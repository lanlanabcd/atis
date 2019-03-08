import dynet as dy
import numpy as np
from embedder import Embedder
from encoder import Encoder
import dynet_utils as du
from token_predictor import construct_token_predictor
from decoder import SequencePredictor
from vocabulary import DEL_TOK, UNK_TOK, EOS_TOK

def flatten_utterances(utterances):
    """ Gets a flat sequence from a sequence of utterances.

    Inputs:
        utterances (list of list of str): Utterances to concatenate.

    Returns:
        list of str, representing the flattened sequence with separating
            delimiter tokens.
    """
    sequence = []
    for i, utterance in enumerate(utterances):
        sequence.extend(utterance)
        if i < len(utterances) - 1:
            sequence.append(DEL_TOK)

    return sequence


def get_token_indices(token, index_to_token):
    """ Maps from a gold token (string) to a list of indices.

    Inputs:
        token (string): String to look up.
        index_to_token (list of tokens): Ordered list of tokens.

    Returns:
        list of int, representing the indices of the token in the probability
            distribution.
    """
    if token in index_to_token:
        if len(set(index_to_token)) == len(index_to_token):  # no duplicates
            return [index_to_token.index(token)]
        else:
            indices = []
            for index, other_token in enumerate(index_to_token):
                if token == other_token:
                    indices.append(index)
            assert len(indices) == len(set(indices))
            return indices
    else:
        return [index_to_token.index(UNK_TOK)]

class Seq2SeqModel():
    def __init__(self, params, input_vocabulary, output_vocabulary, anonymizer):
        self.params = params

        self._pc = dy.ParameterCollection()

        # Create the input embeddings
        self.input_embedder = Embedder(self._pc,
                                       params.input_embedding_size,
                                       name="input-embedding",
                                       vocabulary=input_vocabulary,
                                       anonymizer=anonymizer)

        # Create the output embeddings
        self.output_embedder = Embedder(self._pc,
                                        params.output_embedding_size,
                                        name="output-embedding",
                                        vocabulary=output_vocabulary,
                                        anonymizer=anonymizer)

        # Create the encoder
        encoder_input_size = params.input_embedding_size
        if params.discourse_level_lstm:
            encoder_input_size += params.encoder_state_size / 2

        self.utterance_encoder = Encoder(params.encoder_num_layers,
                                         encoder_input_size,
                                         params.encoder_state_size,
                                         self._pc)

        # Positional embedder for utterances
        attention_key_size = params.encoder_state_size
        if params.state_positional_embeddings:
            attention_key_size += params.positional_embedding_size
            self.positional_embedder = Embedder(
                self._pc,
                params.positional_embedding_size,
                name="positional-embedding",
                num_tokens=params.maximum_utterances)

        # Create the discourse-level LSTM parameters
        if params.discourse_level_lstm:
            self.discourse_lstms = du.create_multilayer_lstm_params(
                1, params.encoder_state_size, params.encoder_state_size / 2, self._pc, "LSTM-t")
            self.initial_discourse_state = du.add_params(self._pc, tuple(
                [params.encoder_state_size / 2]), "V-turn-state-0")

        # Snippet encoder
        final_snippet_size = 0
        if params.use_snippets and not params.previous_decoder_snippet_encoding:
            snippet_encoding_size = int(params.encoder_state_size / 2)
            final_snippet_size = params.encoder_state_size
            if params.snippet_age_embedding:
                snippet_encoding_size -= int(
                    params.snippet_age_embedding_size / 4)
                self.snippet_age_embedder = Embedder(
                    self._pc,
                    params.snippet_age_embedding_size,
                    name="snippet-age-embedding",
                    num_tokens=params.max_snippet_age_embedding)
                final_snippet_size = params.encoder_state_size \
                    + params.snippet_age_embedding_size / 2

            self.snippet_encoder = Encoder(params.snippet_num_layers,
                                           params.output_embedding_size,
                                           snippet_encoding_size,
                                           self._pc)
        token_predictor = construct_token_predictor(self._pc,
                                                    params,
                                                    output_vocabulary,
                                                    attention_key_size,
                                                    final_snippet_size,
                                                    anonymizer,
                                                    mytoken_predictor=False)

        self.decoder = SequencePredictor(
            params,
            params.output_embedding_size +
            attention_key_size,
            self.output_embedder,
            self._pc,
            token_predictor,
            )

        self.trainer = dy.AdamTrainer(
            self._pc, alpha=params.initial_learning_rate)
        self.dropout = 0.

    def train_step(self, batch):
        """Training step for a batch of examples.

                Input:
                    batch (list of examples): Batch of examples used to update.
                """
        dy.renew_cg(autobatching=True)

        losses = []
        total_gold_tokens = 0

        batch.start()
        while not batch.done():
            example = batch.next()

            # First, encode the input sequences.
            input_sequences = example.histories(
                self.params.maximum_utterances - 1) + [example.input_sequence()]
            final_state, utterance_hidden_states = self._encode_input_sequences(
                input_sequences)

            # Add positional embeddings if appropriate
            if self.params.state_positional_embeddings:
                utterance_hidden_states = self._add_positional_embeddings(
                    utterance_hidden_states, input_sequences)

            # Encode the snippets
            snippets = []
            if self.params.use_snippets:
                snippets = self._encode_snippets(example.previous_query(), snippets)

            # Decode
            flat_seq = []
            for sequence in input_sequences:
                flat_seq.extend(sequence)

            #final_state, 一个list，是最后一个utterance每一步输出的hidden_state（h),
            # utterance_hiddenstate是一个list，
            #里面每一个元素是一个utterance的所有state（h和c）构成的list
            decoder_results = self.decoder(
                final_state,
                utterance_hidden_states,
                self.params.train_maximum_sql_length,
                snippets=snippets,
                gold_sequence=example.gold_query(),
                dropout_amount=self.dropout,
                input_sequence=flat_seq)
            all_scores = [
                step.scores for step in decoder_results.predictions]
            all_alignments = [
                step.aligned_tokens for step in decoder_results.predictions]
            loss = du.compute_loss(example.gold_query(),
                                   all_scores,
                                   all_alignments,
                                   get_token_indices)
            losses.append(loss)
            total_gold_tokens += len(example.gold_query())

        average_loss = dy.esum(losses) / total_gold_tokens
        average_loss.forward()
        average_loss.backward()
        self.trainer.update()
        loss_scalar = average_loss.value()

        return loss_scalar

    def eval_step(self, batch):
        pass

    def _add_positional_embeddings(self, hidden_states, utterances):
        grouped_states = []

        start_index = 0
        for utterance in utterances:
            grouped_states.append(
                hidden_states[start_index:start_index + len(utterance)])
            start_index += len(utterance)
        assert len(hidden_states) == sum([len(seq) for seq in grouped_states])

        assert sum([len(seq) for seq in grouped_states]) \
               == sum([len(utterance) for utterance in utterances])

        new_states = []
        flat_sequence = []

        num_utterances_to_keep = min(
            self.params.maximum_utterances, len(utterances))
        for i, (states, utterance) in enumerate(zip(
                grouped_states[-num_utterances_to_keep:], utterances[-num_utterances_to_keep:])):
            positional_sequence = []
            index = num_utterances_to_keep - i - 1
            for state in states:
                positional_sequence.append(dy.concatenate(
                    [state, self.positional_embedder(index)]))

            assert len(positional_sequence) == len(utterance), \
                "Expected utterance and state sequence length to be the same, " \
                + "but they were " + str(len(utterance)) \
                + " and " + str(len(positional_sequence))
            new_states.extend(positional_sequence)
            flat_sequence.extend(utterance)
        return new_states, flat_sequence
    def _initialize_discourse_states(self):
        discourse_state = self.initial_discourse_state

        discourse_lstm_states = [lstm.initial_state([dy.zeros((lstm.spec[2],)),
                                                     dy.zeros((lstm.spec[2],))])
                                 for lstm in self.discourse_lstms]
        return discourse_state, discourse_lstm_states

    def _encode_with_discourse_lstm(self, utterances):
        """ Encodes the utterances using a discourse-level LSTM, instead of concatenating.

        Inputs:
            utterances (list of list of str): Utterances.
        """
        hidden_states = []

        discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        final_state = None
        for utterance in utterances:
            final_state, utterance_states = self.utterance_encoder(
                utterance,
                lambda token: dy.concatenate([self.input_embedder(token), discourse_state]),
                dropout_amount=self.dropout)

            hidden_states.extend(utterance_states)

            _, discourse_state, discourse_lstm_states = du.forward_one_multilayer(
                final_state, discourse_lstm_states, self.dropout)

        return final_state, hidden_states

    def _encode_input_sequences(self, utterances):
        """ Encodes the input sequences.

        Inputs:
            utterances (list of list of str): Utterances to process.
        """
        if self.params.discourse_level_lstm:
            return self._encode_with_discourse_lstm(utterances)
        else:
            flat_utterances = flatten_utterances(utterances)
            final_state, hidden_states = self.utterance_encoder(
                flat_utterances, self.input_embedder, dropout_amount=self.dropout)

            states_no_delimiters = []
            start_utterance_index = 0
            for utterance in utterances:
                states_no_delimiters.extend(
                    hidden_states[start_utterance_index:start_utterance_index + len(utterance)])
                start_utterance_index += len(utterance) + 1

            return final_state, states_no_delimiters

class InteractionModel():
    # initialization, construct all the parameters for the model
    def __init__(self, params, input_vocab, output_vocab):
        self.pc = dy.ParameterCollection()
        self.input_vocab_len = len(input_vocab.inorder_tokens)
        self.output_vocab_len = len(output_vocab.inorder_tokens)

        self.input_embedder = self.pc.add_lookup_parameters(
            dim=(self.input_vocab_len, params.input_embedding_size),
            init="normal")

        self.output_embedder = self.pc.add_lookup_parameters(
            dim=(len(output_vocab.inorder_tokens), params.output_embedding_size),
            init="normal")

        self.encoder = Encoder(params, self.pc)
        self.decoder = Decoder(params, self.pc, self.output_vocab_len)
        self.output_vocabulary = output_vocab

    def encode(self, params, input_sequence):
        states = self.encoder(params, input_sequence, self.input_embedder)
        return states

    def decode(self,params, encoder_states):
        outputs = self.decoder(params, self.output_embedder, encoder_states, self.output_vocabulary)
        return outputs

    def all_process(self, params, input_sequence):
        encoder_states = self.encoder(params, input_sequence, self.input_embedder)
        outputs = self.decoder(params, self.output_embedder, encoder_states, self.output_vocabulary)
        return outputs


class UtteranceModel():
    # initialization, construct all the parameters for the model
    def __init__(self, params, input_vocab, output_vocab):
        self.pc = dy.ParameterCollection()
        self.input_vocab_len = len(input_vocab.inorder_tokens)
        self.output_vocab_len = len(output_vocab.inorder_tokens)

        self.input_embedder = self.pc.add_lookup_parameters(
            dim=(self.input_vocab_len, params.input_embedding_size),
            init="normal")

        self.output_embedder = self.pc.add_lookup_parameters(
            dim=(len(output_vocab.inorder_tokens), params.output_embedding_size),
            init="normal")

        self.encoder = Encoder(params, self.pc)
        self.decoder = Decoder(params, self.pc, self.output_vocab_len)

    def encode(self, params, input_sequence):
        states = self.encoder(params, input_sequence, self.input_embedder)
        return states[-1]



class Decoder():
    def __init__(self, params, pc, output_vocab_len):
        self.lstm = dy.VanillaLSTMBuilder(layers=1,
                                          hidden_dim=params.decoder_state_size,
                                          input_dim=params.output_embedding_size + params.decoder_state_size,
                                          model=pc)

        self.initial_input = pc.add_parameters(params.output_embedding_size)
        self.Wa = pc.add_parameters((params.encoder_state_size, params.decoder_state_size))
        self.Wm = pc.add_parameters((output_vocab_len, params.decoder_state_size + params.encoder_state_size))
        self.bm = pc.add_parameters((output_vocab_len, 1))

    def __call__(self, params, embedder, encoder_states, output_vocabulary):
        state = self.lstm.initial_state([dy.zeroes(params.decoder_state_size), params.decoder_state_size])
        keep_looping = 1
        looptime = 0
        output = dy.concatenate((self.initial_input, np.zeros(params.encoder_state_size)))
        outputs = []

        while keep_looping:
            # one_step forward
            state = state.add_input(output)
            hidden_state = dy.reshape(state.h()[0], (1, params.decoder_state_size)) # encoder_state_size * sent_length
            # Attention
            query = dy.concatenate(encoder_states, d=1)
            weight = hidden_state * query
            s = dy.softmax(weight, d=1)
            context = query * dy.transpose(s)   # encoder_state_size * 1
            # Predict
            pred_vec = dy.concatenate((context, dy.reshape(state, (params.decoder_state_size, 1))), d=0)
            probability = self.Wm * pred_vec + self.bm
            max_index = np.argmax(probability.npvalue(), axis=0)
            token = output_vocabulary.inorder_tokens[max_index]

            output = dy.concatenate((embedder[max_index], dy.reshape(context, (params.encoder_state_size))))
            outputs.append(token)
            looptime += 1
            if token == EOS_TOK or looptime > params.train_maximum_sql_length:
                keep_looping = False
        return outputs


class Encoder():
    def __init__(self, params, pc):
        # single layer LSTM
        self.forward_lstm = dy.VanillaLSTMBuilder(layers=1,
                                                  hidden_dim=params.encoder_state_size,
                                                  input_dim=params.input_embedding_size,
                                                  model=pc)
        self.backward_lstm = dy.VanillaLSTMBuilder(layers=1,
                                                   hidden_dim=params.encoder_state_size,
                                                   input_dim=params.input_embedding_size,
                                                   model=pc)

    # input_sequence: list of indexes
    def __call__(self, params, input_sequence, embedder):
        forward_states = self.forward_one_lstm(self.forward_lstm, params.encoder_state_size, input_sequence, embedder)
        input_sequence = input_sequence[::-1]
        backward_states = self.forward_one_lstm(self.backward_lstm, params.encoder_state_size, input_sequence, embedder)
        states = [dy.concatenate([forward_states[i], backward_states[i]]) for i in range(len(forward_states))]
        return states

    def forward_one_lstm(self, lstm, hidden_size, input_sequence, embedder):
        state = lstm.initial_state([dy.zeroes(hidden_size), dy.zeroes(hidden_size)])
        hidden_states = []
        for index in input_sequence:
            lstm_input = embedder[index]
            state = state.add_input(lstm_input)
            hidden_states.append(state.h()[0])
        return hidden_states

