import dynet as dy
import numpy as np

<<<<<<< HEAD
EOS_TOK = "_EOS"

=======
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8

class UtteranceModel():
    # initialization, construct all the parameters for the model
    def __init__(self, params, input_vocab, output_vocab):
        self.pc = dy.ParameterCollection()
<<<<<<< HEAD
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
=======

        self.input_embedder = self.pc.add_lookup_parameters(
            dim=(len(input_vocab), params.input_embedding_size),
            init="normal")

        self.output_embedder = self.pc.add_lookup_parameters(
            dim=(len(output_vocab), params.output_embedding_size),
            init="normal")

        self.encoder = Encoder(params, self.pc)
        #self.decoder = Decoder(params, self.pc)
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8

    def encode(self, params, input_sequence):
        states = self.encoder(params, input_sequence, self.input_embedder)
        return states[-1]


class Decoder():
<<<<<<< HEAD
    def __init__(self, params, pc, output_vocab_len):
=======
    def __init__(self, params, pc):
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8
        self.lstm = dy.VanillaLSTMBuilder(layers=1,
                                          hidden_dim=params.decoder_state_size,
                                          input_dim=params.output_embedding_size + params.decoder_state_size,
                                          model=pc)
<<<<<<< HEAD

=======
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8
        self.initial_input = pc.add_parameters(params.output_embedding_size)
        self.Wa = pc.add_parameters((params.encoder_state_size, params.decoder_state_size))
        self.Wm = pc.add_parameters((output_vocab_len, params.decoder_state_size + params.encoder_state_size))
        self.bm = pc.add_parameters((output_vocab_len, 1))

<<<<<<< HEAD
    def __call__(self, params, embedder, encoder_states, output_vocabulary):
=======
    def __call__(self, params, embedder, encoder_states):
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8
        state = self.lstm.initial_state([dy.zeroes(params.decoder_state_size), params.decoder_state_size])
        keep_looping = 1
        looptime = 0
        output = dy.concatenate((self.initial_input, np.zeros(params.encoder_state_size)))
        outputs = []
<<<<<<< HEAD

        while keep_looping:
            # one_step forward
            state = state.add_input(output)
            hidden_state = dy.reshape(state.h()[0], (1, params.decoder_state_size)) # encoder_state_size * sent_length
=======
        while keep_looping:
            # one_step forward
            state = state.add_input(output)
            hidden_state = dy.reshape(state.h()[0], (1, params.decoder_state_size))
            # query.shape = [encoder_state_size, length_of_sentence]
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8
            # Attention
            query = dy.concatenate(encoder_states, d=1)
            weight = hidden_state * query
            s = dy.softmax(weight, d=1)
            context = query * dy.transpose(s)   # encoder_state_size * 1
            # Predict
            pred_vec = dy.concatenate((context, dy.reshape(state, (params.decoder_state_size, 1))), d=0)
            probability = self.Wm * pred_vec + self.bm
            max_index = np.argmax(probability.npvalue(), axis=0)
<<<<<<< HEAD
            token = output_vocabulary.inorder_tokens[max_index]

            output = dy.concatenate((embedder[max_index], dy.reshape(context, (params.encoder_state_size))))
            outputs.append(token)
            looptime += 1
            if token == EOS_TOK or looptime > params.train_maximum_sql_length:
=======

            output = dy.concatenate((embedder[max_index], dy.reshape(context, (params.encoder_state_size))))
            outputs.append(max_index)
            looptime += 1
            if output == EOS_TOK or looptime > params.train_maximum_sql_length:
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8
                keep_looping = False


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

