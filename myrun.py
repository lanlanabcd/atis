from parse_args import interpret_args
import atis_data
import mymodel


def train(params, model, data):
    input_sequence = [20, 30, 10, 5, 1, 3, 17]
    expression = model.encode(params, input_sequence)
    print(expression.npvalue().shape)


def main():
    params = interpret_args()
    data = atis_data.ATISDataset(params)
<<<<<<< HEAD
    utterance_batches = data.get_utterance_batches(params.batch_size,
                                                   max_output_length=params.train_maximum_sql_length,
                                                   randomize=not params.deterministic)
=======
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8
    # model_level代表模型复杂程度，目前在实现level0，即普通ENCODER-DECODER模型
    if params.model_level ==0:
        model = mymodel.UtteranceModel(params, data.input_vocabulary, data.output_vocabulary)
    if params.train:
        train(params, model, data)


if __name__ == "__main__":
    main()