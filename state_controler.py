from my_vocab import Vocabulary
import sql2graph
import numpy as np

END = -1
id2label = {0: 'operation_list', 1: 'aggregator_list', 2: 'key_word', 3: 'conjunction', 4: 'all_any', 5: 'functional_word',
            6: 'unknown_token', 7: 'value', 8: 'anon_symbol', 9: 'column', 10: 'table', 11: 'neg', 12: 'bracket', 13: 'end_token',
            14: 'eos'}
mask_dict = {0: [1, 2, 10], 1: [10], 2: [9], 3: [1, 5, 10], 4: [10, 11, 12], 5: [5], 6: [9], 7: [0], 8: [7, 8, 5, 4, 10],
             9: [2, 3, 13], 10: [9], 11: [5], 12: [10], 13: [9], 14: [7, 8], 15: [7, 8], 16: [13], -1: [14]}

test = ['distinct', 'flight', 'flight_id', '<C>', '<C>', 'flight', 'departure_time', '=', '<S>', 'min', 'flight', 'departure_time', '<C>', '<C>', 'flight', 'from_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "\\'BOSTON\\'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', 'and', '<C>', 'flight', 'to_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "'ATLANTA'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', 'and', '<C>', '<C>', 'flight', 'from_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "'BOSTON'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', 'and', '<C>', 'flight', 'to_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "'ATLANTA'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>']


class Controller:
    def __init__(self, output_vocabulary):
        self.vocab = output_vocabulary
        self.state = 0
        self.stack = []
        self.graph = sql2graph.node()
        self.top = self.graph

    def initialize(self):
        self.graph = sql2graph.node()
        self.top = self.graph
        self.state = 0

    def mask(self):
        valid_list = self.vocab.get_index_by_id_list(mask_dict[self.state])
        """
        for num in valid_list:
            print(self.vocab.inorder_tokens[num])
        """
        res = np.zeros(len(self.vocab))
        for num in valid_list:
            res[num] += 1
        return np.reshape(res, [-1, 1])

    def update(self, token):
        label = self.vocab.label2id[self.vocab.token_to_label(token)]
        #print("state: ", self.state)
        #print(token, label)
        # initial state, key words, columns & aggregators are allowed
        if self.state == 0:
            # eg: select [ max ] -> table
            if label == 1:
                self.state = 1
                # first aggregator
                if not self.top.column:
                    self.top.aggregate = token
                else:
                    self.top.additional_agg.append(token)
            # eg: select [ distinct ] -> table
            elif label == 2:
                self.state = 0
                self.top.distinct = True
            # eg: select [ table ] -> column
            elif label == 10:
                self.state = 2
                if not self.top.table:
                    self.top.table = token
                else:
                    self.top.additional_col.append(token)
            else:
                raise(AssertionError("Exception Occurred!"))
        # select max/distinct ...
        elif self.state == 1:
            assert label == 10
            # eg: select [ table ] -> column
            self.state = 2
            if not self.top.table:
                self.top.table = token
            else:
                self.top.additional_col.append(token)
        # select table...
        elif self.state == 2:
            # eg: select distinct [  ]
            if label != 9:
                if token != "dual_carrier":
                    raise(AssertionError("Error Occurred!"))
            self.state = 3
            if not self.top.column:
                self.top.column = token
            else:
                self.top.additional_col[-1] += "." + token
        # select table.column ...
        elif self.state == 3:
            # eg: table.column, [ max ] -> table
            if label == 1:
                self.state = 1
                self.top.additional_agg.append(token)
            # eg: table.column from table where [ <C> ]
            elif label == 5:
                self.stack.append('C')
                # TO BE COMPLETED
                self.top.constraint = sql2graph.Constraint()
                self.state = 4
            # eg: table.column [ table ] -> column
            elif label == 10:
                self.state = 2
                self.top.additional_col.append(token)
            else:
                raise(AssertionError("Exception Occurred!"))
        # ...from table where <C>...
        elif self.state == 4:
            # eg: [ not ] -> table
            if label == 11:
                self.state = 4
            # eg: where [ ( ] -> <C>
            elif label == 12:
                self.state = 5
            # eg: where [ table] -> column
            elif label == 10:
                self.state = 6
            else:
                if token != '1' and label != 8:
                    raise(AssertionError("Exception Occurred!"))
                else:
                    self.state = 7
        # ... not( ...
        elif self.state == 5:
            assert label == 5
            # eg: not [ <C> ]
            self.state = 4
            self.stack.append('C')
        # where table...
        elif self.state == 6:
            # eg: where table [ column ] -> operator
            assert label == 9
            self.state = 7
        # where table.column...
        elif self.state == 7:
            # eg: where table.column [ operator ] -> value/<S>/null
            assert label == 0
            self.state = 8
            if token == 'is null' or token == 'is not null':
                self.state = 9
            if token == 'between' or token == 'not between':
                self.state = 14
        # where table.column op...
        elif self.state == 8:
            # eg: where table.column > [ value ]
            if label == 7 or label == 8:
                self.state = 9
            # eg: where table.column in [ <S> ]
            elif label == 5:
                self.state = 0
                self.stack.append('S')
            # eg: where table.column > [ all ] -> <S>
            elif label == 4:
                self.state = 11
            # eg: where table.column < [ table ] -> column
            elif label == 10:
                self.state = 10
            else:
                raise(AssertionError("Exception Occurred!"))
        # where <C>...
        elif self.state == 9:
            # eg: where <C> [ group by ] -> table
            if label == 2:
                self.state = 12
            # eg: where <C> [ and ] -> not/(<C>)/table
            elif label == 3:
                self.state = 5
            # eg: where <C> [ <EOT> ]
            elif label == 13:
                if not self.stack:
                    self.state = END
                else:
                    top = self.stack.pop(-1)
                    self.state = 9
            else:
                raise(AssertionError("Exception Occurred!"))
        # where table.column > table...
        elif self.state == 10:
            # eg: table.column > table [ column ] -> and/<EOT>/group by
            assert label == 9
            self.state = 9
        # table.column > all...
        elif self.state == 11:
            # eg: table.column > all [ <S> ]
            assert label == 5
            self.stack.append('S')
            self.state = 0
        # ...group by...
        elif self.state == 12:
            assert label == 10
            # eg: group by [ table ] -> column
            self.state = 13
        elif self.state == 13:
            assert label == 9
            # eg: group by table [ column ] -> <EOT>
            self.state = 16
        elif self.state == 14:
            # eg: between [ value ] -> value
            assert label == 7 or label == 8
            self.state = 15
        elif self.state == 15:
            # eg: between value and [ value ]
            assert label == 7 or label == 8
            self.state = 9
        elif self.state == 16:
            # eg: group by table.column [ <EOT> ]
            assert label == 13
            self.state = END
        elif self.state == END:
            assert label == 14


if __name__ == "__main__":
    path = "/Users/mac/PycharmProjects/atis/processed_data/interactions"
    vocab = Vocabulary(path)
    controller = Controller(vocab)
    controller.initialize()
    print(controller.mask())



