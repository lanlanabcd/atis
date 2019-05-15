import my_vocab
from graph import node, Constraint
import numpy as np

END = -1
id2label = {0: 'operation_list', 1: 'aggregator_list', 2: 'key_word', 3: 'conjunction', 4: 'all_any', 5: 'functional_word',
            6: 'unknown_token', 7: 'value', 8: 'anon_symbol', 9: 'column', 10: 'table', 11: 'neg', 12: 'bracket', 13: 'end_token',
            14: 'eos', 15: 'from'}
mask_dict = {0: [1, 2, 10], 1: [10], 2: [9], 3: [1, 10, 15], 4: [10, 11, 12], 5: [5], 6: [9], 7: [0], 8: [7, 8, 5, 4, 10],
             9: [2, 3, 13], 10: [9], 11: [5], 12: [10], 13: [9], 14: [7, 8], 15: [7, 8], 16: [13], 17: [5, 10], -1: [14]}

test = ['distinct', 'flight', 'flight_id', '<C>', '<C>', 'flight', 'departure_time', '=', '<S>', 'min', 'flight', 'departure_time', '<C>', '<C>', 'flight', 'from_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "\\'BOSTON\\'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', 'and', '<C>', 'flight', 'to_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "'ATLANTA'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', 'and', '<C>', '<C>', 'flight', 'from_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "'BOSTON'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', 'and', '<C>', 'flight', 'to_airport', 'in', '<S>', 'airport_service', 'airport_code', '<C>', 'airport_service', 'city_code', 'in', '<S>', 'city', 'city_code', '<C>', 'city', 'city_name', '=', "'ATLANTA'", '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>', '<EOT>']


class Controller:
    def __init__(self, output_vocabulary):
        self.vocab = output_vocabulary
        self.state = 0
        self.stack = []
        self.graph = node()
        self.top = self.graph
        self.sql_stack = []
        self.next_negation = False

    def initialize(self):
        self.graph = node()
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
        # print("state: ", self.state)
        # print(token, label)
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
                if self.top.column:
                    self.top.additional_col.append(token)
            else:
                raise(AssertionError("Exception Occurred!"))
        # select max/distinct ...
        elif self.state == 1:
            assert label == 10
            # eg: select [ table ] -> column
            self.state = 2
            if self.top.column:
                self.top.additional_col.append(token)
        # select table...
        elif self.state == 2:
            # eg: select distinct table [ column ]
            if label != 9:
                if token != "dual_carrier":
                    raise(AssertionError("Error Occurred!"))
            self.state = 3
            if not self.top.column:
                self.top.column = token
            else:
                self.top.additional_col[-1] += "." + token
                if len(self.top.additional_col) != len(self.top.additional_agg):
                    self.top.additional_agg.append("")
        # select table.column ...
        elif self.state == 3:
            # eg: table.column, [ max ] -> table
            if label == 1:
                self.state = 1
                self.top.additional_agg.append(token)
            # eg: table.column from table where [ <C> ]
            elif label == 5:
                self.stack.append('C')
                self.sql_stack.append(self.top)
                self.top.constraint = Constraint()
                self.top = self.top.constraint
                self.state = 4
            # eg: table.column [ table ] -> column
            elif label == 10:
                self.state = 2
                self.top.additional_col.append(token)
            elif label == 15:
                self.state = 17
            else:
                raise(AssertionError("Exception Occurred!"))
        # ...from table where <C>...
        elif self.state == 4:
            # eg: [ not ] -> table
            if label == 11:
                self.state = 4
                #self.top.negation = True
                self.next_negation = True
            # eg: where [ ( ] -> <C>
            elif label == 12:
                self.state = 5
                self.top.cons.append(Constraint())
                self.sql_stack.append(self.top)
                self.top = self.top.cons[-1]
                if self.next_negation:
                    self.top.negation = True
                    self.next_negation = False

            # eg: where [ table] -> column
            elif label == 10:
                self.state = 6
                self.top.cons.append(node())
                self.sql_stack.append(self.top)
                self.top = self.top.cons[-1]
                self.top.table = token
                if self.next_negation:
                    self.top.negation = True
                    self.next_negation = False
            else:
                if token != '1' and label != 8:
                    raise(AssertionError("Exception Occurred!"))
                else:
                    self.top.cons.append(node())
                    self.sql_stack.append(self.top)
                    self.top = self.top.cons[-1]
                    self.top.column = token
                    self.state = 7
                if self.next_negation:
                    self.top.negation = True
                    self.next_negation = False
        # ... ( ...
        elif self.state == 5:
            assert label == 5
            # eg: ( [ <C> ]
            self.state = 4
            self.stack.append('C')
        # where table...
        elif self.state == 6:
            # eg: where table [ column ] -> operator
            assert label == 9
            self.state = 7
            self.top.column = token
        # where table.column...
        elif self.state == 7:
            # eg: where table.column [ operator ] -> value/<S>/null
            assert label == 0
            self.state = 8
            if token == 'is null' or token == 'is not null':
                self.state = 9
            if token == 'between' or token == 'not between':
                self.state = 14
            self.top.operation = token
        # where table.column op...
        elif self.state == 8:
            # eg: where table.column > [ value ]
            if label == 7 or label == 8:
                self.top.value = token
                self.top.constraint = Constraint()
                self.state = 9
            # eg: where table.column in [ <S> ]
            elif label == 5:
                self.state = 0
                if self.top.operation == 'in':
                    self.top.value = [self.top.table, self.top.column]
                    self.top.table = None
                    self.top.column = None
                    self.top.create_subgraph = True
                else:
                    self.top.value = node()
                    self.sql_stack.append(self.top)
                    self.top = self.top.value
                self.stack.append('S')
            # eg: where table.column > [ all ] -> <S>
            elif label == 4:
                self.state = 11
                self.top.operation += " " + token
            # eg: where table.column < [ table ] -> column
            elif label == 10:
                self.state = 10
                self.top.value = token
            else:
                raise(AssertionError("Exception Occurred!"))
        # where <C>...
        elif self.state == 9:
            # eg: where <C> [ group by ] -> table
            if label == 2:
                assert len(self.sql_stack) == 1
                self.state = 12
            # eg: where <C> [ and ] -> not/(<C>)/table
            elif label == 3:
                self.state = 5
                self.top.conjunction = token
            # eg: where <C> [ <EOT> ]
            elif label == 13:
                if not self.stack:
                    self.state = END
                    self.top = self.sql_stack.pop(-1)
                else:
                    top = self.stack.pop(-1)
                    self.state = 9
                    self.top = self.sql_stack.pop(-1)
                    if isinstance(self.top, node):
                        if isinstance(self.top.value, node):
                            self.top = self.sql_stack.pop(-1)
            else:
                raise(AssertionError("Exception Occurred!"))
        # where table.column > table...
        elif self.state == 10:
            # eg: table.column > table [ column ] -> and/<EOT>/group by
            assert label == 9
            self.state = 9
            self.top.value += "." + token
        # table.column > all...
        elif self.state == 11:
            # eg: table.column > all [ <S> ]
            assert label == 5
            self.stack.append('S')
            self.top.value = node()
            self.top = self.top.value
            self.state = 0
        # ...group by...
        elif self.state == 12:
            assert label == 10
            # eg: group by [ table ] -> column
            self.state = 13
            self.sql_stack[0].group_by = token
        elif self.state == 13:
            assert label == 9
            # eg: group by table [ column ] -> <EOT>
            self.state = 16
            self.sql_stack[0].group_by += "." + token
        elif self.state == 14:
            # eg: between [ value ] -> value
            assert label == 7 or label == 8
            self.state = 15
            self.top.value = [token]
        elif self.state == 15:
            # eg: between value and [ value ]
            assert label == 7 or label == 8
            self.state = 9
            self.top.value.append(token)
        elif self.state == 16:
            # eg: group by table.column [ <EOT> ]
            assert label == 13
            self.top = self.sql_stack.pop(-1)
            self.state = END
        elif self.state == 17:
            # eg: select table.column from [ table ]
            if label == 5:
                self.stack.append('C')
                self.sql_stack.append(self.top)
                self.top.constraint = Constraint()
                self.top = self.top.constraint
                self.state = 4
            elif label == 10:
                if self.top.table:
                    self.top.additional_table.append(token)
                else:
                    self.top.table = token

        elif self.state == END:
            assert label == 14


if __name__ == "__main__":
    path = "/Users/mac/PycharmProjects/atis/processed_data/interactions"
    vocab = my_vocab.Vocabulary(path)
    controller = Controller(vocab)
    controller.initialize()
    print(controller.mask())



