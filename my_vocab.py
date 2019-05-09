import pymysql
import pickle

vocab = {'operation_list' : ['>', '<', '=', '>=', '<=', 'LIKE', 'is null', 'is not null', 'between', 'in', 'not between',
                             'not in'],
'aggregator_list' : ['MAX', 'count', 'MIN', 'SUM', 'AVG'],
'key_word' : ['group by', 'distinct'],
'conjunction' : ['AND', 'OR'],
'all_any' : ['all', 'any'],
'functional_word' : ['<C>', '<S>'],
'end_token': ['<EOT>'],
'unknown_token' : ['_UNK'],
'neg' : ['not'],
'bracket': ['('],
'eos': ['_EOS']}

inorder_label_list = ['operation_list', 'aggregator_list', 'key_word', 'conjunction', 'all_any', 'functional_word',
                      'unknown_token', 'value', 'anon_symbol', 'column', 'table', 'neg', 'bracket', 'end_token', 'eos']

value_indicator = ['>', '<', '=', '>=', '<=', 'LIKE', 'BETWEEN', 'AND']


def value_wash(vocab):
    all_tokens = []
    for (label, token_one_type) in vocab.items():
        if label != "value":
            all_tokens.extend(token_one_type)
    final_value = []
    for i, token in enumerate(vocab["value"]):
        if token not in all_tokens:
            final_value.append(token)
    vocab["value"] = final_value


class Vocabulary:
    def __init__(self, path1, path2):
        table_dict, columns = generate_table_and_column()
        print(table_dict)
        print(columns)
        print(set(columns) & set(table_dict.keys()))
        anonymize, values = collect_anon(path1, path2, table_dict.keys(), columns)
        vocab["value"] = values
        vocab["anon_symbol"] = anonymize
        vocab["column"] = columns
        vocab["table"] = list(table_dict.keys())
        vocab["column"].append("*")
        value_wash(vocab)
        self.raw_vocab = vocab
        self.inorder_tokens = []
        self.token2id = {}
        self.id2label = {}
        self.label2id = {}
        self.token2label = {}
        for i, label in enumerate(inorder_label_list):
            self.id2label[i] = label
            self.label2id[label] = i
        for (label, token_one_type) in vocab.items():
            self.inorder_tokens.extend(token_one_type)
            for token in token_one_type:
                self.token2label[token] = label
        for i, token in enumerate(self.inorder_tokens):
            self.token2id[token] = i
        self.tokens = set(self.inorder_tokens)
        print(self.id2label)

    def __len__(self):
        return len(self.inorder_tokens)

    def token_to_id(self, token):
        return self.token2id[token]

    def id_to_token(self, id):
        return self.inorder_tokens[id]

    def get_index_by_label_id(self, label_id):
        return [self.token_to_id(token) for token in self.raw_vocab[self.id2label[label_id]]]

    def get_index_by_id_list(self, ids):
        result = []
        for id in ids:
            result.extend(self.get_index_by_label_id(id))
        return result

    def token_to_label(self, token):
        return self.token2label[token]


def collect_anon(train_path, dev_path, tables, columns):
    anonimizers = []
    values = []
    interactions = pickle.load(open(train_path, "rb"))
    for utterances in interactions.examples:
        for utterance in utterances.utterances:
            for i, word in enumerate(utterance.gold_query_to_use):
                if "#" in word:
                    anonimizers.append(word)
                if word in value_indicator:
                    token = utterance.gold_query_to_use[i+1]
                    if token != '(':
                        values.append(token)
                    """
                    if '.' not in token:
                        values.append(token)
                    else:
                        if not (token.split('.')[0] in tables and token.split('.')[1] in columns):
                            values.append(token)
                    """
    interactions = pickle.load(open(dev_path, "rb"))
    for utterances in interactions.examples:
        for utterance in utterances.utterances:
            for i, word in enumerate(utterance.gold_query_to_use):
                if word == "'CO'":
                    p = input("FOUND. CONTINUE?")
                if "#" in word:
                    anonimizers.append(word)
                if word in value_indicator:
                    token = utterance.gold_query_to_use[i+1]
                    if token != '(':
                        values.append(token)
    values = set(values) - set(anonimizers)
    print(values)
    return list(set(anonimizers)), list(set(values))


def generate_table_and_column():
    table_dict = {}
    tables = []
    all_columns = []
    db = pymysql.connect(user='root', password='mysql12928', database='atis3')
    cursor = db.cursor()
    cursor.execute("use atis3")
    cursor.execute("show tables")
    res = cursor.fetchall()
    for t in res:
        tables.append(t[0])
        columns = []
    for table in tables:
        cursor.execute("select column_name from information_schema.columns where table_name = '{}'".format(table))
        res = cursor.fetchall()
        table_dict[table] = []
        for column in res:
            table_dict[table].append(column[0])
            all_columns.append(column[0])
    return table_dict, all_columns


if __name__ == "__main__":
    path = "/Users/mac/PycharmProjects/atis/processed_data/interactions"
    ppath = "/Users/mac/PycharmProjects/atis/dev_interactions"
    output_vocab = Vocabulary(ppath)
    print(len(output_vocab))
    print(output_vocab.id2label)
    print(output_vocab.get_index_by_id_list([0, 1, 10]))
    print(vocab)
