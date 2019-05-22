import pickle
import json
import my_vocab
import pymysql
from sql2graph import parse_graph
from state_controler import Controller
from parse_args import interpret_args
from sql_util import execution_results
from decimal import Decimal

agg_list = ['MAX', 'count', 'MIN', 'SUM', 'AVG']


def handle_agg(raw_sql):
    for agg in agg_list:
        if agg in raw_sql:
            index = raw_sql.index(agg)
            #assert raw_sql[index+1] == '(' and raw_sql[index+3] == ')'
            res = raw_sql[:index]
            res.append(agg + '(' + raw_sql[index+2] + ')')
            res.extend(raw_sql[index+4:])
            return " ".join(res)
    return " ".join(raw_sql)


def eval_new(data, controller, cursor, copy=False):
    query_match = 0
    execute_match = 0
    syntax = 0
    graph_fail = 0
    execute_fail = 0
    not_match = 0
    copy_index = 0
    file_not_match1 = open("/Users/mac/PycharmProjects/atis/eval_results/EVAL-new-not_match-ans.txt", "w")
    file_not_match2 = open("/Users/mac/PycharmProjects/atis/eval_results/EVAL-new-not_match-gold.txt", "w")
    file_constrcut_fail = open("/Users/mac/PycharmProjects/atis/eval_results/EVAL-new-construct_fail.txt", "w")
    for cnt, item in enumerate(data):
        print(cnt)
        if len(item['gold_query']) == len(item['prediction']):
            copy_index += 1
        if item['flat_prediction'] in item['flat_gold_queries']:
            query_match += 1
            execute_match += 1
            syntax += 1
            print("Exact Match!")
            print("==========")
            continue
        controller.initialize()

        ans = item['flat_prediction']
        try:
            graph = controller.ans_to_graph(ans)
        except:
            print("Constructing Graph Failed!")
            file_constrcut_fail.write(str(ans)+ '\n' + '\n')
            print("==========")
            graph_fail += 1
            continue
        sql, _ = parse_graph(graph, True)
        """
        try:
            cursor.execute(sql)
        except:
            print("SQL Execution Failed!")
            print("==========")
            execute_fail += 1
            continue
        res = cursor.fetchall()
        """
        syntactic, semantic, res = execution_results(sql, params.database_username, params.database_password, 3)
        syntax += syntactic
        gold_sql = item['flat_gold_queries'][0]
        correct_flag = 0
        for gold in eval(item['gold_tables']):
            if set(res) == set(gold):
                correct_flag = 1
        if correct_flag:
            execute_match += 1
            print("Query match.")
        else:
            print("Query not match.")
            print(gold)
            print(res)
            file_not_match1.write(str(ans) + '\n' + '\n')
            file_not_match2.write(str(gold_sql) + '\n' + '\n')
            not_match += 1
        print("==============")
    print("Exact match: ", float(query_match) / len(data))
    print("Result match: ", float(execute_match) / len(data))
    print("Syntax Correct: ", float(syntax) / len(data))
    if params.copy:
        print("Copy Correct: ", float(copy_index) / len(data))
    print("***************")
    print("Construction Fail: ", float(graph_fail) / len(data))
    print("Execution Fail: ", float(execute_fail) / len(data))
    print("Not Match: ", float(not_match) / len(data))
    return float(query_match) / len(data), float(execute_match) / len(data)


def eval_old(data, cursor, params):
    query_match = 0
    syntax = 0
    execute_match = 0
    for item in data:
        if item['flat_prediction'] in item['flat_gold_queries']:
            query_match += 1
            execute_match += 1
            syntax += 1
            continue
        sql = handle_agg(item['flat_prediction'])
        syntactic, semantic, res = execution_results(sql, params.database_username, params.database_password, 3)
        correct_flag = 0
        for gold in eval(item['gold_tables']):
            if set(res) == set(gold) and syntactic and semantic:
                correct_flag = 1
        syntax += syntactic
        res1 = []
        if correct_flag:
            execute_match += 1
            continue
        print("==============")
    print("Exact match: ", float(query_match) / len(data))
    print("Result match: ", float(execute_match) / len(data))
    print("Syntax Correct: ", float(syntax) / len(data))
    return float(query_match) / len(data), float(execute_match) / len(data)


def check_gold(data, cursor):
    cnt = 0
    empty_cnt = 0
    for item in data:
        sql = handle_agg(item['flat_gold_queries'][0])
        cursor.execute(sql)
        res = set(cursor.fetchall())
        gold = set(eval(item['gold_tables'])[0])
        if res == gold:
            cnt += 1
        if gold == {}:
            empty_cnt += 1
    print("GOLD MATCH: ", float(cnt) / len(data))
    print("EMPTY RESULT: ", float(empty_cnt) / len(data))


if __name__ == "__main__":
    db = pymysql.connect(user='root', password='mysql12928', database='atis3')
    cursor = db.cursor()
    cursor.execute("SET sql_mode='IGNORE_SPACE';")
    cursor.execute("use atis3")
    params = interpret_args()
    eval_file = open(params.eval_file, "rb")
    eval_data = eval_file.readlines()
    eval_data = [json.loads(line) for line in eval_data]
    if params.new_version:
        vocab = pickle.load(open("/Users/mac/PycharmProjects/atis/vocab_no_anon", "rb"))
        controller = Controller(vocab)
        eval_new(eval_data, controller, cursor, params.copy)
    else:
        check_gold(eval_data, cursor)
        #eval_old(eval_data, cursor, params)