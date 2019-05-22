import re
import pickle
import pymysql
import my_vocab
import state_controler
from graph import node, Constraint

operation_list = ['>', '<', '=', '>=', '<=', 'LIKE']
aggregator_list = ['MAX', 'count', 'MIN', 'SUM', 'AVG']


def get_right_bracket(sql):
    assert sql[0] == '('
    depth = 0
    for i, word in enumerate(sql):
        if word == '(':
            depth += 1
        if word == ')':
            depth -= 1
        if depth == 0:
            return i


class exception(Exception):
    def __init__(self, str):
        print(str)


def handle_where_clause(head_node, sql_str):
    new_constraint = Constraint()
    cons = []
    negation_flag = False
    next_negation = False
    i = 0
    word = sql_str[i]
    while (word != ')' and word != ';') or negation_flag:
        # nesting
        if word == '(':
            nesting_bound = get_right_bracket(sql_str[i:])
            nested_constraint = handle_where_clause(head_node, sql_str[i+1:i+nesting_bound])
            cons.append(nested_constraint)
            i += nesting_bound
        # negation
        elif word == 'not':
            negation_flag = True
            negation_bound = i + 1 + get_right_bracket(sql_str[i+1:])
            nested_constraint = handle_where_clause(head_node, sql_str[i+2:negation_bound])
            nested_constraint.negation = True
            cons.append(nested_constraint)
            cur_len = len(cons)
            i = negation_bound
        # conjunction
        elif word == 'AND' or word == 'OR':
            new_constraint.conjunction = word
        # where column op value
        elif sql_str[i+1] in operation_list:
            # value is a single word
            all_flag = False
            op = sql_str[i+1]
            if '.' in sql_str[i]:
                table = sql_str[i].split('.')[0]
                col = sql_str[i].split('.')[1]
            else:
                table = None
                col = sql_str[i]
            if sql_str[i+2] == 'ALL':
                op += " all"
                i += 1
            elif sql_str[i+2] == 'ANY':
                op += " any"
                i += 1
            if sql_str[i+2] != '(':
                val = sql_str[i+2]
                cons.append(node(table, col, op, val))
                i += 2
            # value is a subgraph or something else
            else:
                index = get_right_bracket(sql_str[i+2:])
                if sql_str[i+3] != 'SELECT':
                    raise exception("value in a bracket but not a subgraph!")
                new_node = construct_graph(sql_str[i+3:i+2+index])
                cons.append(node(table, col, op, new_node))
                i = i + 2 + index
        # where column between value and value
        elif sql_str[i+1] == 'BETWEEN':
            assert sql_str[i+3] == 'AND'
            if '.' in sql_str[i]:
                cons.append(node(sql_str[i].split('.')[0], sql_str[i].split('.')[1], 'between', [sql_str[i+2], \
                                                                                                 sql_str[i+4]]))
            else:
                cons.append(node(None, sql_str[i], 'between', [sql_str[i+2], sql_str[i+4]]))
            i += 4
        # where column is not null/is null
        elif sql_str[i+1] == 'IS' or sql_str[i+1] == 'is':
            if sql_str[i+2] == 'NULL':
                cons.append(node(sql_str[i].split('.')[0], sql_str[i].split('.')[1], 'is null', None))
                i += 2
            else:
                assert sql_str[i+2] == 'NOT NULL'
                cons.append(node(sql_str[i].split('.')[0], sql_str[i].split('.')[1], 'is not null', None))
                i += 2
        # where column in (subgraph)
        elif sql_str[i+1] == 'IN':
            assert sql_str[i+2] == '('
            bracket_index = get_right_bracket(sql_str[i+2:])
            new_node = construct_graph(sql_str[i+3:i+2+bracket_index])
            new_node.value = [sql_str[i].split('.')[0], sql_str[i].split('.')[1]]
            if next_negation:
                new_node.negation = True
                next_negation = False
            cons.append(new_node)
            i = i + 2 + bracket_index
        elif sql_str[i+1] == 'NOT':
            assert (sql_str[i+2] == 'BETWEEN' and sql_str[i+4] == 'AND') or sql_str[i+2] == 'IN'
            if sql_str[i+2] == 'BETWEEN':
                if '.' in sql_str[i]:
                    cons.append(node(sql_str[i].split('.')[0], sql_str[i].split('.')[1], 'not between', [sql_str[i+3], \
                                                                                                 sql_str[i+5]]))
                else:
                    cons.append(node(None, sql_str[i], 'not between', [sql_str[i+3], sql_str[i+5]]))
                i += 5
            else:
                next_negation = True
                sql_str[i+1] = sql_str[i]
        elif sql_str[i+1] == 'not':
            assert sql_str[i+2] == 'IN'
            next_negation = True
            # pass the table/column to the next position.
            sql_str[i+1] = sql_str[i]

        elif word == 'GROUP' and sql_str[i+1] == 'BY':
            head_node.group_by = sql_str[i+2]
            i += 2
        else:
            raise AssertionError("else occured!")
        i += 1
        """
        if negation_flag:
            # to ensure not assigning negation to the previously existing constraint
            if cons and len(cons) > cur_len:
                cons[-1].negation = True
            if i > negation_bound:
                negation_flag = False
        """
        if i >= len(sql_str):
            break
        word = sql_str[i]
    new_constraint.cons = cons
    return new_constraint


def construct_graph(sql):
        head_node = node()
        sql_str = sql
        i = 0
        selecting_column = 1
        while i < len(sql_str):
            word = sql_str[i]
            if word == 'SELECT':
                head_node.create_subgraph = True
                # select distinct table.column from table
                if sql_str[i+1] == 'DISTINCT':
                    head_node.column = sql_str[i+2].split('.')[1]
                    head_node.table = sql_str[i+2].split('.')[0]
                    head_node.distinct = True
                    i += 3
                # select aggregate(table.column) from table
                elif sql_str[i+1] in aggregator_list:
                    assert sql_str[i+2] == '('
                    for j, word in enumerate(sql_str[i+2:]):
                        if word == ')':
                            record = j
                            break
                    if sql_str[i+3] == 'DISTINCT':
                        head_node.distinct = True
                        sql_str[i+3] = sql_str[i+4]
                    if '.' in sql_str[i+3]:
                        head_node.column = sql_str[i+3].split('.')[1]
                        head_node.table = sql_str[i+3].split('.')[0]
                    else:
                        head_node.column = sql_str[i+3]
                    head_node.aggregate = sql_str[i+1]
                    i += 4
                    if head_node.distinct:
                        i += 1
                # select table.column from table
                else:
                    head_node.column = sql_str[i+1].split('.')[1]
                    head_node.table = sql_str[i+1].split('.')[0]
                    i += 2
            elif word == ',':
                if selecting_column:
                    if '.' in sql_str[i+1]:
                        head_node.additional_col.append(sql_str[i+1])
                        head_node.additional_agg.append("")
                    # select col1, agg(col2) from table
                    elif sql_str[i+1] in aggregator_list:
                        assert sql_str[i+2] == '(' and sql_str[i+4] == ')'
                        head_node.additional_agg.append(sql_str[i+1])
                        head_node.additional_col.append(sql_str[i+3])
                        i += 2
                    else:
                        raise AssertionError('multi-columns without table!')
                else:
                    head_node.additional_table.append(sql_str[i+1])
                i += 2
            elif word == 'FROM':
                head_node.table = sql_str[i+1]
                selecting_column = 0
                i += 2
            elif word == 'WHERE':
                i += 1
                new_constraint = handle_where_clause(head_node, sql_str[i:])
                head_node.constraint = new_constraint
                break
            # group by table.column
            elif word == 'GROUP':
                assert sql_str[i+1] == 'BY' and '.' in sql_str[i+2]
                head_node.group_by = sql_str[i+2]
                i += 3
            else:
                i += 1
        return head_node


def col_and_table(head_node):
    new_ans = []
    if head_node.table:
        new_ans = [head_node.table, head_node.column]
        return head_node.table + "." + head_node.column, new_ans
    else:
        new_ans = [head_node.column]
        return head_node.column, new_ans


def handle_cons(cons, if_and, conjunction):
    new_ans = []
    sql = ""
    if if_and:
        new_ans.append(conjunction)
        sql += " " + conjunction + " "
    new_ans.append('<C>')
    if cons.negation:
        sql += " not ("
        new_ans.append('not')
    if isinstance(cons, node):
        if not cons.create_subgraph:
            if cons.operation == 'between' or cons.operation == 'not between':
                sql += col_and_table(cons)[0] + " " + cons.operation + " " + cons.value[0] + " and " + cons.value[1]
                new_ans.extend(col_and_table(cons)[1])
                new_ans.append(cons.operation)
                new_ans.append(cons.value[0])
                new_ans.append(cons.value[1])
            elif cons.operation == 'is not null' or cons.operation == 'is null':
                sql += col_and_table(cons)[0] + " " + cons.operation + " "
                new_ans.extend(col_and_table(cons)[1])
                new_ans.append(cons.operation)
            elif not isinstance(cons.value, node):
                new_ans.extend(col_and_table(cons)[1])
                if "all" not in cons.operation and "any" not in cons.operation:
                    new_ans.append(cons.operation)
                else:
                    new_ans.append(cons.operation.split(' ')[0])
                    new_ans.append(cons.operation.split(' ')[1])
                new_ans.append(cons.value)
                try:
                    sql += col_and_table(cons)[0] + " " + cons.operation + " " + cons.value
                except:
                    print("here")
            else:
                new_ans.extend(col_and_table(cons)[1])
                if "all" not in cons.operation and "any" not in cons.operation:
                    new_ans.append(cons.operation)
                else:
                    new_ans.append(cons.operation.split(' ')[0])
                    new_ans.append(cons.operation.split(' ')[1])
                new_ans.append('<S>')
                sub_sql, sub_ans = parse_graph(cons.value, False)
                sql += col_and_table(cons)[0] + " " + cons.operation + " (" + sub_sql + ") "
                new_ans.extend(sub_ans)
        else:
            new_ans.append(cons.value[0])
            new_ans.append(cons.value[1])
            new_ans.append('in')
            new_ans.append('<S>')
            sql += cons.value[0] + '.' + cons.value[1] + " in ( "
            sub_sql, sub_ans = parse_graph(cons, False)
            sql += sub_sql + " )"
            new_ans.extend(sub_ans)
    else:
        sql += " ( "
        new_ans.append('(')
        i = 0
        if_and = False
        while i < len(cons.cons):
            sub_cons = cons.cons[i]
            new_sql, cons_ans = handle_cons(sub_cons, if_and, cons.conjunction)
            sql += new_sql
            new_ans.extend(cons_ans)
            if not if_and:
                if_and = True
            i += 1
        sql += " ) "
    if cons.negation:
        sql += " ) "
    new_ans.append('<EOT>')
    return sql, new_ans


def handle_additional_col(new_ans, add):
    agg_exist = False
    for agg in aggregator_list:
        res = re.findall(agg + "\((.*?)\)", add)
        if res:
            agg_exist = True
            new_ans.append(agg)
            if '.' in res[0]:
                new_ans.append(res[0].split('.')[0])
                new_ans.append(res[0].split('.')[1])
            else:
                new_ans.append(res[0])
    if not agg_exist:
        if '.' in add:
            new_ans.append(add.split('.')[0])
            new_ans.append(add.split('.')[1])
        else:
            new_ans.append(add)




def get_aggregated_col(col, agg):
    assert len(col) == len(agg)
    res = []
    for i in range(len(col)):
        if agg[i]:
            res.append(agg[i] + '(' + col[i] + ')')
        else:
            res.append(col[i])
    return res


def parse_graph(head_node, if_first):
    new_ans = []
    sql = "select "
    if head_node.distinct:
        sql += "distinct "
        new_ans.append("distinct")
    if not head_node.aggregate:
        sql += col_and_table(head_node)[0]
        new_ans.extend(col_and_table(head_node)[1])
        """
        for add in head_node.additional_col:
            assert '.' in add or '*' in add
            handle_additional_col(new_ans, add)
        """
        for i in range(len(head_node.additional_col)):
            if head_node.additional_agg[i]:
                new_ans.append(head_node.additional_agg[i])
            if '.' in head_node.additional_col[i]:
                new_ans.append(head_node.additional_col[i].split('.')[0])
                new_ans.append(head_node.additional_col[i].split('.')[1])
            else:
                new_ans.append(head_node.additional_col[i])
        if head_node.additional_col:
            res = get_aggregated_col(head_node.additional_col, head_node.additional_agg)
            sql += " , " + " , ".join(res)
    # select aggregate(column) from table
    # Here we made an assumption that only one column appears between the brackets.
    else:
        new_ans.append(head_node.aggregate)
        new_ans.append(head_node.table)
        new_ans.append(head_node.column)
        sql += head_node.aggregate + "(" + head_node.column
        sql += ") "
    sql += " from " + head_node.table
    new_ans.append("from")
    new_ans.append(head_node.table)
    if head_node.additional_table:
        sql += " , " + " , ".join(head_node.additional_table)
        new_ans.extend(head_node.additional_table)
    if head_node.constraint:
        sql += " where "
        if_and = False
        i = 0
        while i < len(head_node.constraint.cons):
            cons = head_node.constraint.cons[i]
            cons_sql, cons_newans = handle_cons(cons, if_and, head_node.constraint.conjunction)
            sql += cons_sql
            new_ans.extend(cons_newans)
            if not if_and:
                if_and = True
            i += 1
        sql += " "
    if head_node.group_by:
        sql += " group by " + head_node.group_by
        new_ans.append("group by")
        new_ans.append(head_node.group_by.split('.')[0])
        new_ans.append(head_node.group_by.split('.')[1])
    new_ans.append('<EOT>')
    return sql, new_ans


def extract_entity(sql):
    entities = re.findall("('.*?')", sql)
    sql = re.sub("'.*?'", "#entity", sql)
    return sql, entities


def extract(li):
    entities = []
    for word in li:
        if "'" in word:
            entities.append(word)
    return entities


def recover_entity(sql_list, entities):
    entity_num = 0
    for i, word in enumerate(sql_list):
        if word == "#entity":
            sql_list[i] = entities[entity_num]
            entity_num += 1
    return " ".join(sql_list)


def set_environment(path, ppath):
    db = pymysql.connect(user='root', password='mysql12928', database='atis3')
    cursor = db.cursor()
    data = pickle.load(open(ppath, "rb"))
    vocab = my_vocab.Vocabulary(path, ppath)
    controller = state_controler.Controller(vocab)
    return controller, data, cursor


def convert_to_new_answer(ori_sql):
    graph = construct_graph(ori_sql)
    line, new_ans = parse_graph(graph, True)
    return new_ans


def transfer_dataset(dataset, controller=None, name=None, maximum=None):
    for interaction in dataset.examples:
        for utterance in interaction.utterances:
            for i, gold in enumerate(utterance.all_gold_queries):
                new_ans = convert_to_new_answer(gold[0])
                if 'rank' in new_ans:
                    print(gold[0])
                    print(new_ans)
                    new_ans[new_ans.index('rank')] = 'ranks'
                utterance.all_gold_queries[i] = [new_ans, utterance.all_gold_queries[i][1]]
                #print(new_ans)
                #print(i)
            new_ans = convert_to_new_answer(utterance.gold_query_to_use)
            if 'rank' in new_ans:
                new_ans[new_ans.index('rank')] = 'ranks'
            utterance.gold_query_to_use = new_ans
            """
            if controller:
                controller.initialize()
                for token in new_ans:
                    controller.update(token)
                if controller.stack:
                    raise(AssertionError("Error: Stack Not Empty!"))
            """
    for interaction in dataset.examples:
        new_answers = []
        valids = []
        for i, utterance in enumerate(interaction.utterances):
            print(len(utterance.gold_query_to_use))
            if len(utterance.gold_query_to_use) > maximum:
                continue
            else:
                print(i)
                valids.append(i)
            if len(valids) > 1:
                cur_ans = utterance.gold_query_to_use
                prev_ans = interaction.utterances[valids[-2]].gold_query_to_use
                leng = len(cur_ans)
                if leng > len(prev_ans):
                    leng = len(prev_ans)
                count = 0
                for j in range(leng):
                    if cur_ans[j] != prev_ans[j]:
                        break
                    if cur_ans[j] == '<S>' or cur_ans[j] == '<C>':
                        count += 1
                new_answers.append(count)
        flag = False
        for i, utterance in enumerate(interaction.utterances):
            if not flag or i not in valids:
                utterance.copy_gold_query = None
                continue
            cnt = 0
            last = new_answers.pop(0)
            if last == 0:
                utterance.copy_gold_query = [0] + utterance.gold_query_to_use
                continue
            for index, token in enumerate(utterance.gold_query_to_use):
                if token == '<S>' or token == '<C>':
                    cnt += 1
                    if cnt == last:
                        utterance.copy_gold_query = [last] + utterance.gold_query_to_use[index:]
                        break
            #print(utterance.gold_query_to_use)
            #print(utterance.input_seq_to_use)
        #print("\n" + "==========")
    pickle.dump(dataset.examples, open("interactions_new_" + name, "wb"))
    print(dataset.examples[0].utterances[0].all_gold_queries)
    print("succeed!")


if __name__ == '__main__':
    path = "/Users/mac/PycharmProjects/atis/train_interactions"
    ppath = "/Users/mac/PycharmProjects/atis/valid_interactions"
    controller, data, cursor = set_environment(path, ppath)
    cnt = 0
    debug = 0
    anonym = 0
    for c1, interaction in enumerate(data.examples):
        for c2, utterance in enumerate(interaction.utterances):
            cnt += 1
            if cnt < 0:
                continue
            if cnt in [92, 388, 463, 464, 2064, 2277, 3946, 3966, 5565, 5566, 1586,
                       764, 768, 1894, 1902, 1929, 1930, 1931]:
                continue
            if debug:
                sql = "( SELECT count ( DISTINCT fare_basis_code ) FROM fare_basis WHERE fare_basis.economy = 'YES' ) ;"
                sql, entity = extract_entity(sql)
                print(sql)
            elif not anonym:
                ori_sql = " ".join(utterance.original_gold_query)
                print(utterance.original_gold_query)
                graph = construct_graph(utterance.original_gold_query)
                line, new_ans = parse_graph(graph, True)
                print(new_ans)
                for token in new_ans:
                    controller.update(token)
                if controller.stack or controller.sql_stack:
                    raise(AssertionError("Error: Stack Not Empty!"))
                print(line)
                new_line, tmp_ans = parse_graph(controller.top, True)
                print(new_line)

                cursor.execute(line)
                res = list(cursor.fetchall())
                cursor.execute(new_line)
                new_res = list(cursor.fetchall())
                gold = utterance.gold_sql_results
                res.sort()
                gold.sort()
                new_res.sort()
                print(cnt)
                if res != new_res:
                    print(res)
                    print(new_res)
                    raise exception("results don't match!")
                """
                if res != gold:
                    print(res)
                    cursor.execute(ori_sql)
                    gold = list(cursor.fetchall())
                    if res != gold:
                        raise exception("results don't match!")
                """
            else:
                print(cnt)
                print(utterance.gold_query_to_use)
                new_ans = convert_to_new_answer(utterance.gold_query_to_use)
                print(new_ans)
                utterance.gold_query_to_use = new_ans
                for token in new_ans:
                    controller.update(token)
                if controller.stack or controller.sql_stack:
                    raise(AssertionError("Error: Stack Not Empty!"))
            print(utterance.original_input_seq)
            controller.initialize()
            print("=======")

    pickle.dump(data, open("interactions_new", "wb"))
