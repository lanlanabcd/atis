class node:
    def __init__(self, table=None, column=None, operation=None, value=None, create_subgraph=False, aggregate = None, \
                 negation=False):
        self.table = table
        self.column = column
        self.create_subgraph = create_subgraph
        self.operation = operation
        self.aggregate = aggregate
        self.value = value
        self.constraint = []
        self.negation = negation
        self.additional_col = []
        self.additional_table = []
        self.additional_agg = []
        self.group_by = None
        self.distinct = False


class Constraint:
    def __init__(self, conjunction = 'and', cons = None, negation = False):
        self.conjunction = conjunction
        if cons:
            self.cons = cons
        else:
            self.cons = []
        self.negation = negation

