class InteractionGraph():
    def __init__(self, params):
        pass
    def step_forward(self, distribution_map, probabilities):
        '''
        :param distribution_map: a list of str, in which this step generates a output word from, no duplicate
        :param probabilities: a list(or a vector? i'm not so sure) of probability of every word
        :return: a new distribution_map and a new probabilities
        '''
        assert len(distribution_map) == len(probabilities)
        new_distribution_map = []
        new_probabilities = []
        return new_distribution_map, new_probabilities

