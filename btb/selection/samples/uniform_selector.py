from btb.selection.samples import SamplesSelector, GenerateRandomVectors, VectorBackToParams


class UniformSelector(object):
    def __init__(self, parameters):
        """
        Very bare_bones sample selector that returns a random set of parameters
        each time.
        """
        super(UniformSelector, self).__init__(parameters)

    def fit(self, X, y):
        return self

    def propose(self):
        """
        Generate and return a random set of parameters.
        """
        return self.create_candidates(1)[0, :]
