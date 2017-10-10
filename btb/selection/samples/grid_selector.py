from btb.selection.samples import *
from btb.utilities import *
from btb.database import *
from sklearn.gaussian_process import GaussianProcess
import numpy as np


class GridSelector(object):
    def __init__(self, parameters, grid_size=3):
        """
        Grid space selector.
        grid_size determines how many possible values to try for each variable,
        i.e. how many blocks in the grid
        """
        super(GridSelector, self).__init__(parameters)
        self.grid_size = grid_size
        self.finished = False

    def fit(self, X, y):
        """
        Not a whole lot here, since this class randomly chooses an untested
        point on the grid.
        """
        self.past_params = X

    def create_candidates(self, n=10000):
        """
        Generate a number of random hyperparameter vectors based on the
        parameter specifications given to the constructor.
        """
        vectors = np.zeros((n, len(self.parameters)))
        for i, (k, struct) in enumerate(self.parameters):
            if struct.type == Key.TYPE_FLOAT_EXP:
                vals = np.round(np.linspace(struct.range[0],
                                            math.log10(struct.range[1]),
                                            self.grid_size), decimals=5)
                column = 10.0 ** np.random.choice(vals, size=n)

            elif struct.type == Key.TYPE_INT:
                vals = np.linspace(struct.range[0], struct.range[1],
                                   self.grid_size)
                column = np.round(np.random.choice(vals, size=n))

            elif struct.type == Key.TYPE_INT_EXP:
                vals = np.linspace(math.log10(struct.range[0]),
                                   math.log10(struct.range[1]),
                                   self.grid_size)
                column = np.round(10.0 ** np.random.choice(vals, size=n))

            elif struct.type == Key.TYPE_FLOAT:
                vals = np.round(np.linspace(struct.range[0], struct.range[1],
                                            self.grid_size), decimals=5)
                column = np.random.choice(vals, size=n)

            vectors[:, i] = column
            i += 1

        return vectors

    def propose(self):
        """
        Use the trained model to propose a new set of parameters.
        """
        # Return the first candidate we haven't seen before
        candidates = self.create_candidates()
        for i in range(candidates.shape[0]):
            if candidates[i, :] not in self.past_params:
                return chosen
