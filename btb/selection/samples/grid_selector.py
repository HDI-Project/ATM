import math
import numpy as np

from btb.key import Key
from btb.selection.samples import SampleSelector


class Grid(SampleSelector):
    def __init__(self, optimizables, **kwargs):
        """
        Grid space selector.
        grid_size determines how many possible values to try for each variable,
        i.e. how many blocks in the grid
        """
        super(Grid, self).__init__(optimizables, **kwargs)
        self.grid_size = kwargs.pop('grid_size', 3)
        self.finished = False
        self._define_grid()

    def _define_grid(self):
        """
        Define the range of possible values for each of the optimizable
        parameters.
        """
        self._grid_values = {}
        for k, struct in self.optimizables:
            if struct.type == Key.TYPE_FLOAT_EXP:
                vals = 10.0 ** np.round(np.linspace(struct.range[0],
                                                    math.log10(struct.range[1]),
                                                    self.grid_size), decimals=5)

            elif struct.type == Key.TYPE_INT:
                vals = np.round(np.linspace(struct.range[0], struct.range[1],
                                            self.grid_size))

            elif struct.type == Key.TYPE_INT_EXP:
                vals = np.round(10.0 ** np.linspace(math.log10(struct.range[0]),
                                                    math.log10(struct.range[1]),
                                                    self.grid_size))

            elif struct.type == Key.TYPE_FLOAT:
                vals = np.round(np.linspace(struct.range[0], struct.range[1],
                                            self.grid_size), decimals=5)

            self._grid_values[k] = vals

    def _last_candidate(self):
        """
        Compute the last candidate vector this class should generate; that way,
        we can figure out whether all candidates have been seen or not.
        """
        vector = np.zeros(len(self.optimizables))
        for j, (k, _) in enumerate(self.optimizables):
            param_index = self.grid_size - 1
            vector[j] = self._grid_values[k][param_index]
        return vector

    def fit(self, X, y):
        """
        Not a whole lot here, since this class randomly chooses an untested
        point on the grid.
        """
        self.finished = False
        self.past_params = X

    def create_candidates(self):
        """
        Generate _all_ of hyperparameter vectors in the grid based on the
        parameter specifications given to the constructor. Unlike in other
        selectors, this function is a generator, and yields one possibility at a
        time until it runs out of possibilities.
        """
        # compute the total number of points in the grid
        total_points = self.grid_size ** len(self.optimizables)

        for i in xrange(total_points):
            vector = np.zeros(len(self.optimizables))
            for j, (k, _) in enumerate(self.optimizables):
                param_index = i % self.grid_size
                vector[j] = self._grid_values[k][param_index]
                i /= self.grid_size
            yield vector

    def propose(self):
        """
        Use the trained model to propose a new set of parameters.
        """
        # Return the first candidate we haven't seen before
        for candidate in self.create_candidates():
            # if we've reached the end of iteration, set the finished flag
            if (candidate == self._last_candidate()).all():
                self.finished = True
            if candidate not in self.past_params:
                return candidate
