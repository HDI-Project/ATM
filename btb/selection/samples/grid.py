from btb.selection.samples import *
from btb.utilities import *
from btb.database import *
from sklearn.gaussian_process import GaussianProcess
import numpy as np


class Grid(SamplesSelector):
    def __init__(self, frozen_set, metric):
        self.frozen_set = frozen_set
        self.metric = metric

    def do_selection(self, past_params):
        """
        Example format:

            past_params = [
                ({...}, y1),
                ({...}, y2),
                ...
            ]
        """
        # first run (no past_params)
        if not past_params:
            vector = GenerateRandomVectorsGRID(1, self.frozen_set.optimizables, 3)
            return VectorBackToParams(vector, self.frozen_set.optimizables, self.frozen_set.frozens,
                                        self.frozen_set.constants)

        # if no continuous hyperparameters, return becuase this hypartition should be only run once (see above)
        if not self.frozen_set.optimizables:
            MarkFrozenSetGriddingDone(self.frozen_set.id)
            return None

        # extract parameters and performances
        params = [x[0] for x in past_params]

        X = ParamsToVectors(params, self.frozen_set.optimizables)

        # randomly generate many vectors
        candidates = GenerateRandomVectorsGRID(10000, self.frozen_set.optimizables, 3)

        is_unrun = False
        chosen = None

        loop_counter = 0
        while (is_unrun is False) and (loop_counter < 10000):
            candidate = candidates[loop_counter]

            is_candidate_in_past_params = False
            for x in X:
                if np.array_equal(candidate, x):
                    is_candidate_in_past_params = True
                    pass

            if is_candidate_in_past_params is False:
                is_unrun = True
                chosen = candidate

            loop_counter += 1

        if not is_unrun:
            MarkFrozenSetGriddingDone(self.frozen_set.id)
            return None

        return VectorBackToParams(chosen, self.frozen_set.optimizables, self.frozen_set.frozens,
                                  self.frozen_set.constants)
