from delphi.selection.samples import *
from delphi.utilities import *
from delphi.database import *
from sklearn.gaussian_process import GaussianProcess
import numpy as np


class GP(SamplesSelector):
    def __init__(self, **kwargs):
        """
        Needs:
        frozen_set, learners, metric
        """
        super(GP, self).__init__(**kwargs)

    def select(self):
        """
        Takes in learner objects from database that
        have been completed.
        """
        past_params = []
        learners = GetLearnersInFrozen(self.frozen_set.id)
        learners = [x for x in learners if x.completed]
        for learner in learners:
            y = float(getattr(learner, self.metric))
            past_params.append((learner.params, y))

        return self.do_selection(past_params)

    def do_selection(self, past_params):
        """
        Based on past parameterizations and their performances,
        select a best candidate for evaluation by randomly generating
        many examples and seeing which has the highest average expected
        regression value.

        Example format:

            past_params = [
                ({...}, y1),
                ({...}, y2),
                ...
            ]
        """
        # extract parameters and performances
        params = [x[0] for x in past_params]
        y = np.array([x[1] for x in past_params])
        X = ParamsToVectors(params, self.frozen_set.optimizables)

        # train a GP
        gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget=np.finfo(np.double).eps * 1000)
        gp.fit(X, y)

        # randomly generate many vectors
        candidates = GenerateRandomVectors(1000000, self.frozen_set.optimizables)
        predictions = gp.predict(candidates)

        # choose one with highest average, convert, and return
        chosen = candidates[np.argmax(predictions)]
        return VectorBackToParams(chosen, self.frozen_set.optimizables, self.frozen_set.frozens,
                                  self.frozen_set.constants)
