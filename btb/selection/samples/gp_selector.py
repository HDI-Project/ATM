import math
import numpy as np
from __future__ import division
from scipy.stats import norm

from btb.key import Key
from btb.selection.samples import SampleSelector
from btb.utilities import *
from sklearn.gaussian_process import GaussianProcess


class GPSelector(SampleSelector):
    def __init__(self, optimizables, **kwargs):
        """
        r_min: the minimum number of past results this selector needs in order
        to use gaussian process for prediction. If not enough results are
        present during a fit(), subsequent calls to propose() will revert to
        uniform selection.
        """
        super(GPSelector, self).__init__(optimizables, **kwargs)
        self.r_min = kwargs.get('r_min', 2)
        self.uniform = True

    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores
        """
        if X.shape[0] < self.r_min:
            self.uniform = True
        else:
            self.uniform = False

        self.gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                                  nugget=np.finfo(np.double).eps * 1000)
        self.gp.fit(X, y)

    def predict(self, X):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)

        returns:
            y: np.ndarray of predicted scores
        """
        return self.gp.predict(X, eval_MSE=True)

    def propose(self):
        """
        Using the probability_of_random value we computed in fit, either return
        the value with the best expected improvement or choose parameters
        randomly.
        """
        if self.uniform:
            # we probably don't have enough
            return UniformSampler().propose()
        else:
            # otherwise do the normal GPEi thing
            return super(GPSelector, self).propose()


class GPEiSelector(GPSelector):
    def _expected_improvement(self, y_est, stdev):
        """
        Expected improvement criterion:
        http://people.seas.harvard.edu/~jsnoek/nips2013transfer.pdf

        Args:
            y_est:  what the GP estimates the value of y will be
            stdev:  uncertainty of the GP's prediction
        """
        z_score = (self.y_best - y_est) / stdev
        standard_normal = norm()
        ei = stdev * (z_score * standard_normal.cdf(z_score) +\
                      standard_normal.pdf(z_score))
        return ei

    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores
        """
        super(GPEiSelector, self).fit(X, y)

        # the only extra thing to do here is save the best y
        self.best_y = max(y)

    def predict(self, X):
        """
        This is mostly the same as the regular GP selector except that we
        compute the expected improvement of each predicted y after the initial
        predict() call.

        Args:
            X: np.ndarray of feature vectors (vectorized parameters)

        returns:
            y: np.ndarray of predicted scores
        """
        y, stdev = self.gp.predict(X, eval_MSE=True)
        ei_y = [self._expected_improvement(y[i], stdev[i])
                for i in range(len(y))]
        return np.array(ei_y)


class GPEiVelocitySelector(GPEiSelector):
    MULTIPLIER = -100
    N_BEST_Y = 5

    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores
        """
        # first, train a gaussian process like normal
        super(GPEiVelocitySelector, self).fit(X, y)

        # get the best few scores so far, and compute the average distance
        # between them.
        best_few_y = sorted(y)[-self.N_BEST_Y:]
        velocities = [y[i+1] - y[i] for i in range(len(best_few_y) - 1)]

        # the probability of returning random params scales inversely with
        # density of top scores.
        self.probability_of_random = np.exp(self.MULTIPLIER *
                                            np.mean(velocities))

    def propose(self):
        """
        Using the probability_of_random value we computed in fit, either return
        the value with the best expected improvement or choose parameters
        randomly.
        """
        if np.random.random() < self.probability_of_random:
            # choose params at random to avoid local minima
            return UniformSampler().propose()
        else:
            # otherwise do the normal GPEi thing
            return super(GPEiVelocitySelector, self).propose()
