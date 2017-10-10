from btb.key import Key, KeyStruct
from btb.selection import Selector
from btb.database import GetLearnersInFrozen
import operator
import numpy as np
import random
import math


class SampleSelector(object):
    def __init__(self, parameters):
        """
        Accepts a list of pamameter metadata structures.
        parameters will look like this:
        [
            ('C', 		KeyStruct(range=(1e-05, 100000), 	type='FLOAT_EXP', 	is_categorical=False)),
            ('degree', 	KeyStruct(range=(2, 4), 			type='INT', 		is_categorical=False)),
            ('coef0', 	KeyStruct(range=(0, 1), 		    type='INT', 		is_categorical=False)),
            ('gamma', 	KeyStruct(range=(1e-05, 100000),	type='FLOAT_EXP', 	is_categorical=False))
        ]
        """
        self.parameters = parameters


    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores

        Returns:
            self: SampleSelector
        """
        pass

    def predict(self, X):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)

        returns:
            y: np.ndarray of predicted scores
        """
        pass

    def create_candidates(self, n=10000):
        """
        Generate a number of random hyperparameter vectors based on the
        parameter specifications given to the constructor.
        """
        vectors = np.zeros((n, len(self.parameters)))
        for i, (k, struct) in enumerate(self.parameters):
            if struct.type == Key.TYPE_FLOAT_EXP:
                random_powers = 10.0 ** np.random.random_integers(
                    math.log10(struct.range[0]), math.log10(struct.range[1]), size=n)
                random_floats = np.random.rand(n)
                column = np.multiply(random_powers, random_floats)

            elif struct.type == Key.TYPE_INT:
                column = np.random.random_integers(struct.range[0],
                                                   struct.range[1], size=n)

            elif struct.type == Key.TYPE_INT_EXP:
                column = 10.0 ** np.random.random_integers(
                    math.log10(struct.range[0]), math.log10(struct.range[1]),
                    size=n)

            elif struct.type == Key.TYPE_FLOAT:
                column = np.random.rand(n)

            vectors[:, i] = column
            i += 1

        return vectors

    def propose(self):
        """
        Use the trained model to propose a new set of parameters.
        """
        candidate_params = self.create_candidates()
        predictions = self.predict(candidate_params)
        best = np.argmax(predictions)
        return candidate_params[best, :]
