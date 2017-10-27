from btb.cpt import Choice, Combination
from btb.enumeration import Enumerator
from btb.enumeration.classification import ClassifierEnumerator
from hyperselection import HyperParameter, ParamTypes
import numpy as np

class EnumeratorKNN(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "n_neighbors" : (1, 20),
        "weights" : ('uniform', 'distance'),
        "algorithm" : ('ball_tree', 'kd_tree', 'brute'),
        "leaf_size" : (1, 50),
        "metric" : ('minkowski', 'euclidean', 'manhattan', 'chebyshev'),
        "p" : (1, 3),
        "_scale" : (True, True),
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "n_neighbors" : HyperParameter(DEFAULT_RANGES["n_neighbors"], ParamTypes.INT, False),
        "weights" : HyperParameter(DEFAULT_RANGES["weights"], ParamTypes.STRING, True),
        "algorithm" : HyperParameter(DEFAULT_RANGES["algorithm"], ParamTypes.STRING, True),
        "leaf_size" : HyperParameter(DEFAULT_RANGES["leaf_size"], ParamTypes.INT, False),
        "metric" : HyperParameter(DEFAULT_RANGES["metric"], ParamTypes.STRING, True),
        "p" : HyperParameter(DEFAULT_RANGES["p"], ParamTypes.INT, False),
        "_scale" : HyperParameter(DEFAULT_RANGES["_scale"], ParamTypes.BOOL, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorKNN, self).__init__(
            ranges or EnumeratorKNN.DEFAULT_RANGES, keys or EnumeratorKNN.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.KNN
        self.create_cpt()

    def create_cpt(self):

        scale = Choice("_scale", self.ranges["_scale"])
        n_neighbors = Choice("n_neighbors", self.ranges["n_neighbors"])
        weights = Choice("weights", self.ranges["weights"])
        leaf_size = Choice("leaf_size", self.ranges["leaf_size"])
        algorithm = Choice("algorithm", self.ranges["algorithm"])
        algorithm.add_condition("ball_tree", [leaf_size])
        algorithm.add_condition("kd_tree", [leaf_size])
        p = Choice("p", self.ranges["p"])
        metric = Choice("metric", self.ranges["metric"])
        metric.add_condition("minkowski", [p])

        knn = Combination([n_neighbors, weights, algorithm, metric, scale])
        knnroot = Choice("function", ["classify_knn"])
        knnroot.add_condition("classify_knn", [knn])

        self.root = knnroot
