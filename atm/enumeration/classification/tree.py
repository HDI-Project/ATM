from atm.cpt import Choice, Combination
from atm.enumeration import Enumerator
from atm.enumeration.classification import ClassifierEnumerator
from btb import HyperParameter, ParamTypes
import numpy as np

class EnumeratorDTC(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "criterion": ("entropy", "gini"),
        "max_features": (0.1, 1.0),
        "max_depth": (2, 10),
        "min_samples_split": (2, 4),
        "min_samples_leaf": (1, 3),
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "criterion": HyperParameter(DEFAULT_RANGES["criterion"], ParamTypes.STRING, True),
        "max_features": HyperParameter(DEFAULT_RANGES["max_features"], ParamTypes.FLOAT, False),
        "max_depth": HyperParameter(DEFAULT_RANGES["max_depth"], ParamTypes.INT, False),
        "min_samples_split": HyperParameter(DEFAULT_RANGES["min_samples_split"], ParamTypes.INT, False),
        "min_samples_leaf": HyperParameter(DEFAULT_RANGES["min_samples_leaf"], ParamTypes.INT, False),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorDTC, self).__init__(
            ranges or EnumeratorDTC.DEFAULT_RANGES, keys or EnumeratorDTC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.DECISION_TREE
        self.create_cpt()

    def create_cpt(self):
        criterion = Choice("criterion", self.ranges["criterion"])
        max_features = Choice("max_features", self.ranges["max_features"])
        max_depth = Choice("max_depth", self.ranges["max_depth"])
        min_samples_split = Choice("min_samples_split", self.ranges["min_samples_split"])
        min_samples_leaf = Choice("min_samples_leaf", self.ranges["min_samples_leaf"])

        dt = Combination([criterion, max_depth, max_features, min_samples_leaf, min_samples_split])
        dtroot = Choice("function", [ClassifierEnumerator.DECISION_TREE])
        dtroot.add_condition(ClassifierEnumerator.DECISION_TREE, [dt])

        self.root = dtroot

class EnumeratorRFC(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "criterion": ("entropy", "gini"),
        "max_features": (0.1, 1.0),
        "max_depth": (2, 5, 10),
        "min_samples_split": (2, 3),
        "min_samples_leaf": (1, 2),
        "n_estimators": (1000, 1000),
        "n_jobs": (-1, -1),
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "criterion": HyperParameter(DEFAULT_RANGES["criterion"], ParamTypes.STRING, True),
        "max_features": HyperParameter(DEFAULT_RANGES["max_features"], ParamTypes.FLOAT, False),
        "max_depth": HyperParameter(DEFAULT_RANGES["max_depth"], ParamTypes.INT, False),
        "min_samples_split": HyperParameter(DEFAULT_RANGES["min_samples_split"], ParamTypes.INT, False),
        "min_samples_leaf": HyperParameter(DEFAULT_RANGES["min_samples_leaf"], ParamTypes.INT, False),
        "n_estimators": HyperParameter(DEFAULT_RANGES["n_estimators"], ParamTypes.INT, True),
        "n_jobs": HyperParameter(DEFAULT_RANGES["n_jobs"], ParamTypes.INT, False),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorRFC, self).__init__(
            ranges or EnumeratorRFC.DEFAULT_RANGES, keys or EnumeratorRFC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.RANDOM_FOREST
        self.create_cpt()

    def create_cpt(self):

        # dt
        criterion = Choice("criterion", self.ranges["criterion"])
        max_features = Choice("max_features", self.ranges["max_features"])
        max_depth = Choice("max_depth", self.ranges["max_depth"])
        min_samples_split = Choice("min_samples_split", self.ranges["min_samples_split"])
        min_samples_leaf = Choice("min_samples_leaf", self.ranges["min_samples_leaf"])
        n_jobs = Choice("n_jobs", self.ranges["n_jobs"])

        # rf
        n_estimators = Choice("n_estimators", self.ranges["n_estimators"])
        rf = Combination([n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split, n_jobs])
        rfroot = Choice("function", [ClassifierEnumerator.RANDOM_FOREST])
        rfroot.add_condition(ClassifierEnumerator.RANDOM_FOREST, [rf])

        self.root = rfroot


class EnumeratorETC(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "criterion": ("entropy", "gini"),
        "max_features": (0.1, 1.0),
        "max_depth": (2, 5, 10),
        "min_samples_split": (2, 3),
        "min_samples_leaf": (1, 2),
        "n_estimators": (1000, 1000),
        "n_jobs": (-1, -1),
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "criterion": HyperParameter(DEFAULT_RANGES["criterion"], ParamTypes.STRING, True),
        "max_features": HyperParameter(DEFAULT_RANGES["max_features"], ParamTypes.FLOAT, False),
        "max_depth": HyperParameter(DEFAULT_RANGES["max_depth"], ParamTypes.INT, False),
        "min_samples_split": HyperParameter(DEFAULT_RANGES["min_samples_split"], ParamTypes.INT, False),
        "min_samples_leaf": HyperParameter(DEFAULT_RANGES["min_samples_leaf"], ParamTypes.INT, False),
        "n_estimators": HyperParameter(DEFAULT_RANGES["n_estimators"], ParamTypes.INT, True),
        "n_jobs": HyperParameter(DEFAULT_RANGES["n_jobs"], ParamTypes.INT, False),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorETC, self).__init__(
            ranges or EnumeratorETC.DEFAULT_RANGES, keys or EnumeratorETC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.EXTRA_TREES
        self.create_cpt()

    def create_cpt(self):

        # dt
        criterion = Choice("criterion", self.ranges["criterion"])
        max_features = Choice("max_features", self.ranges["max_features"])
        max_depth = Choice("max_depth", self.ranges["max_depth"])
        min_samples_split = Choice("min_samples_split", self.ranges["min_samples_split"])
        min_samples_leaf = Choice("min_samples_leaf", self.ranges["min_samples_leaf"])
        n_jobs = Choice("n_jobs", self.ranges["n_jobs"])

        # rf
        n_estimators = Choice("n_estimators", self.ranges["n_estimators"])
        rf = Combination([n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split, n_jobs])
        rfroot = Choice("function", [ClassifierEnumerator.EXTRA_TREES])
        rfroot.add_condition(ClassifierEnumerator.EXTRA_TREES, [rf])

        self.root = rfroot

