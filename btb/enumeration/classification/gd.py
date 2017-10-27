from btb.cpt import Choice, Combination
from btb.enumeration import Enumerator
from btb.enumeration.classification import ClassifierEnumerator
from hyperselection import HyperParameter, ParamTypes
import numpy as np

class EnumeratorSGDC(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "loss" : ('hinge', 'log', 'modified_huber', 'squared_hinge'),
        "penalty" : ('l1', 'l2', 'elasticnet'),
        "alpha" : (10**-5, 10**5),
        "l1_ratio" : (0.0, 1.0),
        "fit_intercept" : (0.0, 1.0),
        "n_iter" : (10, 200),
        "shuffle" : (True, True),
        "epsilon" : (10**-5, 10**5),
        "learning_rate": ("constant", "optimal"),
        "eta0": (10**-5, 10**5),
        "class_weight": (None, None),
        "_scale_minmax" : (True, True),
        "n_jobs" : (-1, -1),
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "loss" : HyperParameter(DEFAULT_RANGES["loss"], ParamTypes.STRING, True),
        "penalty" : HyperParameter(DEFAULT_RANGES["penalty"], ParamTypes.STRING, True),
        "alpha" : HyperParameter(DEFAULT_RANGES["alpha"], ParamTypes.FLOAT_EXP, False),
        "l1_ratio" : HyperParameter(DEFAULT_RANGES["l1_ratio"], ParamTypes.FLOAT, False),
        "fit_intercept" : HyperParameter(DEFAULT_RANGES["fit_intercept"], ParamTypes.INT, True),
        "n_iter" : HyperParameter(DEFAULT_RANGES["n_iter"], ParamTypes.INT, False),
        "shuffle" : HyperParameter(DEFAULT_RANGES["shuffle"], ParamTypes.BOOL, True),
        "epsilon" : HyperParameter(DEFAULT_RANGES["epsilon"], ParamTypes.FLOAT_EXP, False),
        "learning_rate" : HyperParameter(DEFAULT_RANGES["learning_rate"], ParamTypes.STRING, True),
        "eta0" : HyperParameter(DEFAULT_RANGES["eta0"], ParamTypes.FLOAT_EXP, False),
        "class_weight" : HyperParameter(DEFAULT_RANGES["class_weight"], ParamTypes.STRING, True),
        "_scale_minmax" : HyperParameter(DEFAULT_RANGES["_scale_minmax"], ParamTypes.BOOL, True),
        "n_jobs" : HyperParameter(DEFAULT_RANGES["n_jobs"], ParamTypes.INT, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorSGDC, self).__init__(
            ranges or EnumeratorSGDC.DEFAULT_RANGES, keys or EnumeratorSGDC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.SGD
        self.create_cpt()

    def create_cpt(self):

        loss = Choice("loss", self.ranges["loss"])
        penalty = Choice("penalty", self.ranges["penalty"])
        alpha = Choice("alpha", self.ranges["alpha"])
        l1_ratio = Choice("l1_ratio", self.ranges["l1_ratio"])
        fit_intercept = Choice("fit_intercept", self.ranges["fit_intercept"])
        n_iter = Choice("n_iter", self.ranges["n_iter"])
        shuffle = Choice("shuffle", self.ranges["shuffle"])
        learning_rate = Choice("learning_rate", self.ranges["learning_rate"])
        eta0 = Choice("eta0", self.ranges["eta0"])
        class_weight = Choice("class_weight", self.ranges["class_weight"])
        scale = Choice("_scale_minmax", self.ranges["_scale_minmax"])

        sgd = Combination([
                loss, penalty, alpha, l1_ratio, scale,
                fit_intercept, n_iter, shuffle,
                learning_rate, eta0, class_weight])
        sgdroot = Choice("function", [ClassifierEnumerator.SGD])
        sgdroot.add_condition(ClassifierEnumerator.SGD, [sgd])

        self.root = sgdroot

class EnumeratorPAC(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "C" : (10**-5, 10**5),
        "fit_intercept" : (True, True),
        "n_iter" : (10, 200),
        "shuffle" : (True, True), # always shuffle!
        "loss" : ("hinge", "squared_hinge"),
        "_scale" : (True, True),
        "n_jobs" : (-1, -1),
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "C" : HyperParameter(DEFAULT_RANGES["C"], ParamTypes.FLOAT_EXP, False),
        "fit_intercept" : HyperParameter(DEFAULT_RANGES["fit_intercept"], ParamTypes.INT, True),
        "n_iter" : HyperParameter(DEFAULT_RANGES["n_iter"], ParamTypes.INT, False),
        "shuffle" : HyperParameter(DEFAULT_RANGES["shuffle"], ParamTypes.BOOL, True),
        "loss" : HyperParameter(DEFAULT_RANGES["loss"], ParamTypes.STRING, True),
        "_scale" : HyperParameter(DEFAULT_RANGES["_scale"], ParamTypes.BOOL, True),
        "n_jobs" : HyperParameter(DEFAULT_RANGES["n_jobs"], ParamTypes.INT, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorPAC, self).__init__(
            ranges or EnumeratorPAC.DEFAULT_RANGES, keys or EnumeratorPAC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.PASSIVE_AGGRESSIVE
        self.create_cpt()

    def create_cpt(self):

        loss = Choice("loss", self.ranges["loss"])
        fit_intercept = Choice("fit_intercept", self.ranges["fit_intercept"])
        n_iter = Choice("n_iter", self.ranges["n_iter"])
        shuffle = Choice("shuffle", self.ranges["shuffle"])
        C = Choice("C", self.ranges["C"])
        scale = Choice("_scale", self.ranges["_scale"])

        pa = Combination([C, loss, fit_intercept, n_iter, shuffle, scale])
        paroot = Choice("function", [ClassifierEnumerator.PASSIVE_AGGRESSIVE])
        paroot.add_condition(ClassifierEnumerator.PASSIVE_AGGRESSIVE, [pa])

        self.root = paroot
