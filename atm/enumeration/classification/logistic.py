from atm.cpt import Choice, Combination
from atm.enumeration import Enumerator
from atm.enumeration.classification import ClassifierEnumerator
from btb import HyperParameter, ParamTypes
import numpy as np

class EnumeratorLRC(ClassifierEnumerator):

    DEFAULT_RANGES= {
        "C" : (10**-5, 10**5),
        "tol" : (10**-5, 10**5),
        "penalty" : ('l1', 'l2'),
        "dual" : (True, False),
        "fit_intercept" : (True, False),
        "class_weight" : ('auto', 'auto'),
        "_scale" : (True, True)
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "C" : HyperParameter(DEFAULT_RANGES["C"], ParamTypes.FLOAT_EXP, False),
        "tol" : HyperParameter(DEFAULT_RANGES["tol"], ParamTypes.FLOAT_EXP, False),
        "penalty" : HyperParameter(DEFAULT_RANGES["penalty"], ParamTypes.STRING, True),
        "dual" : HyperParameter(DEFAULT_RANGES["dual"], ParamTypes.BOOL, True),
        "fit_intercept" : HyperParameter(DEFAULT_RANGES["fit_intercept"], ParamTypes.BOOL, True),
        "class_weight" : HyperParameter(DEFAULT_RANGES["class_weight"], ParamTypes.STRING, True),
        "_scale" : HyperParameter(DEFAULT_RANGES["_scale"], ParamTypes.STRING, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorLRC, self).__init__(
            ranges or EnumeratorLRC.DEFAULT_RANGES, keys or EnumeratorLRC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.LRC
        self.create_cpt()

    def create_cpt(self):

        C = Choice("C", self.ranges["C"])
        tol = Choice("tol", self.ranges["tol"])
        penalty = Choice("penalty", self.ranges["penalty"])
        dual = Choice("dual", self.ranges["dual"])
        fit_intercept = Choice("fit_intercept", self.ranges["fit_intercept"])
        class_weight = Choice("class_weight", self.ranges["class_weight"])
        scale = Choice("_scale", self.ranges["_scale"])

        penalty.add_condition("l2", [dual])

        logreg = Combination([C, tol, penalty, fit_intercept, class_weight, scale])
        logroot = Choice("function", [ClassifierEnumerator.LRC])
        logroot.add_condition(ClassifierEnumerator.LRC, [logreg])

        self.root = logroot
