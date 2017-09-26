from btb.cpt import Choice, Combination
from btb.enumeration import Enumerator
from btb.enumeration.classification import ClassifierEnumerator
from btb.key import Key, KeyStruct
import numpy as np

class EnumeratorLRC(ClassifierEnumerator):

    DEFAULT_RANGES= {
        "C" : (10**-5, 10**5),
        "tol" : (10**-5, 10**5),
        "penalty" : ('l1', 'l2'),
        "dual" : (True, False),
        "fit_intercept" : (True, False),
        "class_weight" : ('balanced', 'balanced'),
        "_scale" : (True, True)
    }

    DEFAULT_KEYS = {
        # KeyStruct(range, key_type, is_categorical)
        "C" : KeyStruct(DEFAULT_RANGES["C"], Key.TYPE_FLOAT_EXP, False),
        "tol" : KeyStruct(DEFAULT_RANGES["tol"], Key.TYPE_FLOAT_EXP, False),
        "penalty" : KeyStruct(DEFAULT_RANGES["penalty"], Key.TYPE_STRING, True),
        "dual" : KeyStruct(DEFAULT_RANGES["dual"], Key.TYPE_BOOL, True),
        "fit_intercept" : KeyStruct(DEFAULT_RANGES["fit_intercept"], Key.TYPE_BOOL, True),
        "class_weight" : KeyStruct(DEFAULT_RANGES["class_weight"], Key.TYPE_STRING, True),
        "_scale" : KeyStruct(DEFAULT_RANGES["_scale"], Key.TYPE_STRING, True),
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
