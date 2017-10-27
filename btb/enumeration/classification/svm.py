from btb.cpt import Choice, Combination
from btb.enumeration import Enumerator
from btb.enumeration.classification import ClassifierEnumerator
from hyperselection import HyperParameter, ParamTypes
import numpy as np

class EnumeratorSVC(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "C" : (10**-5, 10**5),
        "gamma" : (10**-5, 10**5),
        "kernel" : ("rbf", "poly", "linear", "sigmoid"),
        "degree" : (2, 5),
        "coef0" : (-10**3, 10**3),
        "probability" : (True,),
        "shrinking" : (True,),
        "cache_size" : (15000, 15000),
        "class_weight" : ("auto",),
        "_scale" : (True,),
    }

    DEFAULT_KEYS = {
        # HyperParameter(range, key_type, is_categorical)
        "C" : HyperParameter(DEFAULT_RANGES["C"], ParamTypes.FLOAT_EXP, False),
        "gamma" : HyperParameter(DEFAULT_RANGES["gamma"], ParamTypes.FLOAT_EXP, False),
        "kernel" : HyperParameter(DEFAULT_RANGES["kernel"], ParamTypes.STRING, True),
        "degree" : HyperParameter(DEFAULT_RANGES["degree"], ParamTypes.INT, False),
        "coef0" : HyperParameter(DEFAULT_RANGES["coef0"], ParamTypes.INT, False),
        "probability" : HyperParameter(DEFAULT_RANGES["probability"], ParamTypes.BOOL, True),
        "shrinking" : HyperParameter(DEFAULT_RANGES["shrinking"], ParamTypes.BOOL, True),
        "cache_size" : HyperParameter(DEFAULT_RANGES["cache_size"], ParamTypes.INT, False),
        "class_weight" : HyperParameter(DEFAULT_RANGES["class_weight"], ParamTypes.STRING, True),
        "_scale" : HyperParameter(DEFAULT_RANGES["_scale"], ParamTypes.BOOL, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorSVC, self).__init__(
            ranges or EnumeratorSVC.DEFAULT_RANGES, keys or EnumeratorSVC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.SVC
        self.create_cpt()

    def create_cpt(self):

        C = Choice("C", self.ranges["C"])
        gamma = Choice("gamma", self.ranges["gamma"])
        degree = Choice("degree", self.ranges["degree"])
        coef0 = Choice("coef0", self.ranges["coef0"])
        sigmoid = Combination([gamma, coef0])
        polynomial = Combination([degree, gamma, coef0])
        kernel = Choice("kernel", self.ranges["kernel"])
        kernel.add_condition("rbf", [gamma])
        kernel.add_condition("sigmoid", [sigmoid])
        kernel.add_condition("poly", [polynomial])

        # constant choices
        probability = Choice("probability", self.ranges["probability"])
        shrinking = Choice("shrinking", self.ranges["shrinking"])
        scale = Choice("_scale", self.ranges["_scale"])
        cache_size = Choice("cache_size", self.ranges["cache_size"])
        class_weight = Choice("class_weight", self.ranges["class_weight"])

        svm = Combination([C, kernel, probability, scale, shrinking, class_weight, cache_size])
        svmroot = Choice("function", [ClassifierEnumerator.SVC])
        svmroot.add_condition(ClassifierEnumerator.SVC, [svm])

        self.root = svmroot
