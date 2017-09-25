from btb.cpt import Choice, Combination
from btb.enumeration import Enumerator
from btb.enumeration.classification import ClassifierEnumerator
from btb.key import Key, KeyStruct
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
        # KeyStruct(range, key_type, is_categorical)
        "C" : KeyStruct(DEFAULT_RANGES["C"], Key.TYPE_FLOAT_EXP, False),
        "gamma" : KeyStruct(DEFAULT_RANGES["gamma"], Key.TYPE_FLOAT_EXP, False),
        "kernel" : KeyStruct(DEFAULT_RANGES["kernel"], Key.TYPE_STRING, True),
        "degree" : KeyStruct(DEFAULT_RANGES["degree"], Key.TYPE_INT, False),
        "coef0" : KeyStruct(DEFAULT_RANGES["coef0"], Key.TYPE_INT, False),
        "probability" : KeyStruct(DEFAULT_RANGES["probability"], Key.TYPE_BOOL, True),
        "shrinking" : KeyStruct(DEFAULT_RANGES["shrinking"], Key.TYPE_BOOL, True),
        "cache_size" : KeyStruct(DEFAULT_RANGES["cache_size"], Key.TYPE_INT, False),
        "class_weight" : KeyStruct(DEFAULT_RANGES["class_weight"], Key.TYPE_STRING, True),
        "_scale" : KeyStruct(DEFAULT_RANGES["_scale"], Key.TYPE_BOOL, True),
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