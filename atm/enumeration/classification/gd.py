from atm.cpt import Choice, Combination
from atm.enumeration import Enumerator
from atm.enumeration.classification import ClassifierEnumerator
from atm.key import Key, KeyStruct
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
        # KeyStruct(range, key_type, is_categorical)
        "loss" : KeyStruct(DEFAULT_RANGES["loss"], Key.TYPE_STRING, True),
        "penalty" : KeyStruct(DEFAULT_RANGES["penalty"], Key.TYPE_STRING, True),
        "alpha" : KeyStruct(DEFAULT_RANGES["alpha"], Key.TYPE_FLOAT_EXP, False),
        "l1_ratio" : KeyStruct(DEFAULT_RANGES["l1_ratio"], Key.TYPE_FLOAT, False),
        "fit_intercept" : KeyStruct(DEFAULT_RANGES["fit_intercept"], Key.TYPE_INT, True),
        "n_iter" : KeyStruct(DEFAULT_RANGES["n_iter"], Key.TYPE_INT, False),
        "shuffle" : KeyStruct(DEFAULT_RANGES["shuffle"], Key.TYPE_BOOL, True),
        "epsilon" : KeyStruct(DEFAULT_RANGES["epsilon"], Key.TYPE_FLOAT_EXP, False),
        "learning_rate" : KeyStruct(DEFAULT_RANGES["learning_rate"], Key.TYPE_STRING, True),
        "eta0" : KeyStruct(DEFAULT_RANGES["eta0"], Key.TYPE_FLOAT_EXP, False),
        "class_weight" : KeyStruct(DEFAULT_RANGES["class_weight"], Key.TYPE_STRING, True),
        "_scale_minmax" : KeyStruct(DEFAULT_RANGES["_scale_minmax"], Key.TYPE_BOOL, True),
        "n_jobs" : KeyStruct(DEFAULT_RANGES["n_jobs"], Key.TYPE_INT, True),
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
        # KeyStruct(range, key_type, is_categorical)
        "C" : KeyStruct(DEFAULT_RANGES["C"], Key.TYPE_FLOAT_EXP, False),
        "fit_intercept" : KeyStruct(DEFAULT_RANGES["fit_intercept"], Key.TYPE_INT, True),
        "n_iter" : KeyStruct(DEFAULT_RANGES["n_iter"], Key.TYPE_INT, False),
        "shuffle" : KeyStruct(DEFAULT_RANGES["shuffle"], Key.TYPE_BOOL, True),
        "loss" : KeyStruct(DEFAULT_RANGES["loss"], Key.TYPE_STRING, True),
        "_scale" : KeyStruct(DEFAULT_RANGES["_scale"], Key.TYPE_BOOL, True),
        "n_jobs" : KeyStruct(DEFAULT_RANGES["n_jobs"], Key.TYPE_INT, True),
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