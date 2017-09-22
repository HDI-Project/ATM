from btb.cpt import Choice, Combination
from btb.enumeration import Enumerator
from btb.enumeration.classification import ClassifierEnumerator
from btb.key import Key, KeyStruct
import numpy as np

class EnumeratorGNB(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "_scale_minmax" : (True, True)
    }   

    DEFAULT_KEYS = {
        # KeyStruct(range, key_type, is_categorical)
        "_scale_minmax" : KeyStruct(DEFAULT_RANGES["_scale_minmax"], Key.TYPE_STRING, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorGNB, self).__init__(
            ranges or EnumeratorGNB.DEFAULT_RANGES, keys or EnumeratorGNB.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.GAUSSIAN_NAIVE_BAYES
        self.create_cpt()
        
    def create_cpt(self):
        scale = Choice("_scale_minmax", self.ranges["_scale_minmax"])
        self.root = Choice("function", [ClassifierEnumerator.GAUSSIAN_NAIVE_BAYES])
        self.root.add_condition(ClassifierEnumerator.GAUSSIAN_NAIVE_BAYES, [scale])
      
  
class EnumeratorMNB(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "alpha" : (0.0, 1.0),
        "fit_prior" : (0, 1),
        "class_prior" : (None, None),
        "_scale_minmax" : (False, False),
    }

    DEFAULT_KEYS = {
    	"alpha" : KeyStruct(DEFAULT_RANGES["alpha"], Key.TYPE_FLOAT, False),
    	"fit_prior" : KeyStruct(DEFAULT_RANGES["fit_prior"], Key.TYPE_INT, False),
    	"class_prior" : KeyStruct(DEFAULT_RANGES["class_prior"], Key.TYPE_STRING, True),
    	"_scale_minmax" : KeyStruct(DEFAULT_RANGES["_scale_minmax"], Key.TYPE_STRING, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorMNB, self).__init__(
            ranges or EnumeratorMNB.DEFAULT_RANGES, keys or EnumeratorMNB.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.MULTINOMIAL_NAIVE_BAYES
        self.create_cpt()
        
    def create_cpt(self):
        
        alpha = Choice("alpha", self.ranges["alpha"])
        fit_prior = Choice("fit_prior", self.ranges["fit_prior"])
        class_prior = Choice("class_prior", self.ranges["class_prior"])
        scale = Choice("_scale_minmax", self.ranges["_scale_minmax"])

        mnb = Combination([alpha, fit_prior, scale, class_prior])
        mnbxroot = Choice("function", [ClassifierEnumerator.MULTINOMIAL_NAIVE_BAYES])
        mnbxroot.add_condition(ClassifierEnumerator.MULTINOMIAL_NAIVE_BAYES, [mnb])
        
        self.root = mnbxroot
        
        
        
class EnumeratorBNB(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "alpha" : (0.0, 1.0),
        "binarize" : (0.0, 1.0),
        "fit_prior" : (0, 1),
        "class_prior" : (None, None),
        "_scale" : (True, True),
    }

    DEFAULT_KEYS = {
    	"alpha" : KeyStruct(DEFAULT_RANGES["alpha"], Key.TYPE_FLOAT, False),
    	"binarize" : KeyStruct(DEFAULT_RANGES["alpha"], Key.TYPE_FLOAT, False),
    	"fit_prior" : KeyStruct(DEFAULT_RANGES["fit_prior"], Key.TYPE_INT, False),
    	"class_prior" : KeyStruct(DEFAULT_RANGES["class_prior"], Key.TYPE_STRING, True),
    	"_scale" : KeyStruct(DEFAULT_RANGES["_scale"], Key.TYPE_STRING, True),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorBNB, self).__init__(
            ranges or EnumeratorBNB.DEFAULT_RANGES, keys or EnumeratorBNB.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.BERNOULLI_NAIVE_BAYES
        self.create_cpt()
        
    def create_cpt(self):
        scale = Choice("_scale", self.ranges["_scale"])
        alpha = Choice("alpha", self.ranges["alpha"])
        binarize = Choice("binarize", self.ranges["binarize"]) 
        fit_prior = Choice("fit_prior", self.ranges["fit_prior"])
        class_prior = Choice("class_prior", self.ranges["class_prior"])
        
        bnb = Combination([alpha, binarize, fit_prior, class_prior, scale])
        bnbroot = Choice("function", [ClassifierEnumerator.BERNOULLI_NAIVE_BAYES])
        bnbroot.add_condition(ClassifierEnumerator.BERNOULLI_NAIVE_BAYES, [bnb])
        
        self.root = bnbroot
        
        