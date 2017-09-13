from atm.cpt import Choice, Combination
from atm.enumeration import Enumerator
from atm.enumeration.classification import ClassifierEnumerator
from atm.key import Key, KeyStruct
import numpy as np

class EnumeratorGPC(ClassifierEnumerator):

    DEFAULT_RANGES = {
        "kernel" : ('constant', 'rbf', 'matern', 'rational_quadratic', 'exp_sine_squared'),
        "nu" : (0.5, 1.5, 2.5),
        "length_scale" : (0.01, 100),
        "alpha" : (0.01, 100),
        "periodicity" : (0.01, 100)
    }
    
    DEFAULT_KEYS = {
        # KeyStruct(range, key_type, is_categorical)
        "kernel" : KeyStruct(DEFAULT_RANGES["kernel"], Key.TYPE_STRING, True),
        "nu" : KeyStruct(DEFAULT_RANGES["nu"], Key.TYPE_INT, True),
        "length_scale" : KeyStruct(DEFAULT_RANGES["length_scale"], Key.TYPE_FLOAT, False),
        "alpha" : KeyStruct(DEFAULT_RANGES["alpha"], Key.TYPE_FLOAT, False),
        "periodicity": KeyStruct(DEFAULT_RANGES["periodicity"], Key.TYPE_FLOAT, False),
    }

    def __init__(self, ranges=None, keys=None):
        super(EnumeratorGPC, self).__init__(
            ranges or EnumeratorGPC.DEFAULT_RANGES, keys or EnumeratorGPC.DEFAULT_KEYS)
        self.code = ClassifierEnumerator.GPC
        self.create_cpt()
        
    def create_cpt(self):
        kernel = Choice("kernel", self.ranges["kernel"])
        nu = Choice("nu", self.ranges["nu"])
        length_scale = Choice("length_scale", self.ranges["length_scale"])
        alpha = Choice("alpha", self.ranges["alpha"])
        periodicity = Choice("periodicity", self.ranges["periodicity"])

        matern_params = Combination([nu])
        kernel.add_condition('matern', [matern_params])

        rational_quad_params = Combination([length_scale, alpha])
        kernel.add_condition('rational_quadratic', [rational_quad_params])

        exp_sine_squared_params = Combination([length_scale, periodicity])
        kernel.add_condition('exp_sine_squared', [exp_sine_squared_params])

        gpc = Combination([kernel])
        gpcroot = Choice("function", [ClassifierEnumerator.GPC])
        gpcroot.add_condition(ClassifierEnumerator.GPC, [gpc])
        
        self.root = gpcroot