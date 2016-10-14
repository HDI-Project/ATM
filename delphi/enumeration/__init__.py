from delphi.cpt import Node, Choice, Combination
from delphi.key import Key, KeyStruct

class Enumerator(object):

    def __init__(self, ranges, keys):
        self.ranges = ranges
        self.keys = keys
        self.root = None 
    
    def combinations(self):
        if self.root:
            return self.root.combinations()
        return None

    def get_categorical_keys(self, no_constants=False):
    	categoricals = []
    	for key, struct in self.keys.iteritems():
    		if struct.is_categorical:
    			if no_constants and len(set(struct.range)) == 1:
    				continue
    			categoricals.append(key)
    	return categoricals
    	
    def get_optimizable_keys(self):
    	optimizables = []
    	for key, struct in self.keys.iteritems():
    		if not struct.is_categorical:
    			if len(set(struct.range)) == 1:
    				continue
    			optimizables.append(key)
    	return optimizables
    	
    def get_constant_optimizable_keys(self):
    	constants = []
    	for key, struct in self.keys.iteritems():
    		if not struct.is_categorical:
    			if len(set(struct.range)) == 1:
    				constants.append(key)
    	return constants