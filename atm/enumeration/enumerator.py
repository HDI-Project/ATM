class Enumerator(object):
    """
    This class is initialized with a list of optimizable Hyperparameters, and
    is used to generate frozen sets (possible combinations of categorical
    hyperparameters).
    """
    def __init__(self, parameters, function):
        """
        parameters: dictionary mapping parameter name -> Hyperparameter
        function: (string) name of the function for which this Enumerator
            enumerates. Must be a value defined on ClassifierEnumerator.
        """
        self.parameters = parameters
        self.function = function

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
