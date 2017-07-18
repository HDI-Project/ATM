from delphi.key import Key, KeyStruct
from delphi.selection import Selector
import operator
import numpy as np
import random
import math

SELECTION_SAMPLES_UNIFORM = "uniform"
SELECTION_SAMPLES_GP = "gp"
SELECTION_SAMPLES_GP_EI = "gp_ei"
SELECTION_SAMPLES_GP_EI_TIME = "gp_eitime"
SELECTION_SAMPLES_GP_EI_VEL = "gp_eivel"

class SamplesSelector(Selector):
	def __init__(self, **kwargs):
		"""
		Needs:
		optimizables, frozens, constants
		"""
		super(SamplesSelector, self).__init__(**kwargs)

def GenerateRandomVectors(n, optimizables):
	"""
	Given a set of optimizable key => key struct mappings, 
	randomly generates N vectors but with deterministic ordering
	of corresponding keys in the vector given the set of keys (by sorting).
	
	Optimizables will look like this:
	[
		('C', 		KeyStruct(range=(1e-05, 100000), 	type='FLOAT_EXP', 	is_categorical=False)),
		('degree', 	KeyStruct(range=(2, 4), 			type='INT', 		is_categorical=False)),
		('coef0', 	KeyStruct(range=(0, 1), 			type='INT', 		is_categorical=False)),
		('gamma', 	KeyStruct(range=(1e-05, 100000),	type='FLOAT_EXP', 	is_categorical=False))
	]
	"""
	optimizables.sort(key=operator.itemgetter(0))
	vectors = np.zeros((n, len(optimizables)))
	i = 0
	for k, struct in optimizables:
		#print "For column %d, and key %s for type %s" % (i, k, struct.type)
		vectors[:, i] = DrawRandomValues(n, struct)
		i += 1
	
	if n == 1:
		return vectors[0]
	return vectors
	
def VectorBackToParams(vector, optimizables, frozens, constants):
	"""
	vector is single example with which to convert 
	d optimizable keys back from vector format to dictionaries.
	
	Examples of the format for SVM sigmoid frozen set below:
	
		optimizables = [
			('C', 		KeyStruct(range=(1e-05, 100000), 	type='FLOAT_EXP', 	is_categorical=False)),
			('degree', 	KeyStruct(range=(2, 4), 			type='INT', 		is_categorical=False)),
			('coef0', 	KeyStruct(range=(0, 1), 			type='INT', 		is_categorical=False)),
			('gamma', 	KeyStruct(range=(1e-05, 100000),	type='FLOAT_EXP', 	is_categorical=False))
		]
		
		frozens = (
			('kernel', 'poly'), ('probability', True),
			('_scale', True), ('shrinking', True),
			('class_weight', 'auto')
		)
		
		constants = [
			('cache_size', KeyStruct(range=(15000, 15000), type='INT', is_categorical=False))
		]	
	
	"""
	
	#print "optimizables: %s" % optimizables
	#print "frozens:", frozens
	#print "constants: %s" % constants
	
	optimizables.sort(key=operator.itemgetter(0))
	params = {}
	
	# add the optimizables
	for i, elt in enumerate(vector):
		if getattr(optimizables[i][1], 'type') == 'INT':
			params[optimizables[i][0]] = int(elt)
		elif getattr(optimizables[i][1], 'type') == 'INT_EXP':
			params[optimizables[i][0]] = int(elt)
		elif getattr(optimizables[i][1], 'type') == 'FLOAT':
			params[optimizables[i][0]] = float(elt)
		elif getattr(optimizables[i][1], 'type') == 'FLOAT_EXP':
			params[optimizables[i][0]] = float(elt)
		elif getattr(optimizables[i][1], 'type') == 'BOOL':
			params[optimizables[i][0]] = bool(elt)
		else:
			raise ValueError('Unknown data type: {}'.format(getattr(optimizables[i][1], 'type')))
		
	#print "After optimizables: %s" % params
	
	# add the frozen categorical settings
	for key, value in frozens:
		params[key] = value
	
	#print "After frozens: %s" % params
	
	# and finally the constant values
	for constant_key, struct in constants:
		params[constant_key] = struct.range[0]
		
	#print "After constants: %s" % params
	
	return params
	
def ParamsToVectors(params, optimizables):
	"""
	params is a list of {}
	
	Example of optimizables below:
	
	optimizables = [
		('C', 		KeyStruct(range=(1e-05, 100000), 	type='FLOAT_EXP', 	is_categorical=False)),
		('degree', 	KeyStruct(range=(2, 4), 			type='INT', 		is_categorical=False)),
		('coef0', 	KeyStruct(range=(0, 1), 			type='INT', 		is_categorical=False)),
		('gamma', 	KeyStruct(range=(1e-05, 100000),	type='FLOAT_EXP', 	is_categorical=False))
	]
	
	Creates vectors ready to be optimized by a Gaussian Process.
	"""
	if not isinstance(params, (list, np.ndarray)):
		params = [params]
	
	keys = [k[0] for k in optimizables]
	keys.sort()
	
	vectors = np.zeros((len(params), len(keys)))
	for i, p in enumerate(params):
		for j, k in enumerate(keys):
			vectors[i, j] = p[k]
	return vectors
			
		
def DrawRandomValues(n, struct):
	"""
	Notes np.random.random_integers(lo, hi, size=n) generates 
	n ints in range [lo, hi] INCLUSIVE.
	"""
	if struct.type == Key.TYPE_FLOAT_EXP:
		random_powers = 10.0 ** np.random.random_integers(
			math.log10(struct.range[0]), math.log10(struct.range[1]), size=n)
		random_floats = np.random.rand(n)
		return np.multiply(random_powers, random_floats)
		
	elif struct.type == Key.TYPE_INT:
		return np.random.random_integers(struct.range[0], struct.range[1], size=n)
		
	elif struct.type == Key.TYPE_INT_EXP:
		return 10.0 ** np.random.random_integers(
			math.log10(struct.range[0]), math.log10(struct.range[1]), size=n)
			
	elif struct.type == Key.TYPE_FLOAT:
		return np.random.rand(n)