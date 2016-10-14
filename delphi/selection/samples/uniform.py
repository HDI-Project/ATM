from delphi.selection.samples import SamplesSelector, GenerateRandomVectors, VectorBackToParams

class UniformSampler(SamplesSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		optimizables, frozens, constants
		"""
		super(UniformSampler, self).__init__(**kwargs)
		
	def select(self):
		"""
		Generates a single random vector
		"""
		vector = GenerateRandomVectors(1, self.frozen_set.optimizables)
		params = VectorBackToParams(vector, self.frozen_set.optimizables, 
					self.frozen_set.frozens, self.frozen_set.constants)
		return params