from delphi.selection.frozens import FrozenSelector
import random

class Uniform(FrozenSelector):
	def __init__(self, **kwargs):
		super(Uniform, self).__init__(**kwargs)
		
	def select(self):
		"""
		Simply the uniform random selector.
		"""
		return self.frozen_sets[
			random.randint(0, len(self.frozen_sets) - 1)].id