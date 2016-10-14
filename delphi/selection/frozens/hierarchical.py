from delphi.selection.frozens import FrozenSelector
from delphi.selection.frozens.ucb1 import UCB1
from delphi.selection.bandit import UCB1Bandit, FrozenArm
import math, random

class HierarchicalByAlgorithm(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		frozen_sets
		"""
		super(HierarchicalByAlgorithm, self).__init__(**kwargs)
		
	def select(self):
		"""
		Groups the frozen sets by algorithm and first chooses
		an algorithm based on the traditional UCB1 criteria. 
		
		Next, from that algorithm's frozen sets, makes the 
		final set choice. 
		"""
		
		# group the fsets by algorithm
		by_algorithm = {} # algorithm => [ frozen_sets ]
		for fset in self.frozen_sets:
			if not fset.algorithm in by_algorithm:
				by_algorithm[fset.algorithm] = []
			by_algorithm[fset.algorithm].append(fset)
			
		# now create arms and choose
		algorithm_arms = []
		total_count = 0
		total_rewards = 0.0
		for algorithm, fsets in by_algorithm.iteritems():
			count = sum([x.trained for x in fsets])
			rewards = float(sum([x.rewards for x in fsets]))
			total_rewards += rewards
			total_count += fset.trained
			arm = FrozenArm(count, rewards, algorithm)
			algorithm_arms.append(arm)
			
		random.shuffle(algorithm_arms) # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(algorithm_arms, total_count, total_rewards)
		best_algorithm = bandit.score_arms()
		print "* Hierarchical picked algorithm: %s" % best_algorithm
		
		# now use only the frozen sets from the chosen best algorithm
		best_frozen_sets = by_algorithm[best_algorithm]
		normal_ucb1 = UCB1(frozen_sets=best_frozen_sets)
		return normal_ucb1.select()
		
class HierarchicalRandom(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		frozens
		"""
		super(HierarchicalRandom, self).__init__(**kwargs)
		
	def select(self):
		"""
		Groups the frozen sets randomly and first chooses
		an algorithm based on the traditional UCB1 criteria. 
		
		Next, from that random set's frozen sets, makes the 
		final set choice. 
		"""
		pass