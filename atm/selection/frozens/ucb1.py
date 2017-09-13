from atm.selection.frozens import FrozenSelector
from atm.selection.bandit import UCB1Bandit, FrozenArm
import math, random

class UCB1(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		frozen_sets
		"""
		super(UCB1, self).__init__(**kwargs)
		
	def select(self):
		"""
		Selects the arm which has the highest score
		as determined by UCB1 bandit algorithm.
		
		majority is the percentage of the dataset that is 
		the minority, representing the baseline.
		"""
		arms = []
		total_count = 0
		total_rewards = 0.0
		for fset in self.frozen_sets:
			count = fset.trained
			rewards = float(fset.rewards) 
			total_rewards += rewards
			total_count += fset.trained
			arm = FrozenArm(count, rewards, fset.id)
			arms.append(arm)
			
		random.shuffle(arms) # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		return bandit.score_arms()