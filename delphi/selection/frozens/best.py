from delphi.selection.frozens import FrozenSelector, AverageVelocitiyFromList
from delphi.selection.bandit import UCB1Bandit, FrozenArm
from delphi.database import *
from delphi.utilities import *
import heapq
import math, random
import numpy as np

# minimum number of examples required for ALL frozen
# sets to have evaluated in order to use best K optimizations
K_MIN = 2

class BestKReward(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		k, frozen_sets, metric
		"""
		super(BestKReward, self).__init__(**kwargs)
		
	def select(self):
		"""
		Keeps the frozen set counts intact but only 
		uses the top k learner's scores for usage in 
		rewards for the bandit calculation 
		"""
		print "[*] Selecting frozen with BestKReward..."
		
		all_k = {} # fset id => [ floats ]
		best_k = {}
		learners = GetLearners(self.frozen_sets[0].datarun_id)
		
		# for each learner, add to appropriate inner list
		for learner in learners:
			if learner.frozen_set_id not in all_k:
				all_k[learner.frozen_set_id] = []
				best_k[learner.frozen_set_id] = []
			score = getattr(learner, self.metric)
			if IsNumeric(score):
				all_k[learner.frozen_set_id].append(float(score))
		
		# for each frozen set, heapify and retrieve the top k elements
		not_enough = False # do we have enough to use this method?
		for fset in self.frozen_sets:
			# use heapify to get largest k elements from each all_k list
			best_k[fset.id] = heapq.nlargest(self.k, all_k.get(fset.id, []))
			print "Frozen set %d (%s) has best k: %s" % (fset.id, fset.algorithm, best_k[fset.id])
			
			if fset.trained < K_MIN:
				not_enough = True
		
		if not_enough:
			print "We don't have enough frozen trials for this k! Continuing with normal UCB1..."
		
		arms = []
		total_count = 0
		total_rewards = 0.0
		for fset in self.frozen_sets:
			count = fset.trained
			
			rewards = 0.0
			if not_enough:
				rewards = float(fset.rewards)
			else:
				rewards = sum(best_k[fset.id])
			
			total_rewards += rewards
			total_count += fset.trained
			arm = FrozenArm(count, rewards, fset.id)
			arms.append(arm)
			
		random.shuffle(arms) # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		return bandit.score_arms()
		
class BestKVelocity(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		k, frozen_sets, metric
		"""
		super(BestKVelocity, self).__init__(**kwargs)
		
	def select(self):
		"""
		Keeps the frozen set counts intact but only 
		uses the top k learner's velocities over their last 
		for usage in rewards for the bandit calculation 
		"""
		print "[*] Selecting frozen with BestKVelocity..."
		
		all_k = {} # fset id => [ floats ]
		best_k = {}  # fset id => [ k floats ]
		learners = GetLearners(self.frozen_sets[0].datarun_id)
		avg_velocities = {} # fset id => [ float ]
		
		# for each learner, add to appropriate inner list
		for learner in learners:
			if learner.frozen_set_id not in all_k:
				all_k[learner.frozen_set_id] = []
				best_k[learner.frozen_set_id] = []
				avg_velocities[learner.frozen_set_id] = []
			score = getattr(learner, self.metric)
			if IsNumeric(score):
				all_k[learner.frozen_set_id].append(float(score))
		
		# for each frozen set, heapify and retrieve the top k elements
		not_enough = False
		for fset in self.frozen_sets:
			# use heapify to get largest k elements from each all_k list
			best_k[fset.id] = heapq.nlargest(self.k, all_k.get(fset.id, []))
			avg_velocities[fset.id] = AverageVelocitiyFromList(best_k[fset.id], is_ascending=False)
			print "Frozen set %d (%s) count=%d has best k: %s => velocity: %f" % (
				fset.id, fset.algorithm, fset.trained, best_k[fset.id], avg_velocities[fset.id])
				
			if fset.trained < K_MIN:
				not_enough = True
				
		if not_enough:
			print "We don't have enough frozen trials for this k! Continuing with normal UCB1..."
		
		arms = []
		total_count = 0
		total_rewards = 0.0
		for fset in self.frozen_sets:
			count = fset.trained
			
			rewards = 0.0
			if not_enough:
				rewards = float(fset.rewards)
			else:
				rewards = avg_velocities[fset.id]
			
			total_rewards += rewards
			total_count += fset.trained
			arm = FrozenArm(count, rewards, fset.id)
			arms.append(arm)
			
		random.shuffle(arms) # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		fset_id = bandit.score_arms()
		self.velocity = avg_velocities[fset_id]
		return fset_id
		