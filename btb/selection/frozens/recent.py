from btb.selection.frozens import FrozenSelector, AverageVelocitiyFromList
from btb.selection.bandit import UCB1Bandit, FrozenArm
from btb.database import *
import heapq, time
import math, random
import numpy as np

# minimum number of examples required for ALL frozen
# sets to have evaluated in order to use recent K optimizations
K_MIN = 2

class RecentKReward(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		k, frozen_sets, metric
		"""
		super(RecentKReward, self).__init__(**kwargs)

	def select(self):
		"""
		Keeps the frozen set counts intact but only
		uses the most recent k learner's scores for usage in
		rewards for the bandit calculation
		"""
		all_k = {} # fset id => [ (timestamp, float) ]
		recent_k = {} # fset id => [ k (timestamp, float) ]
		learners = GetLearners(self.frozen_sets[0].datarun_id)

		# for each learner, add to appropriate inner list
		for learner in learners:
			if learner.frozen_set_id not in all_k:
				all_k[learner.frozen_set_id] = []
				recent_k[learner.frozen_set_id] = []
			score = getattr(learner, self.metric)
			timestamp = time.mktime(learner.completed.timetuple())
			if IsNumeric(score):
				all_k[learner.frozen_set_id].append((timestamp, float(score)))

		# for each frozen set, heapify and retrieve the top k elements
		not_enough = False
		for fset in self.frozen_sets:
			# use heapify to get largest k elements from each all_k list
			elts = all_k.get(fset.id, [])
			recent_k[fset.id] = [x[1] for x in heapq.nlargest(self.k, elts)]
			print "Frozen set %d (%s) has recent k: %s" % (fset.id, fset.algorithm, recent_k[fset.id])

			if fset.trained < K_MIN:
				not_enough = True

		if not_enough:
			print "We don't have enough frozen trials for this k! Continuing with normal UCB1..."

		arms = []
		total_count = 0
		total_rewards = 0.0
		for fset in self.frozen_sets:
			count = fset.trained

			# did we have enough?
			if not_enough:
				rewards = float(fset.rewards)
			else:
				rewards = sum(recent_k[fset.id])

			total_rewards += rewards
			total_count += fset.trained
			arm = FrozenArm(count, rewards, fset.id)
			arms.append(arm)

		random.shuffle(arms) # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		return bandit.score_arms()

class RecentKVelocity(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		k, frozen_sets, metric
		"""
		super(RecentKVelocity, self).__init__(**kwargs)

	def select(self):
		"""
		Keeps the frozen set counts intact but only
		uses the most recent k learner's velocities for usage in
		rewards for the bandit calculation
		"""
		all_k = {} # fset id => [ (timestamp, float) ]
		recent_k = {} # fset id => [ k (timestamp, float) ]
		learners = GetLearners(self.frozen_sets[0].datarun_id)
		avg_velocities = {} # fset id => [ float ]

		# for each learner, add to appropriate inner list
		for learner in learners:
			if learner.frozen_set_id not in all_k:
				all_k[learner.frozen_set_id] = []
				recent_k[learner.frozen_set_id] = []
			score = getattr(learner, self.metric)
			timestamp = time.mktime(learner.completed.timetuple())
			if IsNumeric(score):
				all_k[learner.frozen_set_id].append((timestamp, float(score)))

		# for each frozen set, heapify and retrieve the top k elements
		not_enough = False
		for fset in self.frozen_sets:
			# use heapify to get largest k elements from each all_k list
			elts = all_k.get(fset.id, [])
			recent_k[fset.id] = [x[1] for x in heapq.nlargest(self.k, elts)]
			avg_velocities[fset.id] = AverageVelocitiyFromList(recent_k[fset.id], is_ascending=False)

			print "Frozen set %d (%s) has recent k: %s => velocity: %f" % (
				fset.id, fset.algorithm, recent_k[fset.id], avg_velocities[fset.id])

			if fset.trained < K_MIN:
				not_enough = True

		if not_enough:
			print "We don't have enough frozen trials for this k! Continuing with normal UCB1..."

		arms = []
		total_count = 0
		total_rewards = 0.0
		for fset in self.frozen_sets:
			count = fset.trained

			if not_enough:
				# fall back to normal ucb1 bandit
				rewards = float(fset.rewards)
			else:
				rewards = avg_velocities[fset.id]

			total_rewards += rewards
			total_count += fset.trained
			arm = FrozenArm(count, rewards, fset.id)
			arms.append(arm)

		random.shuffle(arms) # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		return bandit.score_arms()
