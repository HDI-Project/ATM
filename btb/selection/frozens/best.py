from btb.selection.frozens import FrozenSelector, AverageVelocityFromList
from btb.selection.bandit import UCB1Bandit, FrozenArm
from btb.utilities import *
import heapq
import math, random
import numpy as np

# minimum number of examples required for ALL frozen
# sets to have evaluated in order to use best K optimizations
K_MIN = 2


class BestKReward(FrozenSelector):
	def __init__(self, choices, **kwargs):
		"""
		Needs:
		"""
		super(BestKReward, self).__init__(choices, **kwargs)
        self.k = kwargs.get('k', K_MIN)

	def select(self, scores):
		"""
        Keeps the frozen set counts intact but only uses the top k learner's
        scores for usage in rewards for the bandit calculation
        TODO: are all of these stateless?
		"""
		print "[*] Selecting frozen with BestKReward..."
        self.scores = {c: sorted(scores.get(c, [])) for c in self.choices}

        if min([len(s) for s in self.scores]) < K_MIN:
            # fall back to normal UCB1
            return blah

        arms = []
		for c, scores in self.scores:
			count = len(scores)
            rewards = sum(scores[-self.k:])
			arms.append(FrozenArm(count, rewards, c))

        total_rewards = sum(a.rewards for a in arms)
        total_count = sum(a.count for a in arms)

		random.shuffle(arms) # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		return bandit.score_arms()

    def old_select(self):
		all_k = {} # fset id => [ floats ]
		best_k = {}
		learners = GetLearners(self.frozen_sets[0].datarun_id)

		# Build lists of scores indexed by frozen set.
        # Iterates over all learners and adds the score for one to a list of
        # scores for its frozen set.
        # TODO: this should all be done by the worker.
		for learner in learners:
			if learner.frozen_set_id not in all_k:
				all_k[learner.frozen_set_id] = []
				best_k[learner.frozen_set_id] = []
			score = getattr(learner, self.metric)
			if IsNumeric(score):
				all_k[learner.frozen_set_id].append(float(score))

		# for each frozen set, heapify and retrieve the top k elements
        # TODO: this seems overengineered?
		not_enough = False # do we have enough to use this method?
		for fset in self.frozen_sets:
			# use heapify to get largest k elements from each all_k list
			best_k[fset.id] = heapq.nlargest(self.k, all_k.get(fset.id, []))
            print "Frozen set %d (%s) has best k: %s" % (fset.id,
                                                         fset.algorithm,
                                                         best_k[fset.id])

            # If one frozen set doesn't have enough, then we're not allowed to
            # use best k for any of them? 🤔
			if fset.trained < K_MIN:
				not_enough = True

		if not_enough:
            print "We don't have enough frozen trials for this k! Continuing "
                  "with normal UCB1..."

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

		random.shuffle(arms)    # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		return bandit.score_arms()


class BestKVelocity(FrozenSelector):
	def __init__(self, **kwargs):
		"""
		Needs:
		k, frozen_sets, metric
		"""
		super(BestKVelocity, self).__init__(**kwargs)

	def select(self, scores):
		"""
        Keeps the frozen set counts intact but only uses the top k learner's
        velocities over their last for usage in rewards for the bandit
        calculation
        TODO: are all of these stateless?
		"""
		print "[*] Selecting frozen with BestKReward..."
        self.scores = {c: sorted(scores.get(c, [])) for c in self.choices}
        if min([len(s) for s in self.scores]) < K_MIN:
            # fall back to normal UCB1
            return blah

        arms = []
		for c, scores in self.scores:
            velocity = np.mean([scores[i+1] - scores[i] for i in
                                range(len(scores) - 1)])
			count = len(scores)
			arms.append(FrozenArm(count, velocity, c))

        total_rewards = sum(a.rewards for a in arms)
        total_count = sum(a.count for a in arms)

		random.shuffle(arms)    # so arms are not picked in ordinal ID order
		bandit = UCB1Bandit(arms, total_count, total_rewards)
		return bandit.score_arms()

	def old_select(self):
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
            avg_velocities[fset.id] = AverageVelocityFromList(best_k[fset.id],
                                                              is_ascending=False)
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

