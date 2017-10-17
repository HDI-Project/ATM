from btb.selection.frozens import FrozenSelector, UCB1
from btb.selection.bandit import UCB1Bandit, FrozenArm
import random

class HierarchicalByAlgorithm(FrozenSelector):
	def __init__(self, choices, **kwargs):
		"""
		Needs:
            by_algorithm: {str -> list} grouping of frozen set choices by ML
                algorithm
		"""
		super(HierarchicalByAlgorithm, self).__init__(choices, **kwargs)
        self.by_algorithm = kwargs.pop('by_algorithm')

	def select(self, choice_scores):
		"""
        Groups the frozen sets by algorithm and first chooses an algorithm based
        on the traditional UCB1 criteria.

        Next, from that algorithm's frozen sets, makes the final set choice.
		"""
        choice_scores = {c: s for c, s in choice_scores if c in self.choices}

		# create arms and choose algorithm
		algorithm_arms = []
        for algorithm, choices in self.by_algorithm.iteritems():
            # only make arms for algorithms that have options
            if not set(choices) & set(choice_scores.keys()):
                continue

			count = sum(len(choice_scores.get(c, [])) for c in choices)
			rewards = sum(sum(choice_scores.get(c, [])) for c in choices)
			frozen_arms.append(FrozenArm(count, rewards, algorithm))

        total_rewards = sum(a.rewards for a in algorithm_arms)
        total_count = sum(a.count for a in algorithm_arms)

		random.shuffle(algorithm_arms)
		bandit = UCB1Bandit(algorithm_arms, total_count, total_rewards)
		best_algorithm = bandit.score_arms()

		# now use only the frozen sets from the chosen best algorithm
		best_subset = self.by_algorithm[best_algorithm]
		normal_ucb1 = UCB1(choices=best_subset)
		return normal_ucb1.select(choice_scores)

class HierarchicalRandom(FrozenSelector):
	def select(self):
		"""
        Groups the frozen sets randomly and first chooses a random subset based
        on the traditional UCB1 criteria.

        Next, from that random set's frozen sets, makes the final set choice.
		"""
		pass
