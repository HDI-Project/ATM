from btb.selection.frozens import FrozenSelector, UCB1
from btb.selection.bandit import UCB1Bandit, FrozenArm
import random
import numpy as np

# minimum number of examples required for ALL frozen
# sets to have evaluated in order to use recent K optimizations
K_MIN = 2


class RecentKReward(FrozenSelector):
    def __init__(self, choices, **kwargs):
        """
        Needs:
            k: number of best scores to consider
        """
        super(RecentKReward, self).__init__(choices, **kwargs)
        self.k = kwargs.pop('k', K_MIN)
        self.ucb1 = UCB1(choices, **kwargs)

    def select(self, choice_scores):
        """
        Keeps the frozen set counts intact but only uses the top k learner's
        scores for usage in rewards for the bandit calculation
        """
        # if we don't have enough scores to do K-selection, fall back to UCB1
        if min([len(s) for s in choice_scores]) < K_MIN:
            return self.ucb1.select(choice_scores)

        choice_scores = {c: s for c, s in choice_scores if c in self.choices}
        arms = []
        # all scores are already in chronological order
        for choice, scores in choice_scores:
            count = len(scores)
            rewards = sum(scores[-self.k:])
            arms.append(FrozenArm(count, rewards, choice))

        total_rewards = sum(a.rewards for a in arms)
        total_count = sum(a.count for a in arms)

        random.shuffle(arms) # so arms are not picked in ordinal ID order
        bandit = UCB1Bandit(arms, total_count, total_rewards)
        return bandit.score_arms()


class RecentKVelocity(FrozenSelector):
    def __init__(self, **kwargs):
        """
        Needs:
            k: number of best scores to consider
        """
        super(RecentKVelocity, self).__init__(choices, **kwargs)
        self.k = kwargs.get('k', K_MIN)
        self.ucb1 = UCB1(choices, **kwargs)

    def select(self, choice_scores):
        """
        Keeps the frozen set counts intact but only uses the top k learner's
        velocities over their last for usage in rewards for the bandit
        calculation
        """
        # if we don't have enough scores to do K-selection, fall back to UCB1
        if min([len(s) for s in choice_scores]) < K_MIN:
            return self.ucb1.select(choice_scores)

        choice_scores = {c: s for c, s in choice_scores if c in self.choices}
        arms = []
        # all scores are already in chronological order
        for choice, scores in choice_scores:
            count = len(scores)
            # truncate to the highest k scores and compute the velocity of those
            scores = scores[-self.k:]
            velocity = np.mean([scores[i+1] - scores[i] for i in
                                range(len(scores) - 1)])
            arms.append(FrozenArm(count, velocity, choice))

        total_rewards = sum(a.rewards for a in arms)
        total_count = sum(a.count for a in arms)

        random.shuffle(arms)    # so arms are not picked in ordinal ID order
        bandit = UCB1Bandit(arms, total_count, total_rewards)
        return bandit.score_arms()
