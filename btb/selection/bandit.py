import math
import random

class UCB1Bandit(object):
    """
    Simple multi-armed bandit class to aid in frozen set selection.
    TODO: what is UCB1?
    """

    COUNT_PADDING = 1.0
    REWARDS_PADDING = 1.0

    def __init__(self, arms, count, rewards):
        """
        Args:
            arms = list of FrozenArm objects
            count = total count of learners tried in this run
            rewards = total rewards earned in this run
        """
        self.arms = arms
        self.scores = [0.0] * len(arms)   # a score for each of the arms

        # don't want to divide by zero
        # also don't want the first round of results have outsized effect
        self.count = count or UCB1Bandit.COUNT_PADDING

        # as far as I can tell this is never used, removing it for now
        #self.rewards = rewards or UCB1Bandit.REWARDS_PADDING

    def score_arms(self):
        """
        TODO: where is this math from?
        """
        for i, arm in enumerate(self.arms):
            avg_reward = float(arm.rewards) / float(arm.count)
            score = avg_reward + math.sqrt(2.0 * math.log(self.count) /
                                           float(arm.count))
            self.scores[i] = score
            arm.score = score

        idx = self.scores.index(max(self.scores))
        return self.arms[idx].frozen_id


class FrozenArm(object):

    COUNT_PADDING = 1.0
    REWARDS_PADDING = 1.0

    def __init__(self, count, rewards, frozen_id):
        """
        Args:
            count = total number of learners tried in this fset
            rewards = total rewards earned in this fset
            frozen_id = id of this fset
        """
        self.count = count + FrozenArm.COUNT_PADDING
        self.rewards = rewards + FrozenArm.REWARDS_PADDING
        self.frozen_id = frozen_id
        self.score = -1
