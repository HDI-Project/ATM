import math, random

class UCB1Bandit:

    COUNT_PADDING = 1.0
    REWARDS_PADDING = 1.0

    def __init__(self, arms, count, rewards):
        """
            arms = list of FrozenArm objects
            count = total count of learners tried in this run
            rewards = total rewards earned in this run
        """
        self.arms = arms
        self.n_arms = len(arms)
        self.scores = [0.0] * self.n_arms # a score for each of the arms
        
        # we pad these values in order not
        # to divide by zero and insulate from
        # the first round of results having high effect
        self.count = count 
        self.rewards = rewards
        
        if self.count == 0:
        	self.count += UCB1Bandit.COUNT_PADDING
        if self.rewards == 0:
        	self.rewards += UCB1Bandit.REWARDS_PADDING
        
    def score_arms(self):
        for i, arm in enumerate(self.arms):
            avg_reward = float(arm.rewards)  / float(arm.count)
            score = avg_reward + math.sqrt(2.0 * math.log(self.count) / float(arm.count))
            self.scores[i] = score
            arm.score = score
        idx = self.scores.index(max(self.scores))
        return self.arms[idx].frozen_id
            
class FrozenArm:

    COUNT_PADDING = 1.0
    REWARDS_PADDING = 1.0

    def __init__(self, count, rewards, frozen_id):
        """
            count = total number of learners tried in this fset
            rewards = total rewards earned in this fset
            frozen_id = id of this fset
        """
        self.count = count + FrozenArm.COUNT_PADDING
        self.rewards = rewards + FrozenArm.REWARDS_PADDING
        self.frozen_id = frozen_id
        self.score = -1
