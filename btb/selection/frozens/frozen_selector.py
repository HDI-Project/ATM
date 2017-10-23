import random
from btb.selection.bandit import UCB1Bandit, FrozenArm


class FrozenSelector(object):
    def __init__(self, choices, **kwargs):
        """
        choices: a list of discrete choices from which the selector must choose
        at every call to select().
        """
        self.choices = choices

    def select(self, choice_scores):
        """
        Select the next best choice to make
        choice_scores: map of {choice -> [scores]} for each possible choice. The
            caller is responsible for making sure each choice that is possible at
            this juncture is represented in the dict, even those with no scores.
        Score lists should be in ascending chronological order (earliest first)
        e.g.
        {
            1: [0.56, 0.61, 0.33, 0.67],
            2: [0.25, 0.58],
            3: [0.60, 0.65, 0.68]
        }
        """
        pass


class Uniform(FrozenSelector):
    """
    Select a choice uniformly at random.
    """
    def select(self, choice_scores):
        return self.choices[random.randint(0, len(self.choices) - 1)]


class UCB1(FrozenSelector):
    """
    The default FrozenSelector implementation.
    Uses the scores to create a vanilla UCB1 bandit and return its best arm.
    """
    def select(self, choice_scores):
        """
        Selects the arm which has the highest score as determined by UCB1 bandit
        algorithm.
        """
        choice_scores = {c: s for c, s in choice_scores.items()
                         if c in self.choices}
        arms = []
        for choice, scores in choice_scores.items():
            count = len(scores)
            rewards = sum(scores)
            arms.append(FrozenArm(count, rewards, choice))

        total_rewards = sum(a.rewards for a in arms)
        total_count = sum(a.count for a in arms)

        # so arms are not picked in ordinal ID order
        random.shuffle(arms)
        bandit = UCB1Bandit(arms, total_count, total_rewards)
        return bandit.score_arms()
