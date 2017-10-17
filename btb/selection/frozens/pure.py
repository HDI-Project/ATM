from btb.selection.frozens import FrozenSelector
import numpy as np

K_MIN = 3

class PureBestKVelocity(FrozenSelector):
    def __init__(self, choices, **kwargs):
        """
        Needs:
        k, frozen_sets, metric
        """
        super(PureBestKVelocity, self).__init__(choices, **kwargs)
        self.k = kwargs.pop('k', K_MIN)

    def select(self, choice_scores):
        """
        Select the choice with the highest best-K velocity. If any choices
        don't have MIN_K scores yet, return the one with the fewest.
        """
        choice_scores = {c: s for c, s in choice_scores if c in self.choices}
        score_counts = {c: len(s) for c, s in choice_scores}
        if min(score_counts.values()) < K_MIN:
            print "We don't have enough frozen trials for this k! Attempt to "\
                  "get all sets to same K_MIN..."
            # return the choice with the fewest scores so far
            return min(score_counts, key=score_counts.get)

        velocities = {}
        for c, scores in choice_scores:
            # truncate to the highest k scores and compute the velocity of those
            scores = sorted(scores)[-self.k:]
            velocities[c] = np.mean([scores[i+1] - scores[i] for i in
                                     range(len(scores) - 1)])

        # return the frozen set with higest average velocity
        return max(velocities, key=velocities.get)
