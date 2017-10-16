import numpy as np

SELECTION_FROZENS_UNIFORM = "uniform"
SELECTION_FROZENS_UCB1 = "ucb1"
SELECTION_FROZENS_BEST_K = "bestk"
SELECTION_FROZENS_BEST_K_VEL = "bestkvel"
SELECTION_FROZENS_RECENT_K = "recentk"
SELECTION_FROZENS_RECENT_K_VEL = "recentkvel"
SELECTION_FROZENS_HIER_ALG = "hieralg"
SELECTION_FROZENS_HIER_RAND = "hierrand"
SELECTION_FROZENS_PURE_BEST_K_VEL = "purebestkvel"


class FrozenSelector(object):
    def __init__(self, choices, **kwargs):
        """
        choices: a list of discrete choices from which the selector must choose
        at every call to select().
        """
        self.choices = choices

    def select(self, scores):
        """
        Select the next best choice to make
        scores: map of {choice -> [scores]} for each choice
        score lists should be in ascending chronological order (earliest first)
        e.g.
        {
            1: [0.56, 0.61, 0.33, 0.67],
            2: [0.25, 0.58],
            3: [0.60, 0.65, 0.68]
        }
        """
        pass


def AverageVelocityFromList(seq, is_ascending):
	"""
	Assumes unit (1) time steps in between measurements.

	A list of n elements will use (n-1) differences to calculate
	the average.
	"""
	if not is_ascending:
		# reverse, always use ascending
		seq = seq[::-1]

	velocities = []
	for i, score in enumerate(seq):
		if i < len(seq) - 1:
			velocities.append(seq[i+1] - seq[i])

	return np.mean(velocities)
