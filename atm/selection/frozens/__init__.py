from atm.selection import Selector
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

class FrozenSelector(Selector):
	def __init__(self, **kwargs):
		super(FrozenSelector, self).__init__(**kwargs)
		
def AverageVelocitiyFromList(seq, is_ascending):
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