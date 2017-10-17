import numpy as np
from frozen_selector import FrozenSelector, Uniform, UCB1
from best import BestKReward, BestKVelocity
from pure import PureBestKVelocity
from recent import RecentKReward, RecentKVelocity
from hierarchical import HierarchicalByAlgorithm, # HierarchicalRandom

SELECTION_FROZENS_UNIFORM = "uniform"
SELECTION_FROZENS_UCB1 = "ucb1"
SELECTION_FROZENS_BEST_K = "bestk"
SELECTION_FROZENS_BEST_K_VEL = "bestkvel"
SELECTION_FROZENS_PURE_BEST_K_VEL = "purebestkvel"
SELECTION_FROZENS_RECENT_K = "recentk"
SELECTION_FROZENS_RECENT_K_VEL = "recentkvel"
SELECTION_FROZENS_HIER_ALG = "hieralg"
#SELECTION_FROZENS_HIER_RAND = "hierrand"
