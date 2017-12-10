# sample tuning
from btb.tuning import Uniform as UniformTuner, GP, GPEi, GPEiVelocity
# frozen set selectors
from btb.selection import Uniform as UniformSelector, UCB1,\
                                     BestKReward, BestKVelocity,\
                                     RecentKReward, RecentKVelocity,\
                                     HierarchicalByAlgorithm, PureBestKVelocity

# TODO: figure out how to handle these better
SQL_DIALECTS = ['sqlite', 'mysql']
METRICS = ['f1', 'roc_auc', 'accuracy', 'mu_sigma']
SCORE_TARGETS = ['cv', 'test']
BUDGET_TYPES = ['none', 'learner', 'walltime']
ALGORITHMS = ['logreg', 'svm', 'sgd', 'dt', 'et', 'rf', 'gnb', 'mnb', 'bnb',
              'gp', 'pa', 'knn', 'dbn', 'mlp']
TUNERS = ['uniform', 'gp', 'gp_ei', 'gp_eivel']
SELECTORS = ['uniform', 'ucb1', 'bestk', 'bestkvel', 'purebestkvel', 'recentk',
             'recentkvel', 'hieralg']
DATARUN_STATUS = ['pending', 'running', 'complete']
LEARNER_STATUS = ['running', 'errored', 'complete']
FROZEN_STATUS = ['incomplete', 'errored', 'gridding_done']

TIME_FMT = "%y-%m-%d %H:%M"

TUNERS_MAP = {
    'uniform': UniformTuner,
    'gp': GP,
    'gp_ei': GPEi,
    'gp_eivel': GPEiVelocity,
}

SELECTORS_MAP = {
    'uniform': UniformSelector,
    'ucb1': UCB1,
    'bestk': BestKReward,
    'bestkvel': BestKVelocity,
    'purebestkvel': PureBestKVelocity,
    'recentk': RecentKReward,
    'recentkvel': RecentKVelocity,
    'hieralg': HierarchicalByAlgorithm,
}

ALGORITHMS_MAP = {
    'logreg': 'logistic_regression.json',
    'svm': 'support_vector_machine.json',
    'sgd': 'stochastic_gradient_descent.json',
    'dt': 'decision_tree.json',
    'et': 'extra_trees.json',
    'rf': 'random_forest.json',
    'gnb': 'gaussian_naive_bayes.json',
    'mnb': 'multinomial_naive_bayes.json',
    'bnb': 'bernoulli_naive_bayes.json',
    'gp': 'gaussian_process.json',
    'pa': 'passive_aggressive.json',
    'knn': 'k_nearest_neighbors.json',
    'dbn': 'deep_belief_network.json',
    'mlp': 'multi_layer_perceptron.json',
}

class LearnerStatus:
    RUNNING = 'running'
    ERRORED = 'errored'
    COMPLETE = 'complete'

class RunStatus:
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETE = 'complete'

class FrozenStatus:
    INCOMPLETE = 'incomplete'
    GRIDDING_DONE = 'gridding_done'
    ERRORED = 'errored'
