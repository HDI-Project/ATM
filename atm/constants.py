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
