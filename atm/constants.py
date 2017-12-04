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
LEARNER_STATUS = ['started', 'errored', 'complete']

class LearnerStatus:
    STARTED = 'started'
    ERRORED = 'errored'
    COMPLETE = 'complete'

class RunStatus:
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETE = 'complete'
