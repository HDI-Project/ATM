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

class Defaults:
    """ Default values for all required arguments """
    SQL_DIALECT = 'sqlite'
    DATABASE = 'atm.db'
    TRAIN_PATH = 'data/pollution_1.csv'
    OUTPUT_FOLDER = 'data/processed'
    LABEL_COLUMN = 'class'
    ALGORITHMS = ['logreg', 'dt', 'gp', 'knn']
    PRIORITY = 1
    BUDGET_TYPE = 'learner'
    LEARNER_BUDGET = 100
    WALLTIME_BUDGET = 10    # minutes
    TUNER = 'uniform'
    SELECTOR = 'uniform'
    R_MIN = 2
    K_WINDOW = 3
    GRIDDING = 0        # no gridding
    METRIC = 'f1'
    SCORE_TARGET = 'cv'
