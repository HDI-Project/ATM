from ConfigParser import ConfigParser

class Config(ConfigParser):

    # headings
    CLOUD = "cloud"
    GIT = "git"
    IMG = "images"
    TYPE = "types"
    RUN = "run"
    BUDGET = "budget"
    STRATEGY = "strategy"
    
    # subheading: images
    IMG_WORKER = "worker-image"
    
    # subheading: types
    TYPE_WORKER = "worker-type"
    
    # subheading: Cloud
    CLOUD_SECURITY = "security-groups"
    CLOUD_USER = "username"
    CLOUD_PASS = "password"
    CLOUD_TENANT = "tenant-name"
    CLOUD_SERVICE = "service-type"
    CLOUD_AUTH_URL = "auth-url"
    CLOUD_KEY = "key"
    CLOUD_PORT = "port"
    
    # subheading: Git
    GIT_USER = "username"
    GIT_PASS = "password"
    GIT_REPO = "repo"
    
    # subheading: Run
    RUN_ALGORITHMS = "algorithms"
    RUN_DELPHI = "delphi-path"
    RUN_MODELS_DIR = "models-dir"
    RUN_DROP_VALUES = "drop-values"
    RUN_TEST_RATIO = 'test-ratio'
    RUN_NAME = 'name'
    
    # subheading: budget
    BUDGET_TYPE = "budget-type"
    BUDGET_LEARNER = "learner-budget"
    BUDGET_WALLTIME = "walltime-budget"
    
    # subheading: strategy
    STRATEGY_SELECTION = "sample_selection"
    STRATEGY_FROZENS = "frozen_selection"
    STRATEGY_METRIC = "metric"
    STRATEGY_K = "k_window"
    STRATEGY_R = "r_min"
    
    ###########################
    # constants
    CONST_NONE = "none"
    CONST_RANDOM = "random"
    CONST_LEARNER = "learner"
    CONST_WALLTIME = "walltime"
    
    def __init__(self, cfgpath):
        ConfigParser.__init__(self)
        self.configpath = cfgpath
        if self.configpath:
            self.read(self.configpath)
            #self.content = open(cfgpath, "r").read()
     