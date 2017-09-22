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
    AWS = "aws"
    DATAHUB = "datahub"
    DATA = "data"
    MODE = "mode"

    # subheading: data
    DATA_ALLDATAPATH = "alldatapath"
    DATA_TRAINPATH = "trainpath"
    DATA_TESTPATH = "testpath"
    DATA_DESCRIPTION = "data-description"
    DATA_FILELIST = "data-filelist"

    # subheading: mode
    MODE_RUNMODE = "run-mode"

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
    RUN_BTB = "btb-path"
    RUN_MODELS_DIR = "models-dir"
    RUN_DROP_VALUES = "drop-values"
    RUN_TEST_RATIO = 'test-ratio'
    RUN_NAME = 'name'
    RUN_PRIORITY = "priority"

    # subheading: budget
    BUDGET_TYPE = "budget-type"
    BUDGET_LEARNER = "learner-budget"
    BUDGET_WALLTIME = "walltime-budget"

    # subheading: strategy
    STRATEGY_SELECTION = "sample_selection"
    STRATEGY_FROZENS = "frozen_selection"
    STRATEGY_METRIC = "metric"
    STRATEGY_SCORE_TARGET = "score_target"
    STRATEGY_K = "k_window"
    STRATEGY_R = "r_min"

    # subheading: aws
    AWS_ACCESS_KEY = "access_key"
    AWS_SECRET_KEY = "secret_key"
    AWS_NUM_INSTANCES = "num_instances"
    AWS_NUM_WORKERS_PER_INSTACNCES = "num_workers_per_instance"
    AWS_S3_BUCKET = "s3_bucket"
    AWS_S3_FOLDER = "s3_folder"
    AWS_EC2_KEY_PAIR = "ec2_key_pair"
    AWS_EC2_INSTANCE_TYPE = "ec2_instance_type"
    AWS_EC2_REGION = "ec2_region"
    AWS_EC2_AMIS = "ec2_amis"
    AWS_EC2_USERNAME = "ec2_username"
    AWS_EC2_KEYFILE = "ec2_keyfile"

    # subheading: datahub
    DATAHUB_DIALECT = "dialect"
    DATAHUB_DATABASE = "database"
    DATAHUB_USERNAME = "username"
    DATAHUB_PASSWORD = "password"
    DATAHUB_HOST = "host"
    DATAHUB_PORT = "port"
    DATAHUB_QUERY = "query"

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

