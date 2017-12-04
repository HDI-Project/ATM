import yaml


class Defaults:
    """ Default values for all required arguments """
    SQL_DIALECT = 'sqlite'
    DATABASE = 'atm.db'
    TRAIN_PATH = 'data/samples/pollution_1.csv'
    OUTPUT_FOLDER = 'data/processed/'
    LABEL_COLUMN = 'class'
    ALGORITHMS = ['logreg', 'dt', 'gp', 'knn']
    PRIORITY = 1
    BUDGET_TYPE = 'learner'
    BUDGET = {'learner': 100, 'walltime': 10}
    TUNER = 'uniform'
    SELECTOR = 'uniform'
    R_MIN = 2
    K_WINDOW = 3
    GRIDDING = 0        # no gridding
    METRIC = 'f1'
    SCORE_TARGET = 'cv'


class AWSConfig(object):
    """ Stores configuration for AWS S3 and EC2 connections """
    def __init__(self, **kwargs):
        # universal config
        self.access_key = kwargs.get('access_key')
        self.secret_key = kwargs.get('secret_key')

        # s3 config
        self.s3_bucket = kwargs.get('s3_bucket')
        self.s3_folder = kwargs.get('s3_folder')

        # ec2 config
        self.ec2_region = kwargs.get('ec2_region')
        self.ec2_amis = kwargs.get('ec2_amis')
        self.ec2_key_pair = kwargs.get('ec2_key_pair')
        self.ec2_keyfile = kwargs.get('ec2_keyfile')
        self.ec2_instance_type = kwargs.get('ec2_instance_type')
        self.ec2_username = kwargs.get('ec2_username')
        self.num_instances = kwargs.get('num_instances')
        self.num_workers_per_instance = kwargs.get('num_workers_per_instance')


class SQLConfig(object):
    """ Stores configuration for SQL database setup & connection """
    def __init__(self, **kwargs):
        self.dialect = kwargs.get('dialect')
        self.database = kwargs.get('database')
        self.username = kwargs.get('username')
        self.password = kwargs.get('password')
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.query = kwargs.get('query')


class RunConfig(object):
    """ Stores configuration for Dataset and Datarun setup """
    def __init__(self, **kwargs):
        # dataset config
        self.train_path = kwargs.get('train_path')
        self.test_path = kwargs.get('test_path')
        self.data_description = kwargs.get('data_description')
        self.output_folder = kwargs.get('output_folder')
        self.label_column = kwargs.get('label_column')

        # datarun config
        self.dataset_id = kwargs.get('dataset_id')
        self.algorithms = kwargs.get('algorithms')
        self.models_dir = kwargs.get('models_dir')
        self.priority = kwargs.get('priority')
        self.budget_type = kwargs.get('budget_type')
        self.budget = kwargs.get('budget')
        self.deadline = kwargs.get('deadline')
        self.tuner = kwargs.get('tuner')
        self.r_min = kwargs.get('r_min')
        self.gridding = kwargs.get('gridding')
        self.selector = kwargs.get('selector')
        self.k_window = kwargs.get('k_window')
        self.metric = kwargs.get('metric')
        self.score_target = kwargs.get('score_target')


def load_config(sql_path=None, aws_path=None, run_path=None, args=None):
    """
    Load config objects from yaml files and command line arguments. Command line
    args override yaml files where applicable.

    sql_path: path to .yaml file with SQL config info
    aws_path: path to .yaml file with AWS config info
    run_path: path to .yaml file with Dataset and Datarun config info
    args: object with miscellaneous arguments attached to it as attributes (the
        type that is produced by ap.parse_arguments()). Any attributes beginning
        with sql_ are sql config arguments, any beginning with aws_ are aws
        config args, and the rest are assumed to be run config.
    """
    sql_args = {}
    aws_args = {}
    run_args = {}

    if sql_path:
        with open(sql_path) as f:
            sql_args = yaml.load(f)

    if aws_path:
        with open(aws_path) as f:
            aws_args = yaml.load(f)

    if run_path:
        with open(run_path) as f:
            run_args = yaml.load(f)

    # use args to override some yaml config values
    if args is not None:
        sql_args.update({k.replace('sql_', ''): v for k, v in vars(args).items()
                         if 'sql_' in k})
        aws_args.update({k.replace('aws_', ''): v for k, v in vars(args).items()
                         if 'aws_' in k})
        run_args.update({k: v for k, v in vars(args).items()
                         if 'sql_' not in k and 'aws_' not in k})

    # It's ok if there are some extra arguments that get passed in here; only
    # kwargs that correspond to real config values will be stored on the config
    # objects.
    sqlconf = SQLConfig(**sql_args)
    awsconf = AWSConfig(**aws_args)
    runconf = RunConfig(**run_args)

    return sqlconf, awsconf, runconf
