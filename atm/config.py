import yaml
from argparse import ArgumentError

from atm.constants import *


class Config(object):
    """
    Class which stores configuration for one aspect of ATM. Subclasses of
    Config should define, in PARAMETERS and DEFAULTS, respectively, the list of
    all configurable parameters and any default values for those parameters
    other than None. The object can be initialized with any number of keyword
    arguments; only kwargs that are in PARAMETERS will be used. This means you
    can (relatively) safely do things like
        args = parser.parse_args()
        conf = Config(**vars(args))
    and only relevant parameters will be set.

    Subclasses do not need to define __init__ or any other methods.
    """
    # list of all parameters which may be set on this config object
    PARAMETERS = []
    # default values for all required parameters
    DEFAULTS = {}

    def __init__(self, **kwargs):
        for key in self.PARAMETERS:
            value = kwargs.get(key)

            # Here, if a keyword argument is set to None, it will be overridden
            # by the default value. AFAIK, this is the only way to deal with
            # keyword args passed in from argparse that weren't set on the
            # command line. That means you shouldn't define any PARAMETERS for
            # which None is a meaningful value; if you do, make sure None is
            # also the default.
            if key in self.DEFAULTS and value is None:
                value = self.DEFAULTS[key]

            setattr(self, key, value)


class AWSConfig(Config):
    """ Stores configuration for AWS S3 and EC2 connections """
    PARAMETERS = [
        # universal config
        'access_key',
        'secret_key',

        # s3 config
        's3_bucket',
        's3_folder',

        # ec2 config
        'ec2_region',
        'ec2_amis',
        'ec2_key_pair',
        'ec2_keyfile',
        'ec2_instance_type',
        'ec2_username',
        'num_instances',
        'num_workers_per_instance'
    ]

    DEFAULTS = {}


class SQLConfig(Config):
    """ Stores configuration for SQL database setup & connection """
    PARAMETERS = [
        'dialect',
        'database',
        'username',
        'password',
        'host',
        'port',
        'query'
    ]

    DEFAULTS = {
        'dialect': 'sqlite',
        'database': 'atm.db',
    }


class RunConfig(Config):
    """ Stores configuration for Dataset and Datarun setup """
    PARAMETERS = [
        # dataset config
        'train_path',
        'test_path',
        'data_description',
        'output_folder',
        'label_column',

        # datarun config
        'dataset_id',
        'algorithms',
        'models_dir',
        'priority',
        'budget_type',
        'budget',
        'deadline',
        'tuner',
        'r_min',
        'gridding',
        'selector',
        'k_window',
        'metric',
        'score_target'
    ]

    DEFAULTS = {
        'train_path': 'data/test/pollution_1.csv',
        'output_folder': 'data/processed/',
        'label_column': 'class',
        'algorithms': ['logreg', 'dt', 'knn'],
        'priority': 1,
        'budget_type': 'learner',
        'budget': 100,
        'tuner': 'uniform',
        'selector': 'uniform',
        'r_min': 2,
        'k_window': 3,
        'gridding': 0,
        'metric': 'f1',
        'score_target': 'cv',
    }


def add_arguments_aws_s3(parser):
    """
    Add all argparse arguments needed to parse AWS S3 configuration from the
    command line. This is separate from aws_ec2 because usually only one set of
    arguments or the other is needed.

    parser: an argparse.ArgumentParser object
    """
    # Config file
    parser.add_argument('--aws-config', help='path to yaml AWS config file')

    # All of these arguments must start with --aws-, and must correspond to
    # keys present in the AWS config example file.
    # AWS API access key pair
    # try... catch because this might be called after aws_s3
    try:
        parser.add_argument('--aws-access-key', help='AWS access key')
        parser.add_argument('--aws-secret-key', help='AWS secret key')
    except ArgumentError:
        pass

    # S3-specific arguments
    parser.add_argument('--aws-s3-bucket', help='AWS S3 bucket to store data')
    parser.add_argument('--aws-s3-folder', help='Folder in AWS S3 bucket in which to store data')

    return parser


def add_arguments_aws_ec2(parser):
    """
    Add all argparse arguments needed to parse AWS EC2 configuration from the
    command line. This is separate from aws_s3 because usually only one set of
    arguments or the other is needed.

    parser: an argparse.ArgumentParser object
    """
    # Config file
    parser.add_argument('--aws-config', help='path to yaml AWS config file')

    # All of these arguments must start with --aws-, and must correspond to
    # keys present in the AWS config example file.
    # AWS API access key pair
    # try... catch because this might be called after aws_s3
    try:
        parser.add_argument('--aws-access-key', help='AWS access key')
        parser.add_argument('--aws-secret-key', help='AWS secret key')
    except ArgumentError:
        pass

    # AWS EC2 configurations
    parser.add-argument('--num-instances', help='Number of EC2 instances to start')
    parser.add-argument('--num-workers-per-instance', help='Number of ATM workers per instances')
    parser.add-argument('--ec2-region', help='Region to start instances in')
    parser.add-argument('--ec2-ami', help='Name of ATM AMI')
    parser.add-argument('--ec2-key-pair', help='AWS key pair to use for EC2 instances')
    parser.add-argument('--ec2-keyfile', help='Local path to key file (must match ec2-key-pair)')
    parser.add-argument('--ec2-instance-type', help='Type of EC2 instance to start')
    parser.add-argument('--ec2-username', help='Username to log into EC2 instance')

    return parser


def add_arguments_sql(parser):
    """
    Add all argparse arguments needed to parse configuration for the ModelHub
    SQL database from the command line.

    parser: an argparse.ArgumentParser object
    """
    # Config file
    parser.add_argument('--sql-config', help='path to yaml SQL config file')

    # All of these arguments must start with --sql-, and must correspond to
    # keys present in the SQL config example file.
    parser.add_argument('--sql-dialect', choices=SQL_DIALECTS,
                        help='Dialect of SQL to use')
    parser.add_argument('--sql-database',
                        help='Name of, or path to, SQL database')
    parser.add_argument('--sql-username', help='Username for SQL database')
    parser.add_argument('--sql-password', help='Password for SQL database')
    parser.add_argument('--sql-host', help='Hostname for database machine')
    parser.add_argument('--sql-port', help='Port used to connect to database')
    parser.add_argument('--sql-query', help='Specify extra login details')

    return parser


def add_arguments_datarun(parser):
    """
    Add all argparse arguments needed to parse dataset and datarun configuration
    from the command line.

    parser: an argparse.ArgumentParser object
    """
    # Config file
    parser.add_argument('--run-config', help='path to yaml datarun config file')

    ##  Dataset Arguments  #########################################################
    ################################################################################
    parser.add_argument('--dataset-id', type=int,
                        help="ID of dataset, if it's already in the database")

    # These are only relevant if dataset_id is not provided
    parser.add_argument('--train-path', help='Path to raw training data')
    parser.add_argument('--test-path', help='Path to raw test data (if applicable)')
    parser.add_argument('--data-description', help='Description of dataset')
    parser.add_argument('--output-folder', help='Path where processed data will be saved')
    parser.add_argument('--label-column', help='Name of the label column in the input data')

    ##  Datarun Arguments  #########################################################
    ################################################################################
    # Notes:
    # - Support vector machines (svm) can take a long time to train. It's not an
    #   error. It's justpart of what happens  when the algorithm happens to explore
    #   a crappy set of parameters on a powerful algo like this.
    # - Stochastic gradient descent (sgd) can sometimes fail on certain parameter
    #   settings as well. Don't worry, they train SUPER fast, and the worker.py will
    #   simply log the error and continue.
    #
    # Algorithm options:
    #   logreg - logistic regression
    #   svm    - support vector machine
    #   sgd    - linear classifier (SVM or logreg) using stochastic gradient descent
    #   dt     - decision tree
    #   et     - extra trees
    #   rf     - random forest
    #   gnb    - gaussian naive bayes
    #   mnb    - multinomial naive bayes
    #   bnb    - bernoulli naive bayes
    #   gp     - gaussian process
    #   pa     - passive aggressive
    #   knn    - K nearest neighbors
    #   dbn    - deep belief network
    #   mlp    - multi-layer perceptron
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS,
                        help='list of algorithms which the datarun will use')
    parser.add_argument('--priority', type=int,
                        help='Priority of the datarun (higher = more important')
    parser.add_argument('--budget-type', choices=BUDGET_TYPES,
                        help='Type of budget to use')
    parser.add_argument('--budget', type=int,
                        help='Value of the budget, either in learners or minutes')
    parser.add_argument('--deadline',
                        help='Deadline for datarun completion. If provided, this '
                        'overrides the walltime budget. Format: ' + TIME_FMT)

    # hyperparameter selection strategy
    # How should ATM sample hyperparameters from a given frozen set?
    #    uniform  - pick randomly! (baseline)
    #    gp       - vanilla Gaussian Process
    #    gp_ei    - Gaussian Process expected improvement criterion
    #    gp_eivel - Gaussian Process expected improvement, with randomness added in
    #              based on velocity of improvement
    parser.add_argument('--tuner', choices=TUNERS,
                        help='type of BTB tuner to use')

    # How should ATM select a particular hyperpartition (frozen set) from the
    # set of all hyperpartitions?
    # Options:
    #   uniform      - pick randomly
    #   ucb1         - UCB1 multi-armed bandit
    #   bestk        - MAB using only the best K runs in each frozen set
    #   bestkvel     - MAB with velocity of best K runs
    #   purebestkvel - always return frozen set with highest velocity
    #   recentk      - MAB with most recent K runs
    #   recentkvel   - MAB with velocity of most recent K runs
    #   hieralg      - hierarchical MAB: choose a classifier first, then choose frozen
    parser.add_argument('--selector', choices=SELECTORS,
                        help='type of BTB selector to use')

    # r_min is the number of random runs performed in each hyperpartition before
    # allowing bayesian opt to select parameters. Consult the thesis to understand
    # what those mean, but essentially:
    #
    #  if (num_learners_trained_in_hyperpartition >= r_min)
    #    # train using sample criteria
    #  else
    #    # train using uniform (baseline)
    parser.add_argument('--r-min',  type=int,
                        help='number of random runs to perform before tuning can occur')

    # k is number that xxx-k methods use. It is similar to r_min, except it is
    # called k_window and determines how much "history" ATM considers for certain
    # frozen selection logics.
    parser.add_argument('--k-window', type=int,
                        help='number of previous scores considered by -k selector methods')

    # gridding determines whether or not sample selection will happen on a grid.
    # If any positive integer, a grid with `gridding` points on each axis is
    # established, and hyperparameter vectors are sampled from this finite space.
    # If 0 (or blank), hyperparameters are sampled from continuous space, and there
    # is no limit to the number of hyperparameter vectors that may be tried.
    parser.add_argument('--gridding', type=int,
                        help='gridding factor (0: no gridding)')

    # Which field to use for judgment of performance
    # options:
    #   f1        - F1 score (harmonic mean of precision and recall)
    #   roc_auc   - area under the Receiver Operating Characteristic curve
    #   accuracy  - percent correct
    #   mu_sigma  - one standard deviation below the average cross-validated F1
    #               score (mu - sigma)
    parser.add_argument('--metric', choices=METRICS, help='type of BTB selector to use')

    # Which data to use for computing judgment score
    #   cv   - cross-validated performance on training data
    #   test - performance on test data
    parser.add_argument('--score-target', choices=SCORE_TARGETS,
                        help='whether to compute metrics by cross-validation or on '
                        'test data (if available)')

    return parser


def load_config(sql_path=None, run_path=None, aws_path=None, args=None):
    """
    Load config objects from yaml files and command line arguments. Command line
    args override yaml files where applicable.

    Args:
        sql_path: path to .yaml file with SQL config info
        run_path: path to .yaml file with Dataset and Datarun config info
        aws_path: path to .yaml file with AWS config info
        args: Namespace object with miscellaneous arguments attached to it as
            attributes (the type that is generated by parser.parse_arguments()). Any
            attributes beginning with sql_ are SQL config arguments, any beginning
            with aws_ are AWS config, and the rest are assumed to be dataset or
            datarun config.

    Returns: sql_conf, run_conf, aws_conf
    """
    sql_args = {}
    run_args = {}
    aws_args = {}

    # check the args object for config paths
    if args is not None:
        arg_vars = vars(args)
        sql_path = sql_path or arg_vars.get('sql_config')
        run_path = run_path or arg_vars.get('run_config')
        aws_path = aws_path or arg_vars.get('aws_config')

    # load any yaml config files for which paths were provided
    if sql_path:
        with open(sql_path) as f:
            sql_args = yaml.load(f)

    if run_path:
        with open(run_path) as f:
            run_args = yaml.load(f)

    if aws_path:
        with open(aws_path) as f:
            aws_args = yaml.load(f)

    # Use args to override yaml config values
    # Any unspecified argparse arguments will be None, so ignore those. We only
    # care about arguments explicitly specified by the user.
    if args is not None:
        sql_args.update({k.replace('sql_', ''): v for k, v in arg_vars.items()
                         if 'sql_' in k and v is not None})
        aws_args.update({k.replace('aws_', ''): v for k, v in arg_vars.items()
                         if 'aws_' in k and v is not None})
        run_args.update({k: v for k, v in arg_vars.items() if 'sql_' not in k
                         and 'aws_' not in k and v is not None})

    # It's ok if there are some extra arguments that get passed in here; only
    # kwargs that correspond to real config values will be stored on the config
    # objects.
    sql_conf = SQLConfig(**sql_args)
    aws_conf = AWSConfig(**aws_args)
    run_conf = RunConfig(**run_args)

    return sql_conf, run_conf, aws_conf
