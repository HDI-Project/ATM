from __future__ import absolute_import

import logging
import os
import re
import socket
import sys
from argparse import ArgumentError, ArgumentTypeError, RawTextHelpFormatter

import yaml

from .constants import *
from .utilities import ensure_directory


class Config(object):
    """
    Class which stores configuration for one aspect of ATM. Subclasses of
    Config should define the list of all configurable parameters and any
    default values for those parameters other than None (in PARAMETERS and
    DEFAULTS, respectively). The object can be initialized with any number of
    keyword arguments; only kwargs that are in PARAMETERS will be used. This
    means you can (relatively) safely do things like
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
        'class_column',

        # datarun config
        'dataset_id',
        'methods',
        'priority',
        'budget_type',
        'budget',
        'deadline',
        'tuner',
        'r_minimum',
        'gridding',
        'selector',
        'k_window',
        'metric',
        'score_target'
    ]

    DEFAULTS = {
        'train_path': os.path.join(DATA_TEST_PATH, 'pollution_1.csv'),
        'class_column': 'class',
        'methods': ['logreg', 'dt', 'knn'],
        'priority': 1,
        'budget_type': 'classifier',
        'budget': 100,
        'tuner': 'uniform',
        'selector': 'uniform',
        'r_minimum': 2,
        'k_window': 3,
        'gridding': 0,
        'metric': 'f1',
        'score_target': 'cv',
    }


class LogConfig(Config):
    PARAMETERS = [
        'log_level_stdout',
        'log_level_file',
        'log_dir',
        'model_dir',
        'metric_dir',
        'verbose_metrics',
    ]

    DEFAULTS = {
        'log_level_stdout': 'ERROR',
        'log_level_file': 'INFO',
        'log_dir': 'logs',
        'model_dir': 'models',
        'metric_dir': 'metrics',
        'verbose_metrics': False,
    }


def initialize_logging(config):
    LEVELS = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NONE': logging.NOTSET
    }

    file_level = LEVELS.get(config.log_level_file.upper(), logging.CRITICAL)
    stdout_level = LEVELS.get(config.log_level_stdout.upper(), logging.CRITICAL)

    handlers = []
    if file_level > logging.NOTSET:
        fmt = '%(asctime)-15s %(name)s - %(levelname)s  %(message)s'
        ensure_directory(config.log_dir)
        path = os.path.join(config.log_dir, socket.gethostname() + '.txt')
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(file_level)
        handlers.append(handler)

    if stdout_level > logging.NOTSET:
        fmt = '%(message)s'
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(stdout_level)
        handlers.append(handler)

    if not len(handlers):
        handlers.append(logging.NullHandler())

    for lib in ['atm', 'btb']:
        logger = logging.getLogger(lib)
        logger.setLevel(min(file_level, stdout_level))

        for h in logger.handlers:
            logger.removeHandler(h)

        for h in handlers:
            logger.addHandler(h)

        logger.propagate = False
        logger.debug('Logging is active.')


def option_or_path(options, regex=CUSTOM_CLASS_REGEX):
    def type_check(s):
        # first, check whether the argument is one of the preconfigured options
        if s in options:
            return s

        # otherwise, check it against the regex, and try to pull out a path to a
        # real file. The regex must extract the path to the file as groups()[0].
        match = re.match(regex, s)
        if match and os.path.isfile(match.groups()[0]):
            return s

        # if both of those fail, there's something wrong
        raise ArgumentTypeError('%s is not a valid option or path!' % s)

    return type_check


def add_arguments_logging(parser):
    """
    Add all argparse arguments needed to parse logging configuration from the
    command line.
    parser: an argparse.ArgumentParser object
    """
    # Config file path
    parser.add_argument('--log-config', help='path to yaml logging config file')

    # paths to saved files
    parser.add_argument('--model-dir',
                        help='Directory where computed models will be saved')
    parser.add_argument('--metric-dir',
                        help='Directory where model metrics will be saved')
    parser.add_argument('--log-dir',
                        help='Directory where logs will be saved')

    # hoe much information to log or save
    parser.add_argument('--verbose-metrics', default=False, action='store_true',
                        help='If set, compute full ROC and PR curves and '
                        'per-label metrics for each classifier')
    parser.add_argument('--log-level-file',
                        help='minimum log level to write to the log file')
    parser.add_argument('--log-level-stdout',
                        help='minimum log level to write to stdout')

    return parser


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
    parser.add_argument('--num-instances', help='Number of EC2 instances to start')
    parser.add_argument('--num-workers-per-instance', help='Number of ATM workers per instances')
    parser.add_argument('--ec2-region', help='Region to start instances in')
    parser.add_argument('--ec2-ami', help='Name of ATM AMI')
    parser.add_argument('--ec2-key-pair', help='AWS key pair to use for EC2 instances')
    parser.add_argument('--ec2-keyfile', help='Local path to key file (must match ec2-key-pair)')
    parser.add_argument('--ec2-instance-type', help='Type of EC2 instance to start')
    parser.add_argument('--ec2-username', help='Username to log into EC2 instance')

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
    # make sure the text for these arguments is formatted correctly
    # this allows newlines in the help strings
    parser.formatter_class = RawTextHelpFormatter

    # Config file
    parser.add_argument('--run-config', help='path to yaml datarun config file')

    ##  Dataset Arguments  #####################################################
    ############################################################################
    parser.add_argument('--dataset-id', type=int,
                        help="ID of dataset, if it's already in the database")

    # These are only relevant if dataset_id is not provided
    parser.add_argument('--train-path', help='Path to raw training data')
    parser.add_argument('--test-path', help='Path to raw test data (if applicable)')
    parser.add_argument('--data-description', help='Description of dataset')
    parser.add_argument('--class-column', help='Name of the class column in the input data')

    ##  Datarun Arguments  #####################################################
    ############################################################################
    # Notes:
    # - Support vector machines (svm) can take a long time to train. It's not an
    #   error, it's just part of what happens when the method happens to explore
    #   a crappy set of parameters on a powerful algo like this.
    # - Stochastic gradient descent (sgd) can sometimes fail on certain
    #   parameter settings as well. Don't worry, they train SUPER fast, and the
    #   worker.py will simply log the error and continue.
    #
    # Method options:
    #   logreg - logistic regression
    #   svm    - support vector machine
    #   sgd    - linear classifier with stochastic gradient descent
    #   dt     - decision tree
    #   et     - extra trees
    #   rf     - random forest
    #   gnb    - gaussian naive bayes
    #   mnb    - multinomial naive bayes
    #   bnb    - bernoulli naive bayes
    #   gp     - gaussian process
    #   pa     - passive aggressive
    #   knn    - K nearest neighbors
    #   mlp    - multi-layer perceptron
    parser.add_argument('--methods', nargs='+',
                        type=option_or_path(METHODS, JSON_REGEX),
                        help='Method or list of methods to use for '
                        'classification. Each method can either be one of the '
                        'pre-defined method codes listed below or a path to a '
                        'JSON file defining a custom method.' +
                        '\n\nOptions: [%s]' % ', '.join(str(s) for s in METHODS))
    parser.add_argument('--priority', type=int,
                        help='Priority of the datarun (higher = more important')
    parser.add_argument('--budget-type', choices=BUDGET_TYPES,
                        help='Type of budget to use')
    parser.add_argument('--budget', type=int,
                        help='Value of the budget, either in classifiers or minutes')
    parser.add_argument('--deadline',
                        help='Deadline for datarun completion. If provided, this '
                        'overrides the configured walltime budget.\nFormat: ' +
                        TIME_FMT.replace('%', '%%'))

    # Which field to use to judge performance, for the sake of AutoML
    # options:
    #   f1        - F1 score (harmonic mean of precision and recall)
    #   roc_auc   - area under the Receiver Operating Characteristic curve
    #   accuracy  - percent correct
    #   cohen_kappa     - measures accuracy, but controls for chance of guessing
    #                     correctly
    #   rank_accuracy   - multiclass only: percent of examples for which the true
    #                     label is in the top 1/3 most likely predicted labels
    #   ap        - average precision: nearly identical to area under
    #               precision/recall curve.
    #   mcc       - matthews correlation coefficient: good for unbalanced classes
    #
    # f1 and roc_auc may be appended with _micro or _macro to use with
    # multiclass problems.
    parser.add_argument('--metric', choices=METRICS,
                        help='Metric by which ATM should evaluate classifiers. '
                        'The metric function specified here will be used to '
                        'compute the "judgment metric" for each classifier.')

    # Which data to use for computing judgment score
    #   cv   - cross-validated performance on training data
    #   test - performance on test data
    #   mu_sigma - lower confidence bound on cv score
    parser.add_argument('--score-target', choices=SCORE_TARGETS,
                        help='Determines which judgment metric will be used to '
                        'search the hyperparameter space. "cv" will use the mean '
                        'cross-validated performance, "test" will use the '
                        'performance on a test dataset, and "mu_sigma" will use '
                        'the lower confidence bound on the CV performance.')

    ##  AutoML Arguments  ######################################################
    ############################################################################
    # hyperparameter selection strategy
    # How should ATM sample hyperparameters from a given hyperpartition?
    #    uniform  - pick randomly! (baseline)
    #    gp       - vanilla Gaussian Process
    #    gp_ei    - Gaussian Process expected improvement criterion
    #    gp_eivel - Gaussian Process expected improvement, with randomness added
    #               in based on velocity of improvement
    #   path to custom tuner, defined in python
    parser.add_argument('--tuner', type=option_or_path(TUNERS),
                        help='Type of BTB tuner to use. Can either be one of '
                        'the pre-configured tuners listed below or a path to a '
                        'custom tuner in the form "/path/to/tuner.py:ClassName".'
                        '\n\nOptions: [%s]' % ', '.join(str(s) for s in TUNERS))

    # How should ATM select a particular hyperpartition from the set of all
    # possible hyperpartitions?
    # Options:
    #   uniform      - pick randomly
    #   ucb1         - UCB1 multi-armed bandit
    #   bestk        - MAB using only the best K runs in each hyperpartition
    #   bestkvel     - MAB with velocity of best K runs
    #   purebestkvel - always return hyperpartition with highest velocity
    #   recentk      - MAB with most recent K runs
    #   recentkvel   - MAB with velocity of most recent K runs
    #   hieralg      - hierarchical MAB: choose a classifier first, then choose
    #                  a partition
    #   path to custom selector, defined in python
    parser.add_argument('--selector', type=option_or_path(SELECTORS),
                        help='Type of BTB selector to use. Can either be one of '
                        'the pre-configured selectors listed below or a path to a '
                        'custom tuner in the form "/path/to/selector.py:ClassName".'
                        '\n\nOptions: [%s]' % ', '.join(str(s) for s in SELECTORS))

    # r_minimum is the number of random runs performed in each hyperpartition before
    # allowing bayesian opt to select parameters. Consult the thesis to
    # understand what those mean, but essentially:
    #
    #  if (num_classifiers_trained_in_hyperpartition >= r_minimum)
    #    # train using sample criteria
    #  else
    #    # train using uniform (baseline)
    parser.add_argument('--r-minimum', type=int,
                        help='number of random runs to perform before tuning can occur')

    # k is number that xxx-k methods use. It is similar to r_minimum, except it is
    # called k_window and determines how much "history" ATM considers for certain
    # partition selection logics.
    parser.add_argument('--k-window', type=int,
                        help='number of previous scores considered by -k selector methods')

    # gridding determines whether or not sample selection will happen on a grid.
    # If any positive integer, a grid with `gridding` points on each axis is
    # established, and hyperparameter vectors are sampled from this finite
    # space. If 0 (or blank), hyperparameters are sampled from continuous
    # space, and there is no limit to the number of hyperparameter vectors that
    # may be tried.
    parser.add_argument('--gridding', type=int,
                        help='gridding factor (0: no gridding)')

    return parser


def load_config(sql_path=None, run_path=None, aws_path=None, log_path=None, **kwargs):
    """
    Load config objects from yaml files and command line arguments. Command line
    args override yaml files where applicable.

    Args:
        sql_path: path to .yaml file with SQL configuration
        run_path: path to .yaml file with Dataset and Datarun configuration
        aws_path: path to .yaml file with AWS configuration
        log_path: path to .yaml file with logging configuration
        **kwargs: miscellaneous arguments specifying individual configuration
            parameters. Any kwargs beginning with sql_ are SQL config
            arguments, any beginning with aws_ are AWS config.

    Returns: sql_conf, run_conf, aws_conf, log_conf
    """
    sql_args = {}
    run_args = {}
    aws_args = {}
    log_args = {}

    # kwargs are most likely generated by argparse.
    # Any unspecified argparse arguments will be None, so ignore those. We only
    # care about arguments explicitly specified by the user.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # check the keyword args for config paths
    sql_path = sql_path or kwargs.get('sql_config')
    run_path = run_path or kwargs.get('run_config')
    aws_path = aws_path or kwargs.get('aws_config')

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

    if log_path:
        with open(log_path) as f:
            log_args = yaml.load(f)

    # Use keyword args to override yaml config values
    sql_args.update({k.replace('sql_', ''): v for k, v in kwargs.items()
                     if 'sql_' in k})
    aws_args.update({k.replace('aws_', ''): v for k, v in kwargs.items()
                     if 'aws_' in k})
    run_args.update({k: v for k, v in kwargs.items() if k in
                     RunConfig.PARAMETERS})
    log_args.update({k: v for k, v in kwargs.items() if k in
                     LogConfig.PARAMETERS})

    # It's ok if there are some extra arguments that get passed in here; only
    # kwargs that correspond to real config values will be stored on the config
    # objects.
    sql_conf = SQLConfig(**sql_args)
    aws_conf = AWSConfig(**aws_args)
    run_conf = RunConfig(**run_args)
    log_conf = LogConfig(**log_args)

    return sql_conf, run_conf, aws_conf, log_conf
