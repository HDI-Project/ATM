import argparse
import os
import warnings
import yaml

from datetime import datetime, timedelta
from boto.s3.connection import S3Connection, Key as S3Key

from atm.datawrapper import DataWrapper
from atm.constants import *
from atm.config import *
from atm.utilities import ensure_directory, hash_nested_tuple
from atm.mapping import frozen_sets_from_algorithm_codes
from atm.database import Database


warnings.filterwarnings("ignore")

TIME_FMT = "%y-%m-%d %H:%M"

parser = argparse.ArgumentParser(description="""
Creates a dataset (if necessary) and a datarun and adds them to the ModelHub.
All required arguments have default values. Running this script with no
arguments will create a new dataset with the file in data/pollution_1.csv and a
new datarun with the default arguments listed below.

You can pass yaml configuration files (--sql-config, --aws-config, --run-config)
instead of passing individual arguments. Any arguments in the config files will
override arguments passed on the command line. See the examples in the config/
folder for more information. """)


##  Config files  #############################################################
###############################################################################
parser.add_argument('--sql-config', help='path to yaml SQL config file')
parser.add_argument('--aws-config', help='path to yaml AWS config file')
parser.add_argument('--run-config', help='path to yaml datarun config file')


##  Database Arguments  ########################################################
################################################################################
# All of these arguments must start with --sql-, and must correspond to
# keys present in the SQL config example file.
parser.add_argument('--sql-dialect', choices=SQL_DIALECTS,
                    default=Defaults.SQL_DIALECT, help='Dialect of SQL to use')
parser.add_argument('--sql-database', default=Defaults.DATABASE,
                    help='Name of, or path to, SQL database')
parser.add_argument('--sql-username', help='Username for SQL database')
parser.add_argument('--sql-password', help='Password for SQL database')
parser.add_argument('--sql-host', help='Hostname for database machine')
parser.add_argument('--sql-port', help='Port used to connect to database')
parser.add_argument('--sql-query', help='Specify extra login details')


##  AWS Arguments  #############################################################
################################################################################
# All of these arguments must start with --aws-, and must correspond to
# keys present in the AWS config example file.
parser.add_argument('--aws-access-key', help='AWS access key')
parser.add_argument('--aws-secret-key', help='AWS secret key')
parser.add_argument('--aws-s3-bucket', help='AWS S3 bucket to store data')
parser.add_argument('--aws-s3-folder', help='Folder in AWS S3 bucket in which to store data')


##  Dataset Arguments  #########################################################
################################################################################
parser.add_argument('--dataset-id', type=int,
                    help="ID of dataset, if it's already in the database")
parser.add_argument('--train-path', default=Defaults.TRAIN_PATH,
                    help='Path to raw training data')
parser.add_argument('--test-path', help='Path to raw test data (if applicable)')
parser.add_argument('--data-description', help='Description of dataset')
parser.add_argument('--output-folder', default=Defaults.OUTPUT_FOLDER,
                    help='Path where processed data will be saved')
parser.add_argument('--label-column', default=Defaults.LABEL_COLUMN,
                    help='Name of the label column in the input data')
parser.add_argument('--upload-data', action='store_true',
                    help='Whether to upload processed data to s3')


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
                    default=list(Defaults.ALGORITHMS),
                    help='list of algorithms which the datarun will use')
parser.add_argument('--priority', type=int, default=Defaults.PRIORITY,
                    help='Priority of the datarun (higher = more important')
parser.add_argument('--budget-type', choices=BUDGET_TYPES,
                    default=Defaults.BUDGET_TYPE, help='Type of budget to use')
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
parser.add_argument('--tuner', choices=TUNERS, default=Defaults.TUNER,
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
parser.add_argument('--selector', choices=SELECTORS, default=Defaults.SELECTOR,
                    help='type of BTB selector to use')

# r_min is the number of random runs performed in each hyperpartition before
# allowing bayesian opt to select parameters. Consult the thesis to understand
# what those mean, but essentially:
#
#  if (num_learners_trained_in_hyperpartition >= r_min)
#    # train using sample criteria
#  else
#    # train using uniform (baseline)
parser.add_argument('--r-min',  type=int, default=Defaults.R_MIN,
                    help='number of random runs to perform before tuning can occur')

# k is number that xxx-k methods use. It is similar to r_min, except it is
# called k_window and determines how much "history" ATM considers for certain
# frozen selection logics.
parser.add_argument('--k-window', type=int, default=Defaults.K_WINDOW,
                    help='number of previous scores considered by -k selector methods')

# gridding determines whether or not sample selection will happen on a grid.
# If any positive integer, a grid with `gridding` points on each axis is
# established, and hyperparameter vectors are sampled from this finite space.
# If 0 (or blank), hyperparameters are sampled from continuous space, and there
# is no limit to the number of hyperparameter vectors that may be tried.
parser.add_argument('--gridding', type=int, default=Defaults.GRIDDING,
                    help='gridding factor (0: no gridding)')

# Which field to use for judgment of performance
# options:
#   f1        - F1 score (harmonic mean of precision and recall)
#   roc_auc   - area under the Receiver Operating Characteristic curve
#   accuracy  - percent correct
#   mu_sigma  - one standard deviation below the average cross-validated F1
#               score (mu - sigma)
parser.add_argument('--metric', choices=METRICS, default=Defaults.METRIC,
                    help='type of BTB selector to use')

# Which data to use for computing judgment score
#   cv   - cross-validated performance on training data
#   test - performance on test data
parser.add_argument('--score-target', choices=SCORE_TARGETS,
                    default=Defaults.SCORE_TARGET,
                    help='whether to compute metrics by cross-validation or on '
                    'test data (if available)')


def create_dataset(db, train_path, test_path=None, output_folder=None,
                   label_column=None, data_description=None):
    """
    Create a dataset and add it to the ModelHub database.

    db: initialized Database object
    train_path: path to raw training data
    test_path: path to raw test data
    output_folder: folder where processed ('wrapped') data will be saved
    label_column: name of csv column representing the label
    data_description: description of the dataset (max 1000 chars)
    """
    # create the name of the dataset from the path to the data
    name = os.path.basename(train_path)
    name = name.replace("_train", "").replace(".csv", "")

    # parse data and create data wrapper for vectorization and label encoding
    if train_path and test_path:
        dw = DataWrapper(name, output_folder, label_column,
                         trainfile=train_path, testfile=test_path)
    elif train_path:
        dw = DataWrapper(name, output_folder, label_column,
                         traintestfile=train_path)
    else:
        raise Exception("No valid training or testing files!")

    # process the data into the form ATM needs and save it to disk
    dw.wrap()
    stats = dw.get_statistics()

    # enter dataset into database
    session = db.get_session()
    dataset = db.Dataset(name=name,
                         description=data_description,
                         train_path=train_path,
                         test_path=test_path,
                         wrapper=dw,
                         label_column=int(stats['label_column']),
                         n_examples=int(stats['n_examples']),
                         k_classes=int(stats['k_classes']),
                         d_features=int(stats['d_features']),
                         majority=float(stats['majority']),
                         size_kb=int(stats['datasize_bytes']) / 1000)
    session.add(dataset)
    session.commit()
    return dataset


def upload_data(train_path, test_path, access_key, secret_key, s3_bucket,
                s3_folder=None):
    """
    Upload processed train/test data to an AWS bucket.

    train_path: path to processed training data
    test_path: path to processed test data
    access_key: AWS API access key
    secret_key: AWS secret API key
    s3_bucket: path to s3 bucket where data will be saved
    s3_folder: optional path within bucket where data will be saved
    """
    print 'Uploading train and test files to AWS S3 bucket', s3_bucket

    conn = S3Connection(aws_key, aws_secret)
    bucket = conn.get_bucket(s3_bucket)
    ktrain = S3Key(bucket)

    if s3_folder:
        aws_train_path = os.path.join(s3_folder, train_path)
        aws_test_path = os.path.join(s3_folder, test_path)
    else:
        aws_train_path = train_path
        aws_test_path = test_path

    ktrain.key = aws_train_path
    ktrain.set_contents_from_filename(train_path)
    ktest = S3Key(bucket)
    ktest.key = aws_test_path
    ktest.set_contents_from_filename(test_path)


def create_datarun(db, dataset_id, tuner, selector, gridding, priority, k_window,
                   r_min, budget_type, budget, deadline, score_target, metric):
    """
    Given a config, creates a set of dataruns for the config and enters them into
    the database. Returns the ID of the created datarun.

    db: initialized Database object
    dataset_id: ID of the dataset this datarun will use
    tuner: string, hyperparameter tuning method
    selector: string, frozen set selection method
    gridding: int or None,
    priority: int, higher priority runs are computed first
    k_window: int, `k` parameter for selection methods that need it
    r_min: int, minimum number of prior examples for tuning to take effect
    budget_type: string, either 'walltime' or 'learners'
    budget: int, total budget for datarun in learners or minutes
    deadline: string-formatted datetime, when the datarun must end by
    score_target: either 'cv' or 'test', indicating which scores the run should
        optimize for
    metric: string, e.g. 'f1' or 'auc'. Metric the run will optimize for.
    """
    # describe the datarun by its tuner and selector
    run_description =  '__'.join([tuner, selector])

    # set the deadline, if applicable
    # TODO
    deadline = None
    if deadline:
        deadline = datetime.strptime(deadline, TIME_FMT)
        budget_type = 'walltime'
    elif budget_type == 'walltime':
        budget = budget or Defaults.WALLTIME_BUDGET
        deadline = datetime.now() + timedelta(minutes=budget)
    else:
        budget = budget or Defaults.LEARNER_BUDGET

    target = score_target + '_judgment_metric'

    # create datarun
    session = db.get_session()
    datarun = db.Datarun(dataset_id=dataset_id,
                         description=run_description,
                         tuner=tuner,
                         selector=selector,
                         gridding=gridding,
                         priority=priority,
                         budget_type=budget_type,
                         budget=budget,
                         deadline=deadline,
                         metric=metric,
                         score_target=target,
                         k_window=k_window,
                         r_min=r_min)
    session.add(datarun)
    session.commit()
    print datarun
    return datarun


def create_frozen_sets(db, datarun, algorithms):
    """
    Create all frozen sets for a given datarun and store them in the ModelHub
    database.
    db: initialized Database object
    datarun: initialized Datarun ORM object
    algorithms: list of codes for the algorithms this datarun will use
    """
    # enumerate all combinations of categorical variables for these algorithms
    print algorithms
    frozen_sets = frozen_sets_from_algorithm_codes(algorithms)

    # create frozen sets
    session = db.get_session()
    session.autoflush = False
    for algorithm, sets in frozen_sets.iteritems():
        for settings, others in sets.iteritems():
            optimizables, constants = others
            fhash = hash_nested_tuple(settings)
            fset = db.FrozenSet(datarun_id=datarun.id,
                                algorithm=algorithm,
                                optimizables=optimizables,
                                constants=constants,
                                frozens=settings,
                                frozen_hash=fhash,
                                is_gridding_done=False)
            session.add(fset)
    session.commit()
    session.close()


def enter_data(sql_config, aws_config, run_config, upload_data=False):
    """
    sql_config: Object with all attributes necessary to initialize a Database.
    aws_config: all attributes necessary to connect to an S3 bucket.
    run_config: all attributes necessary to initialize a Datarun, including
        Dataset info if the dataset has not already been created.
    upload_data: whether to store processed data in the cloud

    Returns: ID of the generated datarun
    """
    # connect to the database
    db = Database(sql_config.dialect, sql_config.database, sql_config.username,
                  sql_config.password, sql_config.host, sql_config.port,
                  sql_config.query)

    # if the user has provided a dataset id, use that. Otherwise, create a new
    # dataset based on the arguments we were passed.
    if run_config.dataset_id is None:
        print 'creating dataset...'
        dataset = create_dataset(db, run_config.train_path, run_config.test_path,
                                 run_config.output_folder, run_config.label_column,
                                 run_config.data_description)
        run_config.dataset_id = dataset.id

        # if we need to upload the train/test data, do it now
        if upload_data:
            upload_data(dataset.wrapper.train_path_out,
                        dataset.wrapper.test_path_out,
                        s3_config.access_key, s3_config.secret_key,
                        s3_config.bucket, s3_config.folder)

    # create and save datarun to database
    print 'creating datarun...'
    datarun = create_datarun(db, run_config.dataset_id, run_config.tuner,
                             run_config.selector, run_config.gridding,
                             run_config.priority, run_config.k_window,
                             run_config.r_min, run_config.budget_type,
                             run_config.budget, run_config.deadline,
                             run_config.score_target, run_config.metric)

    # create frozen sets for the new datarun
    print 'creating frozen sets...'
    create_frozen_sets(db, datarun, run_config.algorithms)

    print 'done!'
    print 'Dataset ID:', run_config.dataset_id
    print 'Datarun ID:', datarun.id
    return datarun.id


if __name__ == '__main__':
    args = parser.parse_args()
    sql_config, aws_config, run_config = load_config(args.sql_config,
                                                     args.aws_config,
                                                     args.run_config, args)
    enter_data(sql_config, aws_config, run_config, args.upload_data)
