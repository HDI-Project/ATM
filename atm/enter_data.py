import argparse
import os
import warnings
import yaml

from datetime import datetime, timedelta
from boto.s3.connection import S3Connection, Key as S3Key

from atm.datawrapper import DataWrapper
from atm.constants import *
from atm.utilities import ensure_directory, hash_nested_tuple
from atm.mapping import frozen_sets_from_algorithm_codes
from atm.database import Database


warnings.filterwarnings("ignore")


TIME_FMT = "%y-%m-%d_%H:%M"


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
#   ucb1         - vanilla multi-armed bandit
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


def create_dataset(db, args):
    # create the name of the dataset from the path to the data
    name = os.path.basename(args.train_path)
    name = name.replace("_train", "").replace(".csv", "")

    # parse data and create data wrapper for vectorization and label encoding
    if args.train_path and args.test_path:
        dw = DataWrapper(name, args.output_folder, args.label_column,
                         trainfile=args.train_path, testfile=args.test_path)
    elif args.train_path:
        dw = DataWrapper(name, args.output_folder, args.label_column,
                         traintestfile=args.train_path)
    else:
        raise Exception("No valid training or testing files!")

    # process the data into the form ATM needs and save it to disk
    local_train_path, local_test_path = dw.wrap()
    stats = dw.get_statistics()

    # enter dataset into database
    session = db.get_session()
    dataset = db.Dataset(name=name,
                         description=args.data_description,
                         train_path=args.train_path,
                         test_path=args.test_path,
                         wrapper=dw,
                         label_column=int(stats['label_column']),
                         n_examples=int(stats['n_examples']),
                         k_classes=int(stats['k_classes']),
                         d_features=int(stats['d_features']),
                         majority=float(stats['majority']),
                         size_kb=int(stats['datasize_bytes']) / 1000)
    session.add(dataset)
    session.commit()

    # if we need to upload the train/test data, do it now
    if args.upload_data:
        upload_data(local_train_path, local_test_path, args)
    return dataset


def upload_data(train_path, test_path, args):
    print 'Uploading train and test files to AWS S3 Bucket', s3_bucket
    aws_key = args.aws_access_key
    aws_secret = args.aws_secret_key
    s3_bucket = args.aws_s3_bucket
    s3_folder = args.aws_s3_folder

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


def create_datarun(db, args):
    """
    Given a config, creates a set of dataruns for the config and enters them into
    the database.
    Returns the IDs of the created dataruns.
    """
    # describe the datarun by its tuner and selector
    run_description =  '__'.join([args.tuner, args.selector])

    # set the deadline, if applicable
    deadline = None
    if args.deadline:
        deadline = datetime.strptime(args.deadline, TIME_FMT)
        args.budget_type = 'walltime'
    elif args.budget_type == 'walltime':
        budget = args.budget or Defaults.WALLTIME_BUDGET
        deadline = datetime.now() + timedelta(minutes=budget)
    else:
        budget = args.budget or Defaults.LEARNER_BUDGET

    target = args.score_target + '_judgment_metric'

    # create datarun
    session = db.get_session()
    datarun = db.Datarun(dataset_id=args.dataset_id,
                         description=run_description,
                         tuner=args.tuner,
                         selector=args.selector,
                         gridding=args.gridding,
                         priority=args.priority,
                         budget_type=args.budget_type,
                         budget=budget,
                         deadline=deadline,
                         metric=args.metric,
                         score_target=target,
                         k_window=args.k_window,
                         r_min=args.r_min)
    session.add(datarun)
    session.commit()
    print datarun
    return datarun


def create_frozen_sets(db, datarun):
    # create all combinations necessary
    frozen_sets = frozen_sets_from_algorithm_codes(args.algorithms)

    # create frozen sets
    session = db.get_session()
    session.autoflush = False
    for algorithm, frozens in frozen_sets.iteritems():
        for fsettings, others in frozens.iteritems():
            optimizables, constants = others
            fhash = hash_nested_tuple(fsettings)
            fset = db.FrozenSet(datarun_id=datarun.id,
                                algorithm=algorithm,
                                optimizables=optimizables,
                                constants=constants,
                                frozens=fsettings,
                                frozen_hash=fhash,
                                is_gridding_done=False)
            session.add(fset)
    session.commit()
    session.close()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.sql_config:
        with open(args.sql_config) as f:
            sqlconf = yaml.load(f)
        for k, v in sqlconf.items():
            args.__set__('sql_' + k, v)

    if args.aws_config:
        with open(args.aws_config) as f:
            awsconf = yaml.load(f)
        for k, v in awsconf.items():
            args.__set__('aws_' + k, v)


    # connect to the database
    db = Database(args.sql_dialect, args.sql_database, args.sql_username,
                  args.sql_password, args.sql_host, args.sql_port,
                  args.sql_query)

    # if the user has provided a dataset id, use that. Otherwise, create a new
    # dataset based on the arguments we were passed.
    if args.dataset_id is None:
        print 'creating dataset...'
        dataset = create_dataset(db, args)
        args.dataset_id = dataset.id

    # create and save datarun to database
    print 'creating datarun...'
    datarun = create_datarun(db, args)

    # create frozen sets for the new datarun
    print 'creating frozen sets...'
    create_frozen_sets(db, datarun)

    print 'done!'
    print 'Dataset ID:', dataset.id
    print 'Datarun ID:', datarun.id
