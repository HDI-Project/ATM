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
from atm.database import Database

warnings.filterwarnings("ignore")


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
    stats = dw.statistics

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
    if deadline:
        deadline = datetime.strptime(deadline, TIME_FMT)
        # this overrides the otherwise configured budget_type
        # TODO: why not walltime and learner budget simultaneously?
        budget_type = 'walltime'
    elif budget_type == 'walltime':
        deadline = datetime.now() + timedelta(minutes=budget)

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
    session = db.get_session()
    session.autoflush = False

    for alg in algorithms:
        # enumerate all combinations of categorical variables for this algorithm
        enum = Enumerator(ALGORITHMS_MAP[alg])
        frozen_sets = enum.get_frozen_sets()
        print 'algorithm', alg, 'has', len(sets), 'frozen sets'

        for fs in frozen_sets:
            fset = db.FrozenSet(datarun_id=datarun.id,
                                algorithm=alg,
                                optimizables=fs.tunables,
                                constants=fs.constants,
                                frozens=fs.frozens,
                                frozen_hash=hash_nested_tuple(fs.frozens),
                                status=FrozenStatus.INCOMPLETE)
            session.add(fset)

    session.commit()
    session.close()

def enter_dataset(db, run_config, aws_config=None, upload_data=False):
    """
    Generate a dataset, and update run_config with the dataset ID.

    db: Database object with active connection to ModelHub
    run_config: all attributes necessary to initialize a Datarun, including
        Dataset info
    aws_config: all attributes necessary to connect to an S3 bucket.
    upload_data: whether to store processed data in the cloud

    Returns: the generated dataset object
    """
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

    return dataset

def enter_datarun(sql_config, run_config, aws_config=None, upload_data=False):
    """
    Generate a datarun, including a dataset if necessary.

    sql_config: Object with all attributes necessary to initialize a Database.
    run_config: all attributes necessary to initialize a Datarun, including
        Dataset info if the dataset has not already been created.
    aws_config: all attributes necessary to connect to an S3 bucket.
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
        dataset = enter_dataset(db, run_config, aws_config=aws_config,
                                upload_data=upload_data)
    else:
        dataset = db.get_dataset(run_config.dataset_id)

    # create and save datarun to database
    print
    print 'creating datarun...'
    datarun = create_datarun(db, run_config.dataset_id, run_config.tuner,
                             run_config.selector, run_config.gridding,
                             run_config.priority, run_config.k_window,
                             run_config.r_min, run_config.budget_type,
                             run_config.budget, run_config.deadline,
                             run_config.score_target, run_config.metric)

    # create frozen sets for the new datarun
    print
    print 'creating frozen sets...'
    create_frozen_sets(db, datarun, run_config.algorithms)
    print
    print '========== Summary =========='
    print 'Dataset ID:', dataset.id
    print 'Training data:', dataset.train_path
    print 'Test data:', (dataset.test_path or '(None)')
    print 'Datarun ID:', datarun.id
    print 'Frozen set selection strategy:', datarun.selector
    print 'Parameter tuning strategy:', datarun.tuner
    print 'Budget: %d (%s)' % (datarun.budget, datarun.budget_type)
    return datarun.id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
Creates a dataset (if necessary) and a datarun and adds them to the ModelHub.
All required arguments have default values. Running this script with no
arguments will create a new dataset with the file in data/pollution_1.csv and a
new datarun with the default arguments listed below.

You can pass yaml configuration files (--sql-config, --aws-config, --run-config)
instead of passing individual arguments. Any arguments in the config files will
override arguments passed on the command line. See the examples in the config/
folder for more information. """)
    # Add argparse arguments for aws, sql, and datarun config
    add_arguments_aws_s3(parser)
    add_arguments_sql(parser)
    add_arguments_datarun(parser)

    # add our own argument, which determines whether to upload data
    parser.add_argument('--upload-data', action='store_true',
                        help='Whether to upload processed data to s3')

    args = parser.parse_args()

    # create config objects from the config files and/or command line args
    sql_config, run_config, aws_config = load_config(sql_path=args.sql_config,
                                                     run_path=args.run_config,
                                                     aws_path=args.aws_config,
                                                     args=args)
    # create and save the dataset and datarun
    enter_datarun(sql_config, run_config, aws_config, upload_data=args.upload_data)
