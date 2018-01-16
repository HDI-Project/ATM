from __future__ import print_function
import argparse
import os
import warnings
import yaml

from datetime import datetime, timedelta
from boto.s3.connection import S3Connection, Key as S3Key

from atm.config import *
from atm.constants import *
from atm.database import Database
from atm.encoder import MetaData
from atm.method import Method
from atm.utilities import ensure_directory, hash_nested_tuple, download_data

warnings.filterwarnings("ignore")


def create_dataset(db, label_column, train_path, test_path=None,
                   data_description=None):
    """
    Create a dataset and add it to the ModelHub database.

    db: initialized Database object
    label_column: name of csv column representing the label
    train_path: path to raw training data
    test_path: path to raw test data
    data_description: description of the dataset (max 1000 chars)
    """
    # create the name of the dataset from the path to the data
    name = os.path.basename(train_path)
    name = name.replace("_train.csv", "").replace(".csv", "")

    # process the data into the form ATM needs and save it to disk
    meta = MetaData(label_column, train_path, test_path)

    # enter dataset into database
    dataset = db.create_dataset(name=name,
                                description=data_description,
                                train_path=train_path,
                                test_path=test_path,
                                label_column=label_column,
                                n_examples=meta.n_examples,
                                k_classes=meta.k_classes,
                                d_features=meta.d_features,
                                majority=meta.majority,
                                size_kb=meta.size / 1000)
    return dataset


def create_datarun(db, dataset, run_config):
    """
    Given a config, creates a set of dataruns for the config and enters them into
    the database. Returns the ID of the created datarun.

    db: initialized Database object
    dataset: Dataset SQLAlchemy ORM object
    run_config: configuration describing the datarun to create
    """
    # describe the datarun by its tuner and selector
    run_description =  '__'.join([run_config.tuner, run_config.selector])

    # set the deadline, if applicable
    deadline = run_config.deadline
    if deadline:
        deadline = datetime.strptime(deadline, TIME_FMT)
        # this overrides the otherwise configured budget_type
        # TODO: why not walltime and classifiers budget simultaneously?
        run_config.budget_type = 'walltime'
    elif run_config.budget_type == 'walltime':
        deadline = datetime.now() + timedelta(minutes=budget)

    target = run_config.score_target + '_judgment_metric'
    datarun = db.create_datarun(dataset_id=dataset.id,
                                description=run_description,
                                tuner=run_config.tuner,
                                selector=run_config.selector,
                                gridding=run_config.gridding,
                                priority=run_config.priority,
                                budget_type=run_config.budget_type,
                                budget=run_config.budget,
                                deadline=deadline,
                                metric=run_config.metric,
                                score_target=target,
                                k_window=run_config.k_window,
                                r_min=run_config.r_min)
    return datarun


def enter_dataset(db, run_config, aws_config=None):
    """
    Generate a dataset, and update run_config with the dataset ID.

    db: Database object with active connection to ModelHub
    run_config: all attributes necessary to initialize a Datarun, including
        Dataset info
    aws_config: all attributes necessary to connect to an S3 bucket.

    Returns: the generated dataset object
    """
    print('downloading data...')
    train_path, test_path = download_data(run_config.train_path,
                                          run_config.test_path, aws_config)
    print('creating dataset...')
    dataset = create_dataset(db, run_config.label_column, train_path, test_path,
                             run_config.data_description)
    run_config.dataset_id = dataset.id

    return dataset


def enter_datarun(sql_config, run_config, aws_config=None,
                  run_per_partition=False):
    """
    Generate a datarun, including a dataset if necessary.

    sql_config: Object with all attributes necessary to initialize a Database.
    run_config: all attributes necessary to initialize a Datarun, including
        Dataset info if the dataset has not already been created.
    aws_config: all attributes necessary to connect to an S3 bucket.

    Returns: ID of the generated datarun
    """
    # connect to the database
    db = Database(sql_config.dialect, sql_config.database, sql_config.username,
                  sql_config.password, sql_config.host, sql_config.port,
                  sql_config.query)

    # if the user has provided a dataset id, use that. Otherwise, create a new
    # dataset based on the arguments we were passed.
    if run_config.dataset_id is None:
        dataset = enter_dataset(db, run_config, aws_config=aws_config)
    else:
        dataset = db.get_dataset(run_config.dataset_id)

    method_parts = {}
    for m in run_config.methods:
        # enumerate all combinations of categorical variables for this method
        method = Method(m)
        method_parts[m] = method.get_hyperpartitions()
        print('method', m, 'has', len(method_parts[m]), 'hyperpartitions')

    # create hyperpartitions and datarun(s)
    run_ids = []
    if not run_per_partition:
        print('saving datarun...')
        datarun = create_datarun(db, dataset, run_config)

    print('saving hyperpartions...')
    for method, parts in method_parts.items():
        for part in parts:
            # if necessary, create a new datarun for each hyperpartition.
            # This setting is useful for debugging.
            if run_per_partition:
                datarun = create_datarun(db, dataset, run_config)
                run_ids.append(datarun.id)

            # create a new hyperpartition in the database
            db.create_hyperpartition(datarun_id=datarun.id,
                                     method=method,
                                     tunables=part.tunables,
                                     constants=part.constants,
                                     categoricals=part.categoricals,
                                     status=PartitionStatus.INCOMPLETE)

    print('done!')
    print()
    print('========== Summary ==========')
    print('Dataset ID:', dataset.id)
    print('Training data:', dataset.train_path)
    print('Test data:', (dataset.test_path or '(None)'))
    if run_per_partition:
        print('Datarun IDs:', ', '.join(map(str, run_ids)))
    else:
        print('Datarun ID:', datarun.id)
    print('Hyperpartition selection strategy:', datarun.selector)
    print('Parameter tuning strategy:', datarun.tuner)
    print('Budget: %d (%s)' % (datarun.budget, datarun.budget_type))
    print()

    return run_ids or datarun.id


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

    args = parser.parse_args()

    # create config objects from the config files and/or command line args
    sql_config, run_config, aws_config = load_config(sql_path=args.sql_config,
                                                     run_path=args.run_config,
                                                     aws_path=args.aws_config,
                                                     **vars(args))
    # create and save the dataset and datarun
    enter_datarun(sql_config, run_config, aws_config)
