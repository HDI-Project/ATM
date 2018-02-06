from __future__ import absolute_import, print_function

import argparse
import os
import warnings
from datetime import datetime, timedelta

from .config import *
from .constants import *
from .database import Database
from .encoder import MetaData
from .method import Method
from .utilities import download_data


def create_dataset(db, run_config, aws_config=None):
    """
    Create a dataset and add it to the ModelHub database.

    db: initialized Database object
    run_config: RunConfig object describing the dataset to create
    aws_config: optional. AWS credentials for downloading data from S3.
    """
    # download data to the local filesystem to extract metadata
    train_local, test_local = download_data(run_config.train_path,
                                            run_config.test_path,
                                            aws_config)

    # create the name of the dataset from the path to the data
    name = os.path.basename(train_local)
    name = name.replace("_train.csv", "").replace(".csv", "")

    # process the data into the form ATM needs and save it to disk
    meta = MetaData(run_config.class_column, train_local, test_local)

    # enter dataset into database
    dataset = db.create_dataset(name=name,
                                description=run_config.data_description,
                                train_path=run_config.train_path,
                                test_path=run_config.test_path,
                                class_column=run_config.class_column,
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
    run_config: RunConfig object describing the datarun to create
    """
    # describe the datarun by its tuner and selector
    run_description = '__'.join([run_config.tuner, run_config.selector])

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
                                r_minimum=run_config.r_minimum)
    return datarun


def enter_data(sql_config, run_config, aws_config=None,
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
        dataset = create_dataset(db, run_config, aws_config=aws_config)
        run_config.dataset_id = dataset.id
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
