#!/usr/bin/python2.7
import pytest
from collections import defaultdict
from os.path import join

from atm.config import RunConfig
from atm.database import Database
from atm.enter_data import create_dataset, create_datarun
from atm.utilities import download_file_s3
from atm.worker import work

from utilities import work_parallel


@pytest.fixture(scope='module')
def datarun_binary():
    """ generate datarun for a 2-class problem """
    dataset = create_dataset(db, label_column='class',
                             train_path='data/test/pollution_1.csv')
    run_config = RunConfig(dataset_id=dataset.id)
    datarun = create_datarun(db, dataset, run_config)

        dataset = enter_dataset(db, run_config, aws_config)
        datarun_ids.extend(enter_datarun(sql_config, run_config, aws_config,
                                         run_per_partition=True))

@pytest.fixture(scope='module')
def datarun_multiclass_3():
    """ generate datarun for a 3-class problem """
    run_config = RunConfig
    'data/test/iris.data.csv'

@pytest.fixture(scope='module')
def datarun_multiclass_5():
    """ generate datarun for a 5-class problem """
    'data/test/bigmultilabeltest.csv'

@pytest.fixture(scope='module')
def db():
    sql_config = SQLConfig(dialect='sqlite', database='atm.db')
    return Database(**vars(sql_config))

def test_binary_data(db, datarun_binary):

