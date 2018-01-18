import os
import json
import pytest

from atm import constants, PROJECT_ROOT
from atm.config import SQLConfig, RunConfig
from atm.database import Database
from atm.enter_data import enter_data, create_dataset, create_datarun


DB_PATH = os.path.join(PROJECT_ROOT, 'test/atm.db')
DATA_URL = 'https://s3.amazonaws.com/mit-dai-delphi-datastore/downloaded/'
BASELINE_PATH = os.path.join(PROJECT_ROOT, 'test/baselines/best_so_far/')
BASELINE_URL = 'https://s3.amazonaws.com/mit-dai-delphi-datastore/best_so_far/'

METHOD_HYPERPARTS = {
	'logreg': 6,
	'svm': 4,
	'sgd': 24,
	'dt': 2,
	'et': 2,
	'rf': 2,
	'gnb': 1,
	'mnb': 1,
	'bnb': 1,
	'gp': 8,
	'pa': 4,
	'knn': 24,
	'mlp': 60,
}


@pytest.fixture()
def db():
    return Database(dialect='sqlite', database=DB_PATH)


@pytest.fixture()
def dataset(db):
    ds = db.get_dataset(1)
    if ds:
        return ds
    else:
        data_path = os.path.join(PROJECT_ROOT, 'data/test/pollution_1.csv')
        return create_dataset(db, 'class', data_path)


def test_enter_dataset(db):
	train_url = DATA_URL + 'pollution_1_train.csv'
	test_url = DATA_URL + 'pollution_1_test.csv'
    run_conf = RunConfig(train_path=train_url)

    enter_dataset(db, run_config)
    assert os.path.exists(train_path_local)


def test_enter_datarun_by_methods(dataset):
    sql_conf = SQLConfig(database=DB_PATH)
    db = Database(**vars(sql_conf))
    run_conf = RunConfig(dataset_id=dataset.id)

    for method, n_parts in METHOD_HYPERPARTS:
        run_conf.methods = [method]
        run_id = enter_datarun(sql_conf, run_conf)

        assert db.get_datarun(run_id)
        with db_session(db):
            run = db.get_datarun(run_id)
            assert run.dataset.id == dataset.id
            assert len(run.hyperpartitions) == n_parts


def test_enter_datarun_all(dataset):
    sql_conf = SQLConfig(database=DB_PATH)
    db = Database(**vars(sql_conf))
    run_conf = RunConfig(dataset_id=dataset.id)

    run_id = enter_datarun(sql_conf, run_conf)

    with db_session(db):
        run = db.get_datarun(run_id)
        assert run.dataset.id == dataset.id
        assert len(run.hyperpartitions) == sum(METHOD_HYPERPARTS.values())


def test_run_per_partition(dataset):
    sql_conf = SQLConfig(database=DB_PATH)
    run_conf = RunConfig(dataset_id=dataset.id, methods=['logreg'])

    run_id = enter_datarun(sql_conf, run_conf, run_per_partition)

    with db_session(db):
        runs = db.get_datarun(run_id)
        assert len(runs) == METHOD_HYPERPARTS['logreg']
		assert all([len(run.hyperparpartitions) == 1 for run in runs])
