import os
import json
import pytest

from atm import constants, PROJECT_ROOT
from atm.config import SQLConfig, RunConfig
from atm.database import Database
from atm.enter_data import enter_data, create_dataset, create_datarun


DB_PATH = os.path.join(PROJECT_ROOT, 'test/atm.db')


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


def test_enter_dataset():
    run_conf = RunConfig(train_path='https://')
    sql_conf = SQLConfig(database=DB_PATH)
    db = Database(**vars(sql_conf))

    enter_dataset(db, run_config)
    assert os.path.exists(local_path)


def test_enter_data_methods(dataset):
    run_conf = RunConfig(dataset_id=dataset.id)
    sql_conf = SQLConfig(database=DB_PATH)
    db = Database(**vars(sql_conf))

    method_hyperparts = json.load()
    for method, n_parts in method_hyperparts:
        run_conf.methods = [method]
        run_id = enter_data(sql_conf, run_conf)

        assert db.get_datarun(run_id)
        with db_session(db):
            run = db.get_datarun(run_id)
            assert run.dataset.id == dataset.id
            assert len(run.hyperpartitions) == n_parts


def test_enter_data_all():
    run_conf = RunConfig(dataset_id=dataset.id)
    sql_conf = SQLConfig(database=DB_PATH)
    db = Database(**vars(sql_conf))

    method_hyperparts = json.load()
    for method, n_parts in method_hyperparts:
        run_conf.methods = [method]
        run_id = enter_data(sql_conf, run_conf)

        assert db.get_datarun(run_id)
        with db_session(db):
            run = db.get_datarun(run_id)
            assert run.dataset.id == dataset.id
            assert len(run.hyperpartitions) == n_parts


def test_run_per_partition(db):
    run_conf = RunConfig(methods=['logreg'])
    sql_conf = SQLConfig()
