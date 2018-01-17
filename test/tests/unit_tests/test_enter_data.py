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



def test_run_per_partition(db):
    run_conf = RunConfig(methods=['logreg'])
    sql_conf = SQLConfig()

def test_methods(db):
    run_conf = RunConfig()
    sql_conf = SQLConfig()
    method_hyperparams = json.load()
    for method in :
        run_conf.methods = [method]
        run_id = enter_data(sql_conf, run_conf)
        assert
