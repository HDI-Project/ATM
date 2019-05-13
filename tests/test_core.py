import os

import pytest

from atm import PROJECT_ROOT
from atm.config import DatasetConfig, RunConfig, SQLConfig
from atm.core import ATM
from atm.database import Database, db_session
from atm.dataloader import get_local_path

DB_PATH = '/tmp/atm.db'
DB_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data/modelhub/test/')
DATA_URL = 'https://s3.amazonaws.com/mit-dai-delphi-datastore/downloaded/'
BASELINE_PATH = os.path.join(PROJECT_ROOT, 'data/baselines/best_so_far/')
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


@pytest.fixture
def db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    db = Database(dialect='sqlite', database=DB_PATH)
    # load cached ModelHub state. This database snapshot has one dataset
    # (pollution_1.csv) and two dataruns, one complete and one with 33/100
    # classifiers finished.
    db.from_csv(DB_CACHE_PATH)
    return db


@pytest.fixture
def dataset(db):
    return db.get_dataset(1)


def test_create_dataset(db):
    train_url = DATA_URL + 'pollution_1_train.csv'
    test_url = DATA_URL + 'pollution_1_test.csv'

    sql_conf = SQLConfig({'sql_database': DB_PATH})

    train_path_local = get_local_path('pollution_test.csv', train_url, None)
    if os.path.exists(train_path_local):
        os.remove(train_path_local)

    test_path_local = get_local_path('pollution_test_test.csv', test_url, None)
    if os.path.exists(test_path_local):
        os.remove(test_path_local)

    dataset_conf = DatasetConfig({
        'name': 'pollution_test',
        'train_path': train_url,
        'test_path': test_url,
        'data_description': 'test',
        'class_column': 'class'
    })

    atm = ATM(sql_conf, None, None)

    dataset = atm.create_dataset(dataset_conf)
    dataset = db.get_dataset(dataset.id)

    train, test = dataset.load_()  # This will create the test_path_local

    assert os.path.exists(train_path_local)
    assert os.path.exists(test_path_local)

    assert dataset.train_path == train_url
    assert dataset.test_path == test_url
    assert dataset.description == 'test'
    assert dataset.class_column == 'class'
    assert dataset.n_examples == 40
    assert dataset.d_features == 16
    assert dataset.k_classes == 2
    assert dataset.majority >= 0.5

    # remove test dataset
    if os.path.exists(train_path_local):
        os.remove(train_path_local)

    if os.path.exists(test_path_local):
        os.remove(test_path_local)


def test_enter_data_by_methods(dataset):
    sql_conf = SQLConfig({'sql_database': DB_PATH})
    db = Database(**sql_conf.to_dict())
    run_conf = RunConfig({'dataset_id': dataset.id})

    atm = ATM(sql_conf, None, None)

    for method, n_parts in METHOD_HYPERPARTS.items():
        run_conf.methods = [method]
        run_id = atm.enter_data(None, run_conf)

        with db_session(db):
            run = db.get_datarun(run_id.id)
            assert run.dataset.id == dataset.id
            assert len(run.hyperpartitions) == n_parts


def test_enter_data_all(dataset):
    sql_conf = SQLConfig({'sql_database': DB_PATH})
    db = Database(**sql_conf.to_dict())
    run_conf = RunConfig({'dataset_id': dataset.id, 'methods': METHOD_HYPERPARTS.keys()})

    atm = ATM(sql_conf, None, None)

    run_id = atm.enter_data(None, run_conf)

    with db_session(db):
        run = db.get_datarun(run_id.id)
        assert run.dataset.id == dataset.id
        assert len(run.hyperpartitions) == sum(METHOD_HYPERPARTS.values())


def test_run_per_partition(dataset):
    sql_conf = SQLConfig({'sql_database': DB_PATH})
    db = Database(**sql_conf.to_dict())

    run_conf = RunConfig(
        {
            'dataset_id': dataset.id,
            'methods': ['logreg'],
            'run_per_partition': True
        }
    )

    atm = ATM(sql_conf, None, None)

    run_ids = atm.enter_data(None, run_conf)

    with db_session(db):
        runs = []
        for run_id in run_ids:
            run = db.get_datarun(run_id.id)
            if run is not None:
                runs.append(run)

        assert len(runs) == METHOD_HYPERPARTS['logreg']
        assert all([len(r.hyperpartitions) == 1 for r in runs])
