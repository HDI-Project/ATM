import os

import pytest

from atm.core import ATM
from atm.data import get_local_path
from atm.database import Database

DB_PATH = '/tmp/atm.db'
DATA_URL = 'https://atm-data.s3.amazonaws.com/'
DB_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'modelhub')

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
    # (pollution.csv) and two dataruns, one complete and one with 33/100
    # classifiers finished.
    db.from_csv(DB_CACHE_PATH)

    return db


@pytest.fixture
def dataset(db):
    return db.get_dataset(1)


def test_create_dataset(db):
    train_url = DATA_URL + 'pollution_1_train.csv'
    test_url = DATA_URL + 'pollution_1_test.csv'

    train_path_local = get_local_path('pollution_test.csv', train_url, None)
    if os.path.exists(train_path_local):
        os.remove(train_path_local)

    test_path_local = get_local_path('pollution_test_test.csv', test_url, None)
    if os.path.exists(test_path_local):
        os.remove(test_path_local)

    atm = ATM(database=DB_PATH)

    dataset = atm.add_dataset(
        name='pollution_test',
        train_path=train_url,
        test_path=test_url,
        description='test',
        class_column='class'
    )
    dataset = db.get_dataset(dataset.id)

    train, test = dataset.load()  # This will create the test_path_local

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
