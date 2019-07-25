import json
import os

import pytest

from atm.api import create_app
from atm.core import ATM
from atm.database import Database

DB_PATH = '/tmp/atm.db'
DB_CACHE_PATH = os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), 'modelhub')


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
def atm():
    return ATM(database=DB_PATH)


@pytest.fixture
def client(atm):
    app = create_app(atm)

    app.config['TESTING'] = True
    client = app.test_client()
    yield client


def test_create_app(atm):
    app = create_app(atm)

    assert app.name == 'atm.api'
    assert not app.config['DEBUG']


def test_create_app_debug(atm):
    app = create_app(atm, debug=True)

    assert app.name == 'atm.api'
    assert app.config['DEBUG']


def test_get_home(client):
    res = client.get('/', follow_redirects=False)

    assert res.status == '302 FOUND'
    assert res.location == 'http://localhost/static/swagger/swagger-ui/index.html'


def test_get_dataset(client):
    res = client.get('api/datasets')
    data = json.loads(res.data.decode('utf-8'))

    assert res.status == '200 OK'
    assert data.get('num_results') == 1


def test_post_dataset(client):
    res = client.post(
        'api/datasets',
        json={'train_path': 's3://atm-data/wind_1.csv', 'class_column': 'class'}
    )
    data = json.loads(res.data.decode('utf-8'))

    assert res.status == '201 CREATED'


def test_options_dataset(client):
    res = client.options('api/datasets')

    expected_headers = [
        ('Content-Type', 'text/html; charset=utf-8'),
        ('Access-Control-Allow-Headers', 'Content-Type, Authorization'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Credentials', 'true'),
    ]

    assert set(expected_headers).issubset(set(res.headers.to_list()))


def test_get_datarun(client):
    res = client.get('api/dataruns')
    data = json.loads(res.data.decode('utf-8'))

    assert res.status == '200 OK'
    assert data.get('num_results') == 2


def test_get_hyperpartition(client):
    res = client.get('api/hyperpartitions')
    data = json.loads(res.data.decode('utf-8'))

    assert res.status == '200 OK'
    assert data.get('num_results') == 40


def test_get_classifier(client):
    res = client.get('api/classifiers')
    data = json.loads(res.data.decode('utf-8'))

    assert res.status == '200 OK'
    assert data.get('num_results') == 133
