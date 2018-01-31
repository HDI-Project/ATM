import os
import pytest

from atm import PROJECT_ROOT
from atm.worker import Worker

from btb.tuning import GCP
from btb.selection import HierarchicalByAlgorithm

DB_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data/modelhub/test/')
DB_PATH = '/tmp/atm.db'

DT_PARAMS = {'criterion': 'gini', 'max_features': 0.5, 'max_depth': 3,
             'min_samples_split': 2, 'min_samples_leaf': 1}


@pytest.fixture
def db():
    os.remove(DB_PATH)
    db = Database(dialect='sqlite', database=DB_PATH)
    # load cached ModelHub state
    db.read_csv(DB_CACHE_PATH)


@pytest.fixture
def dataset(db):
    return db.get_dataset(1)


@pytest.fixture
def datarun(db):
    return db.get_datarun(2)


@pytest.fixture
def model(datarun):
    return Model(method=dt, params=DT_PARAMS,
                 judgment_metric='cv_judgment_metric',
                 label_column=datarun.label_column)


@pytest.fixture
def worker(db, datarun):
    return Worker(db, datarun)


def get_new_worker(db, dataset, **kwargs):
    kwargs['methods'] = kwargs.get('methods', ['logreg', 'dt'])
    run_conf = RunConfig(**kwargs)
    datarun = create_datarun(db, dataset, run_conf)
    return Worker(db, datarun)


def test_load_selector_and_tuner(db, dataset):
    worker = get_new_worker(db, dataset, selector='hieralg', k_window=7,
                            tuner='gcp', r_minimum=7, gridding=3)
    assert type(worker.selector) == HierarchicalByAlgorithm
    assert len(worker.selector.choices) == 6
    assert worker.selector.k == 7
    assert worker.selector.by_algorithm['logreg'] == 4
    assert worker.Tuner == GCP


def test_load_custom_selector_and_tuner(db, dataset):
    tuner_path = './mytuner.py'
    selector_path = './myselector.py'
    worker = get_new_worker(db, dataset, selector=selector_path + ':MySelector',
                            tuner=tuner_path + ':MyTuner')
    assert isinstance(worker.selector, CustomSelector)
    assert issubclass(worker.Tuner, CustomTuner)


def test_select_and_tune(worker):
    """
    This won't test that BTB is working correctly, just that the ATM-BTB
    connection is working.
    """
    part = worker.select_hyperpartition()
    params = worker.tune_hyperparameters(part)


def test_tune_hyperparameters():
    pass


def test_test_classifier(db, dataset):
    worker = get_worker(db, dataset, save_files=True)


def test_save_classifier(db, dataset, model):
    worker = get_worker(db, dataset, save_files=True)
    worker.save_classifier(1, )


def test_is_datarun_finished():
    pass


def test_run_classifier():
    pass
