import datetime
import mock
import numpy as np
import os
import pytest
import random
from mock import patch, Mock

from atm import PROJECT_ROOT
from atm.config import RunConfig
from atm.constants import TIME_FMT, METRICS_BINARY
from atm.database import ClassifierStatus, Database, db_session
from atm.enter_data import create_datarun
from atm.model import Model
from atm.utilities import get_local_data_path
from atm.worker import Worker

from btb.tuning import GP, GPEi, Tuner
from btb.selection import BestKReward, HierarchicalByAlgorithm, Selector

DB_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data/modelhub/test/')
DB_PATH = '/tmp/atm.db'
METRIC_DIR = '/tmp/metrics/'
MODEL_DIR = '/tmp/models/'

DT_PARAMS = {'criterion': 'gini', 'max_features': 0.5, 'max_depth': 3,
             'min_samples_split': 2, 'min_samples_leaf': 1}


@pytest.fixture
def db():
    os.remove(DB_PATH)
    db = Database(dialect='sqlite', database=DB_PATH)
    # load cached ModelHub state. This database snapshot has one dataset
    # (pollution_1.csv) and two dataruns, one complete and one with 33/100
    # classifiers finished.
    db.from_csv(DB_CACHE_PATH)
    return db


@pytest.fixture
def dataset(db):
    # return the only dataset
    return db.get_dataset(1)


@pytest.fixture
def datarun(db):
    # return the unfinished datarun
    return db.get_datarun(2)


@pytest.fixture
def model(dataset):
    model = Model(method='dt', params=DT_PARAMS,
                  judgment_metric='cv_judgment_metric',
                  class_column=dataset.class_column)
    train_path, _ = get_local_data_path(dataset.train_path)
    model.train_test(train_path=train_path)


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
                            tuner='gp', r_minimum=7, gridding=3)
    assert type(worker.selector) == HierarchicalByAlgorithm
    assert len(worker.selector.choices) == 6
    assert worker.selector.k == 7
    assert worker.selector.by_algorithm['logreg'] == 4
    assert worker.Tuner == GP


def test_load_custom_selector_and_tuner(db, dataset):
    tuner_path = os.path.join(PROJECT_ROOT, 'tests/utilities/mytuner.py')
    selector_path = os.path.join(PROJECT_ROOT, 'tests/utilities/myselector.py')
    worker = get_new_worker(db, dataset, selector=selector_path + ':MySelector',
                            tuner=tuner_path + ':MyTuner')
    assert isinstance(worker.selector, Selector)
    assert issubclass(worker.Tuner, Tuner)


def test_select_hyperpartition(worker):
    """
    This won't test that BTB is working correctly, just that the ATM-BTB
    connection is working.
    """
    worker.db.get_hyperpartitions = Mock(return_value=[Mock(id=1)])
    clf_mock = Mock(hyperpartition_id=1, cv_judgment_metric=0.5)
    worker.db.get_classifiers = Mock(return_value=[clf_mock])
    worker.selector.select = Mock(return_value=1)

    hp = worker.select_hyperpartition()

    worker.selector.select.assert_called_with({1: [0.5]})
    assert hp.id == 1


def test_tune_hyperparameters(worker):
    """
    This won't test that BTB is working correctly, just that the ATM-BTB
    connection is working.
    """
    hp = worker.db.get_hyperpartition(1)
    clfs = worker.db.get_classifiers(hyperpartition_id=1)

    mock_tuner = Mock()
    mock_tuner.fit = Mock()
    mock_tuner.propose = Mock(return_value=[])

    worker.Tuner = mock_tuner
    params = worker.tune_hyperparameters(hp)

    mock_tuner.assert_called_with(tunables=hp.tunables,
                                  gridding=worker.datarun.gridding,
                                  r_minimum=worker.datarun.r_minimum)
    mock_turner.fit.assert_called()
    mock_turner.propose.assert_called()


    worker.selector.select.assert_called_with({1: [0.5]})
    assert hp.id == 1



def test_test_classifier(db, dataset):
    metric = 'roc_auc'
    worker = get_new_worker(db, dataset, metric=metric, score_target='mu_sigma')

    model, metrics = worker.test_classifier(method='dt', params=DT_PARAMS)
    judge_mets = [m[metric] for m in metrics['cv']]

    assert type(model) == Model
    assert model.judgment_metric == metric
    assert model.cv_judgment_metric == np.mean(judge_mets)
    assert model.cv_judgment_metric_stdev == np.std(judge_mets)


def test_save_classifier(db, datarun, model):
    worker = Worker(db, datarun, model_dir=MODEL_DIR, metric_dir=METRIC_DIR)
    hp = db.get_hyperpartitions(datarun_id=worker.datarun.id)[0]
    classifier = worker.db.start_classifier(hyperpartition_id=hp.id,
                                            datarun_id=worker.datarun.id,
                                            host='localhost',
                                            hyperparameter_values=DT_PARAMS)
    metrics = {'cv': [{k: random.random() for k in METRICS_BINARY}
                      for i in range(5)],
               'test': {k: random.random() for k in METRICS_BINARY}}

    worker.db.complete_classifier = Mock()
    worker.save_classifier(classifier.id, model, metrics)
    worker.db.complete_classifier.assert_called()

    with db_session(worker.db):
        clf = db.get_classifier(classifier.id)

        loaded = load_model(clf, MODEL_DIR)
        assert type(loaded) == Model
        assert loaded.method == model.method
        assert loaded.random_state == model.random_state

        assert load_metrics(clf, METRIC_DIR) == metrics


def test_run_classifier():
    pass


def test_is_datarun_finished(db, dataset, datarun):
    r1 = db.get_datarun(1)
    worker = Worker(db, r1)
    assert worker.is_datarun_finished()

    r2 = db.get_datarun(2)
    worker = Worker(db, r2)
    assert not worker.is_datarun_finished()

    deadline = (datetime.datetime.now() - datetime.timedelta(seconds=1)).strftime(TIME_FMT)
    worker = get_new_worker(db, dataset, deadline=deadline)
    assert worker.is_datarun_finished()
