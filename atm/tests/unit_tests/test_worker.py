import datetime
import os
import random

import numpy as np
import pytest
from btb.selection import BestKVelocity
from btb.selection.selector import Selector
from btb.tuning import GP
from btb.tuning.tuner import BaseTuner
from unittest.mock import ANY, Mock, patch

from atm import PROJECT_ROOT
from atm.config import LogConfig, RunConfig, SQLConfig
from atm.constants import METRICS_BINARY, TIME_FMT
from atm.database import Database, db_session
from atm.enter_data import enter_data
from atm.model import Model
from atm.utilities import download_data, load_metrics, load_model
from atm.worker import ClassifierError, Worker

DB_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data/modelhub/test/')
DB_PATH = '/tmp/atm.db'
METRIC_DIR = '/tmp/metrics/'
MODEL_DIR = '/tmp/models/'

DATASET_ID = 1
DATARUN_ID = 2
HYPERPART_ID = 34
DT_PARAMS = {'criterion': 'gini', 'max_features': 0.5, 'max_depth': 3,
             'min_samples_split': 2, 'min_samples_leaf': 1}


# helper class to allow fuzzy arg matching
class StringWith(object):
    def __init__(self, match):
        self.match = match

    def __eq__(self, other):
        return self.match in other


# helper class to allow incomplete object matching
class ObjWithAttrs(object):
    def __init__(self, **kwargs):
        self.attrs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other):
        return all([getattr(other, k) == v for k, v in self.attrs.items()])

    def __repr__(self):
        return '<%s>' % ', '.join(['%s=%s' % i for i in self.attrs.items()])


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
    return db.get_dataset(DATASET_ID)


@pytest.fixture
def datarun(db):
    # return the unfinished datarun
    return db.get_datarun(DATARUN_ID)


@pytest.fixture
def hyperpartition(db):
    # return a decision tree hyperpartition matching the static params above
    return db.get_hyperpartition(HYPERPART_ID)


@pytest.fixture
def model(dataset):
    train_path, _ = download_data(dataset.train_path)
    model = Model(method='dt', params=DT_PARAMS,
                  judgment_metric='roc_auc',
                  class_column=dataset.class_column)
    model.train_test(train_path=train_path)
    return model


@pytest.fixture
def metrics():
    cv_mets = [{k: random.random() for k in METRICS_BINARY} for i in range(5)]
    test_mets = {k: random.random() for k in METRICS_BINARY}
    return {'cv': cv_mets, 'test': test_mets}


@pytest.fixture
def worker(db, datarun):
    return Worker(db, datarun)


def get_new_worker(**kwargs):
    kwargs['methods'] = kwargs.get('methods', ['logreg', 'dt'])
    sql_conf = SQLConfig(database=DB_PATH)
    run_conf = RunConfig(**kwargs)
    run_id = enter_data(sql_conf, run_conf)
    db = Database(**vars(sql_conf))
    datarun = db.get_datarun(run_id)
    return Worker(db, datarun)


def test_load_selector_and_tuner(db, dataset):
    worker = get_new_worker(selector='bestkvel', k_window=7, tuner='gp')
    assert type(worker.selector) == BestKVelocity
    assert len(worker.selector.choices) == 8
    assert worker.selector.k == 7
    assert worker.Tuner == GP


def test_load_custom_selector_and_tuner(db, dataset):
    tuner_path = os.path.join(PROJECT_ROOT, 'tests/utilities/mytuner.py')
    selector_path = os.path.join(PROJECT_ROOT, 'tests/utilities/myselector.py')
    worker = get_new_worker(selector=selector_path + ':MySelector',
                            tuner=tuner_path + ':MyTuner')
    assert isinstance(worker.selector, Selector)
    assert issubclass(worker.Tuner, BaseTuner)


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


def test_tune_hyperparameters(worker, hyperpartition):
    """
    This won't test that BTB is working correctly, just that the ATM-BTB
    connection is working.
    """
    mock_tuner = Mock()
    worker.Tuner = Mock(return_value=mock_tuner)

    with patch('atm.worker.update_params') as update_params_mock:
        worker.tune_hyperparameters(hyperpartition)

        update_params_mock.assert_called_once_with(
            params=mock_tuner.propose.return_value,
            categoricals=hyperpartition.categoricals,
            constants=hyperpartition.constants
        )

    approximate_tunables = [(k, ObjWithAttrs(range=v.range))
                            for k, v in hyperpartition.tunables]
    mock_tuner.propose.assert_called()


def test_test_classifier(db, dataset):
    metric = 'roc_auc'
    worker = get_new_worker(metric=metric, score_target='mu_sigma')

    model, metrics = worker.test_classifier(method='dt', params=DT_PARAMS)
    judge_mets = [m[metric] for m in metrics['cv']]

    assert type(model) == Model
    assert model.judgment_metric == metric
    assert model.cv_judgment_metric == np.mean(judge_mets)
    assert model.cv_judgment_metric_stdev == np.std(judge_mets)


def test_save_classifier(db, datarun, model, metrics):
    log_conf = LogConfig(model_dir=MODEL_DIR, metric_dir=METRIC_DIR)
    worker = Worker(db, datarun, log_config=log_conf)
    hp = db.get_hyperpartitions(datarun_id=worker.datarun.id)[0]
    classifier = worker.db.start_classifier(hyperpartition_id=hp.id,
                                            datarun_id=worker.datarun.id,
                                            host='localhost',
                                            hyperparameter_values=DT_PARAMS)

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


def test_is_datarun_finished(db, dataset, datarun):
    r1 = db.get_datarun(1)
    worker = Worker(db, r1)
    assert worker.is_datarun_finished()

    r2 = db.get_datarun(2)
    worker = Worker(db, r2)
    assert not worker.is_datarun_finished()

    deadline = (datetime.datetime.now() - datetime.timedelta(seconds=1)).strftime(TIME_FMT)
    worker = get_new_worker(deadline=deadline)
    assert worker.is_datarun_finished()


def test_run_classifier(worker, hyperpartition, model, metrics):
    worker.select_hyperpartition = Mock(return_value=hyperpartition)
    worker.tune_hyperparameters = Mock(return_value=DT_PARAMS)
    worker.test_classifier = Mock(return_value=(model, metrics))
    worker.save_classifier = Mock()
    worker.db = Mock()

    # make sure the function shorts out if the datarun is finished
    worker.is_datarun_finished = Mock(return_value=True)
    worker.run_classifier()
    assert not worker.select_hyperpartition.called
    assert not worker.tune_hyperparameters.called

    # make sure things run smoothly: hyperparameters are chosen and a classifier
    # is created, tested, and saved.
    worker.is_datarun_finished = Mock(return_value=False)
    worker.run_classifier()
    worker.select_hyperpartition.assert_called_once()
    worker.tune_hyperparameters.assert_called_once_with(hyperpartition)
    worker.db.start_classifier.assert_called_once_with(hyperpartition_id=hyperpartition.id,
                                                       datarun_id=worker.datarun.id,
                                                       host=ANY,
                                                       hyperparameter_values=DT_PARAMS)
    worker.test_classifier.assert_called_once_with(hyperpartition.method, DT_PARAMS)
    worker.save_classifier.assert_called_once_with(ANY, model, metrics)

    # make sure hyperpartition specification works
    hp_id = hyperpartition.id + 1
    worker.db.get_hyperpartition = lambda i: ObjWithAttrs(id=i, method='dt',
                                                          datarun_id=worker.datarun.id)
    worker.run_classifier(hyperpartition_id=hp_id)
    worker.tune_hyperparameters.assert_called_with(ObjWithAttrs(id=hp_id))

    # make sure error handling works correctly
    worker.test_classifier.side_effect = ValueError('qwerty')
    with pytest.raises(ClassifierError):
        worker.run_classifier()
    worker.db.mark_classifier_errored.assert_called_with(
        ANY, error_message=StringWith('qwerty'))
