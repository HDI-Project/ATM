import pytest

from atm.worker import Worker

@pytest.fixture
def datarun():
    db = Database(**vars(sql_conf))
    sql_conf = SQLConfig(database=DB_PATH)
    run_conf = RunConfig(dataset_id=dataset.id, methods=['logreg'])

@pytest.fixture
def worker():
    worker = Worker()

