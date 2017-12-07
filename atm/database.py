from sqlalchemy import (create_engine, inspect, exists, Column, Unicode, String,
                        Integer, Boolean, DateTime, Enum,
                        MetaData, Numeric, Table, Text)
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from sqlalchemy import func, and_

import traceback
import random, sys
import os
from datetime import datetime
import warnings
import pdb

from atm.constants import *
from atm.utilities import object_to_base_64, base_64_to_object


ALGORITHM_ROWS = [
	dict(id=1, code='svm', name='Support Vector Machine', probability=True),
	dict(id=2, code='et', name='Extra Trees', probability=True),
	dict(id=3, code='pa', name='Passive Aggressive', probability=False),
	dict(id=4, code='sgd', name='Stochastic Gradient Descent', probability=True),
	dict(id=5, code='rf', name='Random Forest', probability=True),
	dict(id=6, code='mnb', name='Multinomial Naive Bayes', probability=True),
	dict(id=7, code='bnb', name='Bernoulii Naive Bayes', probability=True),
	dict(id=8, code='dbn', name='Deef Belief Network', probability=False),
	dict(id=9, code='logreg', name='Logistic Regression', probability=True),
	dict(id=10, code='gnb', name='Gaussian Naive Bayes', probability=True),
	dict(id=11, code='dt', name='Decision Tree', probability=True),
	dict(id=12, code='knn', name='K Nearest Neighbors', probability=True),
	dict(id=13, code='mlp', name='Multi-Layer Perceptron', probability=True),
	dict(id=14, code='gp', name='Gaussian Process', probability=True),
]

MAX_FROZEN_SET_ERRORS = 3


def try_with_session(default=lambda: None, commit=False):
    """
    Decorator for instance methods on Database that need a sqlalchemy session.

    This wrapping function creates a new session with the Database's engine and
    passes it to the instance method to use. Everything is inside a try-catch
    statement, so if something goes wrong, this prints a nice error string and
    fails gracefully.

    In case of an error, the function passed to this decorator as `default` is
    called (without arguments) to generate a response. This needs to be a
    function instead of just a static argument to avoid issues with leaving
    empty lists ([]) in method signatures.
    """
    def wrap(func):
        def call(db, *args, **kwargs):
            session = db.get_session()
            try:
                result = func(db, session, *args, **kwargs)
                if commit:
                    session.commit()
            except Exception:
                if commit:
                    session.rollback()
                result = default()
                argstr = ', '.join([str(a) for a in args])
                kwargstr = ', '.join(['%s=%s' % kv for kv in kwargs.items()])
                print "Error in %s(%s, %s):" % (func.__name__, argstr, kwargstr)
                print traceback.format_exc()
            finally:
                session.close()

            return result
        return call
    return wrap


class Database(object):
    def __init__(self, dialect, database, username=None, password=None,
                 host=None, port=None, query=None):
        """
        Accepts configuration for a database connection, and defines SQLAlchemy
        ORM objects for all the tables in the database.
        """
        db_url = URL(drivername=dialect, database=database, username=username,
                     password=password, host=host, port=port, query=query)
        self.engine = create_engine(db_url)

        self.get_session = sessionmaker(bind=self.engine,
                                        expire_on_commit=False)
        self.define_tables()
        self.create_algorithms()

    def define_tables(self):
        metadata = MetaData(bind=self.engine)
        Base = declarative_base()

        class Algorithm(Base):
            __tablename__ = 'algorithms'

            id = Column(Integer, primary_key=True, autoincrement=True)
            code = Column(String(15), nullable=False)
            name = Column(String(30), nullable=False)
            probability = Column(Boolean)

            def __repr__(self):
                return "<%s>" % self.name

        self.Algorithm = Algorithm

        class Dataset(Base):
            __tablename__ = 'datasets'

            id = Column(Integer, primary_key=True, autoincrement=True)

            name = Column(String(100), nullable=False)
            description = Column(String(1000))
            train_path = Column(String(200), nullable=False)
            test_path = Column(String(200))
            wrapper64 = Column(String(200), nullable=False)

            label_column = Column(Integer, nullable=False)
            n_examples = Column(Integer, nullable=False)
            k_classes = Column(Integer, nullable=False)
            d_features = Column(Integer, nullable=False)
            majority = Column(Numeric(precision=10, scale=9), nullable=False)
            size_kb = Column(Integer, nullable=False)

            @property
            def wrapper(self):
                return base_64_to_object(self.wrapper64)

            @wrapper.setter
            def wrapper(self, value):
                self.wrapper64 = object_to_base_64(value)

            def __repr__(self):
                base = "<%s: %s, %d classes, %d features, %d examples>"
                return base % (self.name, self.description, self.k_classes,
                               self.d_features, self.n_examples)

        self.Dataset = Dataset

        class Datarun(Base):
            __tablename__ = 'dataruns'

            id = Column(Integer, primary_key=True, autoincrement=True)
            dataset_id = Column(Integer)

            description = Column(String(200), nullable=False)
            priority = Column(Integer)

            selector = Column(Enum(*SELECTORS), nullable=False)
            k_window = Column(Integer)
            tuner = Column(Enum(*TUNERS), nullable=False)
            gridding = Column(Integer, nullable=False)
            r_min = Column(Integer)

            budget_type = Column(Enum(*BUDGET_TYPES))
            budget = Column(Integer)
            deadline = Column(DateTime)

            metric = Column(Enum(*METRICS))
            score_target = Column(Enum(*[s + '_judgment_metric' for s in
                                         SCORE_TARGETS]))

            started = Column(DateTime)
            completed = Column(DateTime)
            status = Column(Enum(*DATARUN_STATUS), default=RunStatus.PENDING)

            def __repr__(self):
                base = "<ID = %d, dataset ID = %s, strategy = %s, budget = %s (%s), status: %s>"
                return base % (self.id, self.dataset_id, self.description,
                               self.budget_type, self.budget, self.status)

        self.Datarun = Datarun

        class FrozenSet(Base):
            __tablename__ = 'frozen_sets'

            id = Column(Integer, primary_key=True, autoincrement=True)
            datarun_id = Column(Integer, nullable=False)
            algorithm = Column(String(15), nullable=False)

            learners = Column(Integer, default=0)
            optimizables64 = Column(Text)
            constants64 = Column(Text)
            frozens64 = Column(Text)
            frozen_hash = Column(String(32))
            status = Column(Enum(*FROZEN_STATUS),
                            default=FrozenStatus.INCOMPLETE)

            @property
            def optimizables(self):
                return base_64_to_object(self.optimizables64)

            @optimizables.setter
            def optimizables(self, value):
                self.optimizables64 = object_to_base_64(value)

            @property
            def frozens(self):
                return base_64_to_object(self.frozens64)

            @frozens.setter
            def frozens(self, value):
                self.frozens64 = object_to_base_64(value)

            @property
            def constants(self):
                return base_64_to_object(self.constants64)

            @constants.setter
            def constants(self, value):
                self.constants64 = object_to_base_64(value)

            def __repr__(self):
                return "<%s: %s>" % (self.algorithm, self.frozens)

        self.FrozenSet = FrozenSet

        class Learner(Base):
            __tablename__ = 'learners'

            id = Column(Integer, primary_key=True, autoincrement=True)
            frozen_set_id = Column(Integer, nullable=False)
            datarun_id = Column(Integer, nullable=False)

            model_path = Column(String(300))
            metric_path = Column(String(300))
            host = Column(String(50))
            params64 = Column(Text, nullable=False)
            trainable_params64 = Column(Text)
            dimensions = Column(Integer)

            cv_judgment_metric = Column(Numeric(precision=20, scale=10))
            cv_judgment_metric_stdev = Column(Numeric(precision=20, scale=10))
            test_judgment_metric = Column(Numeric(precision=20, scale=10))

            started = Column(DateTime)
            completed = Column(DateTime)
            status = Column(Enum(*LEARNER_STATUS), nullable=False)
            error_msg = Column(Text)

            @property
            def params(self):
                return base_64_to_object(self.params64)

            @params.setter
            def params(self, value):
                self.params64 = object_to_base_64(value)

            @property
            def trainable_params(self):
                return base_64_to_object(self.trainable_params64)

            @trainable_params.setter
            def trainable_params(self, value):
                self.trainable_params64 = object_to_base_64(value)

            def __repr__(self):
                return "<%s>" % self.params

        self.Learner = Learner

        Base.metadata.create_all(bind=self.engine)

    @try_with_session()
    def create_algorithms(self, session):
        """ Enter all the default algorithms into the database """
        for r in ALGORITHM_ROWS:
            if not session.query(self.Dataset).get(r['id']):
                args = dict(r)
                del args['id']
                alg = self.Algorithm(**args)
                session.add(alg)
        session.commit()

    @try_with_session()
    def get_dataruns(self, session, ignore_pending=False, ignore_running=False,
                     ignore_complete=True, include_ids=None, exclude_ids=None):
        """
        Return a list of all dataruns matching the chosen filters, sorted by
        priority in descending order.
        Args:
            ignore_pending: if True, ignore dataruns that have not been started
            ignore_running: if True, ignore dataruns that are already running
            ignore_complete: if True, ignore completed dataruns
            include_ids: only include ids from this list
            exclude_ids: don't return any ids from this list
        """
        query = session.query(self.Datarun)
        if ignore_pending:
            query = query.filter(self.Datarun.status != RunStatus.PENDING)
        if ignore_running:
            query = query.filter(self.Datarun.status != RunStatus.RUNNING)
        if ignore_complete:
            query = query.filter(self.Datarun.status != RunStatus.COMPLETE)
        if include_ids:
            exclude_ids = exclude_ids or []
            ids = [i for i in include_ids if i not in exclude_ids]
            query = query.filter(self.Datarun.id.in_(ids))
        elif exclude_ids:
            query = query.filter(self.Datarun.id.notin_(exclude_ids))

        dataruns = query.all()

        if not len(dataruns):
            return None

        return dataruns

    @try_with_session()
    def get_datarun(self, session, datarun_id=None, ignore_complete=True):
        """
        Return a single datarun. If no ID is supplied, return the first datarun
        present in the database (likely lowest id).

        Args:
            datarun_id: return the datarun with this id
            ignore_complete: if True, ignore completed dataruns
        """
        query = session.query(self.Datarun)
        if ignore_complete:
            query = query.filter(self.Datarun.status != RunStatus.COMPLETE)
        if datarun_id:
            query = query.filter(self.Datarun.id == datarun_id)

        dataruns = query.all()

        if not dataruns:
            return None

        # select first datarun with max priority
        max_priority = max([r.priority for r in dataruns])
        return next((r for r in dataruns if r.priority == max_priority), None)

    @try_with_session()
    def get_dataset(self, session, dataset_id):
        """ Return a specific dataset. """
        return session.query(self.Dataset).get(dataset_id)

    @try_with_session(default=lambda: True)
    def is_datatun_gridding_done(self, session, datarun_id):
        """
        Check whether gridding is done for the entire datarun.
        """
        frozen_sets = session.query(self.FrozenSet)\
            .filter(self.FrozenSet.datarun_id == datarun_id).all()

        is_done = True
        for frozen_set in frozen_sets:
            # If any frozen set has not finished gridding or errored out, we are
            # not done.
            if frozen_set.status == FrozenStatus.INCOMPLETE:
                is_done = False

        return is_done

    @try_with_session(default=list)
    def get_frozen_sets(self, session, datarun_id, ignore_gridding_done=True,
                        ignore_errored=True):
        """
        Return all the frozen sets in a given datarun by id.
        By default, only returns incomplete frozen sets.
        """
        query = session.query(self.FrozenSet)\
            .filter(self.FrozenSet.datarun_id == datarun_id)
        if ignore_gridding_done:
            query = query.filter(self.FrozenSet.status != FrozenStatus.GRIDDING_DONE)
        if ignore_errored:
            query = query.filter(self.FrozenSet.status != FrozenStatus.ERRORED)

        return query.all()

    @try_with_session()
    def get_frozen_set(self, session, frozen_set_id):
        """ Return a specific learner.  """
        return session.query(self.FrozenSet).get(frozen_set_id)

    @try_with_session(default=int)
    def get_number_of_frozen_set_errors(self, session, frozen_set_id):
        learners = session.query(self.Learner)\
            .filter(and_(self.Learner.frozen_set_id == frozen_set_id,
                         self.Learner.status == LearnerStatus.ERRORED)).all()
        return len(learners)

    @try_with_session(default=list)
    def get_algorithms_in_datarun(self, session, datarun_id):
        """ Get all algorithms used in a particular datarun. """
        frozen_sets = self.get_frozen_sets(datarun_id)
        algorithms = set(f.algorithm for f in frozen_sets)
        return list(algorithms)

    @try_with_session(default=list)
    def get_frozens_by_algorithm(self, session, datarun_id, algorithm):
        """ Get all frozen sets for a particular algorithm and datarun. """
        return session.query(self.FrozenSet)\
            .filter(self.FrozenSet.datarun_id == datarun_id)\
            .filter(self.FrozenSet.algorithm == algorithm).all()

    @try_with_session(default=list)
    def get_learners_in_frozen(self, session, frozen_set_id):
        """ Returns all learners in a frozen set. """
        return session.query(self.Learner)\
            .filter(self.Learner.frozen_set_id == frozen_set_id).all()

    @try_with_session(default=list)
    def get_learners_in_datarun(self, session, datarun_id):
        """ Returns all learners in a datarun.  """
        return session.query(self.Learner)\
            .filter(self.Learner.datarun_id == datarun_id)\
            .order_by(self.Learner.started).all()

    @try_with_session(default=list)
    def get_complete_learners(self, session, datarun_id):
        """ Returns all complete learners in a datarun.  """
        return session.query(self.Learner)\
            .filter(self.Learner.datarun_id == datarun_id)\
            .filter(self.Learner.status == LearnerStatus.COMPLETE)\
            .order_by(self.Learner.started).all()

    @try_with_session()
    def get_learner(self, session, learner_id):
        """ Returns a specific learner.  """
        return session.query(self.Learner).get(learner_id)

    @try_with_session()
    def get_maximum_y(self, session, datarun_id, score_target):
        """ Returns the maximum value of a numeric column by name, or None. """
        result = session.query(func.max(getattr(self.Learner, score_target)))\
            .filter(self.Learner.datarun_id == datarun_id).one()[0]
        if result:
            return float(result)
        return None

    @try_with_session(default=lambda: (None, 0, 0))
    def get_best_so_far(self, session, datarun_id, score_target, frozen_set_id=None):
        """
        Sort of like get_maximum_y, but retuns the score with the highest lower
        error bound. In other words, what is the highest value of (score.mean -
        2 * score.std) for any learner?
        """
        maximum = 0
        best_val, best_err = 0, 0
        best_id = None

        if score_target == 'cv_judgment_metric':
            query = session.query(self.Learner.id,
                                  self.Learner.cv_judgment_metric,
                                  self.Learner.cv_judgment_metric_stdev)\
                            .filter(self.Learner.datarun_id == datarun_id)\
                            .filter(self.Learner.status == LearnerStatus.COMPLETE)
            if frozen_set_id is not None:
                query = query.filter(self.Learner.frozen_set_id == frozen_set_id)
            result = query.all()

            for idx, val, std in result:
                if val is None or std is None:
                    continue
                if val - 2 * std > maximum:
                    best_id = idx
                    best_val, best_err = float(val), 2 * float(std)
                    maximum = float(val - 2 * std)

        elif score_target == 'test_judgment_metric':
            query = session.query(self.Learner.id,
                                  self.Learner.test_judgment_metric)\
                            .filter(self.Learner.datarun_id == datarun_id)\
                            .filter(self.Learner.status == LearnerStatus.COMPLETE)
            if frozen_set_id is not None:
                query = query.filter(self.Learner.frozen_set_id == frozen_set_id)
            result = query.all()

            for idx, val in result:
                if val is None:
                    continue
                if val > maximum:
                    best_id = idx
                    best_val = float(val)
                    maximum = best_val

        return best_id, best_val, best_err

    @try_with_session(commit=True)
    def create_learner(self, session, frozen_set_id, datarun_id, host, params):
        """
        Save a new, fully qualified learner object to the database.

        Returns: the ID of the newly-created learner
        """
        learner = self.Learner(frozen_set_id=frozen_set_id,
                               datarun_id=datarun_id,
                               host=host,
                               params=params,
                               started=datetime.now(),
                               status=LearnerStatus.RUNNING)
        session.add(learner)
        frozen_set = session.query(self.FrozenSet).get(frozen_set_id)
        frozen_set.learners += 1

        return learner.id

    @try_with_session(commit=True)
    def mark_learner_errored(self, session, learner_id, error_msg):
        """
        Mark an existing learner as having errored, and set the error message.
        """
        learner = session.query(self.Learner).get(learner_id)
        learner.status = LearnerStatus.ERRORED
        learner.error_msg = error_msg
        if self.get_number_of_frozen_set_errors(learner.frozen_set_id) > \
                MAX_FROZEN_SET_ERRORS:
            self.mark_frozen_set_errored(learner.frozen_set_id)

    @try_with_session(commit=True)
    def complete_learner(self, session, learner_id, trainable_params,
                         dimensions, model_path, metric_path,
                         cv_score, cv_stdev, test_score):
        """
        Set all the parameters on a learner that haven't yet been set, and mark
        it as complete.
        """
        learner = session.query(self.Learner).get(learner_id)

        learner.trainable_params = trainable_params
        learner.dimensions = dimensions
        learner.model_path = model_path
        learner.metric_path = metric_path
        learner.cv_judgment_metric = cv_score
        learner.cv_judgment_metric_stdev = cv_stdev
        learner.test_judgment_metric = test_score

        learner.completed = datetime.now()
        learner.status = LearnerStatus.COMPLETE

    @try_with_session(commit=True)
    def mark_frozen_set_gridding_done(self, session, frozen_set_id):
        """
        Mark a frozen set as having all of its possible grid points explored.
        """
        frozen_set = session.query(self.FrozenSet)\
            .filter(self.FrozenSet.id == frozen_set_id).one()
        frozen_set.status = FrozenStatus.GRIDDING_DONE

    @try_with_session(commit=True)
    def mark_frozen_set_errored(self, session, frozen_set_id):
        """
        Mark a frozen set as having had too many learner errors. This will
        prevent more learners from being trained on this frozen set in the
        future.
        """
        frozen_set = session.query(self.FrozenSet)\
            .filter(self.FrozenSet.id == frozen_set_id).one()
        frozen_set.status = FrozenStatus.ERRORED

    @try_with_session(commit=True)
    def mark_datarun_running(self, session, datarun_id):
        """
        Set the status of the Datarun to RUNNING and set the 'started' field to
        the current datetime.
        """
        datarun = session.query(self.Datarun)\
            .filter(self.Datarun.id == datarun_id).one()
        if datarun.status == RunStatus.PENDING:
            datarun.status = RunStatus.RUNNING
            datarun.started = datetime.now()

    @try_with_session(commit=True)
    def mark_datarun_complete(self, session, datarun_id):
        """
        Set the status of the Datarun to COMPLETE and set the 'completed' field
        to the current datetime.
        """
        datarun = session.query(self.Datarun)\
            .filter(self.Datarun.id == datarun_id).one()
        datarun.status = RunStatus.COMPLETE
        datarun.completed = datetime.now()
