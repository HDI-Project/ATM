from sqlalchemy import (create_engine, inspect, exists, Column, Unicode, String,
                        ForeignKey, Integer, Boolean, DateTime, Enum, MetaData,
                        Numeric, Table, Text)
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from sqlalchemy import func, and_

import traceback
import random, sys
import os
import warnings
import pdb
from datetime import datetime
from operator import attrgetter

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

MAX_HYPERPARTITION_ERRORS = 3


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
        """
        Define the SQLAlchemy ORM class for each table in the ModelHub database.

        These must be defined after the Database class is initialized so that
        the database metadata is available (at runtime).
        If the database does not already exist, it will be created. If it does
        exist, it will not be updated with new schema -- after schema changes,
        the database must be destroyed and reinialized.
        """

        metadata = MetaData(bind=self.engine)
        Base = declarative_base()

        class Method(Base):
            __tablename__ = 'methods'

            id = Column(Integer, primary_key=True, autoincrement=True)
            code = Column(String(15), nullable=False)
            name = Column(String(30), nullable=False)
            probability = Column(Boolean)

            def __repr__(self):
                return "<%s (%s)>" % (self.name, self.code)

        self.Method = Method

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
            dataset_id = Column(Integer, ForeignKey('datasets.id'))

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

        class Hyperpartition(Base):
            __tablename__ = 'hyperpartitions'

            id = Column(Integer, primary_key=True, autoincrement=True)
            datarun_id = Column(Integer, ForeignKey('dataruns.id'))
            method = Column(String(15))

            categoricals64 = Column(Text)
            tunables64 = Column(Text)
            constants64 = Column(Text)

            classifiers = Column(Integer, default=0)
            status = Column(Enum(*PARTITION_STATUS),
                            default=PartitionStatus.INCOMPLETE)

            @property
            def categoricals(self):
                """
                A list of categorical variables along with the fixed values
                which define the hyperpartition.
                Each element is a ('name', HyperParameter) tuple.
                """
                return base_64_to_object(self.categoricals64)

            @categoricals.setter
            def categoricals(self, value):
                self.categoricals64 = object_to_base_64(value)

            @property
            def tunables(self):
                """
                A list of parameters which are unspecified and must be selected
                with a Tuner. Each element is a ('name', HyperParameter) tuple.
                """
                return base_64_to_object(self.tunables64)

            @tunables.setter
            def tunables(self, value):
                self.tunables64 = object_to_base_64(value)

            @property
            def constants(self):
                return base_64_to_object(self.constants64)

            @constants.setter
            def constants(self, value):
                self.constants64 = object_to_base_64(value)

            def __repr__(self):
                return "<%s: %s>" % (self.algorithm, self.categoricals)

        self.Hyperpartition = Hyperpartition

        class Classifier(Base):
            __tablename__ = 'classifiers'

            id = Column(Integer, primary_key=True, autoincrement=True)
            datarun_id = Column(Integer, ForeignKey('dataruns.id'))
            hyperpartition_id = Column(Integer, ForeignKey('hyperpartitions.id'))

            model_path = Column(String(300))
            metric_path = Column(String(300))
            params64 = Column(Text, nullable=False)

            trainable_params64 = Column(Text)
            host = Column(String(50))
            dimensions = Column(Integer)

            cv_judgment_metric = Column(Numeric(precision=20, scale=10))
            cv_judgment_metric_stdev = Column(Numeric(precision=20, scale=10))
            test_judgment_metric = Column(Numeric(precision=20, scale=10))

            started = Column(DateTime)
            completed = Column(DateTime)
            status = Column(Enum(*CLASSIFIER_STATUS), nullable=False)
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
                params = ', '.join(['%s: %s' % i for i in self.params.items()])
                return "<id=%d, params=(%s)>" % (self.id, params)

        self.Classifier = Classifier

        Base.metadata.create_all(bind=self.engine)

    @try_with_session()
    def create_algorithms(self, session):
        """ Enter all the default algorithms into the database. """
        for r in ALGORITHM_ROWS:
            if not session.query(self.Dataset).get(r['id']):
                args = dict(r)
                del args['id']
                alg = self.MethodCode(**args)
                session.add(alg)
        session.commit()

    ###########################################################################
    ##  Standard query methods  ###############################################
    ###########################################################################

    @try_with_session()
    def get_dataset(self, session, dataset_id):
        """ Get a specific dataset. """
        return session.query(self.Dataset).get(dataset_id)

    @try_with_session()
    def get_datarun(self, session, datarun_id):
        """ Get a specific datarun. """
        return session.query(self.Datarun).get(datarun_id)

    @try_with_session()
    def get_dataruns(self, session, ignore_pending=False, ignore_running=False,
                     ignore_complete=True, include_ids=None, exclude_ids=None,
                     max_priority=True):
        """
        Get a list of all dataruns matching the chosen filters.

        Args:
            ignore_pending: if True, ignore dataruns that have not been started
            ignore_running: if True, ignore dataruns that are already running
            ignore_complete: if True, ignore completed dataruns
            include_ids: only include ids from this list
            exclude_ids: don't return any ids from this list
            max_priority: only return dataruns which have the highest priority
                of any in the filtered set
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

        if max_priority:
            mp = max(dataruns, key=attrgetter('priority')).priority
            dataruns = [d for d in dataruns if d.priority == mp]

        return dataruns

    @try_with_session()
    def get_hyperpartition(self, session, hyperpartition_id):
        """ Get a specific classifier.  """
        return session.query(self.Hyperpartition).get(hyperpartition_id)

    @try_with_session(default=list)
    def get_hyperpartitions(self, session, dataset_id=None, datarun_id=None,
                        algorithm=None, ignore_gridding_done=True,
                        ignore_errored=True):
        """
        Return all the hyperpartitions in a given datarun by id.
        By default, only returns incomplete hyperpartitions.
        """
        query = session.query(self.Hyperpartition)
        if dataset_id is not None:
            query = query.join(self.Datarun)\
                .filter(self.Datarun.dataset_id == dataset_id)
        if datarun_id is not None:
            query = query.filter(self.Hyperpartition.datarun_id == datarun_id)
        if algorithm is not None:
            query = query.filter(self.Hyperpartition.algorithm == algorithm)
        if ignore_gridding_done:
            query = query.filter(self.Hyperpartition.status != PartitionStatus.GRIDDING_DONE)
        if ignore_errored:
            query = query.filter(self.Hyperpartition.status != PartitionStatus.ERRORED)

        return query.all()

    @try_with_session()
    def get_classifier(self, session, classifier_id):
        """ Get a specific classifier. """
        return session.query(self.Classifier).get(classifier_id)

    @try_with_session()
    def get_classifiers(self, session, dataset_id=None, datarun_id=None,
                     algorithm=None, hyperpartition_id=None, status=None):
        """ Get a set of classifiers, filtered by the passed-in arguments. """
        query = session.query(self.Classifier)
        if dataset_id is not None:
            query = query.join(self.Datarun)\
                .filter(self.Datarun.dataset_id == dataset_id)
        if datarun_id is not None:
            query = query.filter(self.Classifier.datarun_id == datarun_id)
        if algorithm is not None:
            query = query.join(self.Hyperpartition)\
                .filter(self.Hyperpartition.algorithm == algorithm)
        if hyperpartition_id is not None:
            query = query.filter(self.Classifier.hyperpartition_id == hyperpartition_id)
        if status is not None:
            query = query.filter(self.Classifier.status == status)

        return query.all()

    ###########################################################################
    ##  Special-purpose queries  ##############################################
    ###########################################################################

    @try_with_session(default=lambda: True)
    def is_datatun_gridding_done(self, session, datarun_id):
        """
        Check whether gridding is done for the entire datarun.
        """
        hyperpartitions = session.query(self.Hyperpartition)\
            .filter(self.Hyperpartition.datarun_id == datarun_id).all()

        is_done = True
        for hyperpartition in hyperpartitions:
            # If any hyperpartiton has not finished gridding or errored out, we are
            # not done.
            if hyperpartition.status == PartitionStatus.INCOMPLETE:
                is_done = False

        return is_done

    @try_with_session(default=int)
    def get_number_of_hyperpartition_errors(self, session, hyperpartition_id):
        """
        Get the number of classifiers that have errored using a specified
        hyperpartition.
        """
        classifiers = session.query(self.Classifier)\
            .filter(and_(self.Classifier.hyperpartition_id == hyperpartition_id,
                         self.Classifier.status == ClassifierStatus.ERRORED)).all()
        return len(classifiers)

    @try_with_session(default=list)
    def get_algorithms(self, session, dataset_id=None, datarun_id=None,
                       ignore_errored=False, ignore_gridding_done=False):
        """ Get all algorithms used in a particular datarun. """
        hyperpartitions = self.get_hyperpartitions(dataset_id=dataset_id,
                                           datarun_id=datarun_id,
                                           ignore_gridding_done=False,
                                           ignore_errored=False)
        algorithms = set(f.algorithm for f in hyperpartitions)
        return list(algorithms)

    @try_with_session()
    def get_maximum_y(self, session, datarun_id, score_target):
        """ Get the maximum value of a numeric column by name, or None. """
        result = session.query(func.max(getattr(self.Classifier, score_target)))\
            .filter(self.Classifier.datarun_id == datarun_id).one()[0]
        if result:
            return float(result)
        return None

    @try_with_session()
    def get_best_classifier(self, session, score_target='mu_sigma',
                         dataset_id=None, datarun_id=None,
                         algorithm=None, hyperpartition_id=None):
        """
        Get the classifier with the highest lower error bound. In other words, what
        classifier has the highest value of (score.mean - 2 * score.std)?

        score_target: indicates the metric by which to judge the best classifier.
            One of ['mu_sigma', 'cv_judgment_metric', 'test_judgment_metric'].
        """
        if score_target == 'mu_sigma':
            func = lambda l: l.cv_judgment_metric - 2 * l.cv_judgment_metric_stdev
        else:
            func = attrgetter(score_target)

        classifiers = self.get_classifiers(dataset_id=dataset_id,
                                     datarun_id=datarun_id,
                                     algorithm=algorithm,
                                     hyperpartition_id=hyperpartition_id,
                                     status=ClassifierStatus.COMPLETE)

        if not classifiers:
            return None

        best = max(classifiers, key=func)
        return best

    ###########################################################################
    ##  Methods to update the database  #######################################
    ###########################################################################

    @try_with_session(commit=True)
    def create_classifier(self, session, hyperpartition_id, datarun_id, host, params):
        """
        Save a new, fully qualified classifier object to the database.

        Returns: the ID of the newly-created classifier
        """
        classifier = self.Classifier(hyperpartition_id=hyperpartition_id,
                               datarun_id=datarun_id,
                               host=host,
                               params=params,
                               started=datetime.now(),
                               status=ClassifierStatus.RUNNING)
        session.add(classifier)
        hyperpartition = session.query(self.Hyperpartition).get(hyperpartition_id)
        hyperpartition.classifiers += 1

        return classifier.id

    @try_with_session(commit=True)
    def complete_classifier(self, session, classifier_id, trainable_params,
                         dimensions, model_path, metric_path,
                         cv_score, cv_stdev, test_score):
        """
        Set all the parameters on a classifier that haven't yet been set, and mark
        it as complete.
        """
        classifier = session.query(self.Classifier).get(classifier_id)

        classifier.trainable_params = trainable_params
        classifier.dimensions = dimensions
        classifier.model_path = model_path
        classifier.metric_path = metric_path
        classifier.cv_judgment_metric = cv_score
        classifier.cv_judgment_metric_stdev = cv_stdev
        classifier.test_judgment_metric = test_score

        classifier.completed = datetime.now()
        classifier.status = ClassifierStatus.COMPLETE

    @try_with_session(commit=True)
    def mark_classifier_errored(self, session, classifier_id, error_msg):
        """
        Mark an existing classifier as having errored and set the error message. If
        the classifier's hyperpartiton has produced too many erring classifiers, mark it
        as errored as well.
        """
        classifier = session.query(self.Classifier).get(classifier_id)
        classifier.error_msg = error_msg
        classifier.status = ClassifierStatus.ERRORED
        classifier.completed = datetime.now()
        if self.get_number_of_hyperpartition_errors(classifier.hyperpartition_id) > \
                MAX_HYPERPARTITION_ERRORS:
            self.mark_hyperpartition_errored(classifier.hyperpartition_id)

    @try_with_session(commit=True)
    def mark_hyperpartition_gridding_done(self, session, hyperpartition_id):
        """
        Mark a hyperpartiton as having all of its possible grid points explored.
        """
        hyperpartition = session.query(self.Hyperpartition)\
            .filter(self.Hyperpartition.id == hyperpartition_id).one()
        hyperpartition.status = PartitionStatus.GRIDDING_DONE

    @try_with_session(commit=True)
    def mark_hyperpartition_errored(self, session, hyperpartition_id):
        """
        Mark a hyperpartiton as having had too many classifier errors. This will
        prevent more classifiers from being trained on this hyperpartiton in the
        future.
        """
        hyperpartition = session.query(self.Hyperpartition)\
            .filter(self.Hyperpartition.id == hyperpartition_id).one()
        hyperpartition.status = PartitionStatus.ERRORED

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
