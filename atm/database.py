from __future__ import absolute_import, unicode_literals

import hashlib
import json
import os
import pickle
from builtins import object
from datetime import datetime
from operator import attrgetter

import numpy as np
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sqlalchemy import (
    Column, DateTime, Enum, ForeignKey, Integer, MetaData, Numeric, String, Text, and_,
    create_engine, func, inspect)
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.orm.properties import ColumnProperty

from atm.constants import (
    BUDGET_TYPES, CLASSIFIER_STATUS, DATARUN_STATUS, METRICS, PARTITION_STATUS, SCORE_TARGETS,
    ClassifierStatus, PartitionStatus, RunStatus)
from atm.dataloader import load_data
from atm.utilities import base_64_to_object, object_to_base_64

# The maximum number of errors allowed in a single hyperpartition. If more than
# this many classifiers using a hyperpartition error, the hyperpartition will be
# considered broken and ignored for the rest of the datarun.
MAX_HYPERPARTITION_ERRORS = 3


def try_with_session(commit=False):
    """
    Decorator for instance methods on Database that need a sqlalchemy session.

    This wrapping function checks if the Database has an active session yet. If
    not, it wraps the function call in a `with db_session():` block.
    """
    def wrap(func):
        def call(db, *args, **kwargs):
            # if the Database has an active session, don't create a new one
            if db.session is not None:
                result = func(db, *args, **kwargs)
                if commit:
                    db.session.commit()
            else:
                # otherwise, use the session generator
                with db_session(db, commit=commit):
                    result = func(db, *args, **kwargs)

            return result
        return call
    return wrap


class db_session(object):
    def __init__(self, db, commit=False):
        self.db = db
        self.commit = commit

    def __enter__(self):
        self.db.session = self.db.get_session()

    def __exit__(self, type, error, traceback):
        if error is not None:
            self.db.session.rollback()
        elif self.commit:
            self.db.session.commit()

        self.db.session.close()
        self.db.session = None


class Database(object):
    def __init__(self, dialect, database, username=None, password=None,
                 host=None, port=None, query=None):
        """
        Accepts configuration for a database connection, and defines SQLAlchemy
        ORM objects for all the tables in the database.
        """

        # Prepare environment for pymysql
        pymysql.install_as_MySQLdb()
        pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
        pymysql.converters.conversions = pymysql.converters.encoders.copy()
        pymysql.converters.conversions.update(pymysql.converters.decoders)

        db_url = URL(drivername=dialect, database=database, username=username,
                     password=password, host=host, port=port, query=query)
        self.engine = create_engine(db_url)
        self.session = None
        self.get_session = sessionmaker(bind=self.engine,
                                        expire_on_commit=False)

        # create ORM objects for the tables
        self._define_tables()

    def _define_tables(self):
        """
        Define the SQLAlchemy ORM class for each table in the ModelHub database.

        These must be defined after the Database class is initialized so that
        the database metadata is available (at runtime).
        If the database does not already exist, it will be created. If it does
        exist, it will not be updated with new schema -- after schema changes,
        the database must be destroyed and reinialized.
        """
        metadata = MetaData(bind=self.engine)
        Base = declarative_base(metadata=metadata)

        class Dataset(Base):
            __tablename__ = 'datasets'

            id = Column(Integer, primary_key=True, autoincrement=True)
            name = Column(String(100), nullable=False)

            # columns necessary for loading/processing data
            class_column = Column(String(100), nullable=False)
            train_path = Column(String(200), nullable=False)
            test_path = Column(String(200))
            description = Column(String(1000))

            # metadata columns, for convenience
            n_examples = Column(Integer, nullable=False)
            k_classes = Column(Integer, nullable=False)
            d_features = Column(Integer, nullable=False)
            majority = Column(Numeric(precision=10, scale=9), nullable=False)
            size_kb = Column(Integer, nullable=False)

            def load_(self, test_size=0.3, random_state=0, aws_conf=None):
                data = load_data(self.name, self.train_path, aws_conf)

                if self.test_path:
                    if self.name.endswith('.csv'):
                        test_name = self.name.replace('.csv', '_test.csv')
                    else:
                        test_name = self.name + '_test'

                    test_data = load_data(test_name, self.test_path, aws_conf)
                    return data, test_data

                else:
                    return train_test_split(data, test_size=test_size, random_state=random_state)

            def _add_extra_fields(self, aws_conf):
                data = load_data(self.name, self.train_path, aws_conf)

                # compute the portion of labels that are the most common value
                counts = data[self.class_column].value_counts()
                total_features = data.shape[1] - 1
                for column in data.columns:
                    if data[column].dtype == 'object':
                        total_features += len(np.unique(data[column])) - 1

                majority_percentage = float(max(counts)) / float(sum(counts))

                self.n_examples = len(data)
                self.d_features = total_features
                self.majority = majority_percentage
                self.k_classes = len(np.unique(data[self.class_column]))
                self.size_kb = int(np.array(data).nbytes / 1000)

            @staticmethod
            def _make_name(path):
                md5 = hashlib.md5(path.encode('utf-8'))
                return md5.hexdigest()

            def __init__(self, class_column, train_path, name=None, description=None,
                         test_path=None, aws_conf=None, **kwargs):

                self.name = name or self._make_name(train_path)
                self.class_column = class_column
                self.description = description
                self.train_path = train_path
                self.test_path = test_path

                self._add_extra_fields(aws_conf)

            def __repr__(self):
                base = "<%s: %s, %d classes, %d features, %d rows>"
                return base % (self.name, self.description, self.k_classes,
                               self.d_features, self.n_examples)

        class Datarun(Base):
            __tablename__ = 'dataruns'

            # relational columns
            id = Column(Integer, primary_key=True, autoincrement=True)
            dataset_id = Column(Integer, ForeignKey('datasets.id'))
            dataset = relationship('Dataset', back_populates='dataruns')

            description = Column(String(200), nullable=False)
            priority = Column(Integer)

            # hyperparameter selection and tuning settings
            selector = Column(String(200), nullable=False)
            k_window = Column(Integer)
            tuner = Column(String(200), nullable=False)
            gridding = Column(Integer, nullable=False)
            r_minimum = Column(Integer)

            # budget settings
            budget_type = Column(Enum(*BUDGET_TYPES))
            budget = Column(Integer)
            deadline = Column(DateTime)

            # which metric to use for judgment, and how to compute it
            metric = Column(Enum(*METRICS))
            score_target = Column(Enum(*[s + '_judgment_metric' for s in
                                         SCORE_TARGETS]))

            # variables that store the status of the datarun
            start_time = Column(DateTime)
            end_time = Column(DateTime)
            status = Column(Enum(*DATARUN_STATUS), default=RunStatus.PENDING)

            def __repr__(self):
                base = "<ID = %d, dataset ID = %s, strategy = %s, budget = %s (%s), status: %s>"
                return base % (self.id, self.dataset_id, self.description,
                               self.budget_type, self.budget, self.status)

        Dataset.dataruns = relationship('Datarun', order_by='Datarun.id',
                                        back_populates='dataset')

        class Hyperpartition(Base):
            __tablename__ = 'hyperpartitions'

            # relational columns
            id = Column(Integer, primary_key=True, autoincrement=True)
            datarun_id = Column(Integer, ForeignKey('dataruns.id'))
            datarun = relationship('Datarun', back_populates='hyperpartitions')

            # name of or path to a configured classification method
            method = Column(String(255))

            # list of categorical parameters whose values are fixed to define
            # this hyperpartition
            categorical_hyperparameters_64 = Column(Text)

            # list of continuous parameters which are not fixed; their values
            # must be selected by a Tuner
            tunable_hyperparameters_64 = Column(Text)

            # list of categorical or continuous parameters whose values are
            # always fixed. These do not define the hyperpartition, but their
            # values must be passed on to the method. Here for convenience.
            constant_hyperparameters_64 = Column(Text)

            # has the partition had too many errors, or is gridding done?
            status = Column(Enum(*PARTITION_STATUS),
                            default=PartitionStatus.INCOMPLETE)

            @property
            def categoricals(self):
                """
                A list of categorical variables along with the fixed values
                which define this hyperpartition.
                Each element is a ('name', HyperParameter) tuple.
                """
                return base_64_to_object(self.categorical_hyperparameters_64)

            @categoricals.setter
            def categoricals(self, value):
                self.categorical_hyperparameters_64 = object_to_base_64(value)

            @property
            def tunables(self):
                """
                A list of parameters which are unspecified and must be selected
                with a Tuner. Each element is a ('name', HyperParameter) tuple.
                """
                return base_64_to_object(self.tunable_hyperparameters_64)

            @tunables.setter
            def tunables(self, value):
                self.tunable_hyperparameters_64 = object_to_base_64(value)

            @property
            def constants(self):
                return base_64_to_object(self.constant_hyperparameters_64)

            @constants.setter
            def constants(self, value):
                self.constant_hyperparameters_64 = object_to_base_64(value)

            def __repr__(self):
                return "<%s: %s>" % (self.method, self.categoricals)

        Datarun.hyperpartitions = relationship('Hyperpartition',
                                               order_by='Hyperpartition.id',
                                               back_populates='datarun')

        class Classifier(Base):
            __tablename__ = 'classifiers'

            # relational columns
            id = Column(Integer, primary_key=True, autoincrement=True)
            datarun_id = Column(Integer, ForeignKey('dataruns.id'))
            datarun = relationship('Datarun', back_populates='classifiers')
            hyperpartition_id = Column(Integer, ForeignKey('hyperpartitions.id'))
            hyperpartition = relationship('Hyperpartition',
                                          back_populates='classifiers')

            # name of the host where the model was trained
            host = Column(String(50))

            # these columns point to where the output is stored
            model_location = Column(String(300))
            metrics_location = Column(String(300))

            # base 64 encoding of the hyperparameter names and values
            hyperparameter_values_64 = Column(Text, nullable=False)

            # performance metrics
            cv_judgment_metric = Column(Numeric(precision=20, scale=10))
            cv_judgment_metric_stdev = Column(Numeric(precision=20, scale=10))
            test_judgment_metric = Column(Numeric(precision=20, scale=10))

            start_time = Column(DateTime)
            end_time = Column(DateTime)
            status = Column(Enum(*CLASSIFIER_STATUS), nullable=False)
            error_message = Column(Text)

            @property
            def hyperparameter_values(self):
                return base_64_to_object(self.hyperparameter_values_64)

            @hyperparameter_values.setter
            def hyperparameter_values(self, value):
                self.hyperparameter_values_64 = object_to_base_64(value)

            @property
            def mu_sigma_judgment_metric(self):
                # compute the lower confidence bound on the cross-validated
                # judgment metric
                if self.cv_judgment_metric is None:
                    return None
                return (self.cv_judgment_metric - 2 * self.cv_judgment_metric_stdev)

            def __repr__(self):
                params = ', '.join(['%s: %s' % i for i in
                                    list(self.hyperparameter_values.items())])
                return "<id=%d, params=(%s)>" % (self.id, params)

        Datarun.classifiers = relationship('Classifier',
                                           order_by='Classifier.id',
                                           back_populates='datarun')
        Hyperpartition.classifiers = relationship('Classifier',
                                                  order_by='Classifier.id',
                                                  back_populates='hyperpartition')

        self.Dataset = Dataset
        self.Datarun = Datarun
        self.Hyperpartition = Hyperpartition
        self.Classifier = Classifier

        Base.metadata.create_all(bind=self.engine)

    # ##########################################################################
    # #  Save/load the database  ###############################################
    # ##########################################################################

    @try_with_session()
    def to_csv(self, path):
        """
        Save the entire ModelHub database as a set of CSVs in the given
        directory.
        """
        for table in ['datasets', 'dataruns', 'hyperpartitions', 'classifiers']:
            df = pd.read_sql('SELECT * FROM %s' % table, self.session.bind)
            df.to_csv(os.path.join(path, '%s.csv' % table), index=False)

    @try_with_session(commit=True)
    def from_csv(self, path):
        """
        Load a snapshot of the ModelHub database from a set of CSVs in the given
        directory.
        """
        for model, table in [(self.Dataset, 'dataset'),
                             (self.Datarun, 'datarun'),
                             (self.Hyperpartition, 'hyperpartition'),
                             (self.Classifier, 'classifier')]:
            df = pd.read_csv(os.path.join(path, '%ss.csv' % table))

            # parse datetime columns. This is necessary because SQLAlchemy can't
            # interpret strings as datetimes on its own.
            # yes, this is the easiest way to do it
            for c in inspect(model).attrs:
                if not isinstance(c, ColumnProperty):
                    continue
                col = c.columns[0]
                if isinstance(col.type, DateTime):
                    df[c.key] = pd.to_datetime(df[c.key],
                                               infer_datetime_format=True)

            for _, r in df.iterrows():
                # replace NaN and NaT with None
                for k, v in list(r.iteritems()):
                    if pd.isnull(v):
                        r[k] = None

                # insert the row into the database
                create_func = getattr(self, 'create_%s' % table)
                create_func(**r)

    # ##########################################################################
    # #  Standard query methods  ###############################################
    # ##########################################################################

    @try_with_session()
    def get_dataset(self, dataset_id):
        """ Get a specific dataset. """
        return self.session.query(self.Dataset).get(dataset_id)

    @try_with_session()
    def get_datarun(self, datarun_id):
        """ Get a specific datarun. """
        return self.session.query(self.Datarun).get(datarun_id)

    @try_with_session()
    def get_dataruns(self, ignore_pending=False, ignore_running=False,
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
        query = self.session.query(self.Datarun)
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
    def get_hyperpartition(self, hyperpartition_id):
        """ Get a specific classifier.  """
        return self.session.query(self.Hyperpartition).get(hyperpartition_id)

    @try_with_session()
    def get_hyperpartitions(self, dataset_id=None, datarun_id=None, method=None,
                            ignore_gridding_done=True, ignore_errored=True):
        """
        Return all the hyperpartitions in a given datarun by id.
        By default, only returns incomplete hyperpartitions.
        """
        query = self.session.query(self.Hyperpartition)
        if dataset_id is not None:
            query = query.join(self.Datarun)\
                .filter(self.Datarun.dataset_id == dataset_id)
        if datarun_id is not None:
            query = query.filter(self.Hyperpartition.datarun_id == datarun_id)
        if method is not None:
            query = query.filter(self.Hyperpartition.method == method)
        if ignore_gridding_done:
            query = query.filter(self.Hyperpartition.status != PartitionStatus.GRIDDING_DONE)
        if ignore_errored:
            query = query.filter(self.Hyperpartition.status != PartitionStatus.ERRORED)

        return query.all()

    @try_with_session()
    def get_classifier(self, classifier_id):
        """ Get a specific classifier. """
        return self.session.query(self.Classifier).get(classifier_id)

    @try_with_session()
    def get_classifiers(self, dataset_id=None, datarun_id=None, method=None,
                        hyperpartition_id=None, status=None):
        """ Get a set of classifiers, filtered by the passed-in arguments. """
        query = self.session.query(self.Classifier)
        if dataset_id is not None:
            query = query.join(self.Datarun)\
                .filter(self.Datarun.dataset_id == dataset_id)
        if datarun_id is not None:
            query = query.filter(self.Classifier.datarun_id == datarun_id)
        if method is not None:
            query = query.join(self.Hyperpartition)\
                .filter(self.Hyperpartition.method == method)
        if hyperpartition_id is not None:
            query = query.filter(self.Classifier.hyperpartition_id == hyperpartition_id)
        if status is not None:
            query = query.filter(self.Classifier.status == status)

        return query.all()

    # ##########################################################################
    # #  Special-purpose queries  ##############################################
    # ##########################################################################

    @try_with_session()
    def is_datatun_gridding_done(self, datarun_id):
        """
        Check whether gridding is done for the entire datarun.
        """
        datarun = self.get_datarun(datarun_id)
        is_done = True
        for hyperpartition in datarun.hyperpartitions:
            # If any hyperpartiton has not finished gridding or errored out,
            # gridding is not done for the datarun.
            if hyperpartition.status == PartitionStatus.INCOMPLETE:
                is_done = False

        return is_done

    @try_with_session()
    def get_number_of_hyperpartition_errors(self, hyperpartition_id):
        """
        Get the number of classifiers that have errored using a specified
        hyperpartition.
        """
        classifiers = self.session.query(self.Classifier)\
            .filter(and_(self.Classifier.hyperpartition_id == hyperpartition_id,
                         self.Classifier.status == ClassifierStatus.ERRORED)).all()
        return len(classifiers)

    @try_with_session()
    def get_methods(self, dataset_id=None, datarun_id=None,
                    ignore_errored=False, ignore_gridding_done=False):
        """ Get all methods used in a particular datarun. """
        hyperpartitions = self.get_hyperpartitions(dataset_id=dataset_id,
                                                   datarun_id=datarun_id,
                                                   ignore_gridding_done=False,
                                                   ignore_errored=False)
        methods = set(f.method for f in hyperpartitions)
        return list(methods)

    @try_with_session()
    def get_maximum_y(self, datarun_id, score_target):
        """ Get the maximum value of a numeric column by name, or None. """
        query = self.session.query(func.max(getattr(self.Classifier,
                                                    score_target)))
        result = query.filter(self.Classifier.datarun_id == datarun_id).first()
        if result:
            return float(result)
        return None

    @try_with_session()
    def get_best_classifier(self, score_target, dataset_id=None,
                            datarun_id=None, method=None,
                            hyperpartition_id=None):
        """
        Get the classifier with the best judgment metric, as indicated by
        score_target.

        score_target: indicates the metric by which to judge the best classifier.
        """
        classifiers = self.get_classifiers(dataset_id=dataset_id,
                                           datarun_id=datarun_id,
                                           method=method,
                                           hyperpartition_id=hyperpartition_id,
                                           status=ClassifierStatus.COMPLETE)

        if '_judgment_metric' not in score_target:
            score_target += '_judgment_metric'

        if not classifiers:
            return None
        return max(classifiers, key=attrgetter(score_target))

    @try_with_session()
    def load_model(self, classifier_id):
        clf = self.get_classifier(classifier_id)
        with open(clf.model_location, 'rb') as f:
            return pickle.load(f)

    @try_with_session()
    def load_metrics(self, classifier_id):
        clf = self.get_classifier(classifier_id)
        with open(clf.metrics_location, 'r') as f:
            return json.load(f)

    # ##########################################################################
    # #  Methods to update the database  #######################################
    # ##########################################################################

    @try_with_session(commit=True)
    def create_dataset(self, **kwargs):
        dataset = self.Dataset(**kwargs)
        self.session.add(dataset)
        return dataset

    @try_with_session(commit=True)
    def create_datarun(self, **kwargs):
        datarun = self.Datarun(**kwargs)
        self.session.add(datarun)
        return datarun

    @try_with_session(commit=True)
    def create_hyperpartition(self, **kwargs):
        partition = self.Hyperpartition(**kwargs)
        self.session.add(partition)
        return partition

    @try_with_session(commit=True)
    def create_classifier(self, **kwargs):
        classifier = self.Classifier(**kwargs)
        self.session.add(classifier)
        return classifier

    @try_with_session(commit=True)
    def start_classifier(self, hyperpartition_id, datarun_id, host,
                         hyperparameter_values):
        """
        Save a new, fully qualified classifier object to the database.
        Returns: the ID of the newly-created classifier
        """
        classifier = self.Classifier(hyperpartition_id=hyperpartition_id,
                                     datarun_id=datarun_id,
                                     host=host,
                                     hyperparameter_values=hyperparameter_values,
                                     start_time=datetime.now(),
                                     status=ClassifierStatus.RUNNING)
        self.session.add(classifier)
        return classifier

    @try_with_session(commit=True)
    def complete_classifier(self, classifier_id, model_location,
                            metrics_location, cv_score, cv_stdev, test_score):
        """
        Set all the parameters on a classifier that haven't yet been set, and mark
        it as complete.
        """
        classifier = self.session.query(self.Classifier).get(classifier_id)

        classifier.model_location = model_location
        classifier.metrics_location = metrics_location
        classifier.cv_judgment_metric = cv_score
        classifier.cv_judgment_metric_stdev = cv_stdev
        classifier.test_judgment_metric = test_score
        classifier.end_time = datetime.now()
        classifier.status = ClassifierStatus.COMPLETE

    @try_with_session(commit=True)
    def mark_classifier_errored(self, classifier_id, error_message):
        """
        Mark an existing classifier as having errored and set the error message. If
        the classifier's hyperpartiton has produced too many erring classifiers, mark it
        as errored as well.
        """
        classifier = self.session.query(self.Classifier).get(classifier_id)
        classifier.error_message = error_message
        classifier.status = ClassifierStatus.ERRORED
        classifier.end_time = datetime.now()

        noh_errors = self.get_number_of_hyperpartition_errors(classifier.hyperpartition_id)
        if noh_errors > MAX_HYPERPARTITION_ERRORS:
            self.mark_hyperpartition_errored(classifier.hyperpartition_id)

    @try_with_session(commit=True)
    def mark_hyperpartition_gridding_done(self, hyperpartition_id):
        """
        Mark a hyperpartiton as having all of its possible grid points explored.
        """
        hyperpartition = self.get_hyperpartition(hyperpartition_id)
        hyperpartition.status = PartitionStatus.GRIDDING_DONE

    @try_with_session(commit=True)
    def mark_hyperpartition_errored(self, hyperpartition_id):
        """
        Mark a hyperpartiton as having had too many classifier errors. This will
        prevent more classifiers from being trained on this hyperpartiton in the
        future.
        """
        hyperpartition = self.get_hyperpartition(hyperpartition_id)
        hyperpartition.status = PartitionStatus.ERRORED

    @try_with_session(commit=True)
    def mark_datarun_running(self, datarun_id):
        """
        Set the status of the Datarun to RUNNING and set the 'start_time' field
        to the current datetime.
        """
        datarun = self.get_datarun(datarun_id)
        if datarun.status == RunStatus.PENDING:
            datarun.status = RunStatus.RUNNING
            datarun.start_time = datetime.now()

    @try_with_session(commit=True)
    def mark_datarun_complete(self, datarun_id):
        """
        Set the status of the Datarun to COMPLETE and set the 'end_time' field
        to the current datetime.
        """
        datarun = self.get_datarun(datarun_id)
        datarun.status = RunStatus.COMPLETE
        datarun.end_time = datetime.now()
