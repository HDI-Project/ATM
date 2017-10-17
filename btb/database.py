#!/usr/bin/python
# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func, and_

import traceback
import random, sys
import os
from datetime import datetime
import warnings
import pdb

from btb.utilities import *
from btb.config import Config


def try_with_session(default=lambda: None):
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
            except Exception:
                session.rollback()
                result = default()
                argstr = ', ',join(args)
                kwargstr = ', ',join(['%s=%s' % kv for kv in kwargs.items()])
                print "Error in %s(%s, %s):" % (func.__name__, argstr, kwargstr)
                print traceback.format_exc()
            finally:
                session.close()

            return result


class Database(object):
    def __init__(self, config):
        """
        Accepts a config (for database connection info), and defines SQLAlchemy
        ORM objects for all the tables in the database.
        """
        dialect = config.get(Config.DATAHUB, Config.DATAHUB_DIALECT)
        user = config.get(Config.DATAHUB, Config.DATAHUB_USERNAME)
        password = config.get(Config.DATAHUB, Config.DATAHUB_PASSWORD)
        host = config.get(Config.DATAHUB, Config.DATAHUB_HOST)
        port = int(config.get(Config.DATAHUB, Config.DATAHUB_PORT))
        database = config.get(Config.DATAHUB, Config.DATAHUB_DATABASE)
        query = config.get(Config.DATAHUB, Config.DATAHUB_QUERY)

        db_string = '%s://%s:%s@%s:%d/%s?%s' % (
            dialect, user, password, host, port, database, query)
        self.engine = create_engine(db_string)

        self.get_session = sessionmaker(bind=engine, expire_on_commit=False)

    def define_tables(self):
        metadata = MetaData(bind=self.engine)
        Base = declarative_base()

        class Algorithm(Base):
            __table__ = Table('algorithms', metadata, autoload=True)
            def __repr__(self):
                return "<%s>" % self.name

        self.Algorithm = Algorithm

        class Datarun(Base):
            __table__ = Table('dataruns', metadata, autoload=True)

            @property
            def wrapper(self):
                return Base64ToObject(self.datawrapper)

            @wrapper.setter
            def wrapper(self, value):
                self.datawrapper = ObjectToBase64(value)

            def __repr__(self):
                base = "<%s:, frozen: %s, sampling: %s, budget: %s, status: %s>"
                status = ""
                if self.started == None:
                    status = "pending"
                elif self.started != None and self.completed == None:
                    status = "running"
                elif self.started != None and self.completed != None:
                    status = "done"
                return base % (self.name, self.frozen_selection,
                               self.sample_selection, self.budget, status)

        self.Datarun = Datarun

        class FrozenSet(Base):
            __table__ = Table('frozen_sets', metadata, autoload=True)

            @property
            def optimizables(self):
                return Base64ToObject(self.optimizables64)

            @optimizables.setter
            def optimizables(self, value):
                self.optimizables64 = ObjectToBase64(value)

            @property
            def frozens(self):
                return Base64ToObject(self.frozens64)

            @frozens.setter
            def frozens(self, value):
                self.frozens64 = ObjectToBase64(value)

            @property
            def constants(self):
                return Base64ToObject(self.constants64)

            @constants.setter
            def constants(self, value):
                self.constants64 = ObjectToBase64(value)

            def __repr__(self):
                return "<%s: %s>" % (self.algorithm, self.frozen_hash)

        self.FrozenSet = FrozenSet

        class Learner(Base):
            __table__ = Table('learners', metadata, autoload=True)

            @property
            def params(self):
                return Base64ToObject(self.params64)

            @params.setter
            def params(self, value):
                self.params64 = ObjectToBase64(value)

            @property
            def trainable_params(self):
                return Base64ToObject(self.trainable_params64)

            @trainable_params.setter
            def trainable_params(self, value):
                self.trainable_params64 = ObjectToBase64(value)

            @property
            def test_accuracies(self):
                return Base64ToObject(self.test_accuracies64)

            @test_accuracies.setter
            def test_accuracies(self, value):
                self.test_accuracies64 = ObjectToBase64(value)

            @property
            def test_cohen_kappas(self):
                return Base64ToObject(self.test_cohen_kappas64)

            @test_cohen_kappas.setter
            def test_cohen_kappas(self, value):
                self.test_cohen_kappas64 = ObjectToBase64(value)

            @property
            def test_f1_scores(self):
                return Base64ToObject(self.test_f1_scores64)

            @test_f1_scores.setter
            def test_f1_scores(self, value):
                self.test_f1_scores64 = ObjectToBase64(value)

            @property
            def test_roc_curve_fprs(self):
                return Base64ToObject(self.test_roc_curve_fprs64)

            @test_roc_curve_fprs.setter
            def test_roc_curve_fprs(self, value):
                self.test_roc_curve_fprs64 = ObjectToBase64(value)

            @property
            def test_roc_curve_tprs(self):
                return Base64ToObject(self.test_roc_curve_tprs64)

            @test_roc_curve_tprs.setter
            def test_roc_curve_tprs(self, value):
                self.test_roc_curve_tprs64 = ObjectToBase64(value)

            @property
            def test_roc_curve_thresholds(self):
                return Base64ToObject(self.test_roc_curve_thresholds64)

            @test_roc_curve_thresholds.setter
            def test_roc_curve_thresholds(self, value):
                self.test_roc_curve_thresholds64 = ObjectToBase64(value)

            @property
            def test_roc_curve_aucs(self):
                return Base64ToObject(self.test_roc_curve_aucs64)

            @test_roc_curve_aucs.setter
            def test_roc_curve_aucs(self, value):
                self.test_roc_curve_aucs64 = ObjectToBase64(value)

            @property
            def test_pr_curve_precisions(self):
                return Base64ToObject(self.test_pr_curve_precisions64)

            @test_pr_curve_precisions.setter
            def test_pr_curve_precisions(self, value):
                self.test_pr_curve_precisions64 = ObjectToBase64(value)

            @property
            def test_pr_curve_recalls(self):
                return Base64ToObject(self.test_pr_curve_recalls64)

            @test_pr_curve_recalls.setter
            def test_pr_curve_recalls(self, value):
                self.test_pr_curve_recalls64 = ObjectToBase64(value)

            @property
            def test_pr_curve_thresholds(self):
                return Base64ToObject(self.test_pr_curve_thresholds64)

            @test_pr_curve_thresholds.setter
            def test_pr_curve_thresholds(self, value):
                self.test_pr_curve_thresholds64 = ObjectToBase64(value)

            @property
            def test_pr_curve_aucs(self):
                return Base64ToObject(self.test_pr_curve_aucs64)

            @test_pr_curve_aucs.setter
            def test_pr_curve_aucs(self, value):
                self.test_pr_curve_aucs64 = ObjectToBase64(value)

            @property
            def test_rank_accuracies(self):
                return Base64ToObject(self.test_rank_accuracies64)

            @test_rank_accuracies.setter
            def test_rank_accuracies(self, value):
                self.test_rank_accuracies64 = ObjectToBase64(value)

            @property
            def test_mu_sigmas(self):
                return Base64ToObject(self.test_mu_sigmas64)

            @test_mu_sigmas.setter
            def test_mu_sigmas(self, value):
                self.test_mu_sigmas64 = ObjectToBase64(value)

            def __repr__(self):
                return "<%s>" % self.algorithm

        self.Learner = Learner

    @try_with_session()
    def GetDatarun(self, session, datarun_id=None, ignore_completed=True,
                   ignore_grid_complete=False, choose_randomly=True):
        """
        Return a single datarun.
        Args:
            datarun_id: return the datarun with this id
            ignore_completed: if True, ignore completed dataruns
            ignore_grid_complete: if True, ignore dataruns with is_gridding_done
            choose_randomly: if True, choose one of the possible dataruns to
                return randomly. If False, return the first datarun present in
                the database (likely lowest id).
        """
        query = session.query(Datarun)
        if ignore_completed:
            query = query.filter(Datarun.completed == None)
        if ignore_grid_complete:
            query = query.filter(Datarun.is_gridding_done == 0)
        if datarun_id:
            query = query.filter(Datarun.id == datarun_id)

        dataruns = query.all()

        if not dataruns:
            return None

        # select only those with max priority
        max_priority = max([r.priority for r in dataruns])
        candidates = [r for r in dataruns if r.priority == max_priority]

        # choose a random candidate if necessary
        if choose_randomly:
            return candidates[random.randint(0, len(candidates) - 1)]
        return candidates[0]

    @try_with_session(lambda: True)
    def IsGriddingDoneForDatarun(self, session, datarun_id,
                                 min_num_errors_to_exclude=0):
        """ Return a boolean indicating whether gridding is done for the
        specified datarun. """
        frozen_sets = session.query(FrozenSet)\
            .filter(FrozenSet.datarun_id == datarun_id).all()

        for frozen_set in frozen_sets:
            if frozen_set.is_gridding_done == 0:
                if ((min_num_errors_to_exclude > 0) and
                       (GetNumberOfFrozenSetErrors(frozen_set_id=frozen_set.id) <
                        min_num_errors_to_exclude)):
                    is_done = False
                elif min_num_errors_to_exclude == 0:
                    is_done = False

    @try_with_session(list)
    def GetIncompletedFrozenSets(self, session, datarun_id,
                                 min_num_errors_to_exclude=0):
        """ Returns all the incomplete frozen sets in a given datarun by id.
        """
        frozen_sets = session.query(FrozenSet)\
            .filter(and_(FrozenSet.datarun_id == datarun_id,
                         FrozenSet.is_gridding_done == 0)).all()

        if min_num_errors_to_exclude > 0:
            old_list = frozen_sets
            frozen_sets = []

            for frozen_set in old_list:
                if GetNumberOfFrozenSetErrors(frozen_set_id=frozen_set.id) < \
                        min_num_errors_to_exclude:
                    frozen_sets.append(frozen_set)

        return frozen_sets

    @try_with_session(int)
    def GetNumberOfFrozenSetErrors(self, session, frozen_set_id):
        learners = session.query(Learner)\
            .filter(and_(Learner.frozen_set_id == frozen_set_id,
                         Learner.is_error == 1)).all()
        return len(learners)

    @try_with_session(list)
    def GetLearnersInFrozen(self, session, frozen_set_id):
        """ Returns all learners in a frozen set. """
        return session.query(Learner)\
            .filter(Learner.frozen_set_id == frozen_set_id).all()

    @try_with_session(list)
    def GetLearners(self, session, datarun_id):
        """ Returns all learners in a datarun.  """
        return session.query(Learner)\
            .filter(Learner.datarun_id == datarun_id)\
            .order_by(Learner.started).all()

    @try_with_session()
    def GetLearner(self, session, learner_id):
        """ Returns a specific learner.  """
        return session.query(Learner).get(learner_id)

    @try_with_session()
    def GetMaximumY(self, session, datarun_id, metric):
        """ Returns the maximum value of a numeric column by name, or None. """
        result = session.query(func.max(getattr(Learner, metric)))\
            .filter(Learner.datarun_id == datarun_id).one()[0]
        if result:
            return float(result)
        return None

    @try_with_session(lambda: (0,0))
    def get_best_so_far(self, session, datarun_id, metric):
        """
        Sort of like GetMaximumY, but looks for best standard dev below the
        mean.
        """
        maximum = 0
        best_val, best_std = 0, 0

        if metric == 'cv_judgment_metric':
            result = session.query(self.Learner.cv_judgment_metric,
                                   self.Learner.cv_judgment_metric_stdev)\
                            .filter(self.Learner.datarun_id == datarun_id)\
                            .all()
            for val, std in result:
                if val is None or std is None:
                    continue
                if val - std > maximum:
                    best_val, best_std = float(val), float(std)
                    maximum = float(val - std)

        elif metric == 'test_judgment_metric':
            result = session.query(func.max(self.Learner.test_judgment_metric))\
                            .filter(self.Learner.datarun_id == datarun_id)\
                            .one()[0]
            if result is not None and result > maximum:
                best_val = float(result)
                maximum = best_val

        return best_val, best_std

    @try_with_session()
    def MarkFrozenSetGriddingDone(self, session, frozen_set_id):
        frozen_set = session.query(FrozenSet)\
            .filter(FrozenSet.id == frozen_set_id).one()
        frozen_set.is_gridding_done = 1
        session.commit()
        # set any sqlalchemy ORM objects created by this session to the
        # detached state, so they can be used after the session is closed.
        session.expunge_all()

    @try_with_session()
    def MarkDatarunGriddingDone(self, session, datarun_id):
        datarun = session.query(Datarun).filter(Datarun.id == datarun_id).one()
        datarun.is_gridding_done = 1
        session.commit()
        session.expunge_all()

    @try_with_session()
    def MarkDatarunDone(self, session, datarun_id):
        """ Sets the completed field of the Datarun to the current datetime. """
        datarun = session.query(Datarun)\
            .filter(Datarun.id == datarun_id).one()
        datarun.completed = datetime.now()
        session.commit()
