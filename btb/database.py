#!/usr/bin/python
# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, MetaData, Table
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

from btb.utilities import *
from btb.config import Config


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
    def __init__(self, config):
        """
        Accepts a config (for database connection info), and defines SQLAlchemy
        ORM objects for all the tables in the database.
        """
        dialect = config.get(Config.DATAHUB, Config.DATAHUB_DIALECT)
        database = config.get(Config.DATAHUB, Config.DATAHUB_DATABASE)
        user = config.get(Config.DATAHUB, Config.DATAHUB_USERNAME) or None
        password = config.get(Config.DATAHUB, Config.DATAHUB_PASSWORD) or None
        host = config.get(Config.DATAHUB, Config.DATAHUB_HOST) or None
        port = config.get(Config.DATAHUB, Config.DATAHUB_PORT) or None
        query = config.get(Config.DATAHUB, Config.DATAHUB_QUERY) or None

        db_url = URL(drivername=dialect, database=database, username=user,
                     password=password, host=host, port=port, query=query)
        self.engine = create_engine(db_url)

        self.get_session = sessionmaker(bind=self.engine,
                                        expire_on_commit=False)
        self.define_tables()

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
        query = session.query(self.Datarun)
        if ignore_completed:
            query = query.filter(self.Datarun.completed == None)
        if ignore_grid_complete:
            query = query.filter(self.Datarun.is_gridding_done == 0)
        if datarun_id:
            query = query.filter(self.Datarun.id == datarun_id)

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

    @try_with_session(default=lambda: True)
    def IsGriddingDoneForDatarun(self, session, datarun_id,
                                 errors_to_exclude=0):
        """
        Check whether gridding is done for the entire datarun.
        errors_to_exclude = 0 indicates we don't care about errors.
        """
        is_done = True
        frozen_sets = session.query(self.FrozenSet)\
            .filter(self.FrozenSet.datarun_id == datarun_id).all()

        for frozen_set in frozen_sets:
            if not frozen_set.is_gridding_done:
                num_errors = self.GetNumberOfFrozenSetErrors(frozen_set.id)
                if errors_to_exclude == 0 or num_errors < errors_to_exclude:
                    is_done = False

        return is_done

    @try_with_session(default=list)
    def GetIncompleteFrozenSets(self, session, datarun_id,
                                 errors_to_exclude=0):
        """
        Returns all the incomplete frozen sets in a given datarun by id.
        """
        frozen_sets = session.query(self.FrozenSet)\
            .filter(and_(self.FrozenSet.datarun_id == datarun_id,
                         self.FrozenSet.is_gridding_done == 0)).all()

        if not errors_to_exclude:
            return frozen_sets

        old_list = frozen_sets
        frozen_sets = []

        for frozen_set in old_list:
            if (self.GetNumberOfFrozenSetErrors(frozen_set.id) <
                    errors_to_exclude):
                frozen_sets.append(frozen_set)

        return frozen_sets

    @try_with_session()
    def GetFrozenSet(self, session, frozen_set_id):
        """ Returns a specific learner.  """
        return session.query(self.FrozenSet).get(frozen_set_id)

    @try_with_session(default=int)
    def GetNumberOfFrozenSetErrors(self, session, frozen_set_id):
        learners = session.query(self.Learner)\
            .filter(and_(self.Learner.frozen_set_id == frozen_set_id,
                         self.Learner.is_error == 1)).all()
        return len(learners)

    @try_with_session(default=list)
    def GetLearnersInFrozen(self, session, frozen_set_id):
        """ Returns all learners in a frozen set. """
        return session.query(self.Learner)\
            .filter(self.Learner.frozen_set_id == frozen_set_id).all()

    @try_with_session(default=list)
    def GetLearners(self, session, datarun_id):
        """ Returns all learners in a datarun.  """
        return session.query(self.Learner)\
            .filter(self.Learner.datarun_id == datarun_id)\
            .order_by(self.Learner.started).all()

    @try_with_session()
    def GetLearner(self, session, learner_id):
        """ Returns a specific learner.  """
        return session.query(self.Learner).get(learner_id)

    @try_with_session()
    def GetMaximumY(self, session, datarun_id, score_target):
        """ Returns the maximum value of a numeric column by name, or None. """
        result = session.query(func.max(getattr(self.Learner, score_target)))\
            .filter(self.Learner.datarun_id == datarun_id).one()[0]
        if result:
            return float(result)
        return None

    @try_with_session(default=lambda: (0, 0))
    def get_best_so_far(self, session, datarun_id, score_target):
        """
        Sort of like GetMaximumY, but retuns the score with the highest lower
        error bound. In other words, what is the highest value of (score.mean -
        2 * score.std) for any learner?
        """
        maximum = 0
        best_val, best_err = 0, 0

        if score_target == 'cv_judgment_metric':
            result = session.query(self.Learner.cv_judgment_metric,
                                   self.Learner.cv_judgment_metric_stdev)\
                            .filter(self.Learner.datarun_id == datarun_id)\
                            .all()
            for val, std in result:
                if val is None or std is None:
                    continue
                if val - 2 * std > maximum:
                    best_val, best_err = float(val), 2 * float(std)
                    maximum = float(val - 2 * std)

        elif score_target == 'test_judgment_metric':
            result = session.query(func.max(self.Learner.test_judgment_metric))\
                            .filter(self.Learner.datarun_id == datarun_id)\
                            .one()[0]
            if result is not None and result > maximum:
                best_val = float(result)
                maximum = best_val

        return best_val, best_err

    @try_with_session(commit=True)
    def MarkFrozenSetGriddingDone(self, session, frozen_set_id):
        frozen_set = session.query(self.FrozenSet)\
            .filter(self.FrozenSet.id == frozen_set_id).one()
        frozen_set.is_gridding_done = 1

    @try_with_session(commit=True)
    def MarkDatarunGriddingDone(self, session, datarun_id):
        datarun = session.query(self.Datarun)\
            .filter(self.Datarun.id == datarun_id).one()
        datarun.is_gridding_done = 1

    @try_with_session(commit=True)
    def MarkDatarunDone(self, session, datarun_id):
        """ Sets the completed field of the Datarun to the current datetime. """
        datarun = session.query(self.Datarun)\
            .filter(self.Datarun.id == datarun_id).one()
        datarun.completed = datetime.now()
