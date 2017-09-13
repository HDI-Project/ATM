#!/usr/bin/python
# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func, and_

import traceback
import random, sys
from delphi.utilities import *
from delphi.config import Config
import os
from datetime import datetime
import warnings

# TODO: config should only be loaded once
configpath = os.getenv('DELPHI_CONFIG_FILE')
if configpath == None:
    warnings.warn('No config file environmental variable, using default config/atm.cnf')
    configpath = 'config/atm.cnf'
assert os.path.isfile(configpath), 'Configuration file not found. ({})'.format(configpath)
config = Config(configpath)

DIALECT = config.get(Config.DATAHUB, Config.DATAHUB_DIALECT)
DATABASE = config.get(Config.DATAHUB, Config.DATAHUB_DATABASE)
USER = config.get(Config.DATAHUB, Config.DATAHUB_USERNAME)
PASSWORD = config.get(Config.DATAHUB, Config.DATAHUB_PASSWORD)
HOST = config.get(Config.DATAHUB, Config.DATAHUB_HOST)
PORT = int(config.get(Config.DATAHUB, Config.DATAHUB_PORT))
QUERY = config.get(Config.DATAHUB, Config.DATAHUB_QUERY)

Base = declarative_base()
DB_STRING = '%s://%s:%s@%s:%d/%s?%s' % (
    DIALECT, USER, PASSWORD, HOST, PORT, DATABASE, QUERY)
engine = create_engine(DB_STRING)
metadata = MetaData(bind=engine)
Session = sessionmaker(bind=engine, expire_on_commit=False)

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
        return base % (self.name, self.frozen_selection, self.sample_selection, self.budget, status)

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

class Algorithm(Base):
    __table__ = Table('algorithms', metadata, autoload=True)
    def __repr__(self):
        return "<%s>" % self.name

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

    @property
    def cv_judgment_metric(self):
        return Base64ToObject(self.cv_judgment_metric64)

    @cv_judgment_metric.setter
    def cv_judgment_metric(self, value):
        self.cv_judgment_metric64 = ObjectToBase64(value)

    def __repr__(self):
        return "<%s>" % self.algorithm

def GetConnection():
    """
    Returns a database connection.

    [***] DO NOT FORGET TO CLOSE AFTERWARDS:
    >>> connection = GetConnection()
    >>> # do some stuff ...
    >>> connection.close()
    """
    return Session()

def GetAllDataruns():
    """
    Among the incomplete dataruns with maximal priority,
    returns one at random.
    """
    # get all incomplete dataruns
    session = None
    dataruns = []
    try:
        session = GetConnection()

        query = session.query(Datarun)

        dataruns = query.all()
        session.close()

        if not dataruns:
            return []

        return dataruns

    except Exception:
        print "Error in GetDatarun():", traceback.format_exc()

    finally:
        if session:
            session.close()

def GetDatarun(datarun_id=None, ignore_completed=True, ignore_grid_complete=False, chose_randomly=True):
    """
    Among the incomplete dataruns with maximal priority,
    returns one at random.
    """
    # get all incomplete dataruns
    session = None
    dataruns = []
    try:
        session = GetConnection()

        if ignore_completed and ignore_grid_complete: # this is ignored if ignore_completed is true
            query = session.query(Datarun).filter(and_(Datarun.completed == None, Datarun.is_gridding_done == 0))
        elif ignore_completed:
            query = session.query(Datarun).filter(Datarun.completed == None)
        elif ignore_grid_complete: # this is ignored if ignore_completed is true
            query = session.query(Datarun).filter(Datarun.is_gridding_done == 0)
        else:
            query = session.query(Datarun)

        if datarun_id:
            query = query.filter(Datarun.id == datarun_id)

        dataruns = query.all()
        session.close()

        if not dataruns:
            return []

        # select only those with max priority
        max_priority = max([x.priority for x in dataruns])
        candidates = []
        for run in dataruns:
            if run.priority == max_priority:
                candidates.append(run)

        # choose one if there is at least one
        if candidates and chose_randomly:
            chosen = candidates[random.randint(0, len(candidates) - 1)]
            if chosen:
                return chosen
        elif candidates and not chose_randomly:
            chosen = candidates[0]
            if chosen:
                return chosen

        return []

    except Exception:
        print "Error in GetDatarun():", traceback.format_exc()

    finally:
        if session:
            session.close()


def GetFrozenSet(frozen_set_id, increment=False):
    session = None
    frozen_set = None
    try:
        session = GetConnection()
        frozen_set = session.query(FrozenSet).\
            filter(FrozenSet.id == frozen_set_id).one()
        if increment:
            frozen_set.trained += 1
        session.commit()
        session.expunge_all() # so we can use outside the session

    except Exception:
        print "Error in GetFrozenSet():", traceback.format_exc()

    finally:
        if session:
            session.close()

    return frozen_set

def MarkFrozenSetGriddingDone(frozen_set_id):
    session = None
    try:
        session = GetConnection()
        frozen_set = session.query(FrozenSet).filter(FrozenSet.id == frozen_set_id).one()
        frozen_set.is_gridding_done = 1
        session.commit()
        session.expunge_all() # so we can use outside the session

    except Exception:
        print "Error in GetFrozenSet():", traceback.format_exc()

    finally:
        if session:
            session.close()

def MarkDatarunGriddingDone(datarun_id):
    session = None
    try:
        session = GetConnection()
        datarun = session.query(Datarun).filter(Datarun.id == datarun_id).one()
        datarun.is_gridding_done = 1
        session.commit()
        session.expunge_all() # so we can use outside the session

    except Exception:
        print "Error in GetFrozenSet():", traceback.format_exc()

    finally:
        if session:
            session.close()

def GetFrozenSets(datarun_id):
    """
    Returns all the frozen sets in a given datarun by id.
    """
    session = None
    frozen_sets = []
    try:
        session = GetConnection()
        frozen_sets = session.query(FrozenSet).\
            filter(FrozenSet.datarun_id == datarun_id).all()

    except Exception:
        print "Error in GetFrozenSets():", traceback.format_exc()

    finally:
        if session:
            session.close()

    return frozen_sets

def IsGriddingDoneForDatarun(datarun_id, min_num_errors_to_exclude=0):
    """
    Returns all the frozen sets in a given datarun by id.
    """
    session = None
    is_done = True
    try:
        session = GetConnection()
        frozen_sets = session.query(FrozenSet).filter(FrozenSet.datarun_id == datarun_id).all()
        for frozen_set in frozen_sets:
            if frozen_set.is_gridding_done == 0:
                if((min_num_errors_to_exclude > 0) and (GetNumberOfFrozenSetErrors(frozen_set_id=frozen_set.id) < min_num_errors_to_exclude)):
                    is_done = False
                elif min_num_errors_to_exclude == 0:
                    is_done = False

    except Exception:
        print "Error in GetFrozenSets():", traceback.format_exc()

    finally:
        if session:
            session.close()

    return is_done

def GetIncompletedFrozenSets(datarun_id, min_num_errors_to_exclude=0):
    """
    Returns all the frozen sets in a given datarun by id.
    """
    session = None
    frozen_sets = []
    try:
        session = GetConnection()
        frozen_sets = session.query(FrozenSet).\
            filter(and_(FrozenSet.datarun_id == datarun_id, FrozenSet.is_gridding_done == 0)).all()

        if min_num_errors_to_exclude > 0:
            old_list = frozen_sets
            frozen_sets = []

            for frozen_set in old_list:
                if GetNumberOfFrozenSetErrors(frozen_set_id=frozen_set.id) < min_num_errors_to_exclude:
                    frozen_sets.append(frozen_set)


    except Exception:
        print "Error in GetFrozenSets():", traceback.format_exc()

    finally:
        if session:
            session.close()

    return frozen_sets

def GetNumberOfFrozenSetErrors(frozen_set_id):
    session = None
    learners = []
    try:
        session = GetConnection()
        learners = session.query(Learner). \
            filter(and_(Learner.frozen_set_id == frozen_set_id, Learner.is_error == 1)).all()
    except:
        print "Error in GetLearnersInFrozen(%d):" % frozen_set_id, traceback.format_exc()
    finally:
        if session:
            session.close()
    return len(learners)


def MarkDatarunDone(datarun_id):
    """
    Sets the completed field of the Learner to
    the current datetime for a given datarun_id.
    """
    session = None
    try:
        session = GetConnection()
        datarun = session.query(Datarun).\
            filter(Datarun.id == datarun_id).one()
        datarun.completed = datetime.now()
        session.commit()

    except Exception:
        print "Error in MarkDatarunDone():", traceback.format_exc()

    finally:
        if session:
            session.close()


def GetMaximumY(datarun_id, metric, default=0.0):
    """
    Returns the maximum value of a numeric column by name.
    """
    session = GetConnection()
    maximum = default
    try:
        result = session.query(func.max(getattr(Learner, metric))).one()[0]
        if result:
            maximum = float(result)
    except:
        print "Error in GetMaximumY(%d):" % datarun_id, traceback.format_exc()
    finally:
        session.close()
    return maximum

def GetLearnersInFrozen(frozen_set_id):
    """
    Returns all completed learners in
    """
    session = None
    learners = []
    try:
        session = GetConnection()
        learners = session.query(Learner).\
            filter(Learner.frozen_set_id == frozen_set_id).all()
    except:
        print "Error in GetLearnersInFrozen(%d):" % frozen_set_id, traceback.format_exc()
    finally:
        if session:
            session.close()
    return learners

def GetLearners(datarun_id):
    """
    Returns all learners in datarun.
    """
    session = None
    learners = []
    try:
        session = GetConnection()
        learners = session.query(Learner).filter(Learner.datarun_id == datarun_id).order_by(Learner.started).all()
    except:
        print "Error in GetLearners(%d):" % datarun_id, traceback.format_exc()
    finally:
        if session:
            session.close()
    return learners


def GetLearner(learner_id):
    """
    Returns a specific learner.
    """
    session = None
    learner = []
    try:
        session = GetConnection()
        learner = session.query(Learner).filter(Learner.id == learner_id).all()
    except:
        print "Error in GetLearnersInFrozen(%d):" % learner_id, traceback.format_exc()
    finally:
        if session:
            session.close()

    if len(learner) > 1:
        raise RuntimeError('Multiple learners with the same id!')

    return learner[0]
