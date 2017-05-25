#!/usr/bin/python
# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func

import traceback
import random, sys
from delphi.utilities import *
from delphi.config import Config
import os
from datetime import datetime
import warnings

configpath = os.getenv('DELPHI_CONFIG_FILE')
if configpath == None:
    warnings.warn('No config file environmental variable, using default config/delphi.cnf')
    configpath = 'config/delphi.cnf'
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
    def confusion(self):
        return Base64ToObject(self.confusion64)

    @confusion.setter
    def confusion(self, value):
        self.confusion64 = value # This attribute was removed to save space in DataHub

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

def GetDatarun(datarun_id=None):
    """
    Among the incomplete dataruns with maximal priority,
    returns one at random.
    """
    # get all incomplete dataruns
    session = None
    dataruns = []
    try:
        session = GetConnection()

        query = session.query(Datarun).\
            filter(Datarun.completed == None)

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
        if candidates:
            chosen = candidates[random.randint(0, len(candidates) - 1)]
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
        learners = session.query(Learner).\
            filter(Learner.datarun_id == datarun_id).all()
    except:
        print "Error in GetLearnersInFrozen(%d):" % frozen_set_id, traceback.format_exc()
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
