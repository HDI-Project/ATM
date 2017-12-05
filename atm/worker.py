#!/usr/bin/python2.7
from atm.config import *
from atm.constants import *
from atm.utilities import *
from atm.mapping import Mapping, create_wrapper
from atm.model import Model
from atm.database import Database, LearnerStatus
from btb.tuning.constants import Tuners

import argparse
import ast
import datetime
import imp
import os
import pdb
import random
import socket
import sys
import time
import traceback
import warnings
from collections import defaultdict
from decimal import Decimal
from operator import attrgetter

import numpy as np
import pandas as pd
from boto.s3.connection import S3Connection, Key as S3Key

# shhh
warnings.filterwarnings("ignore")

# for garrays
os.environ["GNUMPY_IMPLICIT_CONVERSION"] = "allow"

# get the file system in order
# make sure we have directories where we need them
ensure_directory("models")
ensure_directory("metrics")
ensure_directory("logs")

# name log file after the local hostname
LOG_FILE = "logs/%s.txt" % socket.gethostname()
# how long to wait between training learners (or attempting to)
LOOP_WAIT = 1.

parser = argparse.ArgumentParser(description='Add more learners to database')

##  Config files  #############################################################
###############################################################################
parser.add_argument('--sql-config', help='path to yaml SQL config file')
parser.add_argument('--aws-config', help='path to yaml AWS config file')


##  Database arguments  ########################################################
################################################################################
# All of these arguments must start with --sql-, and must correspond to
# keys present in the SQL config example file.
parser.add_argument('--sql-dialect', choices=SQL_DIALECTS,
                    default=Defaults.SQL_DIALECT, help='Dialect of SQL to use')
parser.add_argument('--sql-database', default=Defaults.DATABASE,
                    help='Name of, or path to, SQL database')
parser.add_argument('--sql-username', help='Username for SQL database')
parser.add_argument('--sql-password', help='Password for SQL database')
parser.add_argument('--sql-host', help='Hostname for database machine')
parser.add_argument('--sql-port', help='Port used to connect to database')
parser.add_argument('--sql-query', help='Specify extra login details')


##  AWS arguments  #############################################################
################################################################################
parser.add_argument('--aws-access-key', help='API access key for AWS')
parser.add_argument('--aws-secret-key', help='API secret key for AWS')
parser.add_argument('--aws-s3-bucket', help='Amazon S3 bucket for data storage')
parser.add_argument('--aws-s3-folder', help='Optional S3 folder')


##  Worker arguments  ##########################################################
################################################################################
parser.add_argument('--cloud-mode', action='store_true', default=False,
                    help='Whether to run this worker in cloud mode')
parser.add_argument('--dataruns', help='Only train on dataruns with these ids',
                    nargs='+')
parser.add_argument('--time', help='Number of seconds to run worker', type=int)
parser.add_argument('--choose-randomly', action='store_true',
                    help='Choose dataruns to work on randomly (default = '
                    'sequential order)')
parser.add_argument('--no-save', dest='save_files', default=True,
                    action='store_const', const=False,
                    help="don't save models and metrics for later")


def _log(msg, stdout=True):
    with open(LOG_FILE, "a") as lf:
        lf.write(msg + "\n")
    if stdout:
        print msg


# Exception thrown when something goes wrong for the worker, but the worker
# handles the error.
class LearnerError(Exception):
    pass


class Worker(object):
    def __init__(self, database, datarun, save_files=False, cloud_mode=False,
                 aws_config=None):
        """
        database: Database object with connection information
        datarun: Datarun ORM object to work on.
        save_files: if True, save model and metrics files to disk or cloud.
        cloud_mode: if True, save learners to the cloud
        aws_config: S3Config object with amazon s3 connection info
        """
        self.db = database
        self.datarun = datarun
        self.save_files = save_files
        self.cloud_mode = cloud_mode
        self.aws_config = aws_config

        # load the Dataset from the database
        self.dataset = self.db.get_dataset(self.datarun.dataset_id)

        # load the Selector and Tuner classes specified by our datarun
        self.load_selector()
        self.load_tuner()

    def load_selector(self):
        # load the class for selecting the frozen set
        # selector will either be a key into SELECTORS_MAP or a path to
        # a file that defines a class called CustomSelector.
        if self.datarun.selector in Mapping.SELECTORS_MAP:
            Selector = Mapping.SELECTORS_MAP[self.datarun.selector]
        else:
            mod = imp.load_source('btb.selection.custom', self.datarun.selector)
            Selector = mod.CustomSelector
        _log("Selector: %s" % Selector)

        # generate the arguments we need to initialize the selector
        frozen_sets = self.db.get_incomplete_frozen_sets(self.datarun.id)
        fs_by_algorithm = defaultdict(list)
        for s in frozen_sets:
            fs_by_algorithm[s.algorithm].append(s.id)
        frozen_set_ids = [s.id for s in frozen_sets]

        # Selector classes support passing in redundant arguments
        self.selector = Selector(choices=frozen_set_ids,
                                 k=self.datarun.k_window,
                                 by_algorithm=dict(fs_by_algorithm))

    def load_tuner(self):
        # tuner will either be a key into TUNERS_MAP or a path to
        # a file that defines a class called CustomTuner.
        if self.datarun.tuner in Mapping.TUNERS_MAP:
            self.Tuner = Mapping.TUNERS_MAP[self.datarun.tuner]
        else:
            mod = imp.load_source('btb.tuning.custom', self.datarun.tuner)
            self.Tuner = mod.CustomTuner
        _log("Tuner: %s" % self.Tuner)

    def load_data(self):
        """
        Downloads a set of train/test data from AWS (if necessary) and then
        loads it from disk into memory.

        Returns: train/test data in the structures consumed by
            wrapper.load_data_from_objects().
            i.e. (trainX, testX, trainY, testY)
        """
        dw = self.dataset.wrapper

        # if the data are not present locally, check the S3 bucket detailed in
        # the config for it.
        if not os.path.isfile(dw.train_path_out):
            ensure_directory(dw.outfolder)
            if download_file_s3(dw.train_path_out, aws_key=self.aws_config.access_key,
                                aws_secret=self.aws_config.access_key,
                                s3_bucket=self.aws_config.s3_bucket,
                                s3_folder=self.aws_config.s3_folder) != dw.train_path_out:
                raise Exception("Something about train dataset caching is wrong...")

        # load the data into matrix format
        trainX = read_atm_csv(dw.train_path_out)
        trainY = trainX[:, self.dataset.label_column]
        trainX = np.delete(trainX, self.dataset.label_column, axis=1)

        if not os.path.isfile(dw.test_path_out):
            ensure_directory(dw.outfolder)
            if download_file_s3(dw.test_path_out, aws_key=self.aws_key,
                                aws_secret=self.aws_secret,
                                s3_bucket=self.s3_bucket,
                                s3_folder=self.s3_folder) != dw.test_path_out:
                raise Exception("Something about test dataset caching is wrong...")

        # load the data into matrix format
        testX = read_atm_csv(dw.test_path_out)
        testY = testX[:, self.dataset.label_column]
        testX = np.delete(testX, self.dataset.label_column, axis=1)

        return trainX, testX, trainY, testY

    def save_learner_cloud(self, local_model_path, local_metric_path):
        """
        Save a learner to the S3 bucket supplied by aws_config. Saves a
        serialized representaion of the model as well as a detailed set
        of metrics.

        local_model_path: path to serialized model in the local file system
        local_metric_path: path to serialized metrics in the local file system
        """
        conn = S3Connection(aws_key, aws_secret)
        bucket = conn.get_bucket(s3_bucket)

        if aws_folder:
            aws_model_path = os.path.join(aws_folder, local_model_path)
            aws_metric_path = os.path.join(aws_folder, local_metric_path)
        else:
            aws_model_path = local_model_path
            aws_metric_path = local_metric_path

        kmodel = S3Key(bucket)
        kmodel.key = aws_model_path
        kmodel.set_contents_from_filename(local_model_path)
        _log('Uploading model at %s to S3 bucket %s' % (s3_bucket,
                                                        local_model_path))

        kmodel = S3Key(bucket)
        kmodel.key = aws_metric_path
        kmodel.set_contents_from_filename(local_metric_path)
        _log('Uploading metrics at %s to S3 bucket %s' % (s3_bucket,
                                                          local_metric_path))

        # delete the local copy of the model & metrics so that they don't fill
        # up the worker instance's hard drive
        _log('Deleting local copies of %s and %s' % (local_model_path,
                                                     local_metric_path))
        os.remove(local_model_path)
        os.remove(local_metric_path)

    def insert_learner(self, frozen_set, performance, params, model, started):
        """
        Inserts a learner and also updates the frozen_sets table.

        frozen_set: frozen set that was used by the learner
        performance: dictionary containing detailed performance data, as
            generated by the Wrapper object that actually tests the learner.
        params: full set of hyperparameters used to generate this model
        model: Model object containing a serializable representation of the
            final model generated by this learner
        started: DateTime at which work was started on this learner.
        """
        # save model to local filesystem
        phash = hash_dict(params)
        rhash = hash_string(self.dataset.name)

        # whether to save model and performance data to the filesystem
        if self.save_files:
            local_model_path = make_model_path("models", phash, rhash,
                                               self.datarun.description)
            model.save(local_model_path)
            _log("Saving model in: %s" % local_model_path)

            local_metric_path = make_metric_path("metrics", phash, rhash,
                                                 self.datarun.description)
            metric_obj = dict(cv=performance['cv_object'],
                              test=performance['test_object'])
            save_metric(local_metric_path, object=metric_obj)
            _log("Saving metrics in: %s" % local_model_path)

            # if necessary, save model and metrics to Amazon S3 bucket
            if self.cloud_mode:
                try:
                    self.save_learner_cloud(local_model_path, local_metric_path)
                except Exception:
                    msg = traceback.format_exc()
                    _log("Error in save_learner_cloud()")
                    self.insert_error(frozen_set, params, msg)
        else:
            local_model_path = None
            local_metric_path = None

        # compile fields
        trainables = model.algorithm.performance()['trainable_params']
        completed = datetime.datetime.now()
        seconds = (completed - started).total_seconds()

        # create learner ORM object, and insert learner into the database
        # TODO: wrap this properly in a 'with session_context():' or make it a
        # method on Database
        session = self.db.get_session()
        learner = self.db.Learner(
            frozen_set_id=frozen_set.id,
            datarun_id=self.datarun.id,
            model_path=local_model_path,
            metric_path=local_metric_path,
            host=get_public_ip(),
            params=params,
            trainable_params=trainables,
            dimensions=model.algorithm.dimensions,
            cv_judgment_metric=performance['cv_judgment_metric'],
            cv_judgment_metric_stdev=performance['cv_judgment_metric_stdev'],
            test_judgment_metric=performance['test_judgment_metric'],
            started=started,
            completed=completed,
            status=LearnerStatus.COMPLETE)
        session.add(learner)

        # update this session's frozen set entry
        frozen_set = session.query(self.db.FrozenSet).get(frozen_set.id)
        frozen_set.trained += 1
        session.commit()
        _log('Saved as learner %d.' % learner.id)

    def insert_error(self, frozen_set, params, error_msg):
        session = None
        try:
            session = self.db.get_session()
            session.autoflush = False
            learner = self.db.Learner(datarun_id=self.datarun.id,
                                      frozen_set_id=frozen_set.id,
                                      params=params,
                                      status=LearnerStatus.ERRORED,
                                      error_msg=error_msg)

            session.add(learner)
            session.commit()
            _log("Successfully reported error")

        except Exception:
            _log("insert_error(): Error saving learner error in db...")
            _log(traceback.format_exc())
        finally:
            if session:
                session.close()

    def select_frozen_set(self):
        """
        Use the frozen set selection method specified by our datarun to choose a
        frozen set of hyperparameters from the ModelHub. Only consider frozen
        sets for which gridding is not complete.
        """
        frozen_sets = self.db.get_incomplete_frozen_sets(self.datarun.id)

        # load learners and build scores lists
        # make sure all frozen sets are present in the dict, even ones that
        # don't have any learners. That way the selector can choose frozen sets
        # that haven't been scored yet.
        frozen_set_scores = {fs.id: [] for fs in frozen_sets}
        learners = self.db.get_complete_learners(self.datarun.id)
        for l in learners:
            # ignore frozen sets for which gridding is done
            if l.frozen_set_id not in frozen_set_scores:
                continue

            # the cast to float is necessary because the score is a Decimal;
            # doing Decimal-float arithmetic throws errors later on.
            score = float(getattr(l, self.datarun.score_target))
            frozen_set_scores[l.frozen_set_id].append(score)

        frozen_set_id = self.selector.select(frozen_set_scores)
        return self.db.get_frozen_set(frozen_set_id)

    def tune_parameters(self, frozen_set):
        """
        Use the hyperparameter tuning method specified by our datarun to choose
        a set of hyperparameters from the potential space.
        """
        # Get parameter metadata for this frozen set
        optimizables = frozen_set.optimizables

        # If there aren't any optimizables, we're done. Return the vector of
        # values in the frozen set and mark the set as finished.
        if not len(optimizables):
            _log("No optimizables for frozen set %d" % frozen_set.id)
            self.db.mark_frozen_set_gridding_done(frozen_set.id)
            return vector_to_params(vector=[], optimizables=optimizables,
                                    frozens=frozen_set.frozens,
                                    constants=frozen_set.constants)

        # Get previously-used parameters: every learner should either be
        # completed or have thrown an error
        learners = [l for l in self.db.get_learners_in_frozen(frozen_set.id)
                    if l.status == LearnerStatus.COMPLETE]

        # Extract parameters and scores as numpy arrays from learners
        X = params_to_vectors([l.params for l in learners], optimizables)
        y = np.array([float(getattr(l, self.datarun.score_target))
                      for l in learners])

        # Initialize the tuner and propose a new set of parameters
        # this has to be initialized with information from the frozen set, so we
        # need to do it fresh for each learner (not in load_tuner)
        tuner = self.Tuner(optimizables,
                           gridding=self.datarun.gridding,
                           r_min=self.datarun.r_min)
        tuner.fit(X, y)
        vector = tuner.propose()

        if vector is None and self.datarun.gridding:
            _log("Gridding done for frozen set %d" % frozen_set.id)
            self.db.mark_frozen_set_gridding_done(frozen_set.id)
            return None

        # Convert the numpy array of parameters to a form that can be
        # interpreted by ATM, then return.
        return vector_to_params(vector=vector, optimizables=optimizables,
                                frozens=frozen_set.frozens,
                                constants=frozen_set.constants)

    def is_datarun_finished(self):
        """
        Check to see whether the datarun is finished. This could be due to the
        budget being exhausted or due to hyperparameter gridding being done.
        """
        frozen_sets = self.db.get_incomplete_frozen_sets(self.datarun.id)
        if not frozen_sets:
            _log("No incomplete frozen sets for datarun present in database.")
            return True

        if self.datarun.budget_type == "learner":
            n_completed = sum([f.trained for f in frozen_sets])
            if n_completed >= self.datarun.budget:
                _log("Learner budget has run out!")
                return True

        elif self.datarun.budget_type == "walltime":
            deadline = self.datarun.deadline
            if datetime.datetime.now() > deadline:
                _log("Walltime budget has run out!")
                return True

        return False

    def test_learner(self, params):
        """
        Given a set of fully-qualified hyperparameters, create and test a
        model.
        Returns: Model object and performance dictionary
        """
        wrapper = create_wrapper(params, self.datarun.metric)
        wrapper.load_data_from_objects(*self.load_data())
        performance = wrapper.start()

        old_best = self.db.get_best_so_far(self.datarun.id,
                                           self.datarun.score_target)
        new_val = performance["cv_judgment_metric"]
        new_err = 2 * performance["cv_judgment_metric_stdev"]

        _log("Judgment metric (%s): %.3f +- %.3f" %
             (self.datarun.metric, new_val, new_err))

        if old_best[0] is not None:
            if (new_val - new_err) > (old_best[1] - old_best[2]):
                _log("New best score! Previous best (learner %s): %.3f +- %.3f" %
                     old_best)
            else:
                _log("Best so far (learner %s): %.3f +- %.3f" % old_best)

        model = Model(algorithm=wrapper, data=self.dataset.wrapper)
        return model, performance

    def run_learner(self):
        """
        Choose hyperparameters, then use them to test and save a Learner.
        """
        # check to see if our work is done
        if self.is_datarun_finished():
            # marked the run as done successfully
            self.db.mark_datarun_complete(self.datarun.id)
            _log("This datarun has ended. Returning...")
            return

        learner_start = datetime.datetime.now()
        try:
            _log("Choosing hyperparameters...")
            # use the multi-arm bandit to choose which frozen set to use next
            frozen_set = self.select_frozen_set()
            # use our tuner to choose a set of parameters for the frozen set
            params = self.tune_parameters(frozen_set)
        except Exception as e:
            _log("Error choosing hyperparameters: datarun=%s" % str(self.datarun))
            _log(traceback.format_exc())
            raise LearnerError

        if params is None:
            _log("No parameters chosen: frozen set %d is finished." %
                 frozen_set.id)
            return

        _log("Chose parameters for algorithm %s:" % frozen_set.algorithm)
        for k, v in params.items():
            _log("\t%s = %s" % (k, v))

        # TODO: this doesn't belong here
        params["function"] = frozen_set.algorithm

        try:
            _log("Testing learner...")
            model, performance = self.test_learner(params)
            _log("Saving learner...")
            self.insert_learner(frozen_set, performance, params, model,
                                learner_start)
        except Exception as e:
            msg = traceback.format_exc()
            _log("Error testing learner: datarun=%s" % str(self.datarun))
            _log(msg)
            self.insert_error(frozen_set, params, msg)
            raise LearnerError()


def work(db, datarun_ids=None, save_files=False, choose_randomly=True,
         cloud_mode=False, aws_config=None, total_time=None):
    """
    Check the ModelHub database for unfinished dataruns, and spawn workers to
    work on them as they are added. This process will continue to run until it
    exceeds total_time or is broken with ctrl-C.
    """
    start_time = datetime.datetime.now()

    # main loop
    while True:
        # get all pending and running dataruns, or all pending/running dataruns
        # from the list we were given
        dataruns = db.get_dataruns(include_ids=datarun_ids)
        if dataruns is None or not len(dataruns):
            _log("No dataruns found. Sleeping %d seconds and trying again." %
                 LOOP_WAIT)
            time.sleep(LOOP_WAIT)
            continue

        max_priority = max([r.priority for r in dataruns])
        priority_runs = [r for r in dataruns if r.priority == max_priority]

        # either choose a run randomly, or take the run with the lowest ID
        if choose_randomly:
            run = random.choice(priority_runs)
        else:
            run = sorted(dataruns, key=attrgetter('id'))[0]

        # say we've started working on this datarun, if we haven't already
        db.mark_datarun_running(run.id)

        _log('=' * 25)
        _log('Computing on datarun %d' % run.id)
        # actual work happens here
        worker = Worker(db, run, save_files=save_files,
                        cloud_mode=cloud_mode, aws_config=aws_config)
        try:
            worker.run_learner()
        except LearnerError as e:
            # the exception has already been handled; just wait a sec so we
            # don't go out of control reporting errors
            _log("Something went wrong. Sleeping %d seconds." % LOOP_WAIT)
            time.sleep(LOOP_WAIT)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        if total_time is not None and elapsed_time >= total_time:
            _log("Total run time for worker exceeded; exiting.")
            break


if __name__ == '__main__':
    args = parser.parse_args()

    sql_config, aws_config, _ = load_config(sql_path=args.sql_config,
                                            aws_path=args.aws_config,
                                            args=args)
    db = Database(**vars(sql_config))

    # lets go
    work(db=db, datarun_ids=args.dataruns,
         choose_randomly=args.choose_randomly,
         save_files=args.save_files,
         cloud_mode=args.cloud_mode,
         aws_config=aws_config,
         total_time=args.time)
