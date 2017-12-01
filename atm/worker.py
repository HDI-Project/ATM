#!/usr/bin/python2.7
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
from decimal import Decimal
from collections import defaultdict

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
parser.add_argument('--datarun-id', help='Only train on datasets with this id')
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


class Worker(object):
    def __init__(self, db, datarun_id=None, save_files=False,
                 choose_randomly=True, cloud_mode=False, aws_config=None):
        """
        db: Database object with connection information
        datarun_id: id of datarun to work on, or None. If None, this worker will
            work on whatever incomplete dataruns it finds.
        save_files: if True, save model and metrics files to disk or cloud.
        choose_randomly: if True, choose a random datarun; if False, use the
            first one in id-order.
        cloud_mode: if True, use cloud mode
        aws_config: dictionary of amazon s3 data
        """
        self.db = db
        self.datarun_id = datarun_id
        self.save_files = save_files
        self.choose_randomly = choose_randomly
        self.cloud_mode = cloud_mode
        if cloud_mode:
            self.aws_key = aws_config['access_key']
            self.aws_secret = aws_config['secret_key']
            self.s3_bucket = aws_config['s3_bucket']
            self.s3_folder = aws_config['s3_folder']

    def save_learner_cloud(self, local_model_path, local_metric_path):

        conn = S3Connection(aws_key, aws_secret)
        bucket = conn.get_bucket(s3_bucket)

        if aws_folder and not aws_folder.isspace():
            aws_model_path = os.path.join(aws_folder, local_model_path)
            aws_metric_path = os.path.join(aws_folder, local_metric_path)
        else:
            aws_model_path = local_model_path
            aws_metric_path = local_metric_path

        kmodel = S3Key(bucket)
        kmodel.key = aws_model_path
        kmodel.set_contents_from_filename(local_model_path)
        _log('Uploading model to S3 bucket {} in {}'.format(s3_bucket,
                                                            local_model_path))

        kmodel = S3Key(bucket)
        kmodel.key = aws_metric_path
        kmodel.set_contents_from_filename(local_metric_path)
        _log('Uploading metric to S3 bucket {} in {}'.format(s3_bucket,
                                                             local_metric_path))

        # delete the local copy of the model & metric so that it doesn't
        # fill up the worker instance's hard drive
        _log('Deleting local copy of {}'.format(local_model_path))
        os.remove(local_model_path)
        _log('Deleting local copy of {}'.format(local_metric_path))
        os.remove(local_metric_path)

    def insert_learner(self, datarun, frozen_set, performance, params, model,
                       started):
        """
        Inserts a learner and also updates the frozen_sets table.

        datarun: datarun object for the learner
        frozen_set: frozen set object
        """
        # save model to local filesystem
        phash = hash_dict(params)
        dataset = self.db.get_dataset(datarun.dataset_id)
        rhash = hash_string(dataset.name)

        # whether to save things to local filesystem
        if self.save_files:
            local_model_path = make_model_path("models", phash, rhash,
                                               datarun.description)
            model.save(local_model_path)
            _log("Saving model in: %s" % local_model_path)

            local_metric_path = make_metric_path("metrics", phash, rhash,
                                                 datarun.description)
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
                    self.insert_error(datarun.id, frozen_set, params, msg)
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
            datarun_id=datarun.id,
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

    def insert_error(self, datarun_id, frozen_set, params, error_msg):
        session = None
        try:
            session = self.db.get_session()
            session.autoflush = False
            learner = self.db.Learner(datarun_id=datarun_id,
                                      frozen_set_id=frozen_set.id,
                                      params=params,
                                      status=LearnerStatus.ERRORED,
                                      error_msg=error_msg)

            session.add(learner)
            session.commit()
            _log("Successfully reported error")

        except Exception:
            _log("insert_error(): Error marking this learner an error..." +
                 traceback.format_exc())
        finally:
            if session:
                session.close()

    def load_data(self, dataset):
        """
        Loads the data from HTTP (if necessary) and then from
        disk into memory.
        """
        dw = dataset.wrapper

        # if the data are not present locally, check the S3 bucket detailed in
        # the config for it.
        if not os.path.isfile(dw.train_path):
            ensure_directory(dw.outfolder)
            if download_file_s3(dw.test_path, aws_key=self.aws_key,
                                aws_secret=self.aws_secret,
                                s3_bucket=self.s3_bucket,
                                s3_folder=self.s3_folder) !=\
                    dw.train_path:
                raise Exception("Something about train dataset caching is wrong...")

        # load the data into matrix format
        trainX = read_atm_csv(dw.train_path)
        trainY = trainX[:, dataset.label_column]
        trainX = np.delete(trainX, dataset.label_column, axis=1)

        if not os.path.isfile(dw.test_path):
            ensure_directory(dw.outfolder)
            if download_file_s3(dw.test_path, aws_key=self.aws_key,
                                aws_secret=self.aws_secret,
                                s3_bucket=self.s3_bucket,
                                s3_folder=self.s3_folder) !=\
                    dw.test_path:
                raise Exception("Something about test dataset caching is wrong...")

        # load the data into matrix format
        testX = read_atm_csv(dw.test_path)
        testY = testX[:, dataset.label_column]
        testX = np.delete(testX, dataset.label_column, axis=1)

        return trainX, testX, trainY, testY

    def get_frozen_selector(self, datarun):
        frozen_sets = self.db.get_incomplete_frozen_sets(datarun.id,
                                                      errors_to_exclude=20)
        if not frozen_sets:
            if self.db.is_datarun_gridding_done(datarun_id=datarun.id):
                self.db.mark_datarun_gridding_done(datarun_id=datarun.id)
            _log("No incomplete frozen sets for datarun present in database.")

        # load the class for selecting the frozen set
        # selector will either be a key into SELECTORS_MAP or a path to
        # a file that defines a class called CustomSelector.
        if datarun.selector in Mapping.SELECTORS_MAP:
            Selector = Mapping.SELECTORS_MAP[datarun.selector]
        else:
            mod = imp.load_source('btb.selection.custom', datarun.selector)
            Selector = mod.CustomSelector
        _log("Selector: %s" % Selector)

        # generate the arguments we need
        frozen_set_ids = [s.id for s in frozen_sets]
        fs_by_algorithm = defaultdict(list)
        for s in frozen_sets:
            fs_by_algorithm[s.algorithm].append(s.id)

        # FrozenSelector classes support passing in redundant arguments
        self.frozen_selector = Selector(choices=frozen_set_ids,
                                        k=datarun.k_window,
                                        by_algorithm=dict(fs_by_algorithm))

    def select_frozen_set(self, datarun):
        frozen_sets = self.db.get_incomplete_frozen_sets(datarun.id,
                                                      errors_to_exclude=20)
        if not frozen_sets:
            if self.db.is_datarun_gridding_done(datarun_id=datarun.id):
                self.db.mark_datarun_gridding_done(datarun_id=datarun.id)
            _log("No incomplete frozen sets for datarun present in database.")
            return None

        # load learners and build scores lists
        # make sure all frozen sets are present in the dict, even ones that
        # don't have any learners. That way the selector can choose frozen sets
        # that haven't been scored yet.
        frozen_set_scores = {fs.id: [] for fs in frozen_sets}
        learners = self.db.get_complete_learners(datarun.id)
        for l in learners:
            # ignore frozen sets for which gridding is done
            if l.frozen_set_id not in frozen_set_scores:
                continue
            # the cast to float is necessary because the score is a Decimal;
            # doing Decimal-float arithmetic throws errors later on.
            score = float(getattr(l, datarun.score_target))
            frozen_set_scores[l.frozen_set_id].append(score)

        frozen_set_id = self.frozen_selector.select(frozen_set_scores)
        frozen_set = self.db.get_frozen_set(frozen_set_id)

        if not frozen_set:
            _log("Invalid frozen set id: %d" % frozen_set_id)
            return None
        return frozen_set

    def tune_parameters(self, datarun, frozen_set):
        # tuner will either be a key into TUNERS_MAP or a path to
        # a file that defines a class called CustomTuner.
        if datarun.tuner in Mapping.TUNERS_MAP:
            Tuner = Mapping.TUNERS_MAP[datarun.tuner]
        else:
            mod = imp.load_source('btb.tuning.custom', datarun.tuner)
            Tuner = mod.CustomTuner
        _log("Tuner: %s" % Tuner)

        # Get parameter metadata for this frozen set
        optimizables = frozen_set.optimizables

        # If there aren't any optimizables, only run this frozen set once
        if not len(optimizables):
            _log("No optimizables for frozen set %d" % frozen_set.id)
            self.db.mark_frozen_set_gridding_done(frozen_set.id)
            return vector_to_params(vector=[], optimizables=optimizables,
                                    frozens=frozen_set.frozens,
                                    constants=frozen_set.constants)

        # Get previously-used parameters
        # every learner should either be completed or have thrown an error
        learners = [l for l in self.db.get_learners_in_frozen(frozen_set.id)
                    if l.status == LearnerStatus.COMPLETE]

        # extract parameters and scores as numpy arrays from learners
        X = params_to_vectors([l.params for l in learners], optimizables)
        y = np.array([float(getattr(l, datarun.score_target))
                      for l in learners])

        # initialize the tuner and propose a new set of parameters
        tuner = Tuner(optimizables, datarun.gridding, r_min=datarun.r_min)
        tuner.fit(X, y)
        vector = tuner.propose()

        if vector is None:
            if datarun.gridding:
                _log("Gridding done for frozen set %d" % frozen_set.id)
                self.db.mark_frozen_set_gridding_done(frozen_set.id)
            else:
                _log("No sample selected for frozen set %d" % frozen_set.id)
            return None

        # convert the numpy array of parameters to useable params
        return vector_to_params(vector=vector, optimizables=optimizables,
                                frozens=frozen_set.frozens,
                                constants=frozen_set.constants)

    def work(self, total_time=None):
        start_time = datetime.datetime.now()

        # main loop
        while True:
            datarun, frozen_set, params = None, None, None
            try:
                # choose datarun to work on
                _log("=" * 25)
                started = datetime.datetime.now()
                datarun = self.db.get_datarun(datarun_id=self.datarun_id,
                                              ignore_grid_complete=False,
                                              choose_randomly=self.choose_randomly)

                if datarun is None:
                    # If desired, we can sleep here and wait for a new datarun
                    _log("No datarun present in database, exiting.")
                    sys.exit()

                _log("Datarun: %s" % datarun)

                self.get_frozen_selector(datarun)

                # check if we've exceeded datarun limits
                budget_type = datarun.budget
                endrun = False

                # check to see if we're over the datarun/time budget
                frozen_sets = self.db.get_incomplete_frozen_sets(datarun.id,
                                                              errors_to_exclude=20)

                if budget_type == "learner":
                    n_completed = sum([f.trained for f in frozen_sets])
                    if n_completed >= datarun.budget:
                        endrun = True
                        _log("Learner budget has run out!")

                elif budget_type == "walltime":
                    deadline = datarun.deadline
                    if datetime.datetime.now() > deadline:
                        endrun = True
                        _log("Walltime budget has run out!")

                if endrun == True:
                    # marked the run as done successfully
                    self.db.mark_datarun_done(datarun.id)
                    _log("This datarun has ended.")
                    time.sleep(2)
                    continue

                # use the multi-arm bandit to choose which frozen set to use next
                frozen_set = self.select_frozen_set(datarun)
                if frozen_set is None:
                    _log("No frozen set found. Sleeping for 10 seconds, then "
                         "trying again.")
                    time.sleep(10)
                    continue

                # use the configured sample selector to choose a set of
                # parameters within the frozen set
                params = self.tune_parameters(datarun, frozen_set)
                if params is None:
                    _log("No parameters chosen: frozen set %d is finished." %
                         frozen_set.id)
                    continue

                _log("Chose parameters for algorithm %s:" % frozen_set.algorithm)
                for k, v in params.items():
                    _log("\t%s = %s" % (k, v))

                params["function"] = frozen_set.algorithm

                _log("Testing learner...")
                wrapper = create_wrapper(params, datarun.metric)
                dataset = self.db.get_dataset(datarun.dataset_id)
                trainX, testX, trainY, testY = self.load_data(dataset)
                wrapper.load_data_from_objects(trainX, testX, trainY, testY)
                performance = wrapper.start()

                _log("Judgment metric (%s): %.3f +- %.3f" %
                     (datarun.metric,
                      performance["cv_judgment_metric"],
                      2 * performance["cv_judgment_metric_stdev"]))

                _log("Saving learner...")
                model = Model(algorithm=wrapper, data=dataset.wrapper)

                # insert learner into the database
                self.insert_learner(datarun, frozen_set, performance,
                                    params, model, started)

                best_y = self.db.get_maximum_y(datarun.id, datarun.score_target)
                _log("Best so far: %.3f" % (best_y or 0))
                #_log("Best so far: %.3f +- %.3f" %
                     #self.db.get_best_so_far(datarun.id,
                                             #datarun.score_target))

            except Exception as e:
                msg = traceback.format_exc()
                if datarun and frozen_set:
                    _log("Error in main work loop: datarun=%s" % str(datarun) + msg)
                    self.insert_error(datarun.id, frozen_set, params, msg)
                else:
                    _log("Error in main work loop (no datarun or frozen set):" + msg)

            _log("Learner finished. Sleeping %d seconds." % LOOP_WAIT)
            time.sleep(LOOP_WAIT)

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            if total_time is not None and elapsed_time >= total_time:
                _log("Total run time for worker exceeded; exiting.")
                break


if __name__ == '__main__':
    args = parser.parse_args()

    sql_config = {}
    sql_options = ['dialect', 'database', 'username', 'password', 'host',
                   'port', 'query']
    if args.sql_config:
        with open(args.sql_config) as f:
            sql_config = yaml.load(f)
    else:
        for opt in sql_options:
            sql_config[opt] = getattr(args, 'sql_' + opt)
    db = Database(**sql_config)

    aws_config = None
    aws_options = ['access_key', 'secret_key', 's3_bucket', 's3_folder']
    if args.cloud_mode:
        if args.aws_config:
            with open(args.aws_config) as f:
                aws_config = yaml.load(f)
        else:
            for opt in aws_options:
                aws_config[opt] = getattr(args, 'aws_' + opt)

    worker = Worker(db=db,
                    datarun_id=args.datarun_id,
                    choose_randomly=args.choose_randomly,
                    save_files=args.save_files,
                    cloud_mode=args.cloud_mode,
                    aws_config=aws_config)
    # lets go
    worker.work(total_time=args.time)
