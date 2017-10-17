from btb.selection.samples.constants import SELECTION_SAMPLES_GRID
from btb.config import Config
from btb.utilities import EnsureDirectory, ParamsToVectors, VectorToParams,\
                          HashDict, HashString, GetPublicIP
from btb.mapping import Mapping, CreateWrapper
from btb.model import Model
from btb.database import Database

import argparse
import ast
import datetime
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
EnsureDirectory("models")
EnsureDirectory("metrics")
EnsureDirectory("logs")

# name log file after the local hostname
LOG_FILE = "logs/%s.txt" % socket.gethostname()
LOOP_WAIT = 6
SQL_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# csv loading utility function
def read_btb_csv(filepath):
    """
    read a csv and return a numpy array.
    this works from the assumption the data has been preprocessed by btb:
    no headers, numerical data only
    """
    num_cols = len(open(filepath).readline().split(','))
    with open(filepath) as f:
        for i, _ in enumerate(f):
            pass
    num_rows = i + 1

    data = np.zeros((num_rows, num_cols))

    with open(filepath) as f:
        for i, line in enumerate(f):
            for j, cell in enumerate(line.split(',')):
                data[i, j] = float(cell)

    return data


def _log(msg, stdout=True):
    with open(LOG_FILE, "a") as lf:
        lf.write(msg + "\n")
    if stdout:
        print msg


class Worker(object):
    def __init__(self, config, datarun_id=None, save_files=False,
                 choose_randomly=True):
        """
        config: Config object with all the info we need.
        datarun_id: id of datarun to work on, or None. If None, this worker will
            work on whatever incomplete dataruns it finds.
        save_files: if True, save model and metrics files to disk or cloud.
        choose_randomly: if True, choose a random datarun; if False, use the
            first one in id-order.
        """
        self.config = config
        self.db = Database(config)
        self.datarun_id = datarun_id
        self.save_files = save_files
        self.choose_randomly = choose_randomly

    def save_learner_cloud(self, local_model_path, local_metric_path):
        aws_key = self.config.get(Config.AWS, Config.AWS_ACCESS_KEY)
        aws_secret = self.config.get(Config.AWS, Config.AWS_SECRET_KEY)
        aws_folder = self.config.get(Config.AWS, Config.AWS_S3_FOLDER)
        s3_bucket = self.config.get(Config.AWS, Config.AWS_S3_BUCKET)

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
        phash = HashDict(params)
        rhash = HashString(datarun.name)

        # whether to save things to local filesystem
        if self.save_files:
            local_model_path = MakeModelPath("models", phash, rhash,
                                             datarun.description)
            model.save(local_model_path)
            _log("Saving model in: %s" % local_model_path)

            local_metric_path = MakeMetricPath("metrics", phash, rhash,
                                               datarun.description)
            metric_obj = dict(cv=performance['cv_object'],
                              test=performance['test_object'])
            SaveMetric(local_metric_path, object=metric_obj)
            _log("Saving metrics in: %s" % local_model_path)

            # mode: cloud or local?
            mode = self.config.get(Config.MODE, Config.MODE_RUNMODE)

            # if necessary, save model and metrics to Amazon S3 bucket
            if mode == 'cloud':
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
        completed = time.strftime('%Y-%m-%d %H:%M:%S')
        trainables = model.algorithm.performance()['trainable_params']
        fmt_completed = datetime.datetime.strptime(completed, SQL_DATE_FORMAT)
        fmt_started = datetime.datetime.strptime(started, SQL_DATE_FORMAT)
        seconds = (fmt_completed - fmt_started).total_seconds()

        # create learner ORM object, and insert learner into the database
        # TODO: wrap this properly in a with session_context(): or the like
        session = self.db.get_session()
        learner = self.db.Learner(
            datarun_id=datarun.id,
            frozen_set_id=frozen_set.id,
            dataname=datarun.name,
            algorithm=frozen_set.algorithm,
            trainpath=datarun.trainpath,
            testpath=datarun.testpath,
            modelpath=local_model_path,
            params_hash=phash,
            params=params,
            trainable_params=trainables,
            cv_judgment_metric=performance['cv_judgment_metric'],
            cv_judgment_metric_stdev=performance['cv_judgment_metric_stdev'],
            test_judgment_metric=performance['test_judgment_metric'],
            metricpath=local_metric_path,
            started=started,
            completed=datetime.datetime.now(),
            host=GetPublicIP(),
            dimensions=model.algorithm.dimensions,
            frozen_hash=frozen_set.frozen_hash,
            seconds=seconds,
            description=datarun.description)
        session.add(learner)

        # update this session's frozen set entry
        # TODO this should be a method on Database
        frozen_set = session.query(self.db.FrozenSet)\
            .filter(self.db.FrozenSet.id == frozen_set.id).one()
        frozen_set.trained += 1
        frozen_set.rewards += Decimal(performance[datarun.score_target])
        session.commit()

    def insert_error(self, datarun_id, frozen_set, params, error_msg):
        session = None
        try:
            session = self.db.get_session()
            session.autoflush = False
            learner = self.db.Learner(datarun_id=datarun_id,
                                      frozen_set_id=frozen_set.id,
                                      errored=datetime.datetime.now(),
                                      is_error=True,
                                      params=params,
                                      error_msg=error_msg,
                                      algorithm=frozen_set.algorithm)
            session.add(learner)
            session.commit()
            _log("Successfully reported error")

        except Exception:
            _log("insert_error(): Error marking this learner an error..." +
                 traceback.format_exc())
        finally:
            if session:
                session.close()

    def load_data(self, datarun):
        """
        Loads the data from HTTP (if necessary) and then from
        disk into memory.
        """
        # download data if necessary
        basepath = os.path.basename(datarun.local_trainpath)

        if not os.path.isfile(datarun.local_trainpath):
            EnsureDirectory("data/processed/")
            if DownloadFileS3(self.config, datarun.local_trainpath) !=\
                    datarun.local_trainpath:
                raise Exception("Something about train dataset caching is wrong...")

        # load the data into matrix format
        trainX = read_btb_csv(datarun.local_trainpath)
        labelcol = datarun.labelcol
        trainY = trainX[:, labelcol]
        trainX = np.delete(trainX, labelcol, axis=1)

        basepath = os.path.basename(datarun.local_testpath)
        if not os.path.isfile(datarun.local_testpath):
            EnsureDirectory("data/processed/")
            if DownloadFileS3(self.config, datarun.local_testpath) !=\
                    datarun.local_testpath:
                raise Exception("Something about test dataset caching is "
                                "wrong...")

        # load the data into matrix format
        testX = read_btb_csv(datarun.local_testpath)
        labelcol = datarun.labelcol
        testY = testX[:, labelcol]
        testX = np.delete(testX, labelcol, axis=1)

        return trainX, testX, trainY, testY

    def get_frozen_selector(self, datarun):
        frozen_sets = self.db.GetIncompleteFrozenSets(datarun.id,
                                                      errors_to_exclude=20)
        if not frozen_sets:
            if self.db.IsGriddingDoneForDatarun(datarun_id=datarun.id,
                                                errors_to_exclude=20):
                self.db.MarkDatarunGriddingDone(datarun_id=datarun.id)
            _log("No incomplete frozen sets for datarun present in database.")

        # load the class for selecting the frozen set
        _log("Frozen Selection: %s" % datarun.frozen_selection)
        Selector = Mapping.SELECTION_FROZENS_MAP[datarun.frozen_selection]

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
        frozen_sets = self.db.GetIncompleteFrozenSets(datarun.id,
                                                      errors_to_exclude=20)
        if not frozen_sets:
            if self.db.IsGriddingDoneForDatarun(datarun_id=datarun.id,
                                                errors_to_exclude=20):
                self.db.MarkDatarunGriddingDone(datarun_id=datarun.id)
            _log("No incomplete frozen sets for datarun present in database.")
            return None

        # load learners and build scores lists
        # make sure all frozen sets are present in the dict, even ones that
        # don't have any learners. That way the selector can choose frozen sets
        # that haven't been scored yet.
        frozen_set_scores = {fs.id: [] for fs in frozen_sets}
        learners = self.db.GetLearners(datarun.id)
        for l in learners:
            score = getattr(l, datarun.score_target)
            frozen_set_scores[l.frozen_set_id].append(score)

        frozen_set_id = self.frozen_selector.select(frozen_set_scores)
        frozen_set = self.db.GetFrozenSet(frozen_set_id)

        if not frozen_set:
            _log("Invalid frozen set id: %d" % frozen_set_id)
            return None
        return frozen_set

    def select_parameters(self, datarun, frozen_set):
        _log("Sample selection: %s" % datarun.sample_selection)
        Selector = Mapping.SELECTION_SAMPLES_MAP[datarun.sample_selection]

        # Get parameter metadata for this frozen set
        optimizables = frozen_set.optimizables

        # If there aren't any optimizables, only run this frozen set once
        if not len(optimizables):
            _log("No optimizables for frozen set %d" % frozen_set.id)
            self.db.MarkFrozenSetGriddingDone(frozen_set.id)
            return VectorToParams(vector=[], optimizables=optimizables,
                                  frozens=frozen_set.frozens,
                                  constants=frozen_set.constants)

        # Get previously-used parameters
        # every learner should either be completed or have thrown an error
        learners = [x for x in self.db.GetLearnersInFrozen(frozen_set.id)
                    if x.completed]

        # extract parameters and scores as numpy arrays from learners
        X = ParamsToVectors([l.params for l in learners], optimizables)
        y = np.array([float(getattr(l, datarun.score_target))
                      for l in learners])

        # initialize the sampler and propose a new set of parameters
        selector = Selector(optimizables, r_min=datarun.r_min)
        selector.fit(X, y)
        vector = selector.propose()

        # check if gridding is done after the latest proposal
        # TODO this might be wrong -- maybe we shouldn't mark this done until
        # the actual learner has been completed.
        if datarun.sample_selection == SELECTION_SAMPLES_GRID:
            if selector.finished:
                _log("Gridding done for frozen set %d" % frozen_set.id)
                self.db.MarkFrozenSetGriddingDone(frozen_set.id)

        if vector is None:
            _log("No sample selected for frozen set %d" % frozen_set.id)
            return None

        # convert the numpy array of parameters to useable params
        return VectorToParams(vector=vector, optimizables=optimizables,
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
                started = time.strftime('%Y-%m-%d %H:%M:%S')
                datarun = self.db.GetDatarun(datarun_id=self.datarun_id,
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
                frozen_sets = self.db.GetIncompleteFrozenSets(datarun.id,
                                                              errors_to_exclude=20)
                n_completed = sum([f.trained for f in frozen_sets])

                if (budget_type == "learner" and
                        n_completed >= datarun.learner_budget):
                    # this run is done
                    self.db.MarkDatarunDone(datarun.id)
                    endrun = True
                    _log("Learner budget has run out!")

                elif budget_type == "walltime":
                    deadline = datarun.deadline
                    if datetime.datetime.now() > deadline:
                        # this run is done
                        self.db.MarkDatarunDone(datarun.id)
                        endrun = True
                        _log("Walltime budget has run out!")

                # did we end the run?
                elif endrun == True:
                    # marked the run as done successfully
                    _log("This datarun has ended, let's find another")
                    time.sleep(2)   # TODO: why is this needed?
                    continue

                # use the multi-arm bandit to choose which frozen set to use next
                frozen_set = self.select_frozen_set(datarun)
                if frozen_set is None:
                    _log("Sleeping for 10 seconds, then trying again.")
                    time.sleep(10)
                    continue

                # use the configured sample selector to choose a set of
                # parameters within the frozen set
                params = self.select_parameters(datarun, frozen_set)
                if params is None:
                    _log("Frozen set finished. No parameters chosen.")
                    continue

                _log("Chose parameters for algorithm %s" % frozen_set.algorithm)
                _log("Params chosen: %s" % params, False)
                print "Params chosen:"
                for k, v in params.items():
                    print "\t%s = %s" % (k, v)

                # train learner
                params["function"] = frozen_set.algorithm
                wrapper = CreateWrapper(params, datarun.metric)
                trainX, testX, trainY, testY = self.load_data(datarun)
                wrapper.load_data_from_objects(trainX, testX, trainY, testY)
                performance = wrapper.start()

                _log("Judgment metric (%s): %.3f +- %.3f" %
                     (datarun.metric,
                      performance["cv_judgment_metric"],
                      2 * performance["cv_judgment_metric_stdev"]))

                model = Model(wrapper, datarun.wrapper)

                # insert learner into the database
                self.insert_learner(datarun, frozen_set, performance,
                                    params, model, started)

                _log("Best so far: %.3f +- %.3f" %
                     self.db.get_best_so_far(datarun.id,
                                             datarun.score_target))

            except Exception as e:
                msg = traceback.format_exc()
                if datarun and frozen_set:
                    _log("Error in main work loop: datarun=%s" % str(datarun) + msg)
                    self.insert_error(datarun.id, frozen_set, params, msg)
                else:
                    _log("Error in main work loop (no datarun or frozen set):" + msg)

            finally:
                time.sleep(LOOP_WAIT)

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            if total_time is not None and elapsed_time >= total_time:
                _log("Total run time for worker exceeded; exiting.")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add more learners to database')
    parser.add_argument('-c', '--configpath', help='Location of config file',
                        default=os.getenv('BTB_CONFIG_FILE', 'config/btb.cnf'), required=False)
    parser.add_argument('-d', '--datarun-id', help='Only train on datasets with this id', default=None, required=False)
    parser.add_argument('-t', '--time', help='Number of seconds to run worker', default=None, required=False)

    # a little confusingly named (seqorder populates choose_randomly?)
    parser.add_argument('-l', '--seqorder', help='work on datasets in sequential order starting with smallest id number, but still max priority (default = random)',
                        dest='choose_randomly', default=True, action='store_const', const=False)
    parser.add_argument('--no-save', help="don't save models and metrics for later",
                        dest='save_files', default=True, action='store_const', const=False)
    args = parser.parse_args()
    config = Config(args.configpath)

    worker = Worker(config=config, datarun_id=args.datarun_id,
                    choose_randomly=args.choose_randomly,
                    save_files=args.save_files)
    # lets go
    worker.work(total_time=args.time)
