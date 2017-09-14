from atm.selection.samples import SELECTION_SAMPLES_UNIFORM, SELECTION_SAMPLES_GP, SELECTION_SAMPLES_GRID
from atm.selection.samples import SELECTION_SAMPLES_GP_EI, SELECTION_SAMPLES_GP_EI_TIME, SELECTION_SAMPLES_GP_EI_VEL
from atm.selection.frozens import SELECTION_FROZENS_UNIFORM, SELECTION_FROZENS_UCB1
from atm.config import Config
from atm.database import *
from atm.utilities import *
from atm.mapping import Mapping, CreateWrapper
from atm.model import Model
import atm.database as db

import datetime
import pandas as pd
from decimal import Decimal
import traceback
import random
import sys
import time
import ast
import argparse
import os

from boto.s3.connection import S3Connection, Key as S3Key

import warnings

warnings.filterwarnings("ignore")

# for garrays
os.environ["GNUMPY_IMPLICIT_CONVERSION"] = "allow"

# get the file system in order
# make sure we have directories where we need them
EnsureDirectory("models")
EnsureDirectory("metrics")
EnsureDirectory("logs")

# open or create our log file. TODO: why the random number?
hostname = GetPublicIP() or random.randint(1, 1e12)
logfile = "logs/%s.txt" % hostname

# misc constant definitions
loop_wait = 6
SQL_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _log(msg, stdout=True):
    with open(logfile, "a") as lf:
        lf.write(msg + "\n")
    if stdout:
        print msg


def InsertLearner(datarun, frozen_set, performance, params, model, started, config):
    """
    Inserts a learner and also updates the frozen_sets table.

    datarun: datarun object for the learner
    frozen_set: frozen set object
    """
    # mode: cloud or local?
    mode = config.get(Config.MODE, Config.MODE_RUNMODE)
    session = None
    total = 5
    tries = total

    while tries:
        try:
            # save model to local filesystem
            phash = HashDict(params)
            rhash = HashString(datarun.name)
            local_model_path = MakeModelPath("models", phash, rhash, datarun.description)
            model.save(local_model_path)
            _log("Saving model in: %s" % local_model_path)

            local_metric_path = MakeMetricPath("metrics", phash, rhash, datarun.description)
            metric_obj = dict(cv=performance['cv_object'], test=performance['test_object'])
            SaveMetric(local_metric_path, object=metric_obj)
            _log("Saving metric in: %s" % local_model_path)

            # save model to Amazon S3 bucket
            if mode == 'cloud':
                aws_key = config.get(Config.AWS, Config.AWS_ACCESS_KEY)
                aws_secret = config.get(Config.AWS, Config.AWS_SECRET_KEY)
                conn = S3Connection(aws_key, aws_secret)
                s3_bucket = config.get(Config.AWS, Config.AWS_S3_BUCKET)
                bucket = conn.get_bucket(s3_bucket)


                if config.get(Config.AWS, Config.AWS_S3_FOLDER) and not config.get(Config.AWS, Config.AWS_S3_FOLDER).isspace():
                    aws_model_path = os.path.join(config.get(Config.AWS, Config.AWS_S3_FOLDER), local_model_path)
                    aws_metric_path = os.path.join(config.get(Config.AWS, Config.AWS_S3_FOLDER), local_metric_path)
                else:
                    aws_model_path = local_model_path
                    aws_metric_path = local_metric_path

                kmodel = S3Key(bucket)
                kmodel.key = aws_model_path
                kmodel.set_contents_from_filename(local_model_path)
                _log('Uploading model to S3 bucket {} in {}'.format(s3_bucket, local_model_path))

                kmodel = S3Key(bucket)
                kmodel.key = aws_metric_path
                kmodel.set_contents_from_filename(local_metric_path)
                _log('Uploading metric to S3 bucket {} in {}'.format(s3_bucket, local_metric_path))

                # delete the local copy of the model & metric so that it doesn't fill up the worker instance's hard drive
                os.remove(local_model_path)
                _log('Deleting local copy of {}'.format(local_model_path))
                os.remove(local_metric_path)
                _log('Deleting local copy of {}'.format(local_metric_path))


            # compile fields
            completed = time.strftime('%Y-%m-%d %H:%M:%S')
            trainables = model.algorithm.performance()['trainable_params']
            fmt_completed = datetime.datetime.strptime(completed, SQL_DATE_FORMAT)
            fmt_started = datetime.datetime.strptime(started, SQL_DATE_FORMAT)
            seconds = (fmt_completed - fmt_started).total_seconds()

            # connect and insert, update
            session = GetConnection()

            # create learner ORM object, and insert into the database
            learner = db.Learner(datarun_id=datarun.id, frozen_set_id=frozen_set.id, dataname=datarun.name,
                                 algorithm=frozen_set.algorithm, trainpath=datarun.trainpath, testpath=datarun.testpath,
                                 modelpath=local_model_path, params_hash=phash, params=params, trainable_params=trainables,
                                 cv_judgment_metric=performance['cv_judgment_metric'],
                                 cv_judgment_metric_stdev=performance['cv_judgment_metric_stdev'],
                                 test_judgment_metric=performance['test_judgment_metric'],
                                 metricpath=local_metric_path, started=started, completed=datetime.datetime.now(),
                                 host=GetPublicIP(), dimensions=model.algorithm.dimensions,
                                 frozen_hash=frozen_set.frozen_hash, seconds=seconds, description=datarun.description)
            session.add(learner)

            # update this session's frozen set entry
            frozen_set = session.query(db.FrozenSet).filter(db.FrozenSet.id == frozen_set.id).one()
            # frozen_set.trained += 1 # we mark before now
            frozen_set.rewards += Decimal(performance[datarun.metric])
            session.commit()
            break

        except Exception:
            msg = traceback.format_exc()
            _log("Error in InsertLearner():, try=%d" % (total - tries) + msg)
            if tries:
                time.sleep((total - tries) ** 2)
                tries -= 1
                continue
            InsertError(datarun.id, frozen_set, params, msg)

        finally:
            if session:
                session.close()


def InsertError(datarun_id, frozen_set, params, error_msg):
    session = None
    try:
        session = GetConnection()
        session.autoflush = False
        learner = db.Learner(datarun_id=datarun_id, frozen_set_id=frozen_set.id,
                             errored=datetime.datetime.now(), is_error=True, params=params,
                             error_msg=error_msg, algorithm=frozen_set.algorithm)
        session.add(learner)
        session.commit()
        _log("Successfully reported error")

    except Exception:
        _log("InsertError(): Error marking this learner an error..." + traceback.format_exc())
    finally:
        if session:
            session.close()

def get_atm_csv_num_lines(filepath):
    with open(filepath) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def get_atm_csv_num_cols(filepath):
    line = open(filepath).readline()
    return len(line.split(','))

# this works from the assumption the data has been preprocessed by atm:
# no headers, numerical data only
def read_atm_csv(filepath):
    num_rows = get_atm_csv_num_lines(filepath)
    num_cols = get_atm_csv_num_cols(filepath)

    data = np.zeros((num_rows, num_cols))

    with open(filepath) as f:
        for i, line in enumerate(f):
            for j, cell in enumerate(line.split(',')):
                data[i, j] = float(cell)

    return data

def LoadData(datarun, config):
    """
    Loads the data from HTTP (if necessary) and then from
    disk into memory.
    """
    # download data if necessary
    basepath = os.path.basename(datarun.local_trainpath)

    if not os.path.isfile(datarun.local_trainpath):
        EnsureDirectory("data/processed/")
        if not DownloadFileS3(config, datarun.local_trainpath ) == datarun.local_trainpath:
            raise Exception("Something about train dataset caching is wrong...")

    # load the data into matrix format
    trainX = read_atm_csv(datarun.local_trainpath)
    labelcol = datarun.labelcol
    trainY = trainX[:, labelcol]
    trainX = np.delete(trainX, labelcol, axis=1)

    basepath = os.path.basename(datarun.local_testpath)
    if not os.path.isfile(datarun.local_testpath):
        EnsureDirectory("data/processed/")
        if not DownloadFileS3(config, datarun.local_testpath) == datarun.local_testpath:
            raise Exception("Something about test dataset caching is wrong...")

    # load the data into matrix format
    testX = read_atm_csv(datarun.local_testpath)
    labelcol = datarun.labelcol
    testY = testX[:, labelcol]
    testX = np.delete(testX, labelcol, axis=1)

    return trainX, testX, trainY, testY


def work(config, datarun_id=None, total_time=None, choose_randomly=True):
    start_time = datetime.datetime.now()
    num_no_dataruns = 0

    # TODO: load these from database
    best_perf, best_err = 0, 0

    # call database method to define ORM objects in the db module
    define_tables(config)

    # main loop
    while True:
        datarun, frozen_set, params = None, None, None
        try:
            # choose datarun to work on
            _log("=" * 25)
            started = time.strftime('%Y-%m-%d %H:%M:%S')
            datarun = GetDatarun(datarun_id=datarun_id, ignore_grid_complete=False, chose_randomly=choose_randomly)
            if not datarun:
                if num_no_dataruns > 10:
                    sys.exit()
                _log("No datarun present in database, will wait and try again...")
                num_no_dataruns += 1
                time.sleep(10)
                continue

            # choose frozen set
            _log("Datarun: %s" % datarun)
            frozen_sets = GetIncompletedFrozenSets(datarun.id, min_num_errors_to_exclude=20)
            if not frozen_sets:
                if IsGriddingDoneForDatarun(datarun_id=datarun.id, min_num_errors_to_exclude=20):
                    MarkDatarunGriddingDone(datarun_id=datarun.id)
                _log("No incomplete frozen sets for datarun present in database, will wait and try again...")
                time.sleep(10)
                continue
            ncompleted = sum([f.trained for f in frozen_sets])

            # check if we've exceeded datarun limits
            budget_type = datarun.budget
            endrun = False

            if budget_type == "learner" and ncompleted >= datarun.learner_budget:
                # this run is done
                MarkDatarunDone(datarun.id)
                endrun = True
                _log("Learner budget has run out!")

            elif budget_type == "walltime":
                deadline = datarun.deadline
                if datetime.datetime.now() > deadline:
                    # this run is done
                    MarkDatarunDone(datarun.id)
                    endrun = True
                    _log("Walltime budget has run out!")

            # did we end the run?
            elif endrun == True:
                # marked the run as done successfully
                _log("This datarun has ended, let's find another")
                time.sleep(2)
                continue

            # otherwise select a frozen set from this run
            _log("Frozen Selection: %s" % datarun.frozen_selection)
            fclass = Mapping.SELECTION_FROZENS_MAP[datarun.frozen_selection]
            best_y = GetMaximumY(datarun.id, datarun.metric, default=0.0)
            fselector = fclass(frozen_sets=frozen_sets, best_y=best_y, k=datarun.k_window, metric=datarun.metric)
            frozen_set_id = fselector.select()
            if not frozen_set_id > 0:
                _log("Invalid frozen set id! %d" % frozen_set_id)

            frozen_set = GetFrozenSet(frozen_set_id, increment=True)
            if not frozen_set:
                _log("Invalid frozen set! %s" % frozen_set)
            _log("Frozen set: %d" % frozen_set.id)

            # choose sampler
            _log("Sample selection: %s" % datarun.sample_selection)
            Sampler = Mapping.SELECTION_SAMPLES_MAP[datarun.sample_selection]
            sampler = None

            N_OPT = datarun.r_min

            ### UNIFORM SAMPLE SELECTION ###
            if datarun.sample_selection == SELECTION_SAMPLES_UNIFORM:
                sampler = Sampler(frozen_set=frozen_set)

            ### GRID SAMPLE SELECTION ###
            elif datarun.sample_selection == SELECTION_SAMPLES_GRID:
                learners = GetLearnersInFrozen(frozen_set_id)
                learners = [x.completed == None for x in learners]  # only completed learners

                sampler = Sampler(frozen_set=frozen_set, learners=learners, metric=datarun.metric)

            ### BASIC GP SAMPLE SELECTION ###
            elif datarun.sample_selection == SELECTION_SAMPLES_GP:
                learners = GetLearnersInFrozen(frozen_set_id)
                learners = [x.completed == None for x in learners]  # only completed learners

                # check if we have enough results to pursue this strategy
                if len(learners) < N_OPT:
                    _log("Not enough previous results, falling back onto strategy: %s" % SELECTION_SAMPLES_UNIFORM)
                    Sampler = Mapping.SELECTION_SAMPLES_MAP[SELECTION_SAMPLES_UNIFORM]
                    sampler = Sampler(frozen_set=frozen_set)
                else:
                    sampler = Sampler(frozen_set=frozen_set, learners=learners, metric=datarun.metric)

            ### GP EXPECTED IMPROVEMENT SAMPLE SELECTION ###
            elif datarun.sample_selection == SELECTION_SAMPLES_GP_EI:
                learners = GetLearnersInFrozen(frozen_set.id)
                learners = [x.completed == None for x in learners]  # only completed learners
                best_y = GetMaximumY(datarun.id, datarun.metric, default=0.0)

                # check if we have enough results to pursue this strategy
                if len(learners) < N_OPT:
                    _log(
                        "Not enough previous results for gp_ei, falling back onto strategy: %s" % SELECTION_SAMPLES_UNIFORM)
                    Sampler = Mapping.SELECTION_SAMPLES_MAP[SELECTION_SAMPLES_UNIFORM]
                    sampler = Sampler(frozen_set=frozen_set)
                else:
                    sampler = Sampler(frozen_set=frozen_set, learners=learners, metric=datarun.metric, best_y=best_y)

            elif datarun.sample_selection == SELECTION_SAMPLES_GP_EI_VEL:
                learners = GetLearnersInFrozen(frozen_set.id)
                learners = [x.completed == None for x in learners]  # only completed learners
                best_y = GetMaximumY(datarun.id, datarun.metric, default=0.0)

                # check if we have enough results to pursue this strategy
                if len(learners) < N_OPT:
                    _log("Not enough previous results for gp_eivel, falling back onto strategy: %s (learners %d < %d)" % (
                        SELECTION_SAMPLES_UNIFORM, len(learners), N_OPT))
                    Sampler = Mapping.SELECTION_SAMPLES_MAP[SELECTION_SAMPLES_UNIFORM]
                    sampler = Sampler(frozen_set=frozen_set)
                else:
                    sampler = Sampler(frozen_set=frozen_set, learners=learners, metric=datarun.metric, best_y=best_y)

            ### GP EXPECTED IMPROVEMENT PER TIME SAMPLE SELECTION ###
            elif datarun.sample_selection == SELECTION_SAMPLES_GP_EI_TIME:
                learners = GetLearnersInFrozen(frozen_set.id)
                learners = [x.completed == None for x in learners]  # only completed learners
                best_y = GetMaximumY(datarun.id, datarun.metric, default=0.0)

                # check if we have enough results to pursue this strategy
                if len(learners) < N_OPT:
                    _log(
                        "Not enough previous results for gp_eitime, falling back onto strategy: %s" % SELECTION_SAMPLES_UNIFORM)
                    Sampler = Mapping.SELECTION_SAMPLES_MAP[SELECTION_SAMPLES_UNIFORM]
                    sampler = Sampler(frozen_set=frozen_set)
                else:
                    sampler = Sampler(frozen_set=frozen_set, learners=learners, metric=datarun.metric, best_y=best_y)

            # select the parameters based on the sampler
            params = sampler.select()
            if params:
                print
                _log("Chose frozen set.")
                _log("Classifier type: %s" % frozen_set.algorithm)
                _log("Params chosen: %s" % params, False)
                print "Params chosen:"
                for k, v in params.items():
                    print "\t%s = %s" % (k, v)

                # train learner
                params["function"] = frozen_set.algorithm
                wrapper = CreateWrapper(params)
                trainX, testX, trainY, testY = LoadData(datarun, config)
                wrapper.load_data_from_objects(trainX, testX, trainY, testY)
                performance = wrapper.start()

                print
                _log("Judgement metric (%s): %.3f +- %.3f" %
                     (wrapper.judgment_metric,
                      performance["cv_judgment_metric"],
                      2 * performance["cv_judgment_metric_stdev"]))

                if ((performance["cv_judgment_metric"] -
                     performance["cv_judgment_metric_stdev"] * 2) >
                        best_perf - best_err):
                    best_perf = performance["cv_judgment_metric"]
                    best_err = performance["cv_judgment_metric_stdev"] * 2

                _log("Best so far: %.3f +- %.3f" %
                     (best_perf, best_err))

                print
                model = Model(wrapper, datarun.wrapper)

                # insert learner into the database
                InsertLearner(datarun, frozen_set, performance, params, model, started, config)

        except Exception as e:
            msg = traceback.format_exc()
            if datarun and frozen_set:
                _log("Error in main work loop: datarun=%s" % str(datarun) + msg)
                InsertError(datarun.id, frozen_set, params, msg)
            else:
                _log("Error in main work loop (no datarun or frozen set):" + msg)

        finally:
            time.sleep(loop_wait)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        if total_time is not None and elapsed_time >= total_time:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add more learners to database')
    parser.add_argument('-c', '--configpath', help='Location of config file',
                        default=os.getenv('ATM_CONFIG_FILE', 'config/atm.cnf'), required=False)
    parser.add_argument('-d', '--datarun-id', help='Only train on datasets with this id', default=None, required=False)
    parser.add_argument('-t', '--time', help='Number of seconds to run worker', default=None, required=False)

    # TODO: a little confusingly named
    parser.add_argument('-l', '--seqorder', help='work on datasets in sequential order starting with smallest id number, but still max priority (default = random)',
                        dest='choose_randomly', default=True, action='store_const', const=False)
    args = parser.parse_args()
    config = Config(args.configpath)

    # les go
    work(config, args.datarun_id, args.time, args.choose_randomly)
