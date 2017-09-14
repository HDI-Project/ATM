from atm.datawrapper import DataWrapper
from atm.config import Config
from atm.database import *
from atm.utilities import *
from atm.mapping import FrozenSetsFromAlgorithmCodes
import atm.database as db

from boto.s3.connection import S3Connection, Key as S3Key
import datetime
import pdb

import warnings

warnings.filterwarnings("ignore")

# TODO: get rid of these hard-coded values
OUTPUT_FOLDER = "data/processed"
LABEL_COLUMN = "class"


def Run(config, runname, description, metric, sample_selection, frozen_selection, budget_type, priority,
        k_window, r_min, algorithm_codes, learner_budget=None, walltime_budget=None, alldatapath=None,
        dataset_description=None, trainpath=None, testpath=None, verbose=True, frozens_separately=False):
    EnsureDirectory("models")
    EnsureDirectory("logs")

    # call database method to define ORM objects in the db module
    db.define_tables(config)

    print "Dataname: %s, description: %s" % (runname, description)

    assert alldatapath or (trainpath and testpath), \
        "Must have either a single file for data or two paths, one for training and the other for testing!"

    # parse data and create data wrapper for vectorization and label encoding
    dw = None
    if alldatapath:
        dw = DataWrapper(runname, OUTPUT_FOLDER, LABEL_COLUMN, traintestfile=alldatapath)
    elif trainpath and testpath:
        dw = DataWrapper(runname, OUTPUT_FOLDER, LABEL_COLUMN, trainfile=trainpath, testfile=testpath)
    else:
        raise Exception("No valid training or testing files!")

    # wrap the data and save it to disk
    local_training_path, local_testing_path = dw.wrap()
    stats = dw.get_statistics()
    print "Training data: %s" % local_training_path
    print "Testing data: %s" % local_testing_path

    # create all combinations necessary
    frozen_sets = FrozenSetsFromAlgorithmCodes(algorithm_codes, verbose=verbose)

    ### create datarun ###
    values = {
        "name": runname,
        "local_trainpath": local_training_path,
        "local_testpath": local_testing_path,
        "labelcol": stats["label_col"],
        "metric": metric,
        "description": description,
        "wrapper": dw,
        "n": int(stats["n_examples"]),
        "k": int(stats["k_classes"]),
        "d": int(stats["d_features"]),
        "majority": float(stats["majority"]),
        "size_kb": int(stats["datasize_bytes"]),
        "k_window": k_window,
        "r_min": r_min, }

    ### dataset description ###
    if dataset_description:
        values["dataset_description"] = dataset_description

    ### priority ###
    if priority:
        values["priority"] = priority

    ### budget restrictions ###
    values["budget"] = budget_type

    if learner_budget:
        values["learner_budget"] = learner_budget

    elif walltime_budget:
        minutes = walltime_budget
        values["walltime_budget_minutes"] = walltime_budget
        values["deadline"] = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
        print "Walltime budget: %d minutes, deadline=%s" % (minutes, str(values["deadline"]))

    # selection strategies
    values["sample_selection"] = sample_selection
    print "Sample selection: %s" % values["sample_selection"]
    values["frozen_selection"] = frozen_selection
    print "Frozen selection: %s" % values["frozen_selection"]

    ### insert datarun ####
    session = GetConnection()
    datarun = None
    datarun_ids = []
    if not frozens_separately:
        datarun = db.Datarun(**values)
        print datarun
        session.add(datarun)
        session.commit()
        print "Datarun ID: %d" % datarun.id
        datarun_id = datarun.id
        datarun_ids.append(datarun_id)

    ### insert frozen sets ###
    session.autoflush = False
    for algorithm, frozens in frozen_sets.iteritems():
        for fsettings, others in frozens.iteritems():
            optimizables, constants = others
            # print fsettings, "=>", optimizables, constants

            if frozens_separately:
                datarun = db.Datarun(**values)
                session.add(datarun)
                session.commit()
                datarun_id = datarun.id
                datarun_ids.append(datarun_id)

            fhash = HashNestedTuple(fsettings)
            fset = db.FrozenSet(**{
                "datarun_id": datarun.id,
                "algorithm": algorithm,
                "optimizables": optimizables,
                "constants": constants,
                "frozens": fsettings,
                "frozen_hash": fhash
            })
            session.add(fset)
    session.commit()
    session.close()

    run_mode = config.get(Config.MODE, Config.MODE_RUNMODE)
    if run_mode == 'cloud':
        aws_key = config.get(Config.AWS, Config.AWS_ACCESS_KEY)
        aws_secret = config.get(Config.AWS, Config.AWS_SECRET_KEY)
        conn = S3Connection(aws_key, aws_secret)
        s3_bucket = config.get(Config.AWS, Config.AWS_S3_BUCKET)
        bucket = conn.get_bucket(s3_bucket)
        ktrain = S3Key(bucket)

        if config.get(Config.AWS, Config.AWS_S3_FOLDER).strip():
            aws_training_path = os.path.join(config.get(Config.AWS, Config.AWS_S3_FOLDER),
                                             local_training_path)
            aws_testing_path = os.path.join(config.get(Config.AWS, Config.AWS_S3_FOLDER),
                                            local_testing_path)
        else:
            aws_training_path = local_training_path
            aws_testing_path = local_testing_path

        ktrain.key = aws_training_path
        ktrain.set_contents_from_filename(local_training_path)
        ktest = S3Key(bucket)
        ktest.key = aws_testing_path
        ktest.set_contents_from_filename(local_testing_path)
        print 'CLOUD MODE: Train and test files uploaded to AWS S3 Bucket {}'.format(s3_bucket)
    else:
        print 'LOCAL MODE: Train and test files only on local drive'

    return datarun_ids
