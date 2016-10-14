from delphi.datawrapper import DataWrapper
from delphi.config import Config
from delphi.utilities import *
from delphi.database import *
from delphi.mapping import FrozenSetsFromAlgorithmCodes
import datetime
import argparse
import sys

import warnings
warnings.filterwarnings("ignore")

"""
[*] For when training and testing are split up:
python newdata.py -r fake -d random --trainpath data/binary/fake_train.csv --testpath data/binary/fake_test.csv

[*] For when training and testing in SAME file:
python newdata.py -r congress -d random -a data/binary/congress.csv

python newdata.py -r spam -d testing -a data/binary/spam.csv
"""

# setup
OUTPUT_FOLDER = "data/processed"
LABEL_COLUMN = 0
PREFIX = "http://people.csail.mit.edu/drevo/datasets-v2/"
EnsureDirectory("models")
EnsureDirectory("logs")

# grab the command line arguments
parser = argparse.ArgumentParser(description='Add more learners to database')
parser.add_argument('-t','--trainpath', help='Path to training CSV', required=False)
parser.add_argument('-v','--testpath', help='Path to testing CSV', required=False)
parser.add_argument('-a','--alldatapath', help='Path to data, all as one file', required=False)
parser.add_argument('-c','--configpath', help='Config file location', required=False, default="experiments/.cnf")
parser.add_argument('-r','--runname', help='Runname', required=True)
parser.add_argument('-q','--quiet', type=int, help='quiet', default=1, required=False)
parser.add_argument('-d','--description', help='description', required=True)
args = parser.parse_args()
verbose = True if args.quiet == 0 else False

print "Dataname: %s, description: %s" % (args.runname, args.description)

assert args.alldatapath or (args.trainpath and args.testpath), \
    "Must have either a single file for data or two paths, one for training and the other for testing!"

# parse data and create data wrapper for vectorization and label encoding
dw = None
if args.alldatapath:
    dw = DataWrapper(args.runname, OUTPUT_FOLDER, LABEL_COLUMN, traintestfile=args.alldatapath)
elif args.trainpath and args.testpath:
    dw = DataWrapper(args.runname, OUTPUT_FOLDER, LABEL_COLUMN, trainfile=args.trainpath, testfile=args.testpath)
else:
    raise Exception("No valid training or testing files!")

# wrap the data and save it to disk
training_path, testing_path = dw.wrap()
stats = dw.get_statistics()
print "Training data: %s" % training_path
print "Testing data: %s" % testing_path
training_http = PREFIX + os.path.basename(training_path)
testing_http = PREFIX + os.path.basename(testing_path)

# create all combinations necessary
config = Config(args.configpath)
algorithm_codes = [x.strip() for x in config.get(Config.RUN, Config.RUN_ALGORITHMS).split(",")]
frozen_sets = FrozenSetsFromAlgorithmCodes(algorithm_codes, verbose=verbose)

### create datarun ###
values = {
	"name" : args.runname, 
	"trainpath" : training_http,
	"testpath" : testing_http,
	"labelcol" : stats["label_col"],
	"metric" : config.get(Config.STRATEGY, Config.STRATEGY_METRIC),
	"description" : args.description,
	"wrapper" : dw,
	"n" : int(stats["n_examples"]),
	"k" : int(stats["k_classes"]),
	"d" : int(stats["d_features"]),
	"majority" : float(stats["majority"]),
	"size_kb" : int(stats["datasize_bytes"]),
	"k_window" : config.get(Config.STRATEGY, Config.STRATEGY_K),
	"r_min" : config.get(Config.STRATEGY, Config.STRATEGY_R),
}

### test for budget restrictions ###
if config.get(Config.BUDGET, Config.BUDGET_TYPE) and config.get(Config.BUDGET, Config.BUDGET_TYPE) != "none":
	values["budget"] = config.get(Config.BUDGET, Config.BUDGET_TYPE)
	print "Budget will be used: True"

if config.get(Config.BUDGET, Config.BUDGET_TYPE) == Config.CONST_LEARNER:
	nlearners = config.getint(Config.BUDGET, Config.BUDGET_LEARNER)
	values["learner_budget"] = nlearners
	print "Learner budget: %d classifiers" % nlearners

elif config.get(Config.BUDGET, Config.BUDGET_TYPE) == Config.CONST_WALLTIME:
	minutes = config.getint(Config.BUDGET, Config.BUDGET_WALLTIME)
	values["walltime_budget_minutes"] = minutes
	values["deadline"] = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
	print "Walltime budget: %d minutes, deadline=%s" % (minutes, str(values["deadline"]))

	
### test for selection strategy changes ###
if config.get(Config.STRATEGY, Config.STRATEGY_SELECTION):
	values["sample_selection"] = config.get(Config.STRATEGY, Config.STRATEGY_SELECTION)
	print "Sample selection: %s" % values["sample_selection"]

### test for frozen strategy changes ###
if config.get(Config.STRATEGY, Config.STRATEGY_FROZENS):
	values["frozen_selection"] = config.get(Config.STRATEGY, Config.STRATEGY_FROZENS)
	print "Frozen selection: %s" % values["frozen_selection"]

### insert datarun ####
session = GetConnection()
datarun = Datarun(**values)
session.add(datarun)
session.commit()

print "Datarun ID: %d" % datarun.id

### insert frozen sets ###
session.autoflush = False
for algorithm, frozens in frozen_sets.iteritems():
	for fsettings, others in frozens.iteritems():
		optimizables, constants = others
		#print fsettings, "=>", optimizables, constants
		fhash = HashNestedTuple(fsettings)
		fset = FrozenSet(**{
			"datarun_id" : datarun.id, 
			"algorithm" : algorithm, 
			"optimizables" :  optimizables, 
			"constants" :  constants, 
			"frozens" :  fsettings, 
			"frozen_hash" : fhash
		})
		session.add(fset)	
session.commit()
session.close()

# tell user to upload files to web PREFIX
print "Upload: %s => %s" % (training_path, training_http)
print "Upload: %s => %s" % (testing_path, testing_http)
