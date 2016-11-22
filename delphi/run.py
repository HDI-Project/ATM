from delphi.datawrapper import DataWrapper
from delphi.config import Config
from delphi.utilities import *
from delphi.database import *
from delphi.mapping import FrozenSetsFromAlgorithmCodes
import datetime

import warnings
warnings.filterwarnings("ignore")

OUTPUT_FOLDER = "data/processed"
LABEL_COLUMN = 0
PREFIX = "http://people.csail.mit.edu/drevo/datasets-v2/"
PREFIX = "http://people.csail.mit.edu/bcollazo/datasets/"   # Added by bcollazo

def Run(runname, description, metric, sample_selection, frozen_selection, budget_type, priority,
		k_window, r_min, algorithm_codes, learner_budget=None, walltime_budget=None, alldatapath=None, dataset_description=None,
		trainpath=None, testpath=None, configpath="config/experiments.cnf", verbose=True, frozens_separately=False):
			
	EnsureDirectory("models")
	EnsureDirectory("logs")
	
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
	training_path, testing_path = dw.wrap()
	stats = dw.get_statistics()
	print "Training data: %s" % training_path
	print "Testing data: %s" % testing_path
	training_http = PREFIX + os.path.basename(training_path)
	testing_http = PREFIX + os.path.basename(testing_path)
	
	# create all combinations necessary
	# config = Config(configpath)
	frozen_sets = FrozenSetsFromAlgorithmCodes(algorithm_codes, verbose=verbose)
	
	### create datarun ###
	values = {
		"name" : runname, 
		"trainpath" : training_http,
		"testpath" : testing_http,
		"local_trainpath" : training_path,
		"local_testpath" : testing_path,
		"labelcol" : stats["label_col"],
		"metric" : metric,
		"description" : description,
		"wrapper" : dw,
		"n" : int(stats["n_examples"]),
		"k" : int(stats["k_classes"]),
		"d" : int(stats["d_features"]),
		"majority" : float(stats["majority"]),
		"size_kb" : int(stats["datasize_bytes"]),
		"k_window" : k_window,
		"r_min" : r_min,}
		
	### dataset description ###
	if priority:
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
		datarun = Datarun(**values)
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
			#print fsettings, "=>", optimizables, constants
			
			if frozens_separately:
				datarun = Datarun(**values)
				session.add(datarun)
				session.commit()
				datarun_id = datarun.id
				datarun_ids.append(datarun_id)
			
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

	return datarun_ids

	
	
