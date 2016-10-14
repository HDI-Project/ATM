from delphi.run import Run
import os

"""
Add here the algorithm codes you'd like to run and compare. Look these up in the 
`algorithms` table in the MySQL database, or alternatively, in the config/hyperdelphi.sql 
file in this repository. You must spell them correctly!

Add each algorithm as a string to the list. 

Notes:
	- SVMs (classify_svm) can take a long time to train. It's not an error. It's just part of what
	happens when the algorithm happens to explore a crappy set of parameters on a powerful algo like this. 
	- SGDs (classify_sgd) can sometimes fail on certain parameter settings as well. Don't worry, they 
	train SUPER fast, and the worker.py will simply log the error and continue. 
"""
#algorithm_codes = ['classify_sgd', 'classify_dt', 'classify_dbn'] # commented out by Bryan
algorithm_codes = ['classify_svm', 'classify_dbn', 'classify_rf', 'classify_pa', 'classify_et', 'classify_sgd', 'classify_gnb', 'classify_mnb', 'classify_bnb', 'classify_knn', 'classify_logreg']


"""
This is where you list CSV files to train on. Follow the CSV conventions on the ALFA online
wiki page. 

Note that if you want Delphi to randomly create a train/test split, just add a single string 
CSV path to the list. If you've already got it split into two files, name them
	
	DATANAME_train.csv
	DATANAME_test.csv 

and add them as a tuple of string, with the TRAINING set as the first item and the TESTING 
as the second string in the tuple. 
"""
csvfiles = [
	#"data/binary/banknote.csv",
	# ("data/binary/congress_train.csv", "data/binary/congress_test.csv"),
]
# Added by Bryan(bcollazo@mit.edu) to run 800dataset experiment
path = "/home/bryan/800datasets"
trains = os.listdir(path)
for t in trains:
    if "test" in t: continue
    trainpath = os.path.join(path, t)
    testpath = os.path.join(path, t[:-9] + "test.csv")
    csvfiles.append((trainpath, testpath))
# End of addition

"""
How many learners would you like Delphi to train in this run? Be aware some classifiers are very
quick to train and others take a long time depending on the size and dimensionality of the data. 
"""
nlearners = 500


"""
Should there be a 
	"learner", or
	"walltime"

budget? You decide here. 
"""
budget_type = "learner"


"""
How should Delphi sample a frozen set that it must explore?
	- uniform: pick randomly! (baseline)
	- gp_ei: Gaussian Process expected improvement criterion
	- gp_eitime: Gaussian Process expected improvement criterion 
				per unit time

The number in the second part of the tuple is the `r_min` parameter. Consult
the thesis to understand what those mean, but essentially: 

	if (num_learners_trained_in_hyperpartition >= r_min)
		# train using sample criteria 
	else
		# train using uniform (baseline)
"""
sample_selectors = [
	# sample selection, r_min
	("uniform", -1),
	("gp_ei", 3),
	("gp_eitime", 3),
]


"""
How should Delphi select a hyperpartition (frozen set) from the current options it has? 

Again, each is a different method, consult the thesis. The second numerical entry in the
tuple is similar to r_min, except it is called k_window and determines how much "history"
Delphi considers for certain frozen selection logics. 
"""
frozen_selectors = [
	# frozen selection, k_window
	("uniform", -1),
	("ucb1", -1),
	("bestkvel", 5),
	("purebestkvel", 5),
	("hieralg", -1),
]
	

"""
What is the priority of this run? Higher is more important. 
"""
priority = 10


"""
What metric should Delphi use to score? Keep this as "cv".
"""
metric = "cv"


# now create the runs and populate the database
# you'll need to start workers after this finishes (or
# before, they will wait patiently!)
# Added by Bryan(bcollazo@mit.edu) for 800dataset Experiment
num_runs = 10
for i in xrange(num_runs): 
# End of Addition
    for csv in csvfiles:
            for sampler, r_min in sample_selectors:
                    for fsampler, k_window in frozen_selectors:
                            args = {
                                    "metric" : metric,
                                    "r_min" : r_min,
                                    "algorithm_codes" : algorithm_codes,
                                    "k_window" : k_window,
                                    "sample_selection" : sampler,
                                    "frozen_selection" : fsampler,
                                    "description" : "__".join([sampler, fsampler]),
                                    "priority" : priority,
                                    "budget_type" : budget_type,
                                    "learner_budget" : nlearners,
                                    "frozens_separately" : False,}
                            
                            if isinstance(csv, tuple):
                                    args["trainpath"] = csv[0]
                                    args["testpath"] = csv[1]
                                    runname = os.path.basename(csv[0])
                                    runname = runname.replace("_train", "")
                                    runname = runname.replace(".csv", "")
                                    args["runname"] = runname
                            else:
                                    args["alldatapath"] = csv
                                    args["runname"] = os.path.basename(csv).replace(".csv", "")
                            
                            Run(**args)  # start this run
