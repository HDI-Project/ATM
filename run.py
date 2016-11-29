from delphi.run import Run
import argparse
import os
import pdb

parser = argparse.ArgumentParser(description='Find best classifier for a given dataset.',epilog='See README.md for further information.')

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
parser.add_argument('-data', dest='csvfiles', nargs='+', help='data file(s)')
parser.add_argument('-train', dest='trainfiles', nargs='+', help='data training file(s)')
parser.add_argument('-test', dest='testfiles', nargs='+', help='data test file(s)')

"""
Description of the dataset. This helps with the analysis of classifier performances across many different problems.
"""
parser.add_argument('-description', dest='dataset_description', type=str, nargs=1, default='', help='description of the dataset (string that must be bookended by quotes ("")')

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
parser.add_argument('-algorithms', dest='algorithm_codes', nargs='+', choices=['classify_svm','classify_et','classify_pa','classify_sgd','classify_rf','classify_mnb','classify_bnb','classify_dbn','classify_logreg','classify_gnb','classify_dt','classify_knn','all'], default='all', help='classifiers to test')

"""
Should there be a 
	"learner", or
	"walltime"

budget? You decide here. 
"""
parser.add_argument('-b', dest='budget_type', choices=['learner','walltime'], nargs=1, default='learner', help='budget type')

"""
How many learners would you like Delphi to train in this run? Be aware some classifiers are very
quick to train and others take a long time depending on the size and dimensionality of the data. 
"""
parser.add_argument('-N', dest='nlearners', type=int, nargs=1, default=250, help='budget amount')

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
parser.add_argument('-gp', dest='gp', choices=['uniform','gp_ei','gp_eitime'], type=str, nargs=1, default='gp_ei', help='parameter estimation strategy')
parser.add_argument('-r_min', dest='r_min', type=int, nargs=1, default=3, help='minimum trials before mab')

"""
How should Delphi select a hyperpartition (frozen set) from the current options it has? 

Again, each is a different method, consult the thesis. The second numerical entry in the
tuple is similar to r_min, except it is called k_window and determines how much "history"
Delphi considers for certain frozen selection logics. 
"""
parser.add_argument('-mab', dest='mab', choices=['uniform','ucb1','bestkvel','purebestkvel','hieralg'], nargs=1, type=str, default='ucb1', help='hyperpartition selection strategy')
parser.add_argument('-k_window', dest='k_window', type=int, nargs=1, default=-1, help='gp memory size')

"""
What is the priority of this run? Higher is more important. 
"""
parser.add_argument('-priority', dest='priority', type=int, nargs=1, default=10, help='job priority (higher = more important)')

"""
What metric should Delphi use to score? Keep this as "cv".
"""
metric = "cv"

clargs = parser.parse_args()

if(bool(clargs.csvfiles)):
    csvfiles = clargs.csvfiles
    
if(bool(clargs.trainfiles) and bool(clargs.testfiles)):
    csvfiles = [(clargs.trainfiles[0], clargs.testfiles[0])]


if(clargs.algorithm_codes == 'all'):
    algorithm_codes = ['classify_svm','classify_et','classify_pa','classify_sgd','classify_rf','classify_mnb','classify_bnb','classify_dbn','classify_logreg','classify_gnb','classify_dt','classify_knn']
else:
    algorithm_codes = clargs.algorithm_codes
    
if(type(clargs.mab) == list):
    mab = clargs.mab[0]
else:
    mab = clargs.mab
    
if(type(clargs.gp) == list):
    gp = clargs.gp[0]
else:
    gp = clargs.gp
    


dataset_description = clargs.dataset_description
nlearners = clargs.nlearners
budget_type = clargs.budget_type
sample_selectors = [(gp, clargs.r_min)]
frozen_selectors = [(mab, clargs.k_window)]
priority = clargs.priority

if(bool(dataset_description) and (len(dataset_description[0]) > 1000)):
    raise ValueError('Dataset description is more than 1000 characters.')

# now create the runs and populate the database
# you'll need to start workers after this finishes (or
# before, they will wait patiently!)
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
                "dataset_description" : dataset_description,
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
                
            #Run(**args)  # start this run
            pdb.run('Run(**args)')