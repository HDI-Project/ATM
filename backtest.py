from __future__ import division

from delphi.predictor import Predictor
from delphi.model import Model
from delphi.utilities import *
from delphi.database import *

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pandas as pd
import traceback, sys
import copy

from multiprocessing.pool import ThreadPool

import matplotlib as mpl
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
mpl.rc('font', **font)
mpl.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

###########
FILTER_ALL = "all"
FILTER_LEADERS = "leaders"
FILTER_HYPERLEADERS = "hyperleaders"
FILTER_DIVERSITY = "diverse"
    
def FilterLearners(learners, filter_type, baseline):
    """
    Takes in a list of learners and filters them according 
    """
    flearners = []
    learners = [x for x in learners if x.test > baseline and x.completed != None]
    
    if filter_type == FILTER_ALL:
        flearners = learners[:]
    
    elif filter_type == FILTER_LEADERS:
        algorithms = set([x.algorithm for x in learners])
        
        alg2bestlearner = {alg : None for alg in algorithms}
        alg2bestlearnerscore = {alg : baseline for alg in algorithms}
        for learner in learners:
            test = float(learner.test)
            alg = learner.algorithm
            if test > alg2bestlearnerscore[alg]:
                alg2bestlearner[alg] = learner
                alg2bestlearnerscore[alg] = test
                
        flearners = [blnr for (alg, blnr) in alg2bestlearner.iteritems() if blnr != None]
        
    elif filter_type == FILTER_HYPERLEADERS:
        
        fhash2learners = GroupBy(learners, "frozen_hash")
        fhash2best = { fhash : None for fhash in fhash2learners.keys() } # fhash => learner
        
        # for each fhash's learners, collect the best
        for fhash, lnrs in fhash2learners.iteritems():
             
             best = 0.0
             bestLearner = None
             for lnr in lnrs:
                score = float(lnr.cv)
                if score > best:
                    best = score
                    bestLearner = lnr
             
             if bestLearner:
                fhash2best[fhash] = bestLearner
        
        # add all the leaders to the list
        for fhash, lnrs in fhash2learners.iteritems(): 
            for lnr in lnrs:
                if lnr:
                    flearners.append(lnr)
        
    elif filter_type == FILTER_DIVERSITY:
        
        ######### TODO: not implemented yet
        flearners = learners[:]
        
    elif filter_type == None:
        flearners = learners[:]
    
    return flearners
            
    
##########

def DownloadAndGetData(learner):

    #path = DownloadFileHTTP(learner.trainpath, verbose=False)
    path = os.path.basename(learner.trainpath) ####
    
    training = np.genfromtxt(path, delimiter=",")
    trainingY = training[:, 0]
    trainingX = training[:, 1:]

    #testpath = DownloadFileHTTP(learner.testpath, verbose=False)
    testpath = os.path.basename(learner.testpath) ####
    
    testing = np.genfromtxt(testpath, delimiter=",")
    testingY = testing[:, 0]
    testingX = testing[:, 1:]
    
    return trainingX, trainingY, testingX, testingY

def backtest_ensemble(datarun_id, ensemble_type):
    """
        Returns:
            grid_bests = [ best at each point i in time ]
            ens_bests = [ best ensemble score at each point in time ]
            models = (time at which it was found, best scoring ensembles as a [])
    """
    # fetch run learners
    session = Session()
    print 'Fetching learners...'
    learners = session.query(Learner).\
        filter(Learner.completed != None).\
        filter(Learner.is_error != True).\
        filter(Learner.datarun_id == datarun_id ).\
        order_by(Learner.completed.asc()).all()
        
    # load testing data
    trainingX, trainingY, testingX, testingY = DownloadAndGetData(learners[0])
 
    # get baseline
    bincounts = np.bincount(testingY.astype(int))
    most_prevalent = float(max(bincounts) / sum(bincounts))
    print "Baseline: %f" % most_prevalent

    # incrementally create ensembles
    models = []
    temp = []
    
    ens_bests = [] # (t at which it happened, score, models)
    ens_best = 0.0
    ens_models = None
    
    grid_bests = []
    grid_best = 0.0

    print "Let's start..."
    for i, learner in enumerate(learners):
        try:
            if float(learner.test) > grid_best:
                grid_best = float(learner.test)
            grid_bests.append(grid_best)

            model = joblib.load(learner.modelpath)
            if ObjHasMethod(model.algorithm.pipeline.steps[-1][1], "predict_proba") and learner.test > most_prevalent:
                models.append(model)
            else:
                # ensemble best doesn't change
                ens_bests.append(max(grid_best, ens_best))
                continue

            predictor = Predictor(models, ensemble_type=ensemble_type)
            if ensemble_type == Predictor.ENSEMBLE_STACKING:
                predictor.setup(trainingX=trainingX, trainingY=trainingY, input_type=Model.INPUT_VECT)
            #print "Added model: %s" % model
            predictions = predictor.predict(testingX, input_type=Model.INPUT_VECT, output_type=Model.OUTPUT_INT)
            acc = accuracy_score(testingY, predictions)
            print "Accuracy with %d models: %f, step %d" % (len(models), acc, i)

            if acc > ens_best:
                ens_best = acc
                ens_models = (i, models)
            ens_bests.append(ens_best)

            if i > 70:
                break
            
        except Exception as e:
            print traceback.format_exc()
            #print "Error for round %d" % i
            pass

    session.close()
    return grid_bests, ens_bests, models
    
def append_best(listref, default):
    if listref:
        listref.append(max(listref))
    else:
        listref.append(default)
    
def backtest(run_id, eftypes):
    """
        Returns:
            (ensemble_type, filter_type) => { 
                "bests" : 
                    [ float, float, ... ],
                "changes" :
                    [ (i, float, learners), ...]
            }
    """
    # fetch run learners
    session = Session()
    print 'Fetching learners...'
    learners = session.query(Learner).\
        filter(Learner.completed != None).\
        filter(Learner.is_error != True).\
        filter(Learner.datarun_id == run_id ).\
        order_by(Learner.completed.asc()).all()
        
    # load testing data
    trainingX, trainingY, testingX, testingY = DownloadAndGetData(learners[0])
    
    # get baseline
    bincounts = np.bincount(testingY.astype(int))
    most_prevalent = float(max(bincounts) / sum(bincounts))
    print "Baseline: %f" % most_prevalent
    
    # variables for all loops
    model_pool = [] 
    learner_to_model = {} # learner_id => model
    stacker_learners = set() # learner_id of models that stack without NaNs
    
    # (prediction_type, filter_type) => { 
    #   "bests" : 
    #       [ float, float, ... ],
    #   "changes" :
    #       [ (i, float, learners), ...]
    # }
    predict_and_filter_map = {} 
    for entry_tup in eftypes:
        predict_and_filter_map[entry_tup] = {
            "bests" : [],
            "changes" : [],} 
    
    # incrementally try to find the best
    for i, learner in enumerate(learners):
        if i > 30:
            break
            
        print "-> Classifier %d" % (i + 1)
        
        # should we skip using this learner?
        skip = False
        
        if not learner.test or learner.test < most_prevalent:
            skip = True
        
        model = None
        try:
            model = joblib.load(learner.modelpath)
            learner_to_model[learner.id] = model
            
            if ObjHasMethod(model.algorithm.pipeline.steps[-1][1], "predict_proba"):
                # this is a valid probabilistic model
                stacker_learners.add(learner.id)
            
            #print "performance:", model.algorithm.perf
        except Exception as e:
            skip = True
            print "Error loading model: %s" % traceback.format_exc()
        
        for tup in eftypes:
            ensemble_type, filter_type = tup
            
            if skip or not model or (ensemble_type == Predictor.ENSEMBLE_STACKING and not ObjHasMethod(model.algorithm.pipeline.steps[-1][1], "predict_proba")):
                append_best(predict_and_filter_map[tup]["bests"], most_prevalent)
                continue
            
            # filter our learners
            filtered = FilterLearners(learners[:i+1], filter_type, most_prevalent)
            filtered_ids = set([x.id for x in filtered])
            
            # create ensemble
            acc = None
            if ensemble_type == None:
                # base case of voting is a single predictor
                predictor = Predictor([model], ensemble_type=Predictor.ENSEMBLE_VOTING) 
                #acc = model.algorithm.test_score 
                #print "single model score", acc
            else:
                try:
                    models = [v for (k,v) in learner_to_model.iteritems() if k in filtered_ids]
                    #print "models", models
                    predictor = Predictor(models, ensemble_type=ensemble_type)
                    if ensemble_type == Predictor.ENSEMBLE_STACKING:
                        predictor.setup(trainingX=trainingX, trainingY=trainingY, input_type=Model.INPUT_VECT)
                
                except ValueError as ve:
                    #print "Stacking failed for model %s: %s" % (learner.modelpath, traceback.format_exc())
                    print "Stacking failed...", traceback.format_exc(), model.algorithm.code
                    append_best(predict_and_filter_map[tup]["bests"], most_prevalent)
                    continue
                
                except NotImplementedError as ne:
                    print "This particular setting of this model (%s) doesn't provide probabilities! %s" % \
                        (model.algorithm.code, traceback.format_exc())
                    append_best(predict_and_filter_map[tup]["bests"], most_prevalent)
                    continue
             
            predictions = predictor.predict(testingX, input_type=Model.INPUT_VECT, output_type=Model.OUTPUT_INT)
            acc = accuracy_score(testingY, predictions)
            
            #print "Test accuracy of (%s, %s) with %d models: %f, step %d" % (ensemble_type, filter_type, len(model_pool), acc, i)
            
            # did we beat the best?
            best = max(predict_and_filter_map[tup]["bests"] + [most_prevalent])
            if acc > best:
                predict_and_filter_map[tup]["bests"].append(acc)
                predict_and_filter_map[tup]["changes"].append((i, acc, predictor))
                print "New best for", tup, ":", (i, acc)
            else:
                predict_and_filter_map[tup]["bests"].append(best)

    return predict_and_filter_map
            
    
########################################
############ Begin logic ###############
########################################

EnsureDirectory("plots/ensembles/")

#runs = [1L] # local
runs = [302L, 303L, 304L, 305L, 306L, 307L, 308L, 309L, 310L, 311L]
name = "MF-Binary"

eftypes = [
    (None, FILTER_ALL), # no ensemble
    #(Predictor.ENSEMBLE_VOTING, FILTER_ALL),
    #(Predictor.ENSEMBLE_VOTING, FILTER_LEADERS),
    (Predictor.ENSEMBLE_VOTING, FILTER_HYPERLEADERS),
    #(Predictor.ENSEMBLE_STACKING, FILTER_ALL),
    #(Predictor.ENSEMBLE_STACKING, FILTER_LEADERS),
    #(Predictor.ENSEMBLE_STACKING, FILTER_HYPERLEADERS),
]

all_bests = {} # tup => [bests, bests, ...]
maxrunlength = 0
for run in runs:
    results = backtest(run, eftypes)
    
    keys = results.keys()
    for tup in keys:
        ensemble_type, filter_type = tup
        bests = results[tup]['bests']
        maxrunlength = max(maxrunlength, len(bests))
        if not tup in all_bests:
            all_bests[tup] = []
        all_bests[tup].append(bests)
        
averages = {}
for tup in all_bests.keys():
    arr = np.array(all_bests[tup][:maxrunlength])
    averages[tup] = np.mean(arr, axis=0)

fig = plt.figure()
ax1 = fig.add_subplot(111)
plots, labels = [], []
    
for tup in averages.keys():
    p1, = ax1.plot(range(len(averages[tup])), averages[tup])
    plots.append(p1)
    
    if ensemble_type == None:
    	labels.append("SelectBest Baseline")
    else:
    	labels.append("%s with %s" % tup)

ax1.set_ylim((0.8, 0.9))
ax1.set_xlim((0, 40))
ax1.legend(plots, labels, loc='lower right')
plt.savefig("plots/ensembles/%s.png" % (name,))



