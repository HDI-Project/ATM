from delphi.database import *
from delphi.utilities import *
from sqlalchemy import desc, asc
import numpy as np
import matplotlib as mpl
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

mpl.rc('font', **font)
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import datetime
import pylab

names = {
    "classify_svm" : "SVM",
    "classify_rf" : "RF",
    "classify_dt" : "DT",
    "classify_logreg" : "LR",
    "classify_et" : "ET",
    "classify_gnb" : "GNB",
    "classify_knn" : "KNN",
    "classify_dbn" : "DBN",
    "classify_sgd" : "SGD",
    "classify_mnb" : "MNB",}

def plot_performance_histogram():
    session = Session()

    EnsureDirectory("plots/performance_histograms/uniform/")

    dataruns = session.query(Datarun).\
        filter(Datarun.description.like("%thesis%")).\
        filter(Datarun.description.like("%uniform%")).\
        filter().all()

    uniform_all_scores = {} # dataname => [scores] for histogram
    for datarun in dataruns:
        learners = session.query(Learner).filter(Learner.datarun_id == datarun.id).all()
        
        scores = []
        for learner in learners:
            if learner.is_error:
                scores.append(0.0)
            else:
                scores.append(float(learner.cv))

        if not datarun.name in uniform_all_scores:
            uniform_all_scores[datarun.name] = []
        uniform_all_scores[datarun.name].extend(scores)


    for dataname in uniform_all_scores.keys():

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        print dataname
        plt.hist(uniform_all_scores[dataname], bins=25)
        
        ax1.set_xlabel("Performance")
        ax1.set_ylabel("Number of Classifiers")
        #ax1.set_title("Classifier Performance Distribution for %s" % dataname)
            
        plt.savefig("plots/performance_histograms/uniform/%s.png" % (dataname,))

    session.close()

def plot_scatters():
    # plot scatters with optimization
    session = Session()

    EnsureDirectory("plots/scatters_optimization/")

    dataruns = session.query(Datarun).\
        filter(Datarun.description.like("%thesis%")).\
        filter().all()

    for datarun in dataruns:

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        markers = ['.']
        colors = ['b', 'r', 'g', 'k', 'c', 'm']
        plots = []
        labels = []

        learners = session.query(Learner).filter(Learner.datarun_id == datarun.id).all()
        print "Run %s has %d learners" % (datarun.name, len(learners))
        unique_algos = set()
        for learner in learners:
            unique_algos.add(learner.algorithm)

        for i, algorithm in enumerate(unique_algos):
            lnrs = [x for x in learners if x.algorithm == algorithm and x.completed]
            scores = []
            times = []
            for l in lnrs:
                if not l.cv:
                    l.cv = 0.0
                scores.append(float(l.cv))
                times.append((l.completed - l.started).total_seconds())

            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plot = ax1.scatter(times, scores, color=color, marker=marker, s=7)
            plots.append(plot)
            labels.append(names[algorithm]) 

        print "Setting axis and titles..."
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.87, box.height])
        ax1.legend(plots, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel("Time (sec)")
        ax1.set_ylabel("10 Fold Mean CV Accuracy")
        ax1.set_title("") #dataname)
        ax1.set_xscale('log')
        
        plt.savefig("plots/scatters_optimization/scatter-%s-%s.png" % (datarun.name, datarun.description))

    session.close()

def plot_performance_curves(errorbars=False):
    """
    In the segments that look like this:

    if not "derya" in name:
            continue 

    You can isolate only the datasets or the 
    optimization techniques that you want to 
    show. 
    """

    # lets plot some performnce curves
    descriptions = ["uniform__",
                    "ucb1__",
                    "bestkvel__",
                    "hieralg__",
                    "purebestkvel__",]
    dnames = ["derya_cell", "mooc_nlp5"]

    EnsureDirectory("plots/performance_curves/")

    session = Session()
    for name in dnames:

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        markers = ['.']
        colors = ['b', 'r', 'g', 'k', 'c', 'm']
        plots = []
        labels = []

        for k, desc in enumerate(descriptions):

            detail = desc

            lstr = "%%%s%%" % desc 
            nstr = "%%%s%%" % name 
            dataruns = session.query(Datarun).\
                filter(Datarun.description.like(lstr)).\
                filter(Datarun.name.like(nstr)).all()

            # collect
            full_dataruns = []
            for datarun in dataruns:
                lnrs = session.query(Learner).\
                    filter(Learner.datarun_id == datarun.id).\
                    filter(Learner.is_error != 1).\
                    order_by(Learner.started.asc()).all()
                if not lnrs:
                    continue
                full_dataruns.append(datarun)

            print "Now plotting all %d runs for %s and %s" % (len(full_dataruns), name, desc)

            runs = np.zeros((len(full_dataruns), 2000))
            for i, datarun in enumerate(full_dataruns):
                learners = session.query(Learner).\
                    filter(Learner.datarun_id == datarun.id).\
                    filter(Learner.is_error != 1).\
                    order_by(Learner.started.asc()).all()

                print "Run %d has %d learners" % (i, len(learners))

                best = float(learners[0].cv)
                print 'best starting at',best
                bests = []
                for j in range(2000):

                    # are we out of range?
                    # some runs are linger than others
                    l = None
                    if j < len(learners):
                        l = learners[j]
                    else:
                        runs[i, j] = best
                        continue

                    # we're not, keep going
                    #if not l.completed:
                    #    l.cv = 0.01
                    score = float(l.cv)
                    if score > best:
                        best = score 
                    runs[i, j] = best
                    bests.append(best)

                #print "BESTS", bests
                #print learners[0].id

            error = np.std(runs, axis=0)
            average = np.mean(runs, axis=0)

            marker = markers[k % len(markers)]
            color = colors[k % len(colors)]
            plot = None
            if not errorbars:
                plot, = ax1.plot(range(len(average)), average, color=color)
            else:
                plot,a,b = ax1.errorbar(range(len(average)), average, color=color, yerr=[error, error])
            plots.append(plot)
            labels.append(desc.replace("__", ""))

        print "Setting axis and titles..."
        #box = ax1.get_position()
        #ax1.set_position([box.x0, box.y0, box.width * 0.84, box.height])
        ax1.legend(plots, labels, loc='lower right')
        ax1.set_xlabel("Classifiers")
        ax1.set_ylabel("Mean Best 10-Fold Average CV Accuracy")
        ax1.set_xscale('log')
        ax1.set_xlim((0, 1400))
        ax1.set_ylim((.78, .9))
        #ax1.set_ylim((.4, .7))
        
        path = "plots/performance_curves/curve-errorbar-%s-%s.png" % (detail.replace("__", ""), name,)
        print "Saving to %s" % path
        plt.savefig(path)

    session.close()

# plot distrubtions hist
def plot_performance_distributions():
    session = Session()
    EnsureDirectory("plots/performance_distribution/")
    
    lstr = "%%%s%%" % "uniform" 
    dataruns = session.query(Datarun).\
        filter(Datarun.description.like(lstr)).all()
    
    name_scores = {} # name => scores
    for datarun in dataruns:
        if not datarun.name in name_scores:
            name_scores[datarun.name] = []
    
        learners = session.query(Learner).\
            filter(Learner.datarun_id == datarun.id).\
            filter(Learner.is_error != 1).all()
        for learner in learners:
            name_scores[datarun.name].append(float(learner.cv))
    
    for name, scores in name_scores.iteritems():
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.hist(scores, bins=20)
        ax1.set_xlabel("Average 10-Fold CV Accuracy")
        ax1.set_ylabel("Number of Classifiers")
        plt.savefig("plots/performance_distribution/" + name + ".png")

def plot_frozen_vs_gp():
    # plot each frozen set
    session = Session()
    partitions = session.query(FrozenSet).filter(FrozenSet.trained != 0).all()
    fhashes = set([x.frozen_hash for x in partitions])

    EnsureDirectory("plots/frozens/")

    print "Unique fhashes: %s" % fhashes

    for fhash in fhashes:
        
        learners = session.query(Learner).filter(Learner.frozen_hash == fhash).all()
        dataname2learners = GroupBy(learners, "dataname")
        
        for dataname, datalnrs in  dataname2learners.iteritems():
        
            desc2learners = GroupBy(datalnrs, "description")
            
            #plt.clf() #close('all')
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            plots = []
            labels = []
        
            for desc, lnrs in desc2learners.iteritems():
        
                scores = [float(x.cv) for x in lnrs]
                bests = []
                best = 0.0
                
                if "gp_ei" in desc:
                    label = "GP"
                    labels.append(label)
                    color = 'red'
                else:
                    label = "Uniform"
                    labels.append(label)
                    color = 'blue'
                
                for score in scores:
                    if score > best:
                        best = score
                    bests.append(best)
                
                plot, = ax1.plot(bests, color=color)
                plots.append(plot)
                
                #print "Plotting %s on %s" % (bests, label)
            
            # plot          
            ax1.legend(plots, labels, loc='lower right')
            ax1.set_xlabel("Classifiers")
            ax1.set_ylabel("Mean Best 10-Fold Average CV Accuracy")
            #ax1.set_xscale('log')
            #ax1.set_ylim((0.55, 0.77))
            #ax1.set_xlim((0, 40))
            
            print "plotting %s in %s" % (dataname, fhash)
            plt.savefig("plots/frozens/%s-%s.png" % (dataname, fhash))

# plot frozens on hypedelphi vs grid on delphi
EnsureDirectory("plots/frozen-svm/")
session = Session()
frozen_sets = session.query(FrozenSet).\
    filter(FrozenSet.algorithm == "classify_svm").\
    filter(FrozenSet.id > 249).all()
    
svm_mapping = {
	# hyper => grid
	"8bc3a3f5338e014de526eecac288ee77" : "ef28d19667802611c34e837daadc63a5", # linear
	"3988b40d28b41c1b5e1c1e981f6ec8c8" : "552ab20ed13e0ec7257a2980ed9a11de", # poly
	"e1c4162efed2645034583c492304e154" : "b19be79c6fd693317d954c34c1056b42", # sigmoid
	"de4c5acd8199b0b8232ebdd849da2901" : "4b2ca30e89102d363f8f355d1b46d4e3", # rbf
}

name_mapping = {
    "8bc3a3f5338e014de526eecac288ee77" : "linear", # linear
    "3988b40d28b41c1b5e1c1e981f6ec8c8" : "poly", # poly
    "e1c4162efed2645034583c492304e154" : "sigmoid", # sigmoid
    "de4c5acd8199b0b8232ebdd849da2901" : "rbf", # rbf
}

dataname = "mooc_forum_binary"

import pandas as pd
grid = pd.read_csv("delphi_grid_may21st.csv")
svms = grid[grid['algorithm'] == 'classify_svm']
churns = svms[svms['dataname'] == dataname]

fhashes = set([x.frozen_hash for x in frozen_sets])
for fhash in fhashes:

    # get hyperdelphi runs
    lnrs = session.query(Learner).\
        filter(Learner.algorithm == "classify_svm").\
        filter(Learner.dataname == dataname).\
        filter(Learner.frozen_hash == fhash).\
        filter(Learner.frozen_set_id > 249).\
        order_by(Learner.started.asc()).all()

    # get grid run
    frozen_rows = churns[churns['frozen_hash'] == svm_mapping[fhash]]
    grid_perfs = frozen_rows['cv'].tolist()
    grid_means, grid_stds = GetRandomBests(grid_perfs, n=1000, initBest=0.0)

    # get hyper runs
    did2learners = GroupBy(lnrs, "datarun_id")
    run_perfs = []
    for did, learners in did2learners.iteritems():
        gp_perfs = [float(x.cv) for x in learners]
        print "There are %d GP perfs and %d grid perfs for frozen set %s/%d, kernel=%s" % (
            len(gp_perfs), len(grid_perfs), fhash, did, name_mapping[fhash])
        bests = GetBests(gp_perfs, initBest=0.0)
        run_perfs.append(bests)

    m = max([len(p) for p in run_perfs])
    avg_hyper_at = []
    stddev_hyper_at = []
    for i in range(m):
        vals = [perfs[i] for perfs in run_perfs if i < len(perfs)]
        avg_hyper_at.append(np.mean(vals))
        stddev_hyper_at.append(np.std(vals))

    avg_hyper_at = GetBests(avg_hyper_at)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    plot1 = ax1.errorbar(range(len(grid_means)), grid_means, yerr=None, color='blue')
    plot2 = ax1.errorbar(range(len(avg_hyper_at)), avg_hyper_at, yerr=None, color='red')

    ax1.legend([plot1, plot2], ['Uniform', 'GP-EI'], loc='lower right')

    '''
    if name_mapping[fhash] == 'linear':
        ymin, ymax = 0.75, 0.88
        xmax = 15

    elif name_mapping[fhash] == 'poly':
        ymin, ymax = 0.7, 0.9
        xmax = 23 

    elif name_mapping[fhash] == 'sigmoid':
        ymin, ymax = 0.6, 0.9
        xmax = 60 

    elif name_mapping[fhash] == 'rbf':
        ymin, ymax = 0.75, 0.9
        xmax = 15 

    ax1.set_ylim((ymin, ymax))
    ax1.set_xlim((0, xmax))
    '''

    plt.savefig("plots/frozen-svm/%s-%s.png" % (dataname, name_mapping[fhash]))

    #print avg_hyper_at
    #print
