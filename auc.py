from delphi.model import Model 
from delphi.database import *
from delphi.utilities import *

from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
import traceback

import matplotlib as mpl
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

mpl.rc('font', **font)
mpl.use('Agg')
import matplotlib.pyplot as plt

EnsureDirectory("plots/auc-kdd/")

def plotROC(fpr, tpr, roc_auc, dataname, alg):
   plt.clf()
   plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % (roc_auc,))
   plt.plot([0, 1], [0, 1], 'k--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.0])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   #plt.title('ROC: lead = %s lag = %s' % (lead, lag))
   plt.legend(loc="lower right")
   plt.savefig("plots/auc-kdd/auc-%s-%s.png" % (dataname, alg))

session = Session()
dataruns = session.query(Datarun).filter(Datarun.description.like("%%")).all()

print dataruns

name2dataruns = GroupBy(dataruns, "name")

for name, datarun in name2dataruns.iteritems():
	
	print "Name: %s" % name
	learners = session.query(Learner).\
		filter(Learner.dataname == name).\
		filter(Learner.completed != None).\
		order_by(Learner.cv.desc()).\
		order_by(Learner.stdev.asc()).\
		limit(100).all()
		
	print "Found %d learners" % len(learners)
		
	best_overall_auc = 0.0
	best_fpr, best_tpr = 0.0, 0.0
	best_overall_alg = None
	
	best_aucs = {}
	best_fprtpr = {}
	
	for learner in learners:
	
		if not learner.algorithm in best_aucs:
			best_aucs[learner.algorithm] = 0.0
			best_fprtpr[learner.algorithm] = (1.0, 0.0)
		
		try:
			testingpath = os.path.join("data", "processed", os.path.basename(learner.testpath))
			testing = pd.read_csv(testingpath)
			testingnp = np.array(testing)
			y_true = testingnp[:, 0]
			testingX = testingnp[:, 1:]
			model = joblib.load(learner.modelpath)
			
			if learner.algorithm == 'classify_dbn':
				pr = model.algorithm.predict(testingX, probability=False)
			else:
				pr = model.algorithm.predict(testingX, probability=True)
			
			fpr, tpr, thresholds = metrics.roc_curve(y_true, pr[:, 0], pos_label=0)
			auc = metrics.auc(fpr, tpr)
			
			print "AUC = %f for %s" % (auc, learner.algorithm)
			
			if auc > best_aucs[learner.algorithm]:
				print "Best AUC found for %s!" % learner.algorithm
				print "MODEL: %s" % learner.modelpath
				best_fprtpr[learner.algorithm] = fpr, tpr
				best_aucs[learner.algorithm] = auc
			
			if auc > best_overall_auc:
				best_overall_auc = auc
				best_overall_alg = learner.algorithm
			
		except Exception as e:
			#print traceback.format_exc()
			continue
	
	# and plot
	if learners:
		for algorithm in best_aucs.keys():
			print "Plotting %s, %s..." % (learners[0].dataname, algorithm)
			fpr, tpr = best_fprtpr[algorithm]
			plotROC(fpr, tpr, best_aucs[algorithm], learners[0].dataname, algorithm)