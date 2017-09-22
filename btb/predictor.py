from btb.model import Model
from btb.utilities import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nolearn.dbn import DBN
from sklearn import tree
import numpy as np

class Predictor(object):

    ENSEMBLE_VOTING = "voting"
    ENSEMBLE_STACKING = "stacking"

    def __init__(self, models, ensemble_type='voting'):
        self.models = models
        self.ensemble_type = ensemble_type

        # in order to use stacking all classifiers must have probabilistic
        # estimates for predictions
        if self.ensemble_type == Predictor.ENSEMBLE_STACKING:
            new_models = []
            for model in self.models:
            	clf = model.algorithm.pipeline.steps[-1][1]
                if ObjHasMethod(clf, "predict_proba"):
                    if model.algorithm.code == "classify_sgd" \
                    	and hasattr(clf, "loss") \
                    	and clf.loss not in ['modified_huber', 'log']:
                    	pass
                    else:
                    	new_models.append(model)
            self.models = new_models

        self.prediction_map = {
            Predictor.ENSEMBLE_VOTING : self.vote, 
            Predictor.ENSEMBLE_STACKING: self.stack,
        }

    def setup(self, trainingX=None, trainingY=None, input_type=Model.INPUT_CSV, stacker=None):
        
        if self.ensemble_type == Predictor.ENSEMBLE_STACKING:

            assert trainingX != None and trainingY != None or stacker != None

            if stacker:
                # we alrady have a pretrained stacker built
                self.stacker = stacker

            else:
                # otherwise, improvise, any simple model will do
                self.stacker = LogisticRegression() #tree.DecisionTreeClassifier() #RandomForestClassifier()

                # get each model's probabilistic input
                
                # test models to see if output is valid before training the stacker
                to_remove = []
                for j, model in enumerate(self.models):
                    output = model.predict(trainingX, input_type=input_type, probability=True)
                    if np.isnan(output).any():
                    	to_remove.append(model)
                
                # remove models that output NaNs
                for model in to_remove:
                	self.models.remove(model)
                
                # now create stacking matrix
                n, m = len(trainingX), len(self.models)
                estimates = np.zeros((n, m))
                for j, model in enumerate(self.models):
                    estimates[:, j] = model.predict(trainingX, input_type=input_type, probability=True)

                # fit the outputs of the ensemble classifiers to the original labels, 
                # forming a model of these outputs => original labeling
                #print "Stacker (%d models) is fitting %s to %s" % (len(self.models), estimates, trainingY)
                self.stacker.fit(estimates, trainingY)

    def predict(self, examples, input_type=Model.INPUT_CSV, output_type=Model.OUTPUT_ORIGINAL):
        return self.prediction_map[self.ensemble_type](examples, input_type, output_type)

    def vote(self, examples, input_type=Model.INPUT_CSV, output_type=Model.OUTPUT_ORIGINAL):
        """
            Takes all of the models in this predictor and uses them to vote on the correct labels.
            Voting method is plurality. 
        """
        n, m = len(examples), len(self.models)
        votes = np.zeros((n, m), dtype=np.int64)
        final_vote = np.zeros((n,), dtype=np.int)

        # get each model's input
        for j, model in enumerate(self.models):
            votes[:, j] = model.predict(examples, input_type=input_type, output_type=Model.OUTPUT_INT).astype(int)
        
        # take the most prevalent answer as the winner        
        for i in range(n):
            final_vote[i] = np.bincount(votes[i, :]).argmax()

        # convert or leave as ints?
        if output_type == Model.OUTPUT_ORIGINAL:
            final_vote = self.models[0].data.decode_labels(final_vote.astype(int))

        return final_vote

    def stack(self, examples, input_type=Model.INPUT_CSV, output_type=Model.OUTPUT_ORIGINAL):
        """
            Uses a probibabilistic ensemble to generate an output vector for each original 
            training example based on the prediction from each of the classifiers in the ensemble. 

            The stacker, pretrained in setup uses the output signals to predict the correct label.
        """
        n, m = len(examples), len(self.models)
        estimates = np.zeros((n, m))

        # get each model's probabilistic input
        for j, model in enumerate(self.models):
            estimates[:, j] = model.predict(examples, input_type=input_type, probability=True)

        # use the stacked model to predict 
        labels = self.stacker.predict(estimates)

        # convert or leave as ints?
        if output_type == Model.OUTPUT_ORIGINAL:
            labels = self.models[0].data.decode_labels(labels.astype(int))

        return labels