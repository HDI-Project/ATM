"""
.. module:: algorithm
   :synopsis: Wrapper around classification algorithm.

"""

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from gdbn.activationFunctions import Softmax, Sigmoid, Linear
from delphi.config import Config
import numpy as np
import time

class Wrapper(object):

    # these are special keys that are used for general purpose
    # things like scaling, normalization, PCA, etc
    FUNCTION = "function"
    SCALE = "_scale"
    PCA = "_pca"
    WHITEN = "_whiten"
    MINMAX = "_scale_minmax"
    PCA_DIMS = "_pca_dimensions"

    # list of all such keys
    DELPHI_KEYS = [
        SCALE, PCA, WHITEN, MINMAX, PCA_DIMS]

    def __init__(self, code, params, learner_class):

        # configuration & database
        self.code = code
        self.params = params
        self.learner_class = learner_class

        # data related
        self.datapath = None
        self.X, self.y = None, None
        self.trainX, self.trainY = None, None
        self.testX, self.testY = None, None
        self.code_content = None

        # pipelining
        self.pipeline = None
        self.steps = []

        # results
        self.scores = []
        self.avg_score = -1

    def start(self):
        self.make_pipeline()
        self.cross_validate()
        self.train_final_model()
        self.prepare_model()
        return self.performance()

    def performance(self):
        self.perf = {
            "testing_acc" : self.test_score, # backward compatibility :(
            "test" : self.test_score, # backward compatibility
            "avg_prediction_time" : self.avg_prediction_time,
            "cv_acc" : np.mean(self.scores), # backward compatibility
            "cv" : np.mean(self.scores), # backward compatibility
            "stdev" : np.std(self.scores),
            "n_folds" : 10,
            "trainable_params": self.trainable_params,
            "testing_confusion" : self.testing_confusion}
        return self.perf

    def load_data_from_objects(self, trainX, testX, trainY, testY):
        self.testX = testX
        self.testY = testY
        self.trainX = trainX
        self.trainY = trainY

    def cross_validate(self):
        self.scores = cross_val_score(self.pipeline, self.trainX, self.trainY, cv=10)

    def train_final_model(self):
        self.pipeline.fit(self.trainX, self.trainY)

        # time the training average
        self.test_score = self.pipeline.score(self.testX, self.testY)

        # time the prediction
        starttime = time.time()
        predictions = self.pipeline.predict(self.testX)
        total = time.time() - starttime
        self.avg_prediction_time = total / float(len(self.testY))

        # make confusion matrix
        self.testing_confusion = confusion_matrix(self.testY, predictions)

    def predict(self, examples, probability=False):
        """
            Examples should be in vectorized format

            Returns integer labels.
        """
        if not probability:
            return self.pipeline.predict(examples)
        else:
            return self.pipeline.predict_proba(examples)

    def prepare_model(self):
        del self.trainX, self.trainY
        del self.testX, self.testY
        del self.datapath

    def special_conversions(self, learner_params):
        """
            TODO: Make this logic into subclasses

            ORRRR, should make each enumerator handle it in a static function
            something like:

            @staticmethod
            def ParamsTransformation(params_dict):
                # does learner-specific changes
                return params_dict

        """
        # do special converstions

        ### RF ###
        if "n_jobs" in learner_params:
            learner_params["n_jobs"] = int(learner_params["n_jobs"])
        if "n_estimators" in learner_params:
            learner_params["n_estimators"] = int(learner_params["n_estimators"])

        ### DT ###
        if "max_features" in learner_params:
            learner_params["max_features"] = int(float(learner_params["max_features"] * self.testX.shape[1]))

        ### PCA ###
        if "_pca" in learner_params:
            del learner_params["_pca"]
            del learner_params["_pca_dimensions"]

        ### DBN ###
        if learner_params["function"] == "classify_dbn":

            #print "AddING stuff for DBNs! %s" % learner_params

            learner_params["layer_sizes"] = [learner_params["inlayer_size"]]

            # set layer topology
            if int(learner_params["num_hidden_layers"]) == 1:
                learner_params["layer_sizes"].append(learner_params["hidden_size_layer1"])
                del learner_params["hidden_size_layer1"]

            elif int(learner_params["num_hidden_layers"]) == 2:
                learner_params["layer_sizes"].append(learner_params["hidden_size_layer1"])
                learner_params["layer_sizes"].append(learner_params["hidden_size_layer2"])
                del learner_params["hidden_size_layer1"]
                del learner_params["hidden_size_layer2"]

            elif int(learner_params["num_hidden_layers"]) == 3:
                learner_params["layer_sizes"].append(learner_params["hidden_size_layer1"])
                learner_params["layer_sizes"].append(learner_params["hidden_size_layer2"])
                learner_params["layer_sizes"].append(learner_params["hidden_size_layer3"])
                del learner_params["hidden_size_layer1"]
                del learner_params["hidden_size_layer2"]
                del learner_params["hidden_size_layer3"]

            learner_params["layer_sizes"].append(learner_params["outlayer_size"])
            learner_params["layer_sizes"] = [int(x) for x in learner_params["layer_sizes"]] # convert to ints

            # set activation function
            if learner_params["output_act_funct"] == "Linear":
                learner_params["output_act_funct"] = Linear()
            elif learner_params["output_act_funct"] == "Sigmoid":
                learner_params["output_act_funct"] = Sigmoid()
            elif learner_params["output_act_funct"] == "Softmax":
                learner_params["output_act_funct"] = Softmax()

            learner_params["epochs"] = int(learner_params["epochs"])

            # delete our fabricated keys
            del learner_params["num_hidden_layers"]
            del learner_params["inlayer_size"]
            del learner_params["outlayer_size"]

            #print "Added stuff for DBNs! %s" % learner_params

        # remove function key and return
        del learner_params["function"]
        return learner_params

    def make_pipeline(self):
        """
            Makes the classifier as well as scaling or
            dimension reduction steps.
        """
        # create a list of steps
        steps = []

        # create a learner with
        learner_params = {k:v for k,v in self.params.iteritems() if k not in Wrapper.DELPHI_KEYS}

        # do special converstions
        learner_params = self.special_conversions(learner_params)
        self.trainable_params = learner_params
        print "Training: %s" % learner_params
        learner = self.learner_class(**learner_params)

        dimensions = None
        if Wrapper.PCA in self.params and self.params[Wrapper.PCA]:
            whiten = False
            if Wrapper.WHITEN in self.params and self.params[Wrapper.WHITEN]:
                whiten = True
            # PCA dimension in our self.params is a float reprsenting percentages of features to use
            percentage_dimensions = float(self.params[Wrapper.PCA_DIMS])
            if percentage_dimensions < 0.99: # if close enough keep all features
                dimensions = int(percentage_dimensions * float(self.testX.shape[1]))
                print "*** Will PCA the data down from %d dimensions to %d" % (self.testX.shape[1], dimensions)
                pca = decomposition.PCA(n_components=dimensions, whiten=whiten)
                steps.append(('pca', pca))

        # keep track of the actual # dimensions we used
        if dimensions:
            self.dimensions = dimensions
        else:
            self.dimensions = self.testX.shape[1]

        # should we scale the data?
        if Wrapper.SCALE in self.params and self.params[Wrapper.SCALE]:
            steps.append(('standard_scale', StandardScaler()))

        elif Wrapper.MINMAX in self.params and self.params[Wrapper.MINMAX]:
            steps.append(('minmax_scale', MinMaxScaler()))

        # add the learner as the final step in the pipeline
        steps.append((self.code, learner))
        self.pipeline = Pipeline(steps)
