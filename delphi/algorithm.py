"""
.. module:: algorithm
   :synopsis: Wrapper around classification algorithm.

"""

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_curve, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from gdbn.activationFunctions import Softmax, Sigmoid, Linear, Tanh
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, ExpSineSquared, RationalQuadratic
import numpy as np
import time

def delphi_cross_val_binary(pipeline, X, y, cv=10):
    skf = StratifiedKFold(n_splits=cv)
    skf.get_n_splits(X, y)

    accuracies = np.zeros(cv)
    f1_scores = np.zeros(cv)
    pr_curve_aucs = np.zeros(cv)
    roc_curve_aucs = np.zeros(cv)
    pr_curve_precisions = []
    pr_curve_recalls = []
    pr_curve_thresholds = []
    roc_curve_fprs = []
    roc_curve_tprs = []
    roc_curve_thresholds = []

    idx = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_probs = pipeline.predict_proba(X_test)
        y_pred_probs = y_pred_probs[:,1] #get probabilites for positive class (1)

        accuracies[idx] = accuracy_score(y_test, y_pred)
        f1_scores[idx] = f1_score(y_test, y_pred)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs, pos_label=1)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs, pos_label=1)

        pr_curve_aucs[idx] = auc(recall, precision)
        roc_curve_aucs[idx] = auc(fpr,tpr)

        pr_curve_precisions.append(precision)
        pr_curve_recalls.append(recall)
        pr_curve_thresholds.append(thresholds)

        roc_curve_fprs.append(fpr)
        roc_curve_tprs.append(tpr)
        roc_curve_thresholds.append(thresholds)

        idx += 1

    cv_results = dict(accuracies=accuracies, f1_scores=f1_scores, pr_curve_aucs=pr_curve_aucs,
                      roc_curve_aucs=roc_curve_aucs, pr_curve_precisions=pr_curve_precisions,
                      pr_curve_recalls=pr_curve_recalls, pr_curve_thresholds=pr_curve_thresholds,
                      roc_curve_fprs=roc_curve_fprs, roc_curve_tprs=roc_curve_tprs,
                      roc_curve_thresholds=roc_curve_thresholds)

    return cv_results


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
            "cv_acc" : np.mean(self.scores['accuracies']), # backward compatibility
            "cv" : np.mean(self.scores['accuracies']), # backward compatibility
            "stdev" : np.std(self.scores['accuracies']),
            "cv_f1_scores" : self.scores['f1_scores'],
            "cv_pr_curve_aucs" : self.scores['pr_curve_aucs'],
            "cv_roc_curve_aucs" : self.scores['roc_curve_aucs'],
            "cv_pr_curve_precisions" : self.scores['pr_curve_precisions'],
            "cv_pr_curve_recalls" : self.scores['pr_curve_recalls'],
            "cv_pr_curve_thresholds" : self.scores['pr_curve_thresholds'],
            "cv_roc_curve_fprs" : self.scores['roc_curve_fprs'],
            "cv_roc_curve_tprs" : self.scores['roc_curve_tprs'],
            "cv_roc_curve_thresholds" : self.scores['roc_curve_thresholds'],
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
        self.scores = delphi_cross_val_binary(self.pipeline, self.trainX, self.trainY, cv=10)

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

        ### PCA ###
        if "_pca" in learner_params:
            del learner_params["_pca"]
            del learner_params["_pca_dimensions"]

        ### GPC ###
        if learner_params["function"] == "classify_gp":
            if learner_params["kernel"] == "constant":
                learner_params["kernel"] = ConstantKernel()
            elif learner_params["kernel"] == "rbf":
                learner_params["kernel"] = RBF()
            elif learner_params["kernel"] == "matern":
                learner_params["kernel"] = Matern(nu=learner_params["nu"])
                del learner_params["nu"]
            elif learner_params["kernel"] == "rational_quadratic":
                learner_params["kernel"] = RationalQuadratic(length_scale=learner_params["length_scale"], alpha=learner_params["alpha"])
                del learner_params["length_scale"]
                del learner_params["alpha"]
            elif learner_params["kernel"] == "exp_sine_squared":
                learner_params["kernel"] = ExpSineSquared(length_scale=learner_params["length_scale"], periodicity=learner_params["periodicity"])
                del learner_params["length_scale"]
                del learner_params["periodicity"]


        ### MLP ###
        if learner_params["function"] == "classify_mlp":

            learner_params["hidden_layer_sizes"] = []

            # set layer topology
            if int(learner_params["num_hidden_layers"]) == 1:
                learner_params["hidden_layer_sizes"].append(learner_params["hidden_size_layer1"])
                del learner_params["hidden_size_layer1"]

            elif int(learner_params["num_hidden_layers"]) == 2:
                learner_params["hidden_layer_sizes"].append(learner_params["hidden_size_layer1"])
                learner_params["hidden_layer_sizes"].append(learner_params["hidden_size_layer2"])
                del learner_params["hidden_size_layer1"]
                del learner_params["hidden_size_layer2"]

            elif int(learner_params["num_hidden_layers"]) == 3:
                learner_params["hidden_layer_sizes"].append(learner_params["hidden_size_layer1"])
                learner_params["hidden_layer_sizes"].append(learner_params["hidden_size_layer2"])
                learner_params["hidden_layer_sizes"].append(learner_params["hidden_size_layer3"])
                del learner_params["hidden_size_layer1"]
                del learner_params["hidden_size_layer2"]
                del learner_params["hidden_size_layer3"]

            learner_params["hidden_layer_sizes"] = [int(x) for x in learner_params["hidden_layer_sizes"]] # convert to ints

            # delete our fabricated keys
            del learner_params["num_hidden_layers"]

        #print "Added stuff for DBNs! %s" % learner_params
        ### DBN ###
        if learner_params["function"] == "classify_dbn":

                # print "AddING stuff for DBNs! %s" % learner_params

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
                learner_params["layer_sizes"] = [int(x) for x in learner_params["layer_sizes"]]  # convert to ints

                # set activation function
                if learner_params["output_act_funct"] == "Linear":
                    learner_params["output_act_funct"] = Linear()
                elif learner_params["output_act_funct"] == "Sigmoid":
                    learner_params["output_act_funct"] = Sigmoid()
                elif learner_params["output_act_funct"] == "Softmax":
                    learner_params["output_act_funct"] = Softmax()
                elif learner_params["output_act_funct"] == "tanh":
                    learner_params["output_act_funct"] = Tanh()

                learner_params["epochs"] = int(learner_params["epochs"])

                # delete our fabricated keys
                del learner_params["num_hidden_layers"]
                del learner_params["inlayer_size"]
                del learner_params["outlayer_size"]

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
