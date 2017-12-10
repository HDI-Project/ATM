"""
.. module:: algorithm
   :synopsis: Wrapper around classification algorithm.

"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
from gdbn.activationFunctions import Softmax, Sigmoid, Linear, Tanh
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, ExpSineSquared, RationalQuadratic
from atm.metrics import Metrics, JUDGMENT_METRICS
from atm.metrics import get_metrics_binary, get_metrics_small_multiclass, \
                        get_metrics_large_multiclass, atm_cross_val
import numpy as np
import time


def create_wrapper(params, judgment_metric):
    learner_config = LEARNERS_MAP[params["function"]]
    learner_class = Enumerator(learner_config).learner_class
    return Wrapper(params["function"], judgment_metric, params, learner_class)


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
    ATM_KEYS = [SCALE, PCA, WHITEN, MINMAX, PCA_DIMS]

    # number of folds for cross-validation (arbitrary, for speed)
    CV_COUNT = 5

    def __init__(self, code, judgment_metric, params, learner_class,
                 compute_metrics=False):
        """
        Arguments
            code: the short algorithm code (as defined in constants.py)
            judgment_metric: string that has a mapping in
                metrics.JUDGMENT_METRICS and indicates which metric should be
                optimized for.
            params: parameters passed to the sklearn classifier constructor
            learner_class: sklearn classifier class
            compute_metrics: bool indicating whether all metrics should be
                computed (True) or just the judgment metric (False)
        """
        # configuration & database
        self.code = code
        self.judgment_metric = JUDGMENT_METRICS[judgment_metric]
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

    @property
    def performance(self):
        return {
            # judgment metrics (what GP uses to decide next parameter values)
            "cv_judgment_metric":           self.cv_scores['judgment_metric'],
            "cv_judgment_metric_stdev":     self.cv_scores['judgment_metric_std'],
            "test_judgment_metric":         self.test_scores['judgment_metric'],
            # Cross Validation Metrics (split train data with CV and test on each fold)
            "cv_object":                    self.cv_scores,
            # Test Metrics (train on all train data, test on test data)
            "test_object":                  self.test_scores,
            # other info
            "avg_prediction_time":          self.avg_prediction_time,
            "n_folds":                      self.CV_COUNT,
            "trainable_params":             self.trainable_params
        }

    def start(self):
        self.make_pipeline()
        self.cross_validate()
        self.train_final_model()
        self.prepare_model()
        return self.performance

    def load_data_from_objects(self, trainX, testX, trainY, testY):
        self.testX = testX
        self.testY = testY
        self.trainX = trainX
        self.trainY = trainY
        self.num_classes = len(np.unique(self.trainY))
        assert self.num_classes > 1

        # default judgment metric is f1 for binary and mu sigma for multiclass.
        if self.judgment_metric is None:
            if self.num_classes == 2:
                self.judgment_metric = 'f1_scores'
            else:
                self.judgment_metric = 'mu_sigmas'

    def cross_validate(self):
        self.cv_scores = atm_cross_val(self.pipeline, self.trainX, self.trainY,
                                       self.num_classes, self.judgment_metric,
                                       cv=self.CV_COUNT)

    def train_final_model(self):
        self.pipeline.fit(self.trainX, self.trainY)

        # time the prediction
        starttime = time.time()
        y_preds = self.pipeline.predict(self.testX)
        total = time.time() - starttime
        self.avg_prediction_time = total / float(len(self.testY))

        if self.num_classes == 2:
            last_step = self.pipeline.steps[-1]
            if last_step[0] == 'classify_sgd' or last_step[0] == 'classify_pa':
                class_1_distance = self.pipeline.decision_function(self.testX)
                class_0_distance = -class_1_distance

                # this isn't a probability
                y_pred_probs = np.column_stack((class_0_distance, class_1_distance))

            else:
                y_pred_probs = self.pipeline.predict_proba(self.testX)

            results = get_metrics_binary(y_true=self.testY, y_pred=y_preds,
                                         y_pred_probs=y_pred_probs)

            self.test_scores = dict(accuracies=results['accuracy'],
                                    f1_scores=results['f1_score'],
                                    pr_curve_aucs=results['pr_curve_auc'],
                                    cohen_kappas=results['cohen_kappa'],
                                    roc_curve_aucs=results['roc_curve_auc'],
                                    pr_curve_precisions=results['pr_curve_precisions'],
                                    pr_curve_recalls=results['pr_curve_recalls'],
                                    pr_curve_thresholds=results['pr_curve_thresholds'],
                                    roc_curve_fprs=results['roc_curve_fprs'],
                                    roc_curve_tprs=results['roc_curve_tprs'],
                                    roc_curve_thresholds=results['roc_curve_thresholds'],
                                    rank_accuracies=None,
                                    mu_sigmas=None)
            self.test_scores['judgment_metric'] = self.test_scores[self.judgment_metric]

        elif self.num_classes <= 5:
            last_step = self.pipeline.steps[-1]
            if last_step[0] == 'classify_sgd' or last_step[0] == 'classify_pa':
                # this isn't a probability
                y_pred_probs = self.pipeline.decision_function(self.testX)

            else:
                y_pred_probs = self.pipeline.predict_proba(self.testX)


            results = get_metrics_small_multiclass(y_true=self.testY,
                                                   y_pred=y_preds,
                                                   y_pred_probs=y_pred_probs)

            self.test_scores = dict(accuracies=results['accuracy'],
                                    f1_scores=results['label_level_f1_scores'],
                                    f1_score_micros=results['f1_score_micro'],
                                    f1_score_macros=results['f1_score_macro'],
                                    pr_curve_aucs=results['label_level_pr_curve_aucs'],
                                    cohen_kappas=results['cohen_kappa'],
                                    roc_curve_aucs=results['pair_level_roc_curve_aucs'],
                                    pr_curve_precisions=results['label_level_pr_curve_precisions'],
                                    pr_curve_recalls=results['label_level_pr_curve_recalls'],
                                    pr_curve_thresholds=results['label_level_pr_curve_thresholds'],
                                    roc_curve_fprs=results['pair_level_roc_curve_fprs'],
                                    roc_curve_tprs=results['pair_level_roc_curve_tprs'],
                                    roc_curve_thresholds=results['pair_level_roc_curve_thresholds'],
                                    rank_accuracies=None,
                                    mu_sigmas=results['mu_sigma'])
            # TODO: Calculate mu-sigma for f1, accuracy, and roc_auc and make it selectable
            self.test_scores['judgment_metric'] = self.test_scores['mu_sigmas']

        else:
            last_step = self.pipeline.steps[-1]
            if last_step[0] == 'classify_sgd' or last_step[0] == 'classify_pa':
                # this isn't a probability
                y_pred_probs = self.pipeline.decision_function(self.testX)

            else:
                y_pred_probs = self.pipeline.predict_proba(self.testX)

            results = get_metrics_large_multiclass(y_true=self.testY,
                                                   y_pred=y_preds,
                                                   y_pred_probs=y_pred_probs,
                                                   rank=5)

            self.test_scores = dict(accuracies=results['accuracy'],
                                    f1_scores=results['label_level_f1_scores'],
                                    f1_score_micros=results['f1_score_micro'],
                                    f1_score_macros=results['f1_score_macro'],
                                    cohen_kappas=results['cohen_kappa'],
                                    roc_curve_fprs=None, roc_curve_tprs=None,
                                    roc_curve_thresholds=None,
                                    roc_curve_aucs=None,
                                    pr_curve_precisions=None,
                                    pr_curve_recalls=None,
                                    pr_curve_thresholds=None,
                                    pr_curve_aucs=None,
                                    rank_accuracies=results['rank_accuracy'],
                                    mu_sigmas=results['mu_sigma'])
            # TODO: Calculate mu-sigma for f1, accuracy, and roc_auc and make it selectable
            self.test_scores['judgment_metric'] = self.test_scores['mu_sigmas']


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
                learner_params["kernel"] = RationalQuadratic(length_scale=learner_params["length_scale"],
                                                             alpha=learner_params["alpha"])
                del learner_params["length_scale"]
                del learner_params["alpha"]
            elif learner_params["kernel"] == "exp_sine_squared":
                learner_params["kernel"] = ExpSineSquared(length_scale=learner_params["length_scale"],
                                                          periodicity=learner_params["periodicity"])
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

            learner_params["hidden_layer_sizes"] = [int(x) for x in
                                                    learner_params["hidden_layer_sizes"]]  # convert to ints

            # delete our fabricated keys
            del learner_params["num_hidden_layers"]

        # print "Added stuff for DBNs! %s" % learner_params
        ### DBN ###
        if learner_params["function"] == "classify_dbn":

            # print "Adding stuff for DBNs! %s" % learner_params
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

        # create a learner with specified parameters
        learner_params = {k: v for k, v in self.params.iteritems() if k not in
                          Wrapper.ATM_KEYS}

        # do special conversions
        learner_params = self.special_conversions(learner_params)
        self.trainable_params = learner_params
        #print "Training: %s" % learner_params
        learner = self.learner_class(**learner_params)

        dimensions = None
        if Wrapper.PCA in self.params and self.params[Wrapper.PCA]:
            whiten = False
            if Wrapper.WHITEN in self.params and self.params[Wrapper.WHITEN]:
                whiten = True
            # PCA dimension in our self.params is a float reprsenting percentages of features to use
            percentage_dimensions = float(self.params[Wrapper.PCA_DIMS])
            if percentage_dimensions < 0.99:  # if close enough keep all features
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
