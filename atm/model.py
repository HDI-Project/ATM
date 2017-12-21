"""
.. module:: wrapper
   :synopsis: Model around classification method.

"""
import numpy as np
import pandas as pd
import time
from importlib import import_module

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from gdbn.activationFunctions import Softmax, Sigmoid, Linear, Tanh
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, \
                                             ExpSineSquared, RationalQuadratic

from atm.constants import *
from atm.encoder import MetaData, DataEncoder
from atm.method import Method
from atm.metrics import cross_validate_pipeline


class Model(object):
    """
    This class contains everything needed to run an end-to-end ATM classifier
    pipeline. It is initialized with a set of parameters and trained like a
    normal sklearn model. This class can be pickled and saved to disk, then
    unpickled outside of ATM and used to classify new datasets.
    """
    # these are special keys that are used for general purpose
    # things like scaling, normalization, PCA, etc
    SCALE = "_scale"
    WHITEN = "_whiten"
    MINMAX = "_scale_minmax"
    PCA = "_pca"
    PCA_DIMS = "_pca_dimensions"

    # list of all such keys
    ATM_KEYS = [SCALE, WHITEN, MINMAX, PCA, PCA_DIMS]

    # number of folds for cross-validation (arbitrary, for speed)
    N_FOLDS = 5

    def __init__(self, code, params, judgment_metric, label_column,
                 testing_ratio=0.3):
        """
        Parameters:
            code: the short method code (as defined in constants.py)
            judgment_metric: string that indicates which metric should be
                optimized for.
            params: parameters passed to the sklearn classifier constructor
            class_: sklearn classifier class
        """
        # configuration & database
        self.code = code
        self.params = params
        self.judgment_metric = judgment_metric
        self.label_column = label_column
        self.testing_ratio = testing_ratio

        # load the classifier method's class
        path = Method(METHODS_MAP[code]).class_path.split('.')
        mod_str, cls_str = '.'.join(path[:-1]), path[-1]
        mod = import_module(mod_str)
        self.class_ = getattr(mod, cls_str)

        # pipelining
        self.pipeline = None

        # persistent random state
        self.random_state = np.random.randint(1e7)

    def load_data(self, path, dropvals=None, sep=','):
        # load data as a Pandas dataframe
        data = pd.read_csv(path, skipinitialspace=True,
                           na_values=dropvals, sep=sep)

        # drop rows with any NA values
        return data.dropna(how='any')

    def make_pipeline(self):
        """
        Makes the classifier as well as scaling or dimension reduction steps.
        """
        # create a list of steps, starting with the data encoder
        steps = []

        # create a classifier with specified parameters
        hyperparameters = {k: v for k, v in self.params.iteritems()
                           if k not in Model.ATM_KEYS}
        atm_params = {k: v for k, v in self.params.iteritems()
                      if k in Model.ATM_KEYS}

        # do special conversions
        hyperparameters = self.special_conversions(hyperparameters)
        self.trainable_params = hyperparameters
        classifier = self.class_(**hyperparameters)

        self.dimensions = self.num_features
        if Model.PCA in atm_params and atm_params[Model.PCA]:
            whiten = (Model.WHITEN in atm_params and
                        atm_params[Model.WHITEN])
            pca_dims = atm_params[Model.PCA_DIMS]
            # PCA dimension in atm_params is a float reprsenting percentages of
            # features to use
            if pca_dims >= 1:
                self.dimensions = int(pca_dims)
            else:
                self.dimensions = int(pca_dims * float(self.num_features))
                print "*** Using PCA to reduce %d features to %d dimensions" %\
                    (self.num_features, self.dimensions)
                pca = decomposition.PCA(n_components=self.dimensions, whiten=whiten)
                steps.append(('pca', pca))

        # should we scale the data?
        if atm_params.get(Model.SCALE):
            steps.append(('standard_scale', StandardScaler()))
        elif self.params.get(Model.MINMAX):
            steps.append(('minmax_scale', MinMaxScaler()))

        # add the classifier as the final step in the pipeline
        steps.append((self.code, classifier))
        self.pipeline = Pipeline(steps)

    def cross_validate(self, X, y):
        binary = self.num_classes == 2
        df, self.cv_scores = cross_validate_pipeline(pipeline=self.pipeline,
                                                     X=X, y=y,
                                                     binary=binary,
                                                     n_folds=self.N_FOLDS)
        self.cv_judgment_metric = np.mean(df[self.judgment_metric])
        self.cv_judgment_metric_std = np.std(df[self.judgment_metric])

    def test_final_model(self, X, y):
        """
        Test the (already trained) model pipeline on the provided test data (X
        and y). Store the resulting metrics in self.test_scores.
        """
        # time the prediction
        starttime = time.time()
        y_preds = self.pipeline.predict(X)
        total = time.time() - starttime
        self.avg_prediction_time = total / float(len(Y))

        binary = self.num_classes == 2
        self.test_scores = test_pipeline(self.pipeline, X, y, binary)
        self.test_judgment_metric = self.test_scores.get(self.judgment_metric)

    def train_test(self, train_path, test_path=None):
        # load train and (maybe) test data
        metadata = MetaData(label_column=self.label_column,
                            train_path=train_path,
                            test_path=test_path)
        self.num_classes = metadata.k_classes
        self.num_features = metadata.d_features

        # if necessary, cast judgment metric into its binary/multiary equivalent
        if self.num_classes == 2:
            if self.judgment_metric in [Metrics.F1_MICRO, Metrics.F1_MACRO]:
                self.judgment_metric = Metrics.F1
            elif self.judgment_metric in [Metrics.ROC_AUC_MICRO,
                                          Metrics.ROC_AUC_MACRO]:
                self.judgment_metric = Metrics.ROC_AUC
        else:
            if self.judgment_metric == Metrics.F1:
                self.judgment_metric = Metrics.F1_MACRO
            elif self.judgment_metric == Metrics.ROC_AUC:
                self.judgment_metric = Metrics.ROC_AUC_MACRO

        # load training data
        train_data = self.load_data(train_path)

        # if necessary, generate permanent train/test split
        if test_path is not None:
            test_data = self.load_data(test_path)
        else:
            train_data, test_data = train_test_split(train_data,
                                                     test_size=self.testing_ratio,
                                                     random_state=self.random_state)

        # extract feature matrix and labels from raw data
        self.encoder = DataEncoder()
        X_train, y_train = self.encoder.fit_transform(train_data)
        X_test, y_test = self.encoder.transform(test_data)

        # create and cross-validate pipeline
        self.make_pipeline()
        self.cross_validate(X_train, y_train)

        # train and test the final model
        self.pipeline.fit(X_train, y_train)
        self.test_final_model(X_test, y_test)

    def special_conversions(self, params):
        """
        TODO: Make this logic into subclasses

        ORRRR, should make each enumerator handle it in a static function
        something like:

        @staticmethod
        def transform_params(params_dict):
            # does classifier-specific changes
            return params_dict

        """
        ### GPC ###
        if self.code == "gp":
            if params["kernel"] == "constant":
                params["kernel"] = ConstantKernel()
            elif params["kernel"] == "rbf":
                params["kernel"] = RBF()
            elif params["kernel"] == "matern":
                params["kernel"] = Matern(nu=params["nu"])
                del params["nu"]
            elif params["kernel"] == "rational_quadratic":
                params["kernel"] = RationalQuadratic(length_scale=params["length_scale"],
                                                             alpha=params["alpha"])
                del params["length_scale"]
                del params["alpha"]
            elif params["kernel"] == "exp_sine_squared":
                params["kernel"] = ExpSineSquared(length_scale=params["length_scale"],
                                                          periodicity=params["periodicity"])
                del params["length_scale"]
                del params["periodicity"]

        ### MLP ###
        if self.code == "mlp":

            params["hidden_layer_sizes"] = []

            # set layer topology
            if int(params["num_hidden_layers"]) == 1:
                params["hidden_layer_sizes"].append(params["hidden_size_layer1"])
                del params["hidden_size_layer1"]

            elif int(params["num_hidden_layers"]) == 2:
                params["hidden_layer_sizes"].append(params["hidden_size_layer1"])
                params["hidden_layer_sizes"].append(params["hidden_size_layer2"])
                del params["hidden_size_layer1"]
                del params["hidden_size_layer2"]

            elif int(params["num_hidden_layers"]) == 3:
                params["hidden_layer_sizes"].append(params["hidden_size_layer1"])
                params["hidden_layer_sizes"].append(params["hidden_size_layer2"])
                params["hidden_layer_sizes"].append(params["hidden_size_layer3"])
                del params["hidden_size_layer1"]
                del params["hidden_size_layer2"]
                del params["hidden_size_layer3"]

            params["hidden_layer_sizes"] = [int(x) for x in
                                                    params["hidden_layer_sizes"]]  # convert to ints

            # delete our fabricated keys
            del params["num_hidden_layers"]

        # print "Added stuff for DBNs! %s" % params
        ### DBN ###
        if self.code == "dbn":

            # print "Adding stuff for DBNs! %s" % params
            params["layer_sizes"] = [params["inlayer_size"]]

            # set layer topology
            if int(params["num_hidden_layers"]) == 1:
                params["layer_sizes"].append(params["hidden_size_layer1"])
                del params["hidden_size_layer1"]

            elif int(params["num_hidden_layers"]) == 2:
                params["layer_sizes"].append(params["hidden_size_layer1"])
                params["layer_sizes"].append(params["hidden_size_layer2"])
                del params["hidden_size_layer1"]
                del params["hidden_size_layer2"]

            elif int(params["num_hidden_layers"]) == 3:
                params["layer_sizes"].append(params["hidden_size_layer1"])
                params["layer_sizes"].append(params["hidden_size_layer2"])
                params["layer_sizes"].append(params["hidden_size_layer3"])
                del params["hidden_size_layer1"]
                del params["hidden_size_layer2"]
                del params["hidden_size_layer3"]

            params["layer_sizes"].append(params["outlayer_size"])
            params["layer_sizes"] = [int(x) for x in params["layer_sizes"]]  # convert to ints

            # set activation function
            if params["output_act_funct"] == "Linear":
                params["output_act_funct"] = Linear()
            elif params["output_act_funct"] == "Sigmoid":
                params["output_act_funct"] = Sigmoid()
            elif params["output_act_funct"] == "Softmax":
                params["output_act_funct"] = Softmax()
            elif params["output_act_funct"] == "tanh":
                params["output_act_funct"] = Tanh()

            params["epochs"] = int(params["epochs"])

            # delete our fabricated keys
            del params["num_hidden_layers"]
            del params["inlayer_size"]
            del params["outlayer_size"]

        # return the updated parameter vector
        return params
