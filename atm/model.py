"""
.. module:: wrapper
   :synopsis: Model around classification method.

"""
from __future__ import print_function
import numpy as np
import pandas as pd
import time
import pdb
import re
from importlib import import_module
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, \
                                             ExpSineSquared, RationalQuadratic

from atm.constants import *
from atm.encoder import MetaData, DataEncoder
from atm.method import Method
from atm.metrics import cross_validate_pipeline, test_pipeline


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
                print("*** Using PCA to reduce %d features to %d dimensions" %
                      (self.num_features, self.dimensions))
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
        df, cv_scores = cross_validate_pipeline(pipeline=self.pipeline,
                                                X=X, y=y, binary=binary,
                                                n_folds=self.N_FOLDS)
        self.cv_judgment_metric = np.mean(df[self.judgment_metric])
        self.cv_judgment_metric_stdev = np.std(df[self.judgment_metric])
        self.mu_sigma_judgment_metric = (self.cv_judgment_metric -
                                         2 * self.cv_judgment_metric_stdev)
        return cv_scores

    def test_final_model(self, X, y):
        """
        Test the (already trained) model pipeline on the provided test data
        (X and y). Store the test judgment metric and return the rest of the
        metrics as a hierarchical dictionary.
        """
        # time the prediction
        starttime = time.time()
        y_preds = self.pipeline.predict(X)

        binary = self.num_classes == 2
        test_scores = test_pipeline(self.pipeline, X, y, binary)

        total = time.time() - starttime
        self.avg_prediction_time = total / float(len(y))
        self.test_judgment_metric = test_scores.get(self.judgment_metric)

        return test_scores

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
        self.encoder = DataEncoder(label_column=self.label_column)
        X_train, y_train = self.encoder.fit_transform(train_data)
        X_test, y_test = self.encoder.transform(test_data)

        # create and cross-validate pipeline
        self.make_pipeline()
        cv_scores = self.cross_validate(X_train, y_train)

        # train and test the final model
        self.pipeline.fit(X_train, y_train)
        test_scores = self.test_final_model(X_test, y_test)
        return {'cv': cv_scores, 'test': test_scores}

    def predict(self, data):
        """
        Use the trained encoder and pipeline to transform training data into
        predicted labels
        """
        X, _ = self.encoder.transform(data)
        return self.pipeline.predict(X)

    def special_conversions(self, params):
        """
        TODO: replace this logic with something better
        """
        # create list parameters
        lists = defaultdict(list)
        element_regex = re.compile('(.*)\[(\d)\]')
        for name, param in params.items():
            # look for variables of the form "param_name[1]"
            match = element_regex.match(name)
            if match:
                # name of the list parameter
                lname = match.groups()[0]
                # index of the list item
                index = int(match.groups()[1])
                lists[lname].append((index, param))

                # drop the element parameter from our list
                del params[name]

        for lname, items in lists.items():
            # drop the list size parameter
            del params[lname + '_size']

            # sort the list by index
            params[lname] = [val for idx, val in sorted(items)]

        ## Gaussian process classifier
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

        # return the updated parameter vector
        return params
