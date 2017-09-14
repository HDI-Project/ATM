"""
.. module:: algorithm
   :synopsis: Wrapper around classification algorithm.

"""

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_curve, accuracy_score, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
from gdbn.activationFunctions import Softmax, Sigmoid, Linear, Tanh
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, ExpSineSquared, RationalQuadratic
import numpy as np
import time
import itertools


# these are the strings that are used to index into results dictionaries
class Metrics:
    ACCURACY = 'accuracies'
    F1 = 'f1_scores'
    F1_MICRO = 'f1_scores_micro'
    F1_MACRO = 'f1_scores_macro'
    F1_MU_SIGMA = 'mu_sigmas'           # mean(f1) - std(f1)
    ROC_AUC = 'roc_curve_aucs'
    ROC_AUC_MICRO = 'roc_curve_aucs_micro'
    ROC_AUC_MACRO = 'roc_curve_aucs_macro'
    PR_AUC = 'pr_curve_aucs'
    COHEN_KAPPA = 'cohen_kappas'
    RANK_ACCURACY = 'rank_accuracies'   # for large multiclass problems


# these are the human-readable strings used in the config files
JUDGMENT_METRICS = {
    'accuracy': Metrics.ACCURACY,
    'f1': Metrics.F1,
    'f1_micro': Metrics.F1_MICRO,
    'f1_macro': Metrics.F1_MACRO,
    'roc_auc': Metrics.ROC_AUC,
    'roc_auc_micro': Metrics.ROC_AUC_MICRO,
    'roc_auc_macro': Metrics.ROC_AUC_MACRO,
}


def get_metrics_binary(y_true, y_pred, y_pred_probs):
    y_pred_prob = y_pred_probs[:, 1]  # get probabilites for positive class (1)

    accuracy = accuracy_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)

    _f1_score = f1_score(y_true, y_pred) # _ in front of name to differentiate it from f1_score() function

    if np.any(np.isnan(y_pred_probs)):
        pr_precisions = 'nan probabilities, cannot compute pr curve'
        pr_recalls = 'nan probabilities, cannot compute pr curve'
        pr_thresholds = 'nan probabilities, cannot compute pr curve'
        roc_fprs = 'nan probabilities, cannot compute roc curve'
        roc_tprs = 'nan probabilities, cannot compute roc curve'
        roc_thresholds = 'nan probabilities, cannot compute roc curve'
        pr_curve_auc = 'nan probabilities, cannot compute pr curve'
        roc_curve_auc = 'nan probabilities, cannot compute roc curve'
    else:
        pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(y_true, y_pred_prob, pos_label=1)
        roc_fprs, roc_tprs, roc_thresholds = roc_curve(y_true, y_pred_prob, pos_label=1)

        pr_curve_auc = auc(pr_recalls, pr_precisions)
        roc_curve_auc = auc(roc_fprs, roc_tprs)

    results = dict(accuracy=accuracy,
                   cohen_kappa=cohen_kappa,
                   f1_score=_f1_score,
                   pr_curve_precisions=pr_precisions,
                   pr_curve_recalls=pr_recalls,
                   pr_curve_thresholds=pr_thresholds,
                   pr_curve_auc=pr_curve_auc,
                   roc_curve_fprs=roc_fprs,
                   roc_curve_tprs=roc_tprs,
                   roc_curve_thresholds=roc_thresholds,
                   roc_curve_auc=roc_curve_auc)

    return results


def get_metrics_small_multiclass(y_true, y_pred, y_pred_probs):
    accuracy = accuracy_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)

    pair_level_roc_curve_fprs = []
    pair_level_roc_curve_tprs = []
    pair_level_roc_curve_thresholds = []
    pair_level_roc_curve_aucs = []

    # for each pair, generate roc curve (positive class is the larger class in the pair, i.e., pair[1])
    for pair in itertools.combinations(y_true, 2):
        if np.any(np.isnan(y_pred_probs[:, int(pair[1])])):
            fpr = 'nan probabilities, cannot compute roc curve'
            tpr = 'nan probabilities, cannot compute roc curve'
            roc_thresholds = 'nan probabilities, cannot compute roc curve'
            roc_auc = 'nan probabilities, cannot compute roc curve'
        else:
            fpr, tpr, roc_thresholds = roc_curve(y_true=y_true, y_score=y_pred_probs[:, int(pair[1])],
                                                 pos_label=pair[1])
            roc_auc = auc(fpr, tpr)

        pair_level_roc_curve_fprs.append((pair, fpr))
        pair_level_roc_curve_tprs.append((pair, tpr))
        pair_level_roc_curve_thresholds.append((pair, roc_thresholds))
        pair_level_roc_curve_aucs.append((pair, roc_auc))

    label_level_f1_scores = []
    label_level_pr_curve_precisions = []
    label_level_pr_curve_recalls = []
    label_level_pr_curve_thresholds = []
    label_level_pr_curve_aucs = []
    f1_scores_vec = np.zeros(len(np.unique(y_true)))

    # for each label, generate F1 and precision-recall curve
    counter = 0
    for label in np.nditer(np.unique(y_true)):
        # set label as positive class, and all other classes as negative class
        y_true_temp = (y_true == label).astype(int)
        y_pred_temp = (y_pred == label).astype(int)
        f1_score_val = f1_score(y_true=y_true_temp, y_pred=y_pred_temp, pos_label=1)

        if np.any(np.isnan(y_pred_probs[:, int(pair[1])])):
            precision = 'nan probabilities, cannot compute pr curve'
            recall = 'nan probabilities, cannot compute pr curve'
            pr_thresholds = 'nan probabilities, cannot compute pr curve'
            pr_auc = 'nan probabilities, cannot compute pr curve'
        else:
            precision, recall, pr_thresholds = precision_recall_curve(y_true=y_true,
                                                                      probas_pred=y_pred_probs[:, int(label)],
                                                                      pos_label=1)
            pr_auc = auc(recall, precision)

        f1_scores_vec[counter] = f1_score_val
        label_level_f1_scores.append((label, f1_score_val))
        label_level_pr_curve_precisions.append((label, precision))
        label_level_pr_curve_recalls.append((label, recall))
        label_level_pr_curve_thresholds.append((label, pr_thresholds))
        label_level_pr_curve_aucs.append((label, pr_auc))

        counter += 1

    mu_sigma = np.mean(f1_scores_vec) - np.std(f1_scores_vec)

    results = dict(accuracy=accuracy,
                   cohen_kappa=cohen_kappa,
                   label_level_f1_scores=label_level_f1_scores,
                   pair_level_roc_curve_fprs=pair_level_roc_curve_fprs,
                   pair_level_roc_curve_tprs=pair_level_roc_curve_tprs,
                   pair_level_roc_curve_thresholds=pair_level_roc_curve_thresholds,
                   pair_level_roc_curve_aucs=pair_level_roc_curve_aucs,
                   label_level_pr_curve_precisions=label_level_pr_curve_precisions,
                   label_level_pr_curve_recalls=label_level_pr_curve_recalls,
                   label_level_pr_curve_thresholds=label_level_pr_curve_thresholds,
                   label_level_pr_curve_aucs=label_level_pr_curve_aucs,
                   mu_sigma=mu_sigma)

    return results


def get_metrics_large_multiclass(y_true, y_pred, y_pred_probs, rank):
    label_level_f1_scores = []
    f1_scores_vec = np.zeros(len(np.unique(y_true)))

    # for each label, generate F1
    counter = 0
    for label in np.nditer(np.unique(y_true)):
        y_true_temp = (y_true == label).astype(int)
        y_pred_temp = (y_pred == label).astype(int)
        f1_score_val = f1_score(y_true=y_true_temp, y_pred=y_pred_temp, pos_label=1)

        f1_scores_vec[counter] = f1_score_val
        label_level_f1_scores.append((label, f1_score_val))

        counter += 1

    mu_sigma = np.mean(f1_scores_vec) - np.std(f1_scores_vec)

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)

    rank_accuracy = rank_n_accuracy(y_true=y_true, y_prob_mat=y_pred_probs, rank=rank)

    results = dict(accuracy=accuracy,
                   cohen_kappa=cohen_kappa,
                   label_level_f1_scores=label_level_f1_scores,
                   mu_sigma=mu_sigma,
                   rank_accuracy=rank_accuracy)

    return results


def atm_cross_val_binary(pipeline, X, y, judgment_metric, cv=10):
    skf = StratifiedKFold(n_splits=cv)
    skf.get_n_splits(X, y)

    accuracies = np.zeros(cv)
    cohen_kappas = np.zeros(cv)
    f1_scores = np.zeros(cv)
    roc_curve_fprs = []
    roc_curve_tprs = []
    roc_curve_thresholds = []
    roc_curve_aucs = np.zeros(cv)
    pr_curve_precisions = []
    pr_curve_recalls = []
    pr_curve_thresholds = []
    pr_curve_aucs = np.zeros(cv)
    rank_accuracies = None
    mu_sigmas = None

    split_id = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        last_step = pipeline.steps[-1]
        if last_step[0] == 'classify_sgd' or last_step[0] == 'classify_pa':
            class_1_distance = pipeline.decision_function(X_test)
            class_0_distance = -class_1_distance

            # this isn't a probability
            y_pred_probs = np.column_stack((class_0_distance, class_1_distance))

        else:
            y_pred_probs = pipeline.predict_proba(X_test)

        results = get_metrics_binary(y_pred=y_pred, y_true=y_test, y_pred_probs=y_pred_probs)

        accuracies[split_id] = results['accuracy']
        cohen_kappas[split_id] = results['cohen_kappa']

        f1_scores[split_id] = results['f1_score']

        pr_curve_precisions.append(results['pr_curve_precisions'])
        pr_curve_recalls.append(results['pr_curve_recalls'])
        pr_curve_thresholds.append(results['pr_curve_thresholds'])

        if results['pr_curve_auc'] == 'nan probabilities, cannot compute pr curve':
            pr_curve_aucs[split_id] = np.nan
        else:
            pr_curve_aucs[split_id] = results['pr_curve_auc']

        roc_curve_fprs.append(results['roc_curve_fprs'])
        roc_curve_tprs.append(results['roc_curve_tprs'])
        roc_curve_thresholds.append(results['roc_curve_thresholds'])

        if results['roc_curve_auc'] == 'nan probabilities, cannot compute roc curve':
            roc_curve_aucs[split_id] = np.nan
        else:
            roc_curve_aucs[split_id] = results['roc_curve_auc']

        split_id += 1

    cv_results = dict(accuracies=accuracies,
                      f1_scores=f1_scores,
                      pr_curve_aucs=pr_curve_aucs,
                      cohen_kappas=cohen_kappas,
                      roc_curve_aucs=roc_curve_aucs,
                      pr_curve_precisions=pr_curve_precisions,
                      pr_curve_recalls=pr_curve_recalls,
                      pr_curve_thresholds=pr_curve_thresholds,
                      roc_curve_fprs=roc_curve_fprs,
                      roc_curve_tprs=roc_curve_tprs,
                      roc_curve_thresholds=roc_curve_thresholds,
                      rank_accuracies=rank_accuracies,
                      mu_sigmas=mu_sigmas)

    cv_results['judgment_metric'] = np.mean(cv_results[judgment_metric])
    cv_results['judgment_metric_std'] = np.std(cv_results[judgment_metric])

    return cv_results


def atm_cross_val_small_multiclass(pipeline, X, y, judgment_metric, cv=10):
    skf = StratifiedKFold(n_splits=cv)
    skf.get_n_splits(X, y)

    accuracies = np.zeros(cv)
    cohen_kappas = np.zeros(cv)
    f1_scores = []
    roc_curve_fprs = []
    roc_curve_tprs = []
    roc_curve_thresholds = []
    roc_curve_aucs = []
    pr_curve_precisions = []
    pr_curve_recalls = []
    pr_curve_thresholds = []
    pr_curve_aucs = []
    rank_accuracies = None
    mu_sigmas = np.zeros(cv)

    split_id = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        last_step = pipeline.steps[-1]
        if last_step[0] == 'classify_sgd' or last_step[0] == 'classify_pa':
            # this isn't a probability
            y_pred_probs = pipeline.decision_function(X_test)

        else:
            y_pred_probs = pipeline.predict_proba(X_test)


        results = get_metrics_small_multiclass(y_true=y_test, y_pred=y_pred, y_pred_probs=y_pred_probs)

        roc_curve_fprs.append((split_id, results['pair_level_roc_curve_fprs']))
        roc_curve_tprs.append((split_id, results['pair_level_roc_curve_tprs']))
        roc_curve_thresholds.append((split_id, results['pair_level_roc_curve_thresholds']))
        roc_curve_aucs.append((split_id, results['pair_level_roc_curve_aucs']))

        f1_scores.append((split_id, results['label_level_f1_scores']))
        pr_curve_precisions.append((split_id, results['label_level_pr_curve_precisions']))
        pr_curve_recalls.append((split_id, results['label_level_pr_curve_recalls']))
        pr_curve_thresholds.append((split_id, results['label_level_pr_curve_thresholds']))
        pr_curve_aucs.append((split_id, results['label_level_pr_curve_aucs']))
        accuracies[split_id] = results['accuracy']
        cohen_kappas[split_id] = results['cohen_kappa']
        mu_sigmas[split_id] = results['mu_sigma']

        split_id += 1

    cv_results = dict(accuracies=accuracies,
                      f1_scores=f1_scores,
                      pr_curve_aucs=pr_curve_aucs,
                      cohen_kappas=cohen_kappas,
                      roc_curve_aucs=roc_curve_aucs,
                      pr_curve_precisions=pr_curve_precisions,
                      pr_curve_recalls=pr_curve_recalls,
                      pr_curve_thresholds=pr_curve_thresholds,
                      roc_curve_fprs=roc_curve_fprs,
                      roc_curve_tprs=roc_curve_tprs,
                      roc_curve_thresholds=roc_curve_thresholds,
                      rank_accuracies=rank_accuracies,
                      mu_sigmas=mu_sigmas)

    cv_results['judgment_metric'] = np.mean(cv_results[judgment_metric])
    cv_results['judgment_metric_std'] = np.std(cv_results[judgment_metric])

    return cv_results


def rank_n_accuracy(y_true, y_prob_mat, rank=5):
    rankings = np.argsort(-y_prob_mat) # negative because we want highest value first
    rankings = rankings[:, 0:rank-1]

    num_samples = len(y_true)
    correct_sample_count = 0.0

    for i in range(num_samples):
        if y_true[i] in rankings[i, :]:
            correct_sample_count += 1.0

    return correct_sample_count / num_samples


def atm_cross_val_large_multiclass(pipeline, X, y, judgment_metric, cv=10, rank=5):
    skf = StratifiedKFold(n_splits=cv)
    skf.get_n_splits(X, y)

    # f1_scores = []
    # rank_accuracies = np.zeros(cv)
    # accuracies = np.zeros(cv)
    # mu_sigma = np.zeros(cv)

    accuracies = np.zeros(cv)
    cohen_kappas = np.zeros(cv)
    f1_scores = []
    roc_curve_fprs = None
    roc_curve_tprs = None
    roc_curve_thresholds = None
    roc_curve_aucs = None
    pr_curve_precisions = None
    pr_curve_recalls = None
    pr_curve_thresholds = None
    pr_curve_aucs = None
    rank_accuracies = np.zeros(cv)
    mu_sigmas = np.zeros(cv)

    split_id = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        last_step = pipeline.steps[-1]
        if last_step[0] == 'classify_sgd' or last_step[0] == 'classify_pa':
            # this isn't a probability
            y_pred_probs = pipeline.decision_function(X_test)

        else:
            y_pred_probs = pipeline.predict_proba(X_test)

        results = get_metrics_large_multiclass(y_true=y_test, y_pred=y_pred, y_pred_probs=y_pred_probs, rank=rank)

        f1_scores.append((split_id, results['label_level_f1_scores']))
        mu_sigmas[split_id] =results['mu_sigma']
        accuracies[split_id] = results['accuracy']
        rank_accuracies[split_id] = results['rank_accuracy']

        split_id += 1

    cv_results = dict(accuracies=accuracies,
                      f1_scores=f1_scores,
                      pr_curve_aucs=pr_curve_aucs,
                      cohen_kappas=cohen_kappas,
                      roc_curve_aucs=roc_curve_aucs,
                      pr_curve_precisions=pr_curve_precisions,
                      pr_curve_recalls=pr_curve_recalls,
                      pr_curve_thresholds=pr_curve_thresholds,
                      roc_curve_fprs=roc_curve_fprs,
                      roc_curve_tprs=roc_curve_tprs,
                      roc_curve_thresholds=roc_curve_thresholds,
                      rank_accuracies=rank_accuracies,
                      mu_sigmas=mu_sigmas)

    cv_results['judgment_metric'] = np.mean(cv_results[judgment_metric])
    cv_results['judgment_metric_std'] = np.std(cv_results[judgment_metric])

    return cv_results


def atm_cross_val(pipeline, X, y, num_classes, judgment_metric, cv=10):
    if num_classes == 2:
        return atm_cross_val_binary(pipeline, X, y, judgment_metric, cv=cv)
    elif (num_classes >= 3) and (num_classes <= 5):
        return atm_cross_val_small_multiclass(pipeline, X, y, judgment_metric, cv=cv)
    else:
        return atm_cross_val_large_multiclass(pipeline, X, y, judgment_metric, cv=cv)


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
    ATM_KEYS = [
        SCALE, PCA, WHITEN, MINMAX, PCA_DIMS]

    # number of folds for cross-validation (arbitrary, for speed)
    CV_COUNT = 5

    def __init__(self, code, judgment_metric, params, learner_class):
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

    def start(self):
        self.make_pipeline()
        self.cross_validate()
        self.train_final_model()
        self.prepare_model()
        return self.performance()

    def performance(self):
        self.perf = {
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

        return self.perf

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
                                    cohen_kappas=results['cohen_kappa'],
                                    f1_scores=results['f1_score'],
                                    roc_curve_fprs=results['roc_curve_fprs'],
                                    roc_curve_tprs=results['roc_curve_tprs'],
                                    roc_curve_thresholds=results['roc_curve_thresholds'],
                                    roc_curve_aucs=results['roc_curve_auc'],
                                    pr_curve_precisions=results['pr_curve_precisions'],
                                    pr_curve_recalls=results['pr_curve_recalls'],
                                    pr_curve_thresholds=results['pr_curve_thresholds'],
                                    pr_curve_aucs=results['pr_curve_auc'],
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
                                    cohen_kappas=results['cohen_kappa'],
                                    f1_scores=results['label_level_f1_scores'],
                                    roc_curve_fprs=results['pair_level_roc_curve_fprs'],
                                    roc_curve_tprs=results['pair_level_roc_curve_tprs'],
                                    roc_curve_thresholds=results['pair_level_roc_curve_thresholds'],
                                    roc_curve_aucs=results['pair_level_roc_curve_aucs'],
                                    pr_curve_precisions=results['label_level_pr_curve_precisions'],
                                    pr_curve_recalls=results['label_level_pr_curve_recalls'],
                                    pr_curve_thresholds=results['label_level_pr_curve_thresholds'],
                                    pr_curve_aucs=results['label_level_pr_curve_aucs'],
                                    rank_accuracies=None,
                                    mu_sigmas=results['mu_sigma'])
            self.test_scores['judgment_metric'] = self.test_scores[self.judgment_metric]

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
                                    cohen_kappas=results['cohen_kappa'],
                                    f1_scores=results['label_level_f1_scores'],
                                    roc_curve_fprs=None, roc_curve_tprs=None,
                                    roc_curve_thresholds=None,
                                    roc_curve_aucs=None,
                                    pr_curve_precisions=None,
                                    pr_curve_recalls=None,
                                    pr_curve_thresholds=None,
                                    pr_curve_aucs=None,
                                    rank_accuracies=results['rank_accuracy'],
                                    mu_sigmas=results['mu_sigma'])
            self.test_scores['judgment_metric'] = self.test_scores[self.judgment_metric]


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

        # create a learner with specified parameters
        learner_params = {k: v for k, v in self.params.iteritems() if k not in
                          Wrapper.ATM_KEYS}

        # do special converstions
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
