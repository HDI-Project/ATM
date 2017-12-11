from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_curve,\
                            accuracy_score, cohen_kappa_score, roc_auc_score

import numpy as np
import itertools
import pdb


# these are the strings that are used to index into results dictionaries
class Metrics:
    ACCURACY = 'accuracies'
    F1 = 'f1_scores'
    F1_MICRO = 'f1_score_micros'
    F1_MACRO = 'f1_score_macros'
    F1_MU_SIGMA = 'mu_sigmas'           # mean(f1) - std(f1)
    ROC_AUC = 'roc_curve_aucs'
    ROC_AUC_MICRO = 'roc_curve_auc_micros'
    ROC_AUC_MACRO = 'roc_curve_auc_macros'
    PR_AUC = 'pr_curve_aucs'
    COHEN_KAPPA = 'cohen_kappas'
    RANK_ACCURACY = 'rank_accuracies'   # for large multiclass problems


# these are the human-readable strings used in the config files
JUDGMENT_METRICS = {
    'accuracy': Metrics.ACCURACY,
    'f1': Metrics.F1,
    'f1_micro': Metrics.F1_MICRO,
    'f1_macro': Metrics.F1_MACRO,
    'f1_mu_sigma': Metrics.F1_MU_SIGMA,
    'roc_auc': Metrics.ROC_AUC,
    'roc_auc_micro': Metrics.ROC_AUC_MICRO,
    'roc_auc_macro': Metrics.ROC_AUC_MACRO,
}

METRIC_DEFAULT_SCORES = {
    'accuracy': 0.0,
    'f1': 0.0,
    'f1_micro': 0.0,
    'f1_macro': 0.0,
    'f1_mu_sigma': 0.0,
    'roc_auc': 0.5,
    'roc_auc_micro': 0.5,
    'roc_auc_macro': 0.5,
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
    for pair in itertools.combinations(np.unique(y_true), 2):
        if np.any(np.isnan(y_pred_probs[:, int(pair[1])])):
            fpr = 'nan probabilities, cannot compute roc curve'
            tpr = 'nan probabilities, cannot compute roc curve'
            roc_thresholds = 'nan probabilities, cannot compute roc curve'
            roc_auc = 'nan probabilities, cannot compute roc curve'
        else:
            fpr, tpr, roc_thresholds = roc_curve(y_true=y_true,
                                                 y_score=y_pred_probs[:, int(pair[1])],
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
    f1_score_micro = f1_score(y_true, y_pred, average='micro')
    f1_score_macro = f1_score(y_true, y_pred, average='macro')

    #n_classes = len(np.unique(y_true))
    #ohe = OneHotEncoder()
    #y_true_mc = ohe.fit_transform(y_true)
    #y_pred_mc = ohe.transform(y_pred)
    #fpr, tpr, roc_auc = {}, {}, {}
    #for i in range(n_classes):
		#fpr[i], tpr[i], _ = roc_curve(y_true_mc[:, i], y_pred_mc[:, i])
		#roc_auc[i] = auc(fpr[i], tpr[i])

    ## Compute micro-average ROC curve and ROC area
    #fpr_micro, tpr_micro, _ = roc_curve(y_true_mc.ravel(), y_pred_mc.ravel())
    #roc_curve_auc_micro = auc(fpr_micro, tpr_micro)
    #all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #mean_tpr = np.zeros_like(all_fpr)
    #for i in range(n_classes):
        #mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    ## Finally average it and compute AUC
    #mean_tpr /= n_classes
    #roc_curve_auc_macro = auc(all_fpr, mean_tpr)

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
                   f1_score_micro=f1_score_micro,
                   f1_score_macro=f1_score_macro,
                   #roc_curve_auc_micro=roc_curve_auc_micro,
                   #roc_curve_auc_macro=roc_curve_auc_macro,
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

    rank_accuracy = rank_n_accuracy(y_true=y_true,
                                    y_prob_mat=y_pred_probs,
                                    rank=rank)

    f1_score_micro = f1_score(y_true, y_pred, average='micro')
    f1_score_macro = f1_score(y_true, y_pred, average='macro')

    #n_classes = len(np.unique(y_true))
    #ohe = OneHotEncoder()
    #y_true_mc = ohe.fit_transform(y_true)
    #y_pred_mc = ohe.transform(y_pred)
    #fpr, tpr, roc_auc = {}, {}, {}
    #for i in range(n_classes):
		#fpr[i], tpr[i], _ = roc_curve(y_true_mc[:, i], y_pred_mc[:, i])
		#roc_auc[i] = auc(fpr[i], tpr[i])

    ## Compute micro-average ROC curve and ROC area
    #fpr_micro, tpr_micro, _ = roc_curve(y_true_mc.ravel(), y_pred_mc.ravel())
    #roc_curve_auc_micro = auc(fpr_micro, tpr_micro)
    #all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #mean_tpr = np.zeros_like(all_fpr)
    #for i in range(n_classes):
        #mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    ## Finally, average it and compute AUC
    #mean_tpr /= n_classes
    #roc_curve_auc_macro = auc(all_fpr, mean_tpr)

    results = dict(accuracy=accuracy,
                   cohen_kappa=cohen_kappa,
                   f1_score_micro=f1_score_micro,
                   f1_score_macro=f1_score_macro,
                   #roc_curve_auc_micro=roc_curve_auc_micro,
                   #roc_curve_auc_macro=roc_curve_auc_macro,
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

        results = get_metrics_binary(y_pred=y_pred,
                                     y_true=y_test,
                                     y_pred_probs=y_pred_probs)

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
    f1_score_micros = np.zeros(cv)
    f1_score_macros = np.zeros(cv)
    roc_curve_fprs = []
    roc_curve_tprs = []
    roc_curve_thresholds = []
    roc_curve_aucs = []
    roc_curve_auc_micros = np.zeros(cv)
    roc_curve_auc_macros = np.zeros(cv)
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


        results = get_metrics_small_multiclass(y_true=y_test, y_pred=y_pred,
                                               y_pred_probs=y_pred_probs)

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
        f1_score_micros[split_id] = results['f1_score_micro']
        f1_score_macros[split_id] = results['f1_score_macro']
        #roc_curve_auc_micros[split_id] = results['roc_curve_auc_micro']
        #roc_curve_auc_macros[split_id] = results['roc_curve_auc_macro']

        split_id += 1

    cv_results = dict(accuracies=accuracies,
                      f1_scores=f1_scores,
                      f1_score_micros=f1_score_micros,
                      f1_score_macros=f1_score_macros,
                      pr_curve_aucs=pr_curve_aucs,
                      cohen_kappas=cohen_kappas,
                      roc_curve_aucs=roc_curve_aucs,
                      #roc_curve_auc_micros=roc_curve_auc_micros,
                      #roc_curve_auc_macros=roc_curve_auc_macros,
                      pr_curve_precisions=pr_curve_precisions,
                      pr_curve_recalls=pr_curve_recalls,
                      pr_curve_thresholds=pr_curve_thresholds,
                      roc_curve_fprs=roc_curve_fprs,
                      roc_curve_tprs=roc_curve_tprs,
                      roc_curve_thresholds=roc_curve_thresholds,
                      rank_accuracies=rank_accuracies,
                      mu_sigmas=mu_sigmas)

    # TODO: Calculate mu-sigma for f1, accuracy, and roc_auc and make it selectable
    cv_results['judgment_metric'] = np.mean(cv_results['mu_sigmas'])
    cv_results['judgment_metric_std'] = np.std(cv_results['mu_sigmas'])

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
    f1_score_micros = np.zeros(cv)
    f1_score_macros = np.zeros(cv)
    roc_curve_fprs = None
    roc_curve_tprs = None
    roc_curve_thresholds = None
    roc_curve_aucs = None
    roc_curve_auc_micros = np.zeros(cv)
    roc_curve_auc_macros = np.zeros(cv)
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

        results = get_metrics_large_multiclass(y_true=y_test, y_pred=y_pred,
                                               y_pred_probs=y_pred_probs,
                                               rank=rank)

        f1_scores.append((split_id, results['label_level_f1_scores']))
        mu_sigmas[split_id] = results['mu_sigma']
        accuracies[split_id] = results['accuracy']
        rank_accuracies[split_id] = results['rank_accuracy']
        f1_score_micros[split_id] = results['f1_score_micro']
        f1_score_macros[split_id] = results['f1_score_macro']
        #roc_curve_auc_micros[split_id] = results['roc_curve_auc_micro']
        #roc_curve_auc_macros[split_id] = results['roc_curve_auc_macro']

        split_id += 1

    cv_results = dict(accuracies=accuracies,
                      f1_scores=f1_scores,
                      f1_score_micros=f1_score_micros,
                      f1_score_macros=f1_score_macros,
                      pr_curve_aucs=pr_curve_aucs,
                      cohen_kappas=cohen_kappas,
                      roc_curve_aucs=roc_curve_aucs,
                      #roc_curve_auc_micros=roc_curve_auc_micros,
                      #roc_curve_auc_macros=roc_curve_auc_macros,
                      pr_curve_precisions=pr_curve_precisions,
                      pr_curve_recalls=pr_curve_recalls,
                      pr_curve_thresholds=pr_curve_thresholds,
                      roc_curve_fprs=roc_curve_fprs,
                      roc_curve_tprs=roc_curve_tprs,
                      roc_curve_thresholds=roc_curve_thresholds,
                      rank_accuracies=rank_accuracies,
                      mu_sigmas=mu_sigmas)

    # assert judgment_metric in ['f1_score_micros', 'f1_score_macros', 'mu_sigmas']
    # TODO: Calculate mu-sigma for f1, accuracy, and roc_auc and make it selectable
    cv_results['judgment_metric'] = np.mean(cv_results['mu_sigmas'])
    cv_results['judgment_metric_std'] = np.std(cv_results['mu_sigmas'])

    return cv_results


def atm_cross_val(pipeline, X, y, num_classes, judgment_metric, cv=10):
    if num_classes == 2:
        return atm_cross_val_binary(pipeline, X, y, judgment_metric, cv=cv)
    elif (num_classes >= 3) and (num_classes <= 5):
        return atm_cross_val_small_multiclass(pipeline, X, y, judgment_metric, cv=cv)
    else:
        return atm_cross_val_large_multiclass(pipeline, X, y, judgment_metric, cv=cv)


