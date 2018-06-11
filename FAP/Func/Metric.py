import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    # AUC = roc_auc_score(y_true, y_pred)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]

    AUC = sorted_scores[len(sorted_scores) // 2]

    # print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return AUC, CI, sorted_scores

def EstimateMetirc(prediction, label, key_word=''):
    if key_word != '':
        key_word += '_'

    metric = {}
    metric[key_word + 'sample_number'] = len(label)
    metric[key_word + 'positive_number'] = np.sum(label)
    metric[key_word + 'negative_number'] = len(label) - np.sum(label)


    fpr, tpr, threshold = roc_curve(label, prediction)
    index = np.argmax(1 - fpr + tpr)
    metric[key_word + 'Yorden Index'] = '{:.4f}'.format(threshold[index])

    pred = np.zeros_like(label)
    pred[prediction >= threshold[index]] = 1
    C = confusion_matrix(label, pred, labels=[1, 0])

    metric[key_word + 'accuracy'] = '{:.4f}'.format(np.where(pred == label)[0].size / label.size)
    if np.sum(C[0, :]) < 1e-6:
        metric[key_word + 'sensitivity'] = 0
    else:
        metric[key_word + 'sensitivity'] = '{:.4f}'.format(C[0, 0] / np.sum(C[0, :]))
    if np.sum(C[1, :]) < 1e-6:
        metric[key_word + 'specificity'] = 0
    else:
        metric[key_word + 'specificity'] = '{:.4f}'.format(C[1, 1]/np.sum(C[1, :]))
    if np.sum(C[:, 0]) < 1e-6:
        metric[key_word + 'positive predictive value'] = 0
    else:
        metric[key_word + 'positive predictive value'] = '{:.4f}'.format(C[0, 0]/np.sum(C[:, 0]))
    if np.sum(C[:, 1]) < 1e-6:
        metric[key_word + 'negative predictive value'] = 0
    else:
        metric[key_word + 'negative predictive value'] = '{:.4f}'.format(C[1, 1]/np.sum(C[:, 1]))

    auc, ci, score = AUC_Confidence_Interval(label, prediction)
    metric[key_word + 'auc'] = '{:.4f}'.format(auc)
    metric[key_word + 'auc 95% CIs'] = '[{:.4f}-{:.4f}]'.format(ci[0], ci[1])

    return metric