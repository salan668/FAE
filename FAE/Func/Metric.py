import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    '''
    This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
    not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
    1000 times. the confidence interval are extracted from the bootstrap result.
    :param y_true: The label, dim should be 1.
    :param y_pred: The prediction, dim should be 1
    :param CI_index: The range of confidence interval. Default is 95%
    :return: The AUC value, a list of the confidence interval, the boot strap result.
    '''

    single_auc = roc_auc_score(y_true, y_pred)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        index_list = list(np.arange(0, len(y_pred), 1))
        # bootstrap by sampling with replacement on the prediction indices
        index_list.pop(int(rng.random_integers(0, len(y_pred) - 1, 1)))

        if len(np.unique(y_true[index_list])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[index_list], y_pred[index_list])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    std_auc = np.std(sorted_scores)
    mean_auc = np.mean(sorted_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]
    # final_auc = (confidence_lower+confidence_upper)/2
    # print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return single_auc, mean_auc, CI, sorted_scores, std_auc

def EstimateMetricCV(cv_info, cv_name, key_word):
    import math

    if cv_name == 'LeaveOneOut':
        # x**2 + x - len(cv_info)+1 = 0
        if key_word == 'train_':
            cv_num = int((math.sqrt(1+4*(len(cv_info) - 1)) + 1) / 2)
        elif key_word == 'val_':
            cv_num = int(len(cv_info) - 1)
        temp_dict_index = [str(index+1) for index in range(cv_num)]

    if cv_name == '5-Folder':
        temp_dict_index = [str(index+1) for index in range(5)]

    if cv_name == '10-Folder':
        temp_dict_index = [str(index+1) for index in range(10)]

    temp_dict = {}
    for cv_index in temp_dict_index:
        temp_dict[cv_index + 'Pred'] = []
        temp_dict[cv_index + 'Label'] = []

    for index in range(len(cv_info)):
        if index > 1:
            ##remove first col
            cv_info_result = cv_info[index]
            cv_index = cv_info_result[1]
            temp_dict[cv_index + 'Pred'].append(cv_info_result[2])
            temp_dict[cv_index + 'Label'].append(cv_info_result[3])

    cv_auc = []
    for index in temp_dict_index:
        # if index == '183':
        pred = temp_dict[index + 'Pred']
        label = temp_dict[index + 'Label']
        try:
            score = roc_auc_score(label, pred)
            cv_auc.append(score)
        except:
            a = index
            print(a)


    return np.std(cv_auc)

def EstimateCVMetirc(cv_info, cv_name, prediction, label, key_word=''):
    '''
    Calculate the medical metric according to prediction and the label.
    :param prediction: The prediction. Dim is 1.
    :param label: The label. Dim is 1
    :param key_word: The word to add in front of the metric key. Usually to separate the training data set, validation
    data set, and the testing data set.
    :return: A dictionary of the calculated metrics
    '''
    if key_word != '':
        key_word += '_'

    metric = {}
    metric[key_word + 'sample_number'] = len(label)
    metric[key_word + 'positive_number'] = np.sum(label)
    metric[key_word + 'negative_number'] = len(label) - np.sum(label)

    fpr, tpr, threshold = roc_curve(label, prediction)
    index = np.argmax(1 - fpr + tpr)
    metric[key_word + 'Youden Index'] = '{:.4f}'.format(threshold[index])

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
    std = EstimateMetricCV(cv_info, cv_name, key_word)
    single_auc, mean_auc, ci, score,_ = AUC_Confidence_Interval(label, prediction)
    metric[key_word + 'auc'] = '{:.4f}'.format(single_auc)
    metric[key_word + 'auc 95% CIs'] = '[{:.4f}-{:.4f}]'.format(ci[0], ci[1])
    metric[key_word + 'auc std'] = '{:.4f}'.format(std)

    return metric

def EstimateMetirc(prediction, label, key_word=''):
    '''
    Calculate the medical metric according to prediction and the label.
    :param prediction: The prediction. Dim is 1.
    :param label: The label. Dim is 1
    :param key_word: The word to add in front of the metric key. Usually to separate the training data set, validation
    data set, and the testing data set.
    :return: A dictionary of the calculated metrics
    '''
    if key_word != '':
        key_word += '_'

    metric = {}
    metric[key_word + 'sample_number'] = len(label)
    metric[key_word + 'positive_number'] = np.sum(label)
    metric[key_word + 'negative_number'] = len(label) - np.sum(label)

    fpr, tpr, threshold = roc_curve(label, prediction)
    index = np.argmax(1 - fpr + tpr)
    metric[key_word + 'Youden Index'] = '{:.4f}'.format(threshold[index])

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

    single_auc, mean_auc, ci, score, std = AUC_Confidence_Interval(label, prediction)
    metric[key_word + 'auc'] = '{:.4f}'.format(single_auc)
    metric[key_word + 'auc 95% CIs'] = '[{:.4f}-{:.4f}]'.format(ci[0], ci[1])
    metric[key_word + 'auc std'] = '{:.4f}'.format(std)

    return metric