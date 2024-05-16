"""
All rights reserved.
--Yang Song, Apr 8th, 2020.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, matthews_corrcoef, classification_report, precision_recall_curve, auc

from BC.Func.DelongAUC import CalculateAUC
from BC.Utility.Constants import *

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.calibration import calibration_curve

def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    '''
    This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
    not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
    1000 times. the confidence interval are extracted from the bootstrap result.

    Ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2820000515%2919%3A9%3C1141%3A%3AAID-SIM479%3E3.0.CO%3B2-F
    :param y_true: The label, dim should be 1.
    :param y_pred: The prediction, dim should be 1
    :param CI_index: The range of confidence interval. Default is 95%
    :return: The AUC value, a list of the confidence interval, the boot strap result.
    '''

    single_auc = roc_auc_score(y_true, y_pred)

    bootstrapped_scores = []

    np.random.seed(42) # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(label_one_sample, pred_one_sample)
        bootstrapped_scores.append(score)

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

def EstimatePrediction(prediction, label, key_word='', cutoff=None):
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
    metric[key_word + NUMBER] = len(label)
    metric[key_word + POS_NUM] = np.sum(label)
    metric[key_word + NEG_NUM] = len(label) - np.sum(label)

    precision, recall, _ = precision_recall_curve(label, prediction)
    metric[key_word + AUC_PR] = '{:.4f}'.format(auc(recall, precision))

    pred = np.zeros_like(label)
    if cutoff is None:
        fpr, tpr, threshold = roc_curve(label, prediction)
        index = np.argmax(1 - fpr + tpr)
        metric[key_word + CUTOFF] = '{:.4f}'.format(threshold[index])
        pred[prediction >= threshold[index]] = 1
    else:
        metric[key_word + CUTOFF] = '{:.4f}'.format(cutoff)
        pred[prediction >= cutoff] = 1

    metric[key_word + ACC] = '{:.4f}'.format(np.where(pred == label)[0].size / label.size)
    metric[key_word + MCC] = '{:.4f}'.format(matthews_corrcoef(label, pred))

    report = classification_report(label, pred, digits=4, output_dict=True)
    metric[key_word + SEN] = '{:.4f}'.format(report['1']['recall'])
    metric[key_word + SPE] = '{:.4f}'.format(report['0']['recall'])
    metric[key_word + PPV] = '{:.4f}'.format(report['1']['precision'])
    metric[key_word + NPV] = '{:.4f}'.format(report['0']['precision'])

    metric[key_word + YI] = '{:.4f}'.format(report['1']['recall'] + report['0']['recall'] - 1)

    roc_auc, std, ci = CalculateAUC(label, np.asarray(prediction))
    metric[key_word + AUC] = '{:.4f}'.format(roc_auc)
    metric[key_word + AUC_CI] = '[{:.4f}-{:.4f}]'.format(ci[0], ci[1])
    metric[key_word + AUC_STD] = '{:.4f}'.format(std)

    return metric
