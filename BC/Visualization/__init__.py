def LegendRename(name_list):
    rename_dict= {'cv_train': 'CV Training', 'cv_val': 'Validation',
                  'balance_train': 'Balance Training',
                  'train': 'Training', 'test': 'Test'}
    new_name_list = [rename_dict[i] for i in name_list]
    return new_name_list

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.calibration import calibration_curve

color_list = sns.color_palette('deep') + sns.color_palette('bright')

def DrawViolinPlot(prediction, label, fig=plt.figure()):
    prediction, label = np.array(prediction), np.array(label)
    positive = prediction[label == 1]
    negative = prediction[label == 0]
    fig.clear()
    axes = fig.add_subplot(1, 1, 1)
    axes.violinplot([negative, positive], showmeans=False, showmedians=False, showextrema=False)
    axes.set_xticks([1, 2])
    axes.set_xticklabels(['Negative', 'Positive'])
    return axes

def DrawCalibrationCurve(prediction, label, fig=plt.figure()):
    F, threshold = calibration_curve(label, prediction, n_bins=100)
    clf_score = metrics.brier_score_loss(label, prediction, pos_label=1)
    fig.clear()
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(threshold, F, "s-", label='{:.3f}'.format(clf_score))
    axes.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes.set_ylabel("Fraction of positives")
    axes.set_ylim([-0.05, 1.05])
    axes.legend(loc="lower right")
    return axes

def DrawBoxPlot(prediction, label, fig=plt.figure()):
    prediction, label = np.array(prediction), np.array(label)
    positive = np.asarray(prediction[label == 1], dtype=object)
    negative = np.asarray(prediction[label == 0], dtype=object)
    fig.clear()
    axes = fig.add_subplot(1, 1, 1)
    axes.boxplot([negative, positive], labels=['Negative', 'Positive'], vert=True)
    return axes


def DrawProbability(prediction, label, cut_off, fig=plt.figure()):
    import pandas as pd
    df = pd.DataFrame({'prob': prediction, 'label': label})
    df = df.sort_values('prob')

    bar_color = [color_list[x] for x in df['label'].values]

    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(len(prediction)), df['prob'].values - cut_off, color=bar_color)
    ax.set_yticks([df['prob'].values.min() - cut_off, cut_off - cut_off, df['prob'].max() - cut_off])
    ax.set_yticklabels(
        ['{:.2f}'.format(df['prob'].values.min()), '{:.2f}'.format(cut_off), '{:.2f}'.format(df['prob'].max())])
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Case Index')
    return ax