from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

color_list = sns.color_palette('deep') + sns.color_palette('bright')

def DrawROCList(pred_list, label_list, name_list='', store_path='', is_show=True):
    '''
    To Draw the ROC curve.
    :param pred_list: The list of the prediction.
    :param label_list: The list of the label.
    :param name_list: The list of the legend name.
    :param store_path: The store path. Support jpg and eps.
    :return: None

    Apr-28-18, Yang SONG [yang.song.91@foxmail.com]
    '''
    if not isinstance(pred_list, list):
        pred_list = [pred_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if not isinstance(name_list, list):
        name_list = [name_list]

    fig = plt.figure()
    for index in range(len(pred_list)):
        fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
        auc = roc_auc_score(label_list[index], pred_list[index])
        name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc)

        plt.plot(fpr, tpr, color=color_list[index], label='ROC curve (AUC = %0.3f)' % auc,linewidth=3)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(name_list, loc="lower right")
    if store_path:
        plt.tight_layout()
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        plt.show()
    plt.close()