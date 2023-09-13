from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import LegendRename

color_list = sns.color_palette('deep') + sns.color_palette('bright')



def DrawROCList(pred_list, label_list, name_list='', store_path='', is_show=True, fig=plt.figure()):
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

    if name_list != ['']:
        name_list = LegendRename(name_list)

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)

    for index in range(len(pred_list)):
        fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
        auc_roc = roc_auc_score(label_list[index], pred_list[index])
        name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc_roc)

        axes.plot(fpr, tpr, color=color_list[index], label='ROC curve (AUC = %0.3f)' % auc_roc, linewidth=3)

    axes.plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_title('Receiver operating characteristic curve')
    axes.legend(name_list, loc="lower right")
    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        plt.show()

    return axes

def DrawPRCurveList(pred_list, label_list, name_list='', store_path='', is_show=True, fig=plt.figure()):
    '''
    To Draw the Precision-Recall curve.
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

    if name_list != ['']:
        name_list = LegendRename(name_list)

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)

    for index in range(len(pred_list)):
        precision, recall, threshold = precision_recall_curve(label_list[index], pred_list[index])
        auc_pr = auc(recall, precision)
        name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc_pr)

        precision, recall = precision.tolist(), recall.tolist()
        precision.insert(0, 0.)
        precision.append(1.)
        recall.insert(0, 1.)
        recall.append(0.)

        axes.plot(recall, precision, color=color_list[index], label='PR-AUC = %0.3f' % auc_pr, linewidth=3)

    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.set_title('Precision-Recall curve')
    axes.legend(name_list, loc="lower left")
    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        plt.show()

    return axes

if __name__ == '__main__':
    pred = np.array([0.2, 0.4, 0.6, 0.8])
    label = np.array([0, 0, 1, 1])
    DrawROCList(pred, label, store_path=r'C:\Users\18\Desktop\tem.jpg')
