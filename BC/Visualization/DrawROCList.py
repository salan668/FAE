from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

color_list = sns.color_palette('deep') + sns.color_palette('bright')


def LegendRename(name_list):
    rename_dict= {'cv_train': 'CV Training', 'cv_val': 'Validation',
                  'balance_train': 'Balance Training',
                  'train': 'Training', 'test': 'Testing'}
    new_name_list = [rename_dict[i] for i in name_list]
    return new_name_list


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

    name_list = LegendRename(name_list)

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)

    for index in range(len(pred_list)):
        fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
        auc = roc_auc_score(label_list[index], pred_list[index])
        name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc)

        axes.plot(fpr, tpr, color=color_list[index], label='ROC curve (AUC = %0.3f)' % auc,linewidth=3)

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

if __name__ == '__main__':
    pred = np.array([0.2, 0.4, 0.6, 0.8])
    label = np.array([0, 0, 1, 1])
    DrawROCList(pred, label, store_path=r'C:\Users\18\Desktop\tem.jpg')
