"""
All rights reserved.
--Yang Song
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Utility.Constants import CV_VAL

color_list = sns.color_palette('deep') + sns.color_palette('bright')

def DrawCurve(x, y_list, std_list=[], xlabel='', ylabel='', title='', name_list=[], store_path='',
              one_se=False, is_show=True, fig=plt.figure()):
    '''
    Draw the curve like ROC
    :param x: the vector of the x
    :param y_list: the list of y vectors. Each of the vector should has same length with x
    :param xlabel: the name of the y axis
    :param ylabel: the name of the x axis
    :param title: the tile of the figure,
    :param name_list: the legend name list corresponding to y list
    :param store_path: the store path, supporting jpg and esp format
    :param is_show: Boolen, if it was set to True, the figure would show.
    :return:
    '''
    if not isinstance(y_list, list):
        y_list = [y_list]

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)

    if std_list == []:
        for index in range(len(y_list)):
            axes.plot(x, y_list[index], color=color_list[index])
        axes.set_xticks(np.linspace(1, len(x), len(x)))
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)
        if name_list != []:
            axes.legend(name_list, loc=4)
    else:

        for index in range(len(y_list)):
            sub_y_list = y_list[index]
            sub_std_list = std_list[index]
            if name_list[index] == CV_VAL:
                axes.errorbar(x, sub_y_list, yerr=sub_std_list, fmt='-o',
                              color=color_list[index], elinewidth=2, capsize=4, alpha=0.7, marker='.', label='CV Validation')
                if one_se:
                    sub_y_list = y_list[index]
                    sub_std_list = std_list[index]
                    sub_one_se = max(sub_y_list) - sub_std_list[sub_y_list.index(max(sub_y_list))]
                    line = np.ones((1, len(x))) * sub_one_se
                    line_list = line.tolist()

                    for index in range(len(sub_y_list)):
                        if sub_y_list[index] >= sub_one_se:
                            best_auc_value = sub_y_list[index]
                            best_auc_feature_number = index + 1

                            axes.plot(x, line_list[0], color='orange', linewidth=1, linestyle="--")
                            axes.plot(best_auc_feature_number, best_auc_value, 'H', linewidth=20, color='black')
                            break

                else:
                    axes.plot(np.argmax(sub_y_list) + 1, np.max(sub_y_list), 'H', linewidth=20, color='black')
                    best_auc_feature_number = np.argmax(sub_y_list) + 1

            else:
                axes.plot(x, y_list[index], color=color_list[index], label=name_list[index])

            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.set_title(title)
            if name_list != []:
                axes.legend(loc=4)


        if len(x) < 21:
            axes.set_xticks(np.linspace(1, len(x), len(x)))
        else:
            sub_ticks_list = list(np.arange(0, len(x)+1, len(x)/5))
            sub_ticks_list[0] = 1
            sub_ticks_list.append(best_auc_feature_number)
            for delete_index in [best_auc_feature_number-1, best_auc_feature_number-2,
                                 best_auc_feature_number+1, best_auc_feature_number+2]:
                if delete_index in sub_ticks_list:
                    sub_ticks_list.remove(delete_index)
            axes.set_xticks(sorted(sub_ticks_list))

    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')
    if is_show:
        axes.show()

    return axes

def DrawBar(x_ticks, y_list, ylabel='', title='', name_list=[], store_path='', is_show=True, fig=plt.figure()):
    if not isinstance(y_list, list):
        y_list = [y_list]

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)
    width = 0.2

    x = np.arange(len(x_ticks))
    for index in range(len(y_list)):
        axes.bar(x + width * index, y_list[index], width, color=color_list[index])

    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.set_xticks(x + width * (len(y_list) - 1) / 2)
    axes.set_xticklabels(x_ticks)

    if name_list != []:
        axes.legend(name_list, loc=4)

    if store_path:
        # plt.tight_layout()
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')
    if is_show:
        axes.show()

    return axes
