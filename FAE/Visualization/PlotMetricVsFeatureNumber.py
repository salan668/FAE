import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
color_list = sns.color_palette('deep') + sns.color_palette('bright')

def DrawCurve(x, y_list, xlabel='', ylabel='', title='', name_list=[], store_path='', is_show=True, fig=plt.figure()):
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

    for index in range(len(y_list)):
        axes.plot(x, y_list[index], color=color_list[index])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
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

    # plt.close(fig)
    return axes

def DrawBar(x_ticks, y_list, ylabel='', title='', name_list=[], store_path='', is_show=True, fig=plt.figure()):
    if not isinstance(y_list, list):
        y_list = [y_list]

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)
    width = 0.3

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

    # plt.close(fig)
    return axes
