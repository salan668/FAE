import matplotlib.pyplot as plt
import seaborn as sns
color_list = sns.color_palette('deep') + sns.color_palette('bright')

def DrawCurve(x, y_list, xlabel='', ylabel='', title='', name_list=[], store_path='', is_show=True):
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

    fig = plt.figure()
    for index in range(len(y_list)):
        plt.plot(x, y_list[index], color=color_list[index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if name_list != []:
        plt.legend(name_list)

    if store_path:
        plt.tight_layout()
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')
    if is_show:
        plt.show()

    plt.close(fig)
