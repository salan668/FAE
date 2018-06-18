import matplotlib.pyplot as plt
import seaborn as sns
color_list = sns.color_palette('deep') + sns.color_palette('bright')

def DrawDoubleYLines(x, y1, y2, xlabel='', ylabel=['', ''], legend=['', ''], store_path=''):
    '''
    Draw the doulbe y-axis lines.
    :param x: The vector of the x axis.
    :param y1: The vector of the y1 axis.
    :param y2: The vector of the y2 axis.
    :param xlabel: The label of the x. Default is ''
    :param ylabel: The list of the y label. Default is ['', '']
    :param legend: The list of the legend. Default is ['', '']
    :param store_path: The store path of the figure. support 'jpg' and 'eps' format.
    :return:
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, color=color_list[0])
    ax1.set_ylabel(ylabel[0])
    ax1.set_xlabel(xlabel)

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, color=color_list[1])
    ax2.set_ylabel(ylabel[1])
    ax2.set_xlabel(xlabel)

    ax1.legend([legend[0]], loc=(.02, .9))
    ax2.legend([legend[1]], loc=(.02, .82))

    if store_path:
        plt.tight_layout()
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')

    plt.show()