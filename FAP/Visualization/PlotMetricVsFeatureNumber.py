import matplotlib.pyplot as plt
import seaborn as sns
color_list = sns.color_palette('deep') + sns.color_palette('bright')

def DrawCurve(x, y_list, xlabel='', ylabel='', title='', name_list=[], store_path='', is_show=True):
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
