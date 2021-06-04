"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/2/1
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

color_list = sns.color_palette('deep') + sns.color_palette('bright')


def DrawIndex(x, ys: list, name_list: list, fig=None, is_show=False):
    if fig is None:
        fig = plt.figure()
    assert(isinstance(ys, list))
    assert(len(ys) == len(name_list))

    fig.clear()
    ax = fig.add_subplot(1, 1, 1)

    for y, color, name in zip(ys, color_list, name_list):
        ax.plot(x, y, color=color, label=name)

    ax.set_xlabel('Feature Number')
    ax.set_xticks(list(np.arange(int(x[0]), int(x[-1]), len(x) // 5)))
    ax.set_ylabel('C-Index')
    ax.legend()

    if is_show:
        plt.show()


if __name__ == '__main__':
    x = np.arange(1, 21)
    y1 = np.linspace(0.5, 0.9, 20)
    y2 = np.linspace(0.9, 0.6, 20)

    DrawIndex(x, [y1, y2], name_list=['a', 'b'], is_show=True)
