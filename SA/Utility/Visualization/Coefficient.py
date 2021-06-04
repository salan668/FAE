"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/21
"""

import matplotlib.pyplot as plt


def ModelHazardRatio(fitter, fig=None, start_ratio=0.1, store_folder=None):
    if fig is None:
        fig = plt.figure()

    fig.clear()
    if start_ratio < 0.1:
        start_ratio = 0.1
    if start_ratio > 0.9:
        start_ratio = 0.1

    ax = fig.add_axes([start_ratio, 0.1, 0.95 - start_ratio, 0.8])
    fitter.fitter.plot(ax=ax)


if __name__ == '__main__':
    from SA.Fitter import CoxPH
    fitter = CoxPH()
    fitter.Load(r'd:\MyCode\SAE\Demo\Result\Mean\PCC\Cluster_6\CoxPH')
    ModelHazardRatio(fitter, start_ratio=0.7)

