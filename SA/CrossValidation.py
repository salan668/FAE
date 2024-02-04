"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/7
"""
from copy import deepcopy
import numpy as np

from SA.DataContainer import DataContainer


class CrossValidation(object):
    def __init__(self, k=5):
        self.k = k

    def Generate(self, dc: DataContainer):
        df = dc.df.copy()

        df = df.reindex(np.random.default_rng(0).permutation(df.index)).sort_values(dc.event_name)
        assignments = np.array((dc.df.shape[0] // self.k + 1) * list(range(1, self.k + 1)))
        assignments = assignments[:dc.df.shape[0]]

        for i in range(1, self.k + 1):
            ix = assignments == i

            train_dc = deepcopy(dc)
            train_dc.df = df.loc[~ix]
            train_dc.UpdateData()

            val_dc = deepcopy(dc)
            val_dc.df = df.loc[ix]
            val_dc.UpdateData()

            yield train_dc, val_dc


if __name__ == '__main__':
    from SA.Fitter import MyCox

    dc = DataContainer()
    dc.Load(r'C:\Users\yangs\Desktop\Radiomics_pvp_hcc_os_top20_train.csv',
            event_name='status', duration_name='time')

    print(dc)
    fitter = MyCox()
    cv = CrossValidation()
    cv.Generate(dc)

