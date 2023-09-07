"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/7
"""
import os
import pickle
import random
from abc import abstractmethod
from lifelines import CoxPHFitter, AalenAdditiveFitter

from lifelines.utils.printer import Printer
from lifelines import utils

from SA.Utility import mylog
from SA.DataContainer import DataContainer


class BaseFitter(object):
    def __init__(self, fitter=None, name=None):
        self.fitter = fitter
        self.name = name

    def Fit(self, dc: DataContainer):
        self.fitter.fit(dc.df, duration_col=dc.duration_name, event_col=dc.event_name)

    def Save(self, store_folder):
        with open(os.path.join(store_folder, 'model.pkl'), 'wb') as f:
            pickle.dump(self.fitter, f)

    def Load(self, store_folder):
        with open(os.path.join(store_folder, 'model.pkl'), 'rb') as f:
            self.fitter = pickle.load(f)

    def Plot(self):
        self.fitter.plot()

    def Summary(self):
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.fitter.print_summary()
        out = f.getvalue()

        return out


class CoxPH(BaseFitter):
    def __init__(self):
        random.seed(0)
        super(CoxPH, self).__init__(CoxPHFitter(), self.__class__.__name__)

    def Fit(self, dc: DataContainer):
        self.fitter.fit(dc.df, duration_col=dc.duration_name, event_col=dc.event_name)


class AalenAdditive(BaseFitter):
    def __init__(self):
        super(AalenAdditive, self).__init__(AalenAdditiveFitter(), self.__class__.__name__)

#
# class Weibull(BaseFitter):
#     def __init__(self):
#         super(Weibull, self).__init__(WeibullAFTFitter(), self.__class__.__name__)


if __name__ == '__main__':
    import numpy as np
    model = CoxPH()
    print(model.name)
    # model = AalenAdditive()
    # print(model.name)

    train_dc = DataContainer()
    train_dc.Load(r'..\..\Demo\train.csv', event_name='status', duration_name='time')
    model.Fit(train_dc)

    result = model.Summary()
    print(result)
    # model.Save(r'..\..\Demo')
    #
    # model_new = AalenAdditive()
    # model_new.Load(r'..\..\Demo')
    # model_new.Summary()
