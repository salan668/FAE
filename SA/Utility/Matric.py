"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/18
"""
import numpy as np
from random import choices
from pycox.evaluation import EvalSurv

from SA.Utility.Constant import *


class Metric(object):
    def __init__(self, bootstrap_n=100):
        self.result = {}
        self.text_result = {}
        self.bootstrap_n = bootstrap_n

    def Bootstrap(self, surv, event: list, duration: list):
        np.random.seed(42)  # control reproducibility

        cindex, brier, nbll = [], [], []
        for _ in range(self.bootstrap_n):
            sampled_index = choices(range(surv.shape[1]), k=surv.shape[1])

            sampled_surv = surv.iloc[:, sampled_index]
            sampled_event = [event[i] for i in sampled_index]
            sampled_duration = [duration[i] for i in sampled_index]

            ev = EvalSurv(sampled_surv, np.array(sampled_duration),
                          np.array(sampled_event).astype(int), censor_surv='km')
            time_grid = np.linspace(min(sampled_duration), max(sampled_duration), 100)

            cindex.append(ev.concordance_td('antolini'))
            brier.append(ev.integrated_brier_score(time_grid))
            nbll.append(ev.integrated_nbll(time_grid))

        return cindex, brier, nbll

    def Run(self, surv, event: list, duration: list):
        cindex, brier, nbll = self.Bootstrap(surv, event, duration)
        self.result[METRIC_CINDEX] = np.mean(cindex)
        self.result[METRIC_CINDEX_STD] = np.std(cindex)
        self.result[METRIC_CINDEX_CI] = (sorted(cindex)[int(np.floor(self.bootstrap_n * 0.025))],
                                         sorted(cindex)[int(np.floor(self.bootstrap_n * 0.975))])
        self.result[METRIC_BRIER] = np.mean(brier)
        self.result[METRIC_BRIER_STD] = np.std(brier)
        self.result[METRIC_BRIER_CI] = (sorted(brier)[int(np.floor(self.bootstrap_n * 0.025))],
                                         sorted(brier)[int(np.floor(self.bootstrap_n * 0.975))])
        self.result[METRIC_NBLL] = np.mean(nbll)
        self.result[METRIC_NBLL_STD] = np.std(nbll)
        self.result[METRIC_NBLL_CI] = (sorted(nbll)[int(np.floor(self.bootstrap_n * 0.025))],
                                         sorted(nbll)[int(np.floor(self.bootstrap_n * 0.975))])

        self.text_result[METRIC_CINDEX] = '{:.3f}'.format(self.result[METRIC_CINDEX])
        self.text_result[METRIC_CINDEX_STD] = '{:.3f}'.format(self.result[METRIC_CINDEX_STD])
        self.text_result[METRIC_CINDEX_CI] = '{:.3f}-{:.3f}'.format(*self.result[METRIC_CINDEX_CI])
        self.text_result[METRIC_BRIER] = '{:.3f}'.format(self.result[METRIC_BRIER])
        self.text_result[METRIC_BRIER_STD] = '{:.3f}'.format(self.result[METRIC_BRIER_STD])
        self.text_result[METRIC_BRIER_CI] = '{:.3f}-{:.3f}'.format(*self.result[METRIC_BRIER_CI])
        self.text_result[METRIC_NBLL] = '{:.3f}'.format(self.result[METRIC_NBLL])
        self.text_result[METRIC_NBLL_STD] = '{:.3f}'.format(self.result[METRIC_NBLL_STD])
        self.text_result[METRIC_NBLL_CI] = '{:.3f}-{:.3f}'.format(*self.result[METRIC_NBLL_CI])

        # ev = EvalSurv(surv, np.array(dc.duration), np.array(dc.event), censor_surv='km')
        # time_grid = np.linspace(dc.duration.min(), dc.duration.max(), 100)
        #
        # self.result[METRIC_CINDEX] = ev.concordance_td('antolini')
        # self.result[METRIC_BRIER] = ev.integrated_brier_score(time_grid)
        # self.result[METRIC_NBLL] = ev.integrated_nbll(time_grid)
