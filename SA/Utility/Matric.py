"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/18
"""
import numpy as np
import random
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score

from SA.Utility.Constant import *


class Metric(object):
    def __init__(self, bootstrap_n=100):
        self.result = {}
        self.text_result = {}
        self.bootstrap_n = bootstrap_n

    @staticmethod
    def _summarize_metric(values):
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]

        if finite_values.size == 0:
            return np.nan, np.nan, (np.nan, np.nan)

        sorted_values = np.sort(finite_values)
        mean_value = float(np.mean(sorted_values))
        std_value = float(np.std(sorted_values))

        low_index = int(np.floor(sorted_values.size * 0.025))
        high_index = int(np.floor(sorted_values.size * 0.975)) - 1 if sorted_values.size > 1 else 0
        high_index = max(high_index, 0)
        ci_value = (float(sorted_values[low_index]), float(sorted_values[high_index]))
        return mean_value, std_value, ci_value

    @staticmethod
    def _format_metric_value(value):
        if np.isnan(value):
            return 'N/A'
        return '{:.3f}'.format(value)

    @staticmethod
    def _format_metric_ci(value):
        if np.isnan(value[0]) or np.isnan(value[1]):
            return 'N/A'
        return '{:.3f}-{:.3f}'.format(*value)

    def Bootstrap(self, surv, event: list, duration: list):
        random.seed(42)  # control reproducibility

        cindex, brier = [], []
        for _ in range(self.bootstrap_n):
            sampled_index = random.choices(range(surv.shape[1]), k=surv.shape[1])

            sampled_surv = surv.iloc[:, sampled_index]
            sampled_event = [event[i] for i in sampled_index]
            sampled_duration = [duration[i] for i in sampled_index]

            if int(sum(sampled_event)) == 0 or int(sum(sampled_event)) == len(sampled_event):
                continue

            risk_scores = -sampled_surv.sum(axis=0).values
            c_idx = concordance_index(sampled_duration, risk_scores, sampled_event)
            cindex.append(c_idx)
            
            min_time = min(sampled_duration)
            max_time = max(sampled_duration)
            if max_time > min_time:
                time_grid = np.linspace(min_time, max_time - 1e-5, 100)
                combined_index = np.union1d(sampled_surv.index, time_grid)
                df_interp = sampled_surv.reindex(combined_index).interpolate(method='index').ffill().bfill().loc[time_grid]
                
                surv_struct = np.array([(bool(e), d) for e, d in zip(sampled_event, sampled_duration)], 
                                       dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
                try:
                    ibs = integrated_brier_score(surv_struct, surv_struct, df_interp.T.values, time_grid)
                    brier.append(ibs)
                except Exception:
                    brier.append(np.nan)
            else:
                brier.append(np.nan)

        return cindex, brier

    def Run(self, surv, event: list, duration: list):
        cindex, brier = self.Bootstrap(surv, event, duration)

        self.result[METRIC_CINDEX], self.result[METRIC_CINDEX_STD], self.result[METRIC_CINDEX_CI] = \
            self._summarize_metric(cindex)
        self.result[METRIC_BRIER], self.result[METRIC_BRIER_STD], self.result[METRIC_BRIER_CI] = \
            self._summarize_metric(brier)

        self.text_result[METRIC_CINDEX] = self._format_metric_value(self.result[METRIC_CINDEX])
        self.text_result[METRIC_CINDEX_STD] = self._format_metric_value(self.result[METRIC_CINDEX_STD])
        self.text_result[METRIC_CINDEX_CI] = self._format_metric_ci(self.result[METRIC_CINDEX_CI])
        self.text_result[METRIC_BRIER] = self._format_metric_value(self.result[METRIC_BRIER])
        self.text_result[METRIC_BRIER_STD] = self._format_metric_value(self.result[METRIC_BRIER_STD])
        self.text_result[METRIC_BRIER_CI] = self._format_metric_ci(self.result[METRIC_BRIER_CI])

        # ev = EvalSurv(surv, np.array(dc.duration), np.array(dc.event), censor_surv='km')
        # time_grid = np.linspace(dc.duration.min(), dc.duration.max(), 100)
        #
        # self.result[METRIC_CINDEX] = ev.concordance_td('antolini')
        # self.result[METRIC_BRIER] = ev.integrated_brier_score(time_grid)
