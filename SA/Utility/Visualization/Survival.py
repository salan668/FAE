"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/20
"""
import lifelines
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from SA.PipelineManager import PipelineManager

color_list = sns.color_palette('deep') + sns.color_palette('bright')


def SurvivalPlot(surv_list, event_list, duration_list, name_list, legend_list, fig=None,
                 is_show_KM=False, store_folder=None):
    assert(len(surv_list) == len(event_list) and len(surv_list) == len(duration_list))
    if fig is None:
        fig = plt.figure()

    km = lifelines.KaplanMeierFitter()
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)

    for index, (surv_df, event, duration, name, legend) in enumerate(zip(surv_list, event_list, duration_list,
                                                                        name_list, legend_list)):
        if is_show_KM:
            km.fit(duration, event, timeline=surv_df.index)
            ax = km.plot_survival_function(color=color_list[index], ax=ax, ci_show=False, linestyle='--', label='{}-KM'.format(name))
        ax.step(list(surv_df.index), surv_df.values.mean(axis=1),
                color=color_list[index], label=legend)

    ax.legend()
    ax.set_ylabel('Survival Function')
    ax.set_xlabel('Time')

    # plt.show()


if __name__ == '__main__':
    pipeline = PipelineManager()
    train_surv_df, train_event, train_duration = pipeline.SurvivalLoad(
        r'd:\MyCode\SAE\Demo\Result\None\None\SelectAll_1\CoxPH\train.csv',
                                           'status', 'time')
    test_surv_df, test_event, test_duration = pipeline.SurvivalLoad(
        r'd:\MyCode\SAE\Demo\Result\None\None\SelectAll_1\CoxPH\test.csv',
        'status', 'time')

    SurvivalPlot([train_surv_df, test_surv_df],
                 [train_event, test_event],
                 [train_duration, test_duration],
                 ['train', 'test'])

    # km = lifelines.KaplanMeierFitter()
    # km.fit(duration, event, timeline=surv_df.index)
    # km.plot_survival_function()
    # plt.step(surv_df.index, surv_df.values.mean(axis=1))
    # plt.show()

