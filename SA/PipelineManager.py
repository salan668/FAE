"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/7
"""
import os
import csv
import numpy as np
import pandas as pd

from SA.CrossValidation import CrossValidation

from SA.Utility import MakeFolder, Metric, MakeFile
from SA.Utility.Constant import *
from SA.Utility.Index2Dict import Index2Dict
from HomeUI.VersionConstant import *
from Utility.EcLog import eclog


class PipelineManager(object):
    def __init__(self, eclog=eclog):
        self.normalizers = None
        self.reducers = None
        self.feature_selectors = None
        self.feature_numbers = None
        self.fitters = None
        self.cv = None
        self.total_num = 0
        self.mylog = eclog

        self.metric = Metric()
        self.result = {}
        self.interp_times = np.array([])

        self.version = VERSION

    def SetNormalizers(self, normalizers: list):
        self.normalizers = normalizers
    def SetReducers(self, reducers: list):
        self.reducers = reducers
    def SetFeatureSelectors(self, feature_selectors: list):
        self.feature_selectors = feature_selectors
    def SetFeatureNumbers(self, feature_numbers: list):
        self.feature_numbers = feature_numbers
    def SetFitters(self, fitters: list):
        self.fitters = fitters
    def SetCV(self, cv):
        self.cv = cv

    def _Merge2Frame(self, surv: pd.DataFrame, dc, group_index=None):
        surv.index = ['{:4f}'.format(ind) for ind in self.interp_times.tolist()]
        merge = pd.concat((surv,
                          pd.DataFrame(dc.event).T.rename({'status': dc.event_name}),
                          pd.DataFrame(dc.duration).T.rename({'status': dc.duration_name})),
                         axis=0)
        if group_index is not None:
            merge.columns = ['{}-CV{}'.format(name, group_index) for name in merge.columns]

        return merge

    def SurvivalSave(self, df, store_path=None):
        if store_path:
            if not os.path.exists(store_path):
                df.to_csv(store_path)
            else:
                temp = pd.read_csv(store_path, index_col=0)
                temp = pd.concat((temp, df), axis=1)
                temp.to_csv(store_path)

    def SurvivalLoad(self, store_path, event_name=None, duration_name=None):
        df = pd.read_csv(store_path, index_col=0)
        if event_name is None:
            event_name = df.index[-2]
        if duration_name is None:
            duration_name = df.index[-1]

        event = df.loc[event_name, :]
        duration = df.loc[duration_name, :]
        df.drop(index=[event_name, duration_name], inplace=True)

        df.index = list(map(float, df.index))
        return df, event, duration

    def _Estimate(self, surv, event, duration, pipeline_name, data_name):
        self.metric.Run(surv, event, duration)
        one_df = pd.DataFrame(self.metric.text_result, index=[pipeline_name])
        self.result[data_name] = pd.concat((self.result[data_name], one_df), axis=0)

    def GetPipelineName(self, modules):
        return '_'.join(modules)

    def SaveInfo(self, store_path):
        with open(os.path.join(store_path, 'pipeline_info.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([VERSION_NAME, self.version])
            writer.writerow([CROSS_VALIDATION, self.cv.k])
            writer.writerow([NORMALIZER] + [one.GetName() for one in self.normalizers])
            writer.writerow([DIMENSION_REDUCTION] + [one.GetName() for one in self.reducers])
            writer.writerow([FEATURE_SELECTOR] + [one.GetName() for one in self.feature_selectors])
            writer.writerow([FEATURE_NUMBER] + self.feature_numbers)
            writer.writerow([FITTER] + [one.name for one in self.fitters])

    def LoadInfo(self, store_folder):
        index_2_dict = Index2Dict()
        info_path = os.path.join(store_folder, 'pipeline_info.csv')
        if not os.path.exists(info_path):
            return False

        with open(info_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == VERSION_NAME:
                    self.version = row[1]
                    if self.version not in ACCEPT_VERSION:
                        return False
                elif row[0] == CROSS_VALIDATION:
                    self.cv = CrossValidation(k=row[1])
                elif row[0] == NORMALIZER:
                    self.normalizers = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == DIMENSION_REDUCTION:
                    self.reducers = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == FEATURE_SELECTOR:
                    self.feature_selectors = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == FEATURE_NUMBER:
                    self.feature_numbers = row[1:]
                elif row[0] == FITTER:
                    self.fitters = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                else:
                    self.mylog.error('Unknow key name {} when loading pipeline info {}'.format(row[0], info_path))
                    raise KeyError
        return True

    def LoadOneDf(self, store_folder, key):
        file_path = MakeFile(store_folder, 'result-{}.csv'.format(key))
        if os.path.exists(file_path):
            self.result[key] = pd.read_csv(file_path, index_col=0)

    def LoadResult(self, store_folder: str):
        if not os.path.isdir(store_folder):
            self.mylog.error('Wrong folder path when loading Result of PipelineManager')
            raise OSError

        self.LoadOneDf(store_folder, CV_TRAIN)
        self.LoadOneDf(store_folder, CV_VAL)
        self.LoadOneDf(store_folder, TRAIN)
        self.LoadOneDf(store_folder, TEST)

        return True and self.LoadInfo(store_folder)

    def RunWithoutCV(self, train_dc, test_dc=None, store_folder=None):
        self.total_num = len(self.normalizers) * \
                         len(self.reducers) * \
                         len(self.feature_selectors) * \
                         len(self.fitters) * \
                         len(self.feature_numbers)

        if len(self.interp_times) == 0:
            self.mylog.error('The interp time does not initialized by CV.')
            raise ValueError

        if not os.path.isdir(store_folder):
            os.mkdir(store_folder)

        num = 0
        self.result[TRAIN], self.result[TEST] = pd.DataFrame(), pd.DataFrame()
        for norm_index, normalizer in enumerate(self.normalizers):
            norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
            normalizer.Fit(train_dc)
            norm_train_dc = normalizer.Transform(train_dc, norm_store_folder, TRAIN)
            if test_dc is not None and not test_dc.IsEmpty():
                norm_test_dc = normalizer.Transform(test_dc, norm_store_folder, TEST)

            for dr_index, reducer in enumerate(self.reducers):
                reduce_store_folder = MakeFolder(norm_store_folder, reducer.GetName())
                reducer.Fit(norm_train_dc)
                dr_train_dc = reducer.Transform(norm_train_dc, reduce_store_folder, TRAIN)
                if test_dc is not None and not test_dc.IsEmpty():
                    dr_test_dc = reducer.Transform(norm_test_dc, reduce_store_folder, TEST)

                for fs_index, feature_selector in enumerate(self.feature_selectors):
                    for fn_index, feature_number in enumerate(self.feature_numbers):
                        fs_store_folder = MakeFolder(reduce_store_folder, '{}_{}'.format(
                            feature_selector.name, feature_number))

                        feature_selector.selected_number = feature_number
                        feature_selector.Fit(dr_train_dc)
                        fs_train_dc = feature_selector.Transform(dr_train_dc, store_folder=fs_store_folder,
                                                                 store_key=TRAIN)
                        if test_dc is not None and not test_dc.IsEmpty():
                            fs_test_dc = feature_selector.Transform(dr_test_dc, store_folder=fs_store_folder,
                                                                    store_key=TEST)

                        for fitter_index, fitter in enumerate(self.fitters):
                            fitter_store_folder = MakeFolder(fs_store_folder, fitter.name)

                            pipeline_name = self.GetPipelineName([
                                normalizer.name, reducer.name, feature_selector.name,
                                str(feature_number), fitter.name
                            ])

                            fitter.Fit(fs_train_dc)
                            fitter.Save(fitter_store_folder)

                            train_surv = fitter.fitter.predict_survival_function(fs_train_dc.df,
                                                                                    times=self.interp_times)
                            train_result = self._Merge2Frame(train_surv, fs_train_dc)
                            self.SurvivalSave(train_result,
                                              MakeFile(fitter_store_folder, '{}.csv'.format(TRAIN)))
                            train_surv.index = list(map(float, train_surv.index))
                            self._Estimate(train_surv,
                                           train_dc.event.values.tolist(), train_dc.duration.values.tolist(),
                                           pipeline_name, TRAIN)

                            if test_dc is not None and not test_dc.IsEmpty():
                                test_surv = fitter.fitter.predict_survival_function(fs_test_dc.df,
                                                                                      times=self.interp_times)
                                test_result = self._Merge2Frame(test_surv, fs_test_dc)
                                self.SurvivalSave(test_result, MakeFile(fitter_store_folder, '{}.csv'.format(TEST)))
                                test_surv.index = list(map(float, test_surv.index))
                                self._Estimate(test_surv,
                                               test_dc.event.values.tolist(), test_dc.duration.values.tolist(),
                                               pipeline_name, TEST)

                            num += 1
                            yield self.total_num, num

        self.result[TRAIN].to_csv(MakeFile(store_folder, 'result-{}.csv'.format(TRAIN)))
        if test_dc is not None and not test_dc.IsEmpty():
            self.result[TEST].to_csv(MakeFile(store_folder, 'result-{}.csv'.format(TEST)))

    def RunCV(self, dc, store_folder=None):
        self.total_num = len(self.normalizers) * \
                         len(self.reducers) * \
                         len(self.feature_selectors) * \
                         len(self.fitters) * \
                         len(self.feature_numbers)

        self.interp_times = np.linspace(min(dc.duration), max(dc.duration), 100)
        if not os.path.isdir(store_folder):
            os.mkdir(store_folder)

        for group_index, (cv_train_dc, cv_val_dc) in enumerate(self.cv.Generate(dc)):
            num = 0
            for norm_index, normalizer in enumerate(self.normalizers):
                norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
                normalizer.Fit(cv_train_dc)
                norm_cv_train_dc = normalizer.Transform(cv_train_dc)
                norm_cv_val_dc = normalizer.Transform(cv_val_dc)

                for dr_index, reducer in enumerate(self.reducers):
                    reduce_store_folder = MakeFolder(norm_store_folder, reducer.GetName())
                    reducer.Fit(norm_cv_train_dc)
                    dr_cv_train_dc = reducer.Transform(norm_cv_train_dc)
                    dr_cv_val_dc = reducer.Transform(norm_cv_val_dc)

                    for fs_index, feature_selector in enumerate(self.feature_selectors):
                        for fn_index, feature_number in enumerate(self.feature_numbers):
                            fs_store_folder = MakeFolder(reduce_store_folder, '{}_{}'.format(
                                feature_selector.name, feature_number))

                            feature_selector.selected_number = feature_number
                            feature_selector.Fit(dr_cv_train_dc)
                            fs_cv_train_dc = feature_selector.Transform(dr_cv_train_dc)
                            fs_cv_val_dc = feature_selector.Transform(dr_cv_val_dc)

                            for fitter_index, fitter in enumerate(self.fitters):
                                fitter_store_folder = MakeFolder(fs_store_folder, fitter.name)
                                fitter.Fit(fs_cv_train_dc)

                                cv_train_surv = fitter.fitter.predict_survival_function(fs_cv_train_dc.df,
                                                                                        times=self.interp_times)
                                cv_train_result = self._Merge2Frame(cv_train_surv, fs_cv_train_dc, group_index=group_index)
                                self.SurvivalSave(cv_train_result, MakeFile(fitter_store_folder, '{}.csv'.format(CV_TRAIN)))

                                cv_val_surv = fitter.fitter.predict_survival_function(fs_cv_val_dc.df,
                                                                                      times=self.interp_times)
                                cv_val_result = self._Merge2Frame(cv_val_surv, fs_cv_val_dc, group_index=group_index)
                                self.SurvivalSave(cv_val_result, MakeFile(fitter_store_folder, '{}.csv'.format(CV_VAL)))

                                num += 1
                                yield self.total_num, num, group_index + 1

    def EstimateCV(self, store_folder, event_name, duration_name):
        self.result[CV_TRAIN], self.result[CV_VAL] = pd.DataFrame(), pd.DataFrame()

        num = 0
        for norm_index, normalizer in enumerate(self.normalizers):
            norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
            for dr_index, reducer in enumerate(self.reducers):
                dr_store_folder = MakeFolder(norm_store_folder, reducer.GetName())
                for fs_index, feature_selector in enumerate(self.feature_selectors):
                    for fn_index, feature_number in enumerate(self.feature_numbers):
                        fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(
                            feature_selector.name, feature_number))
                        for fitter_index, fitter in enumerate(self.fitters):
                            fitter_store_folder = MakeFolder(fs_store_folder, fitter.name)

                            pipeline_name = self.GetPipelineName([
                                normalizer.name, reducer.name, feature_selector.name,
                                str(feature_number), fitter.name
                            ])

                            surv, event, duration = self.SurvivalLoad(MakeFile(
                                fitter_store_folder, '{}.csv'.format(CV_TRAIN)), event_name, duration_name)
                            self._Estimate(surv, event, duration, pipeline_name, CV_TRAIN)

                            surv, event, duration = self.SurvivalLoad(MakeFile(
                                fitter_store_folder, '{}.csv'.format(CV_VAL)), event_name, duration_name)
                            self._Estimate(surv, event, duration, pipeline_name, CV_VAL)

                            num += 1
                            yield self.total_num, num

        self.result[CV_TRAIN].to_csv(MakeFile(store_folder, 'result-{}.csv'.format(CV_TRAIN)))
        self.result[CV_VAL].to_csv(MakeFile(store_folder, 'result-{}.csv'.format(CV_VAL)))


if __name__ == '__main__':
    from SA.Normalizer import NormalizerMinMax, NormalizerZscore, NormalizerMean
    from SA.DimensionReducer import DimensionReducerPcc
    from SA.FeatureSelector import FeatureSelectorCluster, FeatureSelectorAll
    from SA.Fitter import CoxPH

    pipeline = PipelineManager()
    pipeline.SetNormalizers([NormalizerMinMax, NormalizerZscore, NormalizerMean])
    pipeline.SetReducers([DimensionReducerPcc()])
    pipeline.SetFeatureSelectors([FeatureSelectorCluster, FeatureSelectorAll])
    pipeline.SetFeatureNumbers(list(np.arange(1, 11)))
    pipeline.SetFitters([CoxPH()])
    pipeline.SetCV(CrossValidation())
    pipeline.SaveInfo(r'..\..\Demo\Result2')

    # train_dc = DataContainer()
    # train_dc.Load(r'..\..\Demo\train.csv', event_name='status', duration_name='time')
    #
    # test_dc = DataContainer()
    # test_dc.Load(r'..\..\Demo\test.csv', event_name='status', duration_name='time')
    #
    # store_folder = r'..\..\Demo\Result'
    # pipeline.RunCV(train_dc, store_folder=store_folder)
    # pipeline.EstimateCV(store_folder, train_dc.event_name, train_dc.duration_name)
    # # pipeline.interp_times = np.linspace(min(train_dc.duration), max(train_dc.duration), 100)
    # pipeline.RunWithoutCV(train_dc, test_dc, store_folder)


