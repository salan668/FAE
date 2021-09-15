"""
All rights reserved.
--Yang SONG, Jun 17, 2018.
"""

import os
import pickle
import pandas as pd
import csv
import numpy as np
from copy import deepcopy

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.IndexDict import Index2Dict
from BC.FeatureAnalysis.Normalizer import NormalizerNone
from BC.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC
from BC.Func.Metric import EstimatePrediction

from BC.Utility.PathManager import MakeFolder
from BC.Utility.Constants import *

from VersionConstant import *


class PipelinesManager(object):
    def __init__(self, balancer=None, normalizer_list=[NormalizerNone],
                 dimension_reduction_list=[DimensionReductionByPCC()], feature_selector_list=[],
                 feature_selector_num_list=[], classifier_list=[],
                 cross_validation=None, is_hyper_parameter=False, logger=None):
        self.balance = balancer
        self.normalizer_list = normalizer_list
        self.dimension_reduction_list = dimension_reduction_list
        self.feature_selector_list = feature_selector_list
        self.feature_selector_num_list = feature_selector_num_list
        self.classifier_list = classifier_list
        self.cv = cross_validation
        self.is_hyper_parameter = is_hyper_parameter
        self.__logger = logger
        self.version = VERSION

        self.GenerateAucDict()

    def SaveAll(self, store_folder):
        self.SaveAucDict(store_folder)
        self.SavePipelineInfo(store_folder)

    def SavePipelineInfo(self, store_folder):
        with open(os.path.join(store_folder, 'pipeline_info.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([VERSION_NAME, self.version])
            writer.writerow([CROSS_VALIDATION, self.cv.GetName()])
            writer.writerow([BALANCE, self.balance.GetName()])
            writer.writerow([NORMALIZR] + [one.GetName() for one in self.normalizer_list])
            writer.writerow([DIMENSION_REDUCTION] + [one.GetName() for one in self.dimension_reduction_list])
            writer.writerow([FEATURE_SELECTOR] + [one.GetName() for one in self.feature_selector_list])
            writer.writerow([FEATURE_NUMBER] + self.feature_selector_num_list)
            writer.writerow([CLASSIFIER] + [one.GetName() for one in self.classifier_list])

    def SaveAucDict(self, store_folder):
        with open(os.path.join(store_folder, 'auc_metric.pkl'), 'wb') as file:
            pickle.dump(self.__auc_dict, file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(store_folder, 'auc_std_metric.pkl'), 'wb') as file:
            pickle.dump(self.__auc_std_dict, file, pickle.HIGHEST_PROTOCOL)

    def LoadAll(self, store_folder):
        return self.LoadAucDict(store_folder) and self.LoadPipelineInfo(store_folder)

    def GetRealFeatureNum(self, store_folder):
        files = os.listdir(store_folder)
        for file in files:
            if file.find('_features.csv') != -1:
                feature_file = os.path.join(store_folder, file)
                pdf = pd.read_csv(feature_file)
                return len(pdf.columns) - 1
        return 0

    def LoadPipelineInfo(self, store_folder):
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
                    self.cv = index_2_dict.GetInstantByIndex(row[1])
                elif row[0] == BALANCE:
                    self.balance = index_2_dict.GetInstantByIndex(row[1])
                elif row[0] == NORMALIZR:
                    self.normalizer_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == DIMENSION_REDUCTION:
                    self.dimension_reduction_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == FEATURE_SELECTOR:
                    self.feature_selector_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                elif row[0] == FEATURE_NUMBER:
                    feature_number = self.GetRealFeatureNum(store_folder)
                    number = len(row) - 1 if len(row) - 1 > feature_number else feature_number
                    self.feature_selector_num_list = row[1: number]
                elif row[0] == CLASSIFIER:
                    self.classifier_list = [index_2_dict.GetInstantByIndex(index) for index in row[1:]]
                else:
                    print('Unknown name: {}'.format(row[0]))
                    raise KeyError
        return True

    def LoadAucDict(self, store_folder):
        auc_path = os.path.join(store_folder, 'auc_metric.pkl')
        std_path = os.path.join(store_folder, 'auc_std_metric.pkl')
        if not(os.path.exists(auc_path) and os.path.exists(std_path)):
            return False

        with open(auc_path, 'rb') as file:
            self.__auc_dict = pickle.load(file)
        with open(std_path, 'rb') as file:
            self.__auc_std_dict = pickle.load(file)

        return True

    def SaveOneResult(self, pred, label, key_name, case_name, matric_indexs, model_name,
                      store_root='', model_folder=''):
        assert(len(matric_indexs) == 5)
        norm_index, dr_index, fs_index, fn_index, cls_index = matric_indexs

        info = pd.DataFrame({'Pred': pred, 'Label': label}, index=case_name)
        metric = EstimatePrediction(pred, label, key_name)

        self.__auc_dict[key_name][norm_index, dr_index, fs_index, fn_index, cls_index] = \
            metric['{}_{}'.format(key_name, AUC)]
        self.__auc_std_dict[key_name][norm_index, dr_index, fs_index, fn_index, cls_index] = \
            metric['{}_{}'.format(key_name, AUC_STD)]

        if store_root:
            info.to_csv(os.path.join(model_folder, '{}_prediction.csv'.format(key_name)))
            self._AddOneMetric(metric, os.path.join(model_folder, 'metrics.csv'))
            self._MergeOneMetric(metric, key_name, model_name)

    def _AddOneMetric(self, info, store_path):
        if not os.path.exists(store_path):
            df = pd.DataFrame(info, index=['Value']).T
            df.to_csv(store_path)
        else:
            df = pd.read_csv(store_path, index_col=0)
            new_df = pd.DataFrame(info, index=['Value']).T
            df = pd.concat((df, new_df), sort=True, axis=0)
            df.to_csv(store_path)

    def _MergeOneMetric(self, metric, key, model_name):
        save_info = [metric['{}_{}'.format(key, index)] for index in HEADER]
        self.total_metric[key].loc[model_name] = save_info

    def _AddOneCvPrediction(self, store_path, prediction):
        if not os.path.exists(store_path):
            prediction.to_csv(store_path)
        else:
            temp = pd.read_csv(store_path, index_col=0)
            temp = pd.concat((temp, prediction), axis=0)
            temp.to_csv(store_path)

    def GenerateAucDict(self):
        self.total_metric = {TRAIN: pd.DataFrame(columns=HEADER),
                             BALANCE_TRAIN: pd.DataFrame(columns=HEADER),
                             TEST: pd.DataFrame(columns=HEADER),
                             CV_TRAIN: pd.DataFrame(columns=HEADER),
                             CV_VAL: pd.DataFrame(columns=HEADER)}

        self.total_num = len(self.normalizer_list) * \
                         len(self.dimension_reduction_list) * \
                         len(self.feature_selector_list) * \
                         len(self.classifier_list) * \
                         len(self.feature_selector_num_list)

        try:
            matrix = np.zeros((len(self.normalizer_list), len(self.dimension_reduction_list),
                               len(self.feature_selector_list), len(self.feature_selector_num_list),
                               len(self.classifier_list)))
        except:
            matrix = np.zeros(())

        self.__auc_dict = {CV_TRAIN: deepcopy(matrix), CV_VAL: deepcopy(matrix), TEST: deepcopy(matrix),
                           TRAIN: deepcopy(matrix), BALANCE_TRAIN: deepcopy(matrix)}
        self.__auc_std_dict = deepcopy(self.__auc_dict)

    def GetAuc(self):
        return self.__auc_dict

    def GetAucStd(self):
        return self.__auc_std_dict

    def GetStoreName(self, normalizer_name='', dimension_rediction_name='', feature_selector_name='',
                     feature_number='', classifer_name=''):
        case_name = '{}_{}_{}_{}_{}'.format(
            normalizer_name, dimension_rediction_name, feature_selector_name, feature_number, classifer_name
        )
        return case_name

    def SplitFolder(self, pipeline_name, store_root):
        normalizer, dr, fs, fn, cls = pipeline_name.split('_')

        normalizer_folder = os.path.join(store_root, normalizer)
        dr_folder = os.path.join(normalizer_folder, dr)
        fs_folder = os.path.join(dr_folder, '{}_{}'.format(fs, fn))
        cls_folder = os.path.join(fs_folder, cls)

        assert(os.path.isdir(store_root) and os.path.isdir(normalizer_folder) and os.path.isdir(dr_folder) and
               os.path.isdir(fs_folder) and os.path.isdir(cls_folder))
        return normalizer_folder, dr_folder, fs_folder, cls_folder

    def RunWithoutCV(self, train_container, test_container=DataContainer(), store_folder=''):
        self.SavePipelineInfo(store_folder)
        num = 0

        # TODO: Balance后面也可以变成循环处理:
        balance_train_container = self.balance.Run(train_container, store_folder)

        for norm_index, normalizer in enumerate(self.normalizer_list):
            norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
            norm_balance_train_container = normalizer.Run(balance_train_container, norm_store_folder, store_key=BALANCE_TRAIN)
            norm_train_container = normalizer.Transform(train_container, norm_store_folder, store_key=TRAIN)
            norm_test_container = normalizer.Transform(test_container, norm_store_folder, store_key=TEST)

            for dr_index, dr in enumerate(self.dimension_reduction_list):
                dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                if dr:
                    dr_balance_train_container = dr.Run(norm_balance_train_container, dr_store_folder, BALANCE_TRAIN)
                    dr_train_container = dr.Transform(norm_train_container, dr_store_folder, TRAIN)
                    if not test_container.IsEmpty():
                        dr_test_container = dr.Transform(norm_test_container, dr_store_folder, TEST)
                    else:
                        dr_test_container = norm_test_container
                else:
                    dr_balance_train_container = norm_balance_train_container
                    dr_train_container = norm_train_container
                    dr_test_container = norm_test_container

                for fs_index, fs in enumerate(self.feature_selector_list):
                    for fn_index, fn in enumerate(self.feature_selector_num_list):
                        if fs:
                            fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                            fs.SetSelectedFeatureNumber(fn)
                            fs_balance_train_container = fs.Run(dr_balance_train_container, fs_store_folder, BALANCE_TRAIN)
                            fs_train_container = fs.Transform(dr_train_container, fs_store_folder, TRAIN)
                            fs_test_container = fs.Transform(dr_test_container, fs_store_folder, TEST)
                        else:
                            fs_store_folder = dr_store_folder
                            fs_balance_train_container = dr_balance_train_container
                            fs_train_container = dr_train_container
                            fs_test_container = dr_test_container

                        for cls_index, cls in enumerate(self.classifier_list):
                            cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                            model_name = self.GetStoreName(normalizer.GetName(),
                                                           dr.GetName(),
                                                           fs.GetName(),
                                                           str(fn),
                                                           cls.GetName())
                            matrics_index = (norm_index, dr_index, fs_index, fn_index, cls_index)
                            num += 1
                            yield self.total_num, num

                            cls.SetDataContainer(fs_balance_train_container)
                            cls.Fit()
                            cls.Save(cls_store_folder)

                            balance_train_pred = cls.Predict(fs_balance_train_container.GetArray())
                            balance_train_label = fs_balance_train_container.GetLabel()
                            self.SaveOneResult(balance_train_pred, balance_train_label,
                                               BALANCE_TRAIN, fs_balance_train_container.GetCaseName(),
                                               matrics_index, model_name, store_folder, cls_store_folder)

                            train_data = fs_train_container.GetArray()
                            train_label = fs_train_container.GetLabel()
                            train_pred = cls.Predict(train_data)
                            self.SaveOneResult(train_pred, train_label,
                                               TRAIN, fs_train_container.GetCaseName(),
                                               matrics_index, model_name, store_folder, cls_store_folder)

                            if not test_container.IsEmpty():
                                test_data = fs_test_container.GetArray()
                                test_label = fs_test_container.GetLabel()
                                test_pred = cls.Predict(test_data)
                                self.SaveOneResult(test_pred, test_label,
                                                   TEST, fs_test_container.GetCaseName(),
                                                   matrics_index, model_name, store_folder, cls_store_folder)

        self.total_metric[BALANCE_TRAIN].to_csv(os.path.join(store_folder, '{}_results.csv'.format(BALANCE_TRAIN)))
        self.total_metric[TRAIN].to_csv(os.path.join(store_folder, '{}_results.csv'.format(TRAIN)))
        if not test_container.IsEmpty():
            self.total_metric[TEST].to_csv(os.path.join(store_folder, '{}_results.csv'.format(TEST)))

    def RunWithCV(self, train_container, store_folder=''):
        for group, containers in enumerate(self.cv.Generate(train_container)):
            cv_train_container, cv_val_container = containers

            balance_cv_train_container = self.balance.Run(cv_train_container)
            num = 0
            for norm_index, normalizer in enumerate(self.normalizer_list):
                norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
                norm_cv_train_container = normalizer.Run(balance_cv_train_container)
                norm_cv_val_container = normalizer.Transform(cv_val_container)

                for dr_index, dr in enumerate(self.dimension_reduction_list):
                    dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                    if dr:
                        dr_cv_train_container = dr.Run(norm_cv_train_container)
                        dr_cv_val_container = dr.Transform(norm_cv_val_container)
                    else:
                        dr_cv_train_container = norm_cv_train_container
                        dr_cv_val_container = norm_cv_val_container

                    for fs_index, fs in enumerate(self.feature_selector_list):
                        for fn_index, fn in enumerate(self.feature_selector_num_list):
                            if fs:
                                fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                                fs.SetSelectedFeatureNumber(fn)
                                fs_cv_train_container = fs.Run(dr_cv_train_container)
                                fs_cv_val_container = fs.Transform(dr_cv_val_container)
                            else:
                                fs_store_folder = dr_store_folder
                                fs_cv_train_container = dr_cv_train_container
                                fs_cv_val_container = dr_cv_val_container

                            for cls_index, cls in enumerate(self.classifier_list):
                                cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                                model_name = self.GetStoreName(normalizer.GetName(),
                                                               dr.GetName(),
                                                               fs.GetName(),
                                                               str(fn),
                                                               cls.GetName())
                                num += 1
                                yield self.total_num, num, group

                                cls.SetDataContainer(fs_cv_train_container)
                                cls.Fit()

                                cv_train_pred = cls.Predict(fs_cv_train_container.GetArray())
                                cv_train_label = fs_cv_train_container.GetLabel()
                                cv_train_info = pd.DataFrame({'Pred': cv_train_pred, 'Label': cv_train_label,
                                                              'Group': [group for temp in cv_train_label]},
                                                             index=fs_cv_train_container.GetCaseName())

                                cv_val_pred = cls.Predict(fs_cv_val_container.GetArray())
                                cv_val_label = fs_cv_val_container.GetLabel()
                                cv_val_info = pd.DataFrame({'Pred': cv_val_pred, 'Label': cv_val_label,
                                                            'Group': [group for temp in cv_val_label]},
                                                           index=fs_cv_val_container.GetCaseName())

                                if store_folder:
                                    self._AddOneCvPrediction(os.path.join(cls_store_folder,
                                                                         '{}_prediction.csv'.format(CV_TRAIN)),
                                                             cv_train_info)
                                    self._AddOneCvPrediction(os.path.join(cls_store_folder,
                                                                         '{}_prediction.csv'.format(CV_VAL)),
                                                             cv_val_info)

    def MergeCvResult(self, store_folder):
        num = 0
        for norm_index, normalizer in enumerate(self.normalizer_list):
            norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
            for dr_index, dr in enumerate(self.dimension_reduction_list):
                dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                for fs_index, fs in enumerate(self.feature_selector_list):
                    for fn_index, fn in enumerate(self.feature_selector_num_list):
                        fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                        for cls_index, cls in enumerate(self.classifier_list):
                            cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                            model_name = self.GetStoreName(normalizer.GetName(),
                                                           dr.GetName(),
                                                           fs.GetName(),
                                                           str(fn),
                                                           cls.GetName())
                            num += 1
                            yield self.total_num, num

                            # ADD CV Train
                            cv_train_info = pd.read_csv(os.path.join(cls_store_folder,
                                                                     '{}_prediction.csv'.format(CV_TRAIN)),
                                                        index_col=0)
                            cv_train_metric = EstimatePrediction(cv_train_info['Pred'], cv_train_info['Label'],
                                                                 key_word=CV_TRAIN)
                            self.__auc_dict[CV_TRAIN][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_train_metric['{}_{}'.format(CV_TRAIN, AUC)]
                            self.__auc_std_dict[CV_TRAIN][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_train_metric['{}_{}'.format(CV_TRAIN, AUC_STD)]
                            self._AddOneMetric(cv_train_metric, os.path.join(cls_store_folder, 'metrics.csv'))
                            self._MergeOneMetric(cv_train_metric, CV_TRAIN, model_name)

                            # ADD CV Validation
                            cv_val_info = pd.read_csv(os.path.join(cls_store_folder,
                                                                   '{}_prediction.csv'.format(CV_VAL)),
                                                      index_col=0)
                            cv_val_metric = EstimatePrediction(cv_val_info['Pred'], cv_val_info['Label'],
                                                               key_word=CV_VAL)
                            self.__auc_dict[CV_VAL][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_val_metric['{}_{}'.format(CV_VAL, AUC)]
                            self.__auc_std_dict[CV_VAL][norm_index, dr_index, fs_index, fn_index, cls_index] = \
                                cv_val_metric['{}_{}'.format(CV_VAL, AUC_STD)]
                            self._AddOneMetric(cv_val_metric, os.path.join(cls_store_folder, 'metrics.csv'))
                            self._MergeOneMetric(cv_val_metric, CV_VAL, model_name)

        self.total_metric[CV_TRAIN].to_csv(os.path.join(store_folder, '{}_results.csv'.format(CV_TRAIN)))
        self.total_metric[CV_VAL].to_csv(os.path.join(store_folder, '{}_results.csv'.format(CV_VAL)))


if __name__ == '__main__':
    manager = PipelinesManager()

    index_dict = Index2Dict()

    train = DataContainer()
    test = DataContainer()
    train.Load(r'C:\Users\yangs\Desktop\train_numeric_feature.csv')
    test.Load(r'C:\Users\yangs\Desktop\test_numeric_feature.csv')

    faps = PipelinesManager(balancer=index_dict.GetInstantByIndex('UpSampling'),
                            normalizer_list=[index_dict.GetInstantByIndex('Mean')],
                            dimension_reduction_list=[index_dict.GetInstantByIndex('PCC')],
                            feature_selector_list=[index_dict.GetInstantByIndex('ANOVA')],
                            feature_selector_num_list=list(np.arange(1, 18)),
                            classifier_list=[index_dict.GetInstantByIndex('SVM')],
                            cross_validation=index_dict.GetInstantByIndex('5-Fold'))

    # for total, num in faps.RunWithoutCV(train, store_folder=r'..\..\Demo\db2-1'):
    #     print(total, num)
    for total, num, group in faps.RunWithCV(train, store_folder=r'..\..\Demo\db1'):
        print(total, num, group)
    for total, num in faps.MergeCvResult(store_folder=r'..\..\Demo\db2-1'):
        print(total, num)

    print('Done')
