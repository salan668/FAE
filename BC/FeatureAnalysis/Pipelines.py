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
from BC.HyperParamManager.HyperParamManager import GetClassifierHyperParams

from BC.Utility.PathManager import MakeFolder
from BC.Utility.Constants import *

from HomeUI.VersionConstant import *


class PipelinesManager(object):
    def __init__(self, balancer=None, normalizer_list=[NormalizerNone],
                 dimension_reduction_list=[DimensionReductionByPCC()], feature_selector_list=[],
                 feature_selector_num_list=[], classifier_list=[],
                 cv=None, hyper_param={}, random_seed={}, logger=None):
        self.balance = balancer
        self.normalizer_list = normalizer_list
        self.dimension_reduction_list = dimension_reduction_list
        self.feature_selector_list = feature_selector_list
        self.feature_selector_num_list = feature_selector_num_list
        self.classifier_list = classifier_list

        self.cv = cv
        self.hyper_param = hyper_param
        self.random_seed = random_seed

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
            pickle.dump(self.auc_dict, file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(store_folder, 'auc_std_metric.pkl'), 'wb') as file:
            pickle.dump(self.auc_std_dict, file, pickle.HIGHEST_PROTOCOL)

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
            self.auc_dict = pickle.load(file)
        with open(std_path, 'rb') as file:
            self.auc_std_dict = pickle.load(file)

        return True

    def SaveOneResult(self, pred, label, key_name, case_name, matric_indexs, model_name,
                      store_root='', model_folder='', cutoff=None):
        assert(len(matric_indexs) == 5)
        norm_index, dr_index, fs_index, fn_index, cls_index = matric_indexs

        info = pd.DataFrame({'Pred': pred, 'Label': label}, index=case_name)
        metric = EstimatePrediction(pred, label, key_name, cutoff=cutoff)

        self.auc_dict[key_name][norm_index, dr_index, fs_index, fn_index, cls_index] = \
            metric['{}_{}'.format(key_name, AUC)]
        self.auc_std_dict[key_name][norm_index, dr_index, fs_index, fn_index, cls_index] = \
            metric['{}_{}'.format(key_name, AUC_STD)]

        if store_root:
            info.to_csv(os.path.join(model_folder, '{}_prediction.csv'.format(key_name)))
            self._AddOneMetric(metric, os.path.join(model_folder, 'metrics.csv'))
            self._MergeOneMetric(metric, key_name, model_name)
        return metric

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

        self.auc_dict = {CV_TRAIN: deepcopy(matrix), CV_VAL: deepcopy(matrix), TEST: deepcopy(matrix),
                         TRAIN: deepcopy(matrix), BALANCE_TRAIN: deepcopy(matrix)}
        self.auc_std_dict = deepcopy(self.auc_dict)

    def GetAuc(self):
        return self.auc_dict

    def GetAucStd(self):
        return self.auc_std_dict

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

        assert (os.path.isdir(store_root) and os.path.isdir(normalizer_folder) and os.path.isdir(dr_folder) and
                os.path.isdir(fs_folder) and os.path.isdir(cls_folder)), \
            f"One or more directories do not exist: store_root={store_root}, normalizer_folder={normalizer_folder}, dr_folder={dr_folder}, fs_folder={fs_folder}, cls_folder={cls_folder}"

        return normalizer_folder, dr_folder, fs_folder, cls_folder

    def Run(self, train_container, test_container=DataContainer(), store_folder='', is_train_cutoff=False):
        self.SavePipelineInfo(store_folder)
        num = 0

        balance_train_container = self.balance.Run(train_container, store_folder)

        for norm_index, normalizer in enumerate(self.normalizer_list):
            norm_store_folder = MakeFolder(store_folder, normalizer.GetName())
            norm_balance_train_container = normalizer.Run(balance_train_container, norm_store_folder,
                                                          store_key=BALANCE_TRAIN)
            norm_train_container = normalizer.Transform(train_container, norm_store_folder, store_key=TRAIN)
            if not test_container.IsEmpty():
                norm_test_container = normalizer.Transform(test_container, norm_store_folder, store_key=TEST)

            for dr_index, dr in enumerate(self.dimension_reduction_list):
                dr_store_folder = MakeFolder(norm_store_folder, dr.GetName())
                if dr:
                    dr_balance_train_container = dr.Run(norm_balance_train_container, dr_store_folder, BALANCE_TRAIN)
                    dr_train_container = dr.Transform(norm_train_container, dr_store_folder, TRAIN)
                    if not test_container.IsEmpty():
                        dr_test_container = dr.Transform(norm_test_container, dr_store_folder, TEST)
                else:
                    dr_balance_train_container = norm_balance_train_container
                    dr_train_container = norm_train_container
                    if not test_container.IsEmpty():
                        dr_test_container = norm_test_container

                for fs_index, fs in enumerate(self.feature_selector_list):
                    for fn_index, fn in enumerate(self.feature_selector_num_list):
                        if fs:
                            fs_store_folder = MakeFolder(dr_store_folder, '{}_{}'.format(fs.GetName(), fn))
                            fs.SetSelectedFeatureNumber(fn)
                            fs_balance_train_container = fs.Run(dr_balance_train_container, fs_store_folder,
                                                                BALANCE_TRAIN)
                            fs_train_container = fs.Transform(dr_train_container, fs_store_folder, TRAIN)
                            if not test_container.IsEmpty():
                                fs_test_container = fs.Transform(dr_test_container, fs_store_folder, TEST)
                        else:
                            fs_store_folder = dr_store_folder
                            fs_balance_train_container = dr_balance_train_container
                            fs_train_container = dr_train_container
                            if not test_container.IsEmpty():
                                fs_test_container = dr_test_container

                        for cls_index, cls in enumerate(self.classifier_list):
                            cls_store_folder = MakeFolder(fs_store_folder, cls.GetName())
                            model_name = self.GetStoreName(normalizer.GetName(),
                                                           dr.GetName(),
                                                           fs.GetName(),
                                                           str(fn),
                                                           cls.GetName())
                            matrics_index = (norm_index, dr_index, fs_index, fn_index, cls_index)

                            cls.SetDataContainer(fs_balance_train_container)
                            cls.SetSeed(self.random_seed)
                            if cls.GetName() in self.hyper_param.keys():
                                cls.Fit(self.hyper_param[cls.GetName()], self.cv.cv_part)
                            else:
                                cls.Fit(cv_part=self.cv.cv_part)

                            val_pred, val_label = cls.CvPredict(fs_train_container, self.cv.cv_part)

                            # 根据best_param用所有数据重新训练
                            cls.Fit()
                            cls.Save(cls_store_folder)

                            balanced_metric = self.SaveOneResult(cls.Predict(fs_balance_train_container.GetArray()),
                                                                 fs_balance_train_container.GetLabel(),
                                                                 BALANCE_TRAIN,
                                                                 fs_balance_train_container.GetCaseName(),
                                                                 matrics_index, model_name, store_folder,
                                                                 cls_store_folder)

                            if is_train_cutoff:
                                cutoff = float(balanced_metric[BALANCE_TRAIN + '_' + CUTOFF])
                            else:
                                cutoff = None

                            self.SaveOneResult(cls.Predict(fs_train_container.GetArray()),
                                               fs_train_container.GetLabel(),
                                               TRAIN, fs_train_container.GetCaseName(),
                                               matrics_index, model_name, store_folder, cls_store_folder,
                                               cutoff=cutoff)
                            self.SaveOneResult(val_pred, val_label,
                                               CV_VAL, fs_train_container.GetCaseName(),
                                               matrics_index, model_name, store_folder, cls_store_folder,
                                               cutoff=cutoff)

                            if not test_container.IsEmpty():
                                self.SaveOneResult(cls.Predict(fs_test_container.GetArray()),
                                                   fs_test_container.GetLabel(),
                                                   TEST, fs_test_container.GetCaseName(),
                                                   matrics_index, model_name, store_folder, cls_store_folder,
                                                   cutoff=cutoff)

                            num += 1
                            # print(self.total_num, num)
                            yield self.total_num, num

                self.total_metric[BALANCE_TRAIN].to_csv(
                    os.path.join(store_folder, '{}_results.csv'.format(BALANCE_TRAIN)))
                self.total_metric[TRAIN].to_csv(os.path.join(store_folder, '{}_results.csv'.format(TRAIN)))
                self.total_metric[CV_VAL].to_csv(os.path.join(store_folder, '{}_results.csv'.format(CV_VAL)))
                if not test_container.IsEmpty():
                    self.total_metric[TEST].to_csv(os.path.join(store_folder, '{}_results.csv'.format(TEST)))


if __name__ == '__main__':
    manager = PipelinesManager()

    index_dict = Index2Dict(r'..\..')

    train = DataContainer()
    test = DataContainer()
    train.Load(r'..\..\..\Demo\train_numeric_feature.csv')
    test.Load(r'..\..\..\Demo\test_numeric_feature.csv')

    hyper_param = GetClassifierHyperParams(r'D:\MyCode\FAE\FAE\BC\HyperParameters\Classifier')

    faps = PipelinesManager(balancer=index_dict.GetInstantByIndex('SMOTE'),
                            normalizer_list=[index_dict.GetInstantByIndex('Mean')],
                            dimension_reduction_list=[index_dict.GetInstantByIndex('PCC')],
                            feature_selector_list=[index_dict.GetInstantByIndex('ANOVA')],
                            feature_selector_num_list=list(np.arange(1, 3)),
                            classifier_list=[index_dict.GetInstantByIndex('SVM')],
                            cross_validation=index_dict.GetInstantByIndex('5-Fold'),
                            hyper_param=hyper_param)

    faps.Run(train, test, store_folder=r'..\..\..\Demo\DemoRun')
    # for total, num in faps.Run(train, test, store_folder=r'..\..\..\Demo\DemoRun'):
    #     print(total, num)


    print('Done')
