'''.
Jun 17, 2018.
Yang SONG, songyangmri@gmail.com
'''

from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.IndexDict import Index2Dict
from FAE.FeatureAnalysis.Normalizer import NormalizerNone
from FAE.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC
from FAE.FeatureAnalysis.FeatureSelector import FeatureSelector

import os
import pickle
import pandas as pd
import csv
import numpy as np
from copy import deepcopy

class FeatureAnalysisPipelines:
    def __init__(self, normalizer_list=[], dimension_reduction_list=[], feature_selector_list=[],
                 feature_selector_num_list=[], classifier_list=[], cross_validation=None, is_hyper_parameter=False):
        self.__normalizer_list = normalizer_list
        self._dimension_reduction_list = dimension_reduction_list
        self.__feature_selector_list = feature_selector_list
        self.__feature_selector_num_list = feature_selector_num_list
        self.__classifier_list = classifier_list
        self.__cross_validation = cross_validation
        self.__is_hyper_parameter = is_hyper_parameter

        self.GenerateMetircDict()

    def SetNormalizerList(self, normalizer_list):
        self.__normalizer_list = normalizer_list
    def GetNormalizerList(self):
        return self.__normalizer_list
    def SetDimensionReductionList(self, dimensino_reduction_list):
        self._dimension_reduction_list = dimensino_reduction_list
    def GetDimensionReductionList(self):
        return self._dimension_reduction_list
    def SetFeatureSelectorList(self, feature_selector_list):
        self.__feature_selector_list = feature_selector_list
    def GetFeatureSelectorList(self):
        return self.__feature_selector_list
    def SetFeatureNumberList(self, feature_selector_num_list):
        self.__feature_selector_num_list = feature_selector_num_list
    def GetFeatureNumberList(self):
        return self.__feature_selector_num_list
    def SetClassifierList(self, classifier_list):
        self.__classifier_list = classifier_list
    def GetClassifierList(self):
        return self.__classifier_list
    def SetCrossValition(self, cv):
        self.__cross_validation = cv
    def GetCrossValidation(self):
        return self.__cross_validation

    def SaveAll(self, store_folder):
        self.SaveMetricDict(store_folder)
        self.SavePipelineInfo(store_folder)

    def LoadAll(self, store_folder):
        self.LoadMetricDict(store_folder)
        self.LoadPipelineInfo(store_folder)

    def GenerateMetircDict(self):
        try:
            matrix = np.zeros((len(self.__normalizer_list), len(self._dimension_reduction_list),
                               len(self.__feature_selector_list), len(self.__feature_selector_num_list),
                               len(self.__classifier_list)))
        except:
            matrix = np.zeros(())

        self.__auc_matrix_dict = {'train': deepcopy(matrix), 'val': deepcopy(matrix), 'test': deepcopy(matrix), 'all_train': deepcopy(matrix)}
        self.__auc_std_matrix_dict = {'train': deepcopy(matrix), 'val': deepcopy(matrix), 'test': deepcopy(matrix), 'all_train': deepcopy(matrix)}
        self.__accuracy_matrix_dict = {'train': deepcopy(matrix), 'val': deepcopy(matrix), 'test': deepcopy(matrix), 'all_train': deepcopy(matrix)}

    def SavePipelineInfo(self, store_folder):
        with open(os.path.join(store_folder, 'pipeline_info.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            temp_row = []
            temp_row.append('Normalizer')
            for index in self.__normalizer_list:
                temp_row.append(index.GetName())
            writer.writerow(temp_row)

            temp_row = []
            temp_row.append('DimensionReduction')
            for index in self._dimension_reduction_list:
                temp_row.append(index.GetName())
            writer.writerow(temp_row)

            temp_row = []
            temp_row.append('FeatureSelector')
            for index in self.__feature_selector_list:
                temp_row.append(index.GetName())
            writer.writerow(temp_row)

            temp_row = []
            temp_row.append('FeatureNumber')
            for index in self.__feature_selector_num_list:
                temp_row.append(index)
            writer.writerow(temp_row)

            temp_row = []
            temp_row.append('Classifier')
            for index in self.__classifier_list:
                temp_row.append(index.GetName())
            writer.writerow(temp_row)

            temp_row = []
            temp_row.append('CrossValidation')
            temp_row.append(self.__cross_validation.GetName())
            writer.writerow(temp_row)

    def LoadPipelineInfo(self, store_folder):
        index_2_dict = Index2Dict()
        with open(os.path.join(store_folder, 'pipeline_info.csv'), 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'Normalizer':
                    self.__normalizer_list = []
                    for index in row[1:]:
                        self.__normalizer_list.append(index_2_dict.GetInstantByIndex(index))
                elif row[0] == 'DimensionReduction':
                    self._dimension_reduction_list = []
                    for index in row[1:]:
                        self._dimension_reduction_list.append(index_2_dict.GetInstantByIndex(index))
                elif row[0] == 'FeatureSelector':
                    self.__feature_selector_list = []
                    for index in row[1:]:
                        self.__feature_selector_list.append(index_2_dict.GetInstantByIndex(index))
                elif row[0] == 'FeatureNumber':
                    self.__feature_selector_num_list = row[1:]
                elif row[0] == 'Classifier':
                    self.__classifier_list = []
                    for index in row[1:]:
                        self.__classifier_list.append(index_2_dict.GetInstantByIndex(index))
                elif row[0] == 'CrossValidation':
                    self.__cross_validation = index_2_dict.GetInstantByIndex(row[1])
                else:
                    print('Unknown name.')

    def SaveMetricDict(self, store_folder):
        with open(os.path.join(store_folder, 'auc_metric.pkl'), 'wb') as file:
            pickle.dump(self.__auc_matrix_dict, file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(store_folder, 'auc_std_metric.pkl'), 'wb') as file:
            pickle.dump(self.__auc_std_matrix_dict, file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(store_folder, 'accuracy_metric.pkl'), 'wb') as file:
            pickle.dump(self.__accuracy_matrix_dict, file, pickle.HIGHEST_PROTOCOL)

    def LoadMetricDict(self, store_folder):
        with open(os.path.join(store_folder, 'auc_metric.pkl'), 'rb') as file:
            self.__auc_matrix_dict = pickle.load(file)
        with open(os.path.join(store_folder, 'auc_std_metric.pkl'), 'rb') as file:
            self.__auc_std_matrix_dict = pickle.load(file)
        with open(os.path.join(store_folder, 'accuracy_metric.pkl'), 'rb') as file:
            self.__accuracy_matrix_dict = pickle.load(file)

    def GetAUCMetric(self):
        return self.__auc_matrix_dict

    def GetAUCstdMetric(self):
        return self.__auc_std_matrix_dict

    def GetAccuracyMetric(self):
        return self.__accuracy_matrix_dict

    def Run(self, train_data_container, test_data_container=DataContainer(), store_folder='', is_hyper_parameter=False):
        column_list = ['sample_number', 'positive_number', 'negative_number',
                       'auc', 'auc 95% CIs', 'auc std', 'accuracy',
                       'Youden Index', 'sensitivity', 'specificity',
                       'positive predictive value', 'negative predictive value']
        train_df = pd.DataFrame(columns=column_list)
        val_df = pd.DataFrame(columns=column_list)
        test_df = pd.DataFrame(columns=column_list)
        all_train_df = pd.DataFrame(columns=column_list)

        if self.__normalizer_list == []:
            self.__normalizer_list = [NormalizerNone()]

        if self._dimension_reduction_list == []:
            self._dimension_reduction_list = [DimensionReductionByPCC()]

        self.GenerateMetircDict()
        self.SavePipelineInfo(store_folder)

        num = 0
        total_num = len(self.__normalizer_list) * \
                    len(self._dimension_reduction_list) * \
                    len(self.__feature_selector_list) * \
                    len(self.__classifier_list) * \
                    len(self.__feature_selector_num_list)

        for normalizer, normalizer_index in zip(self.__normalizer_list, range(len(self.__normalizer_list))):
            for dimension_reductor, dimension_reductor_index in zip(self._dimension_reduction_list, range(len(self._dimension_reduction_list))):
                for feature_selector, feature_selector_index in zip(self.__feature_selector_list, range(len(self.__feature_selector_list))):
                    for classifier, classifier_index in zip(self.__classifier_list, range(len(self.__classifier_list))):
                        for feature_num, feature_num_index in zip(self.__feature_selector_num_list, range(len(self.__feature_selector_num_list))):
                            num += 1
                            yield normalizer.GetName(), dimension_reductor.GetName(), feature_selector.GetName(), feature_num, \
                                  classifier.GetName(), num, total_num

                            feature_selector.SetSelectedFeatureNumber(feature_num)
                            one_pipeline = OnePipeline(normalizer=normalizer,
                                                       dimension_reduction=dimension_reductor,
                                                       feature_selector=feature_selector,
                                                       classifier=classifier,
                                                       cross_validation=self.__cross_validation)
                            case_name = one_pipeline.GetStoreName()
                            case_store_folder = os.path.join(store_folder, case_name)
                            train_cv_metric, val_cv_metric, test_metric, all_train_metric = one_pipeline.Run(train_data_container,
                                                                                     test_data_container,
                                                                                     case_store_folder,
                                                                                     is_hyper_parameter)
                            
                            self.__auc_matrix_dict['train'][normalizer_index,
                                                     dimension_reductor_index, 
                                                     feature_selector_index, 
                                                     feature_num_index, 
                                                     classifier_index] = train_cv_metric['train_auc']
                            self.__auc_std_matrix_dict['train'][normalizer_index,
                                                     dimension_reductor_index,
                                                     feature_selector_index,
                                                     feature_num_index,
                                                     classifier_index] = train_cv_metric['train_auc std']
                            self.__auc_matrix_dict['all_train'][normalizer_index,
                                                            dimension_reductor_index,
                                                            feature_selector_index,
                                                            feature_num_index,
                                                            classifier_index] = all_train_metric['all_train_auc']
                            self.__auc_std_matrix_dict['all_train'][normalizer_index,
                                                                dimension_reductor_index,
                                                                feature_selector_index,
                                                                feature_num_index,
                                                                classifier_index] = all_train_metric['all_train_auc std']
                            self.__auc_matrix_dict['val'][normalizer_index,
                                                     dimension_reductor_index,
                                                     feature_selector_index,
                                                     feature_num_index,
                                                     classifier_index] = val_cv_metric['val_auc']
                            self.__auc_std_matrix_dict['val'][normalizer_index,
                                                     dimension_reductor_index,
                                                     feature_selector_index,
                                                     feature_num_index,
                                                     classifier_index] = val_cv_metric['val_auc std']


                            if store_folder and os.path.isdir(store_folder):
                                store_path = os.path.join(store_folder, 'train_result.csv')
                                save_info = [train_cv_metric['train_' + index] for index in column_list]
                                train_df.loc[case_name] = save_info
                                train_df.to_csv(store_path)

                                store_path = os.path.join(store_folder, 'all_train_result.csv')
                                save_info = [all_train_metric['all_train_' + index] for index in column_list]
                                all_train_df.loc[case_name] = save_info
                                all_train_df.to_csv(store_path)

                                store_path = os.path.join(store_folder, 'val_result.csv')
                                save_info = [val_cv_metric['val_' + index] for index in column_list]
                                val_df.loc[case_name] = save_info
                                val_df.to_csv(store_path)

                                if not test_data_container.IsEmpty():
                                    self.__auc_matrix_dict['test'][normalizer_index,
                                                                   dimension_reductor_index,
                                                                   feature_selector_index,
                                                                   feature_num_index,
                                                                   classifier_index] = test_metric['test_auc']
                                    self.__auc_std_matrix_dict['test'][normalizer_index,
                                                                   dimension_reductor_index,
                                                                   feature_selector_index,
                                                                   feature_num_index,
                                                                   classifier_index] = test_metric['test_auc std']

                                    store_path = os.path.join(store_folder, 'test_result.csv')
                                    save_info = [test_metric['test_' + index] for index in column_list]
                                    test_df.loc[case_name] = save_info
                                    test_df.to_csv(store_path)

                                self.SaveMetricDict(store_folder)


class OnePipeline:
    def __init__(self, normalizer=None, dimension_reduction=None, feature_selector=None, classifier=None, cross_validation=None):
        self.__normalizer = normalizer
        self.__dimension_reduction = dimension_reduction
        self.__feature_selector = feature_selector
        self.__classifier = classifier
        self.__cv = cross_validation

    def SetNormalizer(self, normalizer):
        self.__normalizer = normalizer
    def GetNormalizer(self):
        return self.__normalizer

    def SetDimensionReduction(self, dimension_reduction):
        self.__dimension_reduction = dimension_reduction
    def GetDimensionReduction(self):
        return self.__dimension_reduction

    def SetFeatureSelector(self, feature_selector):
        self.__feature_selector = feature_selector
    def GetFeatureSelector(self):
        return self.__feature_selector

    def SetClassifier(self, classifier):
        self.__classifier = classifier
    def GetClassifier(self):
        return self.__classifier

    def SetCrossValidation(self, cv):
        self.__cv = cv
    def GetCrossValidatiaon(self):
        return self.__cv

    def SavePipeline(self, feature_number, store_path):
        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Normalizer', self.__normalizer.GetName()])
            writer.writerow(['DimensionReduction', self.__dimension_reduction.GetName()])
            writer.writerow(['FeatureSelector', self.__feature_selector.GetName()])
            writer.writerow(['FeatureNumber', feature_number])
            writer.writerow(['Classifier', self.__classifier.GetName()])
            writer.writerow(['CrossValidation', self.__cv.GetName()])

    def LoadPipeline(self, store_path):
        index_2_dict = Index2Dict()
        feature_number = 0
        with open(store_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'Normalizer':
                    self.__normalizer = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'DimensionReduction':
                    self.__dimension_reduction = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'FeatureSelector':
                    self.__feature_selector = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'FeatureNumber':
                    feature_number = int(row[1])
                if row[0] == 'Classifier':
                    self.__classifier = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'CrossValidation':
                    self.__cv = index_2_dict.GetInstantByIndex(row[1])
        self.__feature_selector.SetSelectedFeatureNumber(feature_number)

    def GetName(self):
        try:
            return self.__feature_selector[-1].GetName() + '-' + self.__classifier.GetName()
        except:
            return self.__feature_selector.GetName() + '-' + self.__classifier.GetName()

    def GetStoreName(self):
        case_name = self.__normalizer.GetName() + '_' + \
                    self.__dimension_reduction.GetName() + '_' + \
                    self.__feature_selector.GetName() + '_' + \
                    str(self.__feature_selector.GetSelectedFeatureNumber()) + '_' + \
                    self.__classifier.GetName()
        return case_name

    def Run(self, train_data_container, test_data_container=DataContainer(), store_folder='', is_hyper_parameter=False):
        raw_train_data_container = deepcopy(train_data_container)
        raw_test_data_conainer = deepcopy(test_data_container)

        if store_folder:
            if not os.path.exists(store_folder):
                os.mkdir(store_folder)

        if not (self.__cv and self.__classifier):
            print('Give CV method and classifier')

        if self.__normalizer:
            raw_train_data_container = self.__normalizer.Run(raw_train_data_container, store_folder)
            if not test_data_container.IsEmpty():
                raw_test_data_conainer = self.__normalizer.Run(raw_test_data_conainer, store_folder, is_test=True)

        if self.__dimension_reduction:
            raw_train_data_container = self.__dimension_reduction.Run(raw_train_data_container, store_folder)
            if not test_data_container.IsEmpty():
                raw_test_data_conainer = self.__dimension_reduction.Transform(raw_test_data_conainer)

        if self.__feature_selector:
            raw_train_data_container = self.__feature_selector.Run(raw_train_data_container, store_folder)
            if not test_data_container.IsEmpty():
                selected_feature_name = raw_train_data_container.GetFeatureName()
                fs = FeatureSelector()
                raw_test_data_conainer = fs.SelectFeatureByName(raw_test_data_conainer, selected_feature_name)

        self.__cv.SetClassifier(self.__classifier)
        train_cv_metric, val_cv_metric, test_metric, all_train_metric = self.__cv.Run(raw_train_data_container,
                                                              raw_test_data_conainer,
                                                              store_folder,
                                                              is_hyper_parameter)

        if store_folder:
            self.SavePipeline(len(raw_train_data_container.GetFeatureName()), os.path.join(store_folder, 'pipeline_info.csv'))

        return train_cv_metric, val_cv_metric, test_metric, all_train_metric

if __name__ == '__main__':
    print(os.getcwd())

    data_container = DataContainer()
    file_path = os.path.abspath(r'..\..\Example\numeric_feature.csv')
    print(file_path)
    data_container.Load(file_path)

    temp = OnePipeline(normalizer=NormalizerZeroCenterAndUnit(), feature_selector=FeatureSelectPipeline([RemoveCosSimilarityFeatures(), FeatureSelectByANOVA(10)]),
                       classifier=SVM(), cross_validation=CrossValidation('5-folder'))
    temp.Run(data_container, store_folder=r'..\..\Example\one_pipeline')



