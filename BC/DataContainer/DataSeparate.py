"""
All rights reserved.
--Yang Song, songyangmri@gmail.com
"""

import os

import numpy as np
from random import shuffle
import pandas as pd

from scipy.stats import levene, ttest_ind, kstest, mannwhitneyu, chi2_contingency
from collections import Counter

from BC.DataContainer.DataContainer import DataContainer


class FeatureType:
    Numeric = 1
    Categorical = 2


class OneFeatureStatistics(object):
    def __init__(self):
        pass

    def EstimateFeatureType(self, array):
        # 目前通过Unique方法进行判断
        if np.unique(array).size < array.size // 100:
            return FeatureType.Categorical
        else:
            return FeatureType.Numeric

    def _CompareNumetricFeatures(self, array1, array2):
        description = {}
        _, description['p-value'] = mannwhitneyu(array1, array2)
        description['method'] = 'Mann-Whitney'
        description['description'] = ['{:.2f}+-{:.2f}'.format(np.mean(array1), np.std(array1)),
                                      '{:.2f}+-{:.2f}'.format(np.mean(array2), np.std(array2))]
        return description

    def _CompareCategoricalFeatures(self, array1, array2):
        df1 = pd.DataFrame(Counter(array1), index=[1])
        df2 = pd.DataFrame(Counter(array2), index=[2])
        df = pd.concat((df1, df2), axis=0)
        df = df.fillna(0)

        descrip1, descrip2 = df.iloc[0, :], df.iloc[1, :]
        descrip1 = ['{}: {}'.format(descrip1.index[x], descrip1.iloc[x]) for x in range(descrip1.size)]
        descrip2 = ['{}: {}'.format(descrip2.index[x], descrip2.iloc[x]) for x in range(descrip2.size)]

        description = {}
        _, description['p-value'], _, _ = chi2_contingency(df.values, correction=True)
        description['method'] = 'Chi-Square'
        description['description'] = [', '.join(descrip1),
                                      ', '.join(descrip2)]
        return description


    def AnalyzeTwoArrays(self, array1, array2, feature_type=None):
        if isinstance(array1, list):
            array1 = np.array(array1)
        if isinstance(array2, list):
            array2 = np.array(array2)
        assert(array1.ndim == 1 and array2.ndim == 1)

        if feature_type is None:
            feature_type = self.EstimateFeatureType(np.concatenate([array1, array2], axis=0))

        if feature_type == FeatureType.Numeric:
            description = self._CompareNumetricFeatures(array1, array2)
        elif feature_type == FeatureType.Categorical:
            description = self._CompareCategoricalFeatures(array1, array2)
        else:
            print('Neither Numeric nor Categorical features.')
            raise KeyError

        return description


class FeatureMatrixStatistics(object):
    def __init__(self):
        self.one_feature_statistics = OneFeatureStatistics()
        pass

    def CompareFeatures(self, data1, data2, name=None):
        assert(type(data1) == type(data2))
        if name is None:
            name = ['Array1', 'Array2']
        assert(isinstance(name, list) and len(name) == 2)

        pvalue_key = 'P'
        result = pd.DataFrame(columns=[name[0], name[1], pvalue_key, 'method'])

        if isinstance(data1, pd.DataFrame):
            array1, array2 = data1.values, data2.values
            feature = list(data1.columns)
        elif isinstance(data1, np.ndarray):
            array1, array2 = data1, data2
            feature = ['Feature{}'.format(x) for x in range(array1.shape[1])]
        elif isinstance(data1, DataContainer):
            array1, array2 = data1.GetArray(), data2.GetArray()
            feature = data1.GetFeatureName()
        else:
            raise TypeError

        for index in range(len(feature)):
            description = self.one_feature_statistics.AnalyzeTwoArrays(array1[:, index], array2[:, index])
            one_df = pd.DataFrame({name[0]: description['description'][0],
                                   name[1]: description['description'][1],
                                   pvalue_key: description['p-value'],
                                   'method': description['method']},
                                  index=[feature[index]])
            result = pd.concat([result, one_df], axis=0)

        return result


class DataSeparate:
    def __init__(self):
        self.clinic_split_result = pd.DataFrame()
        pass

    def __SetNewData(self, data_container, case_index):
        array, label, feature_name, case_name = data_container.GetData()

        new_array = array[case_index, :]
        new_label = label[case_index]
        new_case_name = [case_name[i] for i in case_index]

        new_data_container = DataContainer(array=new_array, label=new_label, case_name=new_case_name, feature_name=feature_name)
        new_data_container.UpdateFrameByData()
        return new_data_container

    def _SplitIndex(self, label, test_percentage):
        training_index_list, testing_index_list = [], []
        for group in range(int(np.max(label)) + 1):
            index = np.where(label == group)[0]

            shuffle(index)
            testing_index = index[:round(len(index) * test_percentage)]
            training_index = index[round(len(index) * test_percentage):]

            training_index_list.extend(training_index)
            testing_index_list.extend(testing_index)
        return training_index_list, testing_index_list

    def RunByTestingPercentage(self, data_container, testing_data_percentage=0.3, clinic_df=pd.DataFrame(),
                               store_folder='', max_loop=100):
        label = data_container.GetLabel()

        training_index_list, testing_index_list = [], []
        if clinic_df.size == 0:
            training_index_list, testing_index_list = self._SplitIndex(label, testing_data_percentage)
        else:
            assert(data_container.GetCaseName() == list(clinic_df.index))
            assert(max_loop >= 1)
            loop_count = 0
            analyzer = FeatureMatrixStatistics()

            for _ in range(max_loop):
                training_index_list, testing_index_list = self._SplitIndex(label, testing_data_percentage)
                train_clinic, test_clinic = clinic_df.iloc[training_index_list], clinic_df.iloc[testing_index_list]

                self.clinic_split_result = analyzer.CompareFeatures(train_clinic, test_clinic, ['train', 'test'])
                if self.clinic_split_result[self.clinic_split_result['P'] < 0.05].size == 0:
                    break
                loop_count += 1
            if loop_count == max_loop:
                print('Try to split {} times, but still failed.'.format(max_loop))
                training_index_list, testing_index_list = self._SplitIndex(label, testing_data_percentage)

        train_data_container = self.__SetNewData(data_container, training_index_list)
        test_data_container = self.__SetNewData(data_container, testing_index_list)

        if store_folder:
            train_data_container.Save(os.path.join(store_folder, 'train_numeric_feature.csv'))
            test_data_container.Save(os.path.join(store_folder, 'test_numeric_feature.csv'))
            if clinic_df.size > 0:
                self.clinic_split_result.to_csv(os.path.join(store_folder, 'clinical_split_statistics.csv'))

        return train_data_container, test_data_container

    def RunByTestingReference(self, data_container, testing_ref_data_container, store_folder=''):
        training_index_list, testing_index_list = [], []

        # TODO: assert data_container include all cases which is in the training_ref_data_container.
        all_name_list = data_container.GetCaseName()
        testing_name_list = testing_ref_data_container.GetCaseName()
        for training_name in testing_name_list:
            if training_name not in all_name_list:
                print('The data container and the training data container are not consistent.')
                return DataContainer(), DataContainer()

        for name, index in zip(all_name_list, range(len(all_name_list))):
            if name in testing_name_list:
                testing_index_list.append(index)
            else:
                training_index_list.append(index)

        train_data_container = self.__SetNewData(data_container, training_index_list)
        test_data_container = self.__SetNewData(data_container, testing_index_list)

        if store_folder:
            train_data_container.Save(os.path.join(store_folder, 'train_numeric_feature.csv'))
            test_data_container.Save(os.path.join(store_folder, 'test_numeric_feature.csv'))

        return train_data_container, test_data_container

if __name__ == '__main__':
    clinics = pd.read_csv(r'..\..\Demo\simulated_clinics.csv', index_col=0)
    container = DataContainer()
    container.Load(r'..\..\Demo\simulated_feature.csv')

    separator = DataSeparate()
    train, test = separator.RunByTestingPercentage(container, 0.3, clinic_df=clinics)

    print(train.GetArray().shape, test.GetArray().shape)
    print(separator.clinic_split_result)


