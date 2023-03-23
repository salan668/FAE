"""
All rights reserved.
--Yang Song, songyangmri@gmail.com
"""

import os

import numpy as np
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from scipy.stats import levene, ttest_ind, kstest, mannwhitneyu, chi2_contingency, normaltest
from collections import Counter

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.FeatureSelector import FeatureSelector


class FeatureType:
    Numeric = 1
    Categorical = 2

class OneFeatureStatistics(object):
    def __init__(self):
        pass

    def EstimateFeatureType(self, array):
        # 目前通过Unique方法进行判断
        if np.unique(array).size <= 9:
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


def GetPvalue(array1, array2, feature_type):
    assert feature_type in ['Category', 'Normal', 'Non-normal']

    def count_list(input):
        if not isinstance(input, list):
            input = list(input)
        dict = {}
        for i in set(input):
            dict[i] = input.count(i)
        return dict

    p_value = 0
    if feature_type == 'Category':
        count1 = count_list(array1)
        count2 = count_list(array2)  # dict
        categories = set(list(count1.keys()) + list(count2.keys()))
        contingency_dict = {}
        for category in categories:
            contingency_dict[category] = [count1[category] if count1[category] else 0,
                                          count2[category] if count2[category] else 0]

        contingency_pd = pd.DataFrame(contingency_dict)
        contingency_array = np.array(contingency_pd)
        _, p_value, _, _ = chi2_contingency(contingency_array)
    elif feature_type == 'Normal':
        _, p_value = ttest_ind(array1, array2)
    elif feature_type == 'Non-normal':
        _, p_value = mannwhitneyu(array1, array2)
    return p_value


class DataSplitterByFeatureCluster(object):
    def __init__(self, parts=30, repeat_times=100, test_ratio=0.3, random_seed=10):
        self.parts = parts
        self.repeat_times = repeat_times
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self.feature_labels = []
        self.current_dc = DataContainer()

    #################################################
    def _DataPreProcess(self, dc):
        data = dc.GetArray()  # get train data
        # min_max, Process the features of each column
        min_max_scaler = preprocessing.MinMaxScaler()
        processed_data = min_max_scaler.fit_transform(data).T
        return processed_data

    def _Cluster(self, dc):
        # According Cluster to selecte features and combine them into a DataContainer
        processed_data = self._DataPreProcess(dc)
        feature_name_list = dc.GetFeatureName()
        k_means = KMeans(n_clusters=self.parts, random_state=self.random_seed, init='k-means++')
        k_means.fit(processed_data)  # training

        count_label = [0 for _ in range(self.parts)]
        count_feature = [[] for _ in range(self.parts)]
        count_distance = [[] for _ in range(self.parts)]

        feature_predict = k_means.labels_
        cluster_centers = k_means.cluster_centers_

        for j in range(len(feature_name_list)):
            count_label[feature_predict[j]] += 1
            count_feature[feature_predict[j]].append(feature_name_list[j])

            cluster_center = cluster_centers[feature_predict[j]]
            distance = np.square(processed_data[j] - cluster_center).sum()
            count_distance[feature_predict[j]].append(distance)

        print('The number of feature in each class \n', count_label)
        min_distance_feature = []
        for k in range(self.parts):
            k_feature = count_feature[k]
            k_distance = count_distance[k]
            idx = k_distance.index(min(k_distance))
            selected_feature = k_feature[idx]
            min_distance_feature.append(selected_feature)
            print('min distance feature in this class {} is {}'.format(k, selected_feature))
            print('its distance is', min(k_distance), 'while max distance is', max(k_distance))
        return min_distance_feature, feature_predict

    def _MergeClinical(self, dc, cli_df):
        # Merge DataContainer and a dataframe of clinical
        if 'label' in cli_df.columns.tolist():
            del cli_df['label']
        elif 'Label' in cli_df.columns.tolist():
            del cli_df['Label']
        df = pd.merge(dc.GetFrame(), cli_df, how='left', left_index=True, right_index=True)
        merge_dc = DataContainer()
        merge_dc.SetFrame(df)
        merge_dc.UpdateFrameByData()
        return merge_dc

    def _EstimateAllFeatureDistribution(self, dc):
        feature_name_list = dc.GetFeatureName()
        distribution = dict()
        for i in range(len(feature_name_list)):
            feature = feature_name_list[i]
            feature_data = dc.GetFrame()[feature]
            _, normal_p = normaltest(feature_data, axis=0)
            if len(set(feature_data)) < 10:  # TODO: a better way to distinguish discrete numeric values
                distribution[feature] = 'Category'
            elif normal_p > 0.05:
                distribution[feature] = 'Normal'
            else:
                distribution[feature] = 'Non-normal'
        # return a dict {"AGE": 'Normal', 'Gender': 'Category', ... }
        return distribution

    def _EstimateDcFeaturePvalue(self, dc1, dc2, feature_type):
        array1, array2 = dc1.GetArray(), dc2.GetArray()
        p_values = {}
        for index, feature in enumerate(dc1.GetFeatureName()):
            p_values[feature] = GetPvalue(array1[:, index], array2[:, index], feature_type[feature])

        return p_values

    #################################################
    def VisualizePartsVariance(self, dc: DataContainer, max_k=None, method='SSE',
                               store_folder=None, is_show=True):
        # method must be one of SSE or SC. SSE denotes xxxx, SC denotes Silhouette Coefficient

        data = dc.GetArray()  # get train data
        processed_data = self._DataPreProcess(dc)

        if max_k is None:
            max_k = min(data.shape[0], 50)

        assert(method in ['SSE', 'SC'])

        score = []
        for k in range(2, max_k):
            print('make cluster k=', k)
            estimator = KMeans(n_clusters=k) 
            estimator.fit(processed_data)
            if method == 'SSE':
                score.append(estimator.inertia_)
            elif method == 'SC':
                score.append(silhouette_score(processed_data, estimator.labels_, metric='euclidean'))
        X = range(2, max_k)
        plt.xlabel('k')
        plt.ylabel(method)
        plt.plot(X, score, 'o-')

        if store_folder and os.path.isdir(store_folder):
            plt.savefig(os.path.join(store_folder, 'ClusteringParameterPlot.jpg'))

        if is_show:
            plt.show()

    def VisualizeCluster(self, dimension='2d', select_feature=None,
                         store_folder=None, is_show=True):
        if len(self.feature_labels) != 0 and self.current_dc.GetFrame().size != 0:
            processed_data = self._DataPreProcess(self.current_dc)

            if select_feature is None:
                select_feature = [0, 1, 2]

            assert dimension in ['2d', '3d']
            if dimension == '2d':
                plt.scatter(processed_data[:, select_feature[0]],
                            processed_data[:, select_feature[1]],
                            s=5, c=self.feature_labels)
            elif dimension == '3d':
                ax = plt.figure().add_subplot(111, projection='3d')
                ax.scatter(processed_data[:, select_feature[0]],
                           processed_data[:, select_feature[1]],
                           processed_data[:, select_feature[2]],
                           s=10, c=self.feature_labels, marker='^')
                ax.set_title('Cluster Result 3D')

            if store_folder and os.path.isdir(store_folder):
                plt.savefig(os.path.join(store_folder, 'ClusteringProcessPlot{}.jpg'.format(dimension)))
            if is_show:
                plt.show()

    def Run(self, dc: DataContainer, output_folder: str, clinical_feature=None):
        self.current_dc = dc
        selected_feature_names, self.feature_labels = self._Cluster(dc)

        fs = FeatureSelector()
        selected_dc = fs.SelectFeatureByName(dc, selected_feature_names)

        if clinical_feature is not None:
            if isinstance(clinical_feature, str):
                clinical_feature = pd.read_csv(clinical_feature, index_col=0)
            assert(isinstance(clinical_feature, pd.DataFrame))

            merge_dc = self._MergeClinical(selected_dc, clinical_feature)
        else:
            merge_dc = selected_dc

        feature_distribution_type = self._EstimateAllFeatureDistribution(merge_dc)  # a dict

        splitter = DataSeparate()

        output_train_dc, output_test_dc = DataContainer(), DataContainer()
        output_p_value = []
        mean_p_value = -1

        for _ in range(self.repeat_times):
            train_dc, test_dc = splitter.RunByTestingPercentage(merge_dc, testing_data_percentage=self.test_ratio)
            feature_p_value = self._EstimateDcFeaturePvalue(train_dc, test_dc, feature_distribution_type)
            if np.mean(list(feature_p_value.values())) > mean_p_value:
                mean_p_value = np.mean(list(feature_p_value.values()))
                output_train_dc, output_test_dc = train_dc, test_dc
                output_p_value = feature_p_value

        if output_folder is not None and os.path.isdir(output_folder):
            output_train_dc.Save(os.path.join(output_folder, 'train.csv'))
            output_test_dc.Save(os.path.join(output_folder, 'test.csv'))

            p_value_df = pd.DataFrame(output_p_value, index=['P Value'])
            distribute_df = pd.DataFrame(feature_distribution_type, index=['Distribution'])
            store_df = pd.concat((p_value_df, distribute_df), axis=0)
            store_df.to_csv(os.path.join(output_folder, 'split_info.csv'))


if __name__ == '__main__':
    # clinics = pd.read_csv(r'..\..\Demo\simulated_clinics.csv', index_col=0)
    # container = DataContainer()
    # container.Load(r'..\..\Demo\simulated_feature.csv')
    #
    # separator = DataSeparate()
    # train, test = separator.RunByTestingPercentage(container, 0.3, clinic_df=clinics)
    #
    # print(train.GetArray().shape, test.GetArray().shape)
    # print(separator.clinic_split_result)
    cluster_split = DataSplitterByFeatureCluster()
    container = DataContainer()
    container.Load(r'.\all_feature.csv')
    output_path = r'.\output'
    clinical_path = r'.\clinical.csv'
    cluster_split.VisualizePartsVariance(container, store_folder=output_path)
    cluster_split.Run(container, output_path, clinical_feature=clinical_path)
    cluster_split.VisualizeCluster(dimension='2d', store_folder=output_path)
    cluster_split.VisualizeCluster(dimension='3d', store_folder=output_path)
