import os
import numbers
import csv

# import pymrmr
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.svm import SVC
from scipy.stats import kruskal
from abc import ABCMeta, abstractmethod
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.utils import safe_mask

from BC.HyperParameterConfig.HyperParamManager import HyperParameterManager


def SaveSelectInfo(feature_name, store_path, is_merge=False):
    info = {}
    info['feature_number'] = len(feature_name)
    if not is_merge:
        info['selected_feature'] = feature_name

    write_info = []
    for key in info.keys():
        temp_list = []
        temp_list.append(key)
        if isinstance(info[key], (numbers.Number, str)):
            temp_list.append(info[key])
        else:
            temp_list.extend(info[key])
        write_info.append(temp_list)

    with open(os.path.join(store_path), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(write_info)

def LoadSelectInfo(store_path):
    with open(os.path.join(store_path), 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        feature_number = 0
        selected_features = []
        for row in reader:
            if row[0] == 'feature_number':
                feature_number = row[1]
            if row[0] == 'selected_feature':
                selected_features = row[1:]
    return feature_number, selected_features


class FeatureSelector(object):
    def __init__(self):
        pass

    def __deepcopy__(self, memodict={}):
        copy_selector = type(self)()
        return copy_selector

    def SelectFeatureByIndex(self, data_container, selected_list, is_replace=False, store_path=''):
        selected_list = sorted(selected_list)
        new_data = data_container.GetArray()[:, selected_list]
        new_feature = [data_container.GetFeatureName()[t] for t in selected_list]

        if is_replace:
            data_container.SetArray(new_data)
            data_container.SetFeatureName(new_feature)
            data_container.UpdateFrameByData()
            new_data_container = deepcopy(data_container)
        else:
            new_data_container = deepcopy(data_container)
            new_data_container.SetArray(new_data)
            new_data_container.SetFeatureName(new_feature)
            new_data_container.UpdateFrameByData()
        if store_path:
            new_data_container.Save(store_path)

        return new_data_container

    def SelectFeatureByName(self, data_container, selected_feature_name, is_replace=False, store_path=''):
        new_data = data_container.GetFrame()[selected_feature_name].values

        if is_replace:
            data_container.SetArray(new_data)
            data_container.SetFeatureName(selected_feature_name)
            data_container.UpdateFrameByData()
            new_data_container = deepcopy(data_container)
        else:
            new_data_container = deepcopy(data_container)
            new_data_container.SetArray(new_data)
            new_data_container.SetFeatureName(selected_feature_name)
            new_data_container.UpdateFrameByData()
        if store_path:
            new_data_container.Save(store_path)

        return new_data_container

    def GetDescription(self):
        text = "Since the number of features is not too high, we did not apply any feature selection method here. " \
               "All features were used to build the final model. "
        return text

    __metaclass__ = ABCMeta

    @abstractmethod
    def Run(self, data_container, store_folder='', store_key=''):
        pass


#################################################################

class RemoveNonNumericFeature(FeatureSelector):
    def __init__(self):
        super(RemoveNonNumericFeature, self).__init__()

    def Run(self, data_container, store_folder='', store_key=''):
        temp_frame = data_container.GetFrame().select_dtypes(include=None, exclude=['object'])
        new_data_container = DataContainer()
        new_data_container.SetFrame(temp_frame)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'numeric_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container.GetFeatureName(), featureinfo_store_path, is_merge=False)

        return new_data_container


class RemoveSameFeatures(FeatureSelector):
    def __init__(self):
        super(RemoveSameFeatures, self).__init__()

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        feature_list = []
        for feature_index in range(data.shape[1]):
            feature = data[:, feature_index]
            unique, counts = np.unique(feature, return_counts=True)
            if np.max(counts) / np.sum(counts) < 0.9:  # This is arbitrarily
                feature_list.append(feature_index)
        return feature_list

    def Run(self, data_container, store_folder='', store_key=''):
        new_data_container = self.SelectFeatureByIndex(data_container, self.GetSelectedFeatureIndex(data_container),
                                                       is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

        return new_data_container


class FeatureSelectBySubName(FeatureSelector):
    def __init__(self, sub_name_list):
        super(FeatureSelectBySubName, self).__init__()
        if isinstance(sub_name_list, str):
            sub_name_list = [sub_name_list]

        self.__sub_name_list = sub_name_list

    def GetSelectFeaturedNameBySubName(self, data_container):
        all_feature_name = data_container.GetFeatureName()
        selected_feature_name_list = []
        for selected_sub_name in self.__sub_name_list:
            for feature_name in all_feature_name:
                if selected_sub_name in feature_name:
                    selected_feature_name_list.append(feature_name)

        selected_feature_name_list = list(sorted(set(selected_feature_name_list)))
        return selected_feature_name_list

    def Run(self, data_container, store_folder='', store_key=''):
        new_data_container = self.SelectFeatureByName(data_container,
                                                      self.GetSelectFeaturedNameBySubName(data_container),
                                                      is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

        return new_data_container


#################################################################
class FeatureSelectByAnalysis(FeatureSelector):
    def __init__(self, name='', selected_feature_number=0):
        super(FeatureSelectByAnalysis, self).__init__()
        self.__selected_feature_number = selected_feature_number
        self._selected_features = []
        self._raw_features = []
        self._name = name

    def SetSelectedFeatureNumber(self, selected_feature_number):
        self.__selected_feature_number = selected_feature_number

    def GetSelectedFeatureNumber(self):
        return self.__selected_feature_number

    def Transform(self, container, store_folder='', store_key=''):
        if container.IsEmpty():
            return container

        fs_container = self.SelectFeatureByName(container, self._selected_features)
        if store_folder and os.path.isdir(store_folder):
            self.SaveDataContainer(fs_container, store_folder, store_key)
        return fs_container

    def SaveDataContainer(self, data_container, store_folder, store_key):
        if store_folder:
            assert(len(store_key) > 0)
            feature_store_path = os.path.join(store_folder, '{}_{}_feature.csv'.format(self._name, store_key))
            data_container.Save(feature_store_path)

    def GetName(self):
        return self._name

    __metaclass__ = ABCMeta

    @abstractmethod
    def Run(self, data_container, store_folder):
        pass

    @abstractmethod
    def SaveInfo(self, store_folder):
        pass


class FeatureSelectByANOVA(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1):
        super(FeatureSelectByANOVA, self).__init__(name='ANOVA', selected_feature_number=selected_feature_number)
        self._f_value = np.array([])
        self._p_value = np.array([])

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        if data.shape[1] < self.GetSelectedFeatureNumber():
            print(
                'ANOVA: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                    data.shape[1], self.GetSelectedFeatureNumber()))
            self.SetSelectedFeatureNumber(data.shape[1])

        fs = SelectKBest(f_classif, k=self.GetSelectedFeatureNumber())
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        f_value, p_value = f_classif(data, label)
        return feature_index.tolist(), f_value, p_value

    def SaveInfo(self, store_folder):
        anova_sort_path = os.path.join(store_folder, '{}_sort.csv'.format(self._name))
        df = pd.DataFrame(data=np.stack((self._f_value, self._p_value), axis=1), index=self._raw_features,
                          columns=['F', 'P'])
        df.to_csv(anova_sort_path)

        featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')
        SaveSelectInfo(self._selected_features, featureinfo_store_path, is_merge=False)


    def GetDescription(self):
        text = "Before build the model, we used analysis of variance (ANOVA) to select features. ANOVA was a common method " \
               "to explore the significant features corresponding to the labels. F-value was calculated to evaluate the relationship " \
               "between features and the label. We sorted features according to the corresponding F-value and selected sepcific " \
               "number of features to build the model. "
        return text

    def Run(self, data_container, store_folder='', store_key=''):
        self._raw_features = data_container.GetFeatureName()
        selected_index, self._f_value, self._p_value = self.GetSelectedFeatureIndex(data_container)
        new_data_container = self.SelectFeatureByIndex(data_container, selected_index, is_replace=False)
        self._selected_features = new_data_container.GetFeatureName()
        if store_folder and os.path.isdir(store_folder):
            self.SaveInfo(store_folder)
            self.SaveDataContainer(new_data_container, store_folder, store_key)

        return new_data_container


class FeatureSelectByRelief(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1, iter_ratio=1):
        super(FeatureSelectByRelief, self).__init__(name='Relief', selected_feature_number=selected_feature_number)
        self.__iter_radio = iter_ratio
        self._weight = None

    def __SortByValue(self, feature_score):
        feature_list = []
        sorted_feature_number_list = []
        for feature_index in range(len(feature_score)):
            feature_list_unit = []
            feature_list_unit.append(feature_score[feature_index])
            feature_list_unit.append(feature_index)
            feature_list.append(feature_list_unit)
        sorted_feature_list = sorted(feature_list, key=lambda x: abs(x[0]), reverse=True)
        for feature_index in range(len(sorted_feature_list)):
            sorted_feature_number_list.append(sorted_feature_list[feature_index][1])

        return sorted_feature_number_list

    def __DistanceNorm(self, Norm, D_value):
        # initialization

        # Norm for distance
        if Norm == '1':
            counter = np.absolute(D_value)
            counter = np.sum(counter)
        elif Norm == '2':
            counter = np.power(D_value, 2)
            counter = np.sum(counter)
            counter = np.sqrt(counter)
        elif Norm == 'Infinity':
            counter = np.absolute(D_value)
            counter = np.max(counter)
        else:
            raise Exception('We will program this later......')

        return counter

    def __SortByRelief(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        # initialization
        (n_samples, n_features) = np.shape(data)
        distance = np.zeros((n_samples, n_samples))
        weight = np.zeros(n_features)

        if self.__iter_radio >= 0.5:
            # compute distance
            for index_i in range(n_samples):
                for index_j in range(index_i + 1, n_samples):
                    D_value = data[index_i] - data[index_j]
                    distance[index_i, index_j] = self.__DistanceNorm('2', D_value)
            distance += distance.T
        else:
            pass

        # Start Iteration
        for iter_num in range(int(self.__iter_radio * n_samples)):
            # initialization
            nearHit = list()
            nearMiss = list()
            distance_sort = list()

            # random extract a sample
            index_i = iter_num
            self_features = data[index_i]

            # search for nearHit and nearMiss
            if self.__iter_radio >= 0.5:
                distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
                for index in range(n_samples):
                    distance_sort.append([distance[index_i, index], index, label[index]])
            else:
                # compute distance respectively
                distance = np.zeros(n_samples)
                for index_j in range(n_samples):
                    D_value = data[index_i] - data[index_j]
                    distance[index_j] = self.__DistanceNorm('2', D_value)
                distance[index_i] = np.max(distance)  # filter self-distance
                for index in range(n_samples):
                    distance_sort.append([distance[index], index, label[index]])
            distance_sort.sort(key=lambda x: x[0])
            for index in range(n_samples):
                if nearHit == [] and distance_sort[index][2] == label[index_i]:
                    # nearHit = distance_sort[index][1];
                    nearHit = data[distance_sort[index][1]]
                elif nearMiss == [] and distance_sort[index][2] != label[index_i]:
                    # nearMiss = distance_sort[index][1]
                    nearMiss = data[distance_sort[index][1]]
                elif nearHit != [] and nearMiss != []:
                    break
                else:
                    continue

            weight = weight - np.power(self_features - nearHit, 2) + np.power(self_features - nearMiss, 2)
        result = self.__SortByValue(weight / (self.__iter_radio * n_samples))
        self._weight = weight
        return result

    def SaveInfo(self, store_folder):
        relief_sort_path = os.path.join(store_folder, '{}_sort.csv'.format(self._name))
        df = pd.DataFrame(data=self._weight, index=self._raw_features, columns=['weight'])
        df.to_csv(relief_sort_path)

        featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')
        SaveSelectInfo(self._selected_features, featureinfo_store_path, is_merge=False)

    def GetSelectedFeatureIndex(self, data_container):
        feature_sort_list = self.__SortByRelief(data_container)
        if len(feature_sort_list) < self.GetSelectedFeatureNumber():
            print(
                'Relief: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                    len(feature_sort_list), self.GetSelectedFeatureNumber()))
            self.SetSelectedFeatureNumber(len(feature_sort_list))
        selected_feature_index = feature_sort_list[:self.GetSelectedFeatureNumber()]
        return selected_feature_index

    def GetDescription(self):
        text = "Before build the model, we used Relief to select features. Relief selects sub data set and find the " \
               "relative features according to the label recursively. "
        return text

    def Run(self, data_container, store_folder='', store_key=''):
        self._raw_features = data_container.GetFeatureName()
        new_data_container = self.SelectFeatureByIndex(data_container, self.GetSelectedFeatureIndex(data_container),
                                                       is_replace=False)
        self._selected_features = new_data_container.GetFeatureName()
        if store_folder and os.path.isdir(store_folder):
            self.SaveInfo(store_folder)
            self.SaveDataContainer(new_data_container, store_folder, store_key)

        return new_data_container


class FeatureSelectByRFE(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1, classifier=SVC(kernel='linear')):
        super(FeatureSelectByRFE, self).__init__(name='RFE', selected_feature_number=selected_feature_number)
        self.__classifier = classifier
        self._rank = None
        self._selected_features = []

    def GetDescription(self):
        text = "Before building the model, we used recursive feature elimination (RFE) to select features. The goal of RFE " \
               "is to select features based on a classifier by recursively considering smaller set of the features. "
        return text

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        if data.shape[1] < self.GetSelectedFeatureNumber():
            print('RFE: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                data.shape[1], self.GetSelectedFeatureNumber()))
            self.SetSelectedFeatureNumber(data.shape[1])

        fs = RFE(self.__classifier, n_features_to_select=self.GetSelectedFeatureNumber(), step=0.05)
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        self._rank = fs.ranking_

        return feature_index.tolist()

    def SaveInfo(self, store_folder):
        # TODO: There should not have all_features variable
        rfe_sort_path = os.path.join(store_folder, '{}_sort.csv'.format(self._name))
        assert (self._rank.size == len(self._raw_features))
        df = pd.DataFrame(data=self._rank, index=self._raw_features, columns=['rank'])
        df.to_csv(rfe_sort_path)

        featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')
        SaveSelectInfo(self._selected_features, featureinfo_store_path, is_merge=False)

    def Run(self, data_container, store_folder='', store_key=''):
        self._raw_features = data_container.GetFeatureName()
        selected_index = self.GetSelectedFeatureIndex(data_container)
        new_data_container = self.SelectFeatureByIndex(data_container, selected_index, is_replace=False)
        self._selected_features = new_data_container.GetFeatureName()
        if store_folder and os.path.isdir(store_folder):
            self.SaveInfo(store_folder)
            self.SaveDataContainer(new_data_container, store_folder, store_key)

        return new_data_container


class FeatureSelectByMrmr(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1):
        super(FeatureSelectByMrmr, self).__init__(name='mRMR', selected_feature_number=selected_feature_number)
        self._hyper_parameter_manager = HyperParameterManager()

    def GetDescription(self):
        text = "Before build the model, we used minimum-Redundancy-Maximum-Relevance (mRMR) to select features. The goal of mRMR " \
               "is to select a feature subset set that best characterizes the statistical property of a target classification variable," \
               "subject to the constraint that these features are mutually as dissimilar to each other as possible, but marginally as similar to the classification variable as possible."
        return text

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        if data.shape[1] < self.GetSelectedFeatureNumber():
            print('mMRM: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                data.shape[1], self.GetSelectedFeatureNumber()))
            self.SetSelectedFeatureNumber(data.shape[1])

        feature_list = ['class'] + data_container.GetFeatureName()
        feature_index = []
        pd_label = pd.DataFrame(label)
        pd_data = pd.DataFrame(data)
        mRMR_input = pd.concat([pd_label, pd_data], axis=1)
        mRMR_input.columns = feature_list
        parameter_list = self.LoadFeatureSelectorParameterList(relative_path=r'HyperParameters\FeatureSelector')
        feature_name = pymrmr.mRMR(mRMR_input, parameter_list[0]['mutual_information'], self.GetSelectedFeatureNumber())
        feature_list.remove('class')

        rank = []
        for index, item in enumerate(feature_name):
            feature_index.append(feature_list.index(item))
            rank.append(index)
        return feature_index, rank, feature_name

    def GetName(self):
        return 'mRMR'

    def LoadFeatureSelectorParameterList(self, relative_path=os.path.join('HyperParameters', 'FeatureSelector')):
        self._hyper_parameter_manager.LoadSpecificConfig(self.GetName(), relative_path=relative_path)
        parameter_list = self._hyper_parameter_manager.GetParameterSetting()
        return parameter_list

    def Run(self, data_container, store_folder=''):
        self._raw_features = data_container.GetFeatureName()
        selected_index, rank, feature_name = self.GetSelectedFeatureIndex(data_container)
        new_data_container = self.SelectFeatureByIndex(data_container, selected_index, is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

            mrmr_sort_path = os.path.join(store_folder, 'mMRM_sort.csv')
            df = pd.DataFrame(data=rank, index=feature_name, columns=['rank'])
            df.to_csv(mrmr_sort_path)

        return new_data_container


class FeatureSelectByKruskalWallis(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1):
        super(FeatureSelectByKruskalWallis, self).__init__(name='KW', selected_feature_number=selected_feature_number)
        self._f_value = np.array([])
        self._p_value = np.array([])

    def KruskalWallisAnalysis(self, array, label):
        args = [array[safe_mask(array, label == k)] for k in np.unique(label)]
        neg, pos = args[0], args[1]
        f_list, p_list = [], []
        for index in range(array.shape[1]):
            f, p = kruskal(neg[:, index], pos[:, index])
            f_list.append(f), p_list.append(p)
        return np.array(f_list), np.array(p_list)

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        if data.shape[1] < self.GetSelectedFeatureNumber():
            print('KW: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                data.shape[1], self.GetSelectedFeatureNumber()))
            self.SetSelectedFeatureNumber(data.shape[1])

        fs = SelectKBest(self.KruskalWallisAnalysis, k=self.GetSelectedFeatureNumber())
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        self._f_value, self._p_value = self.KruskalWallisAnalysis(data, label)
        return feature_index.tolist()

    def SaveInfo(self, store_folder):
        anova_sort_path = os.path.join(store_folder, '{}_sort.csv'.format(self._name))
        df = pd.DataFrame(data=np.stack((self._f_value, self._p_value), axis=1), index=self._raw_features,
                          columns=['F', 'P'])
        df.to_csv(anova_sort_path)

        featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')
        SaveSelectInfo(self._selected_features, featureinfo_store_path, is_merge=False)


    def GetDescription(self):
        text = "Before build the model, we used Kruskal Wallis to select features. KruskalWallis was a common method " \
               "to explore the significant features corresponding to the labels. F-value was calculated to evaluate " \
               "the relationship between features and the label. We sorted features according to the corresponding " \
               "F-value and selected top N features according to validation performance."
        return text

    def Run(self, data_container, store_folder='', store_key=''):
        self._raw_features = data_container.GetFeatureName()
        selected_index = self.GetSelectedFeatureIndex(data_container)
        new_data_container = self.SelectFeatureByIndex(data_container, selected_index, is_replace=False)
        self._selected_features = new_data_container.GetFeatureName()
        if store_folder and os.path.isdir(store_folder):
            self.SaveInfo(store_folder)
            self.SaveDataContainer(new_data_container, store_folder, store_key)

        return new_data_container


################################################################

class FeatureSelectPipeline(FeatureSelector):
    def __init__(self, selector, selected_feature_number=0):
        if isinstance(selector, FeatureSelector):
            selector = [selector]
        self.__selector_list = selector
        self.__selected_feature_number = selected_feature_number

    def SetSelectedFeatureNumber(self, selected_feature_number):
        self.__selected_feature_number = selected_feature_number
        try:
            self.__selector_list[-1].SetSelectedFeatureNumber(selected_feature_number)
        except Exception as e:
            content = 'In FeatureSelectPipeline, the last selector does not have method SetSelectedFeaturNumber: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

    def GetSelectedFeatureNumber(self):
        return self.__selected_feature_number

    def GetName(self):
        try:
            return self.__selector_list[-1].GetName()
        except Exception as e:
            content = 'In FeatureSelectPipeline, the last selector does not have method GetName: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

    # TODO: Add verbose parameter to show the removed feature name in each selector
    def Run(self, data_container, store_folder='', store_key=''):
        input_data_container = data_container
        for fs in self.__selector_list:
            output = fs.Run(input_data_container, store_folder, store_key)
            input_data_container = output
        return output

    def SaveInfo(self, store_folder, all_features):
        for fs in self.__selector_list:
            fs.SaveInfo(store_folder, all_features)

    def SaveDataContainer(self, data_container, store_folder, store_key):
        for fs in self.__selector_list:
            fs.SaveDataContainer(data_container, store_folder, store_key)


################################################################

if __name__ == '__main__':
    from BC.DataContainer.DataContainer import DataContainer
    from BC.FeatureAnalysis.Normalizer import NormalizerZeroCenter
    from BC.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC

    dc = DataContainer()
    pcc = DimensionReductionByPCC()
    fs = FeatureSelectByRelief(selected_feature_number=8)

    dc.Load(r'..\..\Demo\train_numeric_feature.csv')

    dc = NormalizerZeroCenter.Run(dc)
    dc = pcc.Run(dc)
    print(dc.GetArray().shape)
    dc = fs.Run(dc)
    print(dc.GetArray().shape)
