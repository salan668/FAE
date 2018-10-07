from abc import ABCMeta,abstractmethod
import numpy as np
from copy import deepcopy
import pandas as pd
from random import randrange
import os
import numbers
import csv

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from FAE.FeatureAnalysis.ReliefF import ReliefF
from FAE.DataContainer.DataContainer import DataContainer


def SaveSelectInfo(data_container, store_path, is_merge=False):
    info = {}
    info['feature_number'] = len(data_container.GetFeatureName())
    if not is_merge:
        info['selected_feature'] = data_container.GetFeatureName()

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

class FeatureSelector:
    def __init__(self):
        self.__selector = None

    def SelectFeatureByIndex(self, data_container, selected_list, is_replace=False, store_path=''):
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
    def Run(self, data_container, store_folder):
        pass


#################################################################

class RemoveNonNumericFeature(FeatureSelector):
    def __init__(self):
        super(RemoveNonNumericFeature, self).__init__()

    def Run(self, data_container, store_folder=''):
        temp_frame = data_container.GetFrame().select_dtypes(include=None, exclude=['object'])
        new_data_container = DataContainer()
        new_data_container.SetFrame(temp_frame)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'numeric_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

        return new_data_container

class RemoveSameFeatures(FeatureSelector):
    def __init__(self):
        super(RemoveSameFeatures, self).__init__()

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        std_value = np.nan_to_num(np.std(data, axis=0))
        index = np.where(std_value == 0)[0]
        feature_list = list(range(data.shape[1]))
        for ind in index:
            feature_list.remove(ind)

        return feature_list

    def Run(self, data_container, store_folder=''):
        new_data_container = self.SelectFeatureByIndex(data_container, self.GetSelectedFeatureIndex(data_container), is_replace=False)
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

    def Run(self, data_container, store_folder=''):
        new_data_container = self.SelectFeatureByName(data_container, self.GetSelectFeaturedNameBySubName(data_container), is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)
        
        return new_data_container

#################################################################
class FeatureSelectByAnalysis(FeatureSelector):
    def __init__(self, selected_feature_number=0):
        super(FeatureSelectByAnalysis, self).__init__()
        self.__selected_feature_number = selected_feature_number

    def SetSelectedFeatureNumber(self, selected_feature_number):
        self.__selected_feature_number = selected_feature_number

    def GetSelectedFeatureNumber(self):
        return self.__selected_feature_number

    __metaclass__ = ABCMeta
    @abstractmethod
    def Run(self, data_container, store_folder):
        pass

    @abstractmethod
    def GetName(self):
        pass


class FeatureSelectByANOVA(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1):
        super(FeatureSelectByANOVA, self).__init__(selected_feature_number)

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        if data.shape[1] < self.GetSelectedFeatureNumber():
            print('ANOVA: The number of features {:d} in data container is smaller than the required number {:d}'.format(
                data.shape[1], self.GetSelectedFeatureNumber()))
            self.SetSelectedFeatureNumber(data.shape[1])

        fs = SelectKBest(f_classif, k=self.GetSelectedFeatureNumber())
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        f_value, p_value = f_classif(data, label)
        return feature_index.tolist(), f_value, p_value

    def GetName(self):
        return 'ANOVA'

    def GetDescription(self):
        text = "Before build the model, we used analysis of variance (ANOVA) to select features. ANOVA was a commen method " \
               "to explore the significant features corresponding to the labels. F-value was calculated to evaluate the relationship " \
               "between features and the label. We sorted features according to the corresponding F-value and selected sepcific " \
               "number of features to build the model. "
        return text

    def Run(self, data_container, store_folder=''):
        selected_index, f_value, p_value = self.GetSelectedFeatureIndex(data_container)
        new_data_container = self.SelectFeatureByIndex(data_container, selected_index, is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

            anova_sort_path = os.path.join(store_folder, 'anova_sort.csv')
            df = pd.DataFrame(data=np.stack((f_value, p_value), axis=1), index=data_container.GetFeatureName(), columns=['F', 'P'])
            df.to_csv(anova_sort_path)

        return new_data_container

class FeatureSelectByRelief(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1, iter_ratio=0.7):
        super(FeatureSelectByRelief, self).__init__(selected_feature_number)
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
        sorted_feature_list = sorted(feature_list, key=lambda x: x[0], reverse=True)
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

            # start iteration
        for iter_num in range(int(self.__iter_radio * n_samples)):
            # print iter_num;
            # initialization
            nearHit = list()
            nearMiss = list()
            distance_sort = list()

            # random extract a sample
            index_i = randrange(0, n_samples, 1)
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

                    # update weight
            weight = weight - np.power(self_features - nearHit, 2) + np.power(self_features - nearMiss, 2)
        result = self.__SortByValue(weight / (self.__iter_radio * n_samples))
        self._weight = weight
        return result

    def GetSelectedFeatureIndex(self, data_container):
        feature_sort_list = self.__SortByRelief(data_container)
        if len(feature_sort_list) < self.GetSelectedFeatureNumber():
            print('Relief: The number of features {:d} in data container is smaller than the required number {:d}'.format(len(feature_sort_list), self.GetSelectedFeatureNumber()))
            self.SetSelectedFeatureNumber(len(feature_sort_list))
        selected_feature_index = feature_sort_list[:self.GetSelectedFeatureNumber()]
        return selected_feature_index

    def GetName(self):
        return 'Relief'

    def GetDescription(self):
        text = "Before build the model, we used Relief to select features. Relief selects sub data set and find the " \
               "relative features according to the label recursively. "
        return text

    def Run(self, data_container, store_folder=''):
        new_data_container = self.SelectFeatureByIndex(data_container, self.GetSelectedFeatureIndex(data_container), is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            relief_sort_path = os.path.join(store_folder, 'Relief_sort.csv')
            df = pd.DataFrame(data=self._weight, index=data_container.GetFeatureName(), columns=['weight'])
            df.to_csv(relief_sort_path)

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)
        return new_data_container

class FeatureSelectByRFE(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1, classifier=SVC(kernel='linear')):
        super(FeatureSelectByRFE, self).__init__(selected_feature_number)
        self.__classifier = classifier

    def GetDescription(self):
        text = "Before build the model, we used recursive feature elimination (RFE) to select features. The goal of RFE " \
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

        fs = RFE(self.__classifier, self.GetSelectedFeatureNumber(), step=0.05)
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        ranks = fs.ranking_

        return feature_index.tolist(), ranks

    def GetName(self):
        return 'RFE'

    def Run(self, data_container, store_folder=''):
        selected_index, rank = self.GetSelectedFeatureIndex(data_container)
        new_data_container = self.SelectFeatureByIndex(data_container, selected_index, is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

            rfe_sort_path = os.path.join(store_folder, 'RFE_sort.csv')
            df = pd.DataFrame(data=rank, index=data_container.GetFeatureName(), columns=['rank'])
            df.to_csv(rfe_sort_path)

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
        except:
            print('The last selector does not have method SetSelectedFeatureNumber')

    def GetSelectedFeatureNumber(self):
        return self.__selected_feature_number

    def GetName(self):
        try:
            return self.__selector_list[-1].GetName()
        except:
            print('The last selector does not have method GetName')

    #TODO: Add verbose parameter to show the removed feature name in each selector
    def Run(self, data_container, store_folder=''):
        input_data_container = data_container
        for fs in self.__selector_list:
            output = fs.Run(input_data_container, store_folder)
            input_data_container = output
        return output

################################################################

if __name__ == '__main__':
    import os
    print(os.getcwd())
    from FAE.DataContainer.DataContainer import DataContainer
    data_container = DataContainer()
    print(os.path.abspath(r'..\..\Example\numeric_feature.csv'))
    data_container.Load(r'..\..\Example\numeric_feature.csv')
    # data_container.UsualNormalize()

    print(data_container.GetArray().shape)
    print(data_container.GetFeatureName())

    fs = FeatureSelectBySubName(['shape', 'ADC'])

    output = fs.Run(data_container)
    print(output.GetFeatureName())

    # fs1 = RemoveNonNumericFeature()
    # fs1.SetDataContainer(data_container)
    # non_number_data_container = fs1.Run()
    #
    # fs2 = FeatureSelectByANOVA(10)
    # fs2.SetDataContainer(non_number_data_container)
    # output = fs2.Run()

    # feature_selector_list = [RemoveNonNumericFeature(), RemoveCosSimilarityFeatures(), FeatureSelectByANOVA(5)]
    # feature_selector_pipeline = FeatureSelectPipeline(feature_selector_list)
    # feature_selector_pipeline.SetDataContainer(data_container)
    # output = feature_selector_pipeline.Run()

    print(output.GetArray().shape)
