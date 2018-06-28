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
        new_data = data_container.GetFrame()[selected_feature_name].as_matrix()

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

    __metaclass__ = ABCMeta
    @abstractmethod
    def Run(self, data_container):
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

class RemoveCosSimilarityFeatures(FeatureSelector):
    def __init__(self, threshold=0.86):
        super(RemoveCosSimilarityFeatures, self).__init__()
        self.__threshold = threshold

    def __CosSimilarity(self, data1, data2):
        return np.abs(np.dot(data1, data2))

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)

        selected_feature_list = []
        for feature_index in range(data.shape[1]):
            is_similar = False
            for save_index in selected_feature_list:
                if self.__CosSimilarity(data[:, save_index], data[:, feature_index]) > self.__threshold:
                    is_similar = True
                    break
            if not is_similar:
                selected_feature_list.append(feature_index)

        return selected_feature_list

    def Run(self, data_container, store_folder=''):
        new_data_container = self.SelectFeatureByIndex(data_container, self.GetSelectedFeatureIndex(data_container), is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

        return new_data_container

class FeatureSelectByKeyName(FeatureSelector):
    def __init__(self, selected_list, method='or'):
        super(FeatureSelectByKeyName, self).__init__()
        self.__selected_list = selected_list
        self.__method = method

    def __SelectFeatureName(self, data_container):
        temp_df = pd.DataFrame(data=data_container.GetArray(),
                               columns=data_container.GetFeatureName(),
                               index=data_container.GetCaseName())
        sub_names = temp_df.columns.str.split('_', expand=True)

        feature_level = sub_names.levels
        feature_label = sub_names.labels

        feature_table = []
        for i in range(len(feature_level)):
            fea_list = feature_level[i]
            fea_name_no = pd.Series([i for i in range(len(fea_list))], index=fea_list)
            feature_table.append(fea_name_no)

        return temp_df.columns, feature_level, feature_label, feature_table

    # coding the selection like [False, False, True], [False, True, True], [True, False, True]
    # depend on SelectFeatures.
    # need fean_label and feacorres_table produced by SplitFeatureName
    def __CodeSelection(self, sele_list, fean_label, fea_corres_table):
        seles = []
        for i in range(len(sele_list)):
            if sele_list:
                for fean in sele_list[i]:
                    sele = (fean_label[i] == fea_corres_table[i][fean])
                    seles.append(sele)
        if not seles:
            print('No paired selection.')
            return -1
        return seles

    # according to the codedesele like
    # [[False, False, True], [False, True, True], [True, False, True]]
    # need codedsele produced by CodeSelection.
    def __MergeSelection(self, codedsele):
        sele_sum = codedsele[0]

        if self.__method == 'and':
            for sele in codedsele:
                sele_sum = (sele & sele_sum)
                # print("sele_sum:",sele_sum)
        if self.__method == 'or':
            for sele in codedsele:
                sele_sum = (sele | sele_sum)
                # print("sele_sum:",sele_sum)

        # if all boolean in sele_sum are False print no matching
        sele_sum_list = sele_sum.tolist()
        # print ('True in sele_sum:',sele_sum_list.count(True))
        if sele_sum_list.count(True) == 0:
            # print('No matching column\'s name.')
            return -1

        return sele_sum

    #### Feature Split was coded by Lao Wang
    # May-09-18.
    # depend on CodeSelection and MergeSelection
    def GetSelectFeaturedNameByKeyName(self, data_container):
        columns, feature_level, feature_label, feature_table = self.__SelectFeatureName(data_container)
        codedsele = self.__CodeSelection(self.__selected_list, feature_label, feature_table)
        selection = self.__MergeSelection(codedsele)
        selection_columns = columns[selection]
        return list(selection_columns)

    def Run(self, data_container, store_folder=''):
        new_data_container = self.SelectFeatureByName(data_container, self.GetSelectFeaturedNameByKeyName(data_container), is_replace=False)
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
    def Run(self, store_folder):
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
            print('The number of features in data container is smaller than the required number')
            self.SetSelectedFeatureNumber(data.shape[1])

        fs = SelectKBest(f_classif, k=self.GetSelectedFeatureNumber())
        fs.fit(data, label)
        feature_index = fs.get_support(True)
        f_value, p_value = f_classif(data, label)
        return feature_index.tolist(), f_value, p_value

    def GetName(self):
        return 'ANOVA'

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
    def __init__(self, selected_feature_number=1, iter_ratio=0.6):
        super(FeatureSelectByRelief, self).__init__(selected_feature_number)
        self.__iter_radio = iter_ratio

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        if data.shape[1] < self.GetSelectedFeatureNumber():
            print('The number of features in data container is smaller than the required number')
            self.SetSelectedFeatureNumber(data.shape[1])

        relief = ReliefF(n_features_to_keep=self.GetSelectedFeatureNumber())
        relief.fit(data, label)
        selected_feature_index = relief.get_support()
        score = relief.get_score()

        return selected_feature_index.tolist(), score

    def GetName(self):
        return 'Relief'

    def Run(self, data_container, store_folder=''):
        selected_index, score = self.GetSelectedFeatureIndex(data_container)
        new_data_container = self.SelectFeatureByIndex(data_container, selected_index, is_replace=False)
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=False)

            relief_sort_path = os.path.join(store_folder, 'relief_sort.csv')
            df = pd.DataFrame(data=score, index=data_container.GetFeatureName(), columns=['Scores'])
            df.to_csv(relief_sort_path)

        return new_data_container

class FeatureSelectByRFE(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1, classifier=SVC(kernel='linear')):
        super(FeatureSelectByRFE, self).__init__(selected_feature_number)
        self.__classifier = classifier

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)
        label = data_container.GetLabel()

        if data.shape[1] < self.GetSelectedFeatureNumber():
            print('The number of features in data container is smaller than the required number')
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

class FeatureSelectByPCA(FeatureSelectByAnalysis):
    def __init__(self, selected_feature_number=1):
        super(FeatureSelectByPCA, self).__init__()
        self.__selected_feature_number = selected_feature_number

    def SetSelectedFeatureNumber(self, selected_feature_number):
        self.__selected_feature_number = selected_feature_number

    def GetName(self):
        return 'PCA'

    def Run(self, data_container, store_folder=''):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)

        if data.shape[1] < self.__selected_feature_number:
            print('The number of features in data container is smaller than the required number')
            self.__selected_feature_number = data.shape[1]

        fs = PCA(n_components=self.__selected_feature_number)
        fs.fit(data)
        sub_data = fs.transform(data)
        sub_feature_name = ['PCA_feature_'+str(index) for index in range(1, self.__selected_feature_number + 1 )]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(sub_data)
        new_data_container.SetFeatureName(sub_feature_name)
        new_data_container.UpdateFrameByData()
        if store_folder and os.path.isdir(store_folder):
            feature_store_path = os.path.join(store_folder, 'selected_feature.csv')
            featureinfo_store_path = os.path.join(store_folder, 'feature_select_info.csv')

            new_data_container.Save(feature_store_path)
            SaveSelectInfo(new_data_container, featureinfo_store_path, is_merge=True)

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

    fs = FeatureSelectByKeyName([[], [], ['shape', 'firstorder', 'glrlm'], []], method='or')

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