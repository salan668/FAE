from abc import abstractmethod
import numpy as np
import pickle
import os
from copy import deepcopy
import pandas as pd
from scipy.stats import pearsonr

from FAE.DataContainer.DataContainer import DataContainer
from sklearn.decomposition import PCA

class DimensionReduction:
    def __init__(self, model=None, number=0, is_transform=False):
        self.__model = model
        self.__remained_number = number
        self.__is_transform=is_transform

    def SetModel(self, model):
        self.__model = deepcopy(model)

    def GetModel(self):
        return self.__model

    def SetRemainedNumber(self, number):
        self.__remained_number = number

    def GetRemainedNumber(self):
        return self.__remained_number

    def SetTransform(self, is_transform):
        self.__is_transform = is_transform

    def GetTransform(self):
        return self.__is_transform

    def GetDescription(self):
        text = "Since the dimension of feature space is low enough, we did not used any dimension reduction method " \
               "here to reduce the dimension of feature space. "
        return text

class DimensionReductionByPCA(DimensionReduction):
    def __init__(self, number=0):
        super(DimensionReductionByPCA, self).__init__(number=number, is_transform=True)
        super(DimensionReductionByPCA, self).SetModel(PCA(n_components=super(DimensionReductionByPCA, self).GetRemainedNumber()))

        self.__raw_feature_name = []
        self.__pca_feature_name = []

    def GetName(self):
        return 'PCA'

    def SaveInfo(self, store_folder):
        pca_sort_path = os.path.join(store_folder, 'pca_sort.csv')
        df = pd.DataFrame(data=self.GetModel().components_, index=self.__pca_feature_name, columns=self.__raw_feature_name)
        df.to_csv(pca_sort_path)

        pca_path = os.path.join(store_folder, 'pca.pickle')
        if pca_path[-7:] != '.pickle':
            print('Check the store path. ')
        else:
            with open(pca_path, 'wb') as f:
                pickle.dump(self.GetModel(), f)

    def LoadInfo(self, store_folder):
        if os.path.isdir(store_folder):
            pca_path = os.path.join(store_folder, 'pca.pickle')
        elif os.path.isfile(store_folder):
            pca_path = store_folder

        if not pca_path.endswith('.pickle'):
            print('Check the store path. ')
        else:
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
                self.SetModel(pca)
        super(DimensionReductionByPCA, self).SetRemainedNumber(self.GetModel().components_.shape[0])

    def SaveDataContainer(self, data_container, store_folder, is_test=False):
        if is_test:
            container_store_path = os.path.join(store_folder, 'pca_test_feature.csv')
        else:
            container_store_path = os.path.join(store_folder, 'pca_train_feature.csv')
        data_container.Save(container_store_path)

    def SetRemainedNumber(self, number):
        super(DimensionReductionByPCA, self).SetRemainedNumber(number)
        super(DimensionReductionByPCA, self).SetModel(PCA(n_components=super(DimensionReductionByPCA, self).GetRemainedNumber()))

    def Transform(self, data_container):
        data = data_container.GetArray()
        if data.shape[1] != self.GetModel().components_.shape[1]:
            print('Data can not be transformed by existed PCA')
        sub_data = self.GetModel().transform(data)

        sub_feature_name = ['PCA_feature_' + str(index) for index in
                            range(1, super(DimensionReductionByPCA, self).GetRemainedNumber() + 1)]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(sub_data)
        new_data_container.SetFeatureName(sub_feature_name)
        new_data_container.UpdateFrameByData()

        return new_data_container

    def GetDescription(self):
        text = "Since the dimension of feature space was high, we applied principle component analysis (PCA) on the feature matrix. " \
               "The feature vector of the transformed feature matrix was independent to each other. "
        return text


    def Run(self, data_container, store_folder=''):
        data = data_container.GetArray()
        self.SetRemainedNumber(np.min(data.shape))

        self.GetModel().fit(data)
        sub_data = self.GetModel().transform(data)

        sub_feature_name = ['PCA_feature_'+str(index) for index in range(1, super(DimensionReductionByPCA, self).GetRemainedNumber() + 1 )]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(sub_data)
        new_data_container.SetFeatureName(sub_feature_name)
        new_data_container.UpdateFrameByData()

        self.__raw_feature_name = data_container.GetFeatureName()
        self.__pca_feature_name = new_data_container.GetFeatureName()

        if store_folder and os.path.isdir(store_folder):
            self.SaveInfo(store_folder)
            self.SaveDataContainer(data_container, store_folder)

        return new_data_container

class DimensionReductionByPCC(DimensionReduction):
    def __init__(self, threshold=0.999):
        super(DimensionReductionByPCC, self).__init__()
        self.__threshold = threshold
        #TODO: Remove the __selected_index. This is not necessary.
        self.__selected_index = []

        self.__new_feature = []

    def GetName(self):
        return 'PCC'

    def __PCCSimilarity(self, data1, data2):
        return np.abs(pearsonr(data1, data2)[0])

    def SaveInfo(self, store_folder):
        pca_sort_path = os.path.join(store_folder, 'PCC_sort.csv')
        df = pd.DataFrame(data=self.__new_feature)
        df.to_csv(pca_sort_path)

    def LoadInfo(self, store_folder):
        if os.path.isdir(store_folder):
            pcc_path = os.path.join(store_folder, 'PCC_sort.csv')
        elif os.path.isfile(store_folder):
            pcc_path = store_folder

        selected_feature_df = pd.read_csv(pcc_path, index_col=0)
        self.__new_feature = selected_feature_df['0'].values.tolist()

    def SaveDataContainer(self, data_container, store_folder, is_test=False):
        if is_test:
            container_store_path = os.path.join(store_folder, 'PCC_test_feature.csv')
        else:
            container_store_path = os.path.join(store_folder, 'PCC_train_feature.csv')
        data_container.Save(container_store_path)

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        label = data_container.GetLabel()
        data /= np.linalg.norm(data, ord=2, axis=0)

        for feature_index in range(data.shape[1]):
            is_similar = False
            for save_index in self.__selected_index:
                if self.__PCCSimilarity(data[:, save_index], data[:, feature_index]) > self.__threshold:
                    if self.__PCCSimilarity(data[:, save_index], label) < self.__PCCSimilarity(data[:, feature_index],
                                                                                               label):
                        self.__selected_index[self.__selected_index == save_index] = feature_index
                    is_similar = True
                    break
            if not is_similar:
                self.__selected_index.append(feature_index)

    def Transform(self, data_container):
        if len(self.__selected_index) == 0 and len(self.__new_feature) > 0:
            all_featues = data_container.GetFeatureName()
            self.__selected_index = [all_featues.index(temp) for temp in self.__new_feature]
        assert(len(self.__selected_index) > 0)
        new_data = data_container.GetArray()[:, self.__selected_index]
        new_feature = [data_container.GetFeatureName()[t] for t in self.__selected_index]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(new_data)
        new_data_container.SetFeatureName(new_feature)
        new_data_container.UpdateFrameByData()

        return new_data_container

    def Run(self, data_container, store_folder=''):
        self.GetSelectedFeatureIndex(data_container)

        new_data = data_container.GetArray()[:, self.__selected_index]
        self.__new_feature = [data_container.GetFeatureName()[t] for t in self.__selected_index]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(new_data)
        new_data_container.SetFeatureName(self.__new_feature)
        new_data_container.UpdateFrameByData()

        if store_folder and os.path.isdir(store_folder):
            self.SaveInfo(store_folder)
            self.SaveDataContainer(new_data_container, store_folder, is_test=False)

        return new_data_container

    def GetDescription(self):
        text = "Since the dimension of feature space was high, we compared the similarity of each feature pair. " \
               "If the PCC value of the feature pair was larger than 0.86, we removed one of them. After this " \
               "process, the dimension of the feature space was reduced and each feature was independent to each other. "
        return text