from abc import abstractmethod
import numpy as np
import os
from copy import deepcopy
import pandas as pd

from FAE.DataContainer.DataContainer import DataContainer
from sklearn.decomposition import PCA

class DimensionReduction:
    def __init__(self, model=None, number=0):
        self.__model = model
        self.__remained_number = number

    def SetModel(self, model):
        self.__model = model

    def GetModel(self):
        return self.__model

    def SetRemainedNumber(self, number):
        self.__remained_number = number

    def GetRemainedNumber(self):
        return self.__remained_number

    def Transform(self, data_container, store_path=''):
        data = data_container.GetArray()
        sub_feature_name = ['PCA_feature_' + str(index) for index in range(1, self.__remained_number + 1)]
        try:
            sub_data = self.__model.transform(data)
            new_data_container = deepcopy(data_container)
            new_data_container.SetArray(sub_data)
            new_data_container.SetFeatureName(sub_feature_name)
            new_data_container.UpdateFrameByData()

            if store_path:
                new_data_container.Save(store_path)

            return new_data_container
        except:
            print('Cannot transform the data.')

class DimensionReductionByPCA(DimensionReduction):
    def __init__(self, number=0):
        super(DimensionReductionByPCA, self).__init__(number=number)
        super(DimensionReductionByPCA, self).SetModel(PCA(n_components=super(DimensionReductionByPCA, self).GetRemainedNumber()))

    def SetRemainedNumber(self, number):
        super(DimensionReductionByPCA, self).SetRemainedNumber(number)
        super(DimensionReductionByPCA, self).SetModel(PCA(n_components=super(DimensionReductionByPCA, self).GetRemainedNumber()))


    def Run(self, data_container, store_folder=''):
        data = data_container.GetArray()
        data /= np.linalg.norm(data, ord=2, axis=0)

        if data.shape[1] < super(DimensionReductionByPCA, self).GetRemainedNumber():
            print('The number of features in data container is smaller than the required number')
            super(DimensionReductionByPCA, self).SetRemainedNumber(data.shape[1])

        self.GetModel().fit(data)
        sub_data = self.GetModel().transform(data)

        sub_feature_name = ['PCA_feature_'+str(index) for index in range(1, super(DimensionReductionByPCA, self).GetRemainedNumber() + 1 )]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(sub_data)
        new_data_container.SetFeatureName(sub_feature_name)
        new_data_container.UpdateFrameByData()
        if store_folder and os.path.isdir(store_folder):
            container_store_path = os.path.join(store_folder, 'pca_train_feature.csv')
            new_data_container.Save(container_store_path)

            pca_sort_path = os.path.join(store_folder, 'pca_sort.csv')
            df = pd.DataFrame(data=self.GetModel().components_, index=new_data_container.GetFeatureName(),
                              columns=data_container.GetFeatureName())
            df.to_csv(pca_sort_path)

        return new_data_container

