import numpy as np
import os
from abc import abstractmethod
import pandas as pd
from copy import deepcopy

class Normalizer:
    def __init__(self, name, description, method):
        self._slop = np.array([])
        self._interception = np.array([])

        self._name = name
        self._Method = method
        self._description = description

    def Transform(self, data_container, store_folder='', store_key=''):
        if data_container.IsEmpty():
            return data_container

        new_data_container = deepcopy(data_container)
        array = new_data_container.GetArray()
        array -= self._interception
        array /= self._slop
        array = np.nan_to_num(array)

        new_data_container.SetArray(array)
        new_data_container.UpdateFrameByData()

        if store_folder:
            assert(len(store_key) > 0)
            self.SaveNormalDataContainer(new_data_container, store_folder, store_key)

        return new_data_container

    def SaveNormalDataContainer(self, data_container, store_folder, store_key):
        data_container.Save(os.path.join(store_folder, '{}_normalized_{}_feature.csv'.format(self._name, store_key)))

    def SaveInfo(self, store_folder, feature_name):
        df = pd.DataFrame({'feature_name': feature_name, 'slop': self._slop, 'interception': self._interception})
        df.to_csv(os.path.join(store_folder, '{}_normalization_training.csv'.format(self._name)))

    def LoadInfo(self, file_path):
        df = pd.read_csv(file_path)
        self._slop = np.array(df['slop'])
        self._interception = np.array(df['interception'])

    @abstractmethod
    def Run(self, raw_data_container, store_folder='', store_key=''):
        if raw_data_container.IsEmpty():
            return raw_data_container

        data_container = deepcopy(raw_data_container)
        array = data_container.GetArray()
        self._slop, self._interception = self._Method(array)

        data_container = self.Transform(data_container, store_folder, store_key)

        if store_folder:
            self.SaveInfo(store_folder, data_container.GetFeatureName())
        return data_container

    def GetName(self):
        return self._name

    def GetDescription(self):
        return self._description


def NoneNormalizeFunc(array):
    return np.ones((array.shape[1], )), np.zeros((array.shape[1], ))
none_description = "We did not apply any normalization method on the feature matrix. "
NormalizerNone = Normalizer('None', none_description, NoneNormalizeFunc)

def MinMaxNormFunc(array):
    return np.max(array, axis=0) - np.min(array, axis=0), np.min(array, axis=0)
unit_description = "We applied the normalization on the feature matrix. For each feature vector, we calculated the L2 norm " \
               "and divided by it. Then the feature vector was mapped to an unit vector. "
NormalizerMinMax = Normalizer('MinMax', unit_description, MinMaxNormFunc)

def ZNormalizeFunc(array):
    return np.std(array, axis=0), np.mean(array, axis=0)
z_description = "We applied the normalization on the feature matrix. For each feature vector, we calculated the mean " \
               "value and the standard deviation. Each feature vector was subtracted by the mean value and was divided " \
               "by the standard deviation. After normalization process, each vector has zero center and unit standard " \
               "deviation. "
NormalizerZscore = Normalizer('Zscore', z_description, ZNormalizeFunc)


def MeanNormFunc(array):
    return np.max(array, axis=0) - np.min(array, axis=0), np.mean(array, axis=0)
z_0_description = "We applied the normalization on the feature matrix.  Each feature vector was subtracted by the mean " \
               "value of the vector and was divided by the length of it. "
NormalizerMean = Normalizer('Mean', z_0_description, MeanNormFunc)


if __name__ == '__main__':
    # from BC.DataContainer.DataContainer import DataContainer
    #
    # data_container = DataContainer()
    # file_path = os.path.abspath(r'..\..\Example\numeric_feature.csv')
    # print(file_path)
    # data_container.Load(file_path)
    #
    # normalizer = NormalizerZeroCenterAndUnit
    # normalizer.Run(data_container, store_folder=r'..\..\Example\one_pipeline')
    pass