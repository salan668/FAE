import numpy as np
import os
from abc import abstractmethod
import pandas as pd
from copy import deepcopy

class Normalizer:
    def __init__(self, key_name, name, description, method):
        self._slop = np.array([])
        self._interception = np.array([])

        self._key_name = key_name
        self._name = name
        self._Method = method
        self._description = description

    def Transform(self, data_container):
        array = data_container.GetArray()
        array -= self._interception
        array /= self._slop
        array = np.nan_to_num(array)

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(array)
        new_data_container.UpdateFrameByData()
        return new_data_container

    def SaveNormalDataContainer(self, data_container, store_folder, is_test=False):
        if is_test:
            dataset_type = 'testing'
        else:
            dataset_type = 'training'
        data_container.Save(os.path.join(store_folder, '{}_normalized_{}_feature.csv'.format(self._key_name, dataset_type)))

    def SaveInfo(self, store_folder, feature_name):
        df = pd.DataFrame({'feature_name': feature_name, 'slop': self._slop, 'interception': self._interception})
        df.to_csv(os.path.join(store_folder, '{}_normalization_training.csv'.format(self._key_name)), index=None)


    def LoadInfo(self, file_path):
        df = pd.read_csv(file_path)
        self._slop = np.array(df['slop'])
        self._interception = np.array(df['interception'])

    @abstractmethod
    def Run(self, raw_data_container, store_folder='', is_test=False):
        data_container = deepcopy(raw_data_container)
        array = data_container.GetArray()
        if not is_test:
            self._slop, self._interception = self._Method(array)
        data_container = self.Transform(data_container)

        if store_folder:
            self.SaveInfo(store_folder, data_container.GetFeatureName(), self._key_name)
            self.SaveNormalDataContainer(data_container, store_folder, self._key_name, is_test)

        return data_container

    def GetName(self):
        return self._name

    def GetDescription(self):
        return self._description


def NoneNormalizeFunc(array):
    return np.ones((array.shape[1], )), np.zeros((array.shape[1], ))
none_description = "We did not apply any normalization method on the feature matrix. "
NormalizerNone = Normalizer('non', 'NormNone', none_description, NoneNormalizeFunc)

def UnitNormalizeFunc(array):
    return np.linalg.norm(array, axis=0), np.zeros((array.shape[1], ))
unit_description = "We applied the normalization on the feature matrix. For each feature vector, we calculated the L2 norm " \
               "and divided by it. Then the feature vector was mapped to an unit vector. "
NormalizerUnit = Normalizer('unit', 'NormUnit', unit_description, UnitNormalizeFunc)

def ZNormalizeFunc(array):
    return np.std(array, axis=0), np.mean(array, axis=0)
z_description = "We applied the normalization on the feature matrix. For each feature vector, we calculated the mean " \
               "value and the standard deviation. Each feature vector was subtracted by the mean value and was divided " \
               "by the standard deviation. After normalization process, each vector has zero center and unit standard " \
               "deviation. "
NormalizerZeroCenter = Normalizer('zero_center', 'Norm0Center', z_description, ZNormalizeFunc)


def ZCenterNormalizeFunc(array):
    return np.linalg.norm(array, axis=0), np.mean(array, axis=0)
z_0_description = "We applied the normalization on the feature matrix.  Each feature vector was subtracted by the mean " \
               "value of the vector and was divided by the length of it. "
NormalizerZeroCenterAndUnit = Normalizer('zero_center_unit', 'Norm0CenterUnit', z_0_description, ZCenterNormalizeFunc)


if __name__ == '__main__':
    from FAE.DataContainer.DataContainer import DataContainer

    data_container = DataContainer()
    file_path = os.path.abspath(r'..\..\Example\numeric_feature.csv')
    print(file_path)
    data_container.Load(file_path)

    normalizer = NormalizerZeroCenterAndUnit()
    normalizer.Run(data_container, store_folder=r'..\..\Example\one_pipeline')