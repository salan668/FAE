"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/15
"""

import numpy as np
import os
import pandas as pd
from copy import deepcopy

from SA.DataContainer import DataContainer


class Normalizer:
    def __init__(self, name, description, method):
        self._slop = np.array([])
        self._interception = np.array([])

        self._name = name
        self._Method = method
        self._description = description

    def GetName(self):
        return self._name
    def SetName(self, name):
        self._name = name
    name = property(GetName, SetName)

    def Transform(self, dc: DataContainer, store_folder=None, store_key=None):
        if dc.IsEmpty():
            return dc

        new_dc = deepcopy(dc)
        array = new_dc.array
        array -= self._interception
        array /= self._slop
        array = np.nan_to_num(array)

        new_dc.array = array
        new_dc.UpdateFrame()

        if store_folder is not None and store_key is not None:
            assert(len(store_key) > 0)
            self.SaveNormalDataContainer(new_dc, store_folder, store_key)

        return new_dc

    def SaveNormalDataContainer(self, dc: DataContainer, store_folder, store_key):
        dc.Save(os.path.join(store_folder, '{}_normalized_{}_feature.csv'.format(self._name, store_key)))

    def SaveInfo(self, store_folder, feature_name):
        df = pd.DataFrame({'feature_name': feature_name, 'slop': self._slop, 'interception': self._interception})
        df.to_csv(os.path.join(store_folder, '{}_normalization_training.csv'.format(self._name)))

    def LoadInfo(self, file_path):
        df = pd.read_csv(file_path)
        self._slop = np.array(df['slop'])
        self._interception = np.array(df['interception'])

    def Fit(self, raw_data_container, store_folder=None, store_key=None):
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
    file_path = r'C:\Users\yangs\Desktop\Radiomics_pvp_hcc_os_top20_train.csv'
    dc = DataContainer()
    dc.Load(file_path, event_name='status', duration_name='time')

    normal = NormalizerNone
    new_dc = normal.Fit(dc)
    print(new_dc)
    new_dc = normal.Transform(dc)
    print(new_dc)
