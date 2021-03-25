"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/19
"""
import os
import csv
from abc import abstractmethod
from copy import deepcopy

import numpy as np
from scipy.stats import pearsonr

from SA.DataContainer import DataContainer
from SA.FeatureSelector import FeatureSelector
from SA.Utility import mylog

TRANSFORM_TYPE = 'TransformType'

TRANSFORM_TYPE_SUB = 'Sub'
TRANSFORM_TYPE_SUB_FEATURES = 'SubFeatures'

TRANSFORM_TYPE_MERGE = 'Merge'


class DimensionReducer(object):
    def __init__(self, name, transform_type,
                 model=None, sub_features=None):
        assert(transform_type in [TRANSFORM_TYPE_SUB, TRANSFORM_TYPE_MERGE])
        self._name = name
        self.transform_type = transform_type

        self.model = model
        self.sub_features = sub_features

    def GetName(self):
        return self._name
    def SetName(self, name):
        self._name = name
    name = property(GetName, SetName)

    @abstractmethod
    def SaveReducer(self, store_folder):
        pass

    @abstractmethod
    def LoadReducer(self, store_folder):
        pass

    @abstractmethod
    def Fit(self, dc: DataContainer):
        pass

    @abstractmethod
    def Transform(self, dc: DataContainer):
        pass


class DimensionReducerNone(DimensionReducer):
    def __init__(self):
        super(DimensionReducerNone, self).__init__('None', TRANSFORM_TYPE_SUB)
        self.info = []

    def Fit(self, dc):
        self.sub_features = dc.GetFeatureName()

    def Transform(self, dc: DataContainer, store_folder=None, store_key=None):
        assert(dc.GetFeatureName() == self.sub_features)
        if store_folder is not None and store_key is not None:
            dc.Save(os.path.join(store_folder, '{}_reduced_features.csv'.format(store_key)))
        return dc

    def SaveReducer(self, store_folder):
        self.info = []
        self.info.append([TRANSFORM_TYPE, self.transform_type])
        self.info.append([TRANSFORM_TYPE_SUB_FEATURES] + self.sub_features)

        with open(os.path.join(store_folder, '{}.csv'.format(TRANSFORM_TYPE_SUB_FEATURES)),
                  'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.info)

    def LoadReducer(self, store_folder):
        with open(os.path.join(store_folder, '{}.csv'.format(TRANSFORM_TYPE_SUB_FEATURES)),
                  'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == TRANSFORM_TYPE:
                    assert(row [1] == TRANSFORM_TYPE_SUB)
                elif row[0] == TRANSFORM_TYPE_SUB_FEATURES:
                    self.info = row[1:]
                else:
                    mylog.error('The first work of the dimension reducer file is not correct.')
                    raise ValueError


class DimensionReducerPcc(DimensionReducer):
    def __init__(self, threshold=0.99):
        super(DimensionReducerPcc, self).__init__('PCC', TRANSFORM_TYPE_SUB)
        self.info = []
        self.__threshold = threshold

    def __PCCSimilarity(self, data1, data2):
        return np.abs(pearsonr(data1, data2)[0])

    def Fit(self, dc):
        data = deepcopy(dc.array)
        data /= np.linalg.norm(data, ord=2, axis=0)
        event = dc.event.values

        selected_index = []
        for feature_index in range(data.shape[1]):
            is_similar = False
            assert (feature_index not in selected_index)
            for save_index in selected_index:
                if self.__PCCSimilarity(data[:, save_index], data[:, feature_index]) > self.__threshold:
                    if self.__PCCSimilarity(data[:, save_index], event) < \
                            self.__PCCSimilarity(data[:, feature_index], event):
                        selected_index[selected_index == save_index] = feature_index
                    is_similar = True
                    break
            if not is_similar:
                selected_index.append(feature_index)
        selected_index = sorted(selected_index)
        self.sub_features = [dc.feature_name[ind] for ind in selected_index]

    def Transform(self, dc: DataContainer, store_folder=None, store_key=None):
        for sub_feature in self.sub_features:
            if sub_feature not in dc.feature_name:
                mylog.error('DataContainer does not have the sub feature')
                raise KeyError

        new_dc = FeatureSelector().SelectByName(dc, self.sub_features)

        if store_folder is not None and store_key is not None:
            new_dc.Save(os.path.join(store_folder, '{}_reduced_features.csv'.format(store_key)))

        return new_dc

    def SaveReducer(self, store_folder):
        self.info = []
        self.info.append([TRANSFORM_TYPE, self.transform_type])
        self.info.append([TRANSFORM_TYPE_SUB_FEATURES] + self.sub_features)

        with open(os.path.join(store_folder, '{}.csv'.format(TRANSFORM_TYPE_SUB_FEATURES)),
                  'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.info)

    def LoadReducer(self, store_folder):
        with open(os.path.join(store_folder, '{}.csv'.format(TRANSFORM_TYPE_SUB_FEATURES)),
                  'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == TRANSFORM_TYPE:
                    assert(row [1] == TRANSFORM_TYPE_SUB)
                elif row[0] == TRANSFORM_TYPE_SUB_FEATURES:
                    self.info = row[1:]
                else:
                    mylog.error('The first work of the dimension reducer file is not correct.')
                    raise ValueError


if __name__ == '__main__':
    dc = DataContainer()
    dc.Load(r'..\..\Demo\train.csv', event_name='status', duration_name='time')

    reducer = DimensionReducerPcc(threshold=0.99)
    reducer.Fit(dc)
    result = reducer.Transform(dc)

    print(dc)
    print(result)
