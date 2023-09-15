'''.
Jun 17, 2018.
Yang SONG, songyangmri@gmail.com
'''

import os
import copy
import math
from copy import deepcopy

import numpy as np
import pandas as pd
from BC.Utility.Constants import REMOVE_CASE, REMOVE_FEATURE, REMOVE_NONE


def LoadCSVwithChineseInPandas(file_path, **kwargs):
    if 'encoding' not in kwargs.keys():
        return pd.read_csv(file_path, encoding="gbk", **kwargs).sort_index().sort_index(axis=1)
    else:
        return pd.read_csv(file_path, **kwargs).sort_index().sort_index(axis=1)

class DataContainer:
    '''
    DataContainer is the key class of the BC project. It is the node to connect different models. Almost all procesors
    accept DataContainer and return a new DataContainer.
    '''
    def __init__(self, array=np.array([]), label=np.array([]), feature_name=[], case_name=[]):
        self._feature_name = feature_name
        self._case_name = case_name
        self._label = label
        self._array = array
        self._df = pd.DataFrame()

        if array.size != 0 and label.size != 0:
            self.UpdateFrameByData()

    def __deepcopy__(self, memodict={}):
        copy_data_container = type(self)(deepcopy(self.GetArray()),
                                         deepcopy(self.GetLabel()),
                                         deepcopy(self.GetFeatureName()),
                                         deepcopy(self.GetCaseName()))
        return copy_data_container

    def __IsNumber(self, input_data):
        result = False
        try:
            float(input_data)
            result = True
        except ValueError:
            pass

        if result:
            temp = float(input_data)
            if np.isnan(temp):
                return False
            if np.isinf(temp):
                return False

        if not result:
            try:
                import unicodedata
                unicodedata.numeric(input_data)
                result = True
            except (TypeError, ValueError):
                pass

        return result

    def IsValidNumber(self, input_data):
        return self.__IsNumber(input_data) and not math.isnan(float(input_data))

    def IsEmpty(self):
        return self._df.size <= 0

    def IsBinaryLabel(self):
        return len(np.unique(self._label)) == 2

    def FindInvalidLabelIndex(self):
        for index in range(self._label.shape[0]):
            if self._label[index] != 0 and self._label[index] != 1:
                return index

    def FindInvalidNumber(self):
        for row_index in range(self._df.shape[0]):
            for col_index in range(self._df.shape[1]):
                if not self.IsValidNumber(self._df.iloc[row_index, col_index]):
                    return row_index, col_index
        return None

    def Save(self, store_path):
        self.UpdateFrameByData()
        self._df.to_csv(store_path, index='CaseID')

    def LoadWithoutCase(self, file_path):
        self.__init__()
        try:
            self._df = pd.read_csv(file_path, header=0)
            self.UpdateDataByFrame()
        except Exception as e:
            print('Check the CSV file path: LoadWithoutCase: \n{}'.format(e.__str__()))

    def LoadwithNonNumeric(self, file_path):
        self.__init__()
        try:
            self._df = pd.read_csv(file_path, header=0, index_col=0)
        except Exception as e:
            print('Check the CSV file path: LoadWithNonNumeirc: \n{}'.format(e.__str__()))

    def Load(self, file_path, is_update=True):
        assert(os.path.exists(file_path))
        self.__init__()
        try:
            self._df = pd.read_csv(file_path, header=0, index_col=0).sort_index().sort_index(axis=1)
            if is_update:
                return self.UpdateDataByFrame()
        except Exception as e:
            print('Check the CSV file path: {}: \n{}'.format(file_path, e.__str__()))

        try:
            self._df = LoadCSVwithChineseInPandas(file_path, header=0, index_col=0)
            if is_update:
                return self.UpdateDataByFrame()
        except Exception as e:
            print('Check the CSV file path: {}: \n{}'.format(file_path, e.__str__()))

        return False

    def Clear(self):
        self._feature_name = []
        self._case_name = []
        self._label = np.array([], dtype=int)
        self._array = np.array([])
        self._df = pd.DataFrame()

    def LoadWithoutLabel(self, file_path, is_update=True):
        assert (os.path.exists(file_path))
        self.__init__()
        try:
            self._df = pd.read_csv(file_path, header=0, index_col=0).sort_index().sort_index(axis=1)
            if is_update:
                self.UpdateDataByFrame(emu_label=True)
            return True
        except Exception as e:
            print('Check the CSV file path: {}: \n{}'.format(file_path, e.__str__()))

        try:
            self._df = LoadCSVwithChineseInPandas(file_path, header=0, index_col=0)
            self.UpdateDataByFrame()
            return True
        except Exception as e:
            print('Check the CSV file path: {}: \n{}'.format(file_path, e.__str__()))

        return False

    def ShowInformation(self):
        print('The number of cases is ', str(len(self._case_name)))
        print('The number of features is ', str(len(self._feature_name)))
        print('The cases are: ', self._case_name)
        print('The features are: ', self._feature_name)

        if len(np.unique(self._label)) == 2:
            positive_number = len(np.where(self._label == np.max(self._label))[0])
            negative_number = len(self._label) - positive_number
            assert(positive_number + negative_number == len(self._label))
            print('The number of positive samples is ', str(positive_number))
            print('The number of negative samples is ', str(negative_number))

    def UpdateDataByFrame(self, emu_label=False):
        self._case_name = [str(one) for one in self._df.index]
        self._feature_name = [str(one) for one in self._df.columns]
        label_name = ''
        if 'label' in self._feature_name:
            label_name = 'label'
        elif 'Label' in self._feature_name:
            label_name = 'Label'
        else:
            if emu_label:
                self._array = np.asarray(self._df[self._feature_name].values, dtype=np.float64)
                self._label = np.asarray(np.zeros(len(self._case_name), dtype=int))
                return True
            print('No "label" in the index')
            return False

        index = self._feature_name.index(label_name)
        self._feature_name.pop(index)
        self._label = np.asarray(self._df[label_name].values, dtype=np.int8)
        self._array = np.asarray(self._df[self._feature_name].values, dtype=np.float64)
        return True

    def UpdateFrameByData(self):
        data = np.concatenate((self._label[..., np.newaxis], self._array), axis=1)
        header = copy.deepcopy(self._feature_name)
        header.insert(0, 'label')
        index = self._case_name

        self._df = pd.DataFrame(data=data, index=index, columns=header).sort_index()

    def RemoveInvalid(self, store_path='', remove_index=REMOVE_NONE):
        array = []
        invalid_case, invalid_feature = [], []
        for case_index in range(self._df.shape[0]):
            sub_array = []
            for feature_index in range(self._df.shape[1]):
                if self.__IsNumber(self._df.iloc[case_index, feature_index]):
                    sub_array.append(True)
                else:
                    sub_array.append(False)
                    invalid_case.append(case_index)
                    invalid_feature.append(feature_index)
            array.append(sub_array)

        invalid_case = list(set(invalid_case))
        invalid_feature = list(set(invalid_feature))
        invalid_df = self._df.iloc[invalid_case, invalid_feature]

        if store_path and invalid_df.size > 0:
            invalid_df.to_csv(store_path)

        if remove_index == REMOVE_CASE:
            self._df.drop(index=invalid_df.index, inplace=True)
            if not self.UpdateDataByFrame():
                return False
        elif remove_index == REMOVE_FEATURE:
            self._df.drop(axis=1, columns=invalid_df.columns, inplace=True)
            if not self.UpdateDataByFrame():
                return False

        return True

    def LoadAndGetData(self, file_path):
        self.Load(file_path)
        return self.GetData()

    def GetData(self):
        return self._array, self._label, self._feature_name, self._case_name
    def GetFrame(self): return deepcopy(self._df)
    def GetArray(self): return deepcopy(self._array)
    def GetLabel(self): return deepcopy(self._label)
    def GetFeatureName(self): return deepcopy(self._feature_name)
    def GetCaseName(self): return deepcopy(self._case_name)

    def SetArray(self, array): self._array = array.astype(np.float64)
    def SetLabel(self, label):self._label = np.asarray(label, dtype=np.int)
    def SetFeatureName(self, feature_name): self._feature_name = feature_name
    def SetCaseName(self, case_name): self._case_name = case_name
    def SetFrame(self, frame):
        if 'label' in list(frame.columns) or 'Label' in list(frame.columns):
            self._df = frame
        else:
            if len(frame.index.tolist()) != self._label.size:
                print('Check the number of frame and the number of labels.')
                return None
            frame.insert(0, 'label', np.asarray(self._label, dtype=int))
            self._df = frame

        self.UpdateDataByFrame()


def main():
    pass


if __name__ == '__main__':
    main()
