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

from Utility.EcLog import eclog


def LoadCSVwithChineseInPandas(file_path, **kwargs):
    if 'encoding' not in kwargs.keys():
        return pd.read_csv(file_path, encoding="gbk", **kwargs)
    else:
        return pd.read_csv(file_path, **kwargs)

class DataContainer:
    '''
    DataContainer is the key class of the FAE project. It is the node to connect different models. Almost all procesors
    accept DataContainer and return a new DataContainer.
    '''
    def __init__(self, array=np.array([]), label=np.array([]), feature_name=[], case_name=[]):
        self.__feature_name = feature_name
        self.__case_name = case_name
        self.__label = label
        self._array = array
        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()

        if array.size != 0 and label.size != 0:
            self.UpdateFrameByData()
        else:
            self.__df = None

    def __deepcopy__(self, memodict={}):
        copy_data_container = type(self)(deepcopy(self.GetArray()),
                                         deepcopy(self.GetLabel()),
                                         deepcopy(self.GetFeatureName()),
                                         deepcopy(self.GetCaseName()))
        return copy_data_container


    def __IsNumber(self, input_data):
        try:
            float(input_data)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(input_data)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def IsValidNumber(self, input_data):
        if not self.__IsNumber(input_data):
            return False

        if math.isnan(float(input_data)):
            return False

        return True

    def IsEmpty(self):
        if self._array.size > 0:
            return False
        else:
            return True

    def IsBinaryLabel(self):
        return len(np.unique(self.__label)) == 2

    def FindNonValidLabelIndex(self):
        for index in range(self.__label.shape[0]):
            if self.__label[index] != 0 and self.__label[index] != 1:
                return index

    def HasNonValidNumber(self):
        array_flat = self._array.flatten()
        for index in range(self._array.size):
            if not self.IsValidNumber(array_flat[index]):
                return True
        return False

    def FindNonValidNumberIndex(self):
        for index0 in range(self._array.shape[0]):
            for index1 in range(self._array.shape[1]):
                if not self.IsValidNumber(self._array[index0,index1]):
                    return index0, index1
        return None, None

    def Save(self, store_path):
        self.UpdateFrameByData()
        self.__df.to_csv(store_path, index='CaseName')

    def LoadWithoutCase(self, file_path):
        self.__init__()
        try:
            self.__df = pd.read_csv(file_path, header=0)
            self.UpdateDataByFrame()
        except Exception as e:
            # self.logger.error('LoadWitoutCase:  ' + str(e))
            print('Check the CSV file path: LoadWithoutCase: \n{}'.format(e.__str__()))


    def LoadwithNonNumeric(self, file_path):
        self.__init__()
        try:
            self.__df = pd.read_csv(file_path, header=0, index_col=0)
        except Exception as e:
            # self.logger.error('LoadWithNonNumeirc:  ' + str(e))
            print('Check the CSV file path: LoadWithNonNumeirc: \n{}'.format(e.__str__()))


    def Load(self, file_path):
        self.__init__()
        try:
            self.__df = pd.read_csv(file_path, header=0, index_col=0)
            self.UpdateDataByFrame()
            return
        except Exception as e:
            # self.logger.error('Load:  ' + str(e))
            print('Check the CSV file path: {}: \n{}'.format(file_path, e.__str__()))

        try:
            self.__df = LoadCSVwithChineseInPandas(file_path, header=0, index_col=0)
            self.UpdateDataByFrame()
        except Exception as e:
            # self.logger.error('Load Chinese CSV:  ' + str(e))
            print('Check the CSV file path: {}: \n{}'.format(file_path, e.__str__()))

    def ShowInformation(self):
        print('The number of cases is ', str(len(self.__case_name)))
        print('The number of features is ', str(len(self.__feature_name)))
        print('The cases are: ', self.__case_name)
        print('The features are: ', self.__feature_name)

        if len(np.unique(self.__label)) == 2:
            positive_number = len(np.where(self.__label == np.max(self.__label))[0])
            negative_number = len(self.__label) - positive_number
            assert(positive_number + negative_number == len(self.__label))
            print('The number of positive samples is ', str(positive_number))
            print('The number of negative samples is ', str(negative_number))

    def UpdateDataByFrame(self):
        self.__case_name = list(self.__df.index)
        self.__feature_name = list(self.__df.columns)
        if 'label' in self.__feature_name:
            label_name = 'label'
            index = self.__feature_name.index('label')
        elif 'Label' in self.__feature_name:
            label_name = 'Label'
            index = self.__feature_name.index('Label')
        else:
            print('No "label" in the index')
            index = np.nan
        self.__feature_name.pop(index)
        self.__label = np.asarray(self.__df[label_name].values, dtype=np.int)
        self._array = np.asarray(self.__df[self.__feature_name].values, dtype=np.float32)

    def UpdateFrameByData(self):
        data = np.concatenate((self.__label[..., np.newaxis], self._array), axis=1)
        header = copy.deepcopy(self.__feature_name)
        header.insert(0, 'label')
        index = self.__case_name

        self.__df = pd.DataFrame(data=data, index=index, columns=header)

    def RemoveUneffectiveFeatures(self):
        removed_index = []
        for index in range(len(self.__feature_name)):
            vector = self._array[:, index]
            if np.where(np.isnan(vector))[0].size > 0:
                removed_index.append(index)

        # Remove the feature name
        removed_feature_name = [self.__feature_name[index] for index in removed_index]
        for feature_name in removed_feature_name:
            self.__feature_name.remove(feature_name)

        new_array = np.delete(self._array, removed_index, axis=1)
        self._array = new_array

        self.UpdateFrameByData()

    def RemoveUneffectiveCases(self):
        removed_index = []
        for index in range(len(self.__case_name)):
            vector = self._array[index, :]
            if np.where(np.isnan(vector))[0].size > 0:
                removed_index.append(index)
                continue
            if self.__label[index] != 0 and self.__label[index] != 1:
                removed_index.append(index)

        # Remove the case name
        removed_case_name = [self.__case_name[index] for index in removed_index]
        for case_name in removed_case_name:
            self.__case_name.remove(case_name)

        new_array = np.delete(self._array, removed_index, axis=0)
        self._array = new_array
        new_label = np.delete(self.__label, removed_index, axis=0)
        self.__label = new_label

        self.UpdateFrameByData()

    def LoadAndGetData(self, file_path):
        self.Load(file_path)
        return self.GetData()
    def GetData(self):
        return self._array, self.__label, self.__feature_name, self.__case_name

    def GetFrame(self): return self.__df
    def GetArray(self): return self._array.astype(np.float64)
    def GetLabel(self): return self.__label
    def GetFeatureName(self): return self.__feature_name
    def GetCaseName(self): return self.__case_name

    def SetArray(self, array): self._array = array
    def SetLabel(self, label):self.__label = np.asarray(label, dtype=np.int)
    def SetFeatureName(self, feature_name): self.__feature_name = feature_name
    def SetCaseName(self, case_name): self.__case_name = case_name
    def SetFrame(self, frame):
        if 'label' in list(frame.columns) or 'Label' in list(frame.columns):
            self.__df = frame
        else:
            if len(frame.index.tolist()) != self.__label.size:
                print('Check the number of fram and the number of labels.')
                return None
            frame.insert(0, 'label', np.asarray(self.__label, dtype=int))
            self.__df = frame

        self.UpdateDataByFrame()


def main():
    import copy
    data = DataContainer()
    data.Load(r'C:\Users\yangs\Desktop\fae_test\data_noncontrast_original.csv')
    new_data = copy.deepcopy(data)
    new_data.ShowInformation()

if __name__ == '__main__':
    main()
