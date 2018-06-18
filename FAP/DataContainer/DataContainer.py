'''.
Jun 17, 2018.
Yang SONG, songyangmri@gmail.com
'''

import numpy as np
import os
import pandas as pd

import copy


class DataContainer:
    '''
    DataContainer is the key class of the FAP project. It is the node to connect different models. Almost all procesors
    accept DataContainer and return a new DataContainer.
    '''
    def __init__(self, array=np.array([]), label=np.array([]), feature_name=[], case_name=[]):
        self.__feature_name = feature_name
        self.__case_name = case_name
        self.__label = label
        self._array = array

        if array.size != 0 and label.size != 0:
            self.UpdateFrameByData()
        else:
            self.__df = None

    def Save(self, store_path):
        self.UpdateFrameByData()
        self.__df.to_csv(store_path)

    def Load(self, file_path):
        self.__init__()
        try:
            self.__df = pd.read_csv(file_path, header=0, index_col=0)
            self.UpdateDataByFrame()
        except:
            print('Check the CSV file path. ')

    def UpdateDataByFrame(self):
        self.__case_name = list(self.__df.index)
        self.__feature_name = list(self.__df.columns)
        index = self.__feature_name.index('label')
        self.__feature_name.pop(index)
        self.__label = self.__df['label'].as_matrix()
        self._array = self.__df[self.__feature_name].as_matrix()

    def UpdateFrameByData(self):
        data = np.concatenate((self.__label[..., np.newaxis], self._array), axis=1)
        header = copy.deepcopy(self.__feature_name)
        header.insert(0, 'label')
        index = self.__case_name

        self.__df = pd.DataFrame(data=data, index=index, columns=header)

    def LoadAndGetData(self, file_path):
        self.Load(file_path)
        return self.GetData()
    def GetData(self):
        return self._array, self.__label, self.__feature_name, self.__case_name

    def GetFrame(self): return self.__df
    def GetArray(self): return self._array
    def GetLabel(self): return self.__label
    def GetFeatureName(self): return self.__feature_name
    def GetCaseName(self): return self.__case_name

    def SetArray(self, array): self._array = array
    def SetLabel(self, label):self.__label = label
    def SetFeatureName(self, feature_name): self.__feature_name = feature_name
    def SetCaseName(self, case_name): self.__case_name = case_name
    def SetFrame(self, frame):
        if 'label' in list(frame.columns):
            self.__df = frame
        else:
            if len(frame.index.tolist()) != self.__label.size:
                print('Check the number of fram and the number of labels.')
                return None
            frame.insert(0, 'label', self.__label)
            self.__df = frame

        self.UpdateDataByFrame()

    def UsualNormalize(self, store_path='', axis=0):
        mean_value = np.average(self._array, axis=axis)
        std_value = np.std(self._array, axis=axis)
        self._array -= mean_value
        self._array /= std_value
        self._array = np.nan_to_num(self._array)

        self.UpdateFrameByData()

        if store_path != '':
            data = np.stack((mean_value, std_value), axis=0)

            header = []
            if axis == 0:
                header = copy.deepcopy(self.__feature_name)
            elif axis == 1:
                header = copy.deepcopy(self.__case_name)
            index = ['mean', 'std']

            assert (len(header) == data.shape[1])
            df = pd.DataFrame(data, columns=header, index=index)
            df.to_csv(store_path)

    def UsualAndL2Normalize(self, store_path='', axis=0):
        mean_value = np.average(self._array, axis=axis)
        std_value = np.std(self._array, axis=axis)
        self._array -= mean_value
        self._array /= std_value
        self._array = np.nan_to_num(self._array)
        self._array /= np.linalg.norm(self._array, ord=2, axis=0)

        self.UpdateFrameByData()

        if store_path != '':
            data = np.stack((mean_value, std_value), axis=0)

            header = []
            if axis == 0:
                header = copy.deepcopy(self.__feature_name)
            elif axis == 1:
                header = copy.deepcopy(self.__case_name)
            index = ['mean', 'std']

            assert (len(header) == data.shape[1])
            df = pd.DataFrame(data, columns=header, index=index)
            df.to_csv(store_path)

    def ArtefactNormalize(self, normalization_file):
        '''
        This function can use the existing file with the infoamtion of the normalization. It is usually used on the fact
        that a learnt model is used to process the testing data set.
        :param normalization_file: the stored file with the information of the normalization.
        :return:
        '''
        df = pd.read_csv(normalization_file, header=0, index_col=0)
        mean_value = df.loc['mean'].values
        std_value = df.loc['std'].values

        if mean_value.size != self._array.shape[1] or mean_value.size != self._array.shape[1]:
            print('Check the data shape and the normalization file')
            return None
        self._array -= mean_value
        self._array /= std_value
        self._array = np.nan_to_num(self._array)

        self.UpdateFrameByData()

def main():
    # test FeatureReader
    # feature_reader = DataContainer()
    # array, label, feature_name, case_name = feature_reader.LoadAndGetData(r'..\Result\numeric_feature.csv')
    # print(array.shape)
    # print(label.shape)
    # print(len(feature_name))
    # print(len(case_name))
    # feature_reader.SaveData(r'..\Result\NewNumericFeature.csv')

    # Test Normalization
    data = DataContainer()
    data.Load(r'..\Result\NumericFeature.csv')
    data.UsualNormalize(r'..\Result\normalization.csv')
    data.ArtefactNormalize(r'..\Result\normalization.csv')



if __name__ == '__main__':
    main()