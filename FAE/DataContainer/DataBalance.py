'''.
Jul 03, 2018.
Yang SONG, songyangmri@gmail.com
'''

import numpy as np
from random import shuffle
import os
import pandas as pd
from abc import abstractmethod

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from FAE.DataContainer.DataContainer import DataContainer


class DataBalance:
    '''
    To deal with the data imbalance.
    '''
    def __init__(self, data_container = DataContainer()):
        self.__data_container = data_container

    def SetDataContainer(self, data_container):
        self.__data_container = data_container

    def GetDataContainer(self):
        return self.__data_container

    @abstractmethod
    def Process(self):
        pass


class UnderSampling(DataBalance):
    def __init__(self, data_container = DataContainer()):
        super(UnderSampling, self).__init__(data_container)

    def GetCaseNameFromAllCaseNames(self, one_case_data):
        one_case_data = np.squeeze(one_case_data)
        all_case_data = self.GetDataContainer().GetArray()
        all_case_name = self.GetDataContainer().GetCaseName()

        if one_case_data.size != all_case_data.shape[1]:
            print('The number of features should be same in DataBalance!')

        for case_index in range(len(all_case_name)):
            if (one_case_data == all_case_data[case_index, :]).all():
                return all_case_name[case_index]
        print('Not Find Case Name')
        return 'Not Find Case Name'

    def Run(self, store_path=''):
        data, label, feature_name, label_name = self.GetDataContainer().GetData()
        rus = RandomUnderSampler(random_state=0)
        data_resampled, label_resampled = rus.fit_sample(data, label)

        new_case_name = []
        for index in range(data_resampled.shape[0]):
            new_case_name.append(self.GetCaseNameFromAllCaseNames(data_resampled[index, :]))

        new_data_container = DataContainer(data_resampled, label_resampled, self.GetDataContainer().GetFeatureName(), new_case_name)
        if store_path != '':
            new_data_container.Save(store_path)
        return new_data_container

class UpSampling(DataBalance):
    def __init__(self, data_container = DataContainer()):
        super(UpSampling, self).__init__(data_container)

    def GetCaseNameFromAllCaseNames(self, one_case_data):
        one_case_data = np.squeeze(one_case_data)
        all_case_data = self.GetDataContainer().GetArray()
        all_case_name = self.GetDataContainer().GetCaseName()

        if one_case_data.size != all_case_data.shape[1]:
            print('The number of features should be same in DataBalance!')

        for case_index in range(len(all_case_name)):
            if (one_case_data == all_case_data[case_index, :]).all():
                return all_case_name[case_index]
        print('Not Find Case Name')
        return 'Not Find Case Name'

    def Run(self, store_path=''):
        data, label, feature_name, label_name = self.GetDataContainer().GetData()
        rus = RandomOverSampler(random_state=0)
        data_resampled, label_resampled = rus.fit_sample(data, label)

        new_case_name = []
        for index in range(data_resampled.shape[0]):
            new_case_name.append(self.GetCaseNameFromAllCaseNames(data_resampled[index, :]))

        new_data_container = DataContainer(data_resampled, label_resampled, self.GetDataContainer().GetFeatureName(),
                                           new_case_name)
        if store_path != '':
            new_data_container.Save(store_path)
        return new_data_container

class SmoteSampling(DataBalance):
    def __init__(self, data_container = DataContainer(), **kwargs):
        super(SmoteSampling, self).__init__(data_container)
        self.__model = SMOTE(**kwargs, random_state=0)

    def Run(self, store_path):
        data, label, feature_name, label_name = self.GetDataContainer().GetData()
        data_resampled, label_resampled = self.__model.fit_sample(data, label)

        new_case_name = ['Generate' + str(index) for index in range(data_resampled.shape[0])]
        new_data_container = DataContainer(data_resampled, label_resampled, self.GetDataContainer().GetFeatureName(),
                                           new_case_name)
        if store_path != '':
            new_data_container.Save(store_path)
        return new_data_container


