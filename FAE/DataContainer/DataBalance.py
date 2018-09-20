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
    def __init__(self):
        pass

    def Run(self, data_container, store_path=''):
        if store_path:
            if os.path.isdir(store_path):
                data_container.Save(os.path.join(store_path, 'non_balance_features.csv'))
            else:
                data_container.Save(store_path)

        return data_container


class DownSampling(DataBalance):
    def __init__(self):
        super(DownSampling, self).__init__()

    def GetCaseNameFromAllCaseNames(self, data_container, one_case_data):
        one_case_data = np.squeeze(one_case_data)
        all_case_data = data_container.GetArray()
        all_case_name = data_container.GetCaseName()

        if one_case_data.size != all_case_data.shape[1]:
            print('The number of features should be same in DataBalance!')

        for case_index in range(len(all_case_name)):
            if (one_case_data == all_case_data[case_index, :]).all():
                return all_case_name[case_index]
        print('Not Find Case Name')
        return 'Not Find Case Name'

    def Run(self, data_container, store_path=''):
        data, label, feature_name, label_name = data_container.GetData()
        rus = RandomUnderSampler(random_state=0)
        data_resampled, label_resampled = rus.fit_sample(data, label)

        new_case_name = []
        for index in range(data_resampled.shape[0]):
            new_case_name.append(self.GetCaseNameFromAllCaseNames(data_container, data_resampled[index, :]))

        new_data_container = DataContainer(data_resampled, label_resampled, data_container.GetFeatureName(), new_case_name)
        if store_path != '':
            if os.path.isdir(store_path):
                new_data_container.Save(os.path.join(store_path, 'downsampling_features.csv'))
            else:
                new_data_container.Save(store_path)
        return new_data_container

class UpSampling(DataBalance):
    def __init__(self):
        super(UpSampling, self).__init__()

    def GetCaseNameFromAllCaseNames(self, data_container, one_case_data):
        one_case_data = np.squeeze(one_case_data)
        all_case_data = data_container.GetArray()
        all_case_name = data_container.GetCaseName()

        if one_case_data.size != all_case_data.shape[1]:
            print('The number of features should be same in DataBalance!')

        for case_index in range(len(all_case_name)):
            if (one_case_data == all_case_data[case_index, :]).all():
                return all_case_name[case_index]
        print('Not Find Case Name')
        return 'Not Find Case Name'

    def Run(self, data_container, store_path=''):
        data, label, feature_name, label_name = data_container.GetData()
        rus = RandomOverSampler(random_state=0)
        data_resampled, label_resampled = rus.fit_sample(data, label)

        new_case_name = []
        for index in range(data_resampled.shape[0]):
            new_case_name.append(self.GetCaseNameFromAllCaseNames(data_container, data_resampled[index, :]))

        new_data_container = DataContainer(data_resampled, label_resampled, data_container.GetFeatureName(),
                                           new_case_name)
        if store_path != '':
            if os.path.isdir(store_path):
                new_data_container.Save(os.path.join(store_path, 'upsampling_features.csv'))
            else:
                new_data_container.Save(store_path)
        return new_data_container

class SmoteSampling(DataBalance):
    def __init__(self, **kwargs):
        super(SmoteSampling, self).__init__()
        self.__model = SMOTE(**kwargs, random_state=0)

    def Run(self, data_container, store_path=''):
        data, label, feature_name, label_name = data_container.GetData()
        data_resampled, label_resampled = self.__model.fit_sample(data, label)

        new_case_name = ['Generate' + str(index) for index in range(data_resampled.shape[0])]
        new_data_container = DataContainer(data_resampled, label_resampled, data_container.GetFeatureName(),
                                           new_case_name)
        if store_path != '':
            if os.path.isdir(store_path):
                new_data_container.Save(os.path.join(store_path, 'smote_features.csv'))
            else:
                new_data_container.Save(store_path)
        return new_data_container


