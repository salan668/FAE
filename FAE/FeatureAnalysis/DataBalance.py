'''.
Jul 03, 2018.
Yang SONG, songyangmri@gmail.com
'''

import numpy as np
import os
from abc import abstractmethod

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

from FAE.DataContainer.DataContainer import DataContainer
from Utility.Constants import BALANCE_UP_SAMPLING, BALANCE_DOWN_SAMPLING, BALANCE_SMOTE, BALANCE_SMOTE_TOMEK
from FAE.HyperParameterConfig.HyperParamManager import RANDOM_SEED


class DataBalance:
    '''
    To deal with the data imbalance.
    '''
    def __init__(self, model, name):
        self._model = model
        self._name = name
        pass

    def GetName(self):
        return self._name

    def GetModel(self):
        return self._model

    @abstractmethod
    def Run(self, data_container, store_path=''):
        pass

    @abstractmethod
    def GetDescription(self):
        pass


class NoneBalance(DataBalance):
    def __init__(self):
        super(NoneBalance, self).__init__(None, 'NoneBalance')

    def Run(self, container, store_path=''):
        if store_path != '':
            if os.path.isdir(store_path):
                container.Save(os.path.join(store_path, '{}_features.csv'.format(self._name)))
            else:
                container.Save(store_path)
        return container

    def GetDescription(self):
        return ''


class DownSampling(DataBalance):
    def __init__(self):
        super(DownSampling, self).__init__(RandomUnderSampler(random_state=RANDOM_SEED[BALANCE_DOWN_SAMPLING]),
                                           BALANCE_DOWN_SAMPLING)

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

    def GetDescription(self):
        return "To Remove the unbalance of the training data set, we sampled the cases to make positive/negative " \
               "samples balance. "

    def Run(self, data_container, store_path=''):
        data, label, feature_name, label_name = data_container.GetData()
        data_resampled, label_resampled = self._model.fit_sample(data, label)

        new_case_name = []
        for index in range(data_resampled.shape[0]):
            new_case_name.append(self.GetCaseNameFromAllCaseNames(data_container, data_resampled[index, :]))

        new_data_container = DataContainer(data_resampled, label_resampled, data_container.GetFeatureName(), new_case_name)
        if store_path != '':
            if os.path.isdir(store_path):
                new_data_container.Save(os.path.join(store_path, '{}_features.csv'.format(self._name)))
            else:
                new_data_container.Save(store_path)
        return new_data_container


class UpSampling(DataBalance):
    def __init__(self):
        super(UpSampling, self).__init__(RandomOverSampler(random_state=RANDOM_SEED[BALANCE_UP_SAMPLING]),
                                         BALANCE_UP_SAMPLING)

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

    def GetDescription(self):
        return "To Remove the unbalance of the training data set, we up-samples by repeating random cases to " \
               "to make positive/negative samples balance. "

    def Run(self, data_container, store_path=''):
        data, label, feature_name, label_name = data_container.GetData()
        data_resampled, label_resampled = self._model.fit_sample(data, label)

        new_case_name = []
        for index in range(data_resampled.shape[0]):
            new_case_name.append(self.GetCaseNameFromAllCaseNames(data_container, data_resampled[index, :]))

        new_data_container = DataContainer(data_resampled, label_resampled, data_container.GetFeatureName(),
                                           new_case_name)
        if store_path != '':
            if os.path.isdir(store_path):
                new_data_container.Save(os.path.join(store_path, '{}_features.csv'.format(self._name)))
            else:
                new_data_container.Save(store_path)
        return new_data_container


class SmoteSampling(DataBalance):
    def __init__(self, **kwargs):
        super(SmoteSampling, self).__init__(SMOTE(**kwargs, random_state=RANDOM_SEED[BALANCE_SMOTE]), BALANCE_SMOTE)

    def GetDescription(self):
        return "To Remove the unbalance of the training data set, we used the Synthetic Minority Oversampling " \
               "TEchnique (SMOTE) to make positive/negative samples balance. "

    def Run(self, data_container, store_path=''):
        data, label, feature_name, label_name = data_container.GetData()
        data_resampled, label_resampled = self._model.fit_sample(data, label)

        new_case_name = ['Generate' + str(index) for index in range(data_resampled.shape[0])]
        new_data_container = DataContainer(data_resampled, label_resampled, data_container.GetFeatureName(),
                                           new_case_name)
        if store_path != '':
            if os.path.isdir(store_path):
                new_data_container.Save(os.path.join(store_path, '{}_features.csv'.format(self._name)))
            else:
                new_data_container.Save(store_path)
        return new_data_container


class SmoteTomekSampling(DataBalance):
    def __init__(self, **kwargs):
        super(SmoteTomekSampling, self).__init__(SMOTETomek(**kwargs, random_state=RANDOM_SEED[BALANCE_SMOTE_TOMEK]),
                                                 BALANCE_SMOTE_TOMEK)

    def GetDescription(self):
        return "To Remove the unbalance of the training data set, we applied an Tomek link after the " \
               "Synthetic Minority Oversampling TEchnique (SMOTE) to make positive/negative samples balance. "

    def Run(self, data_container, store_path=''):
        data, label, feature_name, label_name = data_container.GetData()
        data_resampled, label_resampled = self._model.fit_sample(data, label)

        new_case_name = ['Generate' + str(index) for index in range(data_resampled.shape[0])]
        new_data_container = DataContainer(data_resampled, label_resampled, data_container.GetFeatureName(),
                                           new_case_name)
        if store_path != '':
            if os.path.isdir(store_path):
                new_data_container.Save(os.path.join(store_path, '{}_features.csv'.format(self._name)))
            else:
                new_data_container.Save(store_path)
        return new_data_container


if __name__ == '__main__':
    dc = DataContainer()
    dc.Load(r'..\..\Example\numeric_feature.csv')
    print(dc.GetArray().shape, np.sum(dc.GetLabel()))
    b = SmoteTomekSampling()
    new = b.Run(dc)
    print(new.GetArray().shape, np.sum(new.GetLabel()))
