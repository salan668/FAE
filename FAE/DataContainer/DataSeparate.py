import numpy as np
from random import shuffle
import pandas as pd
import os

from FAE.DataContainer.DataContainer import DataContainer


class DataSeparate:
    def __init__(self, testing_percentage=0.3, training_index=[]):
        self._testing_percentage = testing_percentage
        self._training_index = training_index

    def __SetNewData(self, data_container, case_index):
        array, label, feature_name, case_name = data_container.GetData()

        new_array = array[case_index, :]
        new_label = label[case_index]
        new_case_name = [case_name[i] for i in case_index]

        new_data_container = DataContainer(array=new_array, label=new_label, case_name=new_case_name, feature_name=feature_name)
        new_data_container.UpdateFrameByData()
        return new_data_container

    def Run(self, data_container, store_folder=''):
        data = data_container.GetArray()
        label = data_container.GetLabel()

        if self._training_index == []:
            self._training_index, testing_index_list = [], []
            for group in range(int(np.max(label)) + 1):
                index = np.where(label == group)[0]

                shuffle(index)
                testing_index = index[:round(len(index) * self._testing_percentage)]
                training_index = index[round(len(index) * self._testing_percentage):]

                self._training_index.extend(training_index)
                testing_index_list.extend(testing_index)
        else:
            testing_index_list = [temp for temp in list(range(data.shape[0])) if temp not in self._training_index]

        self._training_index.sort()
        testing_index_list.sort()

        train_data_container = self.__SetNewData(data_container, self._training_index)
        test_data_container = self.__SetNewData(data_container, testing_index_list)

        if store_folder:
            train_data_container.Save(os.path.join(store_folder, 'train_numeric_feature.csv'))
            df_training = pd.DataFrame(self._training_index)
            df_training.to_csv(os.path.join(store_folder, 'training_index.csv'), sep=',', quotechar='"')

            test_data_container.Save(os.path.join(store_folder, 'test_numeric_feature.csv'))
            df_testing = pd.DataFrame(testing_index_list)
            df_testing.to_csv(os.path.join(store_folder, 'testing_index.csv'), sep=',', quotechar='"')

        return train_data_container, test_data_container


if __name__ == '__main__':
    data = DataContainer()
    data.Load(r'..\..\Example\numeric_feature.csv')

    data_separator = DataSeparate()
    data_separator.Run(data, store_folder=r'..\..\Example')

