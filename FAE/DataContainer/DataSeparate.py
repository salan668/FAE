import numpy as np
from random import shuffle
import pandas as pd
import os

from FAE.DataContainer.DataContainer import DataContainer


class DataSeparate:
    def __init__(self):
        pass

    def __SetNewData(self, data_container, case_index):
        array, label, feature_name, case_name = data_container.GetData()

        new_array = array[case_index, :]
        new_label = label[case_index]
        new_case_name = [case_name[i] for i in case_index]

        new_data_container = DataContainer(array=new_array, label=new_label, case_name=new_case_name, feature_name=feature_name)
        new_data_container.UpdateFrameByData()
        return new_data_container

    def RunByTestingPercentage(self, data_container, testing_data_percentage=0.3, store_folder=''):
        label = data_container.GetLabel()

        training_index_list, testing_index_list = [], []
        for group in range(int(np.max(label)) + 1):
            index = np.where(label == group)[0]

            shuffle(index)
            testing_index = index[:round(len(index) * testing_data_percentage)]
            training_index = index[round(len(index) * testing_data_percentage):]

            training_index_list.extend(training_index)
            testing_index_list.extend(testing_index)


        train_data_container = self.__SetNewData(data_container, training_index_list)
        test_data_container = self.__SetNewData(data_container, testing_index_list)

        if store_folder:
            train_data_container.Save(os.path.join(store_folder, 'train_numeric_feature.csv'))
            test_data_container.Save(os.path.join(store_folder, 'test_numeric_feature.csv'))

        return train_data_container, test_data_container

    def RunByTestingReference(self, data_container, testing_ref_data_container, store_folder=''):
        training_index_list, testing_index_list = [], []

        # TODO: assert data_container include all cases which is in the training_ref_data_container.
        all_name_list = data_container.GetCaseName()
        testing_name_list = testing_ref_data_container.GetCaseName()
        for training_name in testing_name_list:
            if training_name not in all_name_list:
                print('The data container and the training data container are not consistent.')
                return DataContainer(), DataContainer()

        for name, index in zip(all_name_list, range(len(all_name_list))):
            if name in testing_name_list:
                testing_index_list.append(index)
            else:
                training_index_list.append(index)

        train_data_container = self.__SetNewData(data_container, training_index_list)
        test_data_container = self.__SetNewData(data_container, testing_index_list)

        if store_folder:
            train_data_container.Save(os.path.join(store_folder, 'train_numeric_feature.csv'))
            test_data_container.Save(os.path.join(store_folder, 'test_numeric_feature.csv'))

        return train_data_container, test_data_container

if __name__ == '__main__':
    data = DataContainer()
    data.Load(r'..\..\Example\numeric_feature.csv')

    data_separator = DataSeparate()
    data_separator.Run(data, store_folder=r'..\..\Example\separate_test')

    ref_data_container = DataContainer()
    ref_data_container.Load(r'..\..\Example\separate_test\train_numeric_feature.csv')

    data_separator.training_ref_data_container = ref_data_container
    data_separator.Run(data, store_folder=r'..\..\Example\separate_test\reload')

