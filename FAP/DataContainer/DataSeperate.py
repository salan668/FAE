'''.
Jun 17, 2018.
Yang SONG, songyangmri@gmail.com
'''

import numpy as np
from random import shuffle
import os
import pandas as pd

from FAP.DataContainer.DataContainer import DataContainer

def SeperateDataToTrainingAndTesting(data, testing_percentage=0.2, label=np.array(()), training_index_list = [], store_folder=''):
    is_label = True
    if label.size == 0:
        label = np.zeros((data.shape[0]), )
        is_label = False
    label = np.asarray(label, dtype=np.uint8)

    if training_index_list == []:
        training_index_list, testing_index_list = [], []
        for group in range(np.max(label) + 1):
            index = np.where(label == group)[0]

            shuffle(index)
            testing_index = index[:round(len(index) * testing_percentage)]
            training_index = index[round(len(index) * testing_percentage):]

            training_index_list.extend(training_index)
            testing_index_list.extend(testing_index)

    else:
        testing_index_list = [temp for temp in list(range(data.shape[0])) if temp not in training_index_list]

    training_index_list.sort()
    testing_index_list.sort()

    training_data = data[training_index_list, ...]
    training_label = label[training_index_list]
    testing_data = data[testing_index_list, ...]
    testing_label = label[testing_index_list]

    if store_folder:
        df_training = pd.DataFrame(training_index_list)
        df_testing = pd.DataFrame(testing_index_list)
        df_training.to_csv(os.path.join(store_folder, 'training_index.csv'), sep=',', quotechar='"')
        df_testing.to_csv(os.path.join(store_folder, 'testing_index.csv'), sep=',', quotechar='"')

    if is_label:
        return {'training_data': training_data,
                'training_label': training_label,
                'testing_data': testing_data,
                'testing_label': testing_label,
                'training_index': training_index_list,
                'testing_index': testing_index_list}
    else:
        return {'training_data': training_data,
                'testing_data': testing_data,
                'training_index': training_index_list,
                'testing_index': testing_index_list}

def GenerateTrainingAndTestingData(csv_file_path, training_index=[], testing_percentage=0.3, is_store_index=False):
    '''
    Seperate the data container into training part and the testing part.
    :param csv_file_path: The file path of the data container
    :param training_index: The index of the training data set. This is usually to compare with different combination
    of the sequences. Default is []
    :param testing_percentage: The percentage of data set is used to separate for testing data set. Default is 30%
    :param is_store_index: To store or not. Default is False.
    :return:
    '''
    data_container = DataContainer()
    data, label, feature_name, case_name = data_container.LoadAndGetData(csv_file_path)
    folder_path = os.path.split(csv_file_path)[0]

    training_folder = os.path.join(folder_path, 'training')
    testing_folder = os.path.join(folder_path, 'testing')

    if not os.path.exists(training_folder):
        os.mkdir(training_folder)
    if not os.path.exists(testing_folder):
        os.mkdir(testing_folder)

    if is_store_index:
        store_folder = os.path.split(csv_file_path)[0]
    else:
        store_folder = ''

    output = SeperateDataToTrainingAndTesting(data, testing_percentage, label, training_index_list=training_index, store_folder=store_folder)

    training_data_contrainer = DataContainer(output['training_data'], output['training_label'], feature_name,
                                             [case_name[temp] for temp in output['training_index']])
    training_data_contrainer.Save(os.path.join(training_folder, 'numeric_feature.csv'))

    testing_data_contrainer = DataContainer(output['testing_data'], output['testing_label'], feature_name,
                                             [case_name[temp] for temp in output['testing_index']])
    testing_data_contrainer.Save(os.path.join(testing_folder, 'numeric_feature.csv'))


if __name__ == '__main__':
    GenerateTrainingAndTestingData(r'C:\MyCode\PythonScript\EyeEnt\lymphoma_MM\T1C_T2\T1_T2', testing_percentage=0.3)
    training_index = pd.read_csv(r'C:\MyCode\PythonScript\EyeEnt\lymphoma_MM\T1C_T2\T1_T2\training\training_index.csv')
    training_index = training_index.values[:, 1].tolist()
    GenerateTrainingAndTestingData(r'C:\MyCode\PythonScript\EyeEnt\lymphoma_MM\T1C_T2\T1', training_index=training_index)
    GenerateTrainingAndTestingData(r'C:\MyCode\PythonScript\EyeEnt\lymphoma_MM\T1C_T2\T2', training_index=training_index)

    training_index = pd.read_csv(r'C:\MyCode\PythonScript\EyeEnt\lymphoma_MM\T1C_T2\T1_T2\training\training_index.csv')
    print(training_index.values[:, 1].tolist())
    training_index = pd.read_csv(r'C:\MyCode\PythonScript\EyeEnt\lymphoma_MM\T1C_T2\T1\training\training_index.csv')
    print(training_index.values[:, 1].tolist())
    training_index = pd.read_csv(r'C:\MyCode\PythonScript\EyeEnt\lymphoma_MM\T1C_T2\T2\training\training_index.csv')
    print(training_index.values[:, 1].tolist())
