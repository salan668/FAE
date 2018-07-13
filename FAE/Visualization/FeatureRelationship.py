import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from FAE.DataContainer.DataContainer import DataContainer
import os
color_list = sns.color_palette('deep') + sns.color_palette('bright')


def DrawValueRelationship(vector_list, vector_name_list, label, label_name_list, store_path=''):
    '''
    Draw value relationship
    :param vector_list: the list store different values. The length must be shorter than 4 (<=3)
    :param vector_name_list: the name of axis. The length must be equal to the length of the value_list
    :param label: The corresponding label.
    :param label_name_list: The name of the legend.
    :return: Appear a figure.

    Apr-27-18, Yang SONG [yang.song.91@foxmail.com]
    '''

    dimension = len(vector_list)
    label = np.asarray(label, dtype=np.int8)
    sub_sets = np.max(label) + 1

    if dimension > 3:
        print('Only can show 3 features once.')
        return None

    if label_name_list == []:
        for index in range(sub_sets):
            label_name_list.append('label='+str(index))

    # Normalize
    normal_value_list = []
    for value in vector_list:
        if len(value) != label.size:
            print('Plase check the value list and label. The length should be same.')
            return None
        value -= np.mean(value)
        value /= np.std(value)
        normal_value_list.append(value)

    # Seperate the Vectors.
    show_info = {}
    for sub_set_index in range(sub_sets):
        show_info[sub_set_index] = {}
        for dimension_index in range(dimension):
            show_info[sub_set_index][dimension_index] = [normal_value_list[dimension_index][index] for index in range(label.size) if label[index] == sub_set_index]
    if dimension == 1:
        label_value_0  =show_info[0][0]
        label_value_1 = show_info[1][0]
        plt.style.use('ggplot')
        plt.hist(label_value_0, label=label_name_list[0], color=color_list[0], alpha=0.7, rwidth=0.5)
        plt.hist(label_value_1, label=label_name_list[1], color=color_list[1], alpha=0.7, rwidth=0.5)
        plt.tick_params(top='off', right='off')
        plt.title(vector_name_list[0])
        plt.xlabel('feature value')
        plt.ylabel('case number')
        plt.legend()
        if store_path:
            if store_path[-3:] == 'jpg':
                plt.savefig(store_path, dpi=300, format='jpeg')
            elif store_path[-3:] == 'eps':
                plt.savefig(store_path, dpi=1200, format='eps')
        plt.show()
    else:
        fig = plt.figure()
        if dimension == 2:
            ax = plt.subplot()
            for sub_set_index in range(sub_sets):
                ax.scatter(show_info[sub_set_index][0], show_info[sub_set_index][1], s=100, c=color_list[sub_set_index])
        if dimension == 3:
            ax = plt.subplot(projection='3d')
            for sub_set_index in range(sub_sets):
                ax.scatter(show_info[sub_set_index][0], show_info[sub_set_index][1], show_info[sub_set_index][2], s=100, c=color_list[sub_set_index])
            ax.set_zlabel(vector_name_list[2])
        ax.set_xlabel(vector_name_list[0])
        ax.set_ylabel(vector_name_list[1])
        plt.legend(label_name_list)

        if store_path:
            plt.tight_layout()
            if store_path[-3:] == 'jpg':
                fig.savefig(store_path, dpi=300, format='jpeg')
            elif store_path[-3:] == 'eps':
                fig.savefig(store_path, dpi=1200, format='eps')

        plt.show()

def DrawFeatureRelationshipAccordingToCsvFile(file_path, selected_feature_name_list, label_name_list, store_path=''):
    '''
    Help draw the feature relationship among different features according to the path of the data container.
    :param file_path: the file path of the csv file
    :param selected_feature_name_list: the features that would be drawn
    :param label_name_list: the name of the label. e.g. ['non-cnacer', 'cancer']
    :param store_path: The store path, supporting jpg and eps format.
    :return:
    '''
    data_container = DataContainer()
    data_container.Load(file_path)
    data_container.UsualNormalize()
    data, label, feature_name, case_name = data_container.GetData()

    if len(selected_feature_name_list) > 3 or len(selected_feature_name_list) < 1:
        print("Please check the length of the feature list. It can only show the relationship of the 1, 2, or 3 features")

    try:
        index = [feature_name.index(t) for t in selected_feature_name_list]
    except:
        print('The selected feature is not in the data container.')
        return
    result_data = []
    for sub_index in index:
        result_data.append(data[:, sub_index])
    DrawValueRelationship(result_data, selected_feature_name_list, label, label_name_list, store_path)


if __name__ == '__main__':
    pass
