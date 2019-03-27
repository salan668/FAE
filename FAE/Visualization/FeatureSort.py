import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

color = sns.color_palette('deep') + sns.color_palette('bright')


def FeatureSort(feature_name, group=np.array(()), group_name=[], value=[], store_path='',
                is_sort=True, is_show=True, fig=plt.figure()):
    '''
    Draw the plot of the sorted feature, an option is to draw different color according to the group and group value.

    :param feature_name: The name of the features
    :param group: the array to map the feature name to the group name, default is zeros
    :param group_name: the group name list to denote the group. default is []
    :param value: The value of each feature_name. Default is []
    :param store_path: The store path, supporting .jpeg and .tif
    :param is_sort: Boolen, to sort the features according to value. Default is True
    :return:

    Apr-29-18, Yang SONG [yang.song.91@foxmail.com]
    '''
    if group.size == 0 and group_name == []:
        group = np.zeros((len(feature_name), ), dtype=np.uint8)
        group_name = ['']

    if value == []:
        value = [len(feature_name) - index for index in range(len(feature_name))]
    else:
        value = np.abs(np.squeeze(value))

    if is_sort:
        sort_index = sorted(range(len(value)), key=lambda k: value[k], reverse=True)
        
        value = [value[index] for index in sort_index]
        feature_name = [feature_name[index] for index in sort_index]
        group = [group[index] for index in sort_index]

    assert(np.max(group) + 1 == len(group_name))
    sub_group = np.zeros((len(group_name), len(feature_name)))
    for index in range(len(feature_name)):
        sub_group[group[index], index] = value[index]
    y = range(len(feature_name))

    fig.clear()
    ax = fig.add_subplot(111)

    for index in range(sub_group.shape[0]):
        ax.barh(y, sub_group[index, :], color=color[index])

    ax.set_yticks(range(len(feature_name)))
    ax.set_yticklabels(feature_name)
    ax.set_xticks([])
    if len(group_name) > 1:
        ax.legend(group_name)

    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        fig.show()

    return ax

def ShortFeatureFullName(feature_full_name):
    if len(feature_full_name) <= 5:
        return feature_full_name

    sub = re.findall("[A-Z]", feature_full_name)
    if len(sub) == 1:
        return feature_full_name[:5]
    elif len(sub) == 0:
        return feature_full_name[:5]
    else:
        return ''.join(sub)

def SeperateRadiomicsFeatures(feature_name):
    '''
    Generate the feature name, group, and group cound according to radiomics features.

    :param feature_name: The generated radiomis featues. which should including sequence_name, image_class,
    feature_class, and the feature_name, like T2_origin_glszm_ZonePercentage
    :return:

    Apr-29-18 Yang SONG [yang.song.91@foxmail.com]
    '''
    sub_feature_name = []
    group = []
    group_name = []

    seq_group = []
    seq_count = 0
    image_class_group = []
    image_class_count = 0
    feature_class_group = []
    feature_class_count = 0

    for feature in feature_name:
        sep = feature.split('_')
        if len(sep) == 2:
            sep = [sep[0], '', '', sep[1]]

        seq = sep[0]
        image_class = sep[1]
        feature_class = sep[-2]

        if not seq in seq_group:
            seq_count += 1
        seq_group.append(seq)
        if not image_class in image_class_group:
            image_class_count += 1
        image_class_group.append(image_class)
        if not feature_class in feature_class_group:
            feature_class_count += 1
        feature_class_group.append(feature_class)

        sub_feature_name.append(ShortFeatureFullName(sep[-1]))

    if seq_count == 1:
        seq_group = ['' for index in range(len(feature_name))]
    if image_class_count == 1:
        image_class_group = ['' for index in range(len(feature_name))]
    if feature_class_count == 1:
        feature_class_group = ['' for index in range(len(feature_name))]

    group_count = 0
    for index in range(len(feature_name)):
        temp_name = seq_group[index] + '-' + image_class_group[index] + '-' + feature_class_group[index]
        if not temp_name in group_name:
            group_name.append(temp_name)
            group.append(group_count)
            group_count += 1
        else:
            group.append(group_name.index(temp_name))

    return sub_feature_name, np.asarray(group, dtype=np.uint8), group_name

def SortRadiomicsFeature(feature_name, value=[], store_path='', is_show=False, fig=plt.figure()):
    sub_feature_name, group, group_name = SeperateRadiomicsFeatures(feature_name)
    FeatureSort(sub_feature_name, group, group_name, value, store_path, is_show=is_show, fig=fig)

def GeneralFeatureSort(feature_name, value=[], store_path='', is_sort=True, max_num=-1, is_show=True, fig=plt.figure()):
    if not isinstance(value, list):
        value = list(value)
    if value == []:
        value = [len(feature_name) - index for index in range(len(feature_name))]

    if is_sort:
        sort_index = sorted(range(len(value)), key=lambda k: value[k], reverse=True)

        value = [value[index] for index in sort_index]
        feature_name = [feature_name[index] for index in sort_index]

    if max_num > 0:
        value = value[:max_num]
        feature_name = feature_name[:max_num]

    fig.clear()
    ax = fig.add_subplot(111)

    ax.barh(range(len(feature_name)), value, color=color[0])
    ax.set_yticks(range(len(feature_name)))
    ax.set_yticklabels(feature_name)
    ax.set_xticks([])

    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            plt.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            plt.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        fig.show()

    return ax

if __name__ == '__main__':
    # feature_name = ['DE', 'SAE', 'SZNUM', 'JE', 'Id']
    feature_name = ['10Per', 'Autoc', 'IR', 'GLV']
    value = [72.9, 45.4, 45.2, 41]
    group = np.array([1, 3, 2, 0])
    group_name = ['ADC--firstorder', 'DWI--firstorder', 'DWI--glcm', 'DWI--glszm']
    # group = [0, 1, 1, 0, 0]
    # group_name = ['GLCM', 'GLSZM']
    # value = 0.1, 0.5, 0.9, 0.2, 0.1
    FeatureSort(feature_name, group, group_name, value, is_show=True)

    # import pandas as pd
    # df = pd.read_csv(r'C:\Users\yangs\Desktop\anova_sort.csv', index_col=0)
    # feature_name = list(df.index)
    # value = list(df['F'])
    # new_feature_name = [ShortFeatureFullName(index) for index in feature_name]
    # GeneralFeatureSort(new_feature_name, value, max_num=4, is_show=True, store_path=r'D:\MyDocs\Document\研究生\毕业\毕业论文\图\组学\ANOVA_sort.jpg')
    # SortRadiomicsFeature(new_feature_name, value, is_show=True)



