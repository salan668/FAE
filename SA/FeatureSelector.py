"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/15
"""

import os
import csv

import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from SA.DataContainer import DataContainer
from SA.Utility.MyLog import mylog


class FeatureSelector(object):
    def __init__(self, method=None, feature_name=None, selected_number=0,
                 name=None, description=None):
        self._Method = method
        self.feature_name = feature_name
        self.name = name
        self._description = description
        self.selected_number = selected_number
        pass

    def _GenerateSubDc(self, dc:DataContainer, new_data: np.array, new_feature: list,
                       is_replace=False, store_path=None):
        if is_replace:
            dc.array = new_data
            dc.feature_name = new_feature
            dc.UpdateFrame()
            new_dc = deepcopy(dc)
        else:
            new_dc = deepcopy(dc)
            new_dc.array = new_data
            new_dc.feature_name = new_feature
            new_dc.UpdateFrame()

        if store_path and isinstance(store_path, str):
            new_dc.Save(store_path)
        return new_dc

    def SelectByIndex(self, dc: DataContainer, selected_list,
                      is_replace=False, store_path=None):
        new_data = dc.GetArray()[:, selected_list]
        new_feature = [dc.GetFeatureName()[t] for t in selected_list]

        return self._GenerateSubDc(dc, new_data, new_feature,
                                   is_replace, store_path)

    def SelectByName(self, dc: DataContainer, selected_feature_name,
                     is_replace=False, store_path=None):
        assert(all(item in dc.feature_name for item in selected_feature_name))
        new_data = dc.GetFrame()[selected_feature_name].values

        return self._GenerateSubDc(dc, new_data, selected_feature_name,
                                   is_replace, store_path)

    def Transform(self, dc, is_replace=False, store_folder=None, store_key=None):
        if store_folder is not None and store_key is not None:
            return self.SelectByName(dc, self.feature_name, is_replace=is_replace,
                                     store_path=os.path.join(store_folder, '{}_sub_features.csv'.format(store_key)))
        else:
            return self.SelectByName(dc, self.feature_name, is_replace=is_replace)

    def Fit(self, dc: DataContainer, select_number=-1):
        if dc.IsEmpty():
            mylog.warning("DataContainer is Empty")
            return dc

        if select_number > 0 and self.name != 'SelectAll':
            self.selected_number = select_number

        new_dc = self._Method(dc, self.selected_number)
        self.feature_name = new_dc.feature_name

    def GetName(self):
        return self.name

    def GetDescription(self):
        return self._description

    def Save(self, store_folder):
        assert(os.path.isdir(store_folder))
        info = [['feature_number', self.selected_number],
                ['selected_feature'] + self.feature_name
                ]

        with open(os.path.join(store_folder, 'selected_features_{}_{}'.format(
            self.selected_number, self._name
        )), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(info)

    def Load(self, store_path):
        with open(store_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'feature_number':
                    self._feature_number = row[1]
                elif row[0] == 'selected_feature':
                    self.feature_name = row[1:]
                else:
                    mylog.error('The first work is not feature_number or selected_feature in the selector file: '
                                '{}'.format(store_path))
                    raise KeyError


def NoneSelect(dc, number=0):
    return dc
description = 'We used all features to build the model. '
FeatureSelectorAll = FeatureSelector(NoneSelect, name='SelectAll', description=description)

def ClusterSelect(dc, number):
    sub_features = []

    if number == 1:
        # mylog.warning('The minimum number of KMeans is 2, Kmeans doesn\'t apply')
        pccs = [abs(pearsonr(dc.df[one_feature].values, dc.event)[0]) for one_feature in dc.feature_name]
        selected_feature = dc.feature_name[pccs.index(max(pccs))]
        sub_features.append(selected_feature)
    else:
        clusters = KMeans(n_clusters=number, random_state=0, init='k-means++', n_init="auto").fit_predict(dc.array.transpose())

        for i in range(number):
            clustering_features = [name for cluster_index, name in zip(clusters, dc.feature_name) if cluster_index == i]
            pccs = [abs(pearsonr(dc.df[one_feature].values, dc.event)[0]) for one_feature in clustering_features]
            selected_feature = clustering_features[pccs.index(max(pccs))]
            sub_features.append(selected_feature)

    sub_dc = FeatureSelector().SelectByName(dc, selected_feature_name=sub_features)
    return sub_dc
description = 'Clustering Selection. '
FeatureSelectorCluster = FeatureSelector(ClusterSelect, name='Cluster', description=description)


if __name__ == '__main__':
    file_path = r'C:\Users\yangs\Desktop\Radiomics_pvp_hcc_os_top20_train.csv'
    dc = DataContainer()
    dc.Load(file_path, event_name='status', duration_name='time')

    fs = FeatureSelectorCluster
    fs.selected_number = 2
    fs.Fit(dc)
    new_dc = fs.Transform(dc)
    print(new_dc)
