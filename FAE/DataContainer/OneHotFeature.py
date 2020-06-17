'''.
Feb 2, 2019.
Yang SONG, songyangmri@gmail.com
'''

import pandas as pd
from copy import deepcopy
from FAE.DataContainer.DataContainer import DataContainer


class FeatureEncodingOneHot():
    def __init__(self):
        pass

    def OneHotOneColumn(self, data_container, feature_list):
        info = data_container.GetFrame()
        feature_name = data_container.GetFeatureName()
        for feature in feature_list:
            assert(feature in feature_name)

        new_info = pd.get_dummies(info, columns=feature_list)
        new_data = DataContainer()
        new_data.SetFrame(new_info)
        return new_data



if __name__ == '__main__':
    data = DataContainer()
    data.Load(r'c:\Users\yangs\Desktop\test.csv')
    info = data.GetFrame()

    new_info = pd.get_dummies(info, columns=['bGs', 'PIRADS', 't2score', 'DWIscore', 'MR_stage'])
    new_info.to_csv(r'c:\Users\yangs\Desktop\test_onehot.csv')