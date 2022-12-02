'''.
Feb 2, 2019.
Yang SONG, songyangmri@gmail.com
'''

import pandas as pd
from copy import deepcopy
from BC.DataContainer.DataContainer import DataContainer


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
    import pandas as pd
    df = pd.read_csv(r'C:\Users\Suns\Desktop\clinic_data2 .csv', delimiter='\t')

    new_info = pd.get_dummies(df, columns=['bingli', 'T', 'N', 'Clinic'])
    new_info.to_csv(r'c:\Users\Suns\Desktop\clinic_data2_onehot.csv')