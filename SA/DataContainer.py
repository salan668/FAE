"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/7
"""
import os
from random import shuffle

import numpy as np
import pandas as pd
from lifelines.utils import to_long_format

from SA.Utility.MyLog import mylog


class DataContainer(object):
    def __init__(self, df=pd.DataFrame(), event_name=None, duration_name=None):
        self._df = df

        self._array = np.array([])
        self._feature_name = None
        self._case_name = None

        self.event_name = event_name
        self.duration_name = duration_name
        self.event = None
        self.duration = None

        if self._df.size != 0 and event_name is not None and duration_name is not None:
            self.UpdateData()

    def GetArray(self):
        return self._array
    def SetArray(self, array: np.array):
        self._array = array
    array = property(GetArray, SetArray)

    def GetFeatureName(self):
        return self._feature_name
    def SetFeatureName(self, feature_name: list):
        self._feature_name = feature_name
    feature_name = property(GetFeatureName, SetFeatureName)

    def GetCaseName(self):
        return self._case_name
    def SetCaseName(self, case_name: list):
        self._case_name = case_name
    case_name = property(GetCaseName, SetCaseName)
    
    def GetFrame(self):
        return self._df
    def SetFrame(self, df: pd.DataFrame):
        self._df = df
    df = property(GetFrame, SetFrame)

    def __str__(self):
        if self.IsEmpty():
            text = 'Empty!'
        else:
            text = "Case Number: {}\nFeature Number {}\nEvent Description: 1/0 = {}/{}".format(
                len(self._case_name), len(self._feature_name),
                np.sum(self.event), len(self._case_name) - np.sum(self.event)
            )
        return text

    def UpdateFrame(self):
        assert(len(self._feature_name) == self._array.shape[1])
        assert(len(self._case_name) == self._array.shape[0])
        df = pd.DataFrame(data=self._array, index=self._case_name, columns=self._feature_name)

        event_df = pd.DataFrame(data=self.event, index=self._case_name, columns=[self.event_name])
        duration_df = pd.DataFrame(data=self.duration, index=self._case_name, columns=[self.duration_name])

        self._df = pd.concat((event_df, duration_df, df), axis=1)

    def UpdateData(self):
        to_long_format(self._df, self.event_name)

        self.event = self._df[self.event_name]
        self.duration = self._df[self.duration_name]

        if np.unique(self.event.to_numpy()).size != 2:
            mylog.error('Key \'{}\' must have only two values'.format(
                self.event_name
            ))
            raise ValueError

        new_df = self.df.drop(columns=[self.event_name, self.duration_name], inplace=False)
        self._array = new_df.values
        self._case_name = list(new_df.index)
        self._feature_name = list(new_df.columns)

    def IsEmpty(self):
        return self._array.size == 0

    def Load(self, file_path, event_name: str, duration_name: str):
        self._df = pd.read_csv(file_path, index_col=0)
        if not (event_name in self._df.columns and duration_name in self._df.columns):
            mylog.error('{} and {} not in the columns of DataFrame ()'.format(
                event_name, duration_name, file_path
            ))
            raise KeyError

        self.event_name = event_name
        self.duration_name = duration_name

        self.UpdateData()

    def Save(self, file_path):
        self._df.to_csv(file_path)


class DataSplitter(object):
    def __init__(self):
        pass

    def _SplitIndex(self, label, test_percentage):
        training_index_list, testing_index_list = [], []
        for group in range(int(np.max(label.values)) + 1):
            index = np.where(label == group)[0]

            shuffle(index)
            testing_index = index[:round(len(index) * test_percentage)]
            training_index = index[round(len(index) * test_percentage):]

            training_index_list.extend(training_index)
            testing_index_list.extend(testing_index)
        return training_index_list, testing_index_list

    def SplitByRatio(self, dc: DataContainer, test_ratio=0.3,
                               store_folder=None, store_name=('train', 'test')):
        training_index_list, testing_index_list = self._SplitIndex(dc.event, test_ratio)

        train_df = dc.df.iloc[training_index_list, :]
        test_df = dc.df.iloc[testing_index_list, :]

        train_dc = DataContainer(df=train_df, event_name=dc.event_name, duration_name=dc.duration_name)
        test_dc = DataContainer(df=test_df, event_name=dc.event_name, duration_name=dc.duration_name)

        if store_folder:
            train_dc.Save(os.path.join(store_folder, '{}.csv'.format(store_name[0])))
            test_dc.Save(os.path.join(store_folder, '{}.csv'.format(store_name[1])))

        return train_dc, test_dc


if __name__ == '__main__':
    file_path = r'..\..\Demo\Radiomics_pvp_hcc_os_top20_train.csv'
    dc = DataContainer()
    dc.Load(file_path, event_name='status', duration_name='time')

    splitter = DataSplitter()
    train_dc, test_dc = splitter.SplitByRatio(dc, store_folder=r'..\..\Demo')
    print(train_dc)
    print(test_dc)
