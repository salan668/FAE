"""
All rights reserved. 
Author: Yang SONG (songyangmri@gmail.com)
"""

import os
import csv
from pathlib import Path
from abc import abstractmethod

import pandas as pd


class SeriesStringMatcher:
    """
    SeriesStringMatch could find a specific series from a series list.
    """
    def __init__(self, include_key= None, exclude_key=None, suffex=('', ), store_name=''):
        self.__include_key = []
        self.__exclude_key = []
        self.__suffex = ('', )
        self.__store_name = store_name
        self.SetIncludeKey(include_key)
        self.SetExcludeKey(exclude_key)
        self.SetSuffex(suffex)

    def LoadConfigFile(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == 'include_key':
                    self.__include_key = row[1:]
                elif row[0] == 'exclude_key':
                    self.__exclude_key= row[1:]
                elif row[0] == 'suffex':
                    self.__suffex = tuple(row[1:])

    def SaveConfigFile(self, file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['include_key'] + list(self.__include_key))
            writer.writerow(['exclude_key'] + list(self.__exclude_key))
            writer.writerow(['suffex'] + list(self.__suffex))

    def SetStoreName(self, store_name):
        self.__store_name = store_name
    def GetStoreName(self):
        return self.__store_name
    store_name = property(GetStoreName, SetStoreName)

    def SetIncludeKey(self, include_key):
        if isinstance(include_key, str):
            self.__include_key = [include_key]
        elif isinstance(include_key, list):
            self.__include_key = include_key
    def GetIncludeKey(self):
        return self.__include_key
    include_key = property(GetIncludeKey, SetIncludeKey)

    def SetExcludeKey(self, exclude_key):
        if isinstance(exclude_key, str) and exclude_key != '':
            self.__exclude_key = [exclude_key]
        elif isinstance(exclude_key, list) and exclude_key != ['']:
            self.__exclude_key = exclude_key
    def GetExcludeKey(self):
        return self.__exclude_key
    exclude_key = property(GetExcludeKey, SetExcludeKey)

    def SetSuffex(self, suffex):
        if isinstance(suffex, str):
            suffex = [suffex]
        if isinstance(suffex, list):
            suffex = tuple(suffex)
        if isinstance(suffex, tuple):
            self.__suffex = suffex

    def GetSuffex(self):
        return self.__suffex
    suffex = property(GetSuffex, SetSuffex)

    def SplitFile(self, file_name: str):
        if file_name.endswith('.nii.gz'):
            return file_name[:-len('.nii.gz')], '.nii.gz'
        else:
            return os.path.splitext(file_name)

    def Match(self, series_list):
        candidate = []
        for file in series_list:
            if file.endswith(self.__suffex):
                if all(key in file for key in self.__include_key):
                    candidate.append(file)

        candidate = [temp for temp in candidate if not any(key in temp for key in self.__exclude_key)]
        return candidate


class FileMatcherManager(object):
    def __init__(self):
        self.matchers = {}
        self.results = pd.DataFrame()
        self.error_info = pd.DataFrame()

    def Clear(self):
        self.results = pd.DataFrame()
        self.error_info = pd.DataFrame()

    def ClearMatcher(self):
        self.matchers = {}
        self.results = pd.DataFrame()
        self.error_info = pd.DataFrame()

    def AddOne(self, name: str, matcher: SeriesStringMatcher):
        if name in self.matchers.keys():
            raise KeyError("{} exists in all matchers.".format(name))

        self.matchers[name] = matcher

    def RemoveOne(self, name: str):
        if name not in self.matchers.keys():
            raise KeyError("{} does not exist in all matchers.".format(name))
        del self.matchers[name]

    def IsAllMatched(self):
        if self.results.size == 0:
            print('Match First')
            return False

        if self.results.isnull().values.any():
            return False
        else:
            return True

    def LoadResult(self, file_path):
        self.results = pd.read_csv(file_path, index_col=0)

    @abstractmethod
    def Match(self, root: Path):
        pass


class UniqueFileMatcherManager(FileMatcherManager):
    """
    结果保存成一个表格，列为每个case，行为每个series name，存储格式每个文件的对应路径
    """
    def __init__(self):
        super().__init__()

    def Match(self, root: Path, store_path=None):
        if not root.is_dir():
            raise OSError('{} is not a folder'.format(root))

        case_name_list = [one.name for one in root.iterdir() if one.is_dir()]
        series_name_list = list(self.matchers.keys())
        self.results = pd.DataFrame(columns=series_name_list, index=case_name_list)
        self.error_info = pd.DataFrame(columns=series_name_list, index=case_name_list)

        for case_name in case_name_list:
            case_folder = root / case_name
            for matcher_name, matcher in self.matchers.items():
                all_series_name = [one.name for one in list(case_folder.glob('*'))]
                candidates = matcher.Match(all_series_name)
                if len(candidates) == 1:
                    self.results.loc[case_name, matcher_name] = case_folder / candidates[0]
                else:
                    self.error_info.loc[case_name, matcher_name] = 'Can not match'

        if store_path is not None:
            self.results.to_csv(str(store_path))

        return self.results

    def EstimateCaseNumber(self, root):
        return len([one.name for one in root.iterdir() if one.is_dir()])

    def MatchVerbose(self, root: Path, store_path=None):
        if not root.is_dir():
            raise OSError('{} is not a folder'.format(root))

        case_name_list = [one.name for one in root.iterdir() if one.is_dir()]
        series_name_list = list(self.matchers.keys())
        self.results = pd.DataFrame(columns=series_name_list, index=case_name_list)
        self.error_info = pd.DataFrame(columns=series_name_list, index=case_name_list)

        for case_name in case_name_list:
            case_folder = root / case_name
            for matcher_name, matcher in self.matchers.items():
                all_series_name = [one.name for one in list(case_folder.glob('*'))]
                candidates = matcher.Match(all_series_name)
                if len(candidates) == 1:
                    self.results.loc[case_name, matcher_name] = case_folder / candidates[0]
                    yield True, case_name, matcher_name
                else:
                    self.error_info.loc[case_name, matcher_name] = 'Can not match'
                    yield False, case_name, matcher_name

        if store_path is not None:
            self.results.to_csv(str(store_path))
