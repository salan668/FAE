"""
SeriesStringMatch could find a specific series from a series list.

All rights reserved.
--Created by Yang Song. On Mar-10-2020
"""

class SeriesStringMatcher:
    def __init__(self, include_key=None, exclude_key=None):
        self.__include_key = []
        self.__exclude_key = []
        self.SetIncludeKey(include_key)
        self.SetExcludeKey(exclude_key)

    def SetIncludeKey(self, include_key):
        if isinstance(include_key, str):
            if include_key == '':
                self.__include_key = []
            else:
                self.__include_key = [include_key]
        elif isinstance(include_key, list):
            if include_key == ['']:
                self.__include_key = []
            else:
                self.__include_key = include_key
        elif include_key is None:
            self.__include_key = []

    def GetIncludeKey(self):
        return self.__include_key
    include_key = property(GetIncludeKey, SetIncludeKey)

    def SetExcludeKey(self, exclude_key):
        if isinstance(exclude_key, str):
            if exclude_key == '':
                self.__exclude_key = []
            else:
                self.__exclude_key = [exclude_key]
        elif isinstance(exclude_key, list):
            if exclude_key == ['']:
                self.__exclude_key = []
            else:
                self.__exclude_key = exclude_key
        elif exclude_key is None:
            self.__exclude_key = []

    def GetExcludeKey(self):
        return self.__exclude_key
    exclude_key = property(GetExcludeKey, SetExcludeKey)

    def Match(self, series_list):
        candidate = []
        for file in series_list:
            if all(key in file for key in self.__include_key):
                candidate.append(file)

        return [temp for temp in candidate if not any(key in temp for key in self.__exclude_key)]

