import os
import json
import traceback
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

class HyperParameterManager:
    def __init__(self):
        self.__param_setting = [{}]
        pass

    def CleanJsonList(self):
        new_param_settting = []
        for one_param in self.__param_setting:
            new_one_param = {}
            for key, item in one_param.items():
                if item != 'None':
                    new_one_param[key] = item
            new_param_settting.append(new_one_param)
        self.__param_setting = deepcopy(new_param_settting)
        del new_param_settting

    def LoadSpecificConfig(self, name, relative_path):
        config_path = os.path.join(relative_path, name + '.json')
        self.__LoadConfig(config_path)

    def __LoadConfig(self, config_path):
        try:
            with open(config_path, 'r') as file:
                json_all = json.load(file, strict=False)
                self.__param_setting = list(ParameterGrid(json_all['setting']))
                self.CleanJsonList()
        except Exception:
            print('traceback.format_exc():\n', traceback.format_exc())
            self.__param_setting = [{}]
            pass


    def GetParameterSetting(self):
        return self.__param_setting