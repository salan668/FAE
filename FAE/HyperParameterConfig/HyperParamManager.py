import os
import json
from sklearn.model_selection import ParameterGrid

class HyperParameterManager:
    def __init__(self):
        self.__param_setting = [{}]
        pass

    def LoadSpecificConfig(self, name):
        config_path = os.path.join(r'FAE\HyperParameterConfig', name + '.json')
        self.__LoadConfig(config_path)

    def __LoadConfig(self, config_path):
        try:
            with open(config_path, 'r') as file:
                json_all = json.load(file, strict=False)
                self.__param_setting = list(ParameterGrid(json_all['setting']))
        except IOError:
            self.__param_setting = [{}]

    def GetParameterSetting(self):
        return self.__param_setting