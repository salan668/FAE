import os
import json
import traceback
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

from BC.Utility.Constants import BALANCE_UP_SAMPLING, BALANCE_DOWN_SAMPLING, BALANCE_SMOTE, BALANCE_SMOTE_TOMEK
from BC.Utility.Constants import CLASSIFIER_AB, CLASSIFIER_AE, CLASSIFIER_DT, CLASSIFIER_GP, CLASSIFIER_LR
from BC.Utility.Constants import CLASSIFIER_LRLasso, CLASSIFIER_RF, CLASSIFIER_SVM


class HyperParameterManager:
    def __init__(self):
        self.param_setting = [{}]
        pass

    def CleanJsonList(self):
        new_param_settting = []
        for one_param in self.param_setting:
            new_one_param = {}
            for key, item in one_param.items():
                if item != 'None':
                    new_one_param[key] = item
            new_param_settting.append(new_one_param)
        self.param_setting = deepcopy(new_param_settting)
        del new_param_settting

    def LoadSpecificConfig(self, name, relative_path):
        config_path = os.path.join(relative_path, name + '.json')
        self.LoadConfig(config_path)

    def LoadConfig(self, config_path):
        try:
            with open(config_path, 'r') as file:
                json_all = json.load(file, strict=False)
                self.param_setting = json_all['setting'][0]
                # self.param_setting = dict(ParameterGrid(json_all['setting']))
                # self.CleanJsonList()
        except Exception:
            print('traceback.format_exc():\n', traceback.format_exc())
            self.param_setting = {}
            pass

    def GetParameterSetting(self):
        return self.param_setting


class RandomSeed:
    def __init__(self, config_path):
        self.random_seed = {}
        self.LoadConfig(config_path)

    def LoadConfig(self, config_path):
        if not os.path.exists(config_path):
            print("Check config path:{}".format(config_path))
            return
        with open(config_path, 'r') as file:
            self.random_seed = json.load(file, strict=False)

            if BALANCE_UP_SAMPLING not in self.random_seed.keys():
                self.random_seed[BALANCE_UP_SAMPLING] = 0
            if BALANCE_DOWN_SAMPLING not in self.random_seed.keys():
                self.random_seed[BALANCE_DOWN_SAMPLING] = 0
            if BALANCE_SMOTE not in self.random_seed.keys():
                self.random_seed[BALANCE_SMOTE] = 0
            if BALANCE_SMOTE_TOMEK not in self.random_seed.keys():
                self.random_seed[BALANCE_SMOTE_TOMEK] = 0

            if CLASSIFIER_AB not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_AB] = 0
            if CLASSIFIER_AE not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_AE] = 0
            if CLASSIFIER_DT not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_DT] = 0
            if CLASSIFIER_GP not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_GP] = 0
            if CLASSIFIER_LR not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_LR] = 0
            if CLASSIFIER_LRLasso not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_LRLasso] = 0
            if CLASSIFIER_RF not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_RF] = 0
            if CLASSIFIER_SVM not in self.random_seed.keys():
                self.random_seed[CLASSIFIER_SVM] = 0

def GetClassifierHyperParams(root):
    param_dict = {}
    for one_file in os.listdir(root):
        if one_file.endswith('.json'):
            one_classifier = one_file.split('.')[0]
            one_hyper_param = HyperParameterManager()
            one_hyper_param.LoadConfig(os.path.join(root, one_file))
            param_dict[one_classifier] = one_hyper_param.GetParameterSetting()
    return param_dict

# root = os.path.abspath(os.getcwd())
# RANDOM_SEED = RandomSeed(os.path.join(root, 'BC', 'HyperParameters', 'RandomSeed.json')).random_seed
# CLASSIFIER_HYPER_PARAMS = GetClassifierHyperParams(os.path.join(root, 'BC', 'HyperParameters', 'Classifier'))

if __name__ == '__main__':
    print(RandomSeed(r'D:\MyCode\FAE\FAE\BC\HyperParameters\RandomSeed.json').random_seed)
    param_dict = GetClassifierHyperParams(r'D:\MyCode\FAE\FAE\BC\HyperParameters\Classifier')
    for key, item in param_dict.items():
        print(key)
        print(item)
