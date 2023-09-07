import yaml
import os
#r'..\RadiomicsParams.yaml'

feature_classes_inface2yaml ={'First Order': 'firstorder',
                              'Shape-based': 'shape',
                              'GLCM': 'glcm',
                              'GLRLM': 'glrlm',
                              'GLSZM': 'glszm',
                              'GLDM': 'gldm',
                              'NGTDM': 'ngtdm'}

image_classes_inface2yaml = {'Exponential': 'Exponential',
                             'Gradient': 'Gradient',
                             'Local Binary Pattern (2D)': 'LBP2D',
                             'Local Binary Pattern (3D)': 'LBP3D',
                             'Laplacian of Gaussian': 'LoG',
                             'Logarithm': 'Logarithm',
                             'Original': 'Original',
                             'Square': 'Square',
                             'Square Root': 'SquareRoot',
                             'Wavelet Transform': 'Wavelet'}



class RadiomicsParamsConfig(object):
    def __init__(self, file_path):
        self.config_path = file_path
        self.feature_classes_key = 'featureClass'
        self.feature_classes = {}
        self.image_classes_key = 'imageType'
        self.image_classes = {}
        self.settings_key = 'settings'
        self.settings = {}

    def LoadConfig(self, config_path):
        self.config_path = config_path

    def SaveConfig(self, config_dict):
        if config_dict is None:
            config_dict = {}

        # Refine the config file for using.
        for image_type in config_dict[self.image_classes_key]:
            if image_type == 'LoG':
                config_dict[self.image_classes_key][image_type] = {'sigma': [1]}
            else:
                config_dict[self.image_classes_key][image_type] = {}

        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file)


if __name__ == '__main__':
    # temp = {"Exponential": 'Exponential',
    #                              "Gradient": 'Gradient',
    #                              "LBP2D": 'Local Binary Pattern (2D)',
    #                              "LBP3D": 'Local Binary Pattern (3D)',
    #                              "LoG": 'Laplacian of Gaussian'}
    # if temp.__contains__('LoG'):
    #     print('Log')
    radiomics_params = RadiomicsParamsConfig(r'RadiomicsParams.yaml')
    radiomics_params.LoadConfig()
    temp = {'Gradient', 'Laplacian of Gaussian'}
    radiomics_params.SetImageClasses(temp)
    feature = {'GLRLM', 'First Order Statistics'}
    radiomics_params.SetFeatureClasses(feature)
    radiomics_params.SaveConfig()

    # feature_classes = radiomics_params.GetFeatureClasses()
    # print(feature_classes)









