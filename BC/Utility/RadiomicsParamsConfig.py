import yaml
import os
#r'..\RadiomicsParams.yaml'

feature_classes_inface2yaml ={'First Order Statistics': 'firstorder',
                              'Shape-based': 'shape',
                              'GLCM': 'glcm',
                              'GLRLM': 'glrlm',
                              'GLSZM': 'glszm',
                              'GLDM': 'gldm',
                              'NGTDM': 'ngtdm'}

feature_classes_yaml2inface = {'firstorder': 'First Order Statistics',
                               'shape': 'Shape-based',
                               'glcm': 'GLCM',
                               'glrlm': 'GLRLM',
                               'glszm': 'GLSZM',
                               'gldm': 'GLDM',
                               'ngtdm': 'NGTDM'}

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

image_classes_yaml2inface = {'Exponential': 'Exponential',
                             'Gradient': 'Gradient',
                             'LBP2D': 'Local Binary Pattern (2D)',
                             'LBP3D': 'Local Binary Pattern (3D)',
                             'LoG': 'Laplacian of Gaussian',
                             'Logarithm': 'Logarithm',
                             'Original': 'Original',
                             'Square': 'Square',
                             'SquareRoot': 'Square Root',
                             'Wavelet': 'Wavelet Transform'}

class  RadiomicsParamsConfig:
    def __init__(self, file_path):
        self.__config_path = file_path
        self.__feature_classes_key = 'featureClass'
        self.__feature_classes = {}
        self.__image_classes_key = 'imageType'
        self.__image_classes = {}

    def LoadConfig(self):
        file = open(self.__config_path, 'r', encoding='utf-8')
        content = file.read()
        # config = yaml.load(content, Loader=yaml.FullLoader)
        config = yaml.load(content)
        self.__image_classes = config[self.__image_classes_key]
        self.__feature_classes = config[self.__feature_classes_key]
        file.close()

    def GetImageClasses(self):
        selected_image_classes = []
        for key in self.__image_classes.keys():
            if image_classes_yaml2inface.__contains__(key):
                selected_image_classes.append(image_classes_yaml2inface[key])
        return selected_image_classes

    def GetFeatureClasses(self):
        selected_feature_classes = []
        for key in self.__feature_classes.keys():
            if feature_classes_yaml2inface.__contains__(key):
                selected_feature_classes.append(feature_classes_yaml2inface[key])
        return selected_feature_classes

    def SetFeatureClasses(self, selected_feature_classes):
        self.__feature_classes.clear()
        for feature in selected_feature_classes:
            temp = feature_classes_inface2yaml.get(feature)
            self.__feature_classes[temp] = None

    def SetImageClasses(self, selected_iamge_classes):
        self.__image_classes.clear()
        for image in selected_iamge_classes:
            temp = image_classes_inface2yaml.get(image)
            self.__image_classes[temp] = {}
        print(self.__image_classes)

    def SaveConfig(self):
        file = open(self.__config_path, 'r', encoding='utf-8')
        content = file.read()
        config = yaml.load(content)
        file.close()
        file = open(self.__config_path, 'w', encoding='utf-8')

        config[self.__image_classes_key] = self.__image_classes
        config[self.__feature_classes_key] = self.__feature_classes
        yaml.dump(config, file)


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









