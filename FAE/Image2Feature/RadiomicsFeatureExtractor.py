from radiomics import featureextractor
import SimpleITK as sitk
import collections
import os
import csv
import copy
import logging
import pandas as pd
import glob

from FAE.Func.Visualization import LoadWaitBar

import traceback

class RadiomicsFeatureExtractor:
    def __init__(self, radiomics_parameter_file, config_file, modality_name_list):
        self.feature_values = []
        self.case_list = []
        self.feature_name_list = []
        self.config_dict = dict()

        self.modality_name_list = modality_name_list
        self.logger = logging.getLogger(__name__)
        try:
            self.extractor = featureextractor.RadiomicsFeaturesExtractor(radiomics_parameter_file)
            self.LoadFileConfig(config_file)
        except:
            traceback.print_exc()
            print('Check the config file path.')

    def LoadFileConfig(self, config_file):
        if config_file:
            with open(config_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    self.config_dict[row[0]] = row[1]

    def __GetFeatureValuesEachModality(self, data_path, roi_path, modality_name):
        try:
            result = self.extractor.execute(data_path, roi_path)
        except Exception as e:
            traceback.print_exc()
            return '', []


        feature_names = []
        feature_values = list(result.values())
        for feature_name in list(result.keys()):
            feature_names.append(modality_name + '_' + feature_name)
        return feature_names, feature_values

    def __GetFeatureValues(self, case_folder):
        feature_dict = {}
        if os.path.exists(os.path.join(case_folder, 'ROI.nii')):
            roi_path = os.path.join(case_folder, 'ROI.nii')
        elif os.path.exists(os.path.join(case_folder, 'ROI.nii.gz')):
            roi_path = os.path.join(case_folder, 'ROI.nii.gz')
        else:
            self.logger.error('Check the ROI file path of case: ' + case_folder)

        quality_feature_path = os.path.join(case_folder, 'QualityFeature.csv')
        if os.path.exists(quality_feature_path):
            with open(quality_feature_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    feature_dict['Quality_' + row[0]] = row[1]

        for data_num in range(len(self.modality_name_list)):
            modality_name = self.modality_name_list[data_num]

            if os.path.exists(os.path.join(case_folder,'data' + str(self.config_dict[str(modality_name)]) + '.nii')):
                data_name = 'data' + str(self.config_dict[str(modality_name)]) + '.nii'
                data_path = os.path.join(case_folder, data_name)
            elif os.path.exists(os.path.join(case_folder,'data' + str(self.config_dict[str(modality_name)]) + '.nii.gz')):
                data_name = 'data' + str(self.config_dict[str(modality_name)]) + '.nii.gz'
                data_path = os.path.join(case_folder, data_name)
            else:
                self.logger.error('Check the Data file path of case: ' + case_folder)

            feature_names_each_modality, feature_values_each_modality = self.__GetFeatureValuesEachModality(data_path, roi_path, modality_name)
            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                if feature_name in self.feature_name_list:
                    feature_dict[feature_name] = feature_value
                else:
                    print('Check the feature name in the feature name list')

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_values = list(feature_dict.values())

        # Add Label
        label_value = 0
        label_path = os.path.join(case_folder, 'label.csv')
        if os.path.exists(label_path):
            with open(label_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    label_value = row[0]

            feature_values.insert(0, label_value)
        else:
            print('No label file!: ', label_path)


        return feature_values

    def __MergeCase(self, case_name, feature_values):
        if case_name in self.case_list:
            print('The case exists!')
            return False
        else:
            self.case_list.append(case_name)
            self.feature_values.append(feature_values)
            return True

    def __InitialFeatureValues(self, case_folder):
        feature_dict = {}
        if os.path.exists(os.path.join(case_folder, 'ROI.nii')):
            roi_path = os.path.join(case_folder, 'ROI.nii')
        elif os.path.exists(os.path.join(case_folder, 'ROI.nii.gz')):
            roi_path = os.path.join(case_folder, 'ROI.nii.gz')
        else:
            self.logger.error('Check the ROI file path of case: ' + case_folder)

        # Add quality feature
        quality_feature_path = os.path.join(case_folder, 'QualityFeature.csv')
        if os.path.exists(quality_feature_path):
            with open(quality_feature_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"',  quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    feature_dict['Quality_' + row[0]] = row[1]

        # Add Radiomics features
        for data_num in range(len(self.modality_name_list)):
            modality_name = self.modality_name_list[data_num]
            data_path = os.path.join(case_folder, 'data' + str(self.config_dict[str(modality_name)]) + '.nii')
            if not os.path.exists(data_path):
                data_path = os.path.join(case_folder, 'data' + str(self.config_dict[str(modality_name)]) + '.nii.gz')

            feature_names_each_modality, feature_values_each_modality = self.__GetFeatureValuesEachModality(data_path, roi_path, modality_name)
            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                feature_dict[feature_name] = feature_value

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_names = list(feature_dict.keys())
        feature_values = list(feature_dict.values())

        # Add Label
        label_value = 0
        label_path = os.path.join(case_folder, 'label.csv')
        if os.path.exists(label_path):
            with open(label_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    label_value = row[0]
            feature_names.insert(0, 'label')
            feature_values.insert(0, label_value)
        else:
            print('No label file!: ', label_path)

        self.feature_name_list = feature_names
        self.feature_values.append(feature_values)
        self.__TempSave('temp.csv')

    def __TempSave(self, store_path):
        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for feature_name, feature_value in zip(self.feature_name_list, self.feature_values[0]):
                writer.writerow([feature_name, feature_value])

    def __IterateCase(self, root_folder, store_path=''):
        case_name_list = os.listdir(root_folder)
        case_name_list.sort()
        for case_name in case_name_list:
            case_path = os.path.join(root_folder, case_name)
            if os.path.isfile(case_path):
                continue
            print(case_name)
            if self.feature_name_list != [] and self.feature_values != []:
                feature_values = self.__GetFeatureValues(case_path)
                self.__MergeCase(case_name, feature_values)
            else:
                self.__InitialFeatureValues(case_path)
                self.case_list.append(case_name)

            if store_path:
                self.Save(store_path)

    def Save(self, store_path):
        header = copy.deepcopy(self.feature_name_list)
        header.insert(0, 'CaseName')
        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            for case_name, feature_value in zip(self.case_list, self.feature_values):
                row = list(map(str, feature_value))
                row.insert(0, case_name)
                writer.writerow(row)

    def Read(self, file_path):
        self.feature_values = []
        self.case_list = []
        self.feature_name_list = []

        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                if row[0] == '':
                    self.feature_name_list = row[1:]
                else:
                    self.case_list.append(row[0])
                    self.feature_values.append(row[1:])

    def Execute(self, root_folder, store_folder=''):
        if not os.path.exists(store_folder):
            os.mkdir(store_folder)
        self.__IterateCase(root_folder, store_path=os.path.join(store_folder, 'features.csv'))

class RadiomicsFeatureExtractorWithoutConfig:
    def __init__(self, radiomics_parameter_file, modality_name_list, ignore_tolorence=False, roi_key=''):
        self.feature_values = []
        self.case_list = []
        self.feature_name_list = []
        self.is_ignore_tolorence=ignore_tolorence
        if roi_key:
            self.roi_key = roi_key
        else:
            self.roi_key = 'roi'

        self.modality_name_list = modality_name_list
        self.logger = logging.getLogger(__name__)
        try:
            self.extractor = featureextractor.RadiomicsFeaturesExtractor(radiomics_parameter_file)
        except:
            traceback.print_exc()
            print('Initial Failed.')

    def __GetFeatureValuesEachModality(self, data_path, roi_path, modality_name):
        try:
            if self.is_ignore_tolorence:
                one_image = sitk.ReadImage(data_path)
                one_mask = sitk.ReadImage(roi_path)
                one_mask.CopyInformation(one_image)
                result = self.extractor.execute(one_image, one_mask)
            else:
                result = self.extractor.execute(data_path, roi_path)
        except Exception as e:
            traceback.print_exc()
            return '', []

        feature_names = []
        feature_values = list(result.values())
        for feature_name in list(result.keys()):
            feature_names.append(modality_name + '_' + feature_name)
        return feature_names, feature_values

    def __GetFeatureValues(self, case_folder):
        feature_dict = {}
        roi_path = glob.glob(os.path.join(case_folder, '*' + self.roi_key + '*'))
        if len(roi_path) != 1:
            self.logger.error('Check the ROI file path of case: ' + case_folder)
            return
        roi_path = roi_path[0]

        quality_feature_path = os.path.join(case_folder, 'QualityFeature.csv')
        if os.path.exists(quality_feature_path):
            with open(quality_feature_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    feature_dict['Quality_' + row[0]] = row[1]

        for modality_name in self.modality_name_list:
            if os.path.exists(os.path.join(case_folder, modality_name + '.nii')):
                data_path = os.path.join(case_folder, modality_name + '.nii')
            elif os.path.exists(os.path.join(case_folder, modality_name + '.nii.gz')):
                data_path = os.path.join(case_folder, modality_name + '.nii.gz')
            else:
                self.logger.error('Check the Data file path of case: ' + case_folder)

            feature_names_each_modality, feature_values_each_modality = self.__GetFeatureValuesEachModality(data_path, roi_path, modality_name)
            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                if feature_name in self.feature_name_list:
                    feature_dict[feature_name] = feature_value
                else:
                    print('Check the feature name in the feature name list')

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_values = list(feature_dict.values())

        # Add Label
        label_value = 0
        label_path = os.path.join(case_folder, 'label.csv')
        if os.path.exists(label_path):
            with open(label_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    label_value = row[0]

            feature_values.insert(0, label_value)
        else:
            print('No label file!: ', label_path)


        return feature_values

    def __MergeCase(self, case_name, feature_values):
        if case_name in self.case_list:
            print('The case exists!')
            return False
        else:
            self.case_list.append(case_name)
            self.feature_values.append(feature_values)
            return True

    def __InitialFeatureValues(self, case_folder):
        feature_dict = {}
        roi_path = glob.glob(os.path.join(case_folder, '*' + self.roi_key + '*'))
        if len(roi_path) != 1:
            self.logger.error('Check the ROI file path of case: ' + case_folder)
            return
        roi_path = roi_path[0]

        # Add quality feature
        quality_feature_path = os.path.join(case_folder, 'QualityFeature.csv')
        if os.path.exists(quality_feature_path):
            with open(quality_feature_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"',  quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    feature_dict['Quality_' + row[0]] = row[1]

        # Add Radiomics features
        for modality_name in self.modality_name_list:
            data_path = os.path.join(case_folder, modality_name + '.nii')
            if not os.path.exists(data_path):
                data_path = os.path.join(case_folder, modality_name + '.nii.gz')

            feature_names_each_modality, feature_values_each_modality = self.__GetFeatureValuesEachModality(data_path, roi_path, modality_name)
            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                feature_dict[feature_name] = feature_value

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_names = list(feature_dict.keys())
        feature_values = list(feature_dict.values())

        # Add Label
        label_value = 0
        label_path = os.path.join(case_folder, 'label.csv')
        if os.path.exists(label_path):
            with open(label_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    label_value = row[0]
            feature_names.insert(0, 'label')
            feature_values.insert(0, label_value)
        else:
            print('No label file!: ', label_path)

        self.feature_name_list = feature_names
        self.feature_values.append(feature_values)
        self.__TempSave('temp.csv')

    def __TempSave(self, store_path):
        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for feature_name, feature_value in zip(self.feature_name_list, self.feature_values[0]):
                writer.writerow([feature_name, feature_value])

    def __IterateCase(self, root_folder, store_path=''):
        case_name_list = os.listdir(root_folder)
        case_name_list.sort()
        for case_name in case_name_list:
            case_path = os.path.join(root_folder, case_name)
            if os.path.isfile(case_path):
                continue
            print(case_name)
            if self.feature_name_list != [] and self.feature_values != []:
                feature_values = self.__GetFeatureValues(case_path)
                self.__MergeCase(case_name, feature_values)
            else:
                self.__InitialFeatureValues(case_path)
                self.case_list.append(case_name)

            if store_path:
                self.Save(store_path)

    def Save(self, store_path):
        header = copy.deepcopy(self.feature_name_list)
        header.insert(0, 'CaseName')
        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            for case_name, feature_value in zip(self.case_list, self.feature_values):
                row = list(map(str, feature_value))
                row.insert(0, case_name)
                writer.writerow(row)

    def Read(self, file_path):
        self.feature_values = []
        self.case_list = []
        self.feature_name_list = []

        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                if row[0] == '':
                    self.feature_name_list = row[1:]
                else:
                    self.case_list.append(row[0])
                    self.feature_values.append(row[1:])

    def Execute(self, root_folder, store_path):
        if store_path and store_path.endswith('.csv'):
            self.__IterateCase(root_folder, store_path=store_path)

def main():
    extractor = RadiomicsFeatureExtractor(r'..\RadiomicsParams.yaml', r'x:\Radiomics_ZhangJing\MM_Ly\FileConfig.csv', ['T1C'])
    extractor.Execute(r'x:\Radiomics_ZhangJing\MM_Ly', store_folder=r'')

    
if __name__ == '__main__':
    main()