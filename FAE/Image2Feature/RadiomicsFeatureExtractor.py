from radiomics import featureextractor
import collections
import os
import csv
import copy
import logging
import SimpleITK as sitk
import glob

class RadiomicsFeatureExtractor:
    def __init__(self, radiomics_parameter_file, has_label=True, ignore_tolerence=False, ignore_diagnostic=True):
        self.feature_values = []
        self.case_list = []
        self.feature_name_list = []
        self.extractor = featureextractor.RadiomicsFeaturesExtractor(radiomics_parameter_file)

        self.logger = logging.getLogger(__name__)

        self.__has_label = has_label
        self.__is_ignore_tolerence = ignore_tolerence
        self.__is_ignore_diagnostic = ignore_diagnostic


    def __GetFeatureValuesEachModality(self, data_path, roi_path, key_name):
        if self.__is_ignore_tolerence:
            image = sitk.ReadImage(data_path)
            roi_image = sitk.ReadImage(roi_path)
            roi_image.CopyInformation(image)
            result = self.extractor.execute(image, roi_image)
        else:
            result = self.extractor.execute(data_path, roi_path)

        feature_names = []
        feature_values = []
        for feature_name, feature_value in zip(list(result.keys()), list(result.values())):
            if self.__is_ignore_diagnostic and 'diagnostics' in feature_name:
                continue
            feature_names.append(key_name + '_' + feature_name)
            feature_values.append(feature_value)
        return feature_names, feature_values

    def __GetFeatureValues(self, case_folder, key_name_list, show_key_name_list, roi_key):
        feature_dict = {}

        if isinstance(roi_key, str):
            roi_key = [roi_key]
        roi_key_path = '*'
        for one_roi_key in roi_key:
            roi_key_path += '{}*'.format(one_roi_key)

        roi_candidate = glob.glob(os.path.join(case_folder, roi_key_path))
        if len(roi_candidate) != 1:
            self.logger.error('Check the ROI file path of case: ' + case_folder)
            return None
        roi_file_path = roi_candidate[0]

        quality_feature_path = os.path.join(case_folder, 'QualityFeature.csv')
        if os.path.exists(quality_feature_path):
            with open(quality_feature_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    feature_dict['Quality_' + row[0]] = row[1]

        for sequence_key, show_key in zip(key_name_list, show_key_name_list):
            sequence_candidate = glob.glob(os.path.join(case_folder, '*{}*'.format(sequence_key)))
            if len(sequence_candidate) != 1:
                self.logger.error('Check the ROI file path of case: ' + case_folder)
                return None
            sequence_file_path = sequence_candidate[0]

            feature_names_each_modality, feature_values_each_modality = \
                self.__GetFeatureValuesEachModality(sequence_file_path, roi_file_path, show_key)

            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                if feature_name in self.feature_name_list:
                    feature_dict[feature_name] = feature_value
                else:
                    print('Check the feature name in the feature name list')

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_values = list(feature_dict.values())

        if self.__has_label:
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

    def __InitialFeatureValues(self, case_folder, key_name_list, show_key_name_list, roi_key):
        feature_dict = {}

        if isinstance(roi_key, str):
            roi_key = [roi_key]
        roi_key_path = '*'
        for one_roi_key in roi_key:
            roi_key_path += '{}*'.format(one_roi_key)
        roi_candidate = glob.glob(os.path.join(case_folder, roi_key_path))

        if len(roi_candidate) != 1:
            self.logger.error('Check the ROI file path of case: ' + case_folder)
            return None
        roi_file_path = roi_candidate[0]

        # Add quality feature
        quality_feature_path = os.path.join(case_folder, 'QualityFeature.csv')
        if os.path.exists(quality_feature_path):
            with open(quality_feature_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"',  quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    feature_dict['Quality_' + row[0]] = row[1]

        # Add Radiomics features
        for sequence_key, show_key in zip(key_name_list, show_key_name_list):
            sequence_candidate = glob.glob(os.path.join(case_folder, '*{}*'.format(sequence_key)))
            if len(sequence_candidate) != 1:
                self.logger.error('Check the data file path of case: ' + case_folder)
                return None
            sequence_file_path = sequence_candidate[0]

            feature_names_each_modality, feature_values_each_modality = self.__GetFeatureValuesEachModality(
                sequence_file_path, roi_file_path, show_key)
            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                feature_dict[feature_name] = feature_value

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_names = list(feature_dict.keys())
        feature_values = list(feature_dict.values())

        # Add Label
        if self.__has_label:
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

    def __IterateCase(self, root_folder, key_name_list, roi_key, show_key_name_list, store_path=''):
        case_name_list = os.listdir(root_folder)
        case_name_list.sort()
        for case_name in case_name_list:
            case_path = os.path.join(root_folder, case_name)
            if not os.path.isdir(case_path):
                continue
            print(case_name)
            try:
                if self.feature_name_list != [] and self.feature_values != []:
                    feature_values = self.__GetFeatureValues(case_path, key_name_list, show_key_name_list, roi_key)
                    self.__MergeCase(case_name, feature_values)
                else:
                    self.__InitialFeatureValues(case_path, key_name_list, show_key_name_list, roi_key)
                    self.case_list.append(case_name)

                if store_path:
                    self.Save(store_path)
            except:
                self.error_list.append(case_name)

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

    def Execute(self, root_folder, key_name_list, roi_key, store_path, show_key_name_list=[]):
        if len(show_key_name_list) == 0:
            show_key_name_list = key_name_list
        assert(len(show_key_name_list) == len(key_name_list))

        if not store_path.endswith('.csv'):
            print('The store path should be a CSV format')
        else:
            self.__IterateCase(root_folder, key_name_list, roi_key, store_path=store_path, show_key_name_list=show_key_name_list)

if __name__ == '__main__':
    extractor = RadiomicsFeatureExtractor(r'OnlyIntensityRadiomicsParams.yaml',
                                          has_label=False)
    extractor.Execute(r'C:\Users\yangs\Desktop\LiuWei',
                      key_name_list=['arterial'],
                      roi_key=['arterial', 'label'],
                      store_path=r'C:\Users\yangs\Desktop\LiuWei\artery.csv')

