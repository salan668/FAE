import collections
import os
import csv
import copy
import glob

import SimpleITK as sitk
from radiomics import featureextractor

from BC.Utility.EcLog import eclog


class RadiomicsFeatureExtractor:
    def __init__(self, radiomics_parameter_file, has_label=True, ignore_tolerence=False, ignore_diagnostic=True):
        self.feature_values = []
        self.case_list = []
        self.feature_name_list = []
        self.extractor = featureextractor.RadiomicsFeatureExtractor(radiomics_parameter_file)
        self.error_list = []
        self._file_name = os.path.split(__file__)

        self.__has_label = has_label
        self.__is_ignore_tolerence = ignore_tolerence
        self.__is_ignore_diagnostic = ignore_diagnostic
        self.error_list = []

    def _RemoveKeyFromPathList(self, path_list, key):
        new_path_list = []
        for p in path_list:
            if key not in str(p):
                new_path_list.append(p)

        return new_path_list


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
        roi_candidate = [one for one in roi_candidate if 'nii' in one]
        if len(roi_candidate) != 1:
            eclog(self._file_name).LogError('Check the ROI file path of case: ' + case_folder)
            return None
        roi_file_path = roi_candidate[0]

        quality_feature_path = os.path.join(case_folder, 'QualityFeature.csv')
        if os.path.exists(quality_feature_path):
            with open(quality_feature_path, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    feature_dict['Quality_' + row[0]] = row[1]

        for sequence_key, show_key in zip(key_name_list, show_key_name_list):
            if isinstance(sequence_key, str):
                sequence_key = [sequence_key]
            sequence_key_path = '*'
            for one_sequence_key in sequence_key:
                sequence_key_path += '{}*'.format(one_sequence_key)

            sequence_candidate = glob.glob(os.path.join(case_folder, sequence_key_path))
            for one_roi_key in roi_key:
                sequence_candidate = self._RemoveKeyFromPathList(sequence_candidate, one_roi_key)
            if len(sequence_candidate) != 1:
                eclog(self._file_name).LogError('Check the data file path of case: ' + sequence_key_path)
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
            if isinstance(feature_values, list) and len(feature_values) == len(self.feature_values[0]):
                self.case_list.append(case_name)
                self.feature_values.append(feature_values)
                return True
            else:
                print('Not extract valid features')
                return False

    def __InitialFeatureValues(self, case_folder, key_name_list, show_key_name_list, roi_key):
        feature_dict = {}

        if isinstance(roi_key, str):
            roi_key = [roi_key]
        roi_key_path = '*'
        for one_roi_key in roi_key:
            roi_key_path += '{}*'.format(one_roi_key)
        roi_candidate = glob.glob(os.path.join(case_folder, roi_key_path))
        roi_candidate = [one for one in roi_candidate if 'nii' in one]

        if len(roi_candidate) != 1:
            eclog(self._file_name).LogError('Check the ROI file path of case: ' + case_folder)
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
            if isinstance(sequence_key, str):
                sequence_key = [sequence_key]
            sequence_key_path = '*'
            for one_sequence_key in sequence_key:
                sequence_key_path += '{}*'.format(one_sequence_key)

            sequence_candidate = glob.glob(os.path.join(case_folder, sequence_key_path))
            for one_roi_key in roi_key:
                sequence_candidate = self._RemoveKeyFromPathList(sequence_candidate, one_roi_key)

            if len(sequence_candidate) != 1:
                eclog(self._file_name).LogError('Check the data file path of case: ' + sequence_key_path)
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
                    if not self.__MergeCase(case_name, feature_values):
                        self.error_list.append(case_name)
                else:
                    self.__InitialFeatureValues(case_path, key_name_list, show_key_name_list, roi_key)
                    self.case_list.append(case_name)

            except Exception as e:
                content = 'In FeatureExtractor, {} extracted failed: '.format(case_name)
                eclog(self._file_name).LogError('{}{}'.format(content, str(e)))
                self.error_list.append(case_name)
                print('{} \n{}'.format(content, e.__str__()))
                raise e

        if store_path:
            self.Save(store_path)

        with open(os.path.join(store_path + '_error.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for error_case in self.error_list:
                writer.writerow([error_case])

    def Save(self, store_path):
        header = copy.deepcopy(self.feature_name_list)
        header.insert(0, 'CaseID')
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
    pass

