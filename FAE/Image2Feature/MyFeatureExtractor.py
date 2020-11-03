

import os
import csv
import copy
import logging

from radiomics.featureextractor import *
from Utility.SeriesMatcher import SeriesStringMatcher


class MyFeatureExtractor:
    def __init__(self, radiomics_parameter_file, ignore_tolerance=False, ignore_diagnostic=True):
        self.feature_values = []
        self.case_list = []
        self.feature_name_list = []
        self.extractor = RadiomicsFeatureExtractor(radiomics_parameter_file)
        self.error_list = []

        self.logger = logging.getLogger(__name__)

        self._ignore_tolerance = ignore_tolerance
        self._ignore_diagnostic = ignore_diagnostic
        self.error_list = []

    def _GetFeatureValuesEachModality(self, data_path, roi_path, key_name):
        if self._ignore_tolerance:
            image = sitk.ReadImage(data_path)
            roi_image = sitk.ReadImage(roi_path)
            roi_image.CopyInformation(image)
            result = self.extractor.execute(image, roi_image)
        else:
            result = self.extractor.execute(data_path, roi_path)

        feature_names = []
        feature_values = []
        for feature_name, feature_value in zip(list(result.keys()), list(result.values())):
            if self._ignore_diagnostic and 'diagnostics' in feature_name:
                continue
            feature_names.append(key_name + '_' + feature_name)
            feature_values.append(feature_value)
        return feature_names, feature_values

    def _GetFeatureValues(self, case_folder, image_matcher_list, show_name_list, roi_matcher):
        feature_dict = dict()
        series_list = os.listdir(case_folder)

        roi_file_path = roi_matcher.Match(series_list)
        assert (len(roi_file_path) == 1)
        roi_file_path = roi_file_path[0]

        for one_matcher, name in zip(image_matcher_list, show_name_list):
            image_file_path = one_matcher.Match(series_list)
            assert (len(image_file_path) == 1)
            image_file_path = image_file_path[0]

            feature_names_each_modality, feature_values_each_modality = \
                self._GetFeatureValuesEachModality(os.path.join(case_folder, image_file_path),
                                                   os.path.join(case_folder, roi_file_path),
                                                   name)

            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                if feature_name in self.feature_name_list:
                    feature_dict[feature_name] = feature_value
                else:
                    print('Check the feature name in the feature name list')

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_values = list(feature_dict.values())

        return feature_values

    def _InitialFeatureValues(self, case_folder, image_matcher_list, show_name_list, roi_matcher):
        feature_dict = {}

        series_list = os.listdir(case_folder)

        roi_file_path = roi_matcher.Match(series_list)
        assert (len(roi_file_path) == 1)
        roi_file_path = roi_file_path[0]

        for one_matcher, name in zip(image_matcher_list, show_name_list):
            image_file_path = one_matcher.Match(series_list)
            assert (len(image_file_path) == 1)
            image_file_path = image_file_path[0]

            feature_names_each_modality, feature_values_each_modality = self._GetFeatureValuesEachModality(
                os.path.join(case_folder, image_file_path),
                os.path.join(case_folder, roi_file_path)
                , name)

            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                feature_dict[feature_name] = feature_value

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_values = list(feature_dict.values())
        feature_names = list(feature_dict.keys())

        self.feature_name_list = feature_names
        self.feature_values.append(feature_values)
        self._TempSave('temp.csv')

    def _MergeCase(self, case_name, feature_values):
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

    def _TempSave(self, store_path):
        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for feature_name, feature_value in zip(self.feature_name_list, self.feature_values[0]):
                writer.writerow([feature_name, feature_value])

    def _IterateCase(self, root_folder, image_matcher_list, roi_matcher, show_name_list, store_path=''):
        case_name_list = os.listdir(root_folder)
        case_name_list.sort()
        for index, case_name in enumerate(case_name_list):
            case_path = os.path.join(root_folder, case_name)
            if not os.path.isdir(case_path):
                continue
            try:
                if self.feature_name_list != [] and self.feature_values != []:
                    feature_values = self._GetFeatureValues(case_path, image_matcher_list, show_name_list, roi_matcher)
                    if not self._MergeCase(case_name, feature_values):
                        self.error_list.append(case_name)
                else:
                    self._InitialFeatureValues(case_path, image_matcher_list, show_name_list, roi_matcher)
                    self.case_list.append(case_name)

            except Exception as e:
                print(e)
                self.error_list.append(case_name)

            finally:
                yield index + 1, case_name

        if store_path:
            self.Save(store_path)

        with open(os.path.join(store_path + '_error.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for error_case in self.error_list:
                writer.writerow([error_case])

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

    def Execute(self, root_folder, image_matcher_list, roi_matcher, show_name_list, store_path):
        if not store_path.endswith('.csv'):
            print('The store path should be a CSV format')
        else:
            for index, case_name in self._IterateCase(root_folder, image_matcher_list, roi_matcher,
                                                      show_name_list=show_name_list, store_path=store_path):
                yield index, case_name

    def Run(self, case_name, image_list, roi_list, show_name_list, store_path):
        self.case_list.append(case_name)
        feature_dict = {}

        for image, roi, name in zip(image_list, roi_list, show_name_list):
            feature_names_each_modality, feature_values_each_modality = self._GetFeatureValuesEachModality(
                image, roi,  name)

            for feature_name, feature_value in zip(feature_names_each_modality, feature_values_each_modality):
                feature_dict[feature_name] = feature_value

        feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
        feature_values = list(feature_dict.values())
        feature_names = list(feature_dict.keys())

        self.feature_name_list = feature_names
        self.feature_values.append(feature_values)
        self.Save(store_path)

def main():
    extractor = MyFeatureExtractor(r'..\..\OnlyIntensityRadiomicsParams.yaml')

    image_matcher = [SeriesStringMatcher(include_key=['t2'], exclude_key='')]
    roi_matcher = SeriesStringMatcher(include_key=['roi'], exclude_key='')

    for index, name in extractor.Execute(r'D:\Data\HuangLi\107cases T2 ROI',
                                         image_matcher, roi_matcher, ['IVIM'], r'..\..\test.csv'):
        print(index, name)


if __name__ == '__main__':
    main()
