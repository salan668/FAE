import os

import numpy as np
import pandas as pd
import xlrd
import xlwt
import copy

from scipy.stats import levene, ttest_ind, kstest, mannwhitneyu, chi2_contingency
from collections import Counter


class FeatureStatistic:
    def __init__(self, feature_folder, selected_feature_list=[]):
        self.feature_folder = feature_folder
        self.selected_feature_list = selected_feature_list
        self.feature_list = []
        self.train_pd = pd.DataFrame()
        self.test_pd = pd.DataFrame()

    def load_train_test_feature(self):

        self.train_pd = pd.read_csv(os.path.join(self.feature_folder, 'train_numeric_feature.csv'), index_col=0)
        self.test_pd = pd.read_csv(os.path.join(self.feature_folder, 'test_numeric_feature.csv'), index_col=0)
        # 判断特征数是否相同
        if len(self.train_pd.columns.tolist()) != len(self.test_pd.columns.tolist()):
            print('feature of train mismatch test')
        else:
            self.feature_list = self.train_pd.columns.tolist()
            # 去除label 和 acquisition date
            if 'label' in self.feature_list:
                self.feature_list.remove('label')
            elif 'Label' in self.feature_list:
                self.feature_list.remove('label')
            if 'acquisition date' in self.feature_list:
                self.feature_list.remove('acquisition date')

    @staticmethod
    def feature_value_continuous(feature_value_array):
        feature_mean = np.mean(feature_value_array)
        feature_std = np.std(feature_value_array)
        return feature_mean, feature_std

    @staticmethod
    def _statistic_continuous(array_0, array_1):

        # 判断方差齐性
        sta_value, p_value_h = levene(array_0, array_1)
        # 判断正态性
        w_train, p_value_train_n = kstest(array_0, 'norm')
        w_test, p_value_test_n = kstest(array_1, 'norm')
        # 如果方差齐性并且都满足正态分布，做U检验
        if p_value_h >= 0.05 and p_value_train_n >= 0.05 and p_value_test_n >= 0.05:
            stat_num, p_value = ttest_ind(array_0, array_1)
            statistic_method = 'T-test'
        else:
            stat_num, p_value = mannwhitneyu(array_0, array_1)
            statistic_method = 'U'

        data_description = [str('%.2f' % np.mean(array_0)) + '±' + str('%.2f' % np.std(array_0)),
                            str('%.2f' % np.mean(array_1)) + '±' + str('%.2f' % np.std(array_1))]
        return data_description, statistic_method,  p_value

    @staticmethod
    def _statistic_discrete(train_data, test_data):
        def class_count(data_array):
            class_dict = {}
            class_counter = Counter(data_array)
            class_array = np.asarray(list(class_counter.keys()))
            count_array = np.asarray(list(class_counter.values()))
            for class_index in range(len(class_array)):
                class_dict[str(class_array[class_index])] = int(count_array[class_index])
            return class_dict
        # 生成列联表
        train_dict = class_count(train_data)
        test_dict = class_count(test_data)

        all_feature_class = [str(i) for i in sorted(list(set(list(train_dict.keys()) + list(test_dict.keys()))))]

        chi2_array = np.zeros((2, len(all_feature_class)))
        for sub_feature_index in range(len(all_feature_class)):
            try:
                chi2_array[0, sub_feature_index] = int(train_dict[all_feature_class[sub_feature_index]])
            except:
                chi2_array[0, sub_feature_index] = int(0)

            try:
                chi2_array[1, sub_feature_index] = int(test_dict[all_feature_class[sub_feature_index]])
            # 如果没有该特征就置零
            except:
                chi2_array[1, sub_feature_index] = int(0)
        print(chi2_array)
        # 计算占比
        percentage_array = copy.copy(chi2_array)
        percentage_array[0, :] = percentage_array[0, :] / np.sum(percentage_array, axis=1)[0]
        percentage_array[1, :] = percentage_array[1, :] / np.sum(percentage_array, axis=1)[1]

        print(percentage_array)
        data_description = pd.DataFrame(data=np.hstack((chi2_array, percentage_array)), index=['train', 'test'],
                                        columns=all_feature_class*2)
        print(data_description)
        # 列联表的自由度为1，所以需要chi2_contingency 的修正项，correction置为true
        a, p_value, b, c = chi2_contingency(chi2_array.T, correction=True)

        statistic_method = 'chi2_contingency'
        return data_description, statistic_method,  p_value

    def statistic_single_feature_sets(self, feature_name):

        # 判断数据是否连续
        train_set = set(self.train_pd[feature_name])
        test_set = set(self.test_pd[feature_name])

        if len(train_set) > 15 or feature_name in ['age', 'Age']:  # 数据连续型
            data_description, statistic_method,  p_value = self._statistic_continuous(self.train_pd[feature_name].get_values(),
                                                                                      self.test_pd[feature_name].get_values())

        else:  # 数据类型分立
            data_description, statistic_method, p_value = self._statistic_discrete(self.train_pd[feature_name],
                                                                                      self.test_pd[feature_name])

        return data_description, statistic_method, p_value

    def statistic_single_feature_labels(self, feature_name):

        data_pd = pd.concat([self.train_pd, self.test_pd], join='inner')

        label_0pd = data_pd.loc[data_pd['label'] == 0]
        label_1pd = data_pd.loc[data_pd['label'] == 1]
        # 判断数据是否连续

        label_0_set = set(label_0pd[feature_name])
        label_1_set = set(label_1pd[feature_name])

        if len(label_0_set) > 15 or feature_name in ['age', 'Age']:  # 数据连续型

            data_description, statistic_method,  p_value = self._statistic_continuous(label_0pd[feature_name].get_values(),
                                                                                      label_1pd[feature_name].get_values())

        else:  # 数据类型分立
            data_description, statistic_method, p_value = self._statistic_discrete(label_0pd[feature_name],
                                                                                      label_1pd[feature_name])

        return data_description, statistic_method, p_value

    def difference_between_cohorts(self, store_path):
        # 创建xls表格并写入列名

        xlsx_f = xlwt.Workbook()
        sheet0 = xlsx_f.add_sheet('feature_statistic', cell_overwrite_ok=True)
        sheet0.write(0, 0, 'feature')
        sheet0.write(0, 1, 'Train')
        sheet0.write(0, 2, 'Test')
        sheet0.write(0, 3, 'P-value')


        # 如果没有指定特征，就会计算所有的特征分布统计
        if len(self.selected_feature_list) == 0:
            self.selected_feature_list = self.feature_list
        col_index = 1
        for feature_index in range(len(self.selected_feature_list)):

            feature_name = self.selected_feature_list[feature_index]
            data_description, statistic_method, p_value = self.statistic_single_feature_sets(feature_name)

            if statistic_method == 'chi2_contingency':
                data_description_array = data_description.get_values()
                data_description_columns = data_description.columns.tolist()
                # data_description_columns前一半是class名称，后一半是class占比

                class_list = data_description_columns[: int(len(data_description_columns)/2)]

                sheet0.write_merge(col_index, col_index + len(class_list)*2-1, 0, 0, feature_name)
                sheet0.write_merge(col_index, col_index + len(class_list)*2-1, 3, 3, str('%.2f' % p_value))
                # 占两行

                # 写入统计数据
                for sub_class_index in range(len(class_list)):
                    # train columns
                    sheet0.write(col_index, 1, 'class ' + class_list[sub_class_index])
                    # test columns
                    sheet0.write(col_index, 2, 'class ' + class_list[sub_class_index])
                    col_index += 1

                    sheet0.write(col_index, 1, str(float(data_description_array[0, sub_class_index])) +
                                 '('+'%.2f%%' % (100*float(data_description_array[0, sub_class_index+len(class_list)]))
                                 + ')')

                    sheet0.write(col_index, 2,
                                 str(float(data_description_array[1, sub_class_index])) +
                                 '('+'%.2f%%' % (100*float(data_description_array[1, sub_class_index+len(class_list)]))
                                 + ')')
                    col_index += 1

            else:
                sheet0.write(col_index, 0, feature_name)
                sheet0.write(col_index, 1, data_description[0])
                sheet0.write(col_index, 2, data_description[1])
                sheet0.write(col_index, 3, str('%.2f' % p_value))
                # 占1行
                col_index += 1
        # 保存文件
        xlsx_f.save(os.path.join(store_path, 'statistics_sets.xls'))

    def difference_between_labels(self, store_path):
        # 创建xls表格并写入列名

        xlsx_f = xlwt.Workbook()
        sheet0 = xlsx_f.add_sheet('feature_statistic', cell_overwrite_ok=True)
        sheet0.write(0, 0, 'feature')
        sheet0.write(0, 1, 'Label 0')
        sheet0.write(0, 2, 'Label 1')
        sheet0.write(0, 3, 'P-value')

        # 如果没有指定特征，就会计算所有的特征分布统计
        if len(self.selected_feature_list) == 0:
            self.selected_feature_list = self.feature_list
        col_index = 1
        for feature_index in range(len(self.selected_feature_list)):

            feature_name = self.selected_feature_list[feature_index]
            data_description, statistic_method, p_value = self.statistic_single_feature_labels(feature_name)

            if statistic_method == 'chi2_contingency':
                data_description_array = data_description.get_values()
                data_description_columns = data_description.columns.tolist()
                # data_description_columns前一半是class名称，后一半是class占比

                class_list = data_description_columns[: int(len(data_description_columns)/2)]

                sheet0.write_merge(col_index, col_index + len(class_list)*2-1, 0, 0, feature_name)
                sheet0.write_merge(col_index, col_index + len(class_list)*2-1, 3, 3, str('%.2f' % p_value))
                # 占两行

                # 写入统计数据
                for sub_class_index in range(len(class_list)):
                    # train columns
                    sheet0.write(col_index, 1, 'class ' + class_list[sub_class_index])
                    # test columns
                    sheet0.write(col_index, 2, 'class ' + class_list[sub_class_index])
                    col_index += 1

                    sheet0.write(col_index, 1, str(float(data_description_array[0, sub_class_index])) +
                                 '('+'%.2f%%' % (100*float(data_description_array[0, sub_class_index+len(class_list)]))
                                 + ')')

                    sheet0.write(col_index, 2,
                                 str(float(data_description_array[1, sub_class_index])) +
                                 '('+'%.2f%%' % (100*float(data_description_array[1, sub_class_index+len(class_list)]))
                                 + ')')
                    col_index += 1

            else:
                sheet0.write(col_index, 0, feature_name)
                sheet0.write(col_index, 1, data_description[0])
                sheet0.write(col_index, 2, data_description[1])
                sheet0.write(col_index, 3, str('%.2f' % p_value))
                # 占1行
                col_index += 1
        # 保存文件
        xlsx_f.save(os.path.join(store_path, 'statistics_labels.xls'))


def IterationFromFeature(original_feature_path, store_path):
    from FAE.DataContainer.DataSeparate import DataSeparate
    from FAE.DataContainer.DataContainer import DataContainer

    def one_echo(original_feature_path, store_path):
        data = DataContainer()
        data.Load(original_feature_path)
        data_separator = DataSeparate()
        train_data_container, test_data_container = \
            data_separator.RunByTestingPercentage(data, testing_data_percentage=0.3)

        feature_statistic = FeatureStatistic(store_path)
        feature_statistic.load_train_test_feature()
        feature_statistic.difference_between_cohorts(store_path)

        statistics_result = pd.read_excel(os.path.join(store_path, 'statistic.xls'))
        print(statistics_result['P-value'])

    one_echo(original_feature_path, store_path)


def load_statistics(original_feature_path, statistics_path):
    original_feature_pd = pd.read_csv(original_feature_path, index_col=0)
    statistics_pd = pd.read_excel(statistics_path)
    sign_feature = statistics_pd.loc[statistics_pd['P-value'] < 0.05]
    sign_feature_list = list(sign_feature['feature'])
    sign_feature_list.insert(0, 'label')
    sign_feature_pd = original_feature_pd[sign_feature_list]
    sign_feature_pd.to_csv(os.path.join(os.path.split(statistics_path)[0], 'selected_feature.csv'))



if __name__ == '__main__':
    # feature_path = r'D:\hospital\CancerHospital\extract_date\0.65\clinical\data_set'
    # store_path = r'D:\hospital\CancerHospital\extract_date\0.65\clinical\data_set'
    # quality_list = ['Quality_age', 'Quality_gender', 'Quality_invoved']

    feature_path = r'D:\hospital\EENT\New_test\statistics'
    store_path = r'D:\hospital\EENT\New_test\statistics'
    # feature_statistic = FeatureStatistic(feature_path)
    # feature_statistic.load_train_test_feature()
    # feature_statistic.difference_between_labels(store_path)
    # IterationFromFeature(feature_path, store_path)
    load_statistics(r'D:\hospital\EENT\New_test\statistics\numeric_feature.csv',
        r'D:\hospital\EENT\New_test\statistics\statistics_labels.xls')
