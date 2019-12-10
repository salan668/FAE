import os

import numpy as np
import pandas as pd
from scipy.stats import levene, ttest_ind, kstest, mannwhitneyu, chi2_contingency
from collections import Counter
import xlrd
import xlwt


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
        # 生成列联表
        train_class_0, train_class_1 = Counter(train_data).most_common()
        test_class_0, test_class_1 = Counter(test_data).most_common()

        chi2_array = np.array([[np.array(train_class_0[-1]), np.array(train_class_1[-1])],
                               [np.array(test_class_0[-1]), np.array(test_class_1[-1])]])

        data_description = pd.DataFrame(data=chi2_array, index=['train', 'test'],
                                        columns=[train_class_0[0], train_class_1[0]])

        # 列联表的自由度为1，所以需要chi2_contingency 的修正项，correction置为true
        a, p_value, b, c = chi2_contingency(chi2_array, correction=True)

        statistic_method = 'chi2_contingency'
        return data_description, statistic_method,  p_value

    def statistic_single_feature(self, feature_name):

        # 判断数据是否连续
        train_set = set(self.train_pd[feature_name])
        test_set = set(self.test_pd[feature_name])

        if len(train_set) > 2 and len(test_set) > 2:  # 数据连续型
            data_description, statistic_method,  p_value = self._statistic_continuous(self.train_pd[feature_name],
                                                                                      self.test_pd[feature_name])

        else:  # 数据类型分立
            data_description, statistic_method, p_value = self._statistic_discrete(self.train_pd[feature_name],
                                                                                      self.test_pd[feature_name])

        return data_description, statistic_method, p_value

    def difference_between_cohorts(self, store_path):
        # 创建xls表格并写入列名

        xlsx_f = xlwt.Workbook()
        sheet0 = xlsx_f.add_sheet('feature_statistic', cell_overwrite_ok=True)
        sheet0.write(0, 0, 'feature')
        sheet0.write_merge(0, 0, 1, 2, 'Train')
        sheet0.write_merge(0, 0, 3, 4, 'Test')
        sheet0.write(0, 5, 'P-value')

        col_index = 0
        # 如果没有指定特征，就会计算所有的特征分布统计
        if len(self.selected_feature_list) == 0:
            self.selected_feature_list = self.feature_list

        for feature_index in range(len(self.selected_feature_list)):

            feature_name = self.selected_feature_list[feature_index]
            data_description, statistic_method, p_value = self.statistic_single_feature(feature_name)
            if statistic_method == 'chi2_contingency':
                # 写入统计数据
                sheet0.write(col_index + 1, 1, 'class ' + str(data_description.columns.tolist()[0]))
                sheet0.write(col_index + 2, 1, float(data_description.get_values()[0, 0]))

                sheet0.write(col_index + 1, 2, 'class ' + str(data_description.columns.tolist()[1]))
                sheet0.write(col_index + 2, 2, float(data_description.get_values()[0, 1]))

                sheet0.write(col_index + 1, 3, 'class ' + str(data_description.columns.tolist()[0]))
                sheet0.write(col_index + 2, 3, float(data_description.get_values()[1, 0]))

                sheet0.write(col_index + 1, 4, 'class ' + str(data_description.columns.tolist()[1]))
                sheet0.write(col_index + 2, 4, float(data_description.get_values()[1, 1]))

                sheet0.write_merge(col_index + 1, col_index + 2, 0, 0, feature_name)
                sheet0.write_merge(col_index + 1, col_index + 2, 5, 5, str('%.2f' % p_value))
                # 占两行
                col_index += 2

            else:
                sheet0.write_merge(col_index + 1, col_index + 1, 1, 2, data_description[0])
                sheet0.write_merge(col_index + 1, col_index + 1, 3, 4, data_description[1])
                sheet0.write(col_index + 1, 0, feature_name)
                sheet0.write(col_index + 1, 5, str('%.2f' % p_value))
                # 占1行
                col_index += 1
        # 保存文件
        xlsx_f.save(os.path.join(store_path, 'statistic.xls'))


if __name__ == '__main__':
    def test():
        feature_path = r'D:\hospital\CancerHospital\extract_date\0.65\data_set'
        store_path = r'D:\hospital\CancerHospital\extract_date\0.65\data_set'
        quality_list = ['Quality_age', 'Quality_gender', 'Quality_invoved']
        feature_statistic = FeatureStatistic(feature_path)
        feature_statistic.load_train_test_feature()
        feature_statistic.difference_between_cohorts(store_path)
    test()
