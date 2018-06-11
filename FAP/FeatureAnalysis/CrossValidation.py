from abc import ABCMeta,abstractmethod
import numpy as np
import os
import numbers
import csv

from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

from FAP.DataContainer.DataContainer import DataContainer
from FAP.FeatureAnalysis.Classifier import Classifier
from FAP.FeatureAnalysis.FeatureSelector import FeatureSelectPipeline, FeatureSelectByAnalysis, FeatureSelector
from FAP.Func.Metric import EstimateMetirc
from FAP.Visualization.PlotMetricVsFeatureNumber import DrawCurve
from FAP.Func.Visualization import LoadWaitBar

class CrossValidation:
    def __init__(self, cv_method):
        self.__classifier = Classifier()

        if cv_method == 'LOO':
            self.__cv = LeaveOneOut
        elif cv_method == '10-folder':
            self.__cv = StratifiedKFold(10)
        elif cv_method == '5-folder':
            self.__cv = StratifiedKFold(5)
        else:
            self.__cv = None

    def SetClassifier(self, classifier):
        self.__classifier = classifier

    def GetClassifier(self):
        return self.__classifier

    def SetCV(self, cv):
        self.__cv = cv

    def GetCV(self):
        return self.__cv

    def SaveCVInfo(self, info, store_path):
        info = dict(sorted(info.items(), key= lambda item: item[0]))

        write_info = []
        for key in info.keys():
            temp_list = []
            temp_list.append(key)
            if isinstance(info[key], (numbers.Number, str)):
                temp_list.append(info[key])
            else:
                temp_list.extend(info[key])
            write_info.append(temp_list)

        # write_info = [[key].extend(info[key]) for key in info.keys()]
        if os.path.isdir(store_path):
            store_path = os.path.join(store_path, 'cv_info.csv')

        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(write_info)

    def Run(self, data_container, test_data_container=DataContainer(), store_folder=''):
        train_pred_list, train_label_list, val_pred_list, val_label_list = [], [], [], []

        data = data_container.GetArray()
        label = data_container.GetLabel()

        for train_index, val_index in self.__cv.split(data, label):
            train_data = data[train_index, :]
            train_label = label[train_index]
            val_data = data[val_index, :]
            val_label = label[val_index]

            self.__classifier.SetData(train_data, train_label)
            self.__classifier.Fit()

            train_prob = self.__classifier.Predict(train_data)
            val_prob = self.__classifier.Predict(val_data)

            train_pred_list.extend(train_prob)
            train_label_list.extend(train_label)
            val_pred_list.extend(val_prob)
            val_label_list.extend(val_label)

        total_train_label = np.asarray(train_label_list, dtype=np.uint8)
        total_train_pred = np.asarray(train_pred_list, dtype=np.float32)
        train_metric = EstimateMetirc(total_train_pred, total_train_label, 'train')

        total_label = np.asarray(val_label_list, dtype=np.uint8)
        total_pred = np.asarray(val_pred_list, dtype=np.float32)
        val_metric = EstimateMetirc(total_pred, total_label, 'val')

        self.__classifier.SetData(data, label)
        self.__classifier.Fit()

        test_metric = {}
        if test_data_container.GetArray().size > 0:
            selected_feature_name = data_container.GetFeatureName()
            fs = FeatureSelector()
            test_data_container = fs.SelectFeatureByName(test_data_container, selected_feature_name)

            test_data = test_data_container.GetArray()
            test_label = test_data_container.GetLabel()
            test_pred = self.__classifier.Predict(test_data)

            test_metric = EstimateMetirc(test_pred, test_label, 'test')

        if store_folder and os.path.isdir(store_folder):
            info = {}
            info.update(train_metric)
            info.update(val_metric)

            np.save(os.path.join(store_folder, 'train_predict.npy'), total_train_pred)
            np.save(os.path.join(store_folder, 'val_predict.npy'), total_pred)
            np.save(os.path.join(store_folder, 'train_label.npy'), total_train_label)
            np.save(os.path.join(store_folder, 'val_label.npy'), total_label)

            if test_data_container.GetArray().size > 0:
                info.update(test_metric)
                np.save(os.path.join(store_folder, 'test_predict.npy'), test_pred)
                np.save(os.path.join(store_folder, 'test_label.npy'), test_label)

            self.__classifier.Save(store_folder)

            self.SaveCVInfo(info, store_folder)

        return train_metric, val_metric, test_metric

class CrossValidationOnFeatureNumber(CrossValidation):
    def __init__(self, cv_method, max_feature_number=1):
        super(CrossValidationOnFeatureNumber, self).__init__(cv_method)
        self.__max_feature_number = max_feature_number
        self.__feature_selector = FeatureSelectByAnalysis()

    def SetMaxFeatureNumber(self, max_feature_number):
        self.__max_feature_number = max_feature_number
    def GetMaxFeatureNumber(self):
        return self.__max_feature_number

    def SetFeatureSelector(self, feature_selector):
        self.__feature_selector = feature_selector
    def GetFeatureSelector(self):
        return self.__feature_selector

    def Run(self, data_container, test_data_container=DataContainer(), store_folder='', metric_name_list=('auc', 'accuracy')):
        train_metric_list = []
        val_metric_list = []
        test_metric_list = []

        for feature_number in range(1, self.__max_feature_number + 1):
            LoadWaitBar(self.__max_feature_number, feature_number)

            feature_store_folder = os.path.join(store_folder, 'feature_'+str(feature_number))
            if not os.path.exists(feature_store_folder):
                os.mkdir(feature_store_folder)

            self.__feature_selector.SetSelectedFeatureNumber(feature_number)
            feature_selected_data_container = self.__feature_selector.Run(data_container, feature_store_folder)
            # feature_selected_data_container.UsualAndL2Normalize()

            train_metric, val_metric, test_metric = super(CrossValidationOnFeatureNumber, self).Run(
                feature_selected_data_container, test_data_container=test_data_container, store_folder=feature_store_folder)

            train_metric_list.append(train_metric)
            val_metric_list.append(val_metric)
            test_metric_list.append(test_metric)

        metric_list = []
        for metric in metric_name_list:
            metric_ditc = {'train': [], 'val': [], 'test': [], 'name': metric}
            for feature_number in range(self.__max_feature_number):
                metric_ditc['train'].append(train_metric_list[feature_number]['train_' + metric])
                metric_ditc['val'].append(val_metric_list[feature_number]['val_' + metric])
                if test_metric_list[0] != {}:
                    metric_ditc['test'].append(test_metric_list[feature_number]['test_' + metric])
            metric_list.append(metric_ditc)

        # Save the Relationship v.s. number of features
        if store_folder and os.path.isdir(store_folder):
            for metric_dict in metric_list:
                if test_metric_list[0] != {}:
                    DrawCurve(range(1, self.__max_feature_number + 1), [metric_dict['train'], metric_dict['val'], metric_dict['test']],
                              xlabel='# Features', ylabel=metric_dict['name'],
                              name_list=['train', 'validation', 'test'], is_show=False,
                              store_path=os.path.join(store_folder, metric_dict['name'] + '_FeatureNum.jpg'))
                else:
                    DrawCurve(range(1, self.__max_feature_number + 1), [metric_dict['train'], metric_dict['val']],
                              xlabel='# Features', ylabel=metric_dict['name'],
                              name_list=['train', 'validation'], is_show=False,
                              store_path=os.path.join(store_folder, metric_dict['name'] + '_FeatureNum.jpg'))

        return_list = []
        test_return_list = []
        for metric_dict in metric_list:
            metric_info = {}
            new_info = val_metric_list[np.argmax(metric_dict['val'])]
            metric_info['feature_number'] = np.argmax(metric_dict['val']) + 1
            for key in new_info.keys():
                metric_info[key[4:]] = new_info[key]
            return_list.append(dict(sorted(metric_info.items(), key=lambda item:item[0])))

            if test_metric_list[0] != {}:
                test_metric_info = {}
                test_new_info = test_metric_list[np.argmax(metric_dict['test'])]
                test_metric_info['feature_number'] = np.argmax(metric_dict['test']) + 1
                for key in test_new_info.keys():
                    test_metric_info[key[5:]] = test_new_info[key]
                test_return_list.append(dict(sorted(test_metric_info.items(), key=lambda item:item[0])))

        return return_list, test_return_list




