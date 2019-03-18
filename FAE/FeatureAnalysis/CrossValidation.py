from abc import ABCMeta,abstractmethod
import numpy as np
import os
import numbers
import csv
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.Classifier import Classifier
from FAE.Func.Metric import EstimateMetirc
from FAE.HyperParameterConfig.HyperParamManager import HyperParameterManager

class CrossValidation:
    '''
    CrossValidation is the base class to explore the hpyer-parameters. Now it supported Leave-one-lout (LOO), 10-folder,
    and 5-folders. A classifier must be set before run CV. A training metric and validation metric will be returned.
    If a testing data container was also set, the test metric will be return.
    '''
    def __init__(self):
        self._raw_classifier = Classifier()
        self.__classifier = Classifier()
        self._hyper_parameter_manager = HyperParameterManager()
        self.__classifier_parameter_list = [{}]

    def SetDefaultClassifier(self):
        self.__classifier = deepcopy(self._raw_classifier)

    def SetClassifier(self, classifier):
        self.__init__()
        self._raw_classifier = deepcopy(classifier)
        self.__classifier = classifier

    def GetClassifier(self):
        return self.__classifier
    classifier = property(GetClassifier, SetClassifier)

    def AutoLoadClassifierParameterList(self, relative_path=r'HyperParameters\Classifier'):
        self._hyper_parameter_manager.LoadSpecificConfig(self.classifier.GetName(), relative_path=relative_path)
        self.__classifier_parameter_list = self._hyper_parameter_manager.GetParameterSetting()

    def SetClassifierParameterList(self, classifier_parameter_list):
        self.__classifier_parameter_list = deepcopy(classifier_parameter_list)

    def GetClassifierParameterList(self):
        return self.__classifier_parameter_list
    classifier_parameter_list = property(GetClassifierParameterList, SetClassifierParameterList)

    def _GetNameOfParamDict(self, param_dict):
        name = ''
        for key, item in param_dict.items():
            name += str(key) + '_' + str(item) + '-'
        return name[:len(name) - 1]

    def SaveResult(self, info, store_path):
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

        write_info.sort()

        # write_info = [[key].extend(info[key]) for key in info.keys()]
        if os.path.isdir(store_path):
            store_path = os.path.join(store_path, 'result.csv')

        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            write_info.sort()
            writer.writerows(write_info)

class CrossValidationLeaveOneOut(CrossValidation):
    def __init__(self):
        super(CrossValidationLeaveOneOut, self).__init__()
        self.__cv = LeaveOneOut()

    def GetCV(self):
        return self.__cv

    def GetName(self):
        return 'LeaveOneOut'

    def GetDescription(self, is_test_data_container=False):
        if is_test_data_container:
            text = "To determine the hyper-parameter (e.g. the number of features) of model, we applied cross validation " \
                   "with leave-one-out on the training data set. The hyper-parameters were set according to the model performance " \
                   "on the validation data set. "
        else:
            text = "To prove the performance of the model, we applied corss validation with leave-one-out on the data set. "

        return text

    def Run(self, data_container, test_data_container=DataContainer(), store_folder=''):
        train_pred_list, train_label_list, val_pred_list, val_label_list = [], [], [], []

        data = data_container.GetArray()
        label = data_container.GetLabel()
        case_name = data_container.GetCaseName()

        train_cv_info = [['CaseName', 'Pred', 'Label']]
        val_cv_info = [['CaseName', 'Pred', 'Label']]

        for train_index, val_index in self.__cv.split(data, label):
            train_data = data[train_index, :]
            train_label = label[train_index]
            val_data = data[val_index, :]
            val_label = label[val_index]

            self.classifier.SetData(train_data, train_label)
            self.classifier.Fit()

            train_prob = self.classifier.Predict(train_data)
            val_prob = self.classifier.Predict(val_data)

            for index in range(len(train_index)):
                train_cv_info.append(
                    [case_name[train_index[index]], train_prob[index], train_label[index]])
            for index in range(len(val_index)):
                val_cv_info.append([case_name[val_index[index]], val_prob[index], val_label[index]])

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

        self.classifier.SetDataContainer(data_container)
        self.classifier.Fit()

        test_metric = {}
        if test_data_container.GetArray().size > 0:
            test_data = test_data_container.GetArray()
            test_label = test_data_container.GetLabel()
            test_case_name = test_data_container.GetCaseName()
            test_pred = self.classifier.Predict(test_data)

            test_metric = EstimateMetirc(test_pred, test_label, 'test')

        if store_folder:
            if not os.path.exists(store_folder):
                os.mkdir(store_folder)

            info = {}
            info.update(train_metric)
            info.update(val_metric)

            np.save(os.path.join(store_folder, 'train_predict.npy'), total_train_pred)
            np.save(os.path.join(store_folder, 'val_predict.npy'), total_pred)
            np.save(os.path.join(store_folder, 'train_label.npy'), total_train_label)
            np.save(os.path.join(store_folder, 'val_label.npy'), total_label)

            with open(os.path.join(store_folder, 'train_cvloo_info.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(train_cv_info)
            with open(os.path.join(store_folder, 'val_cvloo_info.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(val_cv_info)

            if test_data_container.GetArray().size > 0:
                info.update(test_metric)
                np.save(os.path.join(store_folder, 'test_predict.npy'), test_pred)
                np.save(os.path.join(store_folder, 'test_label.npy'), test_label)

                test_result_info = [['CaseName', 'Pred', 'Label']]
                for index in range(len(test_label)):
                    test_result_info.append([test_case_name[index], test_pred[index], test_label[index]])
                with open(os.path.join(store_folder, 'test_info.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(test_result_info)

            self.classifier.Save(store_folder)
            self.SaveResult(info, store_folder)

        return train_metric, val_metric, test_metric

class CrossValidation5Folder(CrossValidation):
    def __init__(self):
        super(CrossValidation5Folder, self).__init__()
        self.__cv = StratifiedKFold(5)

    def GetCV(self):
        return self.__cv

    def GetName(self):
        return '5-Folder'

    def GetDescription(self, is_test_data_container=False):
        if is_test_data_container:
            text = "To determine the hyper-parameter (e.g. the number of features) of model, we applied cross validation " \
                   "with 5-folder on the training data set. The hyper-parameters were set according to the model performance " \
                   "on the validation data set. "
        else:
            text = "To prove the performance of the model, we applied corss validation with 5-folder on the data set. "

        return text

    def Run(self, data_container, test_data_container=DataContainer(), store_folder='', is_hyper_parameter=False):
        train_pred_list, train_label_list, val_pred_list, val_label_list = [], [], [], []

        data = data_container.GetArray()
        label = data_container.GetLabel()
        case_name = data_container.GetCaseName()

        param_metric_train_auc = []
        param_metric_val_auc = []
        param_all = []

        if len(self.classifier_parameter_list) == 1 and is_hyper_parameter:
            self.AutoLoadClassifierParameterList(relative_path=r'HyperParameters\Classifier')

        for parameter in self.classifier_parameter_list:
            self.SetDefaultClassifier()
            self.classifier.SetModelParameter(parameter)

            train_cv_info = [['CaseName', 'Group', 'Pred', 'Label']]
            val_cv_info = [['CaseName', 'Group', 'Pred', 'Label']]
            group_index = 0

            for train_index, val_index in self.__cv.split(data, label):
                group_index += 1

                train_data = data[train_index, :]
                train_label = label[train_index]
                val_data = data[val_index, :]
                val_label = label[val_index]

                self.classifier.SetData(train_data, train_label)
                self.classifier.Fit()

                train_prob = self.classifier.Predict(train_data)
                val_prob = self.classifier.Predict(val_data)

                for index in range(len(train_index)):
                    train_cv_info.append([case_name[train_index[index]], str(group_index), train_prob[index], train_label[index]])
                for index in range(len(val_index)):
                    val_cv_info.append([case_name[val_index[index]], str(group_index), val_prob[index], val_label[index]])

                train_pred_list.extend(train_prob)
                train_label_list.extend(train_label)
                val_pred_list.extend(val_prob)
                val_label_list.extend(val_label)

            total_train_label = np.asarray(train_label_list, dtype=np.uint8)
            total_train_pred = np.asarray(train_pred_list, dtype=np.float32)
            train_cv_metric = EstimateMetirc(total_train_pred, total_train_label, 'train')

            total_val_label = np.asarray(val_label_list, dtype=np.uint8)
            total_val_pred = np.asarray(val_pred_list, dtype=np.float32)
            val_cv_metric = EstimateMetirc(total_val_pred, total_val_label, 'val')

            param_metric_train_auc.append(float(train_cv_metric['train_auc']))
            param_metric_val_auc.append(float(val_cv_metric['val_auc']))
            param_all.append({'total_train_label': total_train_label,
                              'total_train_pred': total_train_pred,
                              'train_metric': train_cv_metric,
                              'train_cv_info': deepcopy(train_cv_info),
                              'total_val_label': total_val_label,
                              'total_val_pred': total_val_pred,
                              'val_metric': val_cv_metric,
                              'val_cv_info': deepcopy(val_cv_info)
                              })

        # find the best parameter
        index = np.argmax(param_metric_val_auc)
        total_train_label = param_all[index]['total_train_label']
        total_train_pred = param_all[index]['total_train_pred']
        train_cv_metric = param_all[index]['train_metric']
        train_cv_info = param_all[index]['train_cv_info']
        total_val_label = param_all[index]['total_val_label']
        total_val_pred = param_all[index]['total_val_pred']
        val_cv_metric = param_all[index]['val_metric']
        val_cv_info = param_all[index]['val_cv_info']

        self.SetDefaultClassifier()
        self.classifier.SetModelParameter(self.classifier_parameter_list[index])
        self.classifier.SetDataContainer(data_container)
        self.classifier.Fit()

        all_train_pred = self.classifier.Predict(data_container.GetArray())
        all_train_label = data_container.GetLabel()
        all_train_metric = EstimateMetirc(all_train_pred, all_train_label, 'all_train')

        test_metric = {}
        if test_data_container.GetArray().size > 0:
            test_data = test_data_container.GetArray()
            test_label = test_data_container.GetLabel()
            test_case_name = test_data_container.GetCaseName()
            test_pred = self.classifier.Predict(test_data)

            test_metric = EstimateMetirc(test_pred, test_label, 'test')

        if store_folder:
            if not os.path.exists(store_folder):
                os.mkdir(store_folder)

            # Save the Parameter:
            if self.classifier_parameter_list[0] != {}:
                with open(os.path.join(store_folder, 'Classifier_Param_Result.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Param', 'Train AUC', 'Val AUC'])
                    for param, param_index in zip(self.classifier_parameter_list, range(len(self.classifier_parameter_list))):
                        writer.writerow([self._GetNameOfParamDict(param),
                                         param_metric_train_auc[param_index],
                                         param_metric_val_auc[param_index]])

            info = {}
            info.update(train_cv_metric)
            info.update(val_cv_metric)
            info.update(all_train_metric)

            np.save(os.path.join(store_folder, 'train_predict.npy'), total_train_pred)
            np.save(os.path.join(store_folder, 'train_label.npy'), total_train_label)
            np.save(os.path.join(store_folder, 'val_predict.npy'), total_val_pred)
            np.save(os.path.join(store_folder, 'val_label.npy'), total_val_label)
            np.save(os.path.join(store_folder, 'all_train_predict.npy'), all_train_pred)
            np.save(os.path.join(store_folder, 'all_train_label.npy'), all_train_label)

            with open(os.path.join(store_folder, 'train_cv5_info.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(train_cv_info)
            with open(os.path.join(store_folder, 'val_cv5_info.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(val_cv_info)

            if test_data_container.GetArray().size > 0:
                info.update(test_metric)
                np.save(os.path.join(store_folder, 'test_predict.npy'), test_pred)
                np.save(os.path.join(store_folder, 'test_label.npy'), test_label)

                test_result_info = [['CaseName', 'Pred', 'Label']]
                for index in range(len(test_label)):
                    test_result_info.append([test_case_name[index], test_pred[index], test_label[index]])
                with open(os.path.join(store_folder, 'test_info.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(test_result_info)

            self.classifier.Save(store_folder)
            self.SaveResult(info, store_folder)

        return train_cv_metric, val_cv_metric, test_metric, all_train_metric

class CrossValidation10Folder(CrossValidation):
    def __init__(self):
        super(CrossValidation10Folder, self).__init__()
        self.__cv = StratifiedKFold(10)

    def GetCV(self):
        return self.__cv

    def GetName(self):
        return '10-Folder'

    def GetDescription(self, is_test_data_container=False):
        if is_test_data_container:
            text = "To determine the hyper-parameter (e.g. the number of features) of model, we applied cross validation " \
                   "with 10-folder on the training data set. The hyper-parameters were set according to the model performance " \
                   "on the validation data set. "
        else:
            text = "To prove the performance of the model, we applied corss validation with 10-folder on the data set. "

        return text

    def Run(self, data_container, test_data_container=DataContainer(), store_folder='', relative_config_path=r'FAE\HyperParameterConfig'):
        train_pred_list, train_label_list, val_pred_list, val_label_list = [], [], [], []

        data = data_container.GetArray()
        label = data_container.GetLabel()
        case_name = data_container.GetCaseName()
        group_index = 0

        train_cv_info = [['CaseName', 'Group', 'Pred', 'Label']]
        val_cv_info = [['CaseName', 'Group', 'Pred', 'Label']]

        param_metric_train_auc = []
        param_metric_val_auc = []
        param_all = []

        if len(self.classifier_parameter_list) == 1:
            self.AutoLoadClassifierParameterList(relative_path=relative_config_path)

        for parameter in self.classifier_parameter_list:
            self.SetDefaultClassifier()
            self.classifier.SetModelParameter(parameter)

            for train_index, val_index in self.__cv.split(data, label):
                group_index += 1

                train_data = data[train_index, :]
                train_label = label[train_index]
                val_data = data[val_index, :]
                val_label = label[val_index]

                self.classifier.SetData(train_data, train_label)
                self.classifier.Fit()

                train_prob = self.classifier.Predict(train_data)
                val_prob = self.classifier.Predict(val_data)

                for index in range(len(train_index)):
                    train_cv_info.append([case_name[train_index[index]], str(group_index), train_prob[index], train_label[index]])
                for index in range(len(val_index)):
                    val_cv_info.append([case_name[val_index[index]], str(group_index), val_prob[index], val_label[index]])

                train_pred_list.extend(train_prob)
                train_label_list.extend(train_label)
                val_pred_list.extend(val_prob)
                val_label_list.extend(val_label)

            total_train_label = np.asarray(train_label_list, dtype=np.uint8)
            total_train_pred = np.asarray(train_pred_list, dtype=np.float32)
            train_cv_metric = EstimateMetirc(total_train_pred, total_train_label, 'train')

            total_val_label = np.asarray(val_label_list, dtype=np.uint8)
            total_val_pred = np.asarray(val_pred_list, dtype=np.float32)
            val_cv_metric = EstimateMetirc(total_val_pred, total_val_label, 'val')

            param_metric_train_auc.append(float(train_cv_metric['train_auc']))
            param_metric_val_auc.append(float(val_cv_metric['val_auc']))
            param_all.append({'total_train_label': total_train_label,
                              'total_train_pred': total_train_pred,
                              'train_metric': train_cv_metric,
                              'train_cv_info': train_cv_info,
                              'total_val_label': total_val_label,
                              'total_val_pred': total_val_pred,
                              'val_metric': val_cv_metric,
                              'val_cv_info': val_cv_info
                              })

        # find the best parameter
        index = np.argmax(param_metric_val_auc)
        total_train_label = param_all[index]['total_train_label']
        total_train_pred = param_all[index]['total_train_pred']
        train_cv_metric = param_all[index]['train_metric']
        train_cv_info = param_all[index]['train_cv_info']
        total_val_label = param_all[index]['total_val_label']
        total_val_pred = param_all[index]['total_val_pred']
        val_cv_metric = param_all[index]['val_metric']
        val_cv_info = param_all[index]['val_cv_info']

        self.SetDefaultClassifier()
        self.classifier.SetModelParameter(self.classifier_parameter_list[index])
        self.classifier.SetDataContainer(data_container)
        self.classifier.Fit()

        all_train_pred = self.classifier.Predict(data_container.GetArray())
        all_train_label = data_container.GetLabel()
        all_train_metric = EstimateMetirc(all_train_pred, all_train_label, 'all_train')

        test_metric = {}
        if test_data_container.GetArray().size > 0:
            test_data = test_data_container.GetArray()
            test_label = test_data_container.GetLabel()
            test_case_name = test_data_container.GetCaseName()
            test_pred = self.classifier.Predict(test_data)

            test_metric = EstimateMetirc(test_pred, test_label, 'test')

        if store_folder:
            if not os.path.exists(store_folder):
                os.mkdir(store_folder)

            # Save the Parameter:
            if self.classifier_parameter_list[0] != {}:
                with open(os.path.join(store_folder, 'Classifier_Param_Result.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Param', 'Train AUC', 'Val AUC'])
                    for param, param_index in zip(self.classifier_parameter_list, range(len(self.classifier_parameter_list))):
                        writer.writerow([self._GetNameOfParamDict(param),
                                         param_metric_train_auc[param_index],
                                         param_metric_val_auc[param_index]])

            info = {}
            info.update(train_cv_metric)
            info.update(val_cv_metric)
            info.update(all_train_metric)

            np.save(os.path.join(store_folder, 'train_predict.npy'), total_train_pred)
            np.save(os.path.join(store_folder, 'train_label.npy'), total_train_label)
            np.save(os.path.join(store_folder, 'val_predict.npy'), total_val_pred)
            np.save(os.path.join(store_folder, 'val_label.npy'), total_val_label)
            np.save(os.path.join(store_folder, 'all_train_predict.npy'), all_train_pred)
            np.save(os.path.join(store_folder, 'all_train_label.npy'), all_train_label)

            with open(os.path.join(store_folder, 'train_cv5_info.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(train_cv_info)
            with open(os.path.join(store_folder, 'val_cv5_info.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(val_cv_info)

            if test_data_container.GetArray().size > 0:
                info.update(test_metric)
                np.save(os.path.join(store_folder, 'test_predict.npy'), test_pred)
                np.save(os.path.join(store_folder, 'test_label.npy'), test_label)

                test_result_info = [['CaseName', 'Pred', 'Label']]
                for index in range(len(test_label)):
                    test_result_info.append([test_case_name[index], test_pred[index], test_label[index]])
                with open(os.path.join(store_folder, 'test_info.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(test_result_info)

            self.classifier.Save(store_folder)
            self.SaveResult(info, store_folder)

        return train_cv_metric, val_cv_metric, test_metric, all_train_metric

if __name__ == '__main__':
    from FAE.DataContainer.DataContainer import DataContainer
    from FAE.FeatureAnalysis.Normalizer import NormalizerZeroCenter
    from FAE.FeatureAnalysis.Classifier import SVM, LR, LDA, LRLasso, GaussianProcess, NaiveBayes, DecisionTree, RandomForest, AE, AdaBoost
    import numpy as np

    train_data_container = DataContainer()
    train_data_container.Load(r'C:\MyCode\FAEGitHub\FAE\Example\withoutshape\non_balance_features.csv')

    normalizer = NormalizerZeroCenter()
    train_data_container = normalizer.Run(train_data_container)

    data = train_data_container.GetArray()
    label = np.asarray(train_data_container.GetLabel())

#     param_list = [
# {"hidden_layer_sizes": [(30,), (100,)],
# "solver": ["adam"],
# "alpha": [0.0001, 0.001],
# "learning_rate_init": [0.001, 0.01]}
# ]
#     from sklearn.model_selection import ParameterGrid
#     pl = ParameterGrid(param_list)

    cv = CrossValidation5Folder()
    cv.SetClassifier(LR())
    # cv.SetClassifierParameterList(pl)
    train_metric, val_metric, test_metric = cv.Run(train_data_container, store_folder=r'C:\Users\SY\Desktop\temp_fae',
                                                   relative_config_path=r'..\..\HyperParameters\Classifier')
    print(train_metric)
    print(val_metric)

    cv.SetClassifier(LRLasso())
    # cv.SetClassifierParameterList(pl)
    train_metric, val_metric, test_metric = cv.Run(train_data_container, store_folder=r'C:\Users\SY\Desktop\temp_fae',
                                                   relative_config_path=r'..\..\HyperParameters\Classifier')
    print(train_metric)
    print(val_metric)