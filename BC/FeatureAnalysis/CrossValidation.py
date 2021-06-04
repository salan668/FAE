"""
All rights reserved.
--Yang Song, Apr 7th, 2020
"""
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from BC.DataContainer.DataContainer import DataContainer


class BaseCrossValidation(object):
    def __init__(self, n_split='', description=''):
        self._description = description
        if n_split == 'all':
            self._cv = LeaveOneOut()
            self._name = 'LeaveOneOut'
        else:
            self._cv = StratifiedKFold(int(n_split), shuffle=False)
            self._name = '{}-Fold'.format(int(n_split))
        pass

    def GetName(self):
        return self._name

    def Generate(self, data_container):
        array, label = data_container.GetArray(), data_container.GetLabel()
        feature_name, case_name = data_container.GetFeatureName(), data_container.GetCaseName()
        for train_index, val_index in self._cv.split(array, label):
            train_array, train_label = array[train_index, :], label[train_index]
            val_array, val_label = array[val_index, :], label[val_index]

            sub_train_container = DataContainer(array=train_array, label=train_label, feature_name=feature_name,
                                                case_name=[case_name[index] for index in train_index])
            sub_val_container = DataContainer(array=val_array, label=val_label, feature_name=feature_name,
                                              case_name=[case_name[index] for index in val_index])
            yield (sub_train_container, sub_val_container)

    def GetDescription(self):
        return self._description


CrossValidation5Fold = BaseCrossValidation(n_split='5',
                                           description="To determine the hyper-parameter (e.g. the number of "
                                                       "features) of model, we applied cross validation with 5-fold "
                                                       "on the training data set. The hyper-parameters were set "
                                                       "according to the model performance on the validation data set. "
                                           )
CrossValidation10Fold = BaseCrossValidation(n_split='10',
                                            description="To determine the hyper-parameter (e.g. the number of "
                                                        "features) of model, we applied cross validation with 10-fold "
                                                        "on the training data set. The hyper-parameters were set "
                                                        "according to the model performance on the validation data set. "
                                            )
CrossValidationLOO = BaseCrossValidation(n_split='all',
                                         description="To determine the hyper-parameter (e.g. the number of features) "
                                                     "of model, we applied cross validation with leave-one-out on the "
                                                     "training data set. The hyper-parameters were set according to "
                                                     "the model performance on the validation data set. ")

if __name__ == '__main__':
    import numpy as np

    data = np.random.random((100, 10))
    label = np.concatenate((np.ones((60,)), np.zeros((40,))), axis=0)

    cv = LeaveOneOut()
    for train, val in cv.split(data, label):
        print(train)
        print(val)
        print('')
