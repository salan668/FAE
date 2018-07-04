import numpy as np
import pickle
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from abc import ABCMeta,abstractmethod
from FAE.DataContainer.DataContainer import DataContainer

class Classifier:
    '''
    This is the base class of the classifer. All the specific classifier need to be artributed from this base class.
    '''
    def __init__(self):
        self.__model = None
        self._x = np.array([])
        self._y = np.array([])
        self._data_container = DataContainer()

    def SetDataContainer(self, data_container):
        data = data_container.GetArray()
        label = data_container.GetLabel()
        try:
            assert(data.shape[0] == label.shape[0])
            if data.ndim == 1:
                data = data[..., np.newaxis]

            self._data_container = data_container
            self._x = data
            self._y = label
        except:
            print('Check the case number of X and y')

    def SetData(self, data, label):
        try:
            assert(data.shape[0] == label.shape[0])
            if data.ndim == 1:
                data = data[..., np.newaxis]

            self._x = data
            self._y = label
        except:
            print('Check the case number of X and y')

    def SetModel(self, model):
        self.__model = model

    def GetModel(self):
        return self.__model

    def Fit(self):
        self.__model.fit(self._x, self._y)

    def Predict(self, x):
        return self.__model.predict(x)

    def Save(self, store_path):
        if os.path.isdir(store_path):
            store_path = os.path.join(store_path, 'model.pickle')

        if store_path[-7:] != '.pickle':
            print('Check the store path. ')
        else:
            with open(store_path, 'wb') as f:
                pickle.dump(self.__model, f)

    def Load(self, store_path):
        if os.path.isdir(store_path):
            store_path = os.path.join(store_path, 'model.pickle')

        if store_path[-7:] != '.pickle':
            print('Check the store path. ')
        else:
            with open(store_path, 'rb') as f:
                self.__model = pickle.load(f)

    @abstractmethod
    def GetName(self):
        pass

class SVM(Classifier):
    def __init__(self, **kwargs):
        super(SVM, self).__init__()
        if not 'kernel' in kwargs.keys():
            kwargs['kernel'] = 'linear'
        if not 'probability' in kwargs.keys():
            kwargs['probability'] = True
        super(SVM, self).SetModel(SVC(random_state=42, **kwargs))

    def GetName(self):
        return 'SVM'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(SVM, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(SVM, self).Predict(x)

    def Save(self, store_path):
        if not os.path.isdir(store_path):
            print('The store function of SVM must be a folder path')
            return

        # Save the coefficients
        try:
            coef_path = os.path.join(store_path, 'svm_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except:
            print("Not support Coef.")

        super(SVM, self).Save(store_path)


class LDA(Classifier):
    def __init__(self, **kwargs):
        super(LDA, self).__init__()
        super(LDA, self).SetModel(LinearDiscriminantAnalysis(**kwargs))

    def GetName(self):
        return 'LDA'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(LDA, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(LDA, self).Predict(x)

    def Save(self, store_path):
        if not os.path.isdir(store_path):
            print('The store function of SVM must be a folder path')
            return

        # Save the coefficients
        try:
            coef_path = os.path.join(store_path, 'lda_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except:
            print("Not support Coef.")

        super(LDA, self).Save(store_path)

class RandomForest(Classifier):
    def __init__(self, **kwargs):
        super(RandomForest, self).__init__()
        super(RandomForest, self).SetModel(RandomForestClassifier(random_state=42, **kwargs))

    def GetName(self):
        return 'RF'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(RandomForest, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(RandomForest, self).Predict(x)

class AE(Classifier):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        super(AE, self).SetModel(MLPClassifier(random_state=42, **kwargs))

    def GetName(self):
        return 'AE'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(AE, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(AE, self).Predict(x)



if __name__ == '__main__':
    import numpy as np
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])


    svm = SVM()
    svm.SetData(X, y)
    svm.Fit()
    print(svm.Predict([[1, 1]]))

    svm = AE()
    svm.SetData(X, y)
    svm.Fit()
    print(svm.Predict([[1, 1]]))

    svm = RandomForest()
    svm.SetData(X, y)
    svm.Fit()
    print(svm.Predict([[1, 1]]))

    svm = LDA()
    svm.SetData(X, y)
    svm.Fit()
    print(svm.Predict([[1, 1]]))


