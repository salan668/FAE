import numpy as np
import pickle
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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

    def GetDescription(self):
        text = "We did not use any classifier. "
        return

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
        if not 'C' in kwargs.keys():
            kwargs['C'] = 1.0
        if not 'probability' in kwargs.keys():
            kwargs['probability'] = True
        super(SVM, self).SetModel(SVC(random_state=42, **kwargs))

        self.__name = 'SVM_'+ kwargs['kernel'] + '_C_' + '{:.3f}'.format(kwargs['C'])

    def GetName(self):
        return 'SVM'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(SVM, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(SVM, self).Predict(x)

    def GetDescription(self):
        text = "We used support vector machine (SVM) as the classifier. SVM was an effective and robust classifier to " \
               "build the model. The kernel function has the ability to map the features into a higher dimension to search " \
               "the hyper-plane for separating the cases with different labels. Here we used the linear kernel function because " \
               "it was easier to explain the coefficients of the features for the final model. "
        return text

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
        if not 'early_stopping' in kwargs.keys():
            kwargs['early_stopping'] = True
        super(AE, self).SetModel(MLPClassifier(random_state=42, **kwargs))

    def GetName(self):
        return 'AE'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(AE, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(AE, self).Predict(x)

class AdaBoost(Classifier):
    def __init__(self, **kwargs):
        super(AdaBoost, self).__init__()
        super(AdaBoost, self).SetModel(AdaBoostClassifier(random_state=42, **kwargs))

    def GetName(self):
        return 'AB'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(AdaBoost, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(AdaBoost, self).Predict(x)

class DecisionTree(Classifier):
    def __init__(self, **kwargs):
        super(DecisionTree, self).__init__()
        super(DecisionTree, self).SetModel(DecisionTreeClassifier(random_state=42, **kwargs))

    def GetName(self):
        return 'DT'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(DecisionTree, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(DecisionTree, self).Predict(x)

class GaussianProcess(Classifier):
    def __init__(self, **kwargs):
        super(GaussianProcess, self).__init__()
        super(GaussianProcess, self).SetModel(GaussianProcessClassifier(random_state=42, **kwargs))

    def GetName(self):
        return 'GP'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(GaussianProcess, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(GaussianProcess, self).Predict(x)

class NativeBayes(Classifier):
    def __init__(self, **kwargs):
        super(NativeBayes, self).__init__()
        super(NativeBayes, self).SetModel(GaussianNB(**kwargs))

    def GetName(self):
        return 'NB'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(NativeBayes, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(NativeBayes, self).Predict(x)

class LR(Classifier):
    def __init__(self, **kwargs):
        super(LR, self).__init__()
        super(LR, self).SetModel(LogisticRegression(**kwargs))

    def GetName(self):
        return 'LR'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(LR, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(LR, self).Predict(x)

    def Save(self, store_path):
        if not os.path.isdir(store_path):
            print('The store function of SVM must be a folder path')
            return

        # Save the coefficients
        try:
            coef_path = os.path.join(store_path, 'lr_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except:
            print("Not support Coef.")

        super(LR, self).Save(store_path)

class LRLasso(Classifier):
    def __init__(self, **kwargs):
        super(LRLasso, self).__init__()
        super(LRLasso, self).SetModel(LogisticRegression(penalty='l1', **kwargs))

    def GetName(self):
        return 'LRLasso'

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(LRLasso, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(LRLasso, self).Predict(x)

    def Save(self, store_path):
        if not os.path.isdir(store_path):
            print('The store function of SVM must be a folder path')
            return

        # Save the coefficients
        try:
            coef_path = os.path.join(store_path, 'lrlasso_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except:
            print("Not support Coef.")

        super(LRLasso, self).Save(store_path)

if __name__ == '__main__':
    import numpy as np
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])

    clf = SVM()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))


    clf = AE()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = RandomForest()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = LDA()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = AdaBoost()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = DecisionTree()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = GaussianProcess()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = NativeBayes()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = LR()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))

    clf = LRLasso()
    clf.SetData(X, y)
    clf.Fit()
    print(clf.GetName(), clf.Predict([[1, 1]]))


