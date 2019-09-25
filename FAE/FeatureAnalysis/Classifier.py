import os
import pickle
from copy import deepcopy

import pandas as pd
import numpy as np
from abc import ABCMeta,abstractmethod
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from FAE.DataContainer.DataContainer import DataContainer
from Utility.EcLog import eclog

class Classifier:
    '''
    This is the base class of the classifer. All the specific classifier need to be artributed from this base class.
    '''
    def __init__(self):
        self.__model = None
        self._x = np.array([])
        self._y = np.array([])
        self._data_container = DataContainer()
        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()

    def __deepcopy__(self, memodict={}):
        copy_classifier = type(self)()
        copy_classifier._data_container = deepcopy(self._data_container)
        copy_classifier._x, copy_classifier._y = deepcopy(self._x), deepcopy(self._y)
        copy_classifier.SetModel(deepcopy(self.__model))
        return copy_classifier

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
        except Exception as e:
            content = 'The case number is not same to the label number: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

    def SetData(self, data, label):
        try:
            assert(data.shape[0] == label.shape[0])
            if data.ndim == 1:
                data = data[..., np.newaxis]

            self._x = data
            self._y = label
        except Exception as e:
            content = 'The case number is not same to the label number: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

    def SetModel(self, model):
        self.__model = model

    def GetModel(self):
        return self.__model

    def SetModelParameter(self, param):
        self.__model.set_params(**param)

    def Fit(self):
        self.__model.fit(self._x, self._y)

    def GetDescription(self):
        text = "We did not use any classifier. "
        return text

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
            coef_path = os.path.join(store_path, 'SVM_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except Exception as e:
            content = 'SVM with specific kernel does not give coef: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

        #Save the intercept_
        try:
            intercept_path = os.path.join(store_path, 'SVM_intercept.csv')
            intercept_df = pd.DataFrame(data=(self.GetModel().intercept_).reshape(1, 1), index=['intercept'], columns=['value'])
            intercept_df.to_csv(intercept_path)
        except Exception as e:
            content = 'SVM with specific kernel does not give intercept: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

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

    def GetDescription(self):
        text = "We used linear discriminant analysis (LDA) as the classifier. LDA was an linear classifier by " \
               "fitting class conditional densities to the data and using Bayes’rule. "
        return text

    def Save(self, store_path):
        if not os.path.isdir(store_path):
            print('The store function of LDA must be a folder path')
            return

        # Save the coefficients
        try:
            coef_path = os.path.join(store_path, 'LDA_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except Exception as e:
            content = 'LDA with specific kernel does not give coef: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

        super(LDA, self).Save(store_path)

class RandomForest(Classifier):
    def __init__(self, **kwargs):
        super(RandomForest, self).__init__()
        if 'n_estimators' not in kwargs.keys():
            super(RandomForest, self).SetModel(RandomForestClassifier(random_state=42, n_estimators=200, **kwargs))
        else:
            super(RandomForest, self).SetModel(RandomForestClassifier(random_state=42, **kwargs))

    def GetName(self):
        return 'RF'

    def GetDescription(self):
        text = "We used random forest as the classifier. Random forest is an ensemble learning method which combining " \
               "multiple decision trees at different subset of the training data set. Random forest is an effective " \
               "method to avoid over-fitting. "
        return text

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

    def GetDescription(self):
        text = "We used multi-layer perceptron (MLP), sometimes called auto-encoder (AE), as the classifier. MLP is based " \
               "neural network with multi-hidden layers to find the mapping from inputted features to the label. Here " \
               "we used 1 hidden layers with 100 hidden units. The non-linear activate function was rectified linear " \
               "unit function and the optimizer was Adam with step 0.001. "
        return text

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

    def GetDescription(self):
        text = "We used AdaBoost as the classifier. AdaBoost is a meta-algorithm that conjunct other type of algorithms " \
               "and combine them to get a final output of boosted classifier. AdaBoost is sensitive to the noise and " \
               "the outlier. Over-fitting can also be avoided by AdaBoost. Here we used decision tree as the base classifier. "
        return text

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

    def GetDescription(self):
        text = "We used decision tree as the classifier. Decision tree is a non-parametric supervised learning method " \
               "and can be used for classification with high interpretation. "
        return text

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

    def GetDescription(self):
        text = "We used Gaussian process as the classifier. Gaussian process combines the features to build a joint " \
               "distribution to estimate the probability of the classification. "
        return text

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(GaussianProcess, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(GaussianProcess, self).Predict(x)

class NaiveBayes(Classifier):
    def __init__(self, **kwargs):
        super(NaiveBayes, self).__init__()
        super(NaiveBayes, self).SetModel(GaussianNB(**kwargs))

    def GetName(self):
        return 'NB'

    def GetDescription(self):
        text = "We used naive Bayes as the classifier. Naive Bayes is a kind of probabilistic classifiers based on Bayes" \
               "theorem. Naive Bayes requires  number of parameters linear in the number of features. "
        return text

    def Predict(self, x, is_probability=True):
        if is_probability:
            return super(NaiveBayes, self).GetModel().predict_proba(x)[:, 1]
        else:
            return super(NaiveBayes, self).Predict(x)

class LR(Classifier):
    def __init__(self, **kwargs):
        super(LR, self).__init__()
        if 'solver' in kwargs.keys():
            super(LR, self).SetModel(LogisticRegression(penalty='none', **kwargs))
        else:
            super(LR, self).SetModel(LogisticRegression(penalty='none', solver='saga', tol=0.01, **kwargs))


    def GetName(self):
        return 'LR'

    def GetDescription(self):
        text = "We used logistic regression as the classifier. Logistic regression is a linear classifier that " \
               "combines all the features. A hyper-plane was searched in the high dimension to separate the samples.  "
        return text

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
            coef_path = os.path.join(store_path, 'LR_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except Exception as e:
            content = 'LR can not load coef: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

        try:
            intercept_path = os.path.join(store_path, 'LR_intercept.csv')
            intercept_df = pd.DataFrame(data=(self.GetModel().intercept_).reshape(1, 1), index=['intercept'], columns=['value'])
            intercept_df.to_csv(intercept_path)
        except Exception as e:
            content = 'LR can not load intercept: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

        super(LR, self).Save(store_path)

class LRLasso(Classifier):
    def __init__(self, **kwargs):
        super(LRLasso, self).__init__()
        if 'solver' in kwargs.keys():
            super(LRLasso, self).SetModel(LogisticRegression(penalty='l1', **kwargs))
        else:
            super(LRLasso, self).SetModel(LogisticRegression(penalty='l1', solver='liblinear', **kwargs))

    def GetName(self):
        return 'LRLasso'

    def GetDescription(self):
        text = "We used logistic regression with LASSO constrain as the classifier. Logistic regression with LASSON " \
               "constrain is a linear classifier based on logistic regression. L1 norm is added in the final lost " \
               "function and the weights was constrained, which make the features sparse. "
        return text

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
            coef_path = os.path.join(store_path, 'LRLasso_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.GetModel().coef_), index=self._data_container.GetFeatureName(), columns=['Coef'])
            df.to_csv(coef_path)
        except Exception as e:
            content = 'LASSO can not load coef: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

        try:
            intercept_path = os.path.join(store_path, 'LRLasso_intercept.csv')
            intercept_df = pd.DataFrame(data=(self.GetModel().intercept_).reshape(1, 1), index=['intercept'], columns=['value'])
            intercept_df.to_csv(intercept_path)
        except Exception as e:
            content = 'LASSO can not load intercept: '
            self.logger.error('{}{}'.format(content, str(e)))
            print('{} \n{}'.format(content, e.__str__()))

        super(LRLasso, self).Save(store_path)

if __name__ == '__main__':
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 0, 0])

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

    clf = NaiveBayes()
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


