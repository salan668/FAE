import os

import numpy as np
import pickle
from copy import deepcopy
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class DimensionReduction:
    def __init__(self, name='', model=None, number=0, is_transform=False):
        self._name = name
        self.__model = model
        self.__remained_number = number
        self.__is_transform = is_transform

    def SetModel(self, model):
        self.__model = deepcopy(model)

    def GetModel(self):
        return self.__model

    def SetRemainedNumber(self, number):
        self.__remained_number = number

    def GetRemainedNumber(self):
        return self.__remained_number

    def SetTransform(self, is_transform):
        self.__is_transform = is_transform

    def GetTransform(self):
        return self.__is_transform

    def GetName(self):
        return self._name

    def GetDescription(self):
        text = "Since the dimension of feature space is low enough, we did not used any dimension reduction method " \
               "here to reduce the dimension of feature space. "
        return text

    def SaveDataContainer(self, data_container, store_folder, store_key):
        if store_folder:
            assert(len(store_key) > 0)
            container_store_path = os.path.join(store_folder, '{}_{}_feature.csv'.format(self._name, store_key))
            data_container.Save(container_store_path)


class DimensionReductionByPCA(DimensionReduction):
    def __init__(self, number=0):
        super(DimensionReductionByPCA, self).__init__(name='PCA', number=number, is_transform=True)
        super(DimensionReductionByPCA, self).SetModel(PCA(n_components=super(DimensionReductionByPCA, self).GetRemainedNumber()))

        self.__raw_feature_name = []
        self.__pca_feature_name = []

    def SaveInfo(self, store_folder):
        pca_sort_path = os.path.join(store_folder, '{}_sort.csv'.format(self._name))
        df = pd.DataFrame(data=self.GetModel().components_, index=self.__pca_feature_name, columns=self.__raw_feature_name)
        df.to_csv(pca_sort_path)

        pca_path = os.path.join(store_folder, '{}.pickle'.format(self._name))
        if pca_path[-7:] != '.pickle':
            print('Check the store path. ')
        else:
            with open(pca_path, 'wb') as f:
                pickle.dump(self.GetModel(), f)

    def LoadInfo(self, store_folder):
        if os.path.isdir(store_folder):
            pca_path = os.path.join(store_folder, '{}.pickle'.format(self._name))
        elif os.path.isfile(store_folder):
            pca_path = store_folder

        if not pca_path.endswith('.pickle'):
            print('Check the store path. ')
        else:
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
                self.SetModel(pca)
        super(DimensionReductionByPCA, self).SetRemainedNumber(self.GetModel().components_.shape[0])

    def SetRemainedNumber(self, number):
        super(DimensionReductionByPCA, self).SetRemainedNumber(number)
        super(DimensionReductionByPCA, self).SetModel(PCA(n_components=super(DimensionReductionByPCA, self).GetRemainedNumber()))

    def Transform(self, data_container, store_folder='', store_key=''):
        data = data_container.GetArray()
        if data.shape[1] != self.GetModel().components_.shape[1]:
            print('Data can not be transformed by existed PCA')
        sub_data = self.GetModel().transform(data)

        sub_feature_name = ['PCA_feature_' + str(index) for index in
                            range(1, super(DimensionReductionByPCA, self).GetRemainedNumber() + 1)]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(sub_data)
        new_data_container.SetFeatureName(sub_feature_name)
        new_data_container.UpdateFrameByData()

        if store_folder:
            self.SaveDataContainer(data_container, store_folder, store_key)

        return new_data_container

    def GetDescription(self):
        text = "Since the dimension of feature space was high, we applied principle component analysis (PCA) on the feature matrix. " \
               "The feature vector of the transformed feature matrix was independent to each other. "
        return text


    def Run(self, data_container, store_folder='', store_key=''):
        data = data_container.GetArray()
        self.SetRemainedNumber(np.min(data.shape))

        self.GetModel().fit(data)
        sub_data = self.GetModel().transform(data)

        sub_feature_name = ['PCA_feature_'+str(index) for index in range(1, super(DimensionReductionByPCA, self).GetRemainedNumber() + 1 )]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(sub_data)
        new_data_container.SetFeatureName(sub_feature_name)
        new_data_container.UpdateFrameByData()

        self.__raw_feature_name = data_container.GetFeatureName()
        self.__pca_feature_name = new_data_container.GetFeatureName()

        if store_folder and os.path.isdir(store_folder):
            self.SaveInfo(store_folder)
            self.SaveDataContainer(data_container, store_folder, store_key)

        return new_data_container


class DimensionReductionByPCC(DimensionReduction):
    def __init__(self, threshold=0.99):
        super(DimensionReductionByPCC, self).__init__(name='PCC')
        self.__threshold = threshold
        #TODO: Remove the __selected_index. This is not necessary.
        self.__selected_index = []

        self.__new_feature = []

    def __PCCSimilarity(self, data1, data2):
        return np.abs(pearsonr(data1, data2)[0])

    def SaveInfo(self, store_folder):
        pca_sort_path = os.path.join(store_folder, '{}_sort.csv'.format(self._name))
        df = pd.DataFrame(data=self.__new_feature)
        df.to_csv(pca_sort_path)

    def LoadInfo(self, store_folder):
        if os.path.isdir(store_folder):
            pcc_path = os.path.join(store_folder, '_sort.csv'.format(self._name))
        elif os.path.isfile(store_folder):
            pcc_path = store_folder

        selected_feature_df = pd.read_csv(pcc_path, index_col=0)
        self.__new_feature = selected_feature_df['0'].values.tolist()

    def GetSelectedFeatureIndex(self, data_container):
        data = data_container.GetArray()
        label = data_container.GetLabel()
        data /= np.linalg.norm(data, ord=2, axis=0)
        self.__selected_index = []

        for feature_index in range(data.shape[1]):
            is_similar = False
            assert(feature_index not in self.__selected_index)
            for save_index in self.__selected_index:
                if self.__PCCSimilarity(data[:, save_index], data[:, feature_index]) > self.__threshold:
                    if self.__PCCSimilarity(data[:, save_index], label) < self.__PCCSimilarity(data[:, feature_index],
                                                                                               label):
                        self.__selected_index[self.__selected_index.index(save_index)] = feature_index
                    is_similar = True
                    break
            if not is_similar:
                self.__selected_index.append(feature_index)
        self.__selected_index = sorted(self.__selected_index)

    def Transform(self, data_container, store_folder='', store_key=''):
        if len(self.__selected_index) == 0 and len(self.__new_feature) > 0:
            all_featues = data_container.GetFeatureName()
            self.__selected_index = [all_featues.index(temp) for temp in self.__new_feature]
        assert(len(self.__selected_index) > 0)
        new_data = data_container.GetArray()[:, self.__selected_index]
        new_feature = [data_container.GetFeatureName()[t] for t in self.__selected_index]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(new_data)
        new_data_container.SetFeatureName(new_feature)
        new_data_container.UpdateFrameByData()

        if store_folder and os.path.isdir(store_folder):
            self.SaveDataContainer(new_data_container, store_folder, store_key)

        return new_data_container

    def Run(self, data_container, store_folder='', store_key=''):
        self.GetSelectedFeatureIndex(data_container)

        new_data = data_container.GetArray()[:, self.__selected_index]
        self.__new_feature = [data_container.GetFeatureName()[t] for t in self.__selected_index]

        new_data_container = deepcopy(data_container)
        new_data_container.SetArray(new_data)
        new_data_container.SetFeatureName(self.__new_feature)
        new_data_container.UpdateFrameByData()

        if store_folder and os.path.isdir(store_folder):
            self.SaveDataContainer(new_data_container, store_folder, store_key)
            self.SaveInfo(store_folder)

        return new_data_container

    def GetDescription(self):
        text = "Since the dimension of feature space was high, we compared the similarity of each feature pair. " \
               "If the PCC value of the feature pair was larger than the set value, we removed one of them. After this " \
               "process, the dimension of the feature space was reduced and each feature was independent to each other. "
        return text

class DimensionReductionByVIF(DimensionReduction):
    # 对于特征数较多的情况，VIF计算出来都无穷大，似乎不适合这种情况，故未写完。
    def __init__(self):
        super(DimensionReductionByVIF, self).__init__()

    def CalculateVIF(self, df, thresh=5):
        '''
        Ref: https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python

        Calculates VIF each feature in a pandas dataframe
        A constant must be added to variance_inflation_factor or the results will be incorrect

        :param df: the pandas dataframe containing only the predictor features, not the response variable
        :param thresh: the max VIF value before the feature is removed from the dataframe
        :return: dataframe with features removed
        '''
        const = add_constant(df)
        cols = const.columns
        variables = np.arange(const.shape[1])
        vif_df = pd.Series([variance_inflation_factor(const.values, i)
                            for i in range(const.shape[1])],
                           index=const.columns).to_frame()

        vif_df = vif_df.sort_values(by=0, ascending=False).rename(columns={0: 'VIF'})
        vif_df = vif_df.drop('const')
        vif_df = vif_df[vif_df['VIF'] > thresh]

        col_to_drop = list(vif_df.index)

        for i in col_to_drop:
            df = df.drop(columns=i)

        return df

    def CalculateVIF2(self, df):
        # initialize dictionaries
        vif_dict, tolerance_dict = {}, {}

        # form input data for each exogenous variable
        for exog in df.columns:
            not_exog = [i for i in df.columns if i != exog]
            X, y = df[not_exog], df[exog]

            # extract r-squared from the fit
            r_squared = LinearRegression().fit(X, y).score(X, y)

            # calculate VIF
            vif = 1/(1 - r_squared)
            vif_dict[exog] = vif

            # calculate tolerance
            tolerance = 1 - r_squared
            tolerance_dict[exog] = tolerance

        # return VIF DataFrame
        df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

        return df_vif


if __name__ == '__main__':
    data_path = r'..\..\Demo\train_numeric_feature.csv'
    from BC.DataContainer.DataContainer import DataContainer
    from BC.FeatureAnalysis.Normalizer import NormalizerZeroCenter
    pca = DimensionReductionByPCA()

    dc = DataContainer()
    dc.Load(data_path)
    dc = NormalizerZeroCenter.Run(dc)
    # dc = pca.Run(dc)

    df = pd.DataFrame(dc.GetArray(), index=dc.GetCaseName(), columns=dc.GetFeatureName())
    dr = DimensionReductionByVIF()

    new_df = dr.CalculateVIF(df)

    print(dc.GetArray().shape, new_df.shape)