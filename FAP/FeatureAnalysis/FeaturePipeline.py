from FAP.DataContainer.DataContainer import DataContainer
from FAP.FeatureAnalysis.CrossValidation import CrossValidation, CrossValidationOnFeatureNumber
from FAP.FeatureAnalysis.FeatureSelector import *
from FAP.FeatureAnalysis.Classifier import *

import pandas as pd

class FeatureAnalysisExplore:
    def __init__(self, feature_selector_list=[], classifier_list=[],
                 cv=CrossValidationOnFeatureNumber('5-folder'), max_feature_number=1):
        self.__feature_selector_list = feature_selector_list
        self.__classifier_list = classifier_list
        self.__cv = cv
        self.__max_feature_number = max_feature_number

    def RunOneModel(self, data_container, feature_selector, classifier, cv, store_folder=''):
        feature_selector.SetDataContainer(data_container)
        selected_data_container = feature_selector.Run(store_folder)

        cv.SetClassifier(classifier)
        cv.SetDataContainer(selected_data_container)

        train_metric, val_metric = cv.Run()

        return val_metric

    def Run(self, data_container, test_data_container=DataContainer(), store_folder=''):

        column_list = ['sample_number', 'positive_number', 'negative_number',
                       'auc', 'auc 95% CIs', 'accuracy', 'feature_number',
                       'Yorden Index', 'sensitivity', 'specificity',
                       'positive predictive value', 'negative predictive value']
        df = pd.DataFrame(columns=column_list)
        test_df = pd.DataFrame(columns=column_list)

        for feature_selector in self.__feature_selector_list:
            for classifier in self.__classifier_list:

                print(feature_selector.GetName() + '-' + classifier.GetName() + ':')

                self.__cv.SetClassifier(classifier)
                self.__cv.SetFeatureSelector(feature_selector)
                self.__cv.SetMaxFeatureNumber(self.__max_feature_number)

                model_store_folder = os.path.join(store_folder, feature_selector.GetName() + '-' + classifier.GetName())
                if not os.path.exists(model_store_folder):
                    os.mkdir(model_store_folder)
                val_return_list, test_return_list = self.__cv.Run(data_container, test_data_container=test_data_container,
                                            store_folder=model_store_folder, metric_name_list=('auc', 'accuracy'))

                if store_folder and os.path.isdir(store_folder):
                    val_auc_info = val_return_list[0]
                    store_path = os.path.join(store_folder, 'val_result.csv')
                    save_info = [val_auc_info[index] for index in column_list]
                    df.loc[feature_selector.GetName() + '-' + classifier.GetName()] = save_info
                    df.to_csv(store_path)

                    if test_data_container.GetArray().size > 0:

                        test_auc_info = test_return_list[0]
                        test_store_path = os.path.join(store_folder, 'test_result.csv')
                        test_save_info = [test_auc_info[index] for index in column_list]
                        test_df.loc[feature_selector.GetName() + '-' + classifier.GetName()] = test_save_info
                        test_df.to_csv(test_store_path)

                # return val_return_list, test_return_list


if __name__ == '__main__':
    print(os.getcwd())
    from DataContainer.DataContainer import DataContainer
    import pandas as pd

    data_container = DataContainer()
    data_container.Load(r'..\tempResult\NumericFeature.csv')
    data_container.UsualNormalize()

    column_list = ['sample_number', 'positive_number', 'negative_number',
                   'auc', 'auc 95% CIs', 'accuracy', 'feature_number',
                   'Yorden Index', 'sensitivity', 'specificity',
                   'positive predictive value', 'negative predictive value']

    df = pd.DataFrame(columns=column_list)

    # Set Feature Selector List
    feature_selector_list = []
    feature_selector_list.append(FeatureSelectPipeline([RemoveSameFeatures(), RemoveCosSimilarityFeatures(), FeatureSelectByANOVA()]))
    feature_selector_list.append(FeatureSelectPipeline([RemoveSameFeatures(), RemoveCosSimilarityFeatures(), FeatureSelectByRelief()]))
    feature_selector_list.append(FeatureSelectPipeline([RemoveSameFeatures(), RemoveCosSimilarityFeatures(), FeatureSelectByRFE()]))
    feature_selector_list.append(FeatureSelectPipeline([RemoveSameFeatures(), RemoveCosSimilarityFeatures(), FeatureSelectByPCA()]))

    # Set Classifier List
    classifier_list = []
    classifier_list.append(SVM())
    classifier_list.append(AE(max_iter=1000))
    classifier_list.append(RandomForest())
    classifier_list.append(LDA())

    fae = FeatureAnalysisExplore(feature_selector_list=feature_selector_list, classifier_list=classifier_list, max_feature_number=20)
    fae.Run(data_container, store_folder=r'..\tempResult')


