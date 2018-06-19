import pandas as pd
import os

from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.FeatureSelector import RemoveSameFeatures, RemoveCosSimilarityFeatures, FeatureSelectPipeline
from FAE.FeatureAnalysis.FeatureSelector import FeatureSelectByANOVA, FeatureSelectByRFE, FeatureSelectByPCA, FeatureSelectByRelief
from FAE.FeatureAnalysis.Classifier import SVM, RandomForest, AE, LDA
from FAE.FeatureAnalysis.CrossValidation import CrossValidationOnFeatureNumber
from FAE.FeatureAnalysis.FeaturePipeline import FeatureAnalysisExplore

if __name__ == '__main__':

    print(os.getcwd())

    column_list = ['sample_number', 'positive_number', 'negative_number',
                   'auc', 'auc 95% CIs', 'accuracy', 'feature_number',
                   'Yorden Index', 'accuracy', 'sensitivity', 'specificity',
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

    cv = CrossValidationOnFeatureNumber('5-folder')

    data_container = DataContainer()
    if os.path.exists(r'Example\numeric_feature.csv'):
        data_path = r'Example\numeric_feature.csv'  # Run by Console
    elif os.path.exists(r'numeric_feature.csv'):
        data_path = r'numeric_feature.csv'          # Run by PyCharm
    data_container.Load(data_path)
    data_container.UsualAndL2Normalize()

    fae = FeatureAnalysisExplore(feature_selector_list=feature_selector_list, classifier_list=classifier_list, cv=cv, max_feature_number=20)
    if os.path.exists(r'Result'):
        store_path = r'Result'                      # Run By PyCharm
    elif os.path.exists(r'Example\Result'):
        store_path = r'Example\Result'              # Run By Console

    fae.Run(data_container, store_folder=store_path)
