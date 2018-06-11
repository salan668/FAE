import pandas as pd
import os

from FAP.DataContainer.DataContainer import DataContainer
from FAP.FeatureAnalysis.FeatureSelector import RemoveSameFeatures, RemoveCosSimilarityFeatures, FeatureSelectPipeline
from FAP.FeatureAnalysis.FeatureSelector import FeatureSelectByANOVA, FeatureSelectByRFE, FeatureSelectByPCA, FeatureSelectByRelief
from FAP.FeatureAnalysis.Classifier import SVM, RandomForest, AE, LDA
from FAP.FeatureAnalysis.CrossValidation import CrossValidationOnFeatureNumber
from FAP.FeatureAnalysis.FeaturePipeline import FeatureAnalysisExplore

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
data_container.Load(r'numeric_feature.csv')
data_container.UsualAndL2Normalize()

fae = FeatureAnalysisExplore(feature_selector_list=feature_selector_list, classifier_list=classifier_list, cv=cv, max_feature_number=20)
fae.Run(data_container, store_folder=r'Result')
