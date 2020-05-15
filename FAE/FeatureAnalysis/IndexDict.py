from FAE.FeatureAnalysis.DataBalance import *
from FAE.FeatureAnalysis.Normalizer import *
from FAE.FeatureAnalysis.DimensionReduction import *
from FAE.FeatureAnalysis.FeatureSelector import *
from FAE.FeatureAnalysis.Classifier import *
from FAE.FeatureAnalysis.CrossValidation import *


class Index2Dict:
    def __init__(self):
        pass

    def GetInstantByIndex(self, name):
        if name == NoneBalance().GetName():
            return NoneBalance()
        elif name == UpSampling().GetName():
            return UpSampling()
        elif name == DownSampling().GetName():
            return DownSampling()
        elif name == SmoteSampling().GetName():
            return SmoteSampling()

        elif name == NormalizerNone.GetName():
            return NormalizerNone
        elif name == NormalizerMinMax.GetName():
            return NormalizerMinMax
        elif name == NormalizerZscore.GetName():
            return NormalizerZscore
        elif name == NormalizerMean.GetName():
            return NormalizerMean

        elif name == DimensionReductionByPCA().GetName():
            return DimensionReductionByPCA()
        elif name == 'Cos':
            return DimensionReductionByPCC()
        elif name == DimensionReductionByPCC().GetName():
            return DimensionReductionByPCC()

        elif name == FeatureSelectByRelief().GetName():
            return FeatureSelectByRelief()
        elif name == FeatureSelectByANOVA().GetName():
            return FeatureSelectByANOVA()
        elif name == FeatureSelectByRFE().GetName():
            return FeatureSelectByRFE()
        elif name == FeatureSelectByMrmr().GetName():
            return FeatureSelectByMrmr()
        elif name == FeatureSelectByKruskalWallis().GetName():
            return FeatureSelectByKruskalWallis()

        elif name == SVM().GetName():
            return SVM()
        elif name == LDA().GetName():
            return LDA()
        elif name == AE().GetName():
            return AE()
        elif name == RandomForest().GetName():
            return RandomForest()
        elif name == DecisionTree().GetName():
            return DecisionTree()
        elif name == AdaBoost().GetName():
            return AdaBoost()
        elif name == NaiveBayes().GetName():
            return NaiveBayes()
        elif name == GaussianProcess().GetName():
            return GaussianProcess()
        elif name == LR().GetName():
            return LR()
        elif name == LRLasso().GetName():
            return LRLasso()

        elif name == CrossValidation5Fold.GetName():
            return CrossValidation5Fold
        elif name == CrossValidation10Fold.GetName():
            return CrossValidation10Fold
        elif name == CrossValidationLOO.GetName():
            return CrossValidationLOO