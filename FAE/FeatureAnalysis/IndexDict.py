from FAE.FeatureAnalysis.Normalizer import *
from FAE.FeatureAnalysis.DimensionReduction import *
from FAE.FeatureAnalysis.FeatureSelector import *
from FAE.FeatureAnalysis.Classifier import *

from copy import deepcopy

class Index2Dict:
    def __init__(self):
        pass

    def GetInstantByIndex(self, name):
        if name == NormalizerNone().GetName():
            return NormalizerNone()
        elif name == NormalizerUnit().GetName():
            return NormalizerUnit()
        elif name == NormalizerZeroCenter().GetName():
            return NormalizerZeroCenter()
        elif name == NormalizerZeroCenterAndUnit().GetName():
            return NormalizerZeroCenterAndUnit()
        elif name == DimensionReductionByPCA().GetName():
            return DimensionReductionByPCA()
        elif name == DimensionReductionByCos().GetName():
            return DimensionReductionByCos()
        elif name == FeatureSelectByRelief().GetName():
            return FeatureSelectByRelief()
        elif name == FeatureSelectByANOVA().GetName():
            return FeatureSelectByANOVA()
        elif name == FeatureSelectByRFE().GetName():
            return FeatureSelectByRFE()
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
        elif name == NativeBayes().GetName():
            return NativeBayes()
        elif name == GaussianProcess().GetName():
            return GaussianProcess()
        elif name == LR().GetName():
            return LR()
        elif name == LRLasso().GetName():
            return LRLasso()