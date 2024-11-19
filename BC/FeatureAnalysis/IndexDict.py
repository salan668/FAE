from BC.FeatureAnalysis.DataBalance import *
from BC.FeatureAnalysis.Normalizer import *
from BC.FeatureAnalysis.DimensionReduction import *
from BC.FeatureAnalysis.FeatureSelector import *
from BC.FeatureAnalysis.Classifier import *
from BC.FeatureAnalysis.CrossValidation import *

from BC.Utility.Constants import *
from BC.HyperParameterConfig.HyperParamManager import RandomSeed


class Index2Dict:
    def __init__(self, root=None):
        if root is None:
            self.random_seed = RandomSeed(os.path.join('BC', 'HyperParameters', 'RandomSeed.json')).random_seed
        else:
            self.random_seed = RandomSeed(os.path.join(root, 'BC', 'HyperParameters', 'RandomSeed.json')).random_seed

    def GetInstantByIndex(self, name):
        if name == NoneBalance().GetName():
            return NoneBalance()
        elif name == UpSampling().GetName():
            return UpSampling(random_state=self.random_seed[BALANCE_UP_SAMPLING])
        elif name == DownSampling().GetName():
            return DownSampling(random_state=self.random_seed[BALANCE_DOWN_SAMPLING])
        elif name == SmoteSampling().GetName():
            return SmoteSampling(random_state=self.random_seed[BALANCE_SMOTE])

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
            return SVM(random_state=self.random_seed[CLASSIFIER_SVM])
        elif name == LDA().GetName():
            return LDA()
        elif name == AE().GetName():
            return AE(random_state=self.random_seed[CLASSIFIER_AE])
        elif name == RandomForest().GetName():
            return RandomForest(random_state=self.random_seed[CLASSIFIER_RF])
        elif name == DecisionTree().GetName():
            return DecisionTree(random_state=self.random_seed[CLASSIFIER_DT])
        elif name == AdaBoost().GetName():
            return AdaBoost(random_state=self.random_seed[CLASSIFIER_AB])
        elif name == NaiveBayes().GetName():
            return NaiveBayes()
        elif name == GaussianProcess().GetName():
            return GaussianProcess(random_state=self.random_seed[CLASSIFIER_GP])
        elif name == LR().GetName():
            return LR(random_state=self.random_seed[CLASSIFIER_LR])
        elif name == LRLasso().GetName():
            return LRLasso(random_state=self.random_seed[CLASSIFIER_LRLasso])

        elif name == CrossValidation5Fold.GetName():
            return CrossValidation5Fold
        elif name == CrossValidation10Fold.GetName():
            return CrossValidation10Fold
        elif name == CrossValidationLOO.GetName():
            return CrossValidationLOO