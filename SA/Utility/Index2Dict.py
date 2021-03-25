"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/2/1
"""

from SA.Normalizer import *
from SA.DimensionReducer import *
from SA.FeatureSelector import *
from SA.Fitter import *


class Index2Dict:
    def __init__(self):
        pass

    def GetInstantByIndex(self, name):
        if name == NormalizerNone.name:
            return NormalizerNone
        elif name == NormalizerMinMax.name:
            return NormalizerMinMax
        elif name == NormalizerZscore.name:
            return NormalizerZscore
        elif name == NormalizerMean.name:
            return NormalizerMean

        elif name == DimensionReducerNone().GetName():
            return DimensionReducerNone()
        elif name == DimensionReducerPcc().GetName():
            return DimensionReducerPcc()

        elif name == FeatureSelectorAll.GetName():
            return FeatureSelectorAll
        elif name == FeatureSelectorCluster.GetName():
            return FeatureSelectorCluster

        elif name == CoxPH().name:
            return CoxPH()