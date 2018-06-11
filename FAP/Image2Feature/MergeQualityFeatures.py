# This class should be applied after Radiomics features extracted.
from FAP.Image2Feature.RadiomicsFeatureExtractor import RadiomicsFeatureExtractor

class MergeQualityFeature():
    def __init__(self, extractor):
        self.extractor = extractor

if __name__ == '__main__':
    pass