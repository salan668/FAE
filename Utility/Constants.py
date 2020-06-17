# Pipeline Key
BALANCE = 'Balance'
NORMALIER = 'Normalizer'
DIMENSION_REDUCTION = 'DimensionReduction'
FEATURE_SELECTOR = 'FeatureSelector'
FEATURE_NUMBER = 'FeatureNumber'
CLASSIFIER = 'Classifier'
CROSS_VALIDATION = 'CrossValidation'

REMOVE_NONE = 'RemoveNone'
REMOVE_CASE = 'RemoveCase'
REMOVE_FEATURE = 'RemoveFeature'

# Data Balance Key
BALANCE_UP_SAMPLING = 'UpSampling'
BALANCE_DOWN_SAMPLING = 'DownSampling'
BALANCE_SMOTE = 'SMOTE'
BALANCE_SMOTE_TOMEK = 'SMOTETomek'

# Classifier Key
CLASSIFIER_LR = 'LR'
CLASSIFIER_SVM = 'SVM'
CLASSIFIER_RF = 'RF'
CLASSIFIER_AE = 'AE'
CLASSIFIER_AB = 'AB'
CLASSIFIER_DT = 'DT'
CLASSIFIER_GP = 'GP'
CLASSIFIER_LRLasso = 'LRLasso'

# Metric Key
NUMBER = 'Number'
POS_NUM = 'PosNum'
NEG_NUM = 'NegNum'
AUC = 'AUC'
AUC_CI = '95% CIs'
AUC_STD = 'Std'
ACC = 'Acc'
YI = 'Youden Index'
SEN = 'Sen'
SPE = 'Spe'
PPV = 'PPV'
NPV = 'NPV'
HEADER = [NUMBER, POS_NUM, NEG_NUM, AUC, AUC_CI, AUC_STD, ACC, YI, SEN, SPE, PPV, NPV]


# Type Key
CV_TRAIN = 'cv_train'
CV_VAL = 'cv_val'
BALANCE_TRAIN = 'balance_train'
TRAIN = 'train'
TEST = 'test'

# Version Key
VERSION_NAME = 'Version'
MAJOR = 0
MINOR = 3
PATCH = 5
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, PATCH)
ACCEPT_VERSION = ['0.3.0', '0.3.1', '0.3.2', '0.3.3', '0.3.4', '0.3.5']
