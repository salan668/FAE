# Pipeline Key
BALANCE = 'Balance'
NORMALIZR = 'Normalizer'
DIMENSION_REDUCTION = 'DimensionReduction'
FEATURE_SELECTOR = 'FeatureSelector'
FEATURE_NUMBER = 'FeatureNumber'
CLASSIFIER = 'Classifier'
CROSS_VALIDATION = 'CrossValidation'

REMOVE_NONE = 'RemoveNone'
REMOVE_CASE = 'RemoveCase'
REMOVE_FEATURE = 'RemoveFeature'

# Data Balance Key
BALANCE_NONE = 'Balance_None'
BALANCE_UP_SAMPLING = 'UpSampling'
BALANCE_DOWN_SAMPLING = 'DownSampling'
BALANCE_SMOTE = 'SMOTE'
BALANCE_SMOTE_TOMEK = 'SMOTETomek'

# Classifier Key
CLASSIFIER_LR = 'LR'
CLASSIFIER_LDA = 'LDA'
CLASSIFIER_SVM = 'SVM'
CLASSIFIER_RF = 'RF'
CLASSIFIER_AE = 'AE'
CLASSIFIER_AB = 'AB'
CLASSIFIER_NB = 'NB'
CLASSIFIER_DT = 'DT'
CLASSIFIER_GP = 'GP'
CLASSIFIER_LRLasso = 'LRLasso'

# Metric Key
NUMBER = 'Number'
POS_NUM = 'PosNum'
NEG_NUM = 'NegNum'
AUC = 'AUC'
AUC_PR = 'AUC-PR'
CUTOFF = 'Cutoff'
MCC = 'MCC'
AUC_CI = '95% CIs'
AUC_STD = 'Std'
ACC = 'Acc'
YI = 'Youden Index'
SEN = 'Sen'
SPE = 'Spe'
PPV = 'PPV'
NPV = 'NPV'
HEADER = [NUMBER, POS_NUM, NEG_NUM, AUC, AUC_CI, AUC_STD, AUC_PR, CUTOFF, MCC, ACC, YI, SEN, SPE, PPV, NPV]

# Type Key
CV_TRAIN = 'cv_train'
CV_VAL = 'cv_val'
BALANCE_TRAIN = 'balance_train'
TRAIN = 'train'
TEST = 'test'

# Plot Type
ROC_CURVE = 'ROC'
PR_CURVE = 'PR Curve'
BOX_PLOT = 'Box Plot'
CALIBRATION_CURVE = 'Calibration Curve'
PROBABILITY = 'Probability'
VIOLIN_PLOT = 'Violin Plot'
PLOT_TYPE = [ROC_CURVE, PR_CURVE, BOX_PLOT, CALIBRATION_CURVE, PROBABILITY, VIOLIN_PLOT]