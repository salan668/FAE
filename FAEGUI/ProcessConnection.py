import shutil
import numpy as np
from copy import deepcopy

from PyQt5.QtWidgets import *
from GUI.Process import Ui_Process
from PyQt5.QtCore import *
from Utility.EcLog import eclog
from FAE.FeatureAnalysis.Normalizer import *
from FAE.FeatureAnalysis.DimensionReduction import *
from FAE.FeatureAnalysis.FeatureSelector import *
from FAE.FeatureAnalysis.Classifier import *
from FAE.FeatureAnalysis.FeaturePipeline import FeatureAnalysisPipelines
from FAE.FeatureAnalysis.CrossValidation import *

import os
import struct

class CVRun(QThread):
    signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        pass

    def SetProcessConnectionAndStore_folder(self, process_connection, store_folder):
        self._process_connection = process_connection
        self._store_folder = store_folder

    def run(self):
        for current_normalizer_name, current_dimension_reductor_name, \
            current_feature_selector_name, curreent_feature_num, \
            current_classifier_name, num, total_num \
                in self._process_connection.fae.Run(self._process_connection.training_data_container,
                                                    self._process_connection.testing_data_container,
                                                    self._store_folder,
                                                    self._process_connection.checkHyperParameters.isChecked()):
            text = self._process_connection.GenerateVerboseTest(current_normalizer_name,
                                                     current_dimension_reductor_name,
                                                     current_feature_selector_name,
                                                     current_classifier_name,
                                                     curreent_feature_num,
                                                     num,
                                                     total_num)

            self.signal.emit(text)  # 反馈信号出去

        self.signal.emit(text + "\n DONE!")
        self._process_connection.SetStateAllButtonWhenRunning(True)

class ProcessConnection(QWidget, Ui_Process):
    def __init__(self, parent=None):
        self.training_data_container = DataContainer()
        self.testing_data_container = DataContainer()
        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()
        self.fae = FeatureAnalysisPipelines(logger=self.logger)
        self.__process_normalizer_list = []
        self.__process_dimension_reduction_list = []
        self.__process_feature_selector_list = []
        self.__process_feature_number_list = []
        self.__process_classifier_list = []

        super(ProcessConnection, self).__init__(parent)
        self.setupUi(self)

        self.buttonLoadTrainingData.clicked.connect(self.LoadTrainingData)
        self.buttonLoadTestingData.clicked.connect(self.LoadTestingData)

        self.checkNormalizeUnit.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeZeroCenter.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeUnitWithZeroCenter.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizationAll.clicked.connect(self.SelectAllNormalization)

        self.checkPCA.clicked.connect(self.UpdatePipelineText)
        self.checkRemoveSimilarFeatures.clicked.connect(self.UpdatePipelineText)
        self.checkPreprocessAll.clicked.connect(self.SelectAllPreprocess)

        self.spinBoxMinFeatureNumber.valueChanged.connect(self.MinFeatureNumberChange)
        self.spinBoxMaxFeatureNumber.valueChanged.connect(self.MaxFeatureNumberChange)

        self.checkANOVA.clicked.connect(self.UpdatePipelineText)
        self.checkRFE.clicked.connect(self.UpdatePipelineText)
        self.checkRelief.clicked.connect(self.UpdatePipelineText)
        # self.checkMRMR.clicked.connect(self.UpdatePipelineText)
        self.checkFeatureSelectorAll.clicked.connect(self.SelectAllFeatureSelector)

        self.checkSVM.clicked.connect(self.UpdatePipelineText)
        self.checkLDA.clicked.connect(self.UpdatePipelineText)
        self.checkAE.clicked.connect(self.UpdatePipelineText)
        self.checkRF.clicked.connect(self.UpdatePipelineText)
        self.checkLogisticRegression.clicked.connect(self.UpdatePipelineText)
        self.checkLRLasso.clicked.connect(self.UpdatePipelineText)
        self.checkAdaboost.clicked.connect(self.UpdatePipelineText)
        self.checkDecisionTree.clicked.connect(self.UpdatePipelineText)
        self.checkNaiveBayes.clicked.connect(self.UpdatePipelineText)
        self.checkGaussianProcess.clicked.connect(self.UpdatePipelineText)
        self.checkClassifierAll.clicked.connect(self.SelectAllClassifier)

        self.radio5folder.clicked.connect(self.UpdatePipelineText)
        self.radio10Folder.clicked.connect(self.UpdatePipelineText)
        self.radioLOO.clicked.connect(self.UpdatePipelineText)

        self.buttonRun.clicked.connect(self.Run)

        self.UpdatePipelineText()
        self.SetStateButtonBeforeLoading(False)

    def LoadTrainingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=r'C:\MyCode\FAE\Example', filter="csv files (*.csv)")
        try:
            self.training_data_container.Load(file_name)
            self.SetStateButtonBeforeLoading(True)
            self.lineEditTrainingData.setText(file_name)
            self.UpdateDataDescription()
            self.logger.info('Open CSV file ' + file_name + ' succeed.')
        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            print('出错啦！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected training data mismatch.')


    def LoadTestingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', filter="csv files (*.csv)")
        try:
            self.testing_data_container.Load(file_name)
            self.lineEditTestingData.setText(file_name)
            self.UpdateDataDescription()
            self.logger.info('Loading testing data ' + file_name + ' succeed.' )
        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            print('出错啦！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected testing data mismatch.')

    def GenerateVerboseTest(self, normalizer_name, dimension_reduction_name, feature_selector_name, classifier_name, feature_num,
                       current_num, total_num):

        def formatOneLine(thename, thelist):
            """
            A function that generates a line
            
            I am not sure whether the last ', ' is needed here.
            I am leaving it here to ensure it produces the exactly 
            same line as before.
            Do remove it in the return line if that is desired. 
            """
            name_string = "{:s} / ".format(thename)
            list_string = ", ".join([temp.GetName() for temp in thelist])
            return name_string + list_string + ", "

    
        text_list = ["Current:"]

        text_list.append(formatOneLine(normalizer_name, 
                            self.__process_normalizer_list))
        
        text_list.append(formatOneLine(dimension_reduction_name, 
                            self.__process_dimension_reduction_list))

        text_list.append(formatOneLine(feature_selector_name, 
                            self.__process_feature_selector_list))

        text_list.append("Feature Number: {:d} / [{:d}-{:d}]".format(feature_num, 
                                                                    self.spinBoxMinFeatureNumber.value(), 
                                                                    self.spinBoxMaxFeatureNumber.value()))

        text_list.append(formatOneLine(classifier_name, 
                            self.__process_classifier_list))

        text_list.append("Total process: {:d} / {:d}".format(current_num, total_num))
        return "\n".join(text_list)

    def SetStateAllButtonWhenRunning(self, state):
        self.buttonLoadTrainingData.setEnabled(state)
        self.buttonLoadTestingData.setEnabled(state)
        
        self.SetStateButtonBeforeLoading(state)

    def SetStateButtonBeforeLoading(self, state):
        self.buttonRun.setEnabled(state)
        
        self.checkNormalizeUnit.setEnabled(state)
        self.checkNormalizeZeroCenter.setEnabled(state)
        self.checkNormalizeUnitWithZeroCenter.setEnabled(state)
        self.checkNormalizationAll.setEnabled(state)
        
        self.checkPCA.setEnabled(state)
        self.checkRemoveSimilarFeatures.setEnabled(state)
        self.checkPreprocessAll.setEnabled(state)
        
        self.checkANOVA.setEnabled(state)
        self.checkRFE.setEnabled(state)
        self.checkRelief.setEnabled(state)
        # self.checkMRMR.setEnabled(state)
        self.checkFeatureSelectorAll.setEnabled(state)
        
        self.spinBoxMinFeatureNumber.setEnabled(state)
        self.spinBoxMaxFeatureNumber.setEnabled(state)
        
        self.checkSVM.setEnabled(state)
        self.checkAE.setEnabled(state)
        self.checkLDA.setEnabled(state)
        self.checkRF.setEnabled(state)
        self.checkLogisticRegression.setEnabled(state)
        self.checkLRLasso.setEnabled(state)
        self.checkAdaboost.setEnabled(state)
        self.checkDecisionTree.setEnabled(state)
        self.checkNaiveBayes.setEnabled(state)
        self.checkGaussianProcess.setEnabled(state)
        self.checkClassifierAll.setEnabled(state)
        self.checkHyperParameters.setEnabled(state)

        self.radio5folder.setEnabled(state)
        self.radio10Folder.setEnabled(state)
        self.radioLOO.setEnabled(state)

    def Run(self):
        if self.training_data_container.IsEmpty():
            QMessageBox.about(self, '', 'Training data is empty.')
            self.logger.info('Training data is empty.');
            return

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            if len(os.listdir(store_folder)) > 0:
                reply = QMessageBox.question(self, 'Continue?',
                                             'The folder is not empty, if you click Yes, the data would be clear in this folder', QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    for file in os.listdir(store_folder):
                        if os.path.isdir(os.path.join(store_folder, file)):
                            shutil.rmtree(os.path.join(store_folder, file))
                        else:
                            os.remove(os.path.join(store_folder, file))
                else:
                    return

            self.textEditVerbose.setText(store_folder)
            if self.MakePipelines():
                thread = CVRun()
                thread.moveToThread(QThread())
                thread.SetProcessConnectionAndStore_folder(self, store_folder)

                thread.signal.connect(self.textEditVerbose.setPlainText)
                thread.start()
                self.SetStateAllButtonWhenRunning(False)

                hidden_file_path = os.path.join(store_folder, '.FAEresult4129074093819729087')
                with open(hidden_file_path, 'wb') as file:
                    pass
                file_hidden = os.popen('attrib +h '+ hidden_file_path)
                file_hidden.close()

            else:
                QMessageBox.about(self, 'Pipeline Error', 'Pipeline must include Classifier and CV method')
                self.logger.error('Make pipeline failed. Pipeline must include Classfier and CV method.')

    def MinFeatureNumberChange(self):
        if self.spinBoxMinFeatureNumber.value() > self.spinBoxMaxFeatureNumber.value():
            self.spinBoxMinFeatureNumber.setValue(self.spinBoxMaxFeatureNumber.value())

        self.UpdatePipelineText()

    def MaxFeatureNumberChange(self):
        if self.spinBoxMaxFeatureNumber.value() < self.spinBoxMinFeatureNumber.value():
            self.spinBoxMaxFeatureNumber.setValue(self.spinBoxMinFeatureNumber.value())

        self.UpdatePipelineText()

    def MakePipelines(self):
        self.__process_normalizer_list = []
        if self.checkNormalizeUnit.isChecked():
            self.__process_normalizer_list.append(NormalizerUnit())
        if self.checkNormalizeZeroCenter.isChecked():
            self.__process_normalizer_list.append(NormalizerZeroCenter())
        if self.checkNormalizeUnitWithZeroCenter.isChecked():
            self.__process_normalizer_list.append(NormalizerZeroCenterAndUnit())
        if (not self.checkNormalizeUnit.isChecked()) and (not self.checkNormalizeZeroCenter.isChecked()) and \
                (not self.checkNormalizeUnitWithZeroCenter.isChecked()):
            self.__process_normalizer_list.append(NormalizerNone())

        self.__process_dimension_reduction_list = []
        if self.checkPCA.isChecked():
            self.__process_dimension_reduction_list.append(DimensionReductionByPCA())
        if self.checkRemoveSimilarFeatures.isChecked():
            self.__process_dimension_reduction_list.append(DimensionReductionByPCC())

        self.__process_feature_selector_list = []
        if self.checkANOVA.isChecked():
            self.__process_feature_selector_list.append(FeatureSelectPipeline([FeatureSelectByANOVA()]))
        if self.checkRFE.isChecked():
            self.__process_feature_selector_list.append(FeatureSelectPipeline([FeatureSelectByRFE()]))
        if self.checkRelief.isChecked():
            self.__process_feature_selector_list.append(FeatureSelectPipeline([FeatureSelectByRelief()]))
        # if self.checkMRMR.isChecked():
        #     self.__process_feature_selector_list.append(FeatureSelectPipeline([FeatureSelectByMrmr()]))

        self.__process_feature_number_list = np.arange(self.spinBoxMinFeatureNumber.value(), self.spinBoxMaxFeatureNumber.value() + 1).tolist()

        self.__process_classifier_list = []
        if self.checkSVM.isChecked():
            self.__process_classifier_list.append(SVM())
        if self.checkLDA.isChecked():
            self.__process_classifier_list.append(LDA())
        if self.checkAE.isChecked():
            self.__process_classifier_list.append(AE())
        if self.checkRF.isChecked():
            self.__process_classifier_list.append(RandomForest())
        if self.checkLogisticRegression.isChecked():
            self.__process_classifier_list.append(LR())
        if self.checkLRLasso.isChecked():
            self.__process_classifier_list.append(LRLasso())
        if self.checkAdaboost.isChecked():
            self.__process_classifier_list.append(AdaBoost())
        if self.checkDecisionTree.isChecked():
            self.__process_classifier_list.append(DecisionTree())
        if self.checkGaussianProcess.isChecked():
            self.__process_classifier_list.append(GaussianProcess())
        if self.checkNaiveBayes.isChecked():
            self.__process_classifier_list.append(NaiveBayes())
        if len(self.__process_classifier_list) == 0:
            self.logger.error('Process classifier list length is zero.')
            return False

        if self.radio5folder.isChecked():
            cv = CrossValidation5Folder()
        elif self.radio10Folder.isChecked():
            cv = CrossValidation10Folder()
        elif self.radioLOO.isChecked():
            cv = CrossValidationLeaveOneOut()
        else:
            return False

        self.fae.SetNormalizerList(self.__process_normalizer_list)
        self.fae.SetDimensionReductionList(self.__process_dimension_reduction_list)
        self.fae.SetFeatureSelectorList(self.__process_feature_selector_list)
        self.fae.SetFeatureNumberList(self.__process_feature_number_list)
        self.fae.SetClassifierList(self.__process_classifier_list)
        self.fae.SetCrossValition(cv)
        self.fae.GenerateMetircDict()

        return True

    def UpdateDataDescription(self):
        show_text = ""
        if self.training_data_container.GetArray().size > 0:
            show_text += "The number of training cases: {:d}\n".format(len(self.training_data_container.GetCaseName()))
            show_text += "The number of training features: {:d}\n".format(len(self.training_data_container.GetFeatureName()))
            if len(np.unique(self.training_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(self.training_data_container.GetLabel() == np.max(self.training_data_container.GetLabel()))[0])
                negative_number = len(self.training_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self.training_data_container.GetLabel()))
                show_text += "The number of training positive samples: {:d}\n".format(positive_number)
                show_text += "The number of training negative samples: {:d}\n".format(negative_number)

        show_text += '\n'
        if self.testing_data_container.GetArray().size > 0:
            show_text += "The number of testing cases: {:d}\n".format(len(self.testing_data_container.GetCaseName()))
            show_text += "The number of testing features: {:d}\n".format(
                len(self.testing_data_container.GetFeatureName()))
            if len(np.unique(self.testing_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(
                        self.testing_data_container.GetLabel() == np.max(self.testing_data_container.GetLabel()))[0])
                negative_number = len(self.testing_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self.testing_data_container.GetLabel()))
                show_text += "The number of testing positive samples: {:d}\n".format(positive_number)
                show_text += "The number of testing negative samples: {:d}\n".format(negative_number)

        self.textEditDescription.setText(show_text)

    def UpdatePipelineText(self):
        self.listOnePipeline.clear()

        normalization_text = 'Normalization:\n'
        normalizer_num = 0
        if self.checkNormalizeUnit.isChecked():
            normalization_text += "Normalize unit\n"
            normalizer_num += 1
        if self.checkNormalizeZeroCenter.isChecked():
            normalization_text += "Normalize zero center\n"
            normalizer_num += 1
        if self.checkNormalizeUnitWithZeroCenter.isChecked():
            normalization_text += "Normalize unit with zero center\n"
            normalizer_num += 1
        if normalizer_num == 0:
            normalizer_num = 1
        self.listOnePipeline.addItem(normalization_text)

        preprocess_test = 'Preprocess:\n'
        dimension_reduction_num = 0
        if self.checkPCA.isChecked():
            preprocess_test += "PCA\n"
            dimension_reduction_num += 1
        if self.checkRemoveSimilarFeatures.isChecked():
            preprocess_test += "Remove Similary Features\n"
            dimension_reduction_num += 1
        if dimension_reduction_num == 0:
            dimension_reduction_num = 1
        self.listOnePipeline.addItem(preprocess_test)

        feature_selection_text = "Feature Selection:\n"
        if self.spinBoxMinFeatureNumber.value() == self.spinBoxMaxFeatureNumber.value():
            feature_selection_text += "Feature Number: " + str(self.spinBoxMinFeatureNumber.value()) + "\n"
        else:
            feature_selection_text += "Feature Number range: {:d}-{:d}\n".format(self.spinBoxMinFeatureNumber.value(),
                                                                                 self.spinBoxMaxFeatureNumber.value())
        feature_num = self.spinBoxMaxFeatureNumber.value() - self.spinBoxMinFeatureNumber.value() + 1

        feature_selector_num = 0
        if self.checkANOVA.isChecked():
            feature_selection_text += "ANOVA\n"
            feature_selector_num += 1
        if self.checkRFE.isChecked():
            feature_selection_text += "RFE\n"
            feature_selector_num += 1
        if self.checkRelief.isChecked():
            feature_selection_text += "Relief\n"
            feature_selector_num += 1
        # if self.checkMRMR.isChecked():
        #     feature_selection_text += "mRMR\n"
        #     feature_selector_num += 1
        if feature_selector_num == 0:
            feature_selection_text += "None\n"
            feature_selector_num = 1
        self.listOnePipeline.addItem(feature_selection_text)


        classifier_text = 'Classifier:\n'
        classifier_num = 0
        if self.checkSVM.isChecked():
            classifier_text += "SVM\n"
            classifier_num += 1
        if self.checkLDA.isChecked():
            classifier_text += "LDA\n"
            classifier_num += 1
        if self.checkAE.isChecked():
            classifier_text += "AE\n"
            classifier_num += 1
        if self.checkRF.isChecked():
            classifier_text += "RF\n"
            classifier_num += 1
        if self.checkLogisticRegression.isChecked():
            classifier_text += "Logistic Regression\n"
            classifier_num += 1
        if self.checkLRLasso.isChecked():
            classifier_text += "Logistic Regression with Lasso\n"
            classifier_num += 1
        if self.checkAdaboost.isChecked():
            classifier_text += "Adaboost\n"
            classifier_num += 1
        if self.checkDecisionTree.isChecked():
            classifier_text += "Decision Tree\n"
            classifier_num += 1
        if self.checkGaussianProcess.isChecked():
            classifier_text += "Gaussian Process\n"
            classifier_num += 1
        if self.checkNaiveBayes.isChecked():
            classifier_text += "Naive Bayes\n"
            classifier_num += 1

        if classifier_num == 0:
            classifier_num = 1
        self.listOnePipeline.addItem(classifier_text)

        cv_method = "Cross Validation:\n"
        if self.radio5folder.isChecked():
            cv_method += "5-Folder\n"
        elif self.radio10Folder.isChecked():
            cv_method += "10-folder\n"
        elif self.radioLOO.isChecked():
            cv_method += "LeaveOneOut\n"

        self.listOnePipeline.addItem(cv_method)

        self.listOnePipeline.addItem("Total number of pipelines is:\n{:d}"
                                     .format(normalizer_num * dimension_reduction_num * feature_selector_num * feature_num * classifier_num))

    def SelectAllNormalization(self):
        if self.checkNormalizationAll.isChecked():
            self.checkNormalizeZeroCenter.setChecked(True)
            self.checkNormalizeUnitWithZeroCenter.setChecked(True)
            self.checkNormalizeUnit.setChecked(True)
        else:
            self.checkNormalizeZeroCenter.setChecked(False)
            self.checkNormalizeUnitWithZeroCenter.setChecked(False)
            self.checkNormalizeUnit.setChecked(False)

        self.UpdatePipelineText()

    def SelectAllPreprocess(self):
        if self.checkPreprocessAll.isChecked():
            self.checkPCA.setChecked(True)
            self.checkRemoveSimilarFeatures.setChecked(True)
        else:
            self.checkPCA.setChecked(False)
            self.checkRemoveSimilarFeatures.setChecked(False)

        self.UpdatePipelineText()

    def SelectAllFeatureSelector(self):
        if self.checkFeatureSelectorAll.isChecked():
            self.checkANOVA.setChecked(True)
            self.checkRFE.setChecked(True)
            self.checkRelief.setChecked(True)
            # self.checkMRMR.setChecked(True)
        else:
            self.checkANOVA.setChecked(False)
            self.checkRFE.setChecked(False)
            self.checkRelief.setChecked(False)
            # self.checkMRMR.setChecked(False)

        self.UpdatePipelineText()

    def SelectAllClassifier(self):
        if self.checkClassifierAll.isChecked():
            self.checkSVM.setChecked(True)
            self.checkAE.setChecked(True)
            self.checkLDA.setChecked(True)
            self.checkRF.setChecked(True)
            self.checkLogisticRegression.setChecked(True)
            self.checkLRLasso.setChecked(True)
            self.checkAdaboost.setChecked(True)
            self.checkDecisionTree.setChecked(True)
            self.checkGaussianProcess.setChecked(True)
            self.checkNaiveBayes.setChecked(True)
        else:
            self.checkSVM.setChecked(False)
            self.checkAE.setChecked(False)
            self.checkLDA.setChecked(False)
            self.checkRF.setChecked(False)
            self.checkLogisticRegression.setChecked(False)
            self.checkLRLasso.setChecked(False)
            self.checkAdaboost.setChecked(False)
            self.checkDecisionTree.setChecked(False)
            self.checkGaussianProcess.setChecked(False)
            self.checkNaiveBayes.setChecked(False)

        self.UpdatePipelineText()
