"""
All rights reserved.
--Yang Song
"""
import sys
import shutil
import traceback

from PyQt5.QtWidgets import *
from sympy import hyper

from BC.GUI.Process import Ui_Process
from PyQt5.QtCore import *
from BC.FeatureAnalysis.DataBalance import *
from BC.FeatureAnalysis.Normalizer import *
from BC.FeatureAnalysis.DimensionReduction import *
from BC.FeatureAnalysis.FeatureSelector import *
from BC.FeatureAnalysis.Classifier import *
from BC.FeatureAnalysis.Pipelines import PipelinesManager
from BC.FeatureAnalysis.CrossValidation import *

from BC.FeatureAnalysis.IndexDict import Index2Dict
from BC.HyperParameterConfig.HyperParamManager import GetClassifierHyperParams


class CVRun(QThread):
    signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        pass

    def SetProcessConnectionAndStore_folder(self, process_connection, store_folder, is_train_cutoff=False):
        self.process_connection = process_connection
        self.store_folder = store_folder
        self.is_train_cutoff = is_train_cutoff

    def run(self):
        self.signal.emit('Start Building')
        valid_features = RemoveSameFeatures().Run(self.process_connection.training_data_container).GetFeatureName()
        try:
            train_dc = FeatureSelector().SelectFeatureByName(
                self.process_connection.training_data_container, valid_features
            )
        except Exception as e:
            print('Selecting valid features wrong')
            print(traceback.format_exc())
            raise Exception

        if not self.process_connection.testing_data_container.IsEmpty():
            test_dc = FeatureSelector().SelectFeatureByName(
                self.process_connection.testing_data_container, valid_features)
        else:
            test_dc = DataContainer()

        for total, num in self.process_connection.pipeline_manager.Run(
            train_dc, test_dc, self.store_folder, self.is_train_cutoff
        ):
            text = "Model Building: {} / {}".format(num, total)
            self.signal.emit(text)
        self.signal.emit("Model Building: Done")

        # for total, num, group in self.process_connection.pipeline_manager.RunWithCV(
        #         train_dc,
        #         self.store_folder):
        #     text = "Cross Validation:\nGroup {}: {} / {}...".format(int(group) + 1, num, total)
        #     self.signal.emit(text)
        # self.signal.emit("Cross Validation: Done.\n\n")
        #
        # if not self.process_connection.testing_data_container.IsEmpty():
        #     test_dc = FeatureSelector().SelectFeatureByName(
        #         self.process_connection.testing_data_container, valid_features)
        # else:
        #     test_dc = DataContainer()
        # for total, num in self.process_connection.pipeline_manager.RunWithoutCV(train_dc, test_dc,
        #                                                                         self.store_folder,
        #                                                                         self.is_train_cutoff):
        #     self.signal.emit("Cross Validation: Done. \n\n Model Developing:\n{} / {}...".format(num, total))
        # self.signal.emit("Cross Validation: Done.\n\nModel Developing: Done")

        # for total, num in self.process_connection.pipeline_manager.MergeCvResult(self.store_folder):
        #     text = "Model Building: Done\n\nMerging Results:\n{} / {}...".format(num, total)
        #     self.signal.emit(text)

        self.process_connection.pipeline_manager.SaveAucDict(self.store_folder)

        text = "Model Building: Done\n\nMerging Results:\nDone.\n\nAll Done, please check the result in Visualization."
        self.signal.emit(text)
        self.process_connection.SetStateAllButtonWhenRunning(True)


class ProcessConnection(QWidget, Ui_Process):
    close_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        self.training_data_container = DataContainer()
        self.testing_data_container = DataContainer()
        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()
        self.pipeline_manager = PipelinesManager(logger=self.logger)
        self.__normalizers = []
        self.__dimension_reducers = []
        self.__feature_selectors = []
        self.__feature_number_list = []
        self.__classifiers = []

        self.index2dict = Index2Dict()
        self.thread = CVRun()

        super(ProcessConnection, self).__init__(parent)
        self.setupUi(self)
        self.pcccoefdoubleSpinBox.setValue(self.getDefaultValue())
        self.pcccoefdoubleSpinBox.valueChanged.connect(self.pccValueChange)
        self.buttonLoadTrainingData.clicked.connect(self.LoadTrainingData)
        self.buttonLoadTestingData.clicked.connect(self.LoadTestingData)

        self.buttonDefaultParam.clicked.connect(self.SetDefaultParam)

        self.radioNoneBalance.clicked.connect(self.UpdatePipelineText)
        self.radioDownSampling.clicked.connect(self.UpdatePipelineText)
        self.radioUpSampling.clicked.connect(self.UpdatePipelineText)
        self.radioSmote.clicked.connect(self.UpdatePipelineText)

        self.checkNormalizeNone.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeMinMax.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeZscore.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeMean.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizationAll.clicked.connect(self.SelectAllNormalization)

        self.checkPCA.clicked.connect(self.UpdatePipelineText)
        self.checkRemoveSimilarFeatures.clicked.connect(self.UpdatePipelineText)
        self.checkPreprocessAll.clicked.connect(self.SelectAllPreprocess)

        self.spinBoxMinFeatureNumber.valueChanged.connect(self.MinFeatureNumberChange)
        self.spinBoxMaxFeatureNumber.valueChanged.connect(self.MaxFeatureNumberChange)

        self.checkANOVA.clicked.connect(self.UpdatePipelineText)
        self.checkKW.clicked.connect(self.UpdatePipelineText)
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

    def getDefaultValue(self):
        path = os.getcwd() + '/fae_Settings.ini'
        settings = QSettings(path)
        pcc_coef = 0.99
        if settings.value('PCC_Coef'):
            pcc_coef = float(settings.value('PCC_Coef'))
        return pcc_coef

    def saveDefaultValue(self):
        path = os.getcwd() + '/fae_Settings.ini'
        settings = QSettings(path)
        settings.setValue('PCC_Coef', self.pcccoefdoubleSpinBox.value())
        settings.sync()

    def closeEvent(self, QCloseEvent):
        self.close_signal.emit(True)
        QCloseEvent.accept()

    def LoadTrainingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=r'C:\MyCode\FAE\Example',
                                           filter="csv files (*.csv)")
        if file_name:
            try:
                self.training_data_container.Load(file_name)
                if self.training_data_container.GetLabel().max() != 1:
                    QMessageBox.warning(self, 'Error', 'The Label only should be 0 and 1.')
                    self.training_data_container.Clear()
                    self.lineEditTrainingData.setText(file_name)
                    self.UpdateDataDescription()
                    return
                if (self.training_data_container.GetLabel() == 0).astype(int).sum() < 10 or \
                    (self.training_data_container.GetLabel() == 1).astype(int).sum() < 10:
                    QMessageBox.warning(self, 'Error', 'The cases of each group should be large enough.')
                    self.training_data_container.Clear()
                    self.lineEditTrainingData.setText(file_name)
                    self.UpdateDataDescription()
                    return

                self.SetStateButtonBeforeLoading(True)
                self.lineEditTrainingData.setText(file_name)
                self.UpdateDataDescription()
                self.logger.info('Open CSV file ' + file_name + ' succeed.')
                self.spinBoxMaxFeatureNumber.setValue(len(self.training_data_container.GetFeatureName()))
                self.SetDefaultParam()
            except OSError as reason:
                error_message = 'Error opening CSV file. The reason is ' + str(reason)
                self.logger.log(error_message)
                QMessageBox.warning(self, 'Error', error_message)
            except ValueError:
                error_message = 'Error opening CSV file ' + file_name + ' because of invalid values in the file.'
                self.logger.error(error_message)
                QMessageBox.warning(self, 'Error', error_message)

    def LoadTestingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', filter="csv files (*.csv)")
        if file_name:
            try:
                self.testing_data_container.Load(file_name)
                if self.testing_data_container.GetLabel().max() != 1:
                    QMessageBox.warning(self, 'Error', 'The Label only should be 0 and 1.')
                    self.testing_data_container.Clear()
                    self.lineEditTestingData.setText(file_name)
                    self.UpdateDataDescription()
                    return
                if self.training_data_container.IsEmpty():
                    QMessageBox.warning(self, 'Error', 'Load training data set first.')
                    self.testing_data_container.Clear()
                    self.lineEditTestingData.setText(file_name)
                    self.UpdateDataDescription()
                    return
                if self.training_data_container.GetFeatureName() != self.testing_data_container.GetFeatureName():
                    QMessageBox.warning(self, 'Error', 'Test has different features from Train.')
                    self.testing_data_container.Clear()
                    self.lineEditTestingData.setText(file_name)
                    self.UpdateDataDescription()
                    return

                self.lineEditTestingData.setText(file_name)
                self.UpdateDataDescription()
                self.logger.info('Loading testing data ' + file_name + ' succeed.')
            except OSError as reason:
                self.logger.log('Open SCV file Error, The reason is ' + str(reason))
                print('ERROR！' + str(reason))
            except ValueError:
                self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
                QMessageBox.information(self, 'Error',
                                        'The selected testing data mismatch.')

    def GenerateVerboseTest(self, normalizer_name, dimension_reduction_name, feature_selector_name, classifier_name,
                            feature_num,
                            current_num, total_num):

        def FormatOneLine(the_name, the_list):
            """
            A function that generates a line
            
            I am not sure whether the last ', ' is needed here.
            I am leaving it here to ensure it produces the exactly 
            same line as before.
            Do remove it in the return line if that is desired. 
            """
            name_string = "{:s} / ".format(the_name)
            list_string = ", ".join([temp.GetName() for temp in the_list])
            return name_string + list_string + ", "

        text_list = ["Current:"]

        text_list.append(FormatOneLine(normalizer_name,
                                       self.__process_normalizer_list))

        text_list.append(FormatOneLine(dimension_reduction_name,
                                       self.__dimension_reducers))

        text_list.append(FormatOneLine(feature_selector_name,
                                       self.__feature_selectors))

        text_list.append("Feature Number: {:d} / [{:d}-{:d}]".format(feature_num,
                                                                     self.spinBoxMinFeatureNumber.value(),
                                                                     self.spinBoxMaxFeatureNumber.value()))

        text_list.append(FormatOneLine(classifier_name,
                                       self.__classifiers))

        text_list.append("Total process: {:d} / {:d}".format(current_num, total_num))
        return "\n".join(text_list)

    def SetDefaultParam(self):
        self.radioUpSampling.setChecked(True)

        self.checkNormalizationAll.setChecked(False)
        self.checkNormalizeNone.setChecked(False)
        self.checkNormalizeMean.setChecked(True)
        self.checkNormalizeMinMax.setChecked(False)
        self.checkNormalizeZscore.setChecked(True)

        self.checkPCA.setChecked(False)
        self.checkRemoveSimilarFeatures.setChecked(True)
        self.pcccoefdoubleSpinBox.setValue(0.99)

        self.spinBoxMinFeatureNumber.setValue(1)
        self.spinBoxMaxFeatureNumber.setValue(min(min(len(self.training_data_container.GetCaseName()) // 10 * 3, 
                                                  len(self.training_data_container.GetFeatureName())), 
                                                  30)
            )
        
        self.checkANOVA.setChecked(True)
        self.checkKW.setChecked(True)
        self.checkRFE.setChecked(True)
        self.checkRelief.setChecked(False)

        self.checkSVM.setChecked(True)        
        self.checkLogisticRegression.setChecked(True)
        self.checkLDA.setChecked(True)
        self.checkDecisionTree.setChecked(True)
        self.checkGaussianProcess.setChecked(False)
        self.checkLRLasso.setChecked(False)
        self.checkAE.setChecked(False)
        self.checkRF.setChecked(False)
        self.checkGaussianProcess.setChecked(False)
        self.checkNaiveBayes.setChecked(False)

        self.radio5folder.setChecked(True)

        self.UpdatePipelineText()

    def SetStateAllButtonWhenRunning(self, state):
        if state:
            self.thread.exit()
        self.buttonLoadTrainingData.setEnabled(state)
        self.buttonLoadTestingData.setEnabled(state)

        self.SetStateButtonBeforeLoading(state)

    def SetStateButtonBeforeLoading(self, state):
        self.buttonDefaultParam.setEnabled(state)
        self.buttonRun.setEnabled(state)

        self.radioNoneBalance.setEnabled(state)
        self.radioUpSampling.setEnabled(state)
        self.radioDownSampling.setEnabled(state)
        self.radioSmote.setEnabled(state)

        self.checkNormalizeNone.setEnabled(state)
        self.checkNormalizeMinMax.setEnabled(state)
        self.checkNormalizeZscore.setEnabled(state)
        self.checkNormalizeMean.setEnabled(state)
        self.checkNormalizationAll.setEnabled(state)

        self.checkPCA.setEnabled(state)
        self.checkRemoveSimilarFeatures.setEnabled(state)
        self.checkPreprocessAll.setEnabled(state)

        self.checkANOVA.setEnabled(state)
        self.checkKW.setEnabled(state)
        self.checkRFE.setEnabled(state)
        self.checkRelief.setEnabled(state)
        # self.checkMRMR.setEnabled(state)
        self.checkFeatureSelectorAll.setEnabled(state)

        self.spinBoxMinFeatureNumber.setEnabled(state)
        self.spinBoxMaxFeatureNumber.setEnabled(state)
        self.pcccoefdoubleSpinBox.setEnabled(state)

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
            self.logger.info('Training data is empty.')
            return

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            if len(os.listdir(store_folder)) > 0:
                reply = QMessageBox.question(self, 'Continue?',
                                             'The folder is not empty, if you click Yes, the data would be clear in this folder',
                                             QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    try:
                        for file in os.listdir(store_folder):
                            if os.path.isdir(os.path.join(store_folder, file)):
                                shutil.rmtree(os.path.join(store_folder, file))
                            else:
                                os.remove(os.path.join(store_folder, file))
                    except PermissionError:
                        QMessageBox().about(self, 'Warning', 'Is there any opened files?')
                        return
                    except OSError:
                        QMessageBox().about(self, 'Warning', 'Is there any opened files?')
                        return
                else:
                    return

            if self.MakePipelines():
                self.thread.moveToThread(QThread())
                self.thread.SetProcessConnectionAndStore_folder(self, store_folder, self.checkEstimatebyTraining.isChecked())

                self.thread.signal.connect(self.textEditVerbose.setPlainText)
                self.thread.start()
                self.SetStateAllButtonWhenRunning(False)
            else:
                QMessageBox.about(self, 'Pipeline Error', 'Pipeline must include Classifier and CV method')
                self.logger.error('Make pipeline failed. Pipeline must include Classfier and CV method.')

    def MinFeatureNumberChange(self):
        if self.spinBoxMinFeatureNumber.value() > self.spinBoxMaxFeatureNumber.value():
            self.spinBoxMinFeatureNumber.setValue(self.spinBoxMaxFeatureNumber.value())

        self.UpdatePipelineText()

    def pccValueChange(self):
        if self.pcccoefdoubleSpinBox.value() > 1.0:
            self.pcccoefdoubleSpinBox.setValue(1.0)

        self.UpdatePipelineText()

    def MaxFeatureNumberChange(self):
        if self.spinBoxMaxFeatureNumber.value() < self.spinBoxMinFeatureNumber.value():
            self.spinBoxMaxFeatureNumber.setValue(self.spinBoxMinFeatureNumber.value())

        self.UpdatePipelineText()

    def MakePipelines(self):
        if self.radioNoneBalance.isChecked():
            data_balance = self.index2dict.GetInstantByIndex(BALANCE_NONE)
        elif self.radioDownSampling.isChecked():
            data_balance = self.index2dict.GetInstantByIndex(BALANCE_DOWN_SAMPLING)
        elif self.radioUpSampling.isChecked():
            data_balance = self.index2dict.GetInstantByIndex(BALANCE_UP_SAMPLING)
        elif self.radioSmote.isChecked():
            data_balance = self.index2dict.GetInstantByIndex(BALANCE_SMOTE)
        else:
            return False

        self.__normalizers = []
        if self.checkNormalizeNone.isChecked():
            self.__normalizers.append(NormalizerNone)
        if self.checkNormalizeMinMax.isChecked():
            self.__normalizers.append(NormalizerMinMax)
        if self.checkNormalizeZscore.isChecked():
            self.__normalizers.append(NormalizerZscore)
        if self.checkNormalizeMean.isChecked():
            self.__normalizers.append(NormalizerMean)

        self.__dimension_reducers = []
        if self.checkPCA.isChecked():
            self.__dimension_reducers.append(DimensionReductionByPCA())
        if self.checkRemoveSimilarFeatures.isChecked():
            self.__dimension_reducers.append(DimensionReductionByPCC(self.pcccoefdoubleSpinBox.value()))
        self.saveDefaultValue()

        self.__feature_selectors = []
        if self.checkANOVA.isChecked():
            self.__feature_selectors.append(FeatureSelectByANOVA())
        if self.checkKW.isChecked():
            self.__feature_selectors.append(FeatureSelectByKruskalWallis())
        if self.checkRFE.isChecked():
            self.__feature_selectors.append(FeatureSelectByRFE())
        if self.checkRelief.isChecked():
            self.__feature_selectors.append(FeatureSelectByRelief())
        # if self.checkMRMR.isChecked():
        #     self.__process_feature_selector_list.append(eatureSelectByMrmr())

        self.__feature_number_list = np.arange(self.spinBoxMinFeatureNumber.value(),
                                                       self.spinBoxMaxFeatureNumber.value() + 1).tolist()

        self.__classifiers = []
        if self.checkSVM.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_SVM))
        if self.checkLDA.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_LDA))
        if self.checkAE.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_AE))
        if self.checkRF.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_RF))
        if self.checkLogisticRegression.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_LR))
        if self.checkLRLasso.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_LRLasso))
        if self.checkAdaboost.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_AB))
        if self.checkDecisionTree.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_DT))
        if self.checkGaussianProcess.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_GP))
        if self.checkNaiveBayes.isChecked():
            self.__classifiers.append(self.index2dict.GetInstantByIndex(CLASSIFIER_NB))
        if len(self.__classifiers) == 0:
            self.logger.error('Process classifier list length is zero.')
            return False

        if self.checkHyperParameters.isChecked():
            hyper_param = GetClassifierHyperParams()
        else:
            hyper_param = {}

        if self.radio5folder.isChecked():
            cv = CrossValidation5Fold
            cv_part = 5
        elif self.radio10Folder.isChecked():
            cv = CrossValidation10Fold
            cv_part = 10
        elif self.radioLOO.isChecked():
            cv = CrossValidationLOO
            cv_part = 2
        else:
            return False

        self.pipeline_manager.balance = data_balance
        self.pipeline_manager.normalizer_list = self.__normalizers
        self.pipeline_manager.dimension_reduction_list = self.__dimension_reducers
        self.pipeline_manager.feature_selector_list = self.__feature_selectors
        self.pipeline_manager.feature_selector_num_list = self.__feature_number_list
        self.pipeline_manager.classifier_list = self.__classifiers
        self.pipeline_manager.cv = cv
        self.pipeline_manager.cv_part = cv_part
        self.pipeline_manager.hyper_param = hyper_param
        self.pipeline_manager.GenerateAucDict()

        return True

    def UpdateDataDescription(self):
        show_text = ""
        if self.training_data_container.GetArray().size > 0:
            show_text += f"Training dataset:\nNumber of cases: {len(self.training_data_container.GetCaseName())}\n"
            show_text += f"Number of features: {len(self.training_data_container.GetFeatureName())}\n"
            if len(np.unique(self.training_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(
                        self.training_data_container.GetLabel() == np.max(self.training_data_container.GetLabel()))[0])
                negative_number = len(self.training_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self.training_data_container.GetLabel()))
                show_text += f"Positive: {positive_number}\n"
                show_text += f"Negative: {negative_number}\n"

        show_text += '\n'
        if self.testing_data_container.GetArray().size > 0:
            show_text += f"Test dataset:\nNumber of cases: {len(self.testing_data_container.GetCaseName())}\n"
            show_text += f"Number of features: {len(self.testing_data_container.GetFeatureName())}\n"                
            if len(np.unique(self.testing_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(
                        self.testing_data_container.GetLabel() == np.max(self.testing_data_container.GetLabel()))[0])
                negative_number = len(self.testing_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self.testing_data_container.GetLabel()))
                show_text += f"Positive: {positive_number}\n"
                show_text += f"Negative: {negative_number}\n"

        self.textEditDescription.setText(show_text)

    def UpdatePipelineText(self):
        self.listOnePipeline.clear()

        normalization_text = 'Normalization:\n'
        normalizer_num = 0
        if self.checkNormalizeNone.isChecked():
            normalization_text += "Normalize None\n"
            normalizer_num += 1
        if self.checkNormalizeMinMax.isChecked():
            normalization_text += "Normalize Min-Max\n"
            normalizer_num += 1
        if self.checkNormalizeZscore.isChecked():
            normalization_text += "Normalize Z-score\n"
            normalizer_num += 1
        if self.checkNormalizeMean.isChecked():
            normalization_text += "Normalize Mean\n"
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
            preprocess_test += "Pearson Correlation ({:.2f})\n".format(self.pcccoefdoubleSpinBox.value())
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
        if self.checkKW.isChecked():
            feature_selection_text += "KW\n"
            feature_selector_num += 1
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
            classifier_text += "LASSO\n"
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
            cv_method += "5-Fold\n"
        elif self.radio10Folder.isChecked():
            cv_method += "10-fold\n"
        elif self.radioLOO.isChecked():
            cv_method += "LeaveOneOut\n"

        self.listOnePipeline.addItem(cv_method)

        self.listOnePipeline.addItem("Total number of pipelines is:\n{:d}"
                                     .format(
            normalizer_num * dimension_reduction_num * feature_selector_num * feature_num * classifier_num))

    def SelectAllNormalization(self):
        if self.checkNormalizationAll.isChecked():
            self.checkNormalizeNone.setChecked(True)
            self.checkNormalizeMinMax.setChecked(True)
            self.checkNormalizeZscore.setChecked(True)
            self.checkNormalizeMean.setChecked(True)
        else:
            self.checkNormalizeNone.setChecked(False)
            self.checkNormalizeMinMax.setChecked(False)
            self.checkNormalizeZscore.setChecked(False)
            self.checkNormalizeMean.setChecked(False)

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
            self.checkKW.setChecked(True)
            self.checkRFE.setChecked(True)
            self.checkRelief.setChecked(True)
            # self.checkMRMR.setChecked(True)
        else:
            self.checkANOVA.setChecked(False)
            self.checkKW.setChecked(False)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_frame = ProcessConnection()
    main_frame.show()
    sys.exit(app.exec_())