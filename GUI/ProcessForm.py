"""
All rights reserved.
--Yang Song
"""
import sys
import shutil

from PyQt5.QtWidgets import *
from GUI.Process import Ui_Process
from PyQt5.QtCore import *
from Utility.EcLog import eclog
from FAE.FeatureAnalysis.DataBalance import *
from FAE.FeatureAnalysis.Normalizer import *
from FAE.FeatureAnalysis.DimensionReduction import *
from FAE.FeatureAnalysis.FeatureSelector import *
from FAE.FeatureAnalysis.Classifier import *
from FAE.FeatureAnalysis.Pipelines import PipelinesManager
from FAE.FeatureAnalysis.CrossValidation import *


class CVRun(QThread):
    signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        pass

    def SetProcessConnectionAndStore_folder(self, process_connection, store_folder):
        self._process_connection = process_connection
        self.store_folder = store_folder

    def run(self):
        try:
            for total, num in self._process_connection.fae.RunWithoutCV(self._process_connection.training_data_container,
                                                                        self._process_connection.testing_data_container,
                                                                        self.store_folder):
                text = "Model Developing:\n{} / {}".format(num, total)
                self.signal.emit(text)
        except Exception as e:
            print('Thread RunWithoutCV Failed: ', e.__str__())
            raise Exception

        for total, num, group in self._process_connection.fae.RunWithCV(
                self._process_connection.training_data_container,
                self.store_folder):
            text = "Model Developing:\nDone.\n\n" \
                   "Cross Validation:\nGroup {}: {} / {}".format(int(group) + 1, num, total)
            self.signal.emit(text)

        for total, num in self._process_connection.fae.MergeCvResult(self.store_folder):
            text = "Model Developing:\nDone.\n\nCross Validation:\nDone.\n\n" \
                   "Merging CV Results:\n{} / {}".format(num, total)
            self.signal.emit(text)

        self._process_connection.fae.SaveAucDict(self.store_folder)

        text = "Model Developing:\nDone.\n\nCross Validation:\nDone.\n\nMerging CV Results:\nDone.\n"
        self.signal.emit(text)
        self._process_connection.SetStateAllButtonWhenRunning(True)


class ProcessConnection(QWidget, Ui_Process):
    close_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        self.training_data_container = DataContainer()
        self.testing_data_container = DataContainer()
        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()
        self.fae = PipelinesManager(logger=self.logger)
        self.__normalizers = []
        self.__dimension_reductions = []
        self.__feature_selectors = []
        self.__process_feature_number_list = []
        self.__classifiers = []

        self.thread = CVRun()

        super(ProcessConnection, self).__init__(parent)
        self.setupUi(self)

        self.buttonLoadTrainingData.clicked.connect(self.LoadTrainingData)
        self.buttonLoadTestingData.clicked.connect(self.LoadTestingData)

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

    def closeEvent(self, close_event):
        self.close_signal.emit(True)
        close_event.accept()

    def LoadTrainingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=r'C:\MyCode\FAE\Example',
                                           filter="csv files (*.csv)")
        if file_name:
            try:
                self.training_data_container.Load(file_name)
                self.SetStateButtonBeforeLoading(True)
                self.lineEditTrainingData.setText(file_name)
                self.UpdateDataDescription()
                self.logger.info('Open CSV file ' + file_name + ' succeed.')
                self.spinBoxMaxFeatureNumber.setValue(len(self.training_data_container.GetFeatureName()))
            except OSError as reason:
                self.logger.log('Open SCV file Error, The reason is ' + str(reason))
                print('Load Error！' + str(reason))
            except ValueError:
                self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
                QMessageBox.information(self, 'Error',
                                        'The selected training data mismatch.')

    def LoadTestingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', filter="csv files (*.csv)")
        if file_name:
            try:
                self.testing_data_container.Load(file_name)
                self.lineEditTestingData.setText(file_name)
                self.UpdateDataDescription()
                self.logger.info('Loading testing data ' + file_name + ' succeed.')
            except OSError as reason:
                self.logger.log('Open CSV file Error, The reason is ' + str(reason))
                print('ERROR！' + str(reason))
            except ValueError:
                self.logger.error('Open CSV file ' + file_name + ' Failed. Value error.')
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

        text_list = ["Current:",
                     FormatOneLine(normalizer_name, self.__normalizers),
                     FormatOneLine(dimension_reduction_name, self.__dimension_reductions),
                     FormatOneLine(feature_selector_name, self.__feature_selectors),
                     "Feature Number: {:d} / [{:d}-{:d}]".format(feature_num,
                                                                 self.spinBoxMinFeatureNumber.value(),
                                                                 self.spinBoxMaxFeatureNumber.value()),
                     FormatOneLine(classifier_name, self.__classifiers),
                     "Total process: {:d} / {:d}".format(current_num, total_num)]

        return "\n".join(text_list)

    def SetStateAllButtonWhenRunning(self, state):
        if state:
            self.thread.exit()
        self.buttonLoadTrainingData.setEnabled(state)
        self.buttonLoadTestingData.setEnabled(state)

        self.SetStateButtonBeforeLoading(state)

    def SetStateButtonBeforeLoading(self, state):
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
                                             "Folder not empty. If continue, all data in folder will be cleared.",
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
                # self.thread = CVRun()
                self.thread.moveToThread(QThread())
                self.thread.SetProcessConnectionAndStore_folder(self, store_folder)

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

    def MaxFeatureNumberChange(self):
        if self.spinBoxMaxFeatureNumber.value() < self.spinBoxMinFeatureNumber.value():
            self.spinBoxMaxFeatureNumber.setValue(self.spinBoxMinFeatureNumber.value())

        self.UpdatePipelineText()

    def MakePipelines(self):
        if self.radioNoneBalance.isChecked():
            data_balance = NoneBalance()
        elif self.radioDownSampling.isChecked():
            data_balance = DownSampling()
        elif self.radioUpSampling.isChecked():
            data_balance = UpSampling()
        elif self.radioSmote.isChecked():
            data_balance = SmoteSampling()
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

        self.__dimension_reductions = []
        if self.checkPCA.isChecked():
            self.__dimension_reductions.append(DimensionReductionByPCA())
        if self.checkRemoveSimilarFeatures.isChecked():
            self.__dimension_reductions.append(DimensionReductionByPCC())

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
        #     self.__feature_selectors.append(FeatureSelectByMrmr())

        self.__process_feature_number_list = np.arange(self.spinBoxMinFeatureNumber.value(),
                                                       self.spinBoxMaxFeatureNumber.value() + 1).tolist()

        self.__classifiers = []
        if self.checkSVM.isChecked():
            self.__classifiers.append(SVM())
        if self.checkLDA.isChecked():
            self.__classifiers.append(LDA())
        if self.checkAE.isChecked():
            self.__classifiers.append(AE())
        if self.checkRF.isChecked():
            self.__classifiers.append(RandomForest())
        if self.checkLogisticRegression.isChecked():
            self.__classifiers.append(LR())
        if self.checkLRLasso.isChecked():
            self.__classifiers.append(LRLasso())
        if self.checkAdaboost.isChecked():
            self.__classifiers.append(AdaBoost())
        if self.checkDecisionTree.isChecked():
            self.__classifiers.append(DecisionTree())
        if self.checkGaussianProcess.isChecked():
            self.__classifiers.append(GaussianProcess())
        if self.checkNaiveBayes.isChecked():
            self.__classifiers.append(NaiveBayes())
        if len(self.__classifiers) == 0:
            self.logger.error('Process classifier list length is zero.')
            return False

        if self.radio5folder.isChecked():
            cv = CrossValidation5Fold
        elif self.radio10Folder.isChecked():
            cv = CrossValidation10Fold
        elif self.radioLOO.isChecked():
            cv = CrossValidationLOO
        else:
            return False

        self.fae.balance = data_balance
        self.fae.normalizer_list = self.__normalizers
        self.fae.dimension_reduction_list = self.__dimension_reductions
        self.fae.feature_selector_list = self.__feature_selectors
        self.fae.feature_selector_num_list = self.__process_feature_number_list
        self.fae.classifier_list = self.__classifiers
        self.fae.cv = cv
        self.fae.GenerateAucDict()

        return True

    @staticmethod
    def get_dataset_description(dataset, dataset_name):
        """
        :param dataset: dataset to describe
        :param dataset_name: can be one of "training", "test"
        :return: description string for the specified dataset
        """
        if dataset.GetArray().size == 0:
            return ""
        
        format_string = "Cases in " + dataset_name + ": {:d}\n"

        show_text = format_string.format(len(dataset.GetCaseName()))
        show_text += ("Features in " + dataset_name + " dataset: {:d}\n").format(
            len(dataset.GetFeatureName()))
        if len(np.unique(dataset.GetLabel())) == 2:
            positive_number = len(np.where(dataset.GetLabel() == np.max(dataset.GetLabel()))[0])
            negative_number = len(dataset.GetLabel()) - positive_number
            show_text += ("Positive cases in " + dataset_name + " dataset: {:d}\n").format(positive_number)
            show_text += ("Negative cases in " + dataset_name + " dataset: {:d}\n").format(negative_number)
        return show_text


    def UpdateDataDescription(self):
        show_text = self.get_dataset_description(self.training_data_container, "training")
        if show_text:
            show_text += '\n'
            
        show_text += self.get_dataset_description(self.testing_data_container, "test")

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
            preprocess_test += "Pearson Correlation (0.99)\n"
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

        self.listOnePipeline.addItem("Total number of pipelines is:\n{:d}".format(
            normalizer_num * dimension_reduction_num * feature_selector_num * feature_num * classifier_num))

    def SelectAllNormalization(self):
        checked = self.checkNormalizationAll.isChecked()
        self.checkNormalizeNone.setChecked(checked)
        self.checkNormalizeMinMax.setChecked(checked)
        self.checkNormalizeZscore.setChecked(checked)
        self.checkNormalizeMean.setChecked(checked)

        self.UpdatePipelineText()

    def SelectAllPreprocess(self):
        checked = self.checkPreprocessAll.isChecked()
        self.checkPCA.setChecked(checked)
        self.checkRemoveSimilarFeatures.setChecked(checked)

        self.UpdatePipelineText()

    def SelectAllFeatureSelector(self):
        checked = self.checkFeatureSelectorAll.isChecked()
        self.checkANOVA.setChecked(checked)
        self.checkKW.setChecked(checked)
        self.checkRFE.setChecked(checked)
        self.checkRelief.setChecked(checked)
        # self.checkMRMR.setChecked(checked)

        self.UpdatePipelineText()

    def SelectAllClassifier(self):
        checked = self.checkClassifierAll.isChecked()
        self.checkSVM.setChecked(checked)
        self.checkAE.setChecked(checked)
        self.checkLDA.setChecked(checked)
        self.checkRF.setChecked(checked)
        self.checkLogisticRegression.setChecked(checked)
        self.checkLRLasso.setChecked(checked)
        self.checkAdaboost.setChecked(checked)
        self.checkDecisionTree.setChecked(checked)
        self.checkGaussianProcess.setChecked(checked)
        self.checkNaiveBayes.setChecked(checked)

        self.UpdatePipelineText()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_frame = ProcessConnection()
    main_frame.show()
    sys.exit(app.exec_())