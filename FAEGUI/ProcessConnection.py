
import numpy as np
from copy import deepcopy

from PyQt5.QtWidgets import *
from GUI.Process import Ui_Process

from FAE.FeatureAnalysis.Normalizer import *
from FAE.FeatureAnalysis.DimensionReduction import *
from FAE.FeatureAnalysis.FeatureSelector import *
from FAE.FeatureAnalysis.Classifier import *
from FAE.FeatureAnalysis.FeaturePipeline import FeatureAnalysisPipelines
from FAE.FeatureAnalysis.CrossValidation import CrossValidation

class ProcessConnection(QWidget, Ui_Process):
    def __init__(self, parent=None):
        self.__training_data_container = DataContainer()
        self.__testing_data_container = DataContainer()
        self.__fae = FeatureAnalysisPipelines()

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

        self.checkPCA.clicked.connect(self.UpdatePipelineText)
        self.checkRemoveSimilarFeatures.clicked.connect(self.UpdatePipelineText)

        self.spinBoxMinFeatureNumber.valueChanged.connect(self.MinFeatureNumberChange)
        self.spinBoxMaxFeatureNumber.valueChanged.connect(self.MaxFeatureNumberChange)

        self.checkANOVA.clicked.connect(self.UpdatePipelineText)
        self.checkRFE.clicked.connect(self.UpdatePipelineText)
        self.checkRelief.clicked.connect(self.UpdatePipelineText)

        self.checkSVM.clicked.connect(self.UpdatePipelineText)
        self.checkLDA.clicked.connect(self.UpdatePipelineText)
        self.checkAE.clicked.connect(self.UpdatePipelineText)
        self.checkRF.clicked.connect(self.UpdatePipelineText)

        self.buttonRun.clicked.connect(self.Run)

        self.UpdatePipelineText()

    def LoadTrainingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', directory=r'C:\MyCode\FAE\Example', filter="csv files (*.csv)")
        try:
            self.__training_data_container.Load(file_name)
        except:
            print('Loading Training Data Error')

        self.lineEditTrainingData.setText(file_name)
        self.UpdateDataDescription()

    def LoadTestingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        try:
            self.__testing_data_container.Load(file_name)
        except:
            print('Loading Testing Data Error')

        self.lineEditTestingData.setText(file_name)
        self.UpdateDataDescription()

    def SetVerboseTest(self, normalizer_name, dimension_reduction_name, feature_selector_name, classifier_name, feature_num,
                       current_num, total_num):
        text = "Current:\n"

        text += "{:s} / ".format(normalizer_name)
        for temp in self.__process_normalizer_list:
            text += (temp.GetName() + ", ")
        text += '\n'

        text += "{:s} / ".format(dimension_reduction_name)
        for temp in self.__process_dimension_reduction_list:
            text += (temp.GetName() + ", ")
        text += '\n'

        text += "{:s} / ".format(feature_selector_name)
        for temp in self.__process_feature_selector_list:
            text += (temp.GetName() + ", ")
        text += '\n'

        text += "Feature Number: {:d} / [{:d}-{:d}]\n".format(feature_num, self.spinBoxMinFeatureNumber.value(), self.spinBoxMaxFeatureNumber.value())

        text += "{:s} / ".format(classifier_name)
        for temp in self.__process_classifier_list:
            text += (temp.GetName() + ", ")
        text += '\n'

        text += "Total process: {:d} / {:d}".format(current_num, total_num)

        self.textEditVerbose.setPlainText(text)

    def Run(self):
        if self.__training_data_container.IsEmpty():
            QMessageBox.about(self, '', 'Training data is empty.')
            return

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            if len(os.listdir(store_folder)) > 0:
                QMessageBox.about(self, 'The folder is not empty', 'The folder is not empty')
                return

            self.textEditVerbose.setText(store_folder)
            if self.MakePipelines():
                for current_normalizer_name, current_dimension_reductor_name, \
                    current_feature_selector_name, curreent_feature_num, \
                    current_classifier_name, num, total_num\
                        in self.__fae.Run(self.__training_data_container, self.__testing_data_container, store_folder):
                    self.SetVerboseTest(current_normalizer_name,
                                        current_dimension_reductor_name,
                                        current_feature_selector_name,
                                        current_classifier_name,
                                        curreent_feature_num,
                                        num,
                                        total_num)
                    QApplication.processEvents()

                text = self.textEditVerbose.toPlainText()
                self.textEditVerbose.setPlainText(text + "\n DONE!")
            else:
                QMessageBox.about(self, 'Pipeline Error', 'Pipeline must include Classifier and CV method')

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
            self.__process_dimension_reduction_list.append(DimensionReductionByCos())

        self.__process_feature_selector_list = []
        if self.checkANOVA.isChecked():
            self.__process_feature_selector_list.append(FeatureSelectPipeline([FeatureSelectByANOVA()]))
        if self.checkRFE.isChecked():
            self.__process_feature_selector_list.append(FeatureSelectPipeline([FeatureSelectByRFE()]))
        if self.checkRelief.isChecked():
            self.__process_feature_selector_list.append(FeatureSelectPipeline([FeatureSelectByRelief()]))

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
        if len(self.__process_classifier_list) == 0:
            return False

        cv = CrossValidation()
        if self.radioLeaveOneOut.isChecked():
            cv.SetCV('LOO')
        elif self.radio5folder.isChecked():
            cv.SetCV('5-folder')
        elif self.radio10Folder.isChecked():
            cv.SetCV('10-folder')
        else:
            return False

        self.__fae.SetNormalizerList(self.__process_normalizer_list)
        self.__fae.SetDimensionReductionList(self.__process_dimension_reduction_list)
        self.__fae.SetFeatureSelectorList(self.__process_feature_selector_list)
        self.__fae.SetFeatureNumberList(self.__process_feature_number_list)
        self.__fae.SetClassifierList(self.__process_classifier_list)
        self.__fae.SetCrossValition(cv)
        self.__fae.GenerateMetircDict()

        return True

    def UpdateDataDescription(self):
        show_text = ""
        if self.__training_data_container.GetArray().size > 0:
            show_text += "The number of training cases: {:d}\n".format(len(self.__training_data_container.GetCaseName()))
            show_text += "The number of training features: {:d}\n".format(len(self.__training_data_container.GetFeatureName()))
            if len(np.unique(self.__training_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(self.__training_data_container.GetLabel() == np.max(self.__training_data_container.GetLabel()))[0])
                negative_number = len(self.__training_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self.__training_data_container.GetLabel()))
                show_text += "The number of training positive samples: {:d}\n".format(positive_number)
                show_text += "The number of training negative samples: {:d}\n".format(negative_number)

        show_text += '\n'
        if self.__testing_data_container.GetArray().size > 0:
            show_text += "The number of testing cases: {:d}\n".format(len(self.__testing_data_container.GetCaseName()))
            show_text += "The number of testing features: {:d}\n".format(
                len(self.__testing_data_container.GetFeatureName()))
            if len(np.unique(self.__testing_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(
                        self.__testing_data_container.GetLabel() == np.max(self.__testing_data_container.GetLabel()))[0])
                negative_number = len(self.__testing_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self.__testing_data_container.GetLabel()))
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
        if self.checkPCA.isChecked():
            preprocess_test += "PCA\n"
        if self.checkRemoveSimilarFeatures.isChecked():
            preprocess_test += "Remove Similary Features\n"
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
        if feature_selector_num == 0:
            feature_selector_num = 1
        self.listOnePipeline.addItem(feature_selection_text)


        classifier_test = 'Classifier:\n'
        classifier_num = 0
        if self.checkSVM.isChecked():
            classifier_test += "SVM\n"
            classifier_num += 1
        if self.checkLDA.isChecked():
            classifier_test += "LDA\n"
            classifier_num += 1
        if self.checkAE.isChecked():
            classifier_test += "AE\n"
            classifier_num += 1
        if self.checkRF.isChecked():
            classifier_test += "RF\n"
            classifier_num += 1
        if classifier_num == 0:
            classifier_num = 1
        self.listOnePipeline.addItem(classifier_test)

        self.listOnePipeline.addItem("Total number of pipelines is:\n{:d}"
                                     .format(normalizer_num * feature_selector_num * feature_num * classifier_num))


