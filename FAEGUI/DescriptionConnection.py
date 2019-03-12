from PyQt5.QtWidgets import *
from GUI.Description import Ui_Description

from FAE.FeatureAnalysis.Classifier import *
from FAE.FeatureAnalysis.FeaturePipeline import FeatureAnalysisPipelines, OnePipeline
from FAE.Description.Description import Description

from FAE.Visualization.DrawROCList import DrawROCList
from FAE.Visualization.PlotMetricVsFeatureNumber import DrawCurve, DrawBar
import os

class DescriptionConnection(QWidget, Ui_Description):
    def __init__(self, parent=None):
        self._root_folder = ''
        self._fae = FeatureAnalysisPipelines()
        self._training_data_container = DataContainer()
        self._testing_data_container = DataContainer()
        self._current_pipeline = OnePipeline()
        
        super(DescriptionConnection, self).__init__(parent)
        self.setupUi(self)

        self.buttonLoadTrainingData.clicked.connect(self.LoadTrainingData)
        self.buttonClearTrainingData.clicked.connect(self.ClearTrainingData)
        self.buttonLoadTestingData.clicked.connect(self.LoadTestingData)
        self.buttonClearTestingData.clicked.connect(self.ClearTestingData)

        self.buttonLoadResult.clicked.connect(self.LoadAll)
        self.buttonClearResult.clicked.connect(self.ClearAll)

        self.buttonGenerate.clicked.connect(self.Generate)

        self.__plt_roc = self.canvasROC.getFigure().add_subplot(111)

        # Update ROC canvas
        self.comboNormalizer.currentIndexChanged.connect(self.UpdateROC)
        self.comboDimensionReduction.currentIndexChanged.connect(self.UpdateROC)
        self.comboFeatureSelector.currentIndexChanged.connect(self.UpdateROC)
        self.comboClassifier.currentIndexChanged.connect(self.UpdateROC)
        self.spinBoxFeatureNumber.valueChanged.connect(self.UpdateROC)
        self.checkROCTrain.stateChanged.connect(self.UpdateROC)
        self.checkROCValidation.stateChanged.connect(self.UpdateROC)
        self.checkROCTest.stateChanged.connect(self.UpdateROC)

        self.SetPipelineStateButton(False)

    def Generate(self):
        if self._training_data_container.IsEmpty():
            QMessageBox.about(self, '', 'Load training data at least')
            return

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            roc_path = os.path.join(store_folder, 'ROC.jpg')
            self.canvasROC.getFigure().savefig(roc_path, dpi=300)

            report = Description()
            try:
                report.Run(self._training_data_container, self._current_pipeline, self._root_folder, store_folder, self._testing_data_container)
                os.system("explorer.exe {:s}".format(os.path.normpath(store_folder)))
            except Exception as ex:
                QMessageBox.about(self, 'Description Generate Error: ', ex.__str__())
                self.logger.log('Description Generate Error:  ' + str(ex))


    def LoadTrainingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', directory=r'C:\MyCode\FAE\Example', filter="csv files (*.csv)")
        try:
            self._training_data_container.Load(file_name)
            # self.SetStateButtonBeforeLoading(True)
            self.lineEditTrainingData.setText(file_name)
            self.UpdateDataDescription()
        except Exception as ex:
            QMessageBox.about(self, "Load Error", ex.__str__())
            self.logger.log('Open SCV file Error, The reason is ' + str(ex))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected training data mismatch.')

    def ClearTrainingData(self):
        self._training_data_container = DataContainer()
        # self.SetStateButtonBeforeLoading(False)
        self.lineEditTrainingData.setText("")
        self.UpdateDataDescription()

    def LoadTestingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        try:
            self._testing_data_container.Load(file_name)
            self.lineEditTestingData.setText(file_name)
            self.UpdateDataDescription()
        except Exception as ex:
            QMessageBox.about(self, "Load Error", ex.__str__())
            self.logger.log('Open SCV file Error, The reason is ' + str(ex))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected testing data mismatch.')

    def ClearTestingData(self):
        self._testing_data_container = DataContainer()
        self.lineEditTestingData.setText("")
        self.UpdateDataDescription()
    
    def UpdateDataDescription(self):
        show_text = ''
        if not self._training_data_container.IsEmpty():
            show_text += "The number of training cases: {:d}\n".format(len(self._training_data_container.GetCaseName()))
            show_text += "The number of training features: {:d}\n".format(len(self._training_data_container.GetFeatureName()))
            if len(np.unique(self._training_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(self._training_data_container.GetLabel() == np.max(self._training_data_container.GetLabel()))[0])
                negative_number = len(self._training_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self._training_data_container.GetLabel()))
                show_text += "The number of training positive samples: {:d}\n".format(positive_number)
                show_text += "The number of training negative samples: {:d}\n".format(negative_number)

        show_text += '\n'
        if not self._testing_data_container.IsEmpty():
            show_text += "The number of testing cases: {:d}\n".format(len(self._testing_data_container.GetCaseName()))
            show_text += "The number of testing features: {:d}\n".format(
                len(self._testing_data_container.GetFeatureName()))
            if len(np.unique(self._testing_data_container.GetLabel())) == 2:
                positive_number = len(
                    np.where(
                        self._testing_data_container.GetLabel() == np.max(self._testing_data_container.GetLabel()))[0])
                negative_number = len(self._testing_data_container.GetLabel()) - positive_number
                assert (positive_number + negative_number == len(self._testing_data_container.GetLabel()))
                show_text += "The number of testing positive samples: {:d}\n".format(positive_number)
                show_text += "The number of testing negative samples: {:d}\n".format(negative_number)

        self.textEditDataDescription.setText(show_text)
        
    def LoadAll(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]

            if not os.path.exists(self._root_folder):
                return
            if not r'.FAEresult4129074093819729087' in os.listdir(self._root_folder):
                QMessageBox.about(self, 'Load Error', 'This folder is not supported for import')
                return
            try:
                self.lineEditResultPath.setText(self._root_folder)
                self._fae.LoadAll(self._root_folder)
                self.SetResultDescription()
                self.InitialUi()
            except Exception as ex:
                QMessageBox.about(self, "Load Error", ex.__str__())
                self.logger.log('Load Error, The reason is ' + str(ex))
                self.ClearAll()
                return

            self.SetPipelineStateButton(True)

    def ClearAll(self):
        self.buttonLoadResult.setEnabled(True)
        self.buttonClearResult.setEnabled(False)

        self._fae = FeatureAnalysisPipelines()
        self.textEditDescription.setPlainText('')
        self.lineEditResultPath.setText('')
        self.InitialUi()

        self.checkROCTrain.setChecked(False)
        self.checkROCValidation.setChecked(False)
        self.checkROCTest.setChecked(False)

        self.spinBoxFeatureNumber.setValue(1)

        self.canvasROC.getFigure().clear()
        self.canvasROC.draw()

        self.SetPipelineStateButton(False)

    def SetPipelineStateButton(self, state):
        self.checkROCTrain.setEnabled(state)
        self.checkROCValidation.setEnabled(state)
        self.checkROCTest.setEnabled(state)

        self.comboNormalizer.setEnabled(state)
        self.comboDimensionReduction.setEnabled(state)
        self.comboFeatureSelector.setEnabled(state)
        self.comboClassifier.setEnabled(state)
        self.spinBoxFeatureNumber.setEnabled(state)

        self.buttonClearResult.setEnabled(state)
        self.buttonLoadResult.setEnabled(not state)

        self.buttonGenerate.setEnabled(state)

    def InitialUi(self):
        # Update ROC canvers
        self.comboNormalizer.clear()
        for normalizer in self._fae.GetNormalizerList():
            self.comboNormalizer.addItem(normalizer.GetName())
        self.comboDimensionReduction.clear()
        for dimension_reduction in self._fae.GetDimensionReductionList():
            self.comboDimensionReduction.addItem(dimension_reduction.GetName())
        self.comboClassifier.clear()
        for classifier in self._fae.GetClassifierList():
            self.comboClassifier.addItem(classifier.GetName())
        self.comboFeatureSelector.clear()
        for feature_selector in self._fae.GetFeatureSelectorList():
            self.comboFeatureSelector.addItem(feature_selector.GetName())

        if self._fae.GetFeatureNumberList() != []:
            self.spinBoxFeatureNumber.setMinimum(int(self._fae.GetFeatureNumberList()[0]))
            self.spinBoxFeatureNumber.setMaximum(int(self._fae.GetFeatureNumberList()[-1]))

    def SetResultDescription(self):
        text = "Normalizer:\n"
        for index in self._fae.GetNormalizerList():
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Dimension Reduction:\n"
        for index in self._fae.GetDimensionReductionList():
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Selector:\n"
        for index in self._fae.GetFeatureSelectorList():
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Number:\n"
        text += "{:s} - {:s}\n".format(self._fae.GetFeatureNumberList()[0], self._fae.GetFeatureNumberList()[-1])
        text += '\n'

        text += "Classifier:\n"
        for index in self._fae.GetClassifierList():
            text += (index.GetName() + '\n')
        text += '\n'

        self.textEditDescription.setPlainText(text)

    def UpdateROC(self):
        if (self.comboNormalizer.count() == 0) or \
                (self.comboDimensionReduction.count() == 0) or \
                (self.comboFeatureSelector.count() == 0) or \
                (self.comboClassifier.count() == 0) or \
                (self.spinBoxFeatureNumber.value() == 0):
            return

        case_name = self.comboNormalizer.currentText() + '_' + \
                    self.comboDimensionReduction.currentText() + '_' + \
                    self.comboFeatureSelector.currentText() + '_' + \
                    str(self.spinBoxFeatureNumber.value()) + '_' + \
                    self.comboClassifier.currentText()

        case_folder = os.path.join(self._root_folder, case_name)
        try:
            self._current_pipeline.LoadPipeline(os.path.join(case_folder, 'pipeline_info.csv'))
        except Exception as ex:
            QMessageBox.about(self, "Load Error", ex.__str__())
            self.logger.log('Load Pipeline Error, The reason is ' + str(ex))


        pred_list, label_list, name_list = [], [], []
        if self.checkROCTrain.isChecked():
            train_pred = np.load(os.path.join(case_folder, 'train_predict.npy'))
            train_label = np.load(os.path.join(case_folder, 'train_label.npy'))
            pred_list.append(train_pred)
            label_list.append(train_label)
            name_list.append('train')
        if self.checkROCValidation.isChecked():
            val_pred = np.load(os.path.join(case_folder, 'val_predict.npy'))
            val_label = np.load(os.path.join(case_folder, 'val_label.npy'))
            pred_list.append(val_pred)
            label_list.append(val_label)
            name_list.append('validation')
        if self.checkROCTest.isChecked():
            if os.path.exists(os.path.join(case_folder, 'test_label.npy')):
                test_pred = np.load(os.path.join(case_folder, 'test_predict.npy'))
                test_label = np.load(os.path.join(case_folder, 'test_label.npy'))
                pred_list.append(test_pred)
                label_list.append(test_label)
                name_list.append('Test')

        if len(pred_list) > 0:
            DrawROCList(pred_list, label_list, name_list=name_list, is_show=False, fig=self.canvasROC.getFigure())

        self.canvasROC.draw()