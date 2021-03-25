"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/25
"""
import sys
import shutil

import traceback
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from SA.GUI.Process import Ui_Process
from SA.PipelineManager import PipelineManager
from SA.Normalizer import *
from SA.DimensionReducer import *
from SA.FeatureSelector import *
from SA.Fitter import *
from SA.CrossValidation import CrossValidation
from SA.Utility import mylog


class CVRun(QThread):
    signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._process_form = None
        self.store_folder = None

    def SetProcessForm(self, process_form, store_folder):
        self._process_form = process_form
        self.store_folder = store_folder

    def run(self):
        self._process_form.pipeline_manager.SaveInfo(self.store_folder)

        try:
            for total, num, group in self._process_form.pipeline_manager.RunCV(
                    self._process_form.train_dc,
                    self.store_folder):
                text = "Cross Validation:\nGroup {}: {} / {}".format(int(group), num, total)
                self.signal.emit(text)
        except Exception as e:
            print(traceback.format_exc())
            mylog.error('Thread RunCV Failed: ', e.__str__())

        try:
            for total, num in self._process_form.pipeline_manager.EstimateCV(
                    self.store_folder,
                    self._process_form.train_dc.event_name,
                    self._process_form.train_dc.duration_name):
                text = "Cross Validation:\nDone.\n\n" \
                       "Merging CV Results:\n{} / {}".format(num, total)
                self.signal.emit(text)
        except Exception as e:
            mylog.error('Thread MergeCV Result Failed: ', e.__str__())

        try:
            for total, num in self._process_form.pipeline_manager.RunWithoutCV(self._process_form.train_dc,
                                                                               self._process_form.test_dc,
                                                                               self.store_folder):
                text = "Corss Validation Done.\n\nCV Result Merge Done\n\n" \
                       "Model Developing:\n{} / {}".format(num, total)
                self.signal.emit(text)
        except Exception:
            mylog.error("Thread RunWithoutCV Failed: ", e.__str__())

        text = "Model Developing:\nDone.\n\nCross Validation:\nDone.\n\nMerging CV Results:\nDone.\n"
        self.signal.emit(text)
        self._process_form.SetLoadState(True)
        self._process_form.SetRunState(True)


class ProcessForm(QWidget, Ui_Process):
    close_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        self.train_dc = DataContainer()
        self.test_dc = DataContainer()
        self.event_name, self.duration_name = None, None
        self.pipeline_manager = PipelineManager()

        self.__normalizers = []
        self.__dimension_reducers = []
        self.__feature_selectors = []
        self.__feature_numbers = []
        self.__fitters = []

        self.thread = CVRun()

        super(ProcessForm, self).__init__(parent)
        self.setupUi(self)

        self.buttonLoadTrainingData.clicked.connect(self.LoadTrainingData)
        self.buttonLoadTestingData.clicked.connect(self.LoadTestingData)

        self.comboEvent.currentIndexChanged.connect(self.UpdateEvent)
        self.comboDuration.currentIndexChanged.connect(self.UpdateDuration)
        self.buttonLoad.clicked.connect(self.LoadData)
        self.buttonClear.clicked.connect(self.ClearData)

        self.checkNormalizeNone.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeMinMax.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeZscore.clicked.connect(self.UpdatePipelineText)
        self.checkNormalizeMean.clicked.connect(self.UpdatePipelineText)

        self.checkDimensionReduceNone.clicked.connect(self.UpdatePipelineText)
        self.checkDimensionReducePCC.clicked.connect(self.UpdatePipelineText)

        self.spinBoxMinFeatureNumber.valueChanged.connect(self.FeatureNumberChange)
        self.spinBoxMaxFeatureNumber.valueChanged.connect(self.FeatureNumberChange)

        self.checkFeatureSelectorNone.clicked.connect(self.UpdatePipelineText)
        self.checkFeatureSelectorCluster.clicked.connect(self.UpdatePipelineText)

        self.checkCoxPH.clicked.connect(self.UpdatePipelineText)
        # self.checkAalen.clicked.connect(self.UpdatePipelineText)

        self.spinCvFold.valueChanged.connect(self.UpdatePipelineText)

        self.buttonRun.clicked.connect(self.Run)

        self.UpdatePipelineText()

        self.SetLoadState(True)
        self.buttonLoad.setEnabled(False)
        self.SetRunState(False)

    def closeEvent(self, event):
        self.close_signal.emit(True)
        event.accept()

    # Data Related
    def LoadTrainingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=r'C:\MyCode\FAE\Example',
                                           filter="csv files (*.csv)")
        if file_name:
            self.lineEditTrainingData.setText(file_name)

            df = pd.read_csv(file_name, index_col=0)
            self.comboEvent.clear()
            self.comboDuration.clear()
            self.comboEvent.addItems(df.columns)
            self.comboDuration.addItems(df.columns)
            self.buttonLoad.setEnabled(True)

    def LoadTestingData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=r'C:\MyCode\FAE\Example',
                                           filter="csv files (*.csv)")
        if file_name:
            self.lineEditTestingData.setText(file_name)

    def UpdateEvent(self):
        self.event_name = self.comboEvent.currentText()

    def UpdateDuration(self):
        self.duration_name = self.comboDuration.currentText()

    def LoadData(self):
        if self.lineEditTrainingData.text():
            try:
                self.train_dc.Load(self.lineEditTrainingData.text(),
                                   self.event_name, self.duration_name)
                mylog.info('Open CSV file ' + self.lineEditTrainingData.text() + ' succeed.')
                self.spinBoxMaxFeatureNumber.setValue(len(self.train_dc.feature_name))
            except OSError as reason:
                message = 'Open SCV file Error, The reason is ' + str(reason)
                mylog.error(message)
                QMessageBox.information(self, 'Error', message)
                return
            except ValueError:
                message = 'Open SCV file ' + self.lineEditTrainingData.text() + ' Failed. because of value error.'
                mylog.error(message)
                QMessageBox.information(self, 'Error', message)
                return
            except Exception as e:
                message = 'Unexpected Error: {}'.format(e.__str__())
                mylog.error(message)
                QMessageBox.information(self, 'Error', message)
                return

        if self.lineEditTestingData.text():
            try:
                self.test_dc.Load(self.lineEditTestingData.text(),
                                   self.event_name, self.duration_name)
                mylog.info('Open CSV file ' + self.lineEditTestingData.text() + ' succeed.')
            except OSError as reason:
                message = 'Open SCV file Error, The reason is ' + str(reason)
                mylog.error(message)
                QMessageBox.information(self, 'Error', message)
                return
            except ValueError:
                message = 'Open SCV file ' + self.lineEditTrainingData.text() + ' Failed. because of value error.'
                mylog.error(message)
                QMessageBox.information(self, 'Error', message)
                return
            except KeyError:
                message = 'The event key {} and duration key {} do not exist in testing data set.'.format(
                    self.comboEvent.currentText(), self.comboDuration.currentText()
                )
                mylog.error(message)
                QMessageBox.information(self, 'Error', message)
                return
            except Exception as e:
                message = 'Unexpected Error: {}'.format(e.__str__())
                mylog.error(message)
                QMessageBox.information(self, 'Error', message)
                return
        else:
            self.test_dc = DataContainer()

        self.SetRunState(True)

        self.spinCvFold.setMaximum(len(self.train_dc.case_name))
        self.spinCvFold.setValue(5)
        self.UpdateDataDescription()

    def ClearData(self):
        self.train_dc = DataContainer()
        self.test_dc = DataContainer()
        self.lineEditTrainingData.setText("")
        self.lineEditTestingData.setText("")
        self.comboEvent.clear()
        self.comboDuration.clear()
        self.textEditDescription.clear()
        self.listOnePipeline.clear()
        self.pipeline_manager = PipelineManager()

        self.SetLoadState(True)
        self.buttonLoad.setEnabled(False)
        self.SetRunState(False)

    def UpdateDataDescription(self):
        text = ""
        if not self.train_dc.IsEmpty():
            text += self.train_dc.__str__()
            if not self.test_dc.IsEmpty():
                text += '\n\n'
                text += self.test_dc.__str__()
        self.textEditDescription.setText(text)

    def SetRunState(self, state):
        self.checkNormalizeNone.setEnabled(state)
        self.checkNormalizeMinMax.setEnabled(state)
        self.checkNormalizeZscore.setEnabled(state)
        self.checkNormalizeMean.setEnabled(state)

        self.checkDimensionReduceNone.setEnabled(state)
        self.checkDimensionReducePCC.setEnabled(state)

        self.checkFeatureSelectorNone.setEnabled(state)
        self.checkFeatureSelectorCluster.setEnabled(state)

        self.spinBoxMinFeatureNumber.setEnabled(state)
        self.spinBoxMaxFeatureNumber.setEnabled(state)

        self.checkCoxPH.setEnabled(state)
        # self.checkAalen.setEnabled(state)

        self.spinCvFold.setEnabled(state)

        self.buttonRun.setEnabled(state)

        self.listOnePipeline.setEnabled(state)

    def SetLoadState(self, state):
        self.lineEditTrainingData.setEnabled(state)
        self.lineEditTestingData.setEnabled(state)
        self.buttonLoadTrainingData.setEnabled(state)
        self.buttonLoadTestingData.setEnabled(state)
        self.comboEvent.setEnabled(state)
        self.comboDuration.setEnabled(state)
        self.buttonLoad.setEnabled(state)
        self.buttonClear.setEnabled(state)

    #%% Pipeline
    def FeatureNumberChange(self):
        if self.spinBoxMinFeatureNumber.value() < 1:
            self.spinBoxMinFeatureNumber.setValue(1)
        if self.spinBoxMaxFeatureNumber.value() > len(self.train_dc.feature_name):
            self.spinBoxMaxFeatureNumber.setValue(len(self.train_dc.feature_name))
        if self.spinBoxMinFeatureNumber.value() > self.spinBoxMaxFeatureNumber.value():
            self.spinBoxMinFeatureNumber.setValue(self.spinBoxMaxFeatureNumber.value())

        self.UpdatePipelineText()

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
        if self.checkDimensionReduceNone.isChecked():
            preprocess_test += "None\n"
            dimension_reduction_num += 1
        if self.checkDimensionReducePCC.isChecked():
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
        if self.checkFeatureSelectorNone.isChecked():
            feature_selection_text += "None\n"
            feature_selector_num += 1
        if self.checkFeatureSelectorCluster.isChecked():
            feature_selection_text += "Cluster\n"
            feature_selector_num += 1
        self.listOnePipeline.addItem(feature_selection_text)

        classifier_text = 'Fitter:\n'
        classifier_num = 0
        if self.checkCoxPH.isChecked():
            classifier_text += "CoxPH\n"
            classifier_num += 1
        # if self.checkAalen.isChecked():
        #     classifier_text += "AalenAdaptive\n"
        #     classifier_num += 1
        if classifier_num == 0:
            classifier_num = 1
        self.listOnePipeline.addItem(classifier_text)

        cv_method = "Cross Validation:\n{} fold.\n\n".format(self.spinCvFold.value())

        self.listOnePipeline.addItem(cv_method)

        self.listOnePipeline.addItem("Total number of pipelines is:\n{:d}".format(
            normalizer_num * dimension_reduction_num * feature_selector_num * feature_num * classifier_num
        ))

    #%% Run
    def MakePipelines(self):
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
        if self.checkDimensionReduceNone.isChecked():
            self.__dimension_reducers.append(DimensionReducerNone())
        if self.checkDimensionReducePCC.isChecked():
            self.__dimension_reducers.append(DimensionReducerPcc())

        self.__feature_selectors = []
        if self.checkFeatureSelectorNone.isChecked():
            self.__feature_selectors.append(FeatureSelectorAll)
        if self.checkFeatureSelectorCluster.isChecked():
            self.__feature_selectors.append(FeatureSelectorCluster)

        self.__feature_numbers = np.arange(self.spinBoxMinFeatureNumber.value(),
                                           self.spinBoxMaxFeatureNumber.value() + 1).tolist()

        self.__fitters = []
        if self.checkCoxPH.isChecked():
            self.__fitters.append(CoxPH())
        # if self.checkAalen.isChecked():
        #     self.__fitters.append(AalenAdditive())
        if len(self.__fitters) == 0:
            mylog.error('Process classifier list length is zero.')
            return False

        cv = CrossValidation(k=int(self.spinCvFold.value()))

        self.pipeline_manager.SetCV(cv)
        self.pipeline_manager.SetNormalizers(self.__normalizers)
        self.pipeline_manager.SetReducers(self.__dimension_reducers)
        self.pipeline_manager.SetFeatureNumbers(self.__feature_numbers)
        self.pipeline_manager.SetFeatureSelectors(self.__feature_selectors)
        self.pipeline_manager.SetFitters(self.__fitters)

        return True

    def Run(self):
        if self.train_dc.IsEmpty():
            QMessageBox.about(self, '', 'Training data is empty.')
            mylog.warning('Training data is empty.')
            return

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            if len(os.listdir(store_folder)) > 0:
                reply = QMessageBox.question(self, 'Continue?',
                                             'The folder is not empty, if you click Yes, '
                                             'the data would be clear in this folder',
                                             QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    try:
                        for file in os.listdir(store_folder):
                            if os.path.isdir(os.path.join(store_folder, file)):
                                shutil.rmtree(os.path.join(store_folder, file))
                            else:
                                os.remove(os.path.join(store_folder, file))
                    except PermissionError:
                        mylog.error('Permission Error: {}'.format(store_folder))
                        QMessageBox().about(self, 'Warning', 'Is there any opened files?')
                        return
                    except OSError:
                        mylog.error('Permission Error: {}'.format(store_folder))
                        QMessageBox().about(self, 'Warning', 'Is there any opened files?')
                        return
                else:
                    return

            if self.MakePipelines():
                self.thread.moveToThread(QThread())
                self.thread.SetProcessForm(self, store_folder)

                self.thread.signal.connect(self.textEditVerbose.setPlainText)
                self.thread.start()
                self.SetLoadState(False)
                self.SetRunState(False)
            else:
                mylog.error('Make pipeline failed. Pipeline must include Fitter and CV method.')
                QMessageBox.about(self, 'Pipeline Error', 'Pipeline must include Fitter and CV method')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_frame = ProcessForm()
    main_frame.show()
    sys.exit(app.exec_())
