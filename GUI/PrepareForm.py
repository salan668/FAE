import numpy as np
import os
from copy import deepcopy
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
from GUI.Prepare import Ui_Prepare
from Utility.EcLog import eclog
from FAE.DataContainer.DataContainer import DataContainer
from FAE.DataContainer import DataSeparate
from FAE.FeatureAnalysis.FeatureSelector import RemoveSameFeatures


class PrepareConnection(QWidget, Ui_Prepare):
    close_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(PrepareConnection, self).__init__(parent)
        self.setupUi(self)
        self.data_container = DataContainer()
        self._filename = os.path.split(__file__)[-1]

        self.buttonLoad.clicked.connect(self.LoadData)
        self.buttonRemove.clicked.connect(self.RemoveNonValidValue)

        self.__testing_ref_data_container = DataContainer()
        self.__clinical_ref = pd.DataFrame()

        self.radioSplitRandom.clicked.connect(self.ChangeSeparateMethod)
        self.radioSplitRef.clicked.connect(self.ChangeSeparateMethod)
        self.checkUseClinicRef.clicked.connect(self.RandomSeparateButtonUpdates)
        self.loadTestingReference.clicked.connect(self.LoadTestingReferenceDataContainer)
        self.clearTestingReference.clicked.connect(self.ClearTestingReferenceDataContainer)
        self.loadClinicRef.clicked.connect(self.LoadClinicalRef)
        self.clearClinicRef.clicked.connect(self.ClearClinicalRef)

        self.buttonSave.clicked.connect(self.CheckAndSave)

    def closeEvent(self, QCloseEvent):
        self.close_signal.emit(True)
        QCloseEvent.accept()

    def UpdateTable(self):
        if self.data_container.GetArray().size == 0:
            return

        self.tableFeature.setRowCount(len(self.data_container.GetCaseName()))
        header_name = deepcopy(self.data_container.GetFeatureName())
        header_name.insert(0, 'Label')

        min_col = np.min([len(header_name), 100])
        if min_col == 100:
            header_name = header_name[:100]
            header_name[-1] = '...'

        self.tableFeature.setColumnCount(min_col)
        self.tableFeature.setHorizontalHeaderLabels(header_name)
        self.tableFeature.setVerticalHeaderLabels(list(map(str, self.data_container.GetCaseName())))

        for row_index in range(len(self.data_container.GetCaseName())):
            for col_index in range(min_col):
                if col_index == 0:
                    self.tableFeature.setItem(row_index, col_index,
                                              QTableWidgetItem(str(self.data_container.GetLabel()[row_index])))
                elif col_index < 99:
                    self.tableFeature.setItem(row_index, col_index,
                                              QTableWidgetItem(str(self.data_container.GetArray()[row_index, col_index - 1])))
                else:
                    self.tableFeature.setItem(row_index, col_index,
                                              QTableWidgetItem('...'))

        text = "The number of cases: {:d}\n".format(len(self.data_container.GetCaseName()))
        text += "The number of features: {:d}\n".format(len(self.data_container.GetFeatureName()))
        if len(np.unique(self.data_container.GetLabel())) == 2:
            positive_number = len(np.where(self.data_container.GetLabel() == np.max(self.data_container.GetLabel()))[0])
            negative_number = len(self.data_container.GetLabel()) - positive_number
            assert (positive_number + negative_number == len(self.data_container.GetLabel()))
            text += "The number of positive samples: {:d}\n".format(positive_number)
            text += "The number of negative samples: {:d}\n".format(negative_number)
        self.textInformation.setText(text)

    def LoadData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        try:
            self.data_container.Load(file_name)
            print(eclog(file_name))
        except OSError as reason:
            eclog(self._filename).GetLogger().error('Load CSV Error: {}'.format(reason))
            QMessageBox.about(self, 'Load data Error', reason.__str__())
            print('Error！' + str(reason))
        except ValueError:
            eclog(self._filename).GetLogger().error('Open CSV Error: {}'.format(file_name))
            QMessageBox.information(self, 'Error', 'The selected data file mismatch.')
        self.UpdateTable()

        self.buttonRemove.setEnabled(True)
        self.buttonSave.setEnabled(True)
        self.radioRemoveNonvalidCases.setEnabled(True)
        self.radioRemoveNonvalidFeatures.setEnabled(True)
        self.radioSplitRandom.setEnabled(True)
        self.radioSplitRef.setEnabled(True)

    def LoadTestingReferenceDataContainer(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        try:
            self.__testing_ref_data_container.Load(file_name)
            self.loadTestingReference.setEnabled(False)
            self.clearTestingReference.setEnabled(True)
            self.spinBoxSeparate.setEnabled(False)
        except OSError as reason:
            eclog(self._filename).GetLogger().error('Load Testing Ref Error: {}'.format(reason))
            print('Error！' + str(reason))
        except ValueError:
            eclog(self._filename).GetLogger().error('Open CSV Error: {}'.format(file_name))
            QMessageBox.information(self, 'Error',
                                    'The selected data file mismatch.')

    def ClearTestingReferenceDataContainer(self):
        del self.__testing_ref_data_container
        self.__testing_ref_data_container = DataContainer()
        self.loadTestingReference.setEnabled(True)
        self.clearTestingReference.setEnabled(False)
        self.spinBoxSeparate.setEnabled(False)

    def LoadClinicalRef(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        try:
            self.__clinical_ref = pd.read_csv(file_name, index_col=0)
            if list(self.__clinical_ref.index) != self.data_container.GetCaseName():
                QMessageBox.information(self, 'Error',
                                        'The index of clinical features is not consistent to the data')
                return None
            self.loadClinicRef.setEnabled(False)
            self.clearClinicRef.setEnabled(True)
        except OSError as reason:
            eclog(self._filename).GetLogger().error('Load Clinical Ref Error: {}'.format(reason))
            QMessageBox.information(self, 'Error',
                                    'Can not Open the Files')
        except ValueError:
            eclog(self._filename).GetLogger().error('OpenCSV Error: {}'.format(file_name))
            QMessageBox.information(self, 'Error',
                                    'The selected data file mismatch.')
        return None

    def ClearClinicalRef(self):
        del self.__clinical_ref
        self.__clinical_ref = pd.DataFrame()
        self.loadClinicRef.setEnabled(True)
        self.clearClinicRef.setEnabled(False)

    def RemoveNonValidValue(self):
        if self.radioRemoveNonvalidCases.isChecked():
            self.data_container.RemoveUneffectiveCases()
        elif self.radioRemoveNonvalidFeatures.isChecked():
            self.data_container.RemoveUneffectiveFeatures()

        self.UpdateTable()

    def ChangeSeparateMethod(self):
        if self.radioSplitRandom.isChecked():
            self.spinBoxSeparate.setEnabled(True)
            self.checkUseClinicRef.setEnabled(True)
            self.loadTestingReference.setEnabled(False)
            self.clearTestingReference.setEnabled(False)
        elif self.radioSplitRef.isChecked():
            self.spinBoxSeparate.setEnabled(False)
            self.checkUseClinicRef.setEnabled(False)
            if self.__testing_ref_data_container.IsEmpty():
                self.loadTestingReference.setEnabled(True)
                self.clearTestingReference.setEnabled(False)
            else:
                self.loadTestingReference.setEnabled(False)
                self.clearTestingReference.setEnabled(True)
        self.RandomSeparateButtonUpdates()

    def RandomSeparateButtonUpdates(self):
        if self.checkUseClinicRef.isChecked():
            if self.__clinical_ref.size > 0:
                self.loadClinicRef.setEnabled(False)
                self.clearClinicRef.setEnabled(True)
            else:
                self.loadClinicRef.setEnabled(True)
                self.clearClinicRef.setEnabled(False)
        else:
            self.loadClinicRef.setEnabled(False)
            self.clearClinicRef.setEnabled(False)

    def CheckAndSave(self):
        if self.data_container.IsEmpty():
            QMessageBox.warning(self, "Warning", "There is no data", QMessageBox.Ok)
        elif not self.data_container.IsBinaryLabel():
            QMessageBox.warning(self, "Warning", "There are not 2 Labels", QMessageBox.Ok)
            non_valid_number_index = self.data_container.FindNonValidLabelIndex()
            old_edit_triggers = self.tableFeature.editTriggers()
            self.tableFeature.setEditTriggers(QAbstractItemView.CurrentChanged)
            self.tableFeature.setCurrentCell(non_valid_number_index, 0)
            self.tableFeature.setEditTriggers(old_edit_triggers)
        elif self.data_container.HasNonValidNumber():
            QMessageBox.warning(self, "Warning", "There are nan items", QMessageBox.Ok)
            non_valid_number_index = self.data_container.FindNonValidNumberIndex()
            old_edit_triggers = self.tableFeature.editTriggers()
            self.tableFeature.setEditTriggers(QAbstractItemView.CurrentChanged)
            self.tableFeature.setCurrentCell(non_valid_number_index[0], non_valid_number_index[1]+1)
            self.tableFeature.setEditTriggers(old_edit_triggers)
        else:
            remove_features_with_same_value = RemoveSameFeatures()
            self.data_container = remove_features_with_same_value.Run(self.data_container)

            folder_name = QFileDialog.getExistingDirectory(self, "Save data")
            if folder_name != '':
                data_separate = DataSeparate.DataSeparate()
                try:
                    if self.__testing_ref_data_container.IsEmpty():
                        testing_data_percentage = self.spinBoxSeparate.value()
                        if self.__clinical_ref.size == 0:
                            training_data_container, _, = \
                                data_separate.RunByTestingPercentage(self.data_container,
                                                                     testing_data_percentage,
                                                                     store_folder=folder_name)
                        else:
                            training_data_container, _, = \
                                data_separate.RunByTestingPercentage(self.data_container,
                                                                     testing_data_percentage,
                                                                     clinic_df=self.__clinical_ref,
                                                                     store_folder=folder_name)
                    else:
                        training_data_container, _, = \
                            data_separate.RunByTestingReference(self.data_container,
                                                                self.__testing_ref_data_container,
                                                                folder_name)
                        if training_data_container.IsEmpty():
                            QMessageBox.information(self, 'Error',
                                                    'The testing data does not mismatch, please check the testing data '
                                                    'really exists in current data')
                            return None
                    os.system("explorer.exe {:s}".format(os.path.normpath(folder_name)))
                except Exception as e:
                    content = 'PrepareConnection, splitting failed: '
                    eclog(self._filename).GetLogger().error('Split Error:  ' + e.__str__())
                    QMessageBox.about(self, content, e.__str__())


            else:
                file_name, _ = QFileDialog.getSaveFileName(self, "Save data", filter="csv files (*.csv)")
