import traceback

import numpy as np
import os
from copy import deepcopy
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
from BC.GUI.Prepare import Ui_Prepare
from Utility.EcLog import eclog
from BC.DataContainer.DataContainer import DataContainer
from BC.DataContainer import DataSeparate
from BC.FeatureAnalysis.FeatureSelector import RemoveSameFeatures
from BC.Utility.Constants import REMOVE_CASE, REMOVE_FEATURE


class PrepareConnection(QWidget, Ui_Prepare):
    close_signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(PrepareConnection, self).__init__(parent)
        self.setupUi(self)
        self.data_container = DataContainer()
        self._filename = os.path.split(__file__)[-1]

        self.buttonLoad.clicked.connect(self.LoadData)
        self.buttonRemoveAndExport.clicked.connect(self.RemoveInvalidValue)

        self.testing_ref_data_container = DataContainer()
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
        self.tableFeature.setRowCount(self.data_container.GetFrame().shape[0])
        header_name = deepcopy(list(self.data_container.GetFrame().columns))

        min_col = np.min([len(header_name), 100])
        if min_col == 100:
            header_name = header_name[:100]
            header_name[-1] = '...'

        self.tableFeature.setColumnCount(min_col)
        self.tableFeature.setHorizontalHeaderLabels(header_name)
        self.tableFeature.setVerticalHeaderLabels(list(map(str, self.data_container.GetFrame().index)))

        for row_index in range(self.data_container.GetFrame().shape[0]):
            for col_index in range(min_col):
                if col_index < 99:
                    self.tableFeature.setItem(row_index, col_index,
                                              QTableWidgetItem(str(
                                                  self.data_container.GetFrame().iloc[row_index, col_index])))
                else:
                    self.tableFeature.setItem(row_index, col_index,
                                              QTableWidgetItem('...'))

        text = "The number of cases: {:d}\n".format(self.data_container.GetFrame().shape[0])
        # To process Label temporally
        if 'label' in self.data_container.GetFrame().columns:
            label_name = 'label'
            text += "The number of features: {:d}\n".format(self.data_container.GetFrame().shape[1] - 1)
        elif 'Label' in self.data_container.GetFrame().columns:
            label_name = 'Label'
            text += "The number of features: {:d}\n".format(self.data_container.GetFrame().shape[1] - 1)
        else:
            label_name = ''
            text += "The number of features: {:d}\n".format(self.data_container.GetFrame().shape[1])
        if label_name:
            labels = np.asarray(self.data_container.GetFrame()[label_name].values, dtype=np.int)
            if len(np.unique(labels)) == 2:
                positive_number = len(np.where(labels == np.max(labels))[0])
                negative_number = len(labels) - positive_number
                assert (positive_number + negative_number == len(labels))
                text += "The number of positive samples: {:d}\n".format(positive_number)
                text += "The number of negative samples: {:d}\n".format(negative_number)
        self.textInformation.setText(text)

    def SetButtonsState(self, state):
        self.buttonRemoveAndExport.setEnabled(state)
        self.buttonSave.setEnabled(state)
        self.checkExport.setEnabled(state)
        self.radioRemoveNone.setEnabled(state)
        self.radioRemoveNonvalidCases.setEnabled(state)
        self.radioRemoveNonvalidFeatures.setEnabled(state)
        self.radioSplitRandom.setEnabled(state)
        self.radioSplitRef.setEnabled(state)

    def LoadData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        if file_name:
            try:
                if self.data_container.Load(file_name, is_update=False):
                    self.UpdateTable()
                    self.SetButtonsState(True)

            except OSError as reason:
                eclog(self._filename).GetLogger().error('Load CSV Error: {}'.format(reason))
                QMessageBox.about(self, 'Load data Error', reason.__str__())
                print('Error！' + str(reason))
            except ValueError:
                eclog(self._filename).GetLogger().error('Open CSV Error: {}'.format(file_name))
                QMessageBox.information(self, 'Error', 'The selected data file mismatch.')

    def LoadTestingReferenceDataContainer(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        if file_name:
            try:
                self.testing_ref_data_container.Load(file_name)
                self.loadTestingReference.setEnabled(False)
                self.clearTestingReference.setEnabled(True)
                self.spinBoxSeparate.setEnabled(False)
            except OSError as reason:
                eclog(self._filename).GetLogger().error('Load Testing Ref Error: {}'.format(reason))
                print('Error！' + str(reason))
                self.ClearTestingReferenceDataContainer()
            except ValueError:
                eclog(self._filename).GetLogger().error('Open CSV Error: {}'.format(file_name))
                QMessageBox.information(self, 'Error',
                                        'The selected data file mismatch.')
                self.ClearTestingReferenceDataContainer()

    def ClearTestingReferenceDataContainer(self):
        self.testing_ref_data_container.Clear()

        self.loadTestingReference.setEnabled(True)
        self.clearTestingReference.setEnabled(False)
        self.spinBoxSeparate.setEnabled(True)

    def LoadClinicalRef(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        if file_name:
            try:
                self.__clinical_ref = pd.read_csv(file_name, index_col=0).sort_index()
                if not self.__clinical_ref.index.equals(self.data_container.GetFrame().index):
                    source_case = [case for case in self.__clinical_ref.index if case not in self.data_container.GetCaseName()]
                    dest_case = [case for case in self.data_container.GetCaseName() if
                                   case not in self.__clinical_ref.index]
                    QMessageBox.information(self, 'Error',
                                            'The index of clinical features is not consistent to the data. {} not in clinical infomations, and {} not in source features'.format(dest_case, source_case))
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

    def RemoveInvalidValue(self):
        if not self.data_container.IsEmpty():
            if self.checkExport.isChecked():
                dlg = QFileDialog()
                store_path, _ = dlg.getSaveFileName(self, 'Save CSV feature files', 'features.csv',
                                                   filter="CSV files (*.csv)")

                # folder_name = QFileDialog.getExistingDirectory(self, "Save Invalid data")
                # store_path = os.path.join(folder_name, 'invalid_feature.csv')
            else:
                store_path = ''

            if self.radioRemoveNonvalidCases.isChecked():
                self.data_container.RemoveInvalid(store_path=store_path, remove_index=REMOVE_CASE)
            elif self.radioRemoveNonvalidFeatures.isChecked():
                self.data_container.RemoveInvalid(store_path=store_path, remove_index=REMOVE_FEATURE)
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
            if self.testing_ref_data_container.IsEmpty():
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
        def _colnum_string(n):
            string = ""
            while n > 0:
                n, remainder = divmod(n - 1, 26)
                string = chr(65 + remainder) + string
            return string

        if self.data_container.IsEmpty():
            QMessageBox.warning(self, "Warning", "There is no data", QMessageBox.Ok)
            return None

        invalid_number_index = self.data_container.FindInvalidNumber()
        if invalid_number_index:
            old_edit_triggers = self.tableFeature.editTriggers()
            self.tableFeature.setEditTriggers(QAbstractItemView.CurrentChanged)
            self.tableFeature.setCurrentCell(invalid_number_index[0], invalid_number_index[1])
            self.tableFeature.setEditTriggers(old_edit_triggers)
            QMessageBox.warning(self, "Warning", "There are nan item in Row {}, Col {} ({})".format(
                invalid_number_index[0] + 2, invalid_number_index[1] + 2, _colnum_string(invalid_number_index[1] + 2)), QMessageBox.Ok)
            
            return None

        self.data_container.UpdateDataByFrame()

        if not self.data_container.IsBinaryLabel():
            QMessageBox.warning(self, "Warning", "There are not 2 Labels", QMessageBox.Ok)
            return None

        remove_features_with_same_value = RemoveSameFeatures()
        self.data_container = remove_features_with_same_value.Run(self.data_container)

        if self.radioSplitRandom.isChecked() or self.radioSplitRef.isChecked():
            folder_name = QFileDialog.getExistingDirectory(self, "Save data")
            if folder_name != '':
                data_separate = DataSeparate.DataSeparate()
                try:
                    if self.testing_ref_data_container.IsEmpty():
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
                                                                self.testing_ref_data_container,
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
                    print(traceback.format_exc())

        else:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save data", filter="csv files (*.csv)")
            if file_name:
                self.data_container.Save(file_name)
