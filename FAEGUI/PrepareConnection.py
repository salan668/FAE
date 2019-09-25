import numpy as np
import csv
import os
from copy import deepcopy

from PyQt5.QtWidgets import *
from GUI.Prepare import Ui_Prepare
from Utility.EcLog import eclog

from FAE.DataContainer.DataContainer import DataContainer
from FAE.DataContainer import DataSeparate
from FAE.FeatureAnalysis.FeatureSelector import RemoveSameFeatures
from FAE.DataContainer.DataBalance import UpSampling, DownSampling, SmoteSampling, DataBalance

from PyQt5.QtCore import QItemSelectionModel,QModelIndex


class PrepareConnection(QWidget, Ui_Prepare):
    def __init__(self, parent=None):
        super(PrepareConnection, self).__init__(parent)
        self.setupUi(self)
        self.data_container = DataContainer()

        self.buttonLoad.clicked.connect(self.LoadData)
        self.buttonRemove.clicked.connect(self.RemoveNonValidValue)
        self.loadTestingReference.clicked.connect(self.LoadTestingReferenceDataContainer)
        self.clearTestingReference.clicked.connect(self.ClearTestingReferenceDataContainer)
        self.__testing_ref_data_container = DataContainer()
        self.checkSeparate.clicked.connect(self.SetSeparateStatus)

        self.spinBoxSeparate.setEnabled(False)
        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()

        self.loadTestingReference.setEnabled(False)
        self.clearTestingReference.setEnabled(False)

        self.buttonSave.clicked.connect(self.CheckAndSave)

    def UpdateTable(self):
        if self.data_container.GetArray().size == 0:
            return

        self.tableFeature.setRowCount(len(self.data_container.GetCaseName()))
        header_name = deepcopy(self.data_container.GetFeatureName())
        header_name.insert(0, 'Label')
        self.tableFeature.setColumnCount(len(header_name))
        self.tableFeature.setHorizontalHeaderLabels(header_name)
        self.tableFeature.setVerticalHeaderLabels(list(map(str, self.data_container.GetCaseName())))

        for row_index in range(len(self.data_container.GetCaseName())):
            for col_index in range(len(header_name)):
                if col_index == 0:
                    self.tableFeature.setItem(row_index, col_index, QTableWidgetItem(str(self.data_container.GetLabel()[row_index])))
                else:
                    self.tableFeature.setItem(row_index, col_index, QTableWidgetItem(str(self.data_container.GetArray()[row_index, col_index - 1])))



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
            self.logger.info('Open the file ' + file_name + ' Succeed.')
        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            QMessageBox.about(self, 'Load data Error', reason.__str__())
            print('Error！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',                                    'The selected data file mismatch.')
        self.UpdateTable()

        self.buttonRemove.setEnabled(True)
        self.buttonSave.setEnabled(True)

    def LoadTestingReferenceDataContainer(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open SCV file', filter="csv files (*.csv)")
        try:
            self.__testing_ref_data_container.Load(file_name)
            self.loadTestingReference.setEnabled(False)
            self.clearTestingReference.setEnabled(True)
            self.spinBoxSeparate.setEnabled(False)
        except OSError as reason:
            self.logger.log('Load Testing Reference Error: ' + str(reason))
            print('Error！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected data file mismatch.')

    def ClearTestingReferenceDataContainer(self):
        del self.__testing_ref_data_container
        self.__testing_ref_data_container = DataContainer()
        self.loadTestingReference.setEnabled(True)
        self.clearTestingReference.setEnabled(False)
        self.spinBoxSeparate.setEnabled(False)

    def RemoveNonValidValue(self):
        if self.radioRemoveNonvalidCases.isChecked():
            self.data_container.RemoveUneffectiveCases()
        elif self.radioRemoveNonvalidFeatures.isChecked():
            self.data_container.RemoveUneffectiveFeatures()

        self.UpdateTable()

    def SetSeparateStatus(self):
        if self.checkSeparate.isChecked():
            self.spinBoxSeparate.setEnabled(True)
            self.loadTestingReference.setEnabled(True)
            self.clearTestingReference.setEnabled(False)
        else:
            self.spinBoxSeparate.setEnabled(False)
            self.loadTestingReference.setEnabled(False)
            self.clearTestingReference.setEnabled(False)

    def CheckAndSave(self):
        if self.data_container.IsEmpty():
            QMessageBox.warning(self, "Warning", "There is no data", QMessageBox.Ok)
        elif not self.data_container.IsBinaryLabel():
            QMessageBox.warning(self, "Warning", "There are not 2 Labels", QMessageBox.Ok)
            non_valid_number_Index = self.data_container.FindNonValidLabelIndex()
            old_edit_triggers = self.tableFeature.editTriggers()
            self.tableFeature.setEditTriggers(QAbstractItemView.CurrentChanged)
            self.tableFeature.setCurrentCell(non_valid_number_Index, 0)
            self.tableFeature.setEditTriggers(old_edit_triggers)
        elif self.data_container.HasNonValidNumber():
            QMessageBox.warning(self, "Warning", "There are nan items", QMessageBox.Ok)
            non_valid_number_Index = self.data_container.FindNonValidNumberIndex()
            old_edit_triggers = self.tableFeature.editTriggers()
            self.tableFeature.setEditTriggers(QAbstractItemView.CurrentChanged)
            self.tableFeature.setCurrentCell(non_valid_number_Index[0], non_valid_number_Index[1]+1)
            self.tableFeature.setEditTriggers(old_edit_triggers)
        else:
            remove_features_with_same_value = RemoveSameFeatures()
            self.data_container = remove_features_with_same_value.Run(self.data_container)

            data_balance = DataBalance()
            if self.radioDownSampling.isChecked():
                data_balance = DownSampling()
            elif self.radioUpSampling.isChecked():
                data_balance = UpSampling()
            elif self.radioSmote.isChecked():
                data_balance = SmoteSampling()

            if self.checkSeparate.isChecked():
                folder_name = QFileDialog.getExistingDirectory(self, "Save data")
                if folder_name != '':
                    data_separate = DataSeparate.DataSeparate()
                    try:
                        if self.__testing_ref_data_container.IsEmpty():
                            testing_data_percentage = self.spinBoxSeparate.value()
                            training_data_container, _, = data_separate.RunByTestingPercentage(self.data_container,
                                                                                               testing_data_percentage,
                                                                                               folder_name)
                        else:
                            training_data_container, _, = data_separate.RunByTestingReference(self.data_container,
                                                                                              self.__testing_ref_data_container,
                                                                                              folder_name)
                            if training_data_container.IsEmpty():
                                QMessageBox.information(self, 'Error',
                                                        'The testing data does not mismatch, please check the testing data '
                                                        'really exists in current data')
                                return None
                        data_balance.Run(training_data_container, store_path=folder_name)
                    except Exception as e:
                        content = 'PrepareConnection, splitting failed: '
                        self.logger.error('{}{}'.format(content, str(e)))
                        QMessageBox.about(self, content, e.__str__())


            else:
                file_name, _ = QFileDialog.getSaveFileName(self, "Save data", filter="csv files (*.csv)")
                if file_name != '':
                    data_balance.Run(self.data_container, store_path=file_name)
