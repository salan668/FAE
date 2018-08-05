
import numpy as np
from copy import deepcopy

from PyQt5.QtWidgets import *
from GUI.Prepare import Ui_Prepare

from FAE.DataContainer.DataContainer import DataContainer
from FAE.DataContainer import DataSeparate

from PyQt5.QtCore import QItemSelectionModel,QModelIndex
class PrepareConnection(QWidget, Ui_Prepare):
    def __init__(self, parent=None):
        super(PrepareConnection, self).__init__(parent)
        self.setupUi(self)
        self.data_container = DataContainer()

        self.buttonLoad.clicked.connect(self.LoadData)
        self.buttonRemove.clicked.connect(self.RemoveNonValidValue)
        self.checkSeparate.clicked.connect(self.SetSeparateStatus)
        self.spinBoxSeparate.setEnabled(False)

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
        except:
            print('Error')

        self.UpdateTable()

        self.buttonRemove.setEnabled(True)
        self.buttonSave.setEnabled(True)

    def RemoveNonValidValue(self):
        if self.radioRemoveNonvalidCases.isChecked():
            self.data_container.RemoveUneffectiveCases()
        elif self.radioRemoveNonvalidFeatures.isChecked():
            self.data_container.RemoveUneffectiveFeatures()

        self.UpdateTable()

    def SetSeparateStatus(self):
        if self.checkSeparate.isChecked():
            self.spinBoxSeparate.setEnabled(True)
        else:
            self.spinBoxSeparate.setEnabled(False)

    def CheckAndSave(self):
        if self.data_container.IsEmpty():
            QMessageBox.warning(self, "Warning", "There is no data", QMessageBox.Ok)
        elif self.data_container.HasNonValidNumber():
            QMessageBox.warning(self, "Warning", "There are nan items", QMessageBox.Ok)
            non_valid_number_Index = self.data_container.FindNonValidNumberIndex()
            old_edit_triggers = self.tableFeature.editTriggers()
            self.tableFeature.setEditTriggers(QAbstractItemView.CurrentChanged)
            self.tableFeature.setCurrentCell(non_valid_number_Index[0],non_valid_number_Index[1]+1)
            self.tableFeature.setEditTriggers(old_edit_triggers)
        elif self.checkSeparate.isChecked():
            percentage_testing_data = self.spinBoxSeparate.value()
            folder_name = QFileDialog.getExistingDirectory(self,"Save data")
            if folder_name != '':
                data_seperate = DataSeparate.DataSeparate(percentage_testing_data)
                data_seperate.Run(self.data_container,folder_name)
        else:
            file_name,_ = QFileDialog.getSaveFileName(self,"Save data",filter = "csv files (*.csv)")
            if file_name != '':
                self.data_container.Save(file_name)