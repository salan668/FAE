import os
import sys

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from GUI.FeatureExtraction import Ui_EeatureExtractionDialog
from Utility.EcLog import eclog
from FAE.Image2Feature.RadiomicsFeatureExtractor import RadiomicsFeatureExtractor

from PyQt5.QtCore import QItemSelectionModel, QModelIndex


class FeatureExactionConnection(QDialog, Ui_EeatureExtractionDialog):
    def __init__(self, parent=None):
        super(FeatureExactionConnection, self).__init__(parent)
        self.setupUi(self)
        self._root_folder = ''
        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()
        self.BrowseButton.clicked.connect(self.BrowseSourceFolder)
        self.ExtractionButton.clicked.connect(self.BeginExtraction)
        self.SelectButton.clicked.connect(self.BrowseOutputFile)
        self.check_background_color = "background-color:rgba(255,0,0,64)"
        self.raw_background_color = "background-color:rgba(255, 255, 255, 255)"
        folder, _ = os.path.split(os.path.abspath(sys.argv[0]))
        self.radiomics_config_path = folder + '/RadiomicsParams.yaml'
        self.extraction = False

    def BrowseSourceFolder(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)
        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]
            self.DataPathEdit.setText(self._root_folder)


    def BrowseOutputFile(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getSaveFileName(self, 'Save CSV feature files', 'features.csv',
                                           filter="CSV files (*.csv)")
        self.FeaturePathLineEdit.setText(file_name)


    def SetColor(self, ctrl, check_pass):
        if check_pass:
            ctrl.setStyleSheet(self.raw_background_color)
        else:
            ctrl.setStyleSheet(self.check_background_color)


    def Check(self):
        check_ctrls = {self.DataPathEdit, self.DatakeyLineEdit, self.RoiKeyLineEdit, self.FeaturePathLineEdit}
        result = True
        for ctrl in check_ctrls:
            result &= len(ctrl.text()) > 0
            self.SetColor(ctrl, len(ctrl.text()) > 0)

        return result

    def Execute(self, root_folder, key_name_list, roi_key, store_path, show_key_name_list=[]):
        if len(show_key_name_list) == 0:
            show_key_name_list = key_name_list
        assert (len(show_key_name_list) == len(key_name_list))

    def BeginExtraction(self):
        try:
            if self.Check():
                def GetNameList(name):
                    name_list = name.split(';')
                    return name_list


                omics_extractor = RadiomicsFeatureExtractor(self.radiomics_config_path, has_label=False, ignore_tolerence=True)
                total_cases = len([temp for temp in os.listdir(self.DataPathEdit.text()) if
                                   os.path.isdir(self.DataPathEdit.text())])
                total_cases *= len(GetNameList(self.RoiKeyLineEdit.text()))
                text = "Total {} cases are processed:\n".format(total_cases)
                self.OutputInfo.setText("Begin Extraction...")

                omics_extractor.Execute(self.DataPathEdit.text(), key_name_list=GetNameList(self.DatakeyLineEdit.text()),
                                                                roi_key=GetNameList(self.RoiKeyLineEdit.text()),
                                                                store_path=self.FeaturePathLineEdit.text())
                    #self.ExtractionProgressBar.setValue(int(index / total_cases * 100))
                self.OutputInfo.setText("Succeed!")
                self.extraction = True

        except Exception as e:
            print(e.__str__())
            self.OutputInfo.setText("Failed: {}".format(e.__str__()))
            self.extraction = False

    def SucceedExtraction(self):
        return self.extraction

    def GetExtractionFeaturePath(self):
        return self.FeaturePathLineEdit.text()

