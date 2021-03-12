import os
import sys

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui

from BC.GUI import Ui_FeatureExtraction
from BC.Utility.RadiomicsParamsConfig import RadiomicsParamsConfig
from BC.Utility.EcLog import eclog
from BC.Image2Feature.MyFeatureExtractor import MyFeatureExtractor


class FeatureExtractionForm(QWidget):
    close_signal = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.ui = Ui_FeatureExtraction()
        self._source_path = ''
        self._cur_pattern_item = 0

        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()
        folder, _ = os.path.split(os.path.abspath(sys.argv[0]))
        self.radiomics_config_path = folder + '/RadiomicsParams.yaml'

        self.ui.setupUi(self)
        self.ui.buttonBrowseSourceFolder.clicked.connect(self.BrowseSourceFolder)
        self.ui.buttonBrowseRoiFile.clicked.connect(self.BrowseRoiFile)
        self.ui.buttonAdd.clicked.connect(self.onButtonAddClicked)
        self.ui.listWidgetImageFiles.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.listWidgetImageFiles.customContextMenuRequested[QtCore.QPoint].connect(
            self.onListImageFilesContextMenuRequested)
        self.ui.buttonBrowseFile.clicked.connect(self.BrowsePatternFile)
        self.ui.buttonBrowseOutputFile.clicked.connect(self.BrowseOutputFile)
        self.radiomics_params = RadiomicsParamsConfig(self.radiomics_config_path)
        self.ui.buttonGo.clicked.connect(self.Go)
        self.InitUi()
        self.check_background_color = "background-color:rgba(255,0,0,64)"
        self.raw_background_color = "background-color:rgba(25,35,45,255)"

    def closeEvent(self, QCloseEvent):
        self.close_signal.emit(True)
        QCloseEvent.accept()

    def InitUi(self):
        self.radiomics_params.LoadConfig()
        image_types = self.radiomics_params.GetImageClasses()
        feature_classes = self.radiomics_params.GetFeatureClasses()
        self.UpdateImageClassesFeature(False, image_types)
        self.UpdateFeatureClasses(False, feature_classes)

    def UpdateImageClassesFeature(self, update_data, image_types):
        image_types_ctrl = {self.ui.checkBoxOriginal, self.ui.checkBoxWavelet, self.ui.checkBoxSquare,
                            self.ui.checkBoxSquareRoot, self.ui.checkBoxLoG, self.ui.checkBoxLogarithm,
                            self.ui.checkBoxExponential, self.ui.checkBoxGradient, self.ui.checkBoxLocalBinaryPattern2D,
                            self.ui.checkBoxLocalBinaryPattern3D}
        if update_data:
            for ctrl in image_types_ctrl:
                if ctrl.isChecked():
                    image_types.append(ctrl.text())
        else:
            for ctrl in image_types_ctrl:
                ctrl.setChecked(ctrl.text() in image_types)

    def UpdateFeatureClasses(self, update_data, feature_classes):
        feature_classes_ctrl = {self.ui.checkBoxFirstOrderStatistics, self.ui.checkBoxShapeBased2D,
                                self.ui.checkBoxGLCM, self.ui.checkBoxGLRLM, self.ui.checkBoxGLSZM,
                                self.ui.checkBoxGLDM, self.ui.checkBoxNGTDM}
        if update_data:
            for ctrl in feature_classes_ctrl:
                if ctrl.isChecked():
                    feature_classes.append(ctrl.text())
        else:
            for ctrl in feature_classes_ctrl:
                ctrl.setChecked(ctrl.text() in feature_classes)

    def onListImageFilesContextMenuRequested(self, point):
        current_item = self.ui.listWidgetImageFiles.currentItem()
        if current_item is None:
            return

        pop_menu = QtGui.QMenu()
        delete_action = QtGui.QAction(u'删除', self)
        pop_menu.addAction(delete_action)
        delete_action.triggered.connect(self.DeletePatternItem)
        modify_action = QtGui.QAction(u'修改', self)
        pop_menu.addAction(modify_action)
        modify_action.triggered.connect(self.ModifyPatternItem)
        cancel_modify_action = QtGui.QAction(u'取消修改', self)
        pop_menu.addAction(cancel_modify_action)
        cancel_modify_action.triggered.connect(self.CancelModifyPatternItem)
        pop_menu.exec_(QtGui.QCursor.pos())

    def BrowseSourceFolder(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)
        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]
            self.ui.lineEditSource.setText(self._root_folder)

    def BrowseRoiFile(self):
        dlg = QFileDialog()
        file_path, _ = dlg.getOpenFileName(None, 'Open ROI file', filter="All files (*.*)",
                                           options=QtGui.QFileDialog.DontUseNativeDialog)

        index = file_path.rfind('/')
        if index != -1:
            self.ui.lineEditRoiFileName.setText(file_path[index + 1:])

    def AddItem(self, item_text):
        item = QListWidgetItem(item_text)
        self.ui.listWidgetImageFiles.addItem(item)
        self._cur_pattern_item = self.ui.listWidgetImageFiles.count() - 1

    def onButtonAddClicked(self):
        file_pattern = self.ui.lineEditFileName.text()
        if len(file_pattern) > 0:
            if self.ui.buttonAdd.text() == 'modify':
                current_item = self.ui.listWidgetImageFiles.currentItem()
                current_item.setText(self.ui.lineEditFileName.text())
                self.ui.buttonAdd.setText('Add')
            else:
                self.AddItem(file_pattern)

        self.ui.lineEditFileName.setText('')

    def DeletePatternItem(self):
        try:
            current_item = self.ui.listWidgetImageFiles.currentItem()
            current_item = self.ui.listWidgetImageFiles.takeItem(self.ui.listWidgetImageFiles.row(current_item))
            self.ui.listWidgetImageFiles.removeItemWidget(current_item)
            items = self.ui.listWidgetImageFiles.count()
            print(items)
        except Exception as e:
            print(e.__str__())

    def ModifyPatternItem(self):
        current_item = self.ui.listWidgetImageFiles.currentItem()
        if current_item is not None:
            self.ui.buttonAdd.setText('Modify')
            text = current_item.text()
            self.ui.lineEditFileName.setText(text)

    def CancelModifyPatternItem(self):
        self.ui.buttonAdd.setText('Add')
        self.ui.lineEditFileName.setText('')

    def BrowsePatternFile(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open DICOM File with Pattern name',
                                           filter="DICOM files (*.dcm);;DICOM files (*.dic);;all files (*.*)",
                                           options=QtGui.QFileDialog.DontUseNativeDialog)
        print(file_name)
        index = file_name.rfind('/')
        if index != -1:
            self.AddItem(file_name[index + 1:])

    def BrowseOutputFile(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getSaveFileName(self, 'Save CSV feature files', 'features.csv',
                                           filter="CSV files (*.csv)")
        self.ui.lineEditOutputFile.setText(file_name)

    def SetColor(self, ctrl, check_pass):
        if check_pass:
            ctrl.setStyleSheet(self.raw_background_color)
        else:
            ctrl.setStyleSheet(self.check_background_color)

    def Check(self):
        check_ctrls = {self.ui.lineEditSource, self.ui.lineEditRoiFileName,
                       self.ui.lineEditOutputFile, self.ui.lineEditFeaturePrefix}
        result = True
        for ctrl in check_ctrls:
            result &= len(ctrl.text()) > 0
            self.SetColor(ctrl, len(ctrl.text()) > 0)

        result &= (self.ui.listWidgetImageFiles.count() > 0)
        self.SetColor(self.ui.listWidgetImageFiles, self.ui.listWidgetImageFiles.count() > 0)
        return result

    def Go(self):
        try:
            if self.Check():
                def GetKeyNameList():
                    name_list = list()
                    for index in range(self.ui.listWidgetImageFiles.count()):
                        name_list.append(self.ui.listWidgetImageFiles.item(index).text())
                    return name_list

                def UpdateRadiomicsConfig():
                    image_types = list()
                    self.UpdateImageClassesFeature(True, image_types)
                    feature_classes = list()
                    self.UpdateFeatureClasses(True, feature_classes)
                    self.radiomics_params.SetImageClasses(image_types)
                    self.radiomics_params.SetFeatureClasses(feature_classes)
                    self.radiomics_params.SaveConfig()

                UpdateRadiomicsConfig()
                omics_extractor = MyFeatureExtractor(self.radiomics_config_path, has_label=False)

                total_cases = len([temp for temp in os.listdir(self.ui.lineEditSource.text()) if os.path.isdir(self.ui.lineEditSource.text())])
                text = "Total {} cases are processed:\n".format(total_cases)
                for index, case_name in omics_extractor.Execute(self.ui.lineEditSource.text(), key_name_list=GetKeyNameList(),
                                        show_key_name_list=[self.ui.lineEditFeaturePrefix.text()],
                                        roi_key=[self.ui.lineEditRoiFileName.text()],
                                        store_path=self.ui.lineEditOutputFile.text()):
                    self.ui.progressBarTotal.setValue(int(index / total_cases * 100))
                    text += "Extracting features from {}\n".format(case_name)
                    self.ui.plainTextEditOutput.setPlainText(text)

        except Exception as e:
            print(e.__str__())











