import os
import sys

from PyQt5.QtWidgets import *
from PyQt5 import QtCore

from GUI.FeatureExtraction2 import Ui_FeatureExtraction
from Utility.RadiomicsParamsConfig import RadiomicsParamsConfig
from Utility.SeriesMatcher import SeriesStringMatcher
from FAE.Image2Feature.MyFeatureExtractor import MyFeatureExtractor


class FeatureExtractionForm(QWidget):
    close_signal = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(FeatureExtractionForm, self).__init__()
        self.ui = Ui_FeatureExtraction()
        self._root_folder = ''
        self._patterns = 0
        self._image_patten_list = []
        self._roi_patten_list = []
        self.radiomics_params = RadiomicsParamsConfig('RadiomicsParams.yaml')

        self.ui.setupUi(self)
        self.ui.tableFilePattern.setColumnCount(4)
        self.ui.tableFilePattern.setHorizontalHeaderLabels(["Type", "Name", "Include", "Exclude"])
        self.ui.tableFilePattern.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ui.tableFilePattern.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.ui.buttonBrowseSourceFolder.clicked.connect(self.BrowseSourceFolder)
        self.ui.buttonAddOne.clicked.connect(self.AddOnePattern)
        self.ui.buttonRemoveOne.clicked.connect(self.RemoveOnePattern)
        self.ui.buttonRun.clicked.connect(self.Run)
        self.ui.UseExistConfigcheckBox.clicked.connect(self.UseExitingConfig)
        self.ui.ConfigPushButton.clicked.connect(self.BrowseRadiomicsFeatureCofigFile)
        self.ui.CofnigLineEdit.setText(r'D:\research\exampleMR_NoResampling.yaml')
        self.ui.CofnigLineEdit.setEnabled(False)
        self.ui.ConfigPushButton.setEnabled(False)


    def closeEvent(self, event):
        self.close_signal.emit(True)
        event.accept()

    def BrowseSourceFolder(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)
        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]
            self.ui.lineEditSourceFolder.setText(self._root_folder)

    def _PatternNameExist(self, one_name):
        exist_name_list = [item['name'] for item in self._image_patten_list]
        if one_name in exist_name_list:
            return True
        else:
            return False

    def AddOnePattern(self):
        message = QMessageBox()
        if self.ui.lineEditkeyInclude.text() == '':
            message.about(self, 'Include can not be empty',
                                'Include patterns are used to identify the file')
            return
        if self.ui.lineEditkeyShowName.text() == '':
            message.about(self, 'ShowName can not be empty',
                          'ShowName patterns are used to add pre-name in the feature matrix')
            return

        one_pattern = {'name': self.ui.lineEditkeyShowName.text().split(','),
                       'include': self.ui.lineEditkeyInclude.text().split(','),
                       'exclude': self.ui.lineEditkeyExclude.text().split(',')}
        if self.ui.radioImagePattern.isChecked() and self._PatternNameExist(one_pattern['name']):
            message.about(self, '', 'Same image pattern exists')
            return
        elif self.ui.radioRoiPattern.isChecked() and self._roi_patten_list:
            message.about(self, '', 'There should only one ROI pattern')
            return

        self.ui.tableFilePattern.insertRow(self._patterns)
        if self.ui.radioImagePattern.isChecked():
            self.ui.tableFilePattern.setItem(self._patterns, 0, QTableWidgetItem('Image'))
            self._image_patten_list.append(one_pattern)
        elif self.ui.radioRoiPattern.isChecked():
            self.ui.tableFilePattern.setItem(self._patterns, 0, QTableWidgetItem('ROI'))
            self._roi_patten_list.append(one_pattern)
        self.ui.tableFilePattern.setItem(self._patterns, 1, QTableWidgetItem(','.join(one_pattern['name'])))
        self.ui.tableFilePattern.setItem(self._patterns, 2, QTableWidgetItem(','.join(one_pattern['include'])))
        self.ui.tableFilePattern.setItem(self._patterns, 3, QTableWidgetItem(','.join(one_pattern['exclude'])))

        self._patterns += 1

    def RemoveOnePattern(self):
        if not self.ui.tableFilePattern.selectedIndexes():
            return None
        index = self.ui.tableFilePattern.selectedIndexes()[0].row()

        image_type = self.ui.tableFilePattern.item(index, 0).text()
        if image_type == 'Image':
            one_pattern = {'name': self.ui.tableFilePattern.item(index, 1).text().split(','),
                           'include': self.ui.tableFilePattern.item(index, 2).text().split(','),
                           'exclude': self.ui.tableFilePattern.item(index, 3).text().split(',')}
            self._image_patten_list.remove(one_pattern)
        elif image_type == 'ROI':
            self._roi_patten_list = []

        self._patterns -= 1
        self.ui.tableFilePattern.removeRow(index)

    def _GetImageAndRoiMatcher(self):
        series_matchers = {}
        for one_pattern in self._image_patten_list:
            series_matchers[one_pattern['name'][0]] = SeriesStringMatcher(include_key=one_pattern['include'],
                                                                       exclude_key=one_pattern['exclude'])
        roi_matcher = SeriesStringMatcher(include_key=self._roi_patten_list[0]['include'],
                                          exclude_key=self._roi_patten_list[0]['exclude'])
        return series_matchers, roi_matcher

    def _CheckFiles(self):
        if len(self._roi_patten_list) == 0:
            self._missing_message += 'Must have 1 Roi Pattern. \n'
            return False
        if len(self._image_patten_list) == 0:
            self._missing_message += 'Must have at least 1 Image Pattern. \n'
            return False

        all_match = True
        series_matchers, roi_matcher = self._GetImageAndRoiMatcher()

        for case_name in os.listdir(self._root_folder):
            case_folder = os.path.join(self._root_folder, case_name)
            if not os.path.isdir(case_folder):
                continue
            series_list = os.listdir(case_folder)

            # Match series according to patterns
            for key, one_series_matcher in series_matchers.items():
                result = one_series_matcher.Match(series_list)
                if len(result) != 1:
                    self._missing_message += '{} does not match {}\n'.format(case_name, key)
                    all_match = False
            result = roi_matcher.Match(series_list)
            if len(result) != 1:
                self._missing_message += '{} does not match ROI\n'.format(case_name)
                all_match = False

        return all_match

    def _UpdateImageClassesFeature(self, update_data, image_types):
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

    def _UpdateFeatureClasses(self, update_data, feature_classes):
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

    def UseExitingConfig(self):
         use_exist = self.ui.UseExistConfigcheckBox.isChecked()
         self.ui.CofnigLineEdit.setEnabled(use_exist)
         self.ui.ConfigPushButton.setEnabled(use_exist)

    def BrowseRadiomicsFeatureCofigFile(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open Radiomics Config file', directory=r'D:\research',
                                           filter="Config (*.yaml)")
        if file_name:
            self._radiomics_file = file_name

    def Run(self):
        try:
            def UpdateRadiomicsConfig():
                image_types = list()
                self._UpdateImageClassesFeature(True, image_types)
                feature_classes = list()
                self._UpdateFeatureClasses(True, feature_classes)
                self.radiomics_params.SetImageClasses(image_types)
                self.radiomics_params.SetFeatureClasses(feature_classes)
                self.radiomics_params.SaveConfig()

            dlg = QFileDialog()
            file_name, _ = dlg.getSaveFileName(self, 'Save CSV feature files', 'features.csv',
                                               filter="CSV files (*.csv)")
            if not file_name:
                return None

            self.ui.plainTextOutput.setPlainText('Checking Files ...')
            QApplication.processEvents()

            self._missing_message = ''
            if not self._CheckFiles():
                self.ui.plainTextOutput.appendPlainText(self._missing_message)
                self.ui.plainTextOutput.appendPlainText("Please Check the Data")
                QApplication.processEvents()
                return

            self.ui.plainTextOutput.appendPlainText('Done')
            QApplication.processEvents()

            UpdateRadiomicsConfig()
            if self.ui.UseExistConfigcheckBox.isChecked():
                extractor = MyFeatureExtractor('RadiomicsParams.yaml')
            else:
                extractor = MyFeatureExtractor()

            series_matchers, roi_matcher = self._GetImageAndRoiMatcher()
            name_list, matcher_list = [], []
            for name, matcher in series_matchers.items():
                name_list.append(name)
                matcher_list.append(matcher)

            total_cases = len([temp for temp in os.listdir(self._root_folder) if
                               os.path.isdir(os.path.join(self._root_folder, temp))])
            self.ui.plainTextOutput.appendPlainText("\n\nTotal {} cases are processed:\n".format(total_cases))

            for index, case_name in extractor.Execute(self._root_folder,
                                                      image_matcher_list=matcher_list,
                                                      roi_matcher=roi_matcher,
                                                      show_name_list=name_list,
                                                      store_path=file_name):
                self.ui.plainTextOutput.appendPlainText('{} Done ({}/{})\n'.format(case_name, index, total_cases))
                QApplication.processEvents()

        except Exception as e:
            print(e.__str__())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_frame = FeatureExtractionForm()
    main_frame.show()
    sys.exit(app.exec_())
