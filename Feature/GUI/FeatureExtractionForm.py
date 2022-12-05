"""
All rights reserved.
Author: Yang SONG (songyangmri@gmail.com)
"""
import traceback
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import QThread

from Feature.GUI.FeatureExtraction import Ui_FeatureExtraction
from Feature.RadiomicsParamsConfig import RadiomicsParamsConfig, feature_classes_inface2yaml, image_classes_inface2yaml
from Feature.SeriesMatcher import SeriesStringMatcher
from radiomics.featureextractor import RadiomicsFeatureExtractor

from Feature.FileMatcher import UniqueFileMatcherManager


class FileCheckerThread(QThread):
    progress_signal = QtCore.pyqtSignal(int)
    text_signal = QtCore.pyqtSignal(str)
    finish_signal = QtCore.pyqtSignal(bool)

    def __init__(self, image_match_manager, roi_match_manager, root_folder):
        super().__init__()
        self.image_match_manager = image_match_manager
        self.roi_match_manager = roi_match_manager
        self.root = root_folder

    def MatchOneManager(self, manager, case_number, message):
        self.finish_signal.emit(False)
        self.text_signal.emit(message)
        self.progress_signal.emit(0)

        count = 0
        for state, case_name, series_name in manager.MatchVerbose(self.root):
            if not state:
                message += '{}-{} {}'.format(case_name, series_name, manager.error_info.loc[case_name, series_name])

            self.progress_signal.emit(int(100 * count / case_number))
            self.text_signal.emit(message)

        self.text_signal.emit(message + '\nAll Done.')
        self.progress_signal.emit(100)
        self.finish_signal.emit(True)

        return message

    def run(self):
        case_number = self.image_match_manager.EstimateCaseNumber(self.root)
        message = '\nChecking the Image Path: \n'
        message = self.MatchOneManager(self.image_match_manager, case_number, message)
        message += '\n\nChecking the Roi Path: \n'
        self.MatchOneManager(self.roi_match_manager, case_number, message)

        self.quit()


class FeatureExtractThread(QThread):
    progress_signal = QtCore.pyqtSignal(int)
    text_signal = QtCore.pyqtSignal(str)
    finish_signal = QtCore.pyqtSignal(bool)

    def __init__(self, image_paths, roi_paths, store_path, extractor: RadiomicsFeatureExtractor,
                 only_matrix: bool):
        super().__init__()
        self.image_paths = image_paths
        self.roi_paths = roi_paths
        self.case_number = roi_paths.size
        self.store_path = store_path
        self.extractor = extractor

        self.only_matrix = only_matrix

    def run(self):
        self.finish_signal.emit(False)

        message = 'Extract Featuresing: \n'
        self.text_signal.emit(message)
        self.progress_signal.emit(0)

        all_features = pd.DataFrame()
        count = 0

        for case_name in self.image_paths.index:
            self.text_signal.emit(message)
            one_case_feature = {}
            try:
                for image_name in self.image_paths.columns:
                    roi_image = sitk.ReadImage(str(self.roi_paths.loc[case_name, 'FAE_ROI']))
                    image = sitk.ReadImage(str(self.image_paths.loc[case_name, image_name]))

                    if self.only_matrix:
                        assert (image.GetSize() == roi_image.GetSize())
                        roi_image.CopyInformation(image)

                    for key, value in self.extractor.execute(image, roi_image).items():
                        if 'diagnostics' not in key:
                            one_case_feature['{}_{}'.format(image_name, key)] = value

                all_features = pd.concat([all_features, pd.DataFrame(one_case_feature, index=[case_name])], axis=0)
                self.text_signal.emit(message)
            except Exception as e:
                message += '{}.\n'.format(e.__str__())
                message += '{}.\n'.format(traceback.format_exc())
                self.text_signal.emit(message)
            count += 1
            self.progress_signal.emit(100 * count / self.case_number)

        try:
            all_features.to_csv(str(self.store_path))
            self.text_signal.emit(message + '\nAll Done')
        except PermissionError:
            self.text_signal.emit(message + '\nThe File can not be saved, maybe the feature matrix does not close.')
        except Exception as e:
            self.text_signal.emit(message + '\n{}'.format(traceback.format_exc()))

        self.progress_signal.emit(100)
        self.finish_signal.emit(True)
        self.quit()


class FeatureExtractionForm(QWidget):
    close_signal = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(FeatureExtractionForm, self).__init__()
        self.ui = Ui_FeatureExtraction()
        self._root_folder = ''
        self._patterns = 0
        self._image_patten_list = []
        self._roi_patten_list = []
        self._missing_message = ''

        self.image_matcher_manager = UniqueFileMatcherManager()
        self.roi_matcher_manager = UniqueFileMatcherManager()

        self.radiomics_params = RadiomicsParamsConfig(r'Feature\GUI\RadiomicsParams.yaml')

        self.ui.setupUi(self)
        self.ui.tableFilePattern.setColumnCount(4)
        self.ui.tableFilePattern.setHorizontalHeaderLabels(["Type", "Name", "Include", "Exclude"])
        self.ui.tableFilePattern.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ui.tableFilePattern.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.ui.radioImagePattern.clicked.connect(self.UpdatePatternInterface)
        self.ui.radioRoiPattern.clicked.connect(self.UpdatePatternInterface)
        self.ui.buttonBrowseSourceFolder.clicked.connect(self.LoadDataRoot)
        self.ui.buttonAddOne.clicked.connect(self.AddOnePattern)
        self.ui.buttonRemoveOne.clicked.connect(self.RemoveOnePattern)

        self.ui.useExistConfigcheckBox.clicked.connect(self.UpdateInterfaceByUseExitingConfig)
        self.ui.buttonLoadConfig.clicked.connect(self.BrowseRadiomicsFeatureCofigFile)
        self.ui.configLineEdit.setText(r'')
        self.ui.checkNormalize.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.checkResample.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.checkResegment.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.radioResegmentAbsolute.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.radioResegmenSigma.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.checkRoiFilter.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.radioBinCount.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.radioBinWidth.clicked.connect(self.UpdateFeatureConfigInterface)
        self.ui.buttonSaveConfig.clicked.connect(self.SaveParamConfig)

        self.ui.buttonCheckPath.clicked.connect(self.CheckPath)
        self.ui.buttonExtract.clicked.connect(self.FeatureExtract)

        self.UpdateInterfaceByUseExitingConfig()

    def closeEvent(self, event):
        self.close_signal.emit(True)
        event.accept()

    def UpdateProgressBar(self, value):
        self.ui.progressBar.setValue(value)

    def LoadDataRoot(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)
        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]
            self.ui.lineEditSourceFolder.setText(self._root_folder)

    def UpdatePatternInterface(self):
        self.ui.lineEditStoreName.setEnabled(not self.ui.radioRoiPattern.isChecked())

    def _PatternNameExist(self, one_name):
        exist_name_list = [item['name'] for item in self._image_patten_list]
        if one_name in exist_name_list:
            return True
        else:
            return False

    def AddOneImagePattern(self, name, patten_matcher):
        message = QMessageBox()
        if name == '':
            message.about(self, 'ShowName can not be empty', 'ShowName can not be empty')
            return False
        if name in self.image_matcher_manager.matchers.keys():
            message.about(self, 'Same patten has been saved', 'Same patten has been saved')
            return False

        self.image_matcher_manager.AddOne(name, patten_matcher)
        return True

    def AddRoiPattern(self, patten_matcher):
        message = QMessageBox()
        if len(self.roi_matcher_manager.matchers) > 0:
            message.about(self, 'Only One Roi Pattern', 'There should be only one ROI Pattern')
            return False
        self.roi_matcher_manager.AddOne('FAE_ROI', patten_matcher)
        return True

    def AddOnePattern(self):
        message = QMessageBox()
        if self.ui.lineEditkeyInclude.text() == '':
            message.about(self, 'Include can not be empty',
                          'Include patterns are used to identify the file')
            return

        one_pattern = SeriesStringMatcher(include_key=self.ui.lineEditkeyInclude.text().split(','),
                                          exclude_key=self.ui.lineEditkeyExclude.text().split(','))
        add_result, add_type, name = False, '', ''
        if self.ui.radioImagePattern.isChecked():
            name = self.ui.lineEditStoreName.text()
            add_result = self.AddOneImagePattern(name, one_pattern)
            add_type = 'Image'
        elif self.ui.radioRoiPattern.isChecked():
            name = 'ROI'
            add_result = self.AddRoiPattern(one_pattern)
            add_type = 'Roi'

        if add_result:
            current_row_count = self.ui.tableFilePattern.rowCount()
            self.ui.tableFilePattern.insertRow(current_row_count)
            self.ui.tableFilePattern.setItem(current_row_count, 0, QTableWidgetItem(add_type))
            self.ui.tableFilePattern.setItem(current_row_count, 1, QTableWidgetItem(name))
            self.ui.tableFilePattern.setItem(current_row_count, 2, QTableWidgetItem(','.join(one_pattern.include_key)))
            self.ui.tableFilePattern.setItem(current_row_count, 3, QTableWidgetItem(','.join(one_pattern.exclude_key)))

    def RemoveOnePattern(self):
        if not self.ui.tableFilePattern.selectedIndexes():
            return None

        index = self.ui.tableFilePattern.selectedIndexes()[0].row()
        name = self.ui.tableFilePattern.item(index, 1).text()

        image_type = self.ui.tableFilePattern.item(index, 0).text()
        if image_type == 'Image':
            self.image_matcher_manager.RemoveOne(name)
        elif image_type == 'Roi':
            self.roi_matcher_manager.ClearMatcher()

        self.ui.tableFilePattern.removeRow(index)

    def CheckPath(self):
        if len(self.image_matcher_manager.matchers) == 0 or len(self.roi_matcher_manager.matchers) == 0:
            QMessageBox().about(self, '', 'At least one image pattern and one roi pattern')
            return

        self.image_matcher_manager.Clear()
        self.roi_matcher_manager.Clear()

        self.file_check_thread = FileCheckerThread(self.image_matcher_manager,
                                                   self.roi_matcher_manager, Path(self._root_folder))
        self.file_check_thread.progress_signal.connect(self.UpdateProgressBar)
        self.file_check_thread.text_signal.connect(self.ui.plainTextOutput.setPlainText)
        self.file_check_thread.start()

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
        feature_classes_ctrl = {self.ui.checkBoxFirstOrderStatistics, self.ui.checkBoxShape,
                                self.ui.checkBoxGLCM, self.ui.checkBoxGLRLM, self.ui.checkBoxGLSZM,
                                self.ui.checkBoxGLDM, self.ui.checkBoxNGTDM}
        if update_data:
            for ctrl in feature_classes_ctrl:
                if ctrl.isChecked():
                    feature_classes.append(ctrl.text())
        else:
            for ctrl in feature_classes_ctrl:
                ctrl.setChecked(ctrl.text() in feature_classes)

    def IsCheckAndEnable(self, wdiget):
        return wdiget.isEnabled() and wdiget.isChecked()

    def UpdateFeatureConfigInterface(self):
        self.ui.spinNormalizeScale.setEnabled(self.IsCheckAndEnable(self.ui.checkNormalize))
        self.ui.doubleSpinResampleX.setEnabled(self.IsCheckAndEnable(self.ui.checkResample))
        self.ui.doubleSpinResampleY.setEnabled(self.IsCheckAndEnable(self.ui.checkResample))
        self.ui.doubleSpinResampleZ.setEnabled(self.IsCheckAndEnable(self.ui.checkResample))
        self.ui.radioResegmentAbsolute.setEnabled(self.IsCheckAndEnable(self.ui.checkResegment))
        self.ui.radioResegmenSigma.setEnabled(self.IsCheckAndEnable(self.ui.checkResegment))
        self.ui.doubleSpinReSegmentMin.setEnabled(self.IsCheckAndEnable(self.ui.radioResegmentAbsolute))
        self.ui.doubleSpinReSegmentMax.setEnabled(self.IsCheckAndEnable(self.ui.radioResegmentAbsolute))
        self.ui.spinMinRoiSize.setEnabled(self.IsCheckAndEnable(self.ui.checkRoiFilter))
        self.ui.doubleSpinBinWidth.setEnabled(self.IsCheckAndEnable(self.ui.radioBinWidth))
        self.ui.spinBinCount.setEnabled(self.IsCheckAndEnable(self.ui.radioBinCount))

    def UpdateInterfaceByUseExitingConfig(self):
        use_exist = self.ui.useExistConfigcheckBox.isChecked()
        self.ui.configLineEdit.setEnabled(use_exist)
        self.ui.buttonLoadConfig.setEnabled(use_exist)

        self.ui.checkNormalize.setEnabled(not use_exist)
        self.ui.checkResample.setEnabled(not use_exist)
        self.ui.checkResegment.setEnabled(not use_exist)
        self.ui.checkRoiFilter.setEnabled(not use_exist)
        self.ui.radioBinCount.setEnabled(not use_exist)
        self.ui.radioBinWidth.setEnabled(not use_exist)
        self.ui.checkForce2DExtraction.setEnabled(not use_exist)
        self.ui.checkCorrectMask.setEnabled(not use_exist)
        self.ui.spinLabelValue.setEnabled(not use_exist)
        self.ui.buttonSaveConfig.setEnabled(not use_exist)

        self.ui.checkBoxOriginal.setEnabled(not use_exist)
        self.ui.checkBoxWavelet.setEnabled(not use_exist)
        self.ui.checkBoxSquare.setEnabled(not use_exist)
        self.ui.checkBoxSquareRoot.setEnabled(not use_exist)
        self.ui.checkBoxLoG.setEnabled(not use_exist)
        self.ui.checkBoxLogarithm.setEnabled(not use_exist)
        self.ui.checkBoxExponential.setEnabled(not use_exist)
        self.ui.checkBoxGradient.setEnabled(not use_exist)
        self.ui.checkBoxLocalBinaryPattern2D.setEnabled(not use_exist)
        self.ui.checkBoxLocalBinaryPattern3D.setEnabled(not use_exist)

        self.ui.checkBoxFirstOrderStatistics.setEnabled(not use_exist)
        self.ui.checkBoxShape.setEnabled(not use_exist)
        self.ui.checkBoxGLCM.setEnabled(not use_exist)
        self.ui.checkBoxGLRLM.setEnabled(not use_exist)
        self.ui.checkBoxGLSZM.setEnabled(not use_exist)
        self.ui.checkBoxGLDM.setEnabled(not use_exist)
        self.ui.checkBoxNGTDM.setEnabled(not use_exist)

        self.UpdateFeatureConfigInterface()

        if not use_exist:
            self.radiomics_params = RadiomicsParamsConfig(r'Feature\GUI\RadiomicsParams.yaml')

    def BrowseRadiomicsFeatureCofigFile(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open Radiomics Config file', directory='',
                                           filter="Config (*.yaml)")
        if file_name:
            self.ui.configLineEdit.setText(file_name)
            self.radiomics_params = RadiomicsParamsConfig(file_name)

    def SaveParamConfig(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getSaveFileName(self, 'Open Radiomics Config file', 'radiomics_config_param.yaml',
                                           filter="Config (*.yaml)")
        if file_name:
            config_dict = self.UpdateConfigFile()
            RadiomicsParamsConfig(file_name).SaveConfig(config_dict)

    def UpdateConfigFile(self):
        config_dict = {'featureClass': {}, 'imageType': {}, 'setting': {}}
        for image_type in [self.ui.checkBoxOriginal, self.ui.checkBoxWavelet, self.ui.checkBoxSquare,
                            self.ui.checkBoxSquareRoot, self.ui.checkBoxLoG, self.ui.checkBoxLogarithm,
                            self.ui.checkBoxExponential, self.ui.checkBoxGradient, self.ui.checkBoxLocalBinaryPattern2D,
                            self.ui.checkBoxLocalBinaryPattern3D]:
            if image_type.isChecked():
                config_dict['imageType'][image_classes_inface2yaml[image_type.text()]] = {}
        for feature_type in [self.ui.checkBoxFirstOrderStatistics, self.ui.checkBoxShape,
                                self.ui.checkBoxGLCM, self.ui.checkBoxGLRLM, self.ui.checkBoxGLSZM,
                                self.ui.checkBoxGLDM, self.ui.checkBoxNGTDM]:
            if feature_type.isChecked():
                config_dict['featureClass'][feature_classes_inface2yaml[feature_type.text()]] = None

        # Set the settings
        if self.ui.checkNormalize.isChecked():
            config_dict['setting']['normalize'] = True
            config_dict['setting']['normalizeScale'] = round(self.ui.spinNormalizeScale.value(), 2)
        if self.ui.checkResample.isChecked():
            config_dict['setting']['resampledPixelSpacing'] = [round(self.ui.doubleSpinResampleX.value(), 2),
                                                                round(self.ui.doubleSpinResampleY.value(), 2),
                                                                round(self.ui.doubleSpinResampleZ.value(), 2)]
        if self.ui.checkResegment.isChecked():
            if self.ui.radioResegmentAbsolute.isChecked():
                config_dict['setting']['resegmentMode'] = 'absolute'
                config_dict['setting']['resegmentRange'] = [round(self.ui.doubleSpinReSegmentMin.value(), 2),
                                                             round(self.ui.doubleSpinReSegmentMax.value(), 2)]
            elif self.ui.radioResegmenSigma.isChecked():
                config_dict['setting']['resegmentMode'] = 'sigma'
        if self.ui.checkRoiFilter.isChecked():
            config_dict['setting']['minimumROISize'] = self.ui.spinMinRoiSize.value()
        if self.ui.checkForce2DExtraction.isChecked():
            config_dict['setting']['force2D'] = True
        if self.ui.checkCorrectMask.isChecked():
            config_dict['setting']['correctMask'] = True
        if self.ui.radioBinWidth.isChecked():
            config_dict['setting']['binWidth'] = round(self.ui.doubleSpinBinWidth.value(), 2)
        elif self.ui.radioBinCount.isChecked():
            config_dict['setting']['binCount'] = self.ui.spinBinCount.value()

        config_dict['setting']['label'] = self.ui.spinLabelValue.value()

        config_dict['setting']['geometryTolerance'] = '1e-6'
        config_dict['setting']['additionalInfo'] = False

        return config_dict

    def FeatureExtract(self):
        if not (self.image_matcher_manager.IsAllMatched() and self.roi_matcher_manager.IsAllMatched()):
            QMessageBox().about(self, '', 'Check the File first')
            return

        dlg = QFileDialog()
        file_name, _ = dlg.getSaveFileName(self, 'Save CSV feature files', 'radiomics_features.csv',
                                           filter="CSV files (*.csv)")
        if not file_name:
            return None

        if not self.ui.useExistConfigcheckBox.isChecked():
            config_dict = self.UpdateConfigFile()
            self.radiomics_params.SaveConfig(config_dict)

        self.analysis_thread = FeatureExtractThread(
            self.image_matcher_manager.results, self.roi_matcher_manager.results, file_name, RadiomicsFeatureExtractor(self.radiomics_params.config_path), self.ui.radioExtractCopyInfo.isChecked()
        )

        self.analysis_thread.progress_signal.connect(self.UpdateProgressBar)
        self.analysis_thread.text_signal.connect(self.ui.plainTextOutput.setPlainText)
        self.analysis_thread.start()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_frame = FeatureExtractionForm()
    main_frame.show()
    sys.exit(app.exec_())
