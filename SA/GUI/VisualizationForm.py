"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/27
"""
import os
import traceback

import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5 import QtCore

from SA.Utility import mylog
from SA.Utility.Constant import *
from SA.PipelineManager import PipelineManager
from SA.Fitter import BaseFitter

from SA.GUI.Visualization import Ui_Visualization
from SA.Utility.Visualization import SurvivalPlot, DrawIndex, ModelHazardRatio


def CheckTextInCombo(text, combo):
    return text in [combo.itemText(ind) for ind in range(combo.count())]


class VisualizationForm(QWidget, Ui_Visualization):
    close_signal = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        self._root_folder = ''
        self.sae = PipelineManager()
        self.sheet_dict = dict()
        self.models_name = []
        self._current_fitter = BaseFitter()
        self._is_loading = False

        self.ref_df = pd.DataFrame()

        super(VisualizationForm, self).__init__(parent)
        self.setupUi(self)

        self.buttonLoadResult.clicked.connect(self.LoadResult)
        self.buttonClearResult.clicked.connect(self.ClearAll)
        self.buttonSaveFigure.clicked.connect(self.SaveFigure)

        # Update Sheet
        self.tableClinicalStatistic.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableClinicalStatistic.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.comboSheet.currentIndexChanged.connect(self.UpdateSheet)
        self.tableClinicalStatistic.itemSelectionChanged.connect(self.ShowOneResult)

        self.comboRefFeature.currentIndexChanged.connect(self.ChangeFeature)
        self.radioSurvivalSplitFeature.clicked.connect(self.UpdateSurvival)
        self.radioSurvivalSplitDataset.clicked.connect(self.UpdateSurvival)

        self.comboSurvivalModel.currentIndexChanged.connect(self.UpdateSurvivalCurve)
        self.checkSurvivalTrain.stateChanged.connect(self.UpdateSurvivalCurve)
        self.checkSurvivalCvVal.stateChanged.connect(self.UpdateSurvivalCurve)
        self.checkSurvivalTest.stateChanged.connect(self.UpdateSurvivalCurve)
        self.checkSurvivalKM.stateChanged.connect(self.UpdateSurvivalCurve)
        self.buttonLoadRefData.clicked.connect(self.LoadRefData)
        self.buttonSplitShow.clicked.connect(self.UpdateSurvivalCurve)

        # Feature related
        self.radioContribution.clicked.connect(self.UpdateFeature)
        self.radioVariance.clicked.connect(self.UpdateFeature)

        self.comboCindexModel.currentIndexChanged.connect(self.UpdateCindexChange)
        self.checkCindexTrain.stateChanged.connect(self.UpdateCindexChange)
        self.checkCindexTest.stateChanged.connect(self.UpdateCindexChange)
        self.checkCindexCV.stateChanged.connect(self.UpdateCindexChange)

        self.comboModelContribution.currentIndexChanged.connect(self.UpdateContribution)
        self.spinCoefficientBias.valueChanged.connect(self.UpdateContribution)

        self.ClearAll()

    def closeEvent(self, event):
        self.close_signal.emit(True)
        event.accept()

    def LoadResult(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]

            if not os.path.exists(self._root_folder):
                return
            try:
                if self.sae.LoadResult(self._root_folder):
                    self.lineEditResultPath.setText(self._root_folder)
                    self.SetResultDescription()
                    self.SetResultTable()
                    self.InitialUi()

                    self.buttonSaveFigure.setEnabled(True)
                    self.UpdateFeature()
                    self.buttonClearResult.setEnabled(True)
                    self.buttonLoadResult.setEnabled(False)
                    # self.buttonGenerateDescription.setEnabled(True)

                    self.radioSurvivalSplitDataset.setEnabled(True)
                    self.radioSurvivalSplitFeature.setEnabled(True)
                    self.radioContribution.setEnabled(True)
                    self.radioVariance.setEnabled(True)
                    self.checkSurvivalKM.setEnabled(True)

                    self.radioSurvivalSplitDataset.setChecked(True)
                    self.radioContribution.setChecked(True)

                    self.UpdateFeature()
                    self.UpdateSurvival()

                    self._is_loading = True

                else:
                    QMessageBox().about(self, "Load Failed",
                                        "The results were built by SAE with the previous version and can not be "
                                        "loaded.")
            except Exception as ex:
                QMessageBox.about(self, "Load Error", ex.__str__())
                mylog.error('Load Error, The reason is ' + traceback.format_exception(ex))
                self.ClearAll()
                raise ex

    def ClearAll(self):
        self.buttonLoadResult.setEnabled(True)

        self.lineEditResultPath.clear()
        self.buttonClearResult.setEnabled(False)
        self.buttonSaveFigure.setEnabled(False)
        self.comboSheet.clear()
        self.textEditDescription.clear()
        self.textEditModelDescription.clear()

        self.tableClinicalStatistic.clear()
        self.tableClinicalStatistic.clear()
        self.tableClinicalStatistic.setRowCount(0)
        self.tableClinicalStatistic.setColumnCount(0)
        self.tableClinicalStatistic.setHorizontalHeaderLabels(list([]))
        self.tableClinicalStatistic.setVerticalHeaderLabels(list([]))

        self.comboSurvivalModel.clear()
        self.radioSurvivalSplitDataset.setEnabled(False)
        self.radioSurvivalSplitFeature.setEnabled(False)
        self.checkSurvivalTrain.setEnabled(False)
        self.checkSurvivalCvVal.setEnabled(False)
        self.checkSurvivalTest.setEnabled(False)
        self.checkSurvivalKM.setEnabled(False)
        self.textEditorRefDescription.clear()
        self.lineEditRefSplit.clear()
        self.tableRefData.clear()
        self.buttonLoadRefData.setEnabled(False)
        self.buttonSplitShow.setEnabled(False)

        self.radioContribution.setEnabled(False)
        self.radioVariance.setEnabled(False)
        self.checkCindexTrain.setEnabled(False)
        self.checkCindexCV.setEnabled(False)
        self.checkCindexTest.setEnabled(False)
        self.comboCindexModel.clear()
        self.comboModelContribution.clear()
        self.spinCoefficientBias.setEnabled(False)

        self.canvasFeature.getFigure().clear()
        self.canvasSurvival.getFigure().clear()

        self._root_folder = ''
        self.sae = PipelineManager()
        self.sheet_dict = dict()
        self.models_name = []
        self._current_fitter = BaseFitter()
        self.ref_df = pd.DataFrame()
        self._is_loading = False

    def InitialUi(self):
        self.models_name = list(self.sheet_dict[TRAIN].index)
        self.comboSurvivalModel.addItems(self.models_name)
        self.comboModelContribution.addItems(self.models_name)

        for one in self.models_name:
            part_one = self._FullName2PartName(one)
            if not CheckTextInCombo(part_one, self.comboCindexModel):
                self.comboCindexModel.addItem(part_one)

    def SaveFigure(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            try:
                self.canvasSurvival.getFigure().savefig(os.path.join(store_folder, 'SurvivalCurve.eps'), dpi=1200)
                self.canvasSurvival.getFigure().savefig(os.path.join(store_folder, 'SurvivalCurve.jpg'), dpi=300)
            except Exception as e:
                mylog.error('Saving Survival Curve {}'.format(e.__str__()))
                QMessageBox.about(self, 'Save Figure Failed', 'There is no SurvivalCurve figure.\n' + e.__str__())

            try:
                self.canvasFeature.getFigure().savefig(os.path.join(store_folder, 'Feature Contribution.eps'), dpi=1200)
                self.canvasFeature.getFigure().savefig(os.path.join(store_folder, 'Feature Contribution.jpg'), dpi=300)
            except Exception as e:
                mylog.error('Saving Feature Contribution {}'.format(e.__str__()))
                QMessageBox.about(self, 'Save Figure Failed',
                                  'There is no Feature Contribution figure.\n' + e.__str__())

    def SetResultDescription(self):
        text = 'Version: ' + self.sae.version
        text += '\n'

        text += "Normalizer:\n"
        for index in self.sae.normalizers:
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Dimension Reduction:\n"
        for index in self.sae.reducers:
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Selector:\n"
        for index in self.sae.feature_selectors:
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Number:\n"
        text += "{:s} - {:s}\n".format(self.sae.feature_numbers[0],
                                       self.sae.feature_numbers[-1])
        text += '\n'

        text += "Fitters:\n"
        for index in self.sae.fitters:
            text += (index.name + '\n')
        text += '\n'

        text += 'Cross Validation: ' + self.sae.cv.k

        self.textEditDescription.setPlainText(text)

    def SetResultTable(self):
        self.sheet_dict[TRAIN] = pd.read_csv(os.path.join(self._root_folder, 'result-{}.csv'.format(TRAIN)),
                                             index_col=0)
        self.comboSheet.addItem(TRAIN)
        self.sheet_dict[CV_VAL] = pd.read_csv(os.path.join(self._root_folder, 'result-{}.csv'.format(CV_VAL)),
                                              index_col=0)
        self.comboSheet.addItem(CV_VAL)
        if os.path.exists(os.path.join(self._root_folder, 'result-{}.csv'.format(TEST))):
            self.sheet_dict[TEST] = pd.read_csv(os.path.join(self._root_folder, 'result-{}.csv'.format(TEST)),
                                                index_col=0)
            self.comboSheet.addItem(TEST)

        self.UpdateSheet()

    def UpdateSheet(self):
        self.tableClinicalStatistic.clear()
        self.tableClinicalStatistic.setSortingEnabled(False)
        if self.comboSheet.currentText() == TRAIN:
            df = self.sheet_dict[TRAIN]
        elif self.comboSheet.currentText() == CV_VAL:
            df = self.sheet_dict[CV_VAL]
        elif self.comboSheet.currentText() == TEST:
            df = self.sheet_dict[TEST]
        else:
            mylog.error('Wrong key in the result table, or may click clear')
            return

        df.sort_index(inplace=True)
        self.tableClinicalStatistic.setRowCount(df.shape[0])
        self.tableClinicalStatistic.setColumnCount(df.shape[1] + 1)

        headerlabels = df.columns.tolist()
        headerlabels.insert(0, 'Models Name')
        self.tableClinicalStatistic.setHorizontalHeaderLabels(headerlabels)

        for row_index in range(df.shape[0]):
            for col_index in range(df.shape[1] + 1):
                if col_index == 0:
                    self.tableClinicalStatistic.setItem(row_index, col_index,
                                                        QTableWidgetItem(df.index[row_index]))
                else:
                    self.tableClinicalStatistic.setItem(row_index, col_index,
                                                        QTableWidgetItem(str(df.iloc[row_index, col_index - 1])))

        self.tableClinicalStatistic.setSortingEnabled(True)

#############################################################
    def LoadRefData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=r'C:\MyCode\FAE\Example',
                                           filter="csv files (*.csv)")

        if file_name:
            self.ref_df = pd.read_csv(file_name, index_col=0)

            self.comboRefFeature.setEnabled(True)
            self.buttonSplitShow.setEnabled(True)
            self.textEditorRefDescription.setEnabled(True)
            self.tableRefData.setEnabled(True)

            self.ShowRefData()
            self.comboRefFeature.addItems(self.ref_df.columns)

    def ShowRefData(self):
        self.tableRefData.setRowCount(self.ref_df.shape[0])
        self.tableRefData.setColumnCount(self.ref_df.shape[1])

        self.tableRefData.setHorizontalHeaderLabels(self.ref_df.columns)
        self.tableRefData.setVerticalHeaderLabels(list(map(str, self.ref_df.index)))

        for row_index in range(self.ref_df.shape[0]):
            for col_index in range(self.ref_df.shape[1]):
                self.tableRefData.setItem(row_index, col_index,
                                          QTableWidgetItem(str(
                                              self.ref_df.iloc[row_index, col_index])))

    def ChangeFeature(self):
        self.lineEditRefSplit.setText('')

    def SplitFeatures(self):
        feature = self.ref_df[self.comboRefFeature.currentText()]
        text = self.lineEditRefSplit.text()
        feature_splits = text.split(',')

        sub_cases, sub_legend = [], []
        for split in feature_splits:
            sub_cases.append(list(map(str, list(self.ref_df.index[feature < float(split)]))))
            sub_legend.append('{} < {}'.format(self.comboRefFeature.currentText(), split))

        sub_cases.append(list(map(str, list(self.ref_df.index[feature >= float(feature_splits[-1])]))))
        sub_legend.append('{} >= {}'.format(self.comboRefFeature.currentText(), feature_splits[-1]))
        return sub_cases, sub_legend

    def __AddSurvivalByDataset(self, store_folder, store_key, surv_list, event_list, duration_list):
        surv, event, duration = self.sae.SurvivalLoad(os.path.join(store_folder, '{}.csv'.format(store_key)))
        surv_list.append(surv)
        event_list.append(event)
        duration_list.append(duration)

    def __AddSurvivalBySubcase(self, store_folder, surv_list, event_list, duration_list, sub_case):
        # Get all surv result
        train_surv_path = os.path.join(store_folder, '{}.csv'.format(TRAIN))
        test_surv_path = os.path.join(store_folder, '{}.csv'.format(TEST))

        assert(os.path.exists(train_surv_path))
        surv, event, duration = self.sae.SurvivalLoad(train_surv_path)

        if os.path.exists(test_surv_path):
            test_surv, test_event, test_duration = self.sae.SurvivalLoad(test_surv_path)
            surv = pd.concat((surv, test_surv), axis=1)
            event = pd.concat([event, test_event])
            duration = pd.concat([duration, test_duration])

        # To find the sub-survivals
        not_exist_case, sub_index = [], []
        for case in sub_case:
            if case not in surv.columns:
                not_exist_case.append(case)
            else:
                sub_index.append(list(surv.columns).index(case))

        if len(not_exist_case) == 0:    # Make sure all cases exist
            sub_surv = surv.iloc[:, sub_index]
            sub_event = [event[index] for index in sub_index]
            sub_duration = [duration[index] for index in sub_index]

            surv_list.append(sub_surv)
            event_list.append(sub_event)
            duration_list.append(sub_duration)
        else:
            QMessageBox.information(self, 'Load Ref Failed.',
                                  'The loaded reference cases are not consistent with the result. \n'
                                  'The following cases are not exists: \n'
                                  '{}'.format(not_exist_case))

    def UpdateSurvivalCurve(self):
        if not self._is_loading:
            return None

        pipeline_name = str(self.comboSurvivalModel.currentText())
        if len(pipeline_name) == 0:
            return None

        normalizer, dr, fs, fn, fitter = pipeline_name.split('_')
        fitter_folder = os.path.join(self._root_folder, normalizer, dr, fs + '_' + fn, fitter)
        assert (os.path.exists(fitter_folder))

        surv, event, duration, name_list, legend_list = [], [], [], [], []
        if self.radioSurvivalSplitDataset.isChecked():
            if self.checkSurvivalTrain.isChecked():
                self.__AddSurvivalByDataset(fitter_folder, TRAIN, surv, event, duration)
                name_list.append(TRAIN)
                legend_list.append('{} C-Index={}'.format(TRAIN,
                                                          self.sae.result[TRAIN].loc[pipeline_name][METRIC_CINDEX]))
            if self.checkSurvivalCvVal.isChecked():
                self.__AddSurvivalByDataset(fitter_folder, CV_VAL, surv, event, duration)
                name_list.append(CV_VAL)
                legend_list.append('{} C-Index={}'.format(CV_VAL,
                                                          self.sae.result[CV_VAL].loc[pipeline_name][METRIC_CINDEX]))
            if self.checkSurvivalTest.isChecked():
                try:
                    self.__AddSurvivalByDataset(fitter_folder, TEST, surv, event, duration)
                    name_list.append(TEST)
                    legend_list.append('{} C-Index={}'.format(TEST,
                                                              self.sae.result[TEST].loc[pipeline_name][METRIC_CINDEX]))
                except FileNotFoundError:
                    QMessageBox.about(self, '', 'No Test Data found.')
                    self.checkSurvivalTest.setChecked(False)
                    return

        elif self.radioSurvivalSplitFeature.isChecked():
            if self.ref_df.size == 0 or self.lineEditRefSplit.text == '':
                return

            sub_cases, sub_legends = self.SplitFeatures()

            for sub_case, sub_legend in zip(sub_cases, sub_legends):
                self.__AddSurvivalBySubcase(fitter_folder, surv, event, duration, sub_case)
                name_list.append(sub_legend)
                legend_list.append(sub_legend)

        if len(surv) > 0:
            text = '{} groups were shown: \n'.format(len(name_list))
            for count, (name, one_event) in enumerate(zip(name_list, event)):
                text += 'Group {}: {}/{} cases with event 1/0 \n'.format(
                    count, int(sum(one_event)), int(len(one_event) - sum(one_event)))
            self.textEditorRefDescription.setText(text)

            SurvivalPlot(surv, event, duration, name_list, legend_list=legend_list,
                         fig=self.canvasSurvival.getFigure(),
                         is_show_KM=self.checkSurvivalKM.isChecked())
            self.canvasSurvival.draw()

    def _ChangeSurvivalState(self, dataset_state: bool):
        self.checkSurvivalTrain.setEnabled(dataset_state)
        self.checkSurvivalCvVal.setEnabled(dataset_state)
        self.checkSurvivalTest.setEnabled(dataset_state)
        self.buttonLoadRefData.setEnabled(not dataset_state)
        self.comboRefFeature.setEnabled(not dataset_state)
        self.tableRefData.setEnabled(not dataset_state)
        self.lineEditRefSplit.setEnabled(not dataset_state)
        self.buttonSplitShow.setEnabled(not dataset_state)

    def UpdateSurvival(self):
        if self.radioSurvivalSplitDataset.isChecked():
            self._ChangeSurvivalState(True)
        elif self.radioSurvivalSplitFeature.isChecked():
            self._ChangeSurvivalState(False)

        self.UpdateSurvivalCurve()

#############################################################
    def _ChangeFeatureCanvasState(self, contribution_state):
        self.checkCindexCV.setEnabled(not contribution_state)
        self.checkCindexTrain.setEnabled(not contribution_state)
        self.checkCindexTest.setEnabled(not contribution_state)
        self.comboCindexModel.setEnabled(not contribution_state)

        self.comboModelContribution.setEnabled(contribution_state)
        self.spinCoefficientBias.setEnabled(contribution_state)

    def UpdateFeature(self):
        if self.radioVariance.isChecked():
            self._ChangeFeatureCanvasState(False)
            self.UpdateCindexChange()
        elif self.radioContribution.isChecked():
            self._ChangeFeatureCanvasState(True)
            self.UpdateContribution()

    def UpdateContribution(self):
        if not self._is_loading:
            return None

        pipeline_name = str(self.comboModelContribution.currentText())
        if len(pipeline_name) == 0:
            return None

        normalizer, dr, fs, fn, fitter = pipeline_name.split('_')
        fitter_folder = os.path.join(self._root_folder, normalizer, dr, fs + '_' + fn, fitter)
        assert (os.path.exists(fitter_folder))

        current_fitter = BaseFitter()
        current_fitter.Load(fitter_folder)

        ModelHazardRatio(current_fitter, self.canvasFeature.getFigure(), self.spinCoefficientBias.value())
        self.canvasFeature.draw()

    def _FullName2PartName(self, pipeline_name):
        normalizer, reducer, selector, number, fitter = pipeline_name.split('_')
        return '_'.join([normalizer, reducer, selector, fitter])

    def _PartName2FullName(self, part_name):
        normalizer, reducer, selector, fitter = part_name.split('_')

        pipelines = []
        for number in self.sae.feature_numbers:
            one_pipeline = '_'.join([normalizer, reducer, selector, number, fitter])
            assert(one_pipeline in self.sheet_dict[TRAIN].index)
            pipelines.append(one_pipeline)
        return pipelines

    def UpdateCindexChange(self):
        if not self._is_loading:
            return None

        part_name = str(self.comboCindexModel.currentText())
        if len(part_name) == 0:
            return None

        full_names = self._PartName2FullName(part_name)

        curve_list, label_name = [], []
        if self.checkCindexCV.isChecked():
            curve_list.append(self.sheet_dict[CV_VAL].loc[full_names][METRIC_CINDEX].values.tolist())
            label_name.append(CV_VAL)
        if self.checkCindexTrain.isChecked():
            curve_list.append(self.sheet_dict[TRAIN].loc[full_names][METRIC_CINDEX].values.tolist())
            label_name.append(TRAIN)
        if self.checkCindexTest.isChecked():
            curve_list.append(self.sheet_dict[TEST].loc[full_names][METRIC_CINDEX].values.tolist())
            label_name.append(TEST)

        DrawIndex([int(one) for one in self.sae.feature_numbers],
                  curve_list, label_name,
                  fig=self.canvasFeature.getFigure())
        self.canvasFeature.draw()

#############################################################
    def ShowOneResult(self):
        if not self.tableClinicalStatistic.selectedIndexes():
            return None
        index = self.tableClinicalStatistic.selectedIndexes()[0]
        row = index.row()
        one_item = self.tableClinicalStatistic.item(row, 0)
        pipeline_name = str(one_item.text())
        normalizer, dr, fs, fn, fitter = pipeline_name.split('_')

        fitter_folder = os.path.join(self._root_folder, normalizer, dr, fs + '_' + fn, fitter)
        self._current_fitter.Load(fitter_folder)
        self.textEditModelDescription.setText(self._current_fitter.Summary())

        try:
            if CheckTextInCombo(pipeline_name, self.comboSurvivalModel):
                self.comboSurvivalModel.setCurrentText(pipeline_name)
                self.UpdateSurvivalCurve()

            part_name = self._FullName2PartName(pipeline_name)
            if CheckTextInCombo(part_name, self.comboCindexModel):
                self.comboCindexModel.setCurrentText(part_name)
                self.UpdateFeature()

            if CheckTextInCombo(pipeline_name, self.comboModelContribution):
                self.comboModelContribution.setCurrentText(pipeline_name)
                self.UpdateFeature()

        except Exception as e:
            mylog.error(e)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_frame = VisualizationForm()
    main_frame.show()
    sys.exit(app.exec_())
