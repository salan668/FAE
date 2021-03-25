import re

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import logging

from BC.GUI.Visualization import Ui_Visualization
from BC.FeatureAnalysis.Classifier import *
from BC.FeatureAnalysis.Pipelines import PipelinesManager
from BC.Description.Description import Description
from BC.Visualization.DrawROCList import DrawROCList
from BC.Visualization.PlotMetricVsFeatureNumber import DrawCurve, DrawBar
from BC.Visualization.FeatureSort import GeneralFeatureSort
from BC.Utility.EcLog import eclog
from BC.Utility.Constants import *


class VisualizationConnection(QWidget, Ui_Visualization):
    close_signal = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        self._root_folder = ''
        self._fae = PipelinesManager()
        self.sheet_dict = dict()
        self._filename = os.path.split(__file__)[-1]
        self.__is_ui_ready = False
        self.__is_clear = False

        super(VisualizationConnection, self).__init__(parent)
        self.setupUi(self)

        self.buttonLoadResult.clicked.connect(self.LoadAll)
        self.buttonClearResult.clicked.connect(self.ClearAll)
        self.buttonSave.clicked.connect(self.Save)
        self.buttonGenerateDescription.clicked.connect(self.GenerateDescription)

        self.__plt_roc = self.canvasROC.getFigure().add_subplot(111)
        self.__plt_plot = self.canvasPlot.getFigure().add_subplot(111)
        self.__contribution = self.canvasFeature.getFigure().add_subplot(111)

        # Update Sheet
        self.tableClinicalStatistic.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableClinicalStatistic.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.comboSheet.currentIndexChanged.connect(self.UpdateSheet)
        self.checkMaxFeatureNumber.stateChanged.connect(self.UpdateSheet)
        self.tableClinicalStatistic.itemSelectionChanged.connect(self.ShowOneResult)
        self.checkMaxFeatureNumber.setEnabled(False)

        # Update ROC canvas
        self.comboNormalizer.currentIndexChanged.connect(self.UpdateROC)
        self.comboDimensionReduction.currentIndexChanged.connect(self.UpdateROC)
        self.comboFeatureSelector.currentIndexChanged.connect(self.UpdateROC)
        self.comboClassifier.currentIndexChanged.connect(self.UpdateROC)
        self.spinBoxFeatureNumber.valueChanged.connect(self.UpdateROC)
        self.checkROCCVTrain.stateChanged.connect(self.UpdateROC)
        self.checkROCCVValidation.stateChanged.connect(self.UpdateROC)
        self.checkROCTrain.stateChanged.connect(self.UpdateROC)
        self.checkROCTest.stateChanged.connect(self.UpdateROC)

        # Update Plot canvas
        self.comboPlotX.currentIndexChanged.connect(self.UpdatePlot)
        self.comboPlotY.currentIndexChanged.connect(self.UpdatePlot)
        self.comboPlotNormalizer.currentIndexChanged.connect(self.UpdatePlot)
        self.comboPlotDimensionReduction.currentIndexChanged.connect(self.UpdatePlot)
        self.comboPlotFeatureSelector.currentIndexChanged.connect(self.UpdatePlot)
        self.comboPlotClassifier.currentIndexChanged.connect(self.UpdatePlot)
        self.spinPlotFeatureNumber.valueChanged.connect(self.UpdatePlot)

        self.checkPlotCVTrain.stateChanged.connect(self.UpdatePlot)
        self.checkPlotCVValidation.stateChanged.connect(self.UpdatePlot)
        self.checkPlotTrain.stateChanged.connect(self.UpdatePlot)
        self.checkPlotOneSE.stateChanged.connect(self.UpdatePlot)
        self.checkPlotTest.stateChanged.connect(self.UpdatePlot)

        # Update Contribution canvas
        self.radioContributionFeatureSelector.toggled.connect(self.UpdateContribution)
        self.radioContributionClassifier.toggled.connect(self.UpdateContribution)
        self.comboContributionNormalizor.currentIndexChanged.connect(self.UpdateContribution)
        self.comboContributionDimension.currentIndexChanged.connect(self.UpdateContribution)
        self.comboContributionFeatureSelector.currentIndexChanged.connect(self.UpdateContribution)
        self.comboContributionClassifier.currentIndexChanged.connect(self.UpdateContribution)
        self.spinContributeFeatureNumber.valueChanged.connect(self.UpdateContribution)

    def closeEvent(self, QCloseEvent):
        self.close_signal.emit(True)
        QCloseEvent.accept()

    def LoadAll(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]

            if not os.path.exists(self._root_folder):
                return
            try:
                if self._fae.LoadAll(self._root_folder):
                    self.lineEditResultPath.setText(self._root_folder)
                    self.SetResultDescription()
                    self.SetResultTable()
                    self.InitialUi()

                    self.buttonClearResult.setEnabled(True)
                    self.buttonSave.setEnabled(True)
                    self.buttonLoadResult.setEnabled(False)
                    self.buttonGenerateDescription.setEnabled(True)

                else:
                    QMessageBox().about(self, "Load Failed",
                                        "The results were built by BC with the previous version and can not be "
                                        "loaded.")
            except Exception as ex:
                QMessageBox.about(self, "Load Error", ex.__str__())
                self.logger.log(logging.ERROR, 'Load Error, The reason is ' + str(ex))
                self.ClearAll()
                raise ex

    def ClearAll(self):
        self.__is_clear = True
        self.buttonLoadResult.setEnabled(True)
        self.buttonSave.setEnabled(False)
        self.buttonGenerateDescription.setEnabled(False)
        self.buttonClearResult.setEnabled(False)
        self.checkMaxFeatureNumber.setEnabled(False)

        self.checkROCCVTrain.setChecked(False)
        self.checkROCCVValidation.setChecked(False)
        self.checkROCTrain.setChecked(False)
        self.checkROCTest.setChecked(False)
        self.checkPlotCVTrain.setChecked(False)
        self.checkPlotCVValidation.setChecked(False)
        self.checkPlotTrain.setChecked(False)
        self.checkPlotOneSE.setChecked(False)
        self.checkPlotTest.setChecked(False)
        self.radioContributionFeatureSelector.setChecked(False)
        self.checkMaxFeatureNumber.setChecked(False)
        self.canvasROC.getFigure().clear()
        self.canvasPlot.getFigure().clear()
        self.canvasFeature.getFigure().clear()
        self.__plt_roc = self.canvasROC.getFigure().add_subplot(111)
        self.__plt_plot = self.canvasPlot.getFigure().add_subplot(111)
        self.__contribution = self.canvasFeature.getFigure().add_subplot(111)
        self.canvasROC.draw()
        self.canvasPlot.draw()
        self.canvasFeature.draw()

        self.textEditDescription.clear()
        self.lineEditResultPath.clear()

        self.comboSheet.clear()
        self.comboClassifier.clear()
        self.comboDimensionReduction.clear()
        self.comboNormalizer.clear()
        self.comboFeatureSelector.clear()

        self.comboPlotClassifier.clear()
        self.comboPlotDimensionReduction.clear()
        self.comboPlotFeatureSelector.clear()
        self.comboPlotNormalizer.clear()
        self.comboPlotX.clear()
        self.comboPlotY.clear()

        self.comboContributionNormalizor.clear()
        self.comboContributionDimension.clear()
        self.comboContributionClassifier.clear()
        self.comboContributionFeatureSelector.clear()

        self.spinBoxFeatureNumber.setValue(0)
        self.spinPlotFeatureNumber.setValue(0)
        self.spinPlotFeatureNumber.setEnabled(False)
        self.spinContributeFeatureNumber.setValue(1)

        self.tableClinicalStatistic.clear()
        self.tableClinicalStatistic.setRowCount(0)
        self.tableClinicalStatistic.setColumnCount(0)
        self.tableClinicalStatistic.setHorizontalHeaderLabels(list([]))
        self.tableClinicalStatistic.setVerticalHeaderLabels(list([]))

        self._fae = PipelinesManager()
        self._root_folder = ''
        self.sheet_dict = dict()
        self.__is_ui_ready = False
        self.__is_clear = False

    def Save(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            try:
                self.canvasROC.getFigure().savefig(os.path.join(store_folder, 'ROC.eps'), dpi=1200)
                self.canvasROC.getFigure().savefig(os.path.join(store_folder, 'ROC.jpg'), dpi=300)
            except Exception as e:
                QMessageBox.about(self, 'Save Figure Failed', 'There is no ROC figure.\n' + e.__str__())

            try:
                self.canvasPlot.getFigure().savefig(os.path.join(store_folder, 'Compare.eps'), dpi=1200)
                self.canvasPlot.getFigure().savefig(os.path.join(store_folder, 'Compare.jpg'), dpi=300)
            except Exception as e:
                QMessageBox.about(self, 'Save Figure Failed', 'There is no AUC comparison figure.\n' + e.__str__())

            try:
                self.canvasFeature.getFigure().savefig(os.path.join(store_folder, 'FeatureWeights.eps'), dpi=1200)
                self.canvasFeature.getFigure().savefig(os.path.join(store_folder, 'FeatureWeights.jpg'), dpi=300)
            except Exception as e:
                QMessageBox.about(self, 'Save Figure Failed',
                                  'There is no Feature Contribution figure.\n' + e.__str__())

    def InitialUi(self):
        # Update ROC canvers
        for normalizer in self._fae.normalizer_list:
            self.comboNormalizer.addItem(normalizer.GetName())
        for dimension_reduction in self._fae.dimension_reduction_list:
            self.comboDimensionReduction.addItem(dimension_reduction.GetName())
        for classifier in self._fae.classifier_list:
            self.comboClassifier.addItem(classifier.GetName())
        for feature_selector in self._fae.feature_selector_list:
            self.comboFeatureSelector.addItem(feature_selector.GetName())
        self.spinBoxFeatureNumber.setMinimum(int(self._fae.feature_selector_num_list[0]))
        self.spinBoxFeatureNumber.setMaximum(int(self._fae.feature_selector_num_list[-1]))

        # Update Plot canvars
        if len(self._fae.normalizer_list) > 1:
            self.comboPlotX.addItem('Normaliaztion')
        if len(self._fae.dimension_reduction_list) > 1:
            self.comboPlotX.addItem('Dimension Reduction')
        if len(self._fae.feature_selector_list) > 1:
            self.comboPlotX.addItem('Feature Selector')
        if len(self._fae.classifier_list) > 1:
            self.comboPlotX.addItem('Classifier')
        if len(self._fae.feature_selector_num_list) > 1:
            self.comboPlotX.addItem('Feature Number')

        self.comboPlotY.addItem('AUC')

        for index in self._fae.normalizer_list:
            self.comboPlotNormalizer.addItem(index.GetName())
        for index in self._fae.dimension_reduction_list:
            self.comboPlotDimensionReduction.addItem(index.GetName())
        for index in self._fae.feature_selector_list:
            self.comboPlotFeatureSelector.addItem(index.GetName())
        for index in self._fae.classifier_list:
            self.comboPlotClassifier.addItem(index.GetName())
        self.spinPlotFeatureNumber.setMinimum(int(self._fae.feature_selector_num_list[0]))
        self.spinPlotFeatureNumber.setMaximum(int(self._fae.feature_selector_num_list[-1]))

        # Update Contribution canvas
        for index in self._fae.normalizer_list:
            self.comboContributionNormalizor.addItem(index.GetName())
        for index in self._fae.dimension_reduction_list:
            self.comboContributionDimension.addItem(index.GetName())
        for selector in self._fae.feature_selector_list:
            self.comboContributionFeatureSelector.addItem(selector.GetName())
        for classifier in self._fae.classifier_list:
            specific_name = classifier.GetName() + '_coef.csv'
            if self._SearchSpecificFile(int(self._fae.feature_selector_num_list[0]), specific_name):
                self.comboContributionClassifier.addItem(classifier.GetName())
        self.spinContributeFeatureNumber.setMinimum(int(self._fae.feature_selector_num_list[0]))
        self.spinContributeFeatureNumber.setMaximum(int(self._fae.feature_selector_num_list[-1]))

        self.__is_ui_ready = True

    def __AddOneCurveInRoc(self, pred_list, label_list, name_list, cls_folder, store_key):
        result_path = os.path.join(cls_folder, '{}_prediction.csv'.format(store_key))
        result = pd.read_csv(result_path, index_col=0)
        pred, label = list(result['Pred']), list(result['Label'])
        pred_list.append(pred)
        label_list.append(label)
        name_list.append(store_key)

    def UpdateROC(self):
        if not self.__is_ui_ready:
            return
        if (self.comboNormalizer.count() == 0) or \
                (self.comboDimensionReduction.count() == 0) or \
                (self.comboFeatureSelector.count() == 0) or \
                (self.comboClassifier.count() == 0) or \
                (self.spinBoxFeatureNumber.value() == 0):
            return

        pipeline_name = self._fae.GetStoreName(self.comboNormalizer.currentText(),
                                               self.comboDimensionReduction.currentText(),
                                               self.comboFeatureSelector.currentText(),
                                               str(self.spinBoxFeatureNumber.value()),
                                               self.comboClassifier.currentText())
        cls_folder = self._fae.SplitFolder(pipeline_name, self._root_folder)[3]

        pred_list, label_list, name_list = [], [], []
        if self.checkROCCVTrain.isChecked():
            self.__AddOneCurveInRoc(pred_list, label_list, name_list, cls_folder, CV_TRAIN)
        if self.checkROCCVValidation.isChecked():
            self.__AddOneCurveInRoc(pred_list, label_list, name_list, cls_folder, CV_VAL)
        if self.checkROCTrain.isChecked():
            self.__AddOneCurveInRoc(pred_list, label_list, name_list, cls_folder, TRAIN)
        if self.checkROCTest.isChecked():
            self.__AddOneCurveInRoc(pred_list, label_list, name_list, cls_folder, TEST)

        if len(pred_list) > 0:
            DrawROCList(pred_list, label_list, name_list=name_list, is_show=False, fig=self.canvasROC.getFigure())

        self.canvasROC.draw()

    def _UpdatePlotButtons(self, selected_index):
        index = [0, 0, 0, 0, 0]

        self.comboPlotNormalizer.setEnabled(True)
        self.comboPlotDimensionReduction.setEnabled(True)
        self.comboPlotFeatureSelector.setEnabled(True)
        self.comboPlotClassifier.setEnabled(True)
        self.spinPlotFeatureNumber.setEnabled(True)
        index[0] = self.comboPlotNormalizer.currentIndex()
        index[1] = self.comboPlotDimensionReduction.currentIndex()
        index[2] = self.comboPlotFeatureSelector.currentIndex()
        index[4] = self.comboPlotClassifier.currentIndex()
        index[3] = self.spinPlotFeatureNumber.value() - int(self._fae.feature_selector_num_list[0])

        if selected_index == 0:
            self.comboPlotNormalizer.setEnabled(False)
            index[0] = [temp for temp in range(len(self._fae.normalizer_list))]
        elif selected_index == 1:
            self.comboPlotDimensionReduction.setEnabled(False)
            index[1] = [temp for temp in range(len(self._fae.dimension_reduction_list))]
        elif selected_index == 2:
            self.comboPlotFeatureSelector.setEnabled(False)
            index[2] = [temp for temp in range(len(self._fae.feature_selector_list))]
        elif selected_index == 4:
            self.comboPlotClassifier.setEnabled(False)
            index[4] = [temp for temp in range(len(self._fae.classifier_list))]
        elif selected_index == 3:
            self.spinPlotFeatureNumber.setEnabled(False)
            index[3] = [temp for temp in range(len(self._fae.feature_selector_num_list))]

        return index

    def UpdatePlot(self):
        if (not self.__is_ui_ready) or self.__is_clear:
            return

        if self.comboPlotX.count() == 0:
            return

        x_ticks = []
        x_label = ''
        selected_index = -1
        if self.comboPlotX.currentText() == 'Normaliaztion':
            selected_index = 0
            x_ticks = [instance.GetName() for instance in self._fae.normalizer_list]
            x_label = 'Normalization Method'
        elif self.comboPlotX.currentText() == 'Dimension Reduction':
            selected_index = 1
            x_ticks = [instance.GetName() for instance in self._fae.dimension_reduction_list]
            x_label = 'Dimension Reduction Method'
        elif self.comboPlotX.currentText() == 'Feature Selector':
            selected_index = 2
            x_ticks = [instance.GetName() for instance in self._fae.feature_selector_list]
            x_label = 'Feature Selecotr Method'
        elif self.comboPlotX.currentText() == 'Classifier':
            selected_index = 4
            x_ticks = [instance.GetName() for instance in self._fae.classifier_list]
            x_label = 'Classifier Method'
        elif self.comboPlotX.currentText() == 'Feature Number':
            selected_index = 3
            x_ticks = list(map(int, self._fae.feature_selector_num_list))
            x_label = 'Feature Number'

        max_axis_list = [0, 1, 2, 3, 4]
        max_axis_list.remove(selected_index)
        max_axis = tuple(max_axis_list)

        index = self._UpdatePlotButtons(selected_index)

        show_data = []
        show_data_std = []
        name_list = []

        if self.comboPlotY.currentText() == 'AUC':
            if self.checkPlotCVTrain.isChecked():
                temp = deepcopy(self._fae.GetAuc()[CV_TRAIN])
                auc_std = deepcopy(self._fae.GetAucStd()[CV_TRAIN])
                show_data.append(temp[tuple(index)].tolist())
                show_data_std.append(auc_std[tuple(index)].tolist())
                name_list.append(CV_TRAIN)
            if self.checkPlotCVValidation.isChecked():
                temp = deepcopy(self._fae.GetAuc()[CV_VAL])
                auc_std = deepcopy(self._fae.GetAucStd()[CV_VAL])
                show_data.append(temp[tuple(index)].tolist())
                show_data_std.append(auc_std[tuple(index)].tolist())
                name_list.append(CV_VAL)
            if self.checkPlotTrain.isChecked():
                temp = deepcopy(self._fae.GetAuc()[TRAIN])
                auc_std = deepcopy(self._fae.GetAucStd()[TRAIN])
                show_data.append(temp[tuple(index)].tolist())
                show_data_std.append(auc_std[tuple(index)].tolist())
                name_list.append(TRAIN)
            if self.checkPlotTest.isChecked():
                temp = deepcopy(self._fae.GetAuc()[TEST])
                auc_std = deepcopy(self._fae.GetAucStd()[TEST])
                if temp.size > 0:
                    show_data.append(temp[tuple(index)].tolist())
                    show_data_std.append(auc_std[tuple(index)].tolist())
                    name_list.append(TEST)

        if len(show_data) > 0:
            if selected_index == 3:
                DrawCurve(x_ticks, show_data, show_data_std, xlabel=x_label, ylabel=self.comboPlotY.currentText(),
                          name_list=name_list, is_show=False, one_se=self.checkPlotOneSE.isChecked(),
                          fig=self.canvasPlot.getFigure())
            else:
                DrawBar(x_ticks, show_data, ylabel=self.comboPlotY.currentText(),
                        name_list=name_list, is_show=False, fig=self.canvasPlot.getFigure())

        self.canvasPlot.draw()

    def UpdateContribution(self):
        if (not self.__is_ui_ready) or self.__is_clear:
            return

        try:
            pipeline_name = self._fae.GetStoreName(self.comboContributionNormalizor.currentText(),
                                                   self.comboContributionDimension.currentText(),
                                                   self.comboContributionFeatureSelector.currentText(),
                                                   str(self.spinContributeFeatureNumber.value()),
                                                   self.comboContributionClassifier.currentText())
            norm_folder, dr_folder, fs_folder, cls_folder = self._fae.SplitFolder(pipeline_name, self._root_folder)

            if self.radioContributionFeatureSelector.isChecked():
                file_name = self.comboContributionFeatureSelector.currentText() + '_sort.csv'
                file_path = os.path.join(fs_folder, file_name)

                if file_path:
                    df = pd.read_csv(file_path, index_col=0)
                    value = list(df.iloc[:, 0])

                    sort_by = df.columns.values[0]
                    if sort_by == 'rank':
                        reverse = False
                    elif sort_by == 'F' or sort_by == 'weight':
                        reverse = True
                    else:
                        reverse = False
                        print('Invalid feature selector sort name.')

                    # add positive and negatiove info for coef
                    processed_feature_name = list(df.index)
                    original_value = list(df.iloc[:, 0])
                    for index in range(len(original_value)):
                        if original_value[index] > 0:
                            processed_feature_name[index] = processed_feature_name[index] + ' P'
                        else:
                            processed_feature_name[index] = processed_feature_name[index] + ' N'

                    GeneralFeatureSort(processed_feature_name, value, max_num=self.spinContributeFeatureNumber.value(),
                                       is_show=False, fig=self.canvasFeature.getFigure(), reverse=reverse)

            elif self.radioContributionClassifier.isChecked():
                specific_name = self.comboContributionClassifier.currentText() + '_coef.csv'
                file_path = os.path.join(cls_folder, specific_name)

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0)
                    value = list(np.abs(df.iloc[:, 0]))

                    # add positive and negatiove info for coef
                    processed_feature_name = list(df.index)
                    original_value = list(df.iloc[:, 0])
                    for index in range(len(original_value)):
                        if original_value[index] > 0:
                            processed_feature_name[index] = processed_feature_name[index] + ' P'
                        else:
                            processed_feature_name[index] = processed_feature_name[index] + ' N'

                    GeneralFeatureSort(processed_feature_name, value,
                                       is_show=False, fig=self.canvasFeature.getFigure())
            self.canvasFeature.draw()
        except Exception as e:
            content = 'In Visualization, UpdateContribution failed'
            self.logger.error('{}{}'.format(content, str(e)))
            QMessageBox.about(self, content, e.__str__())

    def SetResultDescription(self):
        text = 'Version: ' + self._fae.version
        text += '\n'

        text += 'Balance: ' + self._fae.balance.GetName()
        text += '\n'

        text += "Normalizer:\n"
        for index in self._fae.normalizer_list:
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Dimension Reduction:\n"
        for index in self._fae.dimension_reduction_list:
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Selector:\n"
        for index in self._fae.feature_selector_list:
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Number:\n"
        text += "{:s} - {:s}\n".format(self._fae.feature_selector_num_list[0],
                                       self._fae.feature_selector_num_list[-1])
        text += '\n'

        text += "Classifier:\n"
        for index in self._fae.classifier_list:
            text += (index.GetName() + '\n')
        text += '\n'

        text += 'Cross Validation: ' + self._fae.cv.GetName()

        self.textEditDescription.setPlainText(text)

    def UpdateSheet(self):
        if self.__is_clear:
            self.comboSheet.setEnabled(False)
            return None

        if self.checkMaxFeatureNumber.isChecked():
            self.comboSheet.setEnabled(False)
        else:
            self.comboSheet.setEnabled(True)

        self.tableClinicalStatistic.clear()
        self.tableClinicalStatistic.setSortingEnabled(False)
        if self.comboSheet.currentText() == TRAIN:
            df = self.sheet_dict[TRAIN]
        elif self.comboSheet.currentText() == CV_VAL:
            df = self.sheet_dict[CV_VAL]
        elif self.comboSheet.currentText() == TEST:
            df = self.sheet_dict[TEST]
        else:
            return

        if self.checkMaxFeatureNumber.isChecked():
            self.sheet_dict[TEST] = pd.read_csv(os.path.join(self._root_folder, '{}_results.csv'.format(TEST)),
                                                index_col=0)
            data = self._fae.GetAuc()[CV_VAL]
            std_data = self._fae.GetAucStd()[CV_VAL]
            df_val = self.sheet_dict[CV_VAL]
            df_test = self.sheet_dict[TEST]
            name_list = []
            for normalizer_index, normalizer in enumerate(self._fae.normalizer_list):
                for dimension_reducer_index, dimension_reducer in enumerate(self._fae.dimension_reduction_list):
                    for feature_selector_index, feature_selector in enumerate(self._fae.feature_selector_list):
                        for classifier_index, classifier in enumerate(self._fae.classifier_list):
                            sub_auc = data[normalizer_index, dimension_reducer_index, feature_selector_index, :,
                                      classifier_index]
                            sub_auc_std = std_data[normalizer_index, dimension_reducer_index, feature_selector_index, :,
                                          classifier_index]
                            one_se = max(sub_auc) - sub_auc_std[np.argmax(sub_auc)]
                            for feature_number_index in range(len(self._fae.feature_selector_num_list)):
                                if data[normalizer_index, dimension_reducer_index,
                                        feature_selector_index, feature_number_index, classifier_index] >= one_se:
                                    name = normalizer.GetName() + '_' + \
                                           dimension_reducer.GetName() + '_' + \
                                           feature_selector.GetName() + '_' + \
                                           str(self._fae.feature_selector_num_list[feature_number_index]) + '_' + \
                                           classifier.GetName()
                                    name_list.append(name)
                                    break

            # choose the selected models from all test result
            df_val = df_val.loc[name_list]
            max_index = df_val[AUC].idxmax()
            sub_serise = df_val.loc[max_index]
            max_array = sub_serise.values.reshape(1, -1)
            max_auc_df = pd.DataFrame(data=max_array, columns=sub_serise.index.tolist(), index=[max_index])
            max_auc_95ci = max_auc_df.at[max_index, AUC_CI]

            max_auc_95ci = re.findall(r"\d+\.?\d*", max_auc_95ci)
            sub_val_df = df_val[(df_val[AUC] >= float(max_auc_95ci[0])) & (df_val[AUC] <= float(max_auc_95ci[1]))]

            index_by_val = sub_val_df.index.tolist()

            df = df_test.loc[index_by_val]

        df.sort_index(inplace=True)

        self.tableClinicalStatistic.setRowCount(df.shape[0])
        self.tableClinicalStatistic.setColumnCount(df.shape[1] + 1)
        headerlabels = df.columns.tolist()
        headerlabels.insert(0, 'models name')
        self.tableClinicalStatistic.setHorizontalHeaderLabels(headerlabels)
        # self.tableClinicalStatistic.setVerticalHeaderLabels(list(df.index))

        for row_index in range(df.shape[0]):
            for col_index in range(df.shape[1] + 1):
                if col_index == 0:
                    self.tableClinicalStatistic.setItem(row_index, col_index,
                                                        QTableWidgetItem(df.index[row_index]))
                else:
                    self.tableClinicalStatistic.setItem(row_index, col_index,
                                                        QTableWidgetItem(str(df.iloc[row_index, col_index - 1])))

        self.tableClinicalStatistic.setSortingEnabled(True)

    def SetResultTable(self):
        self.sheet_dict[TRAIN] = pd.read_csv(os.path.join(self._root_folder, '{}_results.csv'.format(TRAIN)),
                                             index_col=0)
        self.comboSheet.addItem(TRAIN)
        self.sheet_dict[CV_VAL] = pd.read_csv(os.path.join(self._root_folder, '{}_results.csv'.format(CV_VAL)),
                                              index_col=0)
        self.comboSheet.addItem(CV_VAL)
        if os.path.exists(os.path.join(self._root_folder, '{}_results.csv'.format(TEST))):
            self.sheet_dict[TEST] = pd.read_csv(os.path.join(self._root_folder, '{}_results.csv'.format(TEST)),
                                                index_col=0)
            self.comboSheet.addItem(TEST)
            self.checkMaxFeatureNumber.setEnabled(True)

        self.UpdateSheet()

    def _SearchSpecificFile(self, feature_number, specific_file_name):
        for rt, folder, files in os.walk(self._root_folder):
            if specific_file_name in files:
                return os.path.join(rt, specific_file_name)
        return ''

    def ShowOneResult(self):
        try:
            if not self.tableClinicalStatistic.selectedIndexes():
                return None
            index = self.tableClinicalStatistic.selectedIndexes()[0]
            row = index.row()
            one_item = self.tableClinicalStatistic.item(row, 0)
            text = str(one_item.text())
            current_normalizer, current_dr, current_fs, current_fn, current_cls = text.split('_')

            self.comboNormalizer.setCurrentText(current_normalizer)
            self.comboDimensionReduction.setCurrentText(current_dr)
            self.comboFeatureSelector.setCurrentText(current_fs)
            self.comboClassifier.setCurrentText(current_cls)
            self.spinBoxFeatureNumber.setValue(int(current_fn))
            if not (self.checkROCTrain.isChecked() or self.checkROCCVTrain.isChecked() or
                    self.checkROCCVValidation.isChecked() or self.checkROCTrain.isChecked()):
                self.checkROCCVTrain.setCheckState(True)
                self.checkROCCVValidation.setCheckState(True)
            self.UpdateROC()

            # Update the AUC versus feature number
            self.comboPlotNormalizer.setCurrentText(current_normalizer)
            self.comboPlotDimensionReduction.setCurrentText(current_dr)
            self.comboPlotFeatureSelector.setCurrentText(current_fs)
            self.comboPlotClassifier.setCurrentText(current_cls)
            self.comboPlotX.setCurrentText('Feature Number')
            if not (self.checkPlotTrain.isChecked() or
                    self.checkPlotCVTrain.isChecked() or
                    self.checkPlotCVValidation.isChecked()):
                self.checkPlotCVValidation.setCheckState(True)
            self.UpdatePlot()

            # Update the Contribution
            self.comboContributionNormalizor.setCurrentText(current_normalizer)
            self.comboContributionDimension.setCurrentText(current_dr)
            self.comboContributionFeatureSelector.setCurrentText(current_fs)
            self.comboContributionClassifier.setCurrentText(current_cls)
            self.spinContributeFeatureNumber.setValue(int(current_fn))
            self.UpdateContribution()

        except Exception as e:
            content = 'Visualization, ShowOneResult failed: '
            self.logger.error('{}{}'.format(content, str(e)))
            QMessageBox.about(self, content, e.__str__())

    def GenerateDescription(self):
        if (self.comboNormalizer.count() == 0) or \
                (self.comboDimensionReduction.count() == 0) or \
                (self.comboFeatureSelector.count() == 0) or \
                (self.comboClassifier.count() == 0) or \
                (self.spinBoxFeatureNumber.value() == 0):
            return

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            roc_path = os.path.join(store_folder, 'ROC.jpg')
            self.canvasROC.getFigure().savefig(roc_path, dpi=300)

            report = Description()

            try:
                pipeline_name = self._fae.GetStoreName(self.comboNormalizer.currentText(),
                                                       self.comboDimensionReduction.currentText(),
                                                       self.comboFeatureSelector.currentText(),
                                                       str(self.spinBoxFeatureNumber.value()),
                                                       self.comboClassifier.currentText())
                report.Run(pipeline_name, self._root_folder, store_folder)
                os.system("explorer.exe {:s}".format(os.path.normpath(store_folder)))
            except Exception as ex:
                QMessageBox.about(self, 'Description Generate Error: ', ex.__str__())
                print(eclog(self._filename))
                eclog(self._filename).GetLogger().error('Description Generate Error:  ' + ex.__str__())
