from copy import deepcopy

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from GUI.Visualization import Ui_Visualization

from FAE.FeatureAnalysis.Classifier import *
from FAE.FeatureAnalysis.FeaturePipeline import FeatureAnalysisPipelines
#from FAE.Description.Description import Description

from FAE.Visualization.DrawROCList import DrawROCList
from FAE.Visualization.PlotMetricVsFeatureNumber import DrawCurve, DrawBar
from FAE.Visualization.FeatureSort import GeneralFeatureSort, SortRadiomicsFeature

import os

class VisualizationConnection(QWidget, Ui_Visualization):
    def __init__(self, parent=None):
        self._root_folder = ''
        self._fae = FeatureAnalysisPipelines()
        self.sheet_dict = dict()

        super(VisualizationConnection, self).__init__(parent)
        self.setupUi(self)

        self.buttonLoadResult.clicked.connect(self.LoadAll)
        self.buttonClearResult.clicked.connect(self.ClearAll)
        self.buttonSave.clicked.connect(self.Save)

        self.__plt_roc = self.canvasROC.getFigure().add_subplot(111)
        self.__plt_plot = self.canvasPlot.getFigure().add_subplot(111)
        self.__contribution = self.canvasFeature.getFigure().add_subplot(111)

        # Update Sheet
        self.tableClinicalStatistic.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableClinicalStatistic.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.comboSheet.currentIndexChanged.connect(self.UpdateSheet)
        self.checkMaxFeatureNumber.stateChanged.connect(self.UpdateSheet)
        # self.tableClinicalStatistic.doubleClicked.connect(self.ShowOneResult)
        self.tableClinicalStatistic.itemSelectionChanged.connect(self.ShowOneResult)

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
        self.checkPlotTest.stateChanged.connect(self.UpdatePlot)

        # Update Contribution canvas
        self.checkContributionShow.stateChanged.connect(self.UpdateContribution)
        self.radioContributionFeatureSelector.toggled.connect(self.UpdateContribution)
        self.radioContributionClassifier.toggled.connect(self.UpdateContribution)
        self.comboContributionNormalization.currentIndexChanged.connect(self.UpdateContribution)
        self.comboContributionDimension.currentIndexChanged.connect(self.UpdateContribution)
        self.comboContributionFeatureSelector.currentIndexChanged.connect(self.UpdateContribution)
        self.comboContributionClassifier.currentIndexChanged.connect(self.UpdateContribution)
        self.spinContributeFeatureNumber.valueChanged.connect(self.UpdateContribution)

    def LoadAll(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            self._root_folder = dlg.selectedFiles()[0]

            if not os.path.exists(self._root_folder):
                return
            if not r'.FAEresult4129074093819729087' in os.listdir(self._root_folder):
                QMessageBox.about(self, 'Load Error', 'This folder is not supported for import')
                return

            try:
                self.lineEditResultPath.setText(self._root_folder)
                self._fae.LoadAll(self._root_folder)
                self.SetResultDescription()
                self.SetResultTable()
                self.InitialUi()
            except Exception as ex:
                QMessageBox.about(self, "Load Error", ex.__str__())
                self.logger.log('Load Error, The reason is ' + str(ex))
                self.ClearAll()
                return

            self.buttonClearResult.setEnabled(True)
            self.buttonSave.setEnabled(True)
            self.buttonLoadResult.setEnabled(False)

    def ClearAll(self):

        self.buttonLoadResult.setEnabled(True)
        self.buttonSave.setEnabled(False)
        self.buttonClearResult.setEnabled(False)

        self.checkROCCVTrain.setChecked(False)
        self.checkROCCVValidation.setChecked(False)
        self.checkROCTrain.setChecked(False)
        self.checkROCTest.setChecked(False)
        self.checkPlotCVTrain.setChecked(False)
        self.checkPlotCVValidation.setChecked(False)
        self.checkPlotTrain.setChecked(False)
        self.checkPlotTest.setChecked(False)
        self.checkContributionShow.setChecked(False)
        self.radioContributionFeatureSelector.setChecked(True)
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

        self.comboContributionNormalization.clear()
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

        self._fae = FeatureAnalysisPipelines()
        self._root_folder = ''
        self.sheet_dict = dict()

    def Save(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setOption(QFileDialog.ShowDirsOnly)

        if dlg.exec_():
            store_folder = dlg.selectedFiles()[0]
            self.canvasROC.getFigure().savefig(os.path.join(store_folder, 'ROC.eps'), dpi=1200)
            self.canvasROC.getFigure().savefig(os.path.join(store_folder, 'ROC.jpg'), dpi=300)
            self.canvasPlot.getFigure().savefig(os.path.join(store_folder, 'Compare.eps'), dpi=1200)
            self.canvasPlot.getFigure().savefig(os.path.join(store_folder, 'Compare.jpg'), dpi=300)
            self.canvasFeature.getFigure().savefig(os.path.join(store_folder, 'FeatureWeights.eps'), dpi=1200)
            self.canvasFeature.getFigure().savefig(os.path.join(store_folder, 'FeatureWeights.jpg'), dpi=300)

    def InitialUi(self):
        # Update ROC canvers
        for normalizer in self._fae.GetNormalizerList():
            self.comboNormalizer.addItem(normalizer.GetName())
        for dimension_reduction in self._fae.GetDimensionReductionList():
            self.comboDimensionReduction.addItem(dimension_reduction.GetName())
        for classifier in self._fae.GetClassifierList():
            self.comboClassifier.addItem(classifier.GetName())
        for feature_selector in self._fae.GetFeatureSelectorList():
            self.comboFeatureSelector.addItem(feature_selector.GetName())
        self.spinBoxFeatureNumber.setMinimum(int(self._fae.GetFeatureNumberList()[0]))
        self.spinBoxFeatureNumber.setMaximum(int(self._fae.GetFeatureNumberList()[-1]))

        # Update Plot canvars
        if len(self._fae.GetNormalizerList()) > 1:
            self.comboPlotX.addItem('Normaliaztion')
        if len(self._fae.GetDimensionReductionList()) > 1:
            self.comboPlotX.addItem('Dimension Reduction')
        if len(self._fae.GetFeatureSelectorList()) > 1:
            self.comboPlotX.addItem('Feature Selector')
        if len(self._fae.GetClassifierList()) > 1:
            self.comboPlotX.addItem('Classifier')
        if len(self._fae.GetFeatureNumberList()) > 1:
            self.comboPlotX.addItem('Feature Number')

        self.comboPlotY.addItem('AUC')

        for index in self._fae.GetNormalizerList():
            self.comboPlotNormalizer.addItem(index.GetName())
        for index in self._fae.GetDimensionReductionList():
            self.comboPlotDimensionReduction.addItem(index.GetName())
        for index in self._fae.GetFeatureSelectorList():
            self.comboPlotFeatureSelector.addItem(index.GetName())
        for index in self._fae.GetClassifierList():
            self.comboPlotClassifier.addItem(index.GetName())
        self.spinPlotFeatureNumber.setMinimum(int(self._fae.GetFeatureNumberList()[0]))
        self.spinPlotFeatureNumber.setMaximum(int(self._fae.GetFeatureNumberList()[-1]))

        # Update Contribution canvas
        for index in self._fae.GetNormalizerList():
            self.comboContributionNormalization.addItem(index.GetName())
        for index in self._fae.GetDimensionReductionList():
            self.comboContributionDimension.addItem(index.GetName())
        for selector in self._fae.GetFeatureSelectorList():
            self.comboContributionFeatureSelector.addItem(selector.GetName())
        for classifier in self._fae.GetClassifierList():
            specific_name = classifier.GetName() + '_coef.csv'
            if self._SearchSpecificFile(int(self._fae.GetFeatureNumberList()[0]), specific_name):
                self.comboContributionClassifier.addItem(classifier.GetName())
        self.spinContributeFeatureNumber.setMinimum(int(self._fae.GetFeatureNumberList()[0]))
        self.spinContributeFeatureNumber.setMaximum(int(self._fae.GetFeatureNumberList()[-1]))

    def UpdateROC(self):
        if (self.comboNormalizer.count() == 0) or \
                (self.comboDimensionReduction.count() == 0) or \
                (self.comboFeatureSelector.count() == 0) or \
                (self.comboClassifier.count() == 0) or \
                (self.spinBoxFeatureNumber.value() == 0):
            return

        case_name = self.comboNormalizer.currentText() + '_' + \
                    self.comboDimensionReduction.currentText() + '_' + \
                    self.comboFeatureSelector.currentText() + '_' + \
                    str(self.spinBoxFeatureNumber.value()) + '_' + \
                    self.comboClassifier.currentText()

        case_folder = os.path.join(self._root_folder, case_name)

        pred_list, label_list, name_list = [], [], []
        if self.checkROCCVTrain.isChecked():
            train_pred = np.load(os.path.join(case_folder, 'train_predict.npy'))
            train_label = np.load(os.path.join(case_folder, 'train_label.npy'))
            pred_list.append(train_pred)
            label_list.append(train_label)
            name_list.append('CV Train')
        if self.checkROCCVValidation.isChecked():
            val_pred = np.load(os.path.join(case_folder, 'val_predict.npy'))
            val_label = np.load(os.path.join(case_folder, 'val_label.npy'))
            pred_list.append(val_pred)
            label_list.append(val_label)
            name_list.append('CV Validation')
        if self.checkROCTrain.isChecked():
            all_train_pred = np.load(os.path.join(case_folder, 'all_train_predict.npy'))
            all_train_label = np.load(os.path.join(case_folder, 'all_train_label.npy'))
            pred_list.append(all_train_pred)
            label_list.append(all_train_label)
            name_list.append('Train')
        if self.checkROCTest.isChecked():
            if os.path.exists(os.path.join(case_folder, 'test_label.npy')):
                test_pred = np.load(os.path.join(case_folder, 'test_predict.npy'))
                test_label = np.load(os.path.join(case_folder, 'test_label.npy'))
                pred_list.append(test_pred)
                label_list.append(test_label)
                name_list.append('Test')

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
        index[3] = self.spinPlotFeatureNumber.value() - int(self._fae.GetFeatureNumberList()[0])

        if selected_index == 0:
            self.comboPlotNormalizer.setEnabled(False)
            index[0] = [temp for temp in range(len(self._fae.GetNormalizerList()))]
        elif selected_index == 1:
            self.comboPlotDimensionReduction.setEnabled(False)
            index[1] = [temp for temp in range(len(self._fae.GetDimensionReductionList()))]
        elif selected_index == 2:
            self.comboPlotFeatureSelector.setEnabled(False)
            index[2] = [temp for temp in range(len(self._fae.GetFeatureSelectorList()))]
        elif selected_index == 4:
            self.comboPlotClassifier.setEnabled(False)
            index[4] = [temp for temp in range(len(self._fae.GetClassifierList()))]
        elif selected_index == 3:
            self.spinPlotFeatureNumber.setEnabled(False)
            index[3] = [temp for temp in range(len(self._fae.GetFeatureNumberList()))]

        return index

    def UpdatePlot(self):
        if self.comboPlotX.count() == 0:
            return

        x_ticks = []
        x_label = ''
        selected_index = -1
        if self.comboPlotX.currentText() == 'Normaliaztion':
            selected_index = 0
            x_ticks = [instance.GetName() for instance in self._fae.GetNormalizerList()]
            x_label = 'Normalization Method'
        elif self.comboPlotX.currentText() == 'Dimension Reduction':
            selected_index = 1
            x_ticks = [instance.GetName() for instance in self._fae.GetDimensionReductionList()]
            x_label = 'Dimension Reduction Method'
        elif self.comboPlotX.currentText() == 'Feature Selector':
            selected_index = 2
            x_ticks = [instance.GetName() for instance in self._fae.GetFeatureSelectorList()]
            x_label = 'Feature Selecotr Method'
        elif self.comboPlotX.currentText() == 'Classifier':
            selected_index = 4
            x_ticks = [instance.GetName() for instance in self._fae.GetClassifierList()]
            x_label = 'Classifier Method'
        elif self.comboPlotX.currentText() == 'Feature Number':
            selected_index = 3
            x_ticks = list(map(int, self._fae.GetFeatureNumberList()))
            x_label = 'Feature Number'

        max_axis_list = [0, 1, 2, 3, 4]
        max_axis_list.remove(selected_index)
        max_axis = tuple(max_axis_list)

        index = self._UpdatePlotButtons(selected_index)

        show_data = []
        show_data_std =[]
        name_list = []

        if self.comboPlotY.currentText() == 'AUC':
            if self.checkPlotCVTrain.isChecked():
                temp = deepcopy(self._fae.GetAUCMetric()['train'])
                auc_std = deepcopy(self._fae.GetAUCstdMetric()['train'])
                show_data.append(temp[tuple(index)].tolist())
                show_data_std.append(auc_std[tuple(index)].tolist())
                name_list.append('CV Train')
            if self.checkPlotCVValidation.isChecked():
                temp = deepcopy(self._fae.GetAUCMetric()['val'])
                auc_std = deepcopy(self._fae.GetAUCstdMetric()['val'])
                show_data.append(temp[tuple(index)].tolist())
                show_data_std.append(auc_std[tuple(index)].tolist())
                name_list.append('CV Validation')
            if self.checkPlotTrain.isChecked():
                temp = deepcopy(self._fae.GetAUCMetric()['all_train'])
                auc_std = deepcopy(self._fae.GetAUCstdMetric()['all_train'])
                show_data.append(temp[tuple(index)].tolist())
                show_data_std.append(auc_std[tuple(index)].tolist())
                name_list.append('Train')
            if self.checkPlotTest.isChecked():
                temp = deepcopy(self._fae.GetAUCMetric()['test'])
                auc_std = deepcopy(self._fae.GetAUCstdMetric()['test'])
                if temp.size > 0:
                    show_data.append(temp[tuple(index)].tolist())
                    show_data_std.append(auc_std[tuple(index)].tolist())
                    name_list.append('Test')

        if len(show_data) > 0:
            if selected_index == 3:
                DrawCurve(x_ticks, show_data, show_data_std, xlabel=x_label, ylabel=self.comboPlotY.currentText(),
                          name_list=name_list, is_show=False, fig=self.canvasPlot.getFigure())
            else:
                DrawBar(x_ticks, show_data, ylabel=self.comboPlotY.currentText(),
                          name_list=name_list, is_show=False, fig=self.canvasPlot.getFigure())

        self.canvasPlot.draw()

    def UpdateContribution(self):
        if not self.checkContributionShow.isChecked():
            return

        one_result_folder_name = self.comboContributionNormalization.currentText() + '_' + \
                            self.comboContributionDimension.currentText() + '_' + \
                            self.comboContributionFeatureSelector.currentText() + '_' + \
                            str(self.spinContributeFeatureNumber.value()) + '_' + \
                            self.comboContributionClassifier.currentText()
        one_result_folder = os.path.join(self._root_folder, one_result_folder_name)
        # This is compatible witht the previous version
        if not os.path.exists(one_result_folder):
            one_result_folder_name = self.comboContributionNormalization.currentText() + '_Cos_' + \
                                     self.comboContributionFeatureSelector.currentText() + '_' + \
                                     str(self.spinContributeFeatureNumber.value()) + '_' + \
                                     self.comboContributionClassifier.currentText()
            one_result_folder = os.path.join(self._root_folder, one_result_folder_name)

        if self.radioContributionFeatureSelector.isChecked():
            file_name = self.comboContributionFeatureSelector.currentText() + '_sort.csv'
            file_path = os.path.join(one_result_folder, file_name)

            if not os.path.exists(file_path):
                file_name = self.comboContributionFeatureSelector.currentText().lower() + '_sort.csv'
                file_path = os.path.join(one_result_folder, file_name)



            if file_path:

                df = pd.read_csv(file_path, index_col=0)
                value = list(np.abs(df.iloc[:, 0]))

                #add positive and negatiove info for coef
                processed_feature_name = list(df.index)
                original_value = list(df.iloc[:, 0])
                for index in range(len(original_value)):
                    if original_value[index] > 0:
                        processed_feature_name[index] = processed_feature_name[index] + ' P'
                    else:
                        processed_feature_name[index] = processed_feature_name[index] + ' N'

                GeneralFeatureSort(processed_feature_name, value, max_num=self.spinContributeFeatureNumber.value(),
                                   is_show=False, fig=self.canvasFeature.getFigure())

        elif self.radioContributionClassifier.isChecked():
            specific_name = self.comboContributionClassifier.currentText() + '_coef.csv'
            file_path = os.path.join(one_result_folder, specific_name)

            if not os.path.exists(file_path):
                specific_name = self.comboContributionClassifier.currentText().lower() + '_coef.csv'
                file_path = os.path.join(one_result_folder, specific_name)

            if file_path:
                df = pd.read_csv(file_path, index_col=0)
                feature_name = list(df.index)
                value = list(np.abs(df.iloc[:, 0]))

                #add positive and negatiove info for coef
                processed_feature_name = list(df.index)
                original_value = list(df.iloc[:, 0])
                for index in range(len(original_value)):
                    if original_value[index] > 0:
                        processed_feature_name[index] = processed_feature_name[index] + ' P'
                    else:
                        processed_feature_name[index] = processed_feature_name[index] + ' N'

                try:
                    SortRadiomicsFeature(processed_feature_name, value, is_show=False, fig=self.canvasFeature.getFigure())
                except:
                    GeneralFeatureSort(processed_feature_name, value,
                                       is_show=False, fig=self.canvasFeature.getFigure())


        self.canvasFeature.draw()

    def SetResultDescription(self):
        text = "Normalizer:\n"
        for index in self._fae.GetNormalizerList():
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Dimension Reduction:\n"
        for index in self._fae.GetDimensionReductionList():
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Selector:\n"
        for index in self._fae.GetFeatureSelectorList():
            text += (index.GetName() + '\n')
        text += '\n'

        text += "Feature Number:\n"
        text += "{:s} - {:s}\n".format(self._fae.GetFeatureNumberList()[0], self._fae.GetFeatureNumberList()[-1])
        text += '\n'

        text += "Classifier:\n"
        for index in self._fae.GetClassifierList():
            text += (index.GetName() + '\n')
        text += '\n'

        text += 'Cross Validation: ' + self._fae.GetCrossValidation().GetName()

        self.textEditDescription.setPlainText(text)


    def UpdateSheet(self):
        self.tableClinicalStatistic.clear()
        self.tableClinicalStatistic.setSortingEnabled(False)
        if self.comboSheet.currentText() == 'Train':
            data = self._fae.GetAUCMetric()['train']
            std_data = self._fae.GetAUCstdMetric()['train']
            df = self.sheet_dict['train']
        elif self.comboSheet.currentText() == 'Validation':
            data = self._fae.GetAUCMetric()['val']
            std_data = self._fae.GetAUCstdMetric()['val']
            df = self.sheet_dict['val']
        elif self.comboSheet.currentText() == 'Test':
            # Sort according to the AUC of validation data set
            data = self._fae.GetAUCMetric()['val']
            std_data = self._fae.GetAUCstdMetric()['val']
            df = self.sheet_dict['test']
        else:
            return

        if self.checkMaxFeatureNumber.isChecked():
            name_list = []
            for normalizer, normalizer_index in zip(self._fae.GetNormalizerList(), range(len(self._fae.GetNormalizerList()))):
                for dimension_reducer, dimension_reducer_index in zip(self._fae.GetDimensionReductionList(),
                                                                      range(len(self._fae.GetDimensionReductionList()))):
                    for feature_selector, feature_selector_index in zip(self._fae.GetFeatureSelectorList(),
                                                                        range(len(self._fae.GetFeatureSelectorList()))):
                        for classifier, classifier_index in zip(self._fae.GetClassifierList(),
                                                                range(len(self._fae.GetClassifierList()))):
                            sub_auc = data[normalizer_index, dimension_reducer_index, feature_selector_index, :,
                                      classifier_index]
                            sub_auc_std = std_data[normalizer_index, dimension_reducer_index, feature_selector_index, :,
                                      classifier_index]
                            one_se = max(sub_auc)-sub_auc_std[np.argmax(sub_auc)]
                            for feature_number_index in range(len(self._fae.GetFeatureNumberList())):
                                if data[normalizer_index, dimension_reducer_index,
                                        feature_selector_index, feature_number_index,classifier_index] >= one_se:
                                    name = normalizer.GetName() + '_' + dimension_reducer.GetName() + '_' + \
                                    feature_selector.GetName() + '_' + str(self._fae.GetFeatureNumberList()[feature_number_index]) + '_' + \
                                    classifier.GetName()
                                    name_list.append(name)
                                    break
            df = df.loc[name_list]
        df.sort_index(inplace=True)

        self.tableClinicalStatistic.setRowCount(df.shape[0])
        self.tableClinicalStatistic.setColumnCount(df.shape[1]+1)
        headerlabels = df.columns.tolist()
        headerlabels.insert(0, 'models name')
        self.tableClinicalStatistic.setHorizontalHeaderLabels(headerlabels)
        # self.tableClinicalStatistic.setVerticalHeaderLabels(list(df.index))

        for row_index in range(df.shape[0]):
            for col_index in range(df.shape[1]+1):
                if col_index == 0:
                    self.tableClinicalStatistic.setItem(row_index, col_index,
                                                        QTableWidgetItem(df.index[row_index]))
                else:
                    self.tableClinicalStatistic.setItem(row_index, col_index,
                                                        QTableWidgetItem(str(df.iloc[row_index, col_index-1])))

        self.tableClinicalStatistic.setSortingEnabled(True)

    def SetResultTable(self):
        self.sheet_dict['train'] = pd.read_csv(os.path.join(self._root_folder, 'train_result.csv'), index_col=0)
        self.comboSheet.addItem('Train')
        self.sheet_dict['val'] = pd.read_csv(os.path.join(self._root_folder, 'val_result.csv'), index_col=0)
        self.comboSheet.addItem('Validation')
        if os.path.exists(os.path.join(self._root_folder, 'test_result.csv')):
            self.sheet_dict['test'] = pd.read_csv(os.path.join(self._root_folder, 'test_result.csv'), index_col=0)
            self.comboSheet.addItem('Test')

        self.UpdateSheet()

    def _SearchSpecificFile(self, feature_number, specific_file_name, specific_file_name2=''):
        for rt, folder, files in os.walk(self._root_folder):
            for file_name in files:
                # print(file_name)
                if specific_file_name2:
                    if (file_name.lower() == specific_file_name.lower()) and \
                            ('_{:d}_'.format(feature_number) in rt) and \
                            (specific_file_name2 in rt):
                        return os.path.join(rt, file_name)
                else:
                    if (file_name.lower() == specific_file_name.lower()) and ('_{:d}_'.format(feature_number) in rt):
                        return os.path.join(rt, file_name)

        return ''

    def ShowOneResult(self):
        try:
            for index in self.tableClinicalStatistic.selectedIndexes():
                row = index.row()
                one_item = self.tableClinicalStatistic.item(row, 0)
                text = str(one_item.text())
                current_normalizer, current_dimension_reducer, current_feature_selector, current_feature_number, current_classifier = \
                    text.split('_')

            self.comboNormalizer.setCurrentText(current_normalizer)
            self.comboDimensionReduction.setCurrentText(current_dimension_reducer)
            self.comboFeatureSelector.setCurrentText(current_feature_selector)
            self.comboClassifier.setCurrentText(current_classifier)
            self.spinBoxFeatureNumber.setValue(int(current_feature_number))

            if not (self.checkROCTrain.isChecked() or self.checkROCCVTrain.isChecked() or
                    self.checkROCCVValidation.isChecked() or self.checkROCTrain.isChecked()):
                self.checkROCCVTrain.setCheckState(True)

            self.UpdateROC()

        except Exception as e:
            print(e)
            return

        #
        # for item in self.tableClinicalStatistic.selectedItems():
        #     text = str(item.text())
        #     try:
        #         current_normalizer, current_dimension_reducer, current_feature_selector, current_feature_number, current_classifier = \
        #         text.split('_')
        #     except:
        #         return None
        #     break



