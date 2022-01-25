"""
All rights reserved. 
Author: Yang SONG (songyangmri@gmail.com)
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5 import QtCore

from Feature.GUI.FeatureMerge import Ui_FeatureMerge
from Utility.EcLog import eclog


class FeatureMergeForm(QWidget):
    close_signal = QtCore.pyqtSignal(bool)

    def __init__(self, eclog=eclog):
        super().__init__()
        self.ui = Ui_FeatureMerge()
        self.ui.setupUi(self)

        self.feature1 = pd.DataFrame()
        self.feature2 = pd.DataFrame()
        self.eclog = eclog

        self.ui.buttonLoadFeatureMatrix1.clicked.connect(self.LoadFeature1)
        self.ui.buttonLoadFeatureMatrix2.clicked.connect(self.LoadFeature2)
        self.ui.radioMergeTypeCases.clicked.connect(self.UpdateText)
        self.ui.radioMergeTypeFeatures.clicked.connect(self.UpdateText)

        self.ui.buttonMerge.clicked.connect(self.Merge)

    def closeEvent(self, event):
        self.close_signal.emit(True)
        event.accept()

    def _UpdateTable(self, table_view, feature: pd.DataFrame()):
        table_view.setRowCount(feature.shape[0])
        header_name = deepcopy(list(feature.columns))

        min_col = np.min([len(header_name), 100])
        if min_col == 100:
            header_name = header_name[:100]
            header_name[-1] = '...'

        table_view.setColumnCount(min_col)
        table_view.setHorizontalHeaderLabels(header_name)
        table_view.setVerticalHeaderLabels(list(map(str, feature.index)))

        for row_index in range(feature.shape[0]):
            for col_index in range(min_col):
                if col_index < 99:
                    table_view.setItem(row_index, col_index, QTableWidgetItem(str(
                                                  feature.iloc[row_index, col_index])))
                else:
                    table_view.setItem(row_index, col_index, QTableWidgetItem('...'))

    def _LoadButton(self, line_edit, table_view, feature_num):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', filter="csv files (*.csv)")
        if file_name:
            try:
                feature = pd.read_csv(file_name, index_col=0)
                line_edit.setText(file_name)
                self._UpdateTable(table_view, feature)

                if feature_num == 1:
                    self.feature1 = feature
                elif feature_num == 2:
                    self.feature2 = feature
                else:
                    self.eclog.eclogger.error("number feature number is not 1 or 2. It's {}".format(feature_num))
                    raise ValueError
                
            except Exception as e:
                self.eclog.eclogger.error(e)

    def UpdateText(self):
        text = ''
        if self.feature1.size == 0 and self.feature2.size == 0:
            text += 'DESCRIPTION: Make sure the case ID is in the first columne.'
        else:
            if self.feature1.size > 0:
                text += 'The Feature 1 has {} rows and {} columns. \n'.format(self.feature1.shape[0], self.feature1.shape[1])

            if self.feature2.size > 0:
                text += 'The Feature 2 has {} rows and {} columns. \n'.format(self.feature2.shape[0], self.feature2.shape[1])

            if self.ui.radioMergeTypeFeatures.isChecked():
                if self.feature1.shape[0] == self.feature2.shape[0]:
                    text += 'The shape of merge feature is {} x {}'.format(self.feature1.shape[0],
                                                                           self.feature1.shape[1] + self.feature2.shape[1])
                else:
                    text += 'Can not merge features since different rows.'

            elif self.ui.radioMergeTypeCases.isChecked():
                if self.feature1.shape[1] == self.feature2.shape[1]:
                    text += 'The shape of merge feature is {} x {}'.format(self.feature1.shape[0] + self.feature2.shape[0],
                                                                           self.feature1.shape[1])
                else:
                    text += 'Can not merge features since different columns.'

        self.ui.textEditDescription.setText(text)

    def LoadFeature1(self):
        self._LoadButton(self.ui.lineEditFeature1Path, self.ui.tableFeature1, 1)
        self.UpdateText()

    def LoadFeature2(self):
        self._LoadButton(self.ui.lineEditFeature2Path, self.ui.tableFeature2, 2)
        self.UpdateText()

    def _AddPreName(self, pre_name: str, feature_df: pd.DataFrame):
        rename_dict = {key: '{}_{}'.format(pre_name, key) for key in feature_df.columns}
        return feature_df.rename(columns=rename_dict, inplace=False)

    def MergeFeatures(self):
        if self.feature1.shape[0] != self.feature2.shape[0]:
            QMessageBox().about(self, '', 'Feature 1 and Feature 2 have different rows')
            return None
        
        for one_case in self.feature1.index:
            if one_case not in self.feature2.index:
                QMessageBox().about(self, '', '{} does not in Feature 2 index.'.format(one_case))
                return None
        for one_case in self.feature2.index:
            if one_case not in self.feature1.index:
                QMessageBox().about(self, '', '{} does not in Feature 1 index.'.format(one_case))
                return None

        pre_name1 = self.ui.lineEditPreName1.text()
        if pre_name1:
            rename_feature1 = self._AddPreName(pre_name1, self.feature1)
        else:
            rename_feature1 = self.feature1

        pre_name2 = self.ui.lineEditPreName2.text()
        if pre_name2:
            rename_feature2 = self._AddPreName(pre_name2, self.feature2)
        else:
            rename_feature2 = self.feature2

        for one_feature in rename_feature1.columns:
            if one_feature in rename_feature2.columns:
                QMessageBox().about(self, '', '{} exists in both feature matrix. Please add pre-name'.format(one_feature))
                return None
        
        merge = pd.concat([rename_feature1, rename_feature2], axis=1)
        return merge
        
    def MergeCases(self):
        if self.feature1.shape[1] != self.feature2.shape[1]:
            QMessageBox().about(self, '', 'Feature 1 and Feature 2 have different columns')
            return None

        for one_feature in self.feature1.columns:
            if one_feature not in self.feature2.columns:
                QMessageBox().about(self, '', '{} does not in Feature 2 columns.'.format(one_feature))
                return None
        for one_feature in self.feature2.columns:
            if one_feature not in self.feature1.columns:
                QMessageBox().about(self, '', '{} does not in Feature 1 columns.'.format(one_feature))
                return None
        
        for one_case in self.feature1.index:
            if one_case in self.feature2.index:
                QMessageBox().about(self, '', '{} exists in both feature matrix. Please check the case ID'.format(one_case))
                return None
        
        merge = pd.concat([self.feature1, self.feature2], axis=0)
        return merge

    def Merge(self):
        if self.feature1.size == 0 or self.feature2.size ==0:
            QMessageBox().about(self, '', 'Please Load Feature Matrix 1 and 2 both.')
            return None
        
        dlg = QFileDialog()
        file_name, _ = dlg.getSaveFileName(self, 'Save CSV feature files', 'merge_features.csv',
                                            filter="CSV files (*.csv)")
        if not file_name:
            return None
        
        if self.ui.radioMergeTypeCases.isChecked():
            merge_feature = self.MergeCases()
        else:
            merge_feature = self.MergeFeatures()
        
        if merge_feature is None:
            return None
        
        merge_feature.to_csv(file_name)
    

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_frame = FeatureMergeForm(eclog())
    main_frame.show()
    sys.exit(app.exec_())
