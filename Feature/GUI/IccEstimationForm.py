# -*- coding:utf-8 -*-
# All rights reserved. 
# Author: Yang SONG (songyangmri@gmail.com)
# 2024-05-16
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from pingouin import intraclass_corr
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtCore

from Feature.GUI.IccEstimation import Ui_IccEstimation
from Utility.EcLog import eclog


class IccEstimationThread(QThread):
    progress_signal = pyqtSignal(int)
    finish_signal = pyqtSignal(bool)

    def __init__(self, feature1: pd.DataFrame, feature2: pd.DataFrame, store_path: str,
                 icc_type: str):
        super().__init__()
        self.feature1 = feature1
        self.feature2 = feature2
        self.store_path = store_path
        self.icc_type = icc_type

    def run(self):
        icc_df = {}
        total = len(self.feature1.columns)
        self.progress_signal.emit(0)

        for ind, col in enumerate(self.feature1.columns):
            temp_df = pd.DataFrame()
            temp_df[col] = pd.concat([self.feature1[col], self.feature2[col]], axis=0)
            temp_df["raters"] = ['A' for _ in self.feature1.index] + ['B' for _ in self.feature1.index]
            temp_df['cases'] = [ind for ind in self.feature1.index] + [ind for ind in self.feature1.index]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = intraclass_corr(temp_df, targets='cases', raters="raters", ratings=col)

            icc = result.set_index('Type').loc[self.icc_type]['ICC']

            if pd.isna(icc):
                icc = 1

            icc_df[col] = icc
            self.progress_signal.emit(int((ind + 1) / total * 100))

        print(icc_df)
        df = pd.DataFrame(icc_df, index=['ICC'])
        df.to_csv(self.store_path)
        self.progress_signal.emit(100)
        self.finish_signal.emit(True)


class IccEstimationForm(QWidget):
    close_signal = QtCore.pyqtSignal(bool)

    def __init__(self, eclog=eclog):
        super().__init__()
        self.ui = Ui_IccEstimation()
        self.ui.setupUi(self)

        self.feature1 = pd.DataFrame()
        self.feature2 = pd.DataFrame()
        self.eclog = eclog

        self.ui.buttonLoadFeatureMatrix1.clicked.connect(self.LoadFeature1)
        self.ui.buttonLoadFeatureMatrix2.clicked.connect(self.LoadFeature2)

        self.ui.buttonIccEstimate.clicked.connect(self.IccEstimation)

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
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'CSV files (*.csv)')
        if file_name:
            line_edit.setText(file_name)
            feature = pd.read_csv(file_name, index_col=0)
            self._UpdateTable(table_view, feature)
            if feature_num == 1:
                self.feature1 = feature
            else:
                self.feature2 = feature

    def LoadFeature1(self):
        self._LoadButton(self.ui.lineEditFeature1Path, self.ui.tableFeature1, 1)

    def LoadFeature2(self):
        self._LoadButton(self.ui.lineEditFeature2Path, self.ui.tableFeature2, 2)
    def compare_indices(self, df1, df2):
        diff1 = df1.index.difference(df2.index)
        diff2 = df2.index.difference(df1.index)

        message = ''
        if not diff1.empty:
            message += "Indices in df1 not in df2:\n" + "{}\n".format(diff1)
        if not diff2.empty:
            message += "Indices in df2 not in df1:\n" + "{}\n".format(diff2)
        return message

    def compare_columns(self, df1, df2):
        diff1 = df1.columns.difference(df2.columns)
        diff2 = df2.columns.difference(df1.columns)

        message = ''
        if not diff1.empty:
            message += "Columns in df1 not in df2:\n" + "{}\n".format(diff1)
        if not diff2.empty:
            message += "Columns in df2 not in df1:\n" + "{}\n".format(diff2)
        return message

    def IccEstimationFinish(self, state):
        if state:
            QMessageBox.about(self, 'Information', 'The ICC estimation is done')

    def IccEstimation(self):
        if self.feature1.empty or self.feature2.empty:
            QMessageBox.warning(self, 'Warning', 'Please load the feature matrix first')
            self.eclog.error('Please load the feature matrix first')
            return

        self.eclog.info('Icc estimation is running')

        if not self.feature1.index.equals(self.feature2.index):
            messsage = self.compare_indices(self.feature1, self.feature2)
            QMessageBox.warning(self, 'Warning', 'The index of two feature matrix is not the same: \n' + '{}'.format(messsage))
            self.eclog.error('The index of two feature matrix is not the same.\n{}'.format(messsage))
            return

        if not self.feature1.columns.equals(self.feature2.columns):
            messsage = self.compare_columns(self.feature1, self.feature2)
            QMessageBox.warning(self, 'Warning', 'The columns of two feature matrix is not the same: \n' + '{}'.format(messsage))
            self.eclog.error('The columns of two feature matrix is not the same.\n{}'.format(messsage))
            return

        store_path, _ = QFileDialog.getSaveFileName(self, 'Save file', 'ICC_Estimation.csv', 'CSV files (*.csv)')
        if not store_path:
            return

        self.thread = IccEstimationThread(self.feature1, self.feature2, store_path,
                                          self.ui.comboIccType.currentText())
        self.thread.progress_signal.connect(self.ui.progressBar.setValue)
        self.thread.finish_signal.connect(self.IccEstimationFinish)
        self.thread.start()



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_frame = IccEstimationForm(eclog('Eclog.txt').GetLogger())
    main_frame.show()
    sys.exit(app.exec_())
