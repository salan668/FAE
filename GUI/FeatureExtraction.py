# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FeatureExtraction.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_EeatureExtractionDialog(object):
    def setupUi(self, EeatureExtractionDialog):
        EeatureExtractionDialog.setObjectName("EeatureExtractionDialog")
        EeatureExtractionDialog.resize(773, 462)
        self.gridLayout = QtWidgets.QGridLayout(EeatureExtractionDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.line = QtWidgets.QFrame(EeatureExtractionDialog)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 4, 0, 1, 1)
        self.FeaturePathLineEdit = QtWidgets.QLineEdit(EeatureExtractionDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.FeaturePathLineEdit.setFont(font)
        self.FeaturePathLineEdit.setObjectName("FeaturePathLineEdit")
        self.gridLayout.addWidget(self.FeaturePathLineEdit, 3, 1, 1, 1)
        self.label = QtWidgets.QLabel(EeatureExtractionDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(EeatureExtractionDialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.DataPathEdit = QtWidgets.QLineEdit(EeatureExtractionDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.DataPathEdit.setFont(font)
        self.DataPathEdit.setObjectName("DataPathEdit")
        self.gridLayout.addWidget(self.DataPathEdit, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(EeatureExtractionDialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.BrowseButton = QtWidgets.QPushButton(EeatureExtractionDialog)
        self.BrowseButton.setObjectName("BrowseButton")
        self.gridLayout.addWidget(self.BrowseButton, 0, 2, 1, 1)
        self.DatakeyLineEdit = QtWidgets.QLineEdit(EeatureExtractionDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.DatakeyLineEdit.setFont(font)
        self.DatakeyLineEdit.setObjectName("DatakeyLineEdit")
        self.gridLayout.addWidget(self.DatakeyLineEdit, 1, 1, 1, 1)
        self.RoiKeyLineEdit = QtWidgets.QLineEdit(EeatureExtractionDialog)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.RoiKeyLineEdit.setFont(font)
        self.RoiKeyLineEdit.setObjectName("RoiKeyLineEdit")
        self.gridLayout.addWidget(self.RoiKeyLineEdit, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(EeatureExtractionDialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.SelectButton = QtWidgets.QPushButton(EeatureExtractionDialog)
        self.SelectButton.setObjectName("SelectButton")
        self.gridLayout.addWidget(self.SelectButton, 3, 2, 1, 1)
        self.ExtractionButton = QtWidgets.QPushButton(EeatureExtractionDialog)
        self.ExtractionButton.setObjectName("ExtractionButton")
        self.gridLayout.addWidget(self.ExtractionButton, 5, 2, 1, 1)
        self.OutputInfo = QtWidgets.QLabel(EeatureExtractionDialog)
        self.OutputInfo.setObjectName("OutputInfo")
        self.gridLayout.addWidget(self.OutputInfo, 5, 0, 1, 2)

        self.retranslateUi(EeatureExtractionDialog)
        QtCore.QMetaObject.connectSlotsByName(EeatureExtractionDialog)

    def retranslateUi(self, EeatureExtractionDialog):
        _translate = QtCore.QCoreApplication.translate
        EeatureExtractionDialog.setWindowTitle(_translate("EeatureExtractionDialog", "Feature Extraction"))
        self.label.setText(_translate("EeatureExtractionDialog", "DataPath:"))
        self.label_3.setText(_translate("EeatureExtractionDialog", "RoiKey："))
        self.label_2.setText(_translate("EeatureExtractionDialog", "DataKey："))
        self.BrowseButton.setText(_translate("EeatureExtractionDialog", "Browse……"))
        self.label_4.setText(_translate("EeatureExtractionDialog", "Feature Path："))
        self.SelectButton.setText(_translate("EeatureExtractionDialog", "Select"))
        self.ExtractionButton.setText(_translate("EeatureExtractionDialog", "Go!"))
        self.OutputInfo.setText(_translate("EeatureExtractionDialog", "extraction："))


