# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Visualization.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Visualization(object):
    def setupUi(self, Visualization):
        Visualization.setObjectName("Visualization")
        Visualization.resize(1373, 914)
        self.gridLayout_2 = QtWidgets.QGridLayout(Visualization)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.buttonLoadResult = QtWidgets.QPushButton(Visualization)
        self.buttonLoadResult.setObjectName("buttonLoadResult")
        self.verticalLayout.addWidget(self.buttonLoadResult)
        self.buttonClearResult = QtWidgets.QPushButton(Visualization)
        self.buttonClearResult.setEnabled(True)
        self.buttonClearResult.setObjectName("buttonClearResult")
        self.verticalLayout.addWidget(self.buttonClearResult)
        self.lineEditResultPath = QtWidgets.QLineEdit(Visualization)
        self.lineEditResultPath.setObjectName("lineEditResultPath")
        self.verticalLayout.addWidget(self.lineEditResultPath)
        self.textEditDescription = QtWidgets.QTextEdit(Visualization)
        self.textEditDescription.setObjectName("textEditDescription")
        self.verticalLayout.addWidget(self.textEditDescription)
        self.buttonSaveFigure = QtWidgets.QPushButton(Visualization)
        self.buttonSaveFigure.setEnabled(False)
        self.buttonSaveFigure.setObjectName("buttonSaveFigure")
        self.verticalLayout.addWidget(self.buttonSaveFigure)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_7 = QtWidgets.QLabel(Visualization)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.comboSheet = QtWidgets.QComboBox(Visualization)
        self.comboSheet.setObjectName("comboSheet")
        self.verticalLayout_2.addWidget(self.comboSheet)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.tableClinicalStatistic = QtWidgets.QTableWidget(Visualization)
        self.tableClinicalStatistic.setObjectName("tableClinicalStatistic")
        self.tableClinicalStatistic.setColumnCount(0)
        self.tableClinicalStatistic.setRowCount(0)
        self.horizontalLayout_3.addWidget(self.tableClinicalStatistic)
        self.textEditModelDescription = QtWidgets.QTextEdit(Visualization)
        self.textEditModelDescription.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textEditModelDescription.setObjectName("textEditModelDescription")
        self.horizontalLayout_3.addWidget(self.textEditModelDescription)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(2, 2)
        self.horizontalLayout_3.setStretch(3, 3)
        self.verticalLayout_10.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_2 = QtWidgets.QLabel(Visualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(0, 12))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 12))
        self.label_2.setObjectName("label_2")
        self.verticalLayout_9.addWidget(self.label_2)
        self.textEditorRefDescription = QtWidgets.QTextEdit(Visualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEditorRefDescription.sizePolicy().hasHeightForWidth())
        self.textEditorRefDescription.setSizePolicy(sizePolicy)
        self.textEditorRefDescription.setMaximumSize(QtCore.QSize(16777215, 200))
        self.textEditorRefDescription.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textEditorRefDescription.setObjectName("textEditorRefDescription")
        self.verticalLayout_9.addWidget(self.textEditorRefDescription)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.tableRefData = QtWidgets.QTableWidget(Visualization)
        self.tableRefData.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tableRefData.setObjectName("tableRefData")
        self.tableRefData.setColumnCount(0)
        self.tableRefData.setRowCount(0)
        self.verticalLayout_8.addWidget(self.tableRefData)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_9 = QtWidgets.QLabel(Visualization)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout.addWidget(self.label_9)
        self.comboSurvivalModel = QtWidgets.QComboBox(Visualization)
        self.comboSurvivalModel.setObjectName("comboSurvivalModel")
        self.horizontalLayout.addWidget(self.comboSurvivalModel)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 5)
        self.verticalLayout_8.addLayout(self.horizontalLayout)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioSurvivalSplitDataset = QtWidgets.QRadioButton(Visualization)
        self.radioSurvivalSplitDataset.setChecked(True)
        self.radioSurvivalSplitDataset.setObjectName("radioSurvivalSplitDataset")
        self.buttonGroup = QtWidgets.QButtonGroup(Visualization)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioSurvivalSplitDataset)
        self.horizontalLayout_2.addWidget(self.radioSurvivalSplitDataset)
        self.radioSurvivalSplitFeature = QtWidgets.QRadioButton(Visualization)
        self.radioSurvivalSplitFeature.setObjectName("radioSurvivalSplitFeature")
        self.buttonGroup.addButton(self.radioSurvivalSplitFeature)
        self.horizontalLayout_2.addWidget(self.radioSurvivalSplitFeature)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_2)
        self.checkSurvivalKM = QtWidgets.QCheckBox(Visualization)
        self.checkSurvivalKM.setObjectName("checkSurvivalKM")
        self.horizontalLayout_10.addWidget(self.checkSurvivalKM)
        self.verticalLayout_8.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.checkSurvivalTrain = QtWidgets.QCheckBox(Visualization)
        self.checkSurvivalTrain.setChecked(True)
        self.checkSurvivalTrain.setObjectName("checkSurvivalTrain")
        self.verticalLayout_7.addWidget(self.checkSurvivalTrain)
        self.checkSurvivalCvVal = QtWidgets.QCheckBox(Visualization)
        self.checkSurvivalCvVal.setChecked(False)
        self.checkSurvivalCvVal.setObjectName("checkSurvivalCvVal")
        self.verticalLayout_7.addWidget(self.checkSurvivalCvVal)
        self.checkSurvivalTest = QtWidgets.QCheckBox(Visualization)
        self.checkSurvivalTest.setObjectName("checkSurvivalTest")
        self.verticalLayout_7.addWidget(self.checkSurvivalTest)
        self.horizontalLayout_11.addLayout(self.verticalLayout_7)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.buttonLoadRefData = QtWidgets.QPushButton(Visualization)
        self.buttonLoadRefData.setObjectName("buttonLoadRefData")
        self.verticalLayout_3.addWidget(self.buttonLoadRefData)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_10 = QtWidgets.QLabel(Visualization)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_9.addWidget(self.label_10)
        self.comboRefFeature = QtWidgets.QComboBox(Visualization)
        self.comboRefFeature.setObjectName("comboRefFeature")
        self.horizontalLayout_9.addWidget(self.comboRefFeature)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.lineEditRefSplit = QtWidgets.QLineEdit(Visualization)
        self.lineEditRefSplit.setObjectName("lineEditRefSplit")
        self.horizontalLayout_7.addWidget(self.lineEditRefSplit)
        self.buttonSplitShow = QtWidgets.QPushButton(Visualization)
        self.buttonSplitShow.setObjectName("buttonSplitShow")
        self.horizontalLayout_7.addWidget(self.buttonSplitShow)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_11.addLayout(self.verticalLayout_3)
        self.verticalLayout_8.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12.addLayout(self.verticalLayout_8)
        self.canvasSurvival = MatplotlibWidget(Visualization)
        self.canvasSurvival.setMinimumSize(QtCore.QSize(600, 600))
        self.canvasSurvival.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.canvasSurvival.setObjectName("canvasSurvival")
        self.horizontalLayout_12.addWidget(self.canvasSurvival)
        self.verticalLayout_9.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_13.addLayout(self.verticalLayout_9)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_4 = QtWidgets.QLabel(Visualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(0, 12))
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 12))
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.canvasFeature = MatplotlibWidget(Visualization)
        self.canvasFeature.setMinimumSize(QtCore.QSize(400, 400))
        self.canvasFeature.setObjectName("canvasFeature")
        self.verticalLayout_6.addWidget(self.canvasFeature)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_11 = QtWidgets.QLabel(Visualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QtCore.QSize(0, 20))
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_6.addWidget(self.label_11)
        self.comboModelContribution = QtWidgets.QComboBox(Visualization)
        self.comboModelContribution.setObjectName("comboModelContribution")
        self.horizontalLayout_6.addWidget(self.comboModelContribution)
        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(1, 5)
        self.verticalLayout_5.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_12 = QtWidgets.QLabel(Visualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QtCore.QSize(0, 20))
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_8.addWidget(self.label_12)
        self.spinCoefficientBias = QtWidgets.QDoubleSpinBox(Visualization)
        self.spinCoefficientBias.setMinimum(0.1)
        self.spinCoefficientBias.setMaximum(0.9)
        self.spinCoefficientBias.setSingleStep(0.05)
        self.spinCoefficientBias.setProperty("value", 0.1)
        self.spinCoefficientBias.setObjectName("spinCoefficientBias")
        self.horizontalLayout_8.addWidget(self.spinCoefficientBias)
        self.verticalLayout_5.addLayout(self.horizontalLayout_8)
        self.gridLayout.addLayout(self.verticalLayout_5, 1, 1, 1, 1)
        self.radioContribution = QtWidgets.QRadioButton(Visualization)
        self.radioContribution.setChecked(True)
        self.radioContribution.setObjectName("radioContribution")
        self.buttonGroup_2 = QtWidgets.QButtonGroup(Visualization)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.radioContribution)
        self.gridLayout.addWidget(self.radioContribution, 0, 1, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.checkCindexCV = QtWidgets.QCheckBox(Visualization)
        self.checkCindexCV.setChecked(True)
        self.checkCindexCV.setObjectName("checkCindexCV")
        self.horizontalLayout_4.addWidget(self.checkCindexCV)
        self.checkCindexTrain = QtWidgets.QCheckBox(Visualization)
        self.checkCindexTrain.setObjectName("checkCindexTrain")
        self.horizontalLayout_4.addWidget(self.checkCindexTrain)
        self.checkCindexTest = QtWidgets.QCheckBox(Visualization)
        self.checkCindexTest.setObjectName("checkCindexTest")
        self.horizontalLayout_4.addWidget(self.checkCindexTest)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_13 = QtWidgets.QLabel(Visualization)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_5.addWidget(self.label_13)
        self.comboCindexModel = QtWidgets.QComboBox(Visualization)
        self.comboCindexModel.setObjectName("comboCindexModel")
        self.horizontalLayout_5.addWidget(self.comboCindexModel)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 5)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.gridLayout.addLayout(self.verticalLayout_4, 1, 0, 1, 1)
        self.radioVariance = QtWidgets.QRadioButton(Visualization)
        self.radioVariance.setObjectName("radioVariance")
        self.buttonGroup_2.addButton(self.radioVariance)
        self.gridLayout.addWidget(self.radioVariance, 0, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.verticalLayout_6.addLayout(self.gridLayout)
        self.horizontalLayout_13.addLayout(self.verticalLayout_6)
        self.verticalLayout_10.addLayout(self.horizontalLayout_13)
        self.gridLayout_2.addLayout(self.verticalLayout_10, 0, 0, 1, 1)

        self.retranslateUi(Visualization)
        QtCore.QMetaObject.connectSlotsByName(Visualization)

    def retranslateUi(self, Visualization):
        _translate = QtCore.QCoreApplication.translate
        Visualization.setWindowTitle(_translate("Visualization", "Visualization"))
        self.buttonLoadResult.setText(_translate("Visualization", "Load Result"))
        self.buttonClearResult.setText(_translate("Visualization", "Clear"))
        self.buttonSaveFigure.setText(_translate("Visualization", "Save Figure"))
        self.label_7.setText(_translate("Visualization", "Show:"))
        self.label_2.setText(_translate("Visualization", "Survival Prediction"))
        self.label_9.setText(_translate("Visualization", "Model"))
        self.radioSurvivalSplitDataset.setText(_translate("Visualization", "Split By DataSet"))
        self.radioSurvivalSplitFeature.setText(_translate("Visualization", "Split By Feature"))
        self.checkSurvivalKM.setText(_translate("Visualization", "KM Result"))
        self.checkSurvivalTrain.setText(_translate("Visualization", "Train"))
        self.checkSurvivalCvVal.setText(_translate("Visualization", "CV Val"))
        self.checkSurvivalTest.setText(_translate("Visualization", "Test"))
        self.buttonLoadRefData.setText(_translate("Visualization", "Ref Data Load"))
        self.label_10.setText(_translate("Visualization", "Split Feature"))
        self.buttonSplitShow.setText(_translate("Visualization", "Split Show"))
        self.label_4.setText(_translate("Visualization", "Model Visualization"))
        self.label_11.setText(_translate("Visualization", "Model"))
        self.label_12.setText(_translate("Visualization", "Stretch Window"))
        self.radioContribution.setText(_translate("Visualization", "Feature Contribution"))
        self.checkCindexCV.setText(_translate("Visualization", "CV Val"))
        self.checkCindexTrain.setText(_translate("Visualization", "Train"))
        self.checkCindexTest.setText(_translate("Visualization", "Test"))
        self.label_13.setText(_translate("Visualization", "Model"))
        self.radioVariance.setText(_translate("Visualization", "C-index Variance"))
from MatplotlibWidget import MatplotlibWidget
