# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\MyCode\FeatureAnalysisPro\GUI\HomePage.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_HomePage(object):
    def setupUi(self, HomePage):
        HomePage.setObjectName("HomePage")
        HomePage.resize(468, 369)
        self.gridLayout = QtWidgets.QGridLayout(HomePage)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.buttonFeatureExtraction = QtWidgets.QPushButton(HomePage)
        self.buttonFeatureExtraction.setObjectName("buttonFeatureExtraction")
        self.verticalLayout.addWidget(self.buttonFeatureExtraction)
        self.buttonFeaturePreprocessing = QtWidgets.QPushButton(HomePage)
        self.buttonFeaturePreprocessing.setObjectName("buttonFeaturePreprocessing")
        self.verticalLayout.addWidget(self.buttonFeaturePreprocessing)
        self.buttonModelExploration = QtWidgets.QPushButton(HomePage)
        self.buttonModelExploration.setObjectName("buttonModelExploration")
        self.verticalLayout.addWidget(self.buttonModelExploration)
        self.buttonVisulization = QtWidgets.QPushButton(HomePage)
        self.buttonVisulization.setObjectName("buttonVisulization")
        self.verticalLayout.addWidget(self.buttonVisulization)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.verticalLayout.setStretch(0, 2)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(5, 2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.labelIntroduction = QtWidgets.QLabel(HomePage)
        self.labelIntroduction.setTextFormat(QtCore.Qt.AutoText)
        self.labelIntroduction.setAlignment(QtCore.Qt.AlignCenter)
        self.labelIntroduction.setObjectName("labelIntroduction")
        self.horizontalLayout.addWidget(self.labelIntroduction)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.retranslateUi(HomePage)
        QtCore.QMetaObject.connectSlotsByName(HomePage)

    def retranslateUi(self, HomePage):
        _translate = QtCore.QCoreApplication.translate
        HomePage.setWindowTitle(_translate("HomePage", "FeAture Explorer (Research Used Only)"))
        self.buttonFeatureExtraction.setText(_translate("HomePage", "Feature Extraction"))
        self.buttonFeaturePreprocessing.setText(_translate("HomePage", "Feature Preprocessing"))
        self.buttonModelExploration.setText(_translate("HomePage", "Model Exploration"))
        self.buttonVisulization.setText(_translate("HomePage", "Visulization"))
        self.labelIntroduction.setText(_translate("HomePage", "FeAture Explorer\n"
" V.0.1\n"
"\n"
" Yang Song (songyangmri@gmail.com)\n"
"Jing Zhang (zhangjingmri@gmail.com)\n"
"Chengxiu Zhang (cxzhang@phy.ecnu.edu.cn)\n"
"Guang Yang (gyang@phy.ecnu.edu.cn)\n"
"\n"
" Shanghai Key Laboratory of Magnetic Resonance\n"
"East China Normal University\n"
"3663, Zhongshan North Road. Shanghai, China, 200062"))

