# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI\About.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(690, 445)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "FeAture Explorer (FAE)\n"
"\n"
"Author:\n"
"Yang Song\n"
"songyangmri@gmail.com\n"
"\n"
"Jing Zhang\n"
"798582238@qq.com\n"
"\n"
"Guang Yang\n"
"gyang@phy.ecnu.edu.cn\n"
"\n"
"Shanghai Key Laboratory of Magnetic Resonance\n"
"East China Normal Univeristy\n"
"3663, Zhongshan North Rd. Shanghai, China, 200062\n"
" Tel: +86-021-62233873\n"
"\n"
" Version:\n"
"0.1.0"))

