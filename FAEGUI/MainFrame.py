# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI\MainFrame.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

from FAEGUI.PrepareConnection import PrepareConnection
from FAEGUI.ProcessConnection import ProcessConnection
from FAEGUI.VisualizationConnection import VisualizationConnection
from FAEGUI.AboutConnection import AboutConnection

class Ui_TabWidget(object):
    def setupUi(self, TabWidget):

        self.prepare = PrepareConnection()
        self.process = ProcessConnection()
        self.visualization = VisualizationConnection()
        self.about = AboutConnection()

        TabWidget.setObjectName("TabWidget")
        TabWidget.resize(1920, 1000)
        TabWidget.move(0, 0)

        self.tabPrepare = self.prepare
        TabWidget.addTab(self.tabPrepare, "")

        self.tabProcess = self.process
        TabWidget.addTab(self.tabProcess, "")

        self.tabVisualization = self.visualization
        TabWidget.addTab(self.tabVisualization, "")

        self.tabAbout = self.about
        TabWidget.addTab(self.tabAbout, "")

        self.retranslateUi(TabWidget)
        TabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(TabWidget)

    def retranslateUi(self, TabWidget):
        _translate = QtCore.QCoreApplication.translate
        TabWidget.setWindowTitle(_translate("TabWidget", "FeAture Explorer (FAE), v.0.1.0"))

        TabWidget.setTabText(TabWidget.indexOf(self.tabPrepare), _translate("TabWidget", "Prepare"))
        TabWidget.setTabText(TabWidget.indexOf(self.tabProcess), _translate("TabWidget", "Process"))
        TabWidget.setTabText(TabWidget.indexOf(self.tabVisualization), _translate("TabWidget", "Visualization"))
        TabWidget.setTabText(TabWidget.indexOf(self.tabAbout), _translate("TabWidget", "About"))