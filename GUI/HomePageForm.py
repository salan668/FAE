import os
import sys

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui

from Utility.EcLog import eclog
from GUI.HomePage import Ui_HomePage
# from GUI.FeatureExtractionForm import FeatureExtractionForm
from GUI.FeatureExtractionForm2 import FeatureExtractionForm
from GUI.PrepareForm import PrepareConnection
from GUI.ProcessForm import ProcessConnection
from GUI.VisualizationForm import VisualizationConnection
from Utility.Constants import VERSION

class HomePageForm(QDialog, Ui_HomePage):
    def __init__(self, parent=None):
        super(HomePageForm, self).__init__(parent)
        self.setupUi(self)

        self.labelIntroduction.setText(
            "FeAture Explorer\n "
            "V. {}\n\n"
            "Yang Song (songyangmri@gmail.com)\n"
            "Jing Zhang (zhangjingmri@gmail.com)\n"
            "Chengxiu Zhang (cxzhang@phy.ecnu.edu.cn)\n"
            "Guang Yang (gyang@phy.ecnu.edu.cn)\n\n "
            "Shanghai Key Laboratory of Magnetic Resonance\n"
            "East China Normal University\n"
            "3663, Zhongshan North Road. Shanghai, China, 200062".format(VERSION)
        )

        self.feature_extraction = FeatureExtractionForm()
        self.preprocessing = PrepareConnection()
        self.model_exploration = ProcessConnection()
        self.visulization = VisualizationConnection()

        self.buttonFeatureExtraction.clicked.connect(self.OpenFeatureExtraction)
        self.feature_extraction.close_signal.connect(self.CloseFeatureExtraction)
        self.buttonFeaturePreprocessing.clicked.connect(self.OpenPreprocessing)
        self.preprocessing.close_signal.connect(self.ClosePreprocessing)
        self.buttonModelExploration.clicked.connect(self.OpenModelExploration)
        self.model_exploration.close_signal.connect(self.CloseModelExploration)
        self.buttonVisulization.clicked.connect(self.OpenVisulization)
        self.visulization.close_signal.connect(self.CloseVisulization)


    def OpenFeatureExtraction(self):
        self.feature_extraction.show()
        self.hide()
    def CloseFeatureExtraction(self, is_close):
        if is_close:
            self.show()

    def OpenPreprocessing(self):
        self.preprocessing.show()
        self.hide()
    def ClosePreprocessing(self, is_close):
        if is_close:
            self.show()

    def OpenModelExploration(self):
        self.model_exploration.show()
        self.hide()
    def CloseModelExploration(self, is_close):
        if is_close:
            self.show()

    def OpenVisulization(self):
        self.visulization.show()
        self.hide()
    def CloseVisulization(self, is_close):
        if is_close:
            self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_frame = HomePageForm()
    main_frame.show()
    sys.exit(app.exec_())
