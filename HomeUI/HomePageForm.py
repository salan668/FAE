import sys

from PyQt5.QtWidgets import *

from HomeUI.HomePage import Ui_HomePage
from BC.GUI import FeatureExtractionForm, PrepareConnection, ProcessConnection, VisualizationConnection
from VersionConstant import VERSION

from SA.GUI.ProcessForm import ProcessForm
from SA.GUI.VisualizationForm import VisualizationForm


class HomePageForm(QDialog, Ui_HomePage):
    def __init__(self, parent=None):
        super(HomePageForm, self).__init__(parent)
        self.setupUi(self)

        self.labelTitle.setText("FeAture Explorer V. {}".format(VERSION))

        self.feature_extraction = FeatureExtractionForm()
        self.bc_preprocessing = PrepareConnection()
        self.bc_model_exploration = ProcessConnection()
        self.bc_visualization = VisualizationConnection()
        self.sa_model_exploration = ProcessForm()
        self.sa_visualization = VisualizationForm()

        self.buttonFeatureExtraction.clicked.connect(self.OpenFeatureExtraction)
        self.feature_extraction.close_signal.connect(self.CloseFeatureExtraction)

        self.buttonBcFeaturePreprocessing.clicked.connect(self.OpenBcPreprocessing)
        self.bc_preprocessing.close_signal.connect(self.CloseBcPreprocessing)
        self.buttonBcModelExploration.clicked.connect(self.OpenBcModelExploration)
        self.bc_model_exploration.close_signal.connect(self.CloseBcModelExploration)
        self.buttonBcVisulization.clicked.connect(self.OpenBcVisualization)
        self.bc_visualization.close_signal.connect(self.CloseBcVisualization)

        self.buttonSaModelExploration.clicked.connect(self.OpenSaModelExploration)
        self.sa_model_exploration.close_signal.connect(self.CloseSaModelExploration)
        self.buttonSaVisulization.clicked.connect(self.OpenSaVisualization)
        self.sa_visualization.close_signal.connect(self.CloseSaVisualization)


    def OpenFeatureExtraction(self):
        self.feature_extraction.show()
        self.hide()
    def CloseFeatureExtraction(self, is_close):
        if is_close:
            self.show()

    def OpenBcPreprocessing(self):
        self.bc_preprocessing.show()
        self.hide()
    def CloseBcPreprocessing(self, is_close):
        if is_close:
            self.show()

    def OpenBcModelExploration(self):
        self.bc_model_exploration.show()
        self.hide()
    def CloseBcModelExploration(self, is_close):
        if is_close:
            self.show()

    def OpenBcVisualization(self):
        self.bc_visualization.show()
        self.hide()
    def CloseBcVisualization(self, is_close):
        if is_close:
            self.show()

    def OpenSaModelExploration(self):
        self.sa_model_exploration.show()
        self.hide()
    def CloseSaModelExploration(self, is_close):
        if is_close:
            self.show()

    def OpenSaVisualization(self):
        self.sa_visualization.show()
        self.hide()
    def CloseSaVisualization(self, is_close):
        if is_close:
            self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_frame = HomePageForm()
    main_frame.show()
    sys.exit(app.exec_())
