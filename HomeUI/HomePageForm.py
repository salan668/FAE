import os
import sys

from pathlib import Path

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from HomeUI.HomePage import Ui_HomePage
from Feature.GUI import FeatureMergeForm, FeatureExtractionForm, IccEstimationForm
from BC.GUI import PrepareConnection, ProcessConnection, VisualizationConnection, ModelPredictionForm
from HomeUI.VersionConstant import VERSION

from SA.GUI.ProcessForm import ProcessForm
from SA.GUI.VisualizationForm import VisualizationForm

from Utility.EcLog import mylog

from Plugin.PluginManager import PluginManager

class HomePageForm(QDialog, Ui_HomePage):
    def __init__(self, parent=None):
        super(HomePageForm, self).__init__(parent)
        self.setupUi(self)

        self.eclog = mylog

        self.labelTitle.setText("FeAture Explorer V. {}".format(VERSION))

        self.feature_extraction = FeatureExtractionForm()
        self.feature_merge = FeatureMergeForm()
        self.feature_icc_estimation = IccEstimationForm(self.eclog)
        self.bc_preprocessing = PrepareConnection()
        self.bc_model_exploration = ProcessConnection()
        self.bc_visualization = VisualizationConnection()
        self.bc_prediction = ModelPredictionForm()
        self.sa_model_exploration = ProcessForm()
        self.sa_visualization = VisualizationForm()

        self.buttonFeatureExtraction.clicked.connect(self.OpenFeatureExtraction)
        self.feature_extraction.close_signal.connect(self.CloseFeatureExtraction)
        self.buttonFeatureMerge.clicked.connect(self.OpenFeatureMerge)
        self.feature_merge.close_signal.connect(self.CloseFeatureMerge)
        self.buttonIccEstimation.clicked.connect(self.OpenIccEstimation)
        self.feature_icc_estimation.close_signal.connect(self.CloseIccEstimation)

        self.buttonBcFeaturePreprocessing.clicked.connect(self.OpenBcPreprocessing)
        self.bc_preprocessing.close_signal.connect(self.CloseBcPreprocessing)
        self.buttonBcModelExploration.clicked.connect(self.OpenBcModelExploration)
        self.bc_model_exploration.close_signal.connect(self.CloseBcModelExploration)
        self.buttonBcVisulization.clicked.connect(self.OpenBcVisualization)
        self.bc_visualization.close_signal.connect(self.CloseBcVisualization)
        self.buttonBcPrediction.clicked.connect(self.RunBcPrediction)
        self.bc_prediction.close_signal.connect(self.CloseBcPrediction)

        self.buttonSaModelExploration.clicked.connect(self.OpenSaModelExploration)
        self.sa_model_exploration.close_signal.connect(self.CloseSaModelExploration)
        self.buttonSaVisulization.clicked.connect(self.OpenSaVisualization)
        self.sa_visualization.close_signal.connect(self.CloseSaVisualization)

        self.plugins_manager = PluginManager()
        self.buttonPluginRun.clicked.connect(self.RunPlugin)
        self.comboPlugin.currentIndexChanged.connect(self.UpdatePlugin)

        self.LoadPlugins()
        self.UpdatePlugin()

    def OpenFeatureExtraction(self):
        self.feature_extraction.show()
        self.hide()
    def CloseFeatureExtraction(self, is_close):
        if is_close:
            self.show()

    def OpenFeatureMerge(self):
        self.feature_merge.show()
        self.hide()
    def CloseFeatureMerge(self, is_close):
        if is_close:
            self.show()

    def OpenIccEstimation(self):
        self.feature_icc_estimation.show()
        self.hide()
    def CloseIccEstimation(self, is_close):
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

    def RunBcPrediction(self):
        self.bc_prediction.show()
        self.hide()
    def CloseBcPrediction(self, is_close):
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

    def LoadPlugins(self):
        self.comboPlugin.clear()
        plugin_folder = Path(os.getcwd()) / 'Plugin'
        self.plugins_manager.LoadPlugin(plugin_folder)
        for one in self.plugins_manager.plugins.keys():
            self.comboPlugin.addItem(one)

    def UpdatePlugin(self):
        # Set Logo
        if self.comboPlugin.count() == 0:
            return

        current_plugin = self.plugins_manager.plugins[self.comboPlugin.currentText()]
        if current_plugin.figure is not None:
            pixmap = QPixmap(str(current_plugin.figure))

            # wired for use 2 factor.
            pixmap = pixmap.scaled(self.labelPluginFigure.size(), PyQt5.QtCore.Qt.KeepAspectRatio)
            self.labelPluginFigure.setPixmap(pixmap)
        else:
            self.labelPluginFigure.setText('Non Logo')

        if current_plugin.description is not None:
            text = ''
            with open(str(current_plugin.description), 'r') as f:
                for one in f.read():
                    text += one

                self.textBrowser.setText(text)
        else:
            self.textBrowser.setText('None')

    def RunPlugin(self):
        file_path = self.plugins_manager.plugins[self.comboPlugin.currentText()].path
        self.showMinimized()
        os.system(str(file_path))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_frame = HomePageForm()
    main_frame.show()
    sys.exit(app.exec_())
