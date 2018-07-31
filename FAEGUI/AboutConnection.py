from PyQt5.QtWidgets import *
from GUI.About import Ui_Form


class AboutConnection(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(AboutConnection, self).__init__(parent)
        self.setupUi(self)
