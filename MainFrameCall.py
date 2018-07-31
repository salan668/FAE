import sys
from PyQt5.QtWidgets import *
from FAEGUI.MainFrame import Ui_TabWidget
# import qdarkstyle

class MainFrame(QTabWidget, Ui_TabWidget):
    def __init__(self, parent=None):
        super(MainFrame, self).__init__(parent)
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_frame = MainFrame()
    main_frame.show()
    sys.exit(app.exec_())
