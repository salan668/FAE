import sys
from PyQt5.QtWidgets import *
from HomePage.HomePageForm import HomePageForm

# import qdarkstyle


if __name__ == '__main__':
    sys._enablelegacywindowsfsencoding()
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_frame = HomePageForm()
    main_frame.show()
    sys.exit(app.exec_())

