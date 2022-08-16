import sys
from PyQt5.QtWidgets import *
from HomeUI.HomePageForm import HomePageForm


if __name__ == '__main__':
    sys._enablelegacywindowsfsencoding()
    app = QApplication(sys.argv)
    main_frame = HomePageForm()
    main_frame.show()
    sys.exit(app.exec_())

