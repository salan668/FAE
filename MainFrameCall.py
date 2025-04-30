import sys
import os

import multiprocessing
from PyQt5.QtWidgets import *
from HomeUI.HomePageForm import HomePageForm


if __name__ == '__main__':
    current_path = os.getcwd()
    if getattr(sys, 'frozen', False): # 检测是否为打包后的程序 
        # 获取程序所在目录 
        application_path = os.path.dirname(sys.executable) 
    else: # 脚本运行时，获取脚本所在目录 
        application_path = os.path.dirname(os.path.abspath(__file__))# 设置工作目录为程序所在路径 
        os.chdir(application_path)
        current_path = os.getcwd()

    if sys.platform.startswith("win"):
        sys._enablelegacywindowsfsencoding()
        
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    main_frame = HomePageForm()
    main_frame.show()
    sys.exit(app.exec_())

