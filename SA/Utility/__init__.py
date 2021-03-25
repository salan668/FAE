"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/15
"""
import os
from SA.Utility.MyLog import mylog
from SA.Utility.Matric import Metric
from SA.Utility.Constant import *


def MakeFolder(root, folder_name):
    if root:
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        return folder_path
    else:
        return None


def MakeFile(root, file_name):
    if root:
        file_path = os.path.join(root, file_name)
        return file_path
    else:
        return None
