"""
All rights reserved.
--Yang Song, Apr 8th, 2020
"""

import os

def MakeFolder(root, folder_name):
    if root:
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        return folder_path
    else:
        return ''