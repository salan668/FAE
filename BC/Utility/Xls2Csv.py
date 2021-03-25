"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2020/8/20
"""

import os
import pandas as pd


def ConvertXls2Csv(xls_file):
    df = pd.read_excel(xls_file, index_col=0)
    if xls_file.endswith('xls'):
        df.to_csv(os.path.join(os.path.split(xls_file)[0], os.path.split(xls_file)[1].replace('xls', 'csv')))
    elif xls_file.endswith('xlsx'):
        df.to_csv(os.path.join(os.path.split(xls_file)[0], os.path.split(xls_file)[1].replace('xlsx', 'csv')))


if __name__ == '__main__':
    pass