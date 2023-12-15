import os
import glob
import csv

import pandas as pd
from reportlab.lib import colors

from BC.FeatureAnalysis.FeaturePipeline import OnePipeline
from BC.DataContainer.DataContainer import DataContainer
from BC.Description.MyPDFDocument import MyPdfDocument
from BC.FeatureAnalysis.Pipelines import PipelinesManager
from BC.FeatureAnalysis.IndexDict import Index2Dict
from BC.Utility.Constants import *
from HomeUI.VersionConstant import VERSION


class Description:
    def __init__(self, pipeline_manager=PipelinesManager()):
        self.__paragraph_list = []
        self.__current_paragraph = ''
        self.__manager = pipeline_manager
        pass

    def _DataDescription(self, df):
        # Data Description
        data_description_text = "    "
        data_description_text += "We selected {:d} cases as the training data set ({:d}/{:d} = positive/negative)). ". \
            format(int(df.loc['{}_{}'.format(TRAIN, NUMBER)][0]),
                   int(df.loc['{}_{}'.format(TRAIN, POS_NUM)][0]),
                   int(df.loc['{}_{}'.format(TRAIN, NEG_NUM)][0]))

        if '{}_{}'.format(TEST, NUMBER) not in df.index:
            data_description_text += "Since the number of the samples were limited, there was no independent testing data. "
        else:
            data_description_text += "We also selected another {:d} cases as the independent testing data " \
                                     "set ({:d}/{:d} = positive/negative). \n".format(
                int(df.loc['{}_{}'.format(TEST, NUMBER)][0]),
                int(df.loc['{}_{}'.format(TEST, POS_NUM)][0]),
                int(df.loc['{}_{}'.format(TEST, NEG_NUM)][0]))

        return data_description_text

    def _MethodDescription(self, pipeline_name, result_root):
        index_dict = Index2Dict()
        norm_folder, dr_folder, fs_folder, cls_folder = self.__manager.SplitFolder(pipeline_name, result_root)

        normalizer = index_dict.GetInstantByIndex(os.path.split(norm_folder)[1])
        dr = index_dict.GetInstantByIndex(os.path.split(dr_folder)[1])
        fs = index_dict.GetInstantByIndex(os.path.split(fs_folder)[1].split('_')[0])
        cls = index_dict.GetInstantByIndex(os.path.split(cls_folder)[1])

        with open(os.path.join(result_root, 'pipeline_info.csv'), 'r') as file:
            rows = csv.reader(file)
            for row in rows:
                if CROSS_VALIDATION == row[0]:
                    cv = index_dict.GetInstantByIndex(row[1])
                elif BALANCE == row[0]:
                    balance = index_dict.GetInstantByIndex(row[1])

        method_description = "    "
        method_description += balance.GetDescription()
        method_description += normalizer.GetDescription()
        method_description += dr.GetDescription()
        method_description += fs.GetDescription()
        method_description += cls.GetDescription()
        method_description += cv.GetDescription()
        method_description += "\n"

        return method_description

    def _MetricDescription(self, fn, df):
        if '{}_{}'.format(TEST, NUMBER) in df.index:
            result_description = "We found that the model based on {:d} features can get the highest AUC on the " \
                                 "validation data set. The AUC and the accuracy could achieve {:.3f} and {:.3f}, " \
                                 "respectively. In this point, The AUC and the accuracy of the model achieve {:.3f} " \
                                 "and {:.3f} on testing data set. The clinical statistics in the diagonsis and " \
                                 "the selected features were shown in Table 1 and Table 2. The ROC curve was shown " \
                                 "in Figure 1.\n".format(int(fn),
                                                         float(df.loc['{}_{}'.format(CV_VAL, AUC)][0]),
                                                         float(df.loc['{}_{}'.format(CV_VAL, ACC)][0]),
                                                         float(df.loc['{}_{}'.format(TEST, AUC)][0]),
                                                         float(df.loc['{}_{}'.format(TEST, ACC)][0]))

        else:
            result_description = "We found that the model based on {:d} features can get the highest AUC on the " \
                                 "validation data set. The AUC and the accuracy could achieve {:.3f} and {:.3f}, " \
                                 "The clinical statistics in the diagonsis and the selected features were shown in " \
                                 "Table 1 and Table 2. The ROC curve was shown in Figure 1. \n" \
                                 "".format(int(fn),
                                           float(df.loc['{}_{}'.format(CV_VAL, AUC)][0]),
                                           float(df.loc['{}_{}'.format(CV_VAL, ACC)][0]))
        return result_description


    def _StatisticTable(self, df):
        header = "Table 1. Clinical statistics in the diagnosis. "

        if '{}_{}'.format(TEST, NUMBER) in df.index:
            content = [['Statistics', 'Value'],
                       ['Accuracy', str(df.loc['{}_{}'.format(TEST, ACC)][0])],
                       ['AUC', str(df.loc['{}_{}'.format(TEST, AUC)][0])],
                       ['AUC 95% CIs', str(df.loc['{}_{}'.format(TEST, AUC_CI)][0])],
                       ['NPV', str(df.loc['{}_{}'.format(TEST, NPV)][0])],
                       ['PPV', str(df.loc['{}_{}'.format(TEST, PPV)][0])],
                       ['Sensitivity', str(df.loc['{}_{}'.format(TEST, SEN)][0])],
                       ['Specificity', str(df.loc['{}_{}'.format(TEST, SPE)][0])]]
        else:
            content = [['Statistics', 'Value'],
                       ['Accuracy', str(df.loc['{}_{}'.format(CV_VAL, ACC)][0])],
                       ['AUC', str(df.loc['{}_{}'.format(CV_VAL, AUC)][0])],
                       ['AUC 95% CIs', str(df.loc['{}_{}'.format(CV_VAL, AUC_CI)][0])],
                       ['NPV', str(df.loc['{}_{}'.format(CV_VAL, NPV)][0])],
                       ['PPV', str(df.loc['{}_{}'.format(CV_VAL, PPV)][0])],
                       ['Sensitivity', str(df.loc['{}_{}'.format(CV_VAL, SEN)][0])],
                       ['Specificity', str(df.loc['{}_{}'.format(CV_VAL, SPE)][0])]]
        return header, content

    def _FeatureTable(self, fs_folder, cls_folder):
        candidate_file = glob.glob(os.path.join(cls_folder, '*coef.csv'))
        if len(candidate_file) > 0:
            coef = pd.read_csv(candidate_file[0], index_col=0, header=0)
            header = 'Table 2. The coefficients of features in the model. '
            content = [['Features', 'Coef in model']]
            for index in coef.index:
                content.append([str(index), "{:.3f}".format(coef.loc[index].values[0])])

        else:
            with open(os.path.join(fs_folder, 'feature_select_info.csv'), 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == 'selected_feature':
                        features = row[1:]
            header = 'Table 2. The rank of selectedfeatures. '
            content = [['Features', 'Rank']]
            for index in range(len(features)):
                content.append([features[index], str(index + 1)])

        return header, content

    def Run(self, pipeline_name, result_root, store_folder):
        norm_folder, dr_folder, fs_folder, cls_folder = self.__manager.SplitFolder(pipeline_name, result_root)

        metric_path = os.path.join(cls_folder, 'metrics.csv')
        assert (os.path.exists(metric_path))
        df = pd.read_csv(metric_path, index_col=0, header=0)

        data_description = self._DataDescription(df)
        method_description = self._MethodDescription(pipeline_name, result_root)
        statistic_description = "The performance of the model was evaluated using receiver operating " \
                                "characteristic (ROC) curve analysis. The area under the ROC curve (AUC) was " \
                                "calculated for quantification. The accuracy, sensitivity, specificity, " \
                                "positive predictive value (PPV), and negative predictive value (NPV) were also " \
                                "calculated at a cutoff value that maximized the value of the Yorden index. We " \
                                "also estimated the 95% confidence interval by bootstrape with 1000 samples. " \
                                "All above processes were implemented with FeAture Explorer Pro " \
                                "(FAE, V {}) on Python (3.7.6).\n".format(VERSION)
        result_description = self._MetricDescription(os.path.split(fs_folder)[1].split('_')[1], df)

        table_stype = (
            ('FONT', (0, 0), (-1, -1), '%s' % 'Helvetica', 9),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
            ('LINEABOVE', (0, 1), (-1, 1), 1, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        )

        statistic_table_header, statistic_table_content = self._StatisticTable(df)
        feature_table_header, feature_table_content = self._FeatureTable(fs_folder, cls_folder)

        figure_title = "Figure 1. The ROC curve. "

        # Build PDF
        pdf = MyPdfDocument(os.path.join(store_folder, 'description.pdf'))
        pdf.init_report()
        pdf.h1("Materials and Methods")
        pdf.p(data_description)
        pdf.p(method_description)
        pdf.p(statistic_description)

        pdf.h1("Result")
        pdf.p(result_description)
        pdf.table_header(statistic_table_header)
        pdf.table(statistic_table_content, 130, style=table_stype)
        pdf.table_header(feature_table_header)
        pdf.table(feature_table_content, 200, style=table_stype)
        pdf.p("\n\n")
        pdf.image(os.path.join(store_folder, 'ROC.jpg'))
        pdf.table_header(figure_title)

        pdf.end_connect(
            "Thanks for using FAE {}. If you need a specific description, please connect to "
            "Yang Song (songyangmri@gmail.com) or Guang Yang (gyang@phy.ecnu.edu.cn). "
            "Welcome any co-operation and discussion.".format(VERSION))
        pdf.generate()

def GenerateDescription():
    training_data_container = DataContainer()
    training_data_container.Load(r'..\..\Example\numeric_feature.csv')

    one_pipeline = OnePipeline()
    one_pipeline.LoadPipeline(r'C:\MyCode\FAEGitHub\FAE\Example\report_temp\NormUnit_Cos_ANOVA_5_SVM\pipeline_info.csv')

    description = Description()
    description.Run(training_data_container, one_pipeline, r'..\..\Example\report_temp', r'..\..\Example\report')


if __name__ == '__main__':
    GenerateDescription()
