from pdfdocument.document import PDFDocument
import glob
import os
import pandas as pd
import csv

from FAE.FeatureAnalysis.FeaturePipeline import OnePipeline
from FAE.DataContainer.DataContainer import DataContainer

class Description:
    def __init__(self):
        self.__paragraph_list = []
        self.__current_paragraph = ''
        pass

    # def Run(self, training_data_container, pipeline, result_folder, store_folder, testing_data_container=DataContainer()):
    def Run(self, pipeline, result_folder, store_folder):

        # Data Description
        data_description_text = "    "

        all_training_result_csv_path = os.path.join(result_folder, 'all_train_result.csv')
        if not os.path.exists(all_training_result_csv_path):
            print('There is no result here')
            return False

        all_training_result_df = pd.read_csv(all_training_result_csv_path, index_col=0, header=0)
        sample_number = all_training_result_df.iloc[0, :]['sample_number']
        positive_number = all_training_result_df.iloc[0, :]['positive_number']
        negative_number = all_training_result_df.iloc[0, :]['negative_number']
        data_description_text += "We selected {:d} cases as the training data set ({:d}/{:d} = positive/negative)). ".\
            format(int(sample_number), int(positive_number), int(negative_number))

        testing_result_csv_path = os.path.join(result_folder, 'test_result.csv')
        if not os.path.exists(testing_result_csv_path):
            data_description_text += "Since the number of the samples were limited, there was no independent testing data. "
        else:
            testing_result_df = pd.read_csv(testing_result_csv_path, index_col=0, header=0)
            sample_number = testing_result_df.iloc[0, :]['sample_number']
            positive_number = testing_result_df.iloc[0, :]['positive_number']
            negative_number = testing_result_df.iloc[0, :]['negative_number']
            data_description_text += "We also selected another {:d} cases as the independent testing data " \
                                     "set ({:d}/{:d} = positive/negative). \n" \
                                     "".format(int(sample_number), int(positive_number), int(negative_number))

        # Method Description
        method_description_text = "    "
        method_description_text += pipeline.GetNormalizer().GetDescription()
        method_description_text += pipeline.GetDimensionReduction().GetDescription()
        method_description_text += pipeline.GetFeatureSelector().GetDescription()
        method_description_text += pipeline.GetClassifier().GetDescription()
        method_description_text += pipeline.GetCrossValidatiaon().GetDescription()
        method_description_text += "\n"

        statistic_description_text = "    The performance of the model was evaluated using receiver operating characteristic " \
                                     "(ROC) curve analysis. The area under the ROC curve (AUC) was calculated for quantification. " \
                                     "The accuracy, sensitivity, specificity, positive predictive value (PPV), and negative " \
                                     "predictive value (NPV) were also calculated at a cutoff value that maximized the " \
                                     "value of the Yorden index. We also boosted estimation 1000 times and applied paired " \
                                     "t-test to give the 95% confidence interval. All above processes were implemented with " \
                                     "FeAture Explorer (FAE, v0.2.5, https://github.com/salan668/FAE) on Python (3.6.8, https://www.python.org/). \n"

        # Result Description
        result_folder = os.path.join(result_folder, pipeline.GetStoreName())
        result = pd.read_csv(os.path.join(result_folder, 'result.csv'), index_col=0)

        if os.path.exists(testing_result_csv_path):
            result_description_text = "We found that the model based on {:d} features can get the highest AUC on the " \
                                      "validation data set. The AUC and the accuracy could achieve {:.3f} and {:.3f}, respectively. In this point, " \
                                      "The AUC and the accuracy of the model achieve {:.3f} and {:.3f} on testing data set. " \
                                      "The clinical statistics in the diagonsis and the selected features were shown in Table 1 and Table 2. " \
                                      "The ROC curve was shown in Figure 1. \n" \
                                      "".format(int(pipeline.GetFeatureSelector().GetSelectedFeatureNumber()),
                                                float(result.loc['val_auc'].values),
                                                float(result.loc['val_accuracy'].values),
                                                float(result.loc['test_auc'].values),
                                                float(result.loc['test_accuracy'].values)
                                                )

        else:
            result_description_text = "We found that the model based on {:d} features can get the highest AUC on the " \
                                      "validation data set. The AUC and the accuracy could achieve {:.3f} and {:.3f}, respectively. " \
                                      "The clinical statistics in the diagonsis and the selected features were shown in Table 1 and Table 2. " \
                                      "The ROC curve was shown in Figure 1. \n" \
                                      "".format(int(pipeline.GetFeatureSelector().GetSelectedFeatureNumber()),
                                                float(result.loc['val_auc'].values), float(result.loc['val_accuracy'].values))

        from reportlab.lib import colors
        table_stype = (
            ('FONT', (0, 0), (-1, -1), '%s' % 'Helvetica', 9),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
            ('LINEABOVE', (0, 1), (-1, 1), 1, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        )
        table_1_header = "Table 1. Clinical statistics in the diagnosis. "

        if not os.path.exists(testing_result_csv_path):
            table_1 = [['Statistics', 'Value'],
                       ['Accuracy', str(result.loc['val_accuracy'].values[0])],
                       ['AUC', str(result.loc['val_auc'].values[0])],
                       ['AUC 95% CIs', str(result.loc['val_auc 95% CIs'].values[0])],
                       ['NPV', str(result.loc['val_negative predictive value'].values[0])],
                       ['PPV', str(result.loc['val_positive predictive value'].values[0])],
                       ['Sensitivity', str(result.loc['val_sensitivity'].values[0])],
                       ['Specificity', str(result.loc['val_specificity'].values[0])]]
        else:
            table_1 = [['Statistics', 'Value'],
                       ['Accuracy', str(result.loc['test_accuracy'].values[0])],
                       ['AUC', str(result.loc['test_auc'].values[0])],
                       ['AUC 95% CIs', str(result.loc['test_auc 95% CIs'].values[0])],
                       ['NPV', str(result.loc['test_negative predictive value'].values[0])],
                       ['PPV', str(result.loc['test_positive predictive value'].values[0])],
                       ['Sensitivity', str(result.loc['test_sensitivity'].values[0])],
                       ['Specificity', str(result.loc['test_specificity'].values[0])]]

        candidate_file = glob.glob(os.path.join(result_folder, '*coef.csv'))
        if len(candidate_file) > 0:
            coef = pd.read_csv(candidate_file[0], index_col=0, header=0)
            table_2_header = 'Table 2. The coefficients of features in the model. '
            table_2 = [['Features', 'Coef in model']]
            for index in coef.index:
                table_2.append([str(index), "{:.3f}".format(coef.loc[index].values[0])])

        else:
            with open(os.path.join(result_folder, 'feature_select_info.csv'), 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == 'selected_feature':
                        features = row[1:]
            table_2_header = 'Table 2. The selected of features. '
            table_2 = [['Features', 'Rank']]
            for index in range(len(features)):
                table_2.append([features[index], str(index + 1)])

        figure_title = "Figure 1. The ROC curve. "

        # Build PDF
        pdf = PDFDocument(os.path.join(store_folder, 'description.pdf'))
        pdf.init_report()
        pdf.h1("Materials and Methods")
        pdf.p(data_description_text)
        pdf.p(method_description_text)
        pdf.p(statistic_description_text)

        pdf.h1("Result")
        pdf.p(result_description_text)
        pdf.table_header(table_1_header)
        pdf.table(table_1, 130, style=table_stype)
        pdf.table_header(table_2_header)
        pdf.table(table_2, 200, style=table_stype)
        pdf.p("\n\n")
        pdf.image(os.path.join(store_folder, 'ROC.jpg'))
        pdf.table_header(figure_title)

        pdf.end_connect("Thanks for using FAE v.0.2.6. If you need a specific description, please connect to Yang Song (songyangmri@gmail.com) or Guang Yang "
              "(gyang@phy.ecnu.edu.cn). Welcome any co-operation and discussion. ")
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