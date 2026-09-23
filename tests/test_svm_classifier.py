import logging
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.Classifier import SVM


class SvmClassifierTest(unittest.TestCase):
    def setUp(self):
        self.data = DataContainer(
            array=np.array([
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.2, 0.8],
                [0.8, 0.2],
            ]),
            label=np.array([0, 1, 1, 0, 1, 1]),
            feature_name=['feature_1', 'feature_2'],
            case_name=['case_1', 'case_2', 'case_3', 'case_4', 'case_5', 'case_6'],
        )

    def test_non_linear_svm_save_skips_unsupported_linear_outputs(self):
        classifier = SVM(kernel='rbf')
        classifier.SetDataContainer(self.data)
        classifier.Fit()

        with tempfile.TemporaryDirectory() as store_folder:
            with self.assertNoLogs(classifier.logger, logging.WARNING):
                classifier.Save(store_folder)

            self.assertTrue(os.path.exists(os.path.join(store_folder, 'model.pickle')))
            self.assertFalse(os.path.exists(os.path.join(store_folder, 'SVM_coef.csv')))
            self.assertFalse(os.path.exists(os.path.join(store_folder, 'SVM_shap.csv')))

    def test_non_linear_svm_save_removes_stale_linear_outputs(self):
        classifier = SVM(kernel='poly')
        classifier.SetDataContainer(self.data)
        classifier.Fit()

        with tempfile.TemporaryDirectory() as store_folder:
            for filename in ('SVM_coef.csv', 'SVM_shap.csv'):
                with open(os.path.join(store_folder, filename), 'w') as output:
                    output.write('stale')

            classifier.Save(store_folder)

            self.assertFalse(os.path.exists(os.path.join(store_folder, 'SVM_coef.csv')))
            self.assertFalse(os.path.exists(os.path.join(store_folder, 'SVM_shap.csv')))

    def test_non_linear_svm_save_continues_when_stale_output_cleanup_fails(self):
        classifier = SVM(kernel='rbf')
        classifier.SetDataContainer(self.data)
        classifier.Fit()

        with tempfile.TemporaryDirectory() as store_folder:
            stale_path = os.path.join(store_folder, 'SVM_coef.csv')
            with open(stale_path, 'w') as output:
                output.write('stale')

            with patch('BC.FeatureAnalysis.Classifier.os.remove',
                       side_effect=OSError('cleanup denied')):
                with self.assertLogs(classifier.logger, logging.WARNING) as logs:
                    classifier.Save(store_folder)

            self.assertTrue(os.path.exists(os.path.join(store_folder, 'model.pickle')))
            self.assertIn(stale_path, '\n'.join(logs.output))
            self.assertIn('cleanup denied', '\n'.join(logs.output))

    def test_linear_svm_is_saved_when_coefficient_export_fails(self):
        classifier = SVM(kernel='linear')
        classifier.SetData(self.data.GetArray(), self.data.GetLabel())
        classifier.Fit()

        with tempfile.TemporaryDirectory() as store_folder:
            with redirect_stdout(StringIO()):
                with self.assertLogs(classifier.logger, logging.ERROR):
                    classifier.Save(store_folder)

            self.assertTrue(os.path.exists(os.path.join(store_folder, 'model.pickle')))

    @patch('BC.FeatureAnalysis.Classifier.GridSearchCV')
    def test_hyperfit_avoids_loky_process_pool(self, grid_search_class):
        grid_search = grid_search_class.return_value
        grid_search.best_estimator_ = SVM().GetModel()
        classifier = SVM()
        classifier.SetDataContainer(self.data)

        classifier.HyperFit({'C': [0.1, 1.0]}, cv_parts=2)

        self.assertEqual(grid_search_class.call_args.kwargs['n_jobs'], 1)


if __name__ == '__main__':
    unittest.main()
