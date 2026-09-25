import logging
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis import Classifier as classifier_module
from BC.FeatureAnalysis.Classifier import AdaBoost, GaussianProcess
from BC.HyperParamManager.HyperParamManager import GetClassifierHyperParams


class ClassifierHyperparameterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = DataContainer(
            array=np.array([
                [0.0, 0.0],
                [0.0, 0.2],
                [0.2, 0.0],
                [0.2, 0.2],
                [0.8, 0.8],
                [0.8, 1.0],
                [1.0, 0.8],
                [1.0, 1.0],
            ]),
            label=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            feature_name=['feature_1', 'feature_2'],
            case_name=['case_{}'.format(index) for index in range(8)],
        )
        cls.config_folder = (
            Path(__file__).resolve().parents[1]
            / 'BC'
            / 'HyperParameters'
            / 'Classifier'
        )
        cls.output_folder = (
            Path(__file__).resolve().parent
            / '.classifier_hyperparameter_output'
        )

    def setUp(self):
        shutil.rmtree(self.output_folder, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.output_folder, ignore_errors=True)

    def test_has_effective_parameter_grid_recognizes_supported_shapes(self):
        helper = getattr(classifier_module, 'HasEffectiveParameterGrid', None)

        self.assertTrue(callable(helper))
        self.assertFalse(helper({}))
        self.assertTrue(helper({'C': []}))
        self.assertFalse(helper([]))
        self.assertFalse(helper([{}]))
        self.assertTrue(helper([{}, {'C': []}]))
        self.assertFalse(helper(None))
        self.assertFalse(helper(({'C': [1]},)))

    def test_gaussian_process_fit_skips_grid_search_for_empty_grid_list(self):
        classifier = GaussianProcess()
        classifier.SetDataContainer(self.data)

        with patch.object(classifier_module, 'GridSearchCV') as grid_search:
            classifier.Fit([{}], cv_part=2)

        grid_search.assert_not_called()
        self.assertTrue(hasattr(classifier.GetModel(), 'classes_'))

    def test_gaussian_process_hyperfit_skips_grid_search_for_empty_grid_list(self):
        classifier = GaussianProcess()
        classifier.SetDataContainer(self.data)

        with patch.object(classifier_module, 'GridSearchCV') as grid_search:
            classifier.HyperFit([{}], cv_parts=2)

        grid_search.assert_not_called()
        self.assertTrue(hasattr(classifier.GetModel(), 'classes_'))

    def test_lda_svd_config_explicitly_disables_shrinkage(self):
        settings = GetClassifierHyperParams(self.config_folder)['LDA']
        svd_setting = next(
            setting for setting in settings if setting.get('solver') == ['svd']
        )

        self.assertEqual(
            svd_setting,
            {'solver': ['svd'], 'shrinkage': [None]},
        )

    def test_naive_bayes_config_exposes_var_smoothing_grid(self):
        settings = GetClassifierHyperParams(self.config_folder)

        self.assertIn('NB', settings)
        self.assertEqual(
            settings['NB'],
            [{
                'var_smoothing': [
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                ],
            }],
        )

    def test_adaboost_save_skips_unsupported_shap_without_warning(self):
        classifier = AdaBoost(n_estimators=2, random_state=0)
        classifier.SetDataContainer(self.data)
        classifier.Fit()
        self.output_folder.mkdir()

        with self.assertNoLogs(classifier.logger, logging.WARNING):
            classifier.Save(str(self.output_folder))

        self.assertTrue((self.output_folder / 'model.pickle').is_file())
        self.assertFalse((self.output_folder / 'AB_shap.csv').exists())


if __name__ == '__main__':
    unittest.main()
