import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.Classifier import AdaBoost, LR, SVM
from BC.FeatureAnalysis.ExplanationArtifacts import load_verified_explanation


class ClassifierExplanationTest(unittest.TestCase):
    def setUp(self):
        self.real_data = DataContainer(
            array=np.array([[0.0, 0.1], [0.1, 0.2], [0.8, 0.9], [0.9, 0.8]]),
            label=np.array([0, 0, 1, 1]),
            feature_name=['feature_a', 'feature_b'],
            case_name=['case_1', 'case_2', 'case_3', 'case_4'],
        )
        self.balanced_data = DataContainer(
            array=np.array([
                [0.0, 0.1], [0.1, 0.2], [0.8, 0.9], [0.9, 0.8],
                [0.05, 0.15], [0.85, 0.85],
            ]),
            label=np.array([0, 0, 1, 1, 0, 1]),
            feature_name=['feature_a', 'feature_b'],
            case_name=['case_1', 'case_2', 'case_3', 'case_4', 'Generate0', 'Generate1'],
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _status(self):
        with open(self.folder / 'explanation.json', 'r', encoding='utf-8') as handle:
            return json.load(handle)

    def test_rbf_svm_invalidates_old_linear_artifacts(self):
        """A non-linear SVM must not retain a linear model's explanation."""
        for name in ('SVM_shap.csv', 'SVM_shap_features.csv', 'SVM_coef.csv'):
            (self.folder / name).write_text('stale', encoding='utf-8')
        classifier = SVM(kernel='rbf')
        classifier.SetDataContainer(self.balanced_data)
        classifier.Fit()

        classifier.Save(str(self.folder), explanation_container=self.real_data)

        self.assertEqual(self._status()['status'], 'unsupported')
        self.assertFalse((self.folder / 'SVM_shap.csv').exists())
        self.assertFalse((self.folder / 'SVM_shap_features.csv').exists())
        self.assertFalse((self.folder / 'SVM_coef.csv').exists())

    def test_adaboost_invalidates_old_shap_artifact(self):
        """Unsupported AdaBoost SHAP must not expose a prior CSV."""
        (self.folder / 'AB_shap.csv').write_text('stale', encoding='utf-8')
        classifier = AdaBoost(n_estimators=2, random_state=0)
        classifier.SetDataContainer(self.balanced_data)
        classifier.Fit()

        classifier.Save(str(self.folder), explanation_container=self.real_data)

        self.assertEqual(self._status()['status'], 'unsupported')
        self.assertFalse((self.folder / 'AB_shap.csv').exists())

    def test_linear_model_explains_real_cases_after_balanced_fit(self):
        """SHAP rows must represent real cases, never generated balancing rows."""
        classifier = LR()
        classifier.SetDataContainer(self.balanced_data)
        classifier.Fit()

        classifier.Save(str(self.folder), explanation_container=self.real_data)

        result = load_verified_explanation(self.folder, 'LR')
        self.assertIsNotNone(result)
        shap_df, feature_df, metadata = result
        self.assertEqual(list(shap_df.index), self.real_data.GetCaseName())
        self.assertEqual(list(feature_df.index), self.real_data.GetCaseName())
        self.assertNotIn('Generate0', shap_df.index)
        self.assertEqual(metadata['explained_data_kind'], 'real_train')


if __name__ == '__main__':
    unittest.main()
