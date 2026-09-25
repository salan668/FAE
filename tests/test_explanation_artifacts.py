import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from BC.FeatureAnalysis.ExplanationArtifacts import (
    load_verified_explanation,
    write_available_explanation,
    write_unavailable_explanation,
)


class ExplanationArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)
        self.params = {'C': 1.0, 'kernel': 'linear'}
        self.shap_df = pd.DataFrame(
            [[0.1, -0.2], [0.3, 0.4]],
            index=['case_1', 'case_2'],
            columns=['feature_a', 'feature_b'],
        )
        self.feature_df = pd.DataFrame(
            [[3.0, 4.0], [5.0, 6.0]],
            index=['case_1', 'case_2'],
            columns=['feature_a', 'feature_b'],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_model_params(self, params=None):
        with open(self.folder / 'model_param.json', 'w', encoding='utf-8') as handle:
            json.dump(self.params if params is None else params, handle)

    def test_available_artifact_round_trips_only_when_model_params_match(self):
        """Changing saved model parameters must make the SHAP artifact unreadable."""
        metadata = write_available_explanation(
            self.folder, 'SVM', self.params, self.shap_df, self.feature_df,
            'real_train',
        )
        self._write_model_params()

        result = load_verified_explanation(self.folder, 'SVM')

        self.assertEqual(metadata['status'], 'available')
        self.assertIsNotNone(result)
        shap_df, feature_df, loaded_metadata = result
        pd.testing.assert_frame_equal(shap_df, self.shap_df)
        pd.testing.assert_frame_equal(feature_df, self.feature_df)
        self.assertEqual(loaded_metadata['explained_data_kind'], 'real_train')

        self._write_model_params({'C': 3.0, 'kernel': 'linear'})
        self.assertIsNone(load_verified_explanation(self.folder, 'SVM'))

    def test_unavailable_artifact_removes_old_shap_feature_and_coefficient_files(self):
        """Unsupported models must not leave an older explanation displayable."""
        for name in ('SVM_shap.csv', 'SVM_shap_features.csv', 'SVM_coef.csv'):
            (self.folder / name).write_text('stale', encoding='utf-8')

        metadata = write_unavailable_explanation(
            self.folder, 'SVM', {'kernel': 'rbf'}, 'unsupported',
            'non-linear kernel',
        )

        self.assertEqual(metadata['status'], 'unsupported')
        self.assertFalse((self.folder / 'SVM_shap.csv').exists())
        self.assertFalse((self.folder / 'SVM_shap_features.csv').exists())
        self.assertFalse((self.folder / 'SVM_coef.csv').exists())
        self.assertIsNone(load_verified_explanation(self.folder, 'SVM'))

    def test_mismatched_shap_and_feature_rows_are_rejected_before_writing(self):
        """A displayable explanation must have one feature row per SHAP row."""
        mismatched_features = self.feature_df.drop(index='case_2')

        with self.assertRaisesRegex(ValueError, 'same index'):
            write_available_explanation(
                self.folder, 'SVM', self.params, self.shap_df,
                mismatched_features, 'real_train',
            )


if __name__ == '__main__':
    unittest.main()
