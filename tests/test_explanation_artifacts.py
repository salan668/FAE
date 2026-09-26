import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_available_artifact_loads_after_csv_float_round_trip(self):
        """CSV serialization must not invalidate a valid linear-model explanation."""
        shap_df = pd.DataFrame(
            [[0.12345678901234567, -0.9876543210987654]],
            index=['case_1'], columns=['feature_a', 'feature_b'],
        )
        feature_df = pd.DataFrame(
            [[0.12345678901234567, -0.9876543210987654]],
            index=['case_1'], columns=['feature_a', 'feature_b'],
        )

        write_available_explanation(
            self.folder, 'SVM', self.params, shap_df, feature_df,
            'real_train',
        )
        self._write_model_params()

        self.assertIsNotNone(load_verified_explanation(self.folder, 'SVM'))

    def test_load_parses_the_same_shap_bytes_that_it_verifies(self):
        """A replacement after verification must not change the displayed SHAP values."""
        write_available_explanation(
            self.folder, 'SVM', self.params, self.shap_df, self.feature_df,
            'real_train',
        )
        self._write_model_params()
        value_path = self.folder / 'SVM_shap.csv'
        replacement_shap = pd.DataFrame(
            [[99.0, -99.0], [88.0, -88.0]],
            index=['case_1', 'case_2'],
            columns=['feature_a', 'feature_b'],
        )
        original_read_csv = pd.read_csv

        def replace_value_file_before_parse(path, *args, **kwargs):
            if path == value_path:
                replacement_shap.to_csv(value_path)
            return original_read_csv(path, *args, **kwargs)

        with patch(
                'BC.FeatureAnalysis.ExplanationArtifacts.pd.read_csv',
                side_effect=replace_value_file_before_parse):
            result = load_verified_explanation(self.folder, 'SVM')

        self.assertIsNotNone(result)
        pd.testing.assert_frame_equal(result[0], self.shap_df)

    def test_load_rejects_tampered_shap_csv(self):
        """Changing a SHAP value after writing must invalidate the artifact."""
        write_available_explanation(
            self.folder, 'SVM', self.params, self.shap_df, self.feature_df,
            'real_train',
        )
        self._write_model_params()
        tampered_shap = self.shap_df.copy()
        tampered_shap.iloc[0, 0] = 9.9
        tampered_shap.to_csv(self.folder / 'SVM_shap.csv')

        self.assertIsNone(load_verified_explanation(self.folder, 'SVM'))

    def test_load_rejects_tampered_feature_csv(self):
        """Changing an explained feature value must invalidate the artifact."""
        write_available_explanation(
            self.folder, 'SVM', self.params, self.shap_df, self.feature_df,
            'real_train',
        )
        self._write_model_params()
        tampered_features = self.feature_df.copy()
        tampered_features.iloc[0, 0] = 9.9
        tampered_features.to_csv(self.folder / 'SVM_shap_features.csv')

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
