import time
import unittest

import numpy as np

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC


class PccDimensionReductionTest(unittest.TestCase):
    def test_pcc_preserves_greedy_feature_selection_semantics(self):
        label = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        data = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
            [3, 1, 0],
            [3, 1, 1],
        ], dtype=float)
        container = DataContainer(
            data, label,
            ['correlated_weaker', 'correlated_stronger', 'independent'],
            ['case_{}'.format(index) for index in range(len(label))],
        )

        result = DimensionReductionByPCC(threshold=0.85).Run(container)

        self.assertEqual(
            result.GetFeatureName(),
            ['correlated_stronger', 'independent'],
        )

    def test_pcc_handles_266_features_within_one_second(self):
        rng = np.random.RandomState(42)
        data = rng.normal(size=(200, 266))
        label = np.array([0, 1] * 100, dtype=np.int8)
        container = DataContainer(
            data, label,
            ['feature_{}'.format(index) for index in range(data.shape[1])],
            ['case_{}'.format(index) for index in range(data.shape[0])],
        )

        started = time.perf_counter()
        result = DimensionReductionByPCC(threshold=0.99).Run(container)
        elapsed = time.perf_counter() - started

        self.assertEqual(result.GetArray().shape, data.shape)
        self.assertLess(
            elapsed, 1.0,
            'PCC took {:.3f} seconds for 266 features'.format(elapsed),
        )

    def test_pcc_matches_legacy_tie_break_for_scaled_duplicates(self):
        base_feature = np.array([
            -2.0644148031211755,
            -0.6621593396668087,
            -1.2042198455997326,
            1.461975627213524,
            1.7661608779293339,
            -0.3294137519130651,
            0.8407332421435357,
            -0.17998640125235033,
        ])
        label = np.array([1, 1, 1, 1, 0, 1, 0, 0], dtype=np.int8)
        data = np.column_stack((base_feature, 10 * base_feature))
        container = DataContainer(
            data, label, ['original_scale', 'scaled'],
            ['case_{}'.format(index) for index in range(len(label))],
        )

        result = DimensionReductionByPCC(threshold=0.99).Run(container)

        self.assertEqual(result.GetFeatureName(), ['scaled'])

    def test_pcc_matches_legacy_threshold_for_near_constant_features(self):
        rng = np.random.RandomState(142)
        first = rng.normal(size=425)
        noise = rng.normal(size=425)
        correlation = 0.985
        offset = 1e14
        data = np.column_stack((
            offset + first,
            offset + correlation * first
            + np.sqrt(1 - correlation ** 2) * noise,
        ))
        label = np.array([0, 1] * 212 + [0], dtype=np.int8)
        container = DataContainer(
            data, label, ['near_constant_a', 'near_constant_b'],
            ['case_{}'.format(index) for index in range(len(label))],
        )

        result = DimensionReductionByPCC(threshold=0.99).Run(container)

        self.assertEqual(
            result.GetFeatureName(),
            ['near_constant_a', 'near_constant_b'],
        )

    def test_pcc_accepts_an_empty_feature_matrix(self):
        label = np.array([0, 1], dtype=np.int8)
        container = DataContainer(
            np.empty((2, 0)), label, [], ['case_0', 'case_1'],
        )

        result = DimensionReductionByPCC(threshold=0.99).Run(container)

        self.assertEqual(result.GetArray().shape, (2, 0))
        self.assertEqual(result.GetFeatureName(), [])


if __name__ == '__main__':
    unittest.main()
