import unittest

import numpy as np

from BC.FeatureAnalysis.CrossValidation import GetMaximumCvParts, ValidateCvParts


class CrossValidationLimitTest(unittest.TestCase):
    def test_maximum_fold_count_is_smallest_class(self):
        labels = np.array([0] * 12 + [1] * 5)

        maximum = GetMaximumCvParts(labels)

        self.assertEqual(maximum, 5)
        self.assertIsInstance(maximum, int)

    def test_validate_rejects_too_many_folds(self):
        labels = np.array([0] * 12 + [1] * 5)

        with self.assertRaisesRegex(ValueError, 'smallest class has 5'):
            ValidateCvParts(labels, 6)

    def test_maximum_requires_two_represented_classes(self):
        with self.assertRaisesRegex(ValueError, 'at least 2 cases in each class'):
            GetMaximumCvParts(np.array([0, 0]))

    def test_validate_requires_two_cases_per_class(self):
        with self.assertRaisesRegex(ValueError, 'at least 2 cases in each class'):
            ValidateCvParts(np.array([0, 0, 1]), 2)

    def test_validate_returns_valid_fold_count_unchanged(self):
        labels = np.array([0] * 12 + [1] * 5)

        self.assertEqual(ValidateCvParts(labels, 5), 5)


if __name__ == '__main__':
    unittest.main()
