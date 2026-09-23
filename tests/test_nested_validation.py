import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.Classifier import SVM
from BC.FeatureAnalysis.NestedValidation import NestedPipelineEvaluator
from BC.FeatureAnalysis.Normalizer import NormalizerNone


def make_binary_data(case_count=18):
    labels = np.tile(np.array([0, 1]), case_count // 2)
    values = np.column_stack((
        labels.astype(float),
        labels.astype(float) * 2 + np.arange(case_count) / 1000,
    ))
    return DataContainer(
        array=values,
        label=labels,
        feature_name=['feature_1', 'feature_2'],
        case_name=['case_{}'.format(index) for index in range(case_count)],
    )


class RecordingBalance:
    fit_case_counts = []
    fit_case_names = []

    def Run(self, container):
        type(self).fit_case_counts.append(len(container.GetCaseName()))
        type(self).fit_case_names.append(container.GetCaseName())
        return container


class DeterministicClassifier:
    def __init__(self):
        self.choice = 'bad'

    def SetDataContainer(self, data_container):
        self.data_container = data_container

    def SetModelParameter(self, candidate):
        self.choice = candidate['choice']

    def Fit(self):
        pass

    def GetModel(self):
        return self

    def predict(self, array):
        expected = (array[:, 0] >= 0.5).astype(int)
        if self.choice == 'good':
            return expected
        return 1 - expected

    def Predict(self, array):
        return np.where(self.predict(array) == 1, 0.9, 0.1)

    def GetName(self):
        return 'DeterministicClassifier'


class CountingClassifier(DeterministicClassifier):
    fitted_instances = []

    def Fit(self):
        type(self).fitted_instances.append(self)

    def GetName(self):
        return 'CountingClassifier'


class FailingClassifier(DeterministicClassifier):
    def Fit(self):
        raise ValueError('candidate cannot be fitted')

    def GetName(self):
        return 'FailingClassifier'


class EqualAccuracyClassifier(DeterministicClassifier):
    def predict(self, array):
        return (array[:, 0] >= 0.5).astype(int)


class IdentifierProbabilityClassifier:
    def SetDataContainer(self, data_container):
        self.data_container = data_container

    def Fit(self):
        pass

    def Predict(self, array):
        return array[:, 0]

    def GetName(self):
        return 'IdentifierProbabilityClassifier'


class StrictNormalizer:
    run_count = 0
    transform_count = 0

    def Run(self, container):
        type(self).run_count += 1
        return container

    def Transform(self, container):
        type(self).transform_count += 1
        return container


class StrictReducer(StrictNormalizer):
    run_count = 0
    transform_count = 0


class StrictSelector(StrictNormalizer):
    run_count = 0
    transform_count = 0
    selected_numbers = []

    def SetSelectedFeatureNumber(self, feature_number):
        type(self).selected_numbers.append(feature_number)


class MutatingBalance:
    run_count = 0

    def Run(self, container):
        type(self).run_count += 1
        container.SetArray(np.full(container.GetArray().shape, -100.0))
        container.SetCaseName(['mutated'] * len(container.GetCaseName()))
        container.UpdateFrameByData()
        return container


class DeepcopyRecordingProcessor:
    copies = {}

    def __init__(self, role):
        self.role = role
        self.fitted = False
        self.transform_count = 0
        self.selected_feature_number = None

    def __deepcopy__(self, memo):
        copied = type(self)(self.role)
        type(self).copies.setdefault(self.role, []).append(copied)
        return copied

    def Run(self, container):
        self.fitted = True
        return container

    def Transform(self, container):
        self.transform_count += 1
        return container

    def SetSelectedFeatureNumber(self, feature_number):
        self.selected_feature_number = feature_number


class NestedPipelineEvaluatorTest(unittest.TestCase):
    def setUp(self):
        RecordingBalance.fit_case_counts = []
        RecordingBalance.fit_case_names = []
        CountingClassifier.fitted_instances = []
        StrictNormalizer.run_count = 0
        StrictNormalizer.transform_count = 0
        StrictReducer.run_count = 0
        StrictReducer.transform_count = 0
        StrictSelector.run_count = 0
        StrictSelector.transform_count = 0
        StrictSelector.selected_numbers = []
        MutatingBalance.run_count = 0
        DeepcopyRecordingProcessor.copies = {}

    def make_evaluator(self, classifier=None, param_grid=None, cv_parts=3):
        return NestedPipelineEvaluator(
            balance=RecordingBalance(),
            normalizer=NormalizerNone,
            dimension_reducer=None,
            feature_selector=None,
            feature_number=2,
            classifier=classifier or DeterministicClassifier(),
            param_grid=(
                {'choice': ['bad', 'good']}
                if param_grid is None else param_grid
            ),
            cv_parts=cv_parts,
        )

    def test_effective_grid_selects_better_candidate_deterministically(self):
        evaluator = self.make_evaluator()

        selected = evaluator.select_parameters(make_binary_data())

        self.assertEqual(selected, {'choice': 'good'})

    def test_tied_mean_accuracy_selects_first_parameter_grid_candidate(self):
        evaluator = self.make_evaluator(
            classifier=EqualAccuracyClassifier(),
            param_grid={'choice': ['first', 'second']},
        )

        selected = evaluator.select_parameters(make_binary_data())

        self.assertEqual(selected, {'choice': 'first'})

    def test_empty_grid_returns_default_parameters_without_inner_search(self):
        evaluator = self.make_evaluator(param_grid=[{}])

        with patch.object(evaluator, '_score_candidate') as score_candidate:
            selected = evaluator.select_parameters(make_binary_data())

        self.assertEqual(selected, {})
        score_candidate.assert_not_called()

    def test_empty_grid_evaluation_does_not_require_inner_cv(self):
        evaluator = self.make_evaluator(param_grid=[{}], cv_parts=2)

        result = evaluator.evaluate(make_binary_data(case_count=4))

        self.assertEqual(result.prediction.shape, (4,))

    def test_outer_evaluation_preserves_order_and_returns_probabilities(self):
        data = make_binary_data()
        evaluator = NestedPipelineEvaluator(
            balance=RecordingBalance(),
            normalizer=NormalizerNone,
            dimension_reducer=None,
            feature_selector=None,
            feature_number=2,
            classifier=SVM(kernel='linear', random_state=0),
            param_grid=[{}],
            cv_parts=3,
        )

        result = evaluator.evaluate(data)

        np.testing.assert_array_equal(result.label, data.GetLabel())
        self.assertEqual(result.case_names, data.GetCaseName())
        self.assertEqual(result.prediction.shape, (18,))
        self.assertTrue(np.all(result.prediction >= 0))
        self.assertTrue(np.all(result.prediction <= 1))
        self.assertTrue(np.any(
            (result.prediction != 0) & (result.prediction != 1)
        ))

    def test_outer_probabilities_follow_original_case_order(self):
        identifiers = np.array([
            0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
            0.65, 0.75, 0.85, 0.95, 0.12, 0.22,
        ])
        labels = np.array([0] * 6 + [1] * 6)
        data = DataContainer(
            array=np.column_stack((identifiers, labels)),
            label=labels,
            feature_name=['identifier_probability', 'label_signal'],
            case_name=[
                'case_{}'.format(index) for index in range(len(labels))
            ],
        )
        evaluator = NestedPipelineEvaluator(
            balance=RecordingBalance(),
            normalizer=NormalizerNone,
            dimension_reducer=None,
            feature_selector=None,
            feature_number=2,
            classifier=IdentifierProbabilityClassifier(),
            param_grid=[{}],
            cv_parts=3,
        )

        result = evaluator.evaluate(data)

        np.testing.assert_array_equal(result.prediction, identifiers)

    def test_inner_folds_reduce_to_outer_training_smallest_class(self):
        requested_splits = 4
        split_calls = []

        class RecordingStratifiedKFold:
            def __init__(self, *args, **kwargs):
                self.n_splits = kwargs.get(
                    'n_splits',
                    args[0] if args else 5,
                )
                self.delegate = SklearnStratifiedKFold(*args, **kwargs)

            def split(self, array, labels):
                split_calls.append((
                    self.n_splits,
                    tuple(array[:, 1]),
                    np.array(labels, copy=True),
                ))
                return self.delegate.split(array, labels)

        evaluator = self.make_evaluator(cv_parts=requested_splits)
        data = make_binary_data(case_count=8)
        expected_inner_sets = {
            tuple(data.GetArray()[train_indices, 1])
            for train_indices, _ in SklearnStratifiedKFold(
                n_splits=requested_splits,
                shuffle=False,
            ).split(data.GetArray(), data.GetLabel())
        }

        with patch(
            'BC.FeatureAnalysis.NestedValidation.StratifiedKFold',
            RecordingStratifiedKFold,
        ):
            evaluator.evaluate(data)

        outer_calls = [call for call in split_calls if len(call[1]) == 8]
        inner_calls = [call for call in split_calls if len(call[1]) == 6]
        self.assertEqual(
            [n_splits for n_splits, _, _ in outer_calls],
            [requested_splits],
        )
        self.assertEqual(
            {identifiers for _, identifiers, _ in inner_calls},
            expected_inner_sets,
        )
        for n_splits, _, labels in inner_calls:
            _, counts = np.unique(labels, return_counts=True)
            self.assertEqual(n_splits, counts.min())
            self.assertEqual(n_splits, 3)

    def test_outer_validation_is_never_fitted_or_balanced(self):
        data = make_binary_data()
        evaluator = self.make_evaluator(param_grid=[{}])

        evaluator.evaluate(data)

        self.assertEqual(RecordingBalance.fit_case_counts, [12, 12, 12])
        self.assertNotIn(18, RecordingBalance.fit_case_counts)
        for fitted_names in RecordingBalance.fit_case_names:
            self.assertNotEqual(fitted_names, data.GetCaseName())

    def test_each_candidate_fold_uses_a_fresh_classifier(self):
        evaluator = self.make_evaluator(classifier=CountingClassifier())

        evaluator.select_parameters(make_binary_data())

        fitted = CountingClassifier.fitted_instances
        self.assertEqual(len(fitted), 6)
        self.assertEqual(len({id(classifier) for classifier in fitted}), 6)
        self.assertNotIn(evaluator.classifier, fitted)

    def test_each_outer_fold_fits_a_distinct_final_classifier(self):
        evaluator = self.make_evaluator(
            classifier=CountingClassifier(),
            param_grid=[{}],
        )

        evaluator.evaluate(make_binary_data())

        fitted = CountingClassifier.fitted_instances
        self.assertEqual(len(fitted), 3)
        self.assertEqual(len({id(classifier) for classifier in fitted}), 3)
        self.assertNotIn(evaluator.classifier, fitted)

    def test_processing_fits_distinct_copies_without_mutating_originals(self):
        originals = {
            role: DeepcopyRecordingProcessor(role)
            for role in ('balance', 'normalizer', 'reducer', 'selector')
        }
        evaluator = NestedPipelineEvaluator(
            balance=originals['balance'],
            normalizer=originals['normalizer'],
            dimension_reducer=originals['reducer'],
            feature_selector=originals['selector'],
            feature_number=1,
            classifier=DeterministicClassifier(),
            param_grid=[{}],
            cv_parts=3,
        )

        evaluator.evaluate(make_binary_data())

        original_ids = {id(processor) for processor in originals.values()}
        fitted_copies = [
            processor
            for copies in DeepcopyRecordingProcessor.copies.values()
            for processor in copies
        ]
        self.assertEqual(len(fitted_copies), 12)
        self.assertEqual(len({id(processor) for processor in fitted_copies}), 12)
        self.assertTrue(
            original_ids.isdisjoint(id(processor) for processor in fitted_copies)
        )
        for role, original in originals.items():
            role_copies = DeepcopyRecordingProcessor.copies[role]
            self.assertEqual(len(role_copies), 3)
            self.assertEqual(len({id(processor) for processor in role_copies}), 3)
            self.assertNotIn(id(original), {
                id(processor) for processor in role_copies
            })
            self.assertTrue(all(processor.fitted for processor in role_copies))
        for processor in originals.values():
            self.assertFalse(processor.fitted)
            self.assertEqual(processor.transform_count, 0)
            self.assertIsNone(processor.selected_feature_number)

    def test_processing_omits_storage_arguments_and_cannot_mutate_source(self):
        data = make_binary_data()
        original_array = data.GetArray()
        original_labels = data.GetLabel()
        original_features = data.GetFeatureName()
        original_cases = data.GetCaseName()
        evaluator = NestedPipelineEvaluator(
            balance=MutatingBalance(),
            normalizer=StrictNormalizer(),
            dimension_reducer=StrictReducer(),
            feature_selector=StrictSelector(),
            feature_number=1,
            classifier=DeterministicClassifier(),
            param_grid=[{}],
            cv_parts=3,
        )

        evaluator.evaluate(data)

        np.testing.assert_array_equal(data.GetArray(), original_array)
        np.testing.assert_array_equal(data.GetLabel(), original_labels)
        self.assertEqual(data.GetFeatureName(), original_features)
        self.assertEqual(data.GetCaseName(), original_cases)
        self.assertEqual(MutatingBalance.run_count, 3)
        self.assertEqual(StrictNormalizer.run_count, 3)
        self.assertEqual(StrictNormalizer.transform_count, 3)
        self.assertEqual(StrictReducer.run_count, 3)
        self.assertEqual(StrictReducer.transform_count, 3)
        self.assertEqual(StrictSelector.run_count, 3)
        self.assertEqual(StrictSelector.transform_count, 3)
        self.assertEqual(StrictSelector.selected_numbers, [1, 1, 1])

    def test_excessive_outer_cv_parts_are_rejected(self):
        evaluator = self.make_evaluator(param_grid=[{}], cv_parts=4)

        with self.assertRaisesRegex(ValueError, 'smallest class has 3'):
            evaluator.evaluate(make_binary_data(case_count=6))

    def test_candidate_failure_includes_classifier_and_candidate_context(self):
        evaluator = self.make_evaluator(
            classifier=FailingClassifier(),
            param_grid={'choice': ['broken']},
        )

        with self.assertRaises(RuntimeError) as raised:
            evaluator.select_parameters(make_binary_data())

        message = str(raised.exception)
        self.assertIn('FailingClassifier', message)
        self.assertIn("{'choice': 'broken'}", message)
        self.assertIsInstance(raised.exception.__cause__, ValueError)


if __name__ == '__main__':
    unittest.main()
