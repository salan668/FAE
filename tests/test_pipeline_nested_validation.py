import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.Classifier import NaiveBayes
from BC.FeatureAnalysis.CrossValidation import ArbitratyCrossValidation
from BC.FeatureAnalysis.DataBalance import NoneBalance
from BC.FeatureAnalysis.NestedValidation import NestedValidationResult
from BC.FeatureAnalysis.Normalizer import (
    NoneNormalizeFunc,
    Normalizer,
    NormalizerNone,
)
from BC.FeatureAnalysis.Pipelines import PipelinesManager


WORKTREE = Path(__file__).resolve().parents[1]


def make_data(case_prefix, case_count=12):
    labels = np.tile(np.array([0, 1]), case_count // 2)
    values = np.column_stack((
        labels + np.arange(case_count) / 100,
        1 - labels + np.arange(case_count) / 200,
    ))
    return DataContainer(
        array=values,
        label=labels,
        feature_name=['feature_1', 'feature_2'],
        case_name=[
            '{}_{}'.format(case_prefix, index)
            for index in range(case_count)
        ],
    )


class FalsyNoOpProcessor:
    def __init__(self, name):
        self.name = name
        self.selected_feature_number = None

    def __bool__(self):
        return False

    def GetName(self):
        return self.name

    def SetSelectedFeatureNumber(self, feature_number):
        self.selected_feature_number = feature_number

    def Run(self, container, *args, **kwargs):
        return container

    def Transform(self, container, *args, **kwargs):
        return container


class RecordingNaiveBayes(NaiveBayes):
    fit_instances = []
    fit_arguments = []
    prediction_instances = []
    saved_instances = []
    seeded_instances = {}
    selected_parameters = {}

    def Fit(self, *args, **kwargs):
        type(self).fit_instances.append(self)
        type(self).fit_arguments.append((self, args, kwargs))
        return super().Fit(*args, **kwargs)

    def SetSeed(self, seed):
        type(self).seeded_instances[id(self)] = seed
        return super().SetSeed(seed)

    def Predict(self, array, is_probability=True):
        type(self).prediction_instances.append(self)
        return super().Predict(array, is_probability)

    def Save(self, store_path):
        type(self).saved_instances.append((self, Path(store_path)))
        return super().Save(store_path)

    def SetModelParameter(self, param):
        type(self).selected_parameters[id(self)] = dict(param)
        return super().SetModelParameter(param)


class RecordingNaiveBayesNoGrid(RecordingNaiveBayes):
    def GetName(self):
        return 'NB_NO_GRID'


class RecordingEvaluator:
    constructions = []
    evaluations = []
    selections = []

    def __init__(self, balance, normalizer, dimension_reducer,
                 feature_selector, feature_number, classifier,
                 param_grid, cv_parts):
        self.balance = balance
        self.normalizer = normalizer
        self.dimension_reducer = dimension_reducer
        self.feature_selector = feature_selector
        self.feature_number = feature_number
        self.classifier = classifier
        self.param_grid = param_grid
        self.cv_parts = cv_parts
        type(self).constructions.append(self)

    def evaluate(self, data_container):
        type(self).evaluations.append((self, data_container))
        order = np.arange(len(data_container.GetLabel()) - 1, -1, -1)
        return NestedValidationResult(
            prediction=np.linspace(0.05, 0.95, len(order)),
            label=np.array(data_container.GetLabel()[order], copy=True),
            case_names=[
                data_container.GetCaseName()[index] for index in order
            ],
        )

    def select_parameters(self, data_container):
        type(self).selections.append((self, data_container))
        if not self.param_grid:
            return {}
        value = 0.125 if self.normalizer.GetName() == 'NormA' else 0.25
        return {'var_smoothing': value}


class PipelineNestedValidationTest(unittest.TestCase):
    def setUp(self):
        RecordingNaiveBayes.fit_instances = []
        RecordingNaiveBayes.fit_arguments = []
        RecordingNaiveBayes.prediction_instances = []
        RecordingNaiveBayes.saved_instances = []
        RecordingNaiveBayes.seeded_instances = {}
        RecordingNaiveBayes.selected_parameters = {}
        RecordingEvaluator.constructions = []
        RecordingEvaluator.evaluations = []
        RecordingEvaluator.selections = []

    def make_tempdir(self):
        return TemporaryDirectory(
            dir=str(WORKTREE / 'tests'),
            prefix='.pipeline_nested_validation_',
        )

    def test_pipeline_uses_nested_results_and_fresh_final_models(self):
        train = make_data('train')
        test = make_data('test', case_count=6)
        balance = NoneBalance()
        normalizers = [
            Normalizer('NormA', '', NoneNormalizeFunc),
            Normalizer('NormB', '', NoneNormalizeFunc),
        ]
        reducer = FalsyNoOpProcessor('NoReduction')
        selector = FalsyNoOpProcessor('NoSelection')
        classifiers = [
            RecordingNaiveBayes(),
            RecordingNaiveBayesNoGrid(),
        ]
        hyper_param = {
            'NB': {'var_smoothing': [0.125, 0.25]},
        }
        manager = PipelinesManager(
            balancer=balance,
            normalizer_list=normalizers,
            dimension_reduction_list=[reducer],
            feature_selector_list=[selector],
            feature_selector_num_list=[2],
            classifier_list=classifiers,
            cv=ArbitratyCrossValidation(2),
            hyper_param=hyper_param,
            random_seed={'seed': 17},
        )

        with self.make_tempdir() as tempdir:
            with patch(
                'BC.FeatureAnalysis.Pipelines.NestedPipelineEvaluator',
                RecordingEvaluator,
                create=True,
            ):
                progress = list(manager.Run(train, test, tempdir))

            self.assertEqual(
                progress,
                [(manager.total_num, index) for index in range(1, 5)],
            )
            self.assertEqual(len(RecordingEvaluator.constructions), 4)
            self.assertEqual(len(RecordingEvaluator.evaluations), 4)
            self.assertEqual(len(RecordingEvaluator.selections), 4)

            template_by_name = {}
            for evaluator in RecordingEvaluator.constructions:
                self.assertIs(evaluator.balance, balance)
                self.assertIn(evaluator.normalizer, normalizers)
                self.assertIs(evaluator.dimension_reducer, reducer)
                self.assertIs(evaluator.feature_selector, selector)
                self.assertEqual(evaluator.feature_number, 2)
                self.assertEqual(evaluator.cv_parts, 2)
                expected_grid = hyper_param.get(
                    evaluator.classifier.GetName(), {}
                )
                self.assertEqual(evaluator.param_grid, expected_grid)
                self.assertNotIn(evaluator.classifier, classifiers)
                self.assertFalse(
                    hasattr(evaluator.classifier.GetModel(), 'classes_')
                )
                previous = template_by_name.setdefault(
                    evaluator.classifier.GetName(), evaluator.classifier
                )
                self.assertIs(evaluator.classifier, previous)

            for _, supplied_train in RecordingEvaluator.evaluations:
                self.assertIs(supplied_train, train)
            for _, supplied_train in RecordingEvaluator.selections:
                self.assertIs(supplied_train, train)

            expected_case_names = list(reversed(train.GetCaseName()))
            expected_predictions = np.linspace(
                0.05, 0.95, len(expected_case_names)
            )
            expected_labels = np.array(list(reversed(train.GetLabel())))
            for normalizer in normalizers:
                for classifier in classifiers:
                    output = (
                        Path(tempdir)
                        / normalizer.GetName()
                        / reducer.GetName()
                        / classifier.GetName()
                    )
                    prediction = pd.read_csv(
                        output / 'CV_VAL_prediction.csv',
                        index_col=0,
                    )
                    self.assertEqual(
                        prediction.index.tolist(), expected_case_names
                    )
                    np.testing.assert_allclose(
                        prediction['Pred'].to_numpy(), expected_predictions
                    )
                    np.testing.assert_array_equal(
                        prediction['Label'].to_numpy(), expected_labels
                    )

                    with open(
                        output / 'used_hyper_param.json',
                        encoding='utf-8',
                    ) as stream:
                        used_parameters = json.load(stream)
                    if classifier.GetName() == 'NB':
                        expected_value = (
                            0.125
                            if normalizer.GetName() == 'NormA'
                            else 0.25
                        )
                        self.assertEqual(
                            used_parameters['var_smoothing'], expected_value
                        )
                    else:
                        self.assertEqual(
                            used_parameters['var_smoothing'], 1e-9
                        )

        final_models = RecordingNaiveBayes.fit_instances
        self.assertEqual(len(final_models), 4)
        self.assertEqual(len({id(model) for model in final_models}), 4)
        self.assertTrue(
            all(
                args == () and kwargs == {}
                for _, args, kwargs
                in RecordingNaiveBayes.fit_arguments
            )
        )
        self.assertTrue(
            all(model not in classifiers for model in final_models)
        )
        self.assertTrue(
            all(
                not hasattr(classifier.GetModel(), 'classes_')
                for classifier in classifiers
            )
        )
        self.assertEqual(
            RecordingNaiveBayes.seeded_instances,
            {id(model): {'seed': 17} for model in final_models},
        )
        self.assertEqual(
            {id(model) for model, _ in RecordingNaiveBayes.saved_instances},
            {id(model) for model in final_models},
        )
        self.assertEqual(len(RecordingNaiveBayes.prediction_instances), 12)
        for model in final_models:
            self.assertEqual(
                sum(
                    prediction_model is model
                    for prediction_model
                    in RecordingNaiveBayes.prediction_instances
                ),
                3,
            )

        selected_by_model = RecordingNaiveBayes.selected_parameters
        selected_values = sorted(
            parameters['var_smoothing']
            for parameters in selected_by_model.values()
        )
        self.assertEqual(selected_values, [0.125, 0.25])
        for template in template_by_name.values():
            self.assertNotIn(id(template), selected_by_model)
            self.assertFalse(hasattr(template.GetModel(), 'classes_'))

    def test_real_naive_bayes_pipeline_runs_nested_validation(self):
        train = make_data('real_train')
        test = make_data('real_test', case_count=6)
        reducer = FalsyNoOpProcessor('NoReduction')
        selector = FalsyNoOpProcessor('NoSelection')
        manager = PipelinesManager(
            balancer=NoneBalance(),
            normalizer_list=[deepcopy(NormalizerNone)],
            dimension_reduction_list=[reducer],
            feature_selector_list=[selector],
            feature_selector_num_list=[2],
            classifier_list=[NaiveBayes()],
            cv=ArbitratyCrossValidation(2),
            hyper_param={},
            random_seed={'seed': 23},
        )

        with self.make_tempdir() as tempdir:
            progress = list(manager.Run(train, test, tempdir))

            self.assertEqual(progress, [(1, 1)])
            output = (
                Path(tempdir)
                / NormalizerNone.GetName()
                / reducer.GetName()
                / 'NB'
            )
            cv_prediction = pd.read_csv(
                output / 'CV_VAL_prediction.csv',
                index_col=0,
            )
            self.assertEqual(
                cv_prediction.index.tolist(), train.GetCaseName()
            )
            self.assertEqual(len(cv_prediction), len(train.GetLabel()))
            self.assertTrue(
                np.all(cv_prediction['Pred'].to_numpy() >= 0)
            )
            self.assertTrue(
                np.all(cv_prediction['Pred'].to_numpy() <= 1)
            )
            self.assertTrue((output / 'model.pickle').is_file())
            self.assertTrue(
                (output / 'used_hyper_param.json').is_file()
            )


if __name__ == '__main__':
    unittest.main()
