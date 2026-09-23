from copy import deepcopy
from dataclasses import dataclass
import inspect

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.Classifier import HasEffectiveParameterGrid
from BC.FeatureAnalysis.CrossValidation import (
    GetMaximumCvParts,
    ValidateCvParts,
)


@dataclass
class NestedValidationResult:
    prediction: np.ndarray
    label: np.ndarray
    case_names: list


class NestedPipelineEvaluator:
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

    @staticmethod
    def _subset(array, label, feature_names, case_names, indices):
        return DataContainer(
            array=np.array(array[indices], copy=True),
            label=np.array(label[indices], copy=True),
            feature_name=deepcopy(feature_names),
            case_name=[deepcopy(case_names[index]) for index in indices],
        )

    def _fit_processing(self, train_container):
        balance = deepcopy(self.balance)
        normalizer = deepcopy(self.normalizer)
        reducer = deepcopy(self.dimension_reducer)
        selector = deepcopy(self.feature_selector)

        balanced = balance.Run(train_container)
        normalized = normalizer.Run(balanced)
        reduced = (
            reducer.Run(normalized) if reducer is not None else normalized
        )
        if selector is not None:
            selector.SetSelectedFeatureNumber(self.feature_number)
            selected = selector.Run(reduced)
        else:
            selected = reduced
        return selected, normalizer, reducer, selector

    @staticmethod
    def _transform(container, normalizer, reducer, selector):
        transformed = normalizer.Transform(container)
        if reducer is not None:
            transformed = reducer.Transform(transformed)
        if selector is not None:
            transformed = selector.Transform(transformed)
        return transformed

    def _copy_classifier(self):
        copy_method = getattr(self.classifier, '__deepcopy__', None)
        if copy_method is not None:
            try:
                if not inspect.signature(copy_method).parameters:
                    return copy_method()
            except (TypeError, ValueError):
                pass
        return deepcopy(self.classifier)

    def _fit_classifier(self, train_container, candidate):
        classifier = self._copy_classifier()
        classifier.SetDataContainer(train_container)
        if candidate:
            classifier.SetModelParameter(candidate)
        classifier.Fit()
        return classifier

    def _score_candidate(self, data_container, candidate, cv_parts=None):
        array = data_container.GetArray()
        label = data_container.GetLabel()
        feature_names = data_container.GetFeatureName()
        case_names = data_container.GetCaseName()
        if cv_parts is None:
            cv_parts = min(
                self.cv_parts,
                GetMaximumCvParts(label),
            )

        scores = []
        cv = StratifiedKFold(n_splits=cv_parts, shuffle=False)
        for train_indices, validation_indices in cv.split(array, label):
            train = self._subset(
                array, label, feature_names, case_names, train_indices
            )
            validation = self._subset(
                array, label, feature_names, case_names, validation_indices
            )
            validation_label = validation.GetLabel()
            processed_train, normalizer, reducer, selector = (
                self._fit_processing(train)
            )
            processed_validation = self._transform(
                validation, normalizer, reducer, selector
            )
            classifier = self._fit_classifier(processed_train, candidate)
            predicted = classifier.GetModel().predict(
                processed_validation.GetArray()
            )
            scores.append(
                accuracy_score(predicted, validation_label)
            )
        return float(np.mean(scores))

    def _select_parameters(self, data_container, cv_parts):
        best_candidate = None
        best_score = -np.inf
        for candidate in ParameterGrid(self.param_grid):
            try:
                score = self._score_candidate(
                    data_container, candidate, cv_parts
                )
            except Exception as error:
                raise RuntimeError(
                    "Classifier {} failed for candidate {}."
                    .format(self.classifier.GetName(), candidate)
                ) from error
            if score > best_score:
                best_candidate = candidate
                best_score = score
        return best_candidate if best_candidate is not None else {}

    def select_parameters(self, data_container):
        if not HasEffectiveParameterGrid(self.param_grid):
            return {}
        cv_parts = min(
            self.cv_parts,
            GetMaximumCvParts(data_container.GetLabel()),
        )
        return self._select_parameters(data_container, cv_parts)

    def evaluate(self, data_container):
        array = data_container.GetArray()
        label = data_container.GetLabel()
        feature_names = data_container.GetFeatureName()
        case_names = data_container.GetCaseName()
        ValidateCvParts(label, self.cv_parts)

        predictions = np.empty(label.shape[0], dtype=float)
        cv = StratifiedKFold(n_splits=self.cv_parts, shuffle=False)
        for train_indices, validation_indices in cv.split(array, label):
            outer_train = self._subset(
                array, label, feature_names, case_names, train_indices
            )
            outer_validation = self._subset(
                array, label, feature_names, case_names, validation_indices
            )
            candidate = self.select_parameters(outer_train)

            processed_train, normalizer, reducer, selector = (
                self._fit_processing(outer_train)
            )
            processed_validation = self._transform(
                outer_validation, normalizer, reducer, selector
            )
            classifier = self._fit_classifier(processed_train, candidate)
            fold_predictions = np.asarray(
                classifier.Predict(processed_validation.GetArray())
            ).reshape(-1)
            predictions[validation_indices] = fold_predictions

        return NestedValidationResult(
            prediction=predictions,
            label=np.array(label, copy=True),
            case_names=list(deepcopy(case_names)),
        )
