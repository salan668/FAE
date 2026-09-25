# BC Hyperparameter Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BC hyperparameter tuning leakage-free, reject invalid CV settings, and handle every GUI classifier's parameter and explanation outputs without misleading errors.

**Architecture:** Add a focused nested-validation module that executes the existing `DataContainer` processors on fresh copies inside every fold. `PipelinesManager` will use it for out-of-fold predictions and parameter selection, then retain the existing full-data processing and artifact-writing path for the final model.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, imbalanced-learn, PySide6, unittest

---

## File map

- Create `BC/FeatureAnalysis/NestedValidation.py`: fold-local processing, candidate scoring, nested CV predictions, and final parameter selection.
- Modify `BC/FeatureAnalysis/Pipelines.py`: integrate nested evaluation and stop reusing fitted classifiers between pipeline combinations.
- Modify `BC/FeatureAnalysis/Classifier.py`: normalize empty grids, retain the SVM fix, and treat unsupported AdaBoost SHAP as an intentional fallback.
- Modify `BC/FeatureAnalysis/CrossValidation.py`: centralize class-count validation for CV fold counts.
- Modify `BC/GUI/ProcessForm.py`: constrain and revalidate the CV fold selector.
- Modify `BC/HyperParameters/Classifier/LDA.json`: make the SVD branch explicitly clear shrinkage.
- Add `BC/HyperParameters/Classifier/NB.json`: tune GaussianNB `var_smoothing`.
- Modify `tests/test_svm_classifier.py`: retain SVM regression coverage already present in the working tree.
- Create `tests/test_classifier_hyperparameters.py`: parameter-grid, LDA, GP, NB, and AdaBoost regressions.
- Create `tests/test_nested_validation.py`: nested-fold isolation, fresh-instance, parameter selection, and prediction-order tests.
- Create `tests/test_cross_validation.py`: fold-limit and invalid-setting tests.
- Create `tests/test_pipeline_nested_validation.py`: integration between `PipelinesManager` and the nested evaluator.

### Task 1: Stabilize classifier parameter grids and optional outputs

**Files:**
- Modify: `BC/FeatureAnalysis/Classifier.py:95-126, 208-270, 367-390`
- Modify: `BC/HyperParameters/Classifier/LDA.json`
- Create: `BC/HyperParameters/Classifier/NB.json`
- Test: `tests/test_classifier_hyperparameters.py`
- Test: `tests/test_svm_classifier.py`

- [ ] **Step 1: Write failing tests for effective grids and classifier artifacts**

Create tests that assert:

```python
def make_binary_data():
    return DataContainer(
        array=np.array([
            [0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
            [1.0, 1.0], [0.2, 0.8], [0.8, 0.2],
        ]),
        label=np.array([0, 1, 1, 0, 1, 0]),
        feature_name=['feature_1', 'feature_2'],
        case_name=['case_1', 'case_2', 'case_3', 'case_4', 'case_5', 'case_6'],
    )


def test_empty_parameter_grid_fits_without_grid_search():
    classifier = GaussianProcess()
    classifier.SetDataContainer(make_binary_data())
    with patch('BC.FeatureAnalysis.Classifier.GridSearchCV') as search:
        classifier.Fit([{}], cv_part=2)
    search.assert_not_called()


def test_lda_svd_grid_clears_previous_shrinkage():
    settings = GetClassifierHyperParams()['LDA']
    svd = next(item for item in settings if item['solver'] == ['svd'])
    assert svd['shrinkage'] == [None]


def test_naive_bayes_has_effective_parameter_grid():
    settings = GetClassifierHyperParams()['NB']
    assert settings == [{'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}]


def test_adaboost_save_uses_existing_visualization_fallback():
    classifier = AdaBoost(random_state=0)
    classifier.SetDataContainer(make_binary_data())
    classifier.Fit()
    with TemporaryDirectory() as folder:
        with self.assertNoLogs(classifier.logger, logging.WARNING):
            classifier.Save(folder)
        assert not os.path.exists(os.path.join(folder, 'AB_shap.csv'))
        assert os.path.exists(os.path.join(folder, 'model.pickle'))
```

- [ ] **Step 2: Run the tests and verify the expected failures**

Run:

```powershell
& 'C:\Users\SunsServer\miniconda3\envs\fae\python.exe' -m unittest tests.test_classifier_hyperparameters tests.test_svm_classifier -v
```

Expected: the GP test reports that `GridSearchCV` was called, LDA lacks
`shrinkage: [None]`, NB is absent, and AdaBoost logs the unsupported SHAP
warning. Existing SVM tests remain green.

- [ ] **Step 3: Implement effective-grid detection and classifier-specific output behavior**

Add to `Classifier.py`:

```python
def HasEffectiveParameterGrid(param_grid):
    if isinstance(param_grid, dict):
        return bool(param_grid)
    if isinstance(param_grid, list):
        return any(isinstance(candidate, dict) and candidate for candidate in param_grid)
    return False
```

Use it in both `Fit` and `HyperFit` instead of checking only collection length.
Retain `n_jobs=1`. Remove the
`self._SaveShap(store_folder, explainer_type='tree')` call from `AdaBoost.Save`
because
SHAP 0.49.1 does not support `AdaBoostClassifier`; continue calling the base
`Save`.

Update `LDA.json`:

```json
{"solver": ["svd"], "shrinkage": [null]}
```

Create `NB.json`:

```json
{
  "_commit": "Gaussian Naive Bayes variance smoothing.",
  "setting": [
    {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}
  ]
}
```

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2. Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit**

```powershell
git add BC\FeatureAnalysis\Classifier.py BC\HyperParameters\Classifier\LDA.json BC\HyperParameters\Classifier\NB.json tests\test_classifier_hyperparameters.py tests\test_svm_classifier.py
git commit -m "fix(bc): stabilize classifier hyperparameter grids" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Validate cross-validation fold counts

**Files:**
- Modify: `BC/FeatureAnalysis/CrossValidation.py`
- Modify: `BC/GUI/ProcessForm.py:158-190, 361-405`
- Test: `tests/test_cross_validation.py`

- [ ] **Step 1: Write failing tests for class-aware fold limits**

```python
class CrossValidationLimitTest(unittest.TestCase):
    def test_maximum_fold_count_is_smallest_class(self):
        labels = np.array([0] * 12 + [1] * 5)
        self.assertEqual(GetMaximumCvParts(labels), 5)

    def test_validate_rejects_too_many_folds(self):
        labels = np.array([0] * 12 + [1] * 5)
        with self.assertRaisesRegex(ValueError, 'smallest class has 5'):
            ValidateCvParts(labels, 6)

    def test_validate_requires_two_cases_per_class(self):
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            ValidateCvParts(np.array([0, 0, 1]), 2)
```

- [ ] **Step 2: Run and verify failure**

```powershell
& 'C:\Users\SunsServer\miniconda3\envs\fae\python.exe' -m unittest tests.test_cross_validation -v
```

Expected: import failure because the validation functions do not exist.

- [ ] **Step 3: Add reusable validation**

Add to `CrossValidation.py`:

```python
import numpy as np


def GetMaximumCvParts(labels):
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError('Cross-validation requires at least 2 cases in each class.')
    return int(counts.min())


def ValidateCvParts(labels, cv_parts):
    maximum = GetMaximumCvParts(labels)
    if cv_parts > maximum:
        raise ValueError(
            'Cross-validation cannot use {} folds because the smallest class has {} cases.'
            .format(cv_parts, maximum)
        )
    return cv_parts
```

In `LoadTrainingData`, replace the total-case maximum with
`GetMaximumCvParts(self.training_data_container.GetLabel())`. At the start of
`Run`, call `ValidateCvParts`; show its message with `QMessageBox.warning` and
return before opening the destination dialog when validation fails.

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2. Expected: all three tests pass.

- [ ] **Step 5: Commit**

```powershell
git add BC\FeatureAnalysis\CrossValidation.py BC\GUI\ProcessForm.py tests\test_cross_validation.py
git commit -m "fix(bc): validate folds against class counts" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Implement fold-local processing

**Files:**
- Create: `BC/FeatureAnalysis/NestedValidation.py`
- Test: `tests/test_nested_validation.py`

- [ ] **Step 1: Write failing tests for processing isolation**

Use a `RecordingBalance` test double whose class-level `fit_case_counts` list
records the number of cases passed to every `Run` call. Use the real
`NormalizerNone` and SVM wrapper so the evaluator exercises production data
and classifier APIs:

```python
def make_binary_data(case_count=18):
    values = np.arange(case_count * 2, dtype=float).reshape(case_count, 2)
    labels = np.tile(np.array([0, 1]), case_count // 2)
    return DataContainer(
        array=values,
        label=labels,
        feature_name=['feature_1', 'feature_2'],
        case_name=['case_{}'.format(index) for index in range(case_count)],
    )


class RecordingBalance:
    fit_case_counts = []

    def Run(self, container, store_path=''):
        type(self).fit_case_counts.append(len(container.GetCaseName()))
        return container


def test_outer_validation_never_fits_processing_on_all_cases():
    RecordingBalance.fit_case_counts = []
    data = make_binary_data()
    evaluator = NestedPipelineEvaluator(
        balance=RecordingBalance(),
        normalizer=NormalizerNone,
        dimension_reducer=None,
        feature_selector=None,
        feature_number=2,
        classifier=SVM(kernel='linear', random_state=0),
        param_grid={'C': [0.1, 1.0]},
        cv_parts=3,
    )
    result = evaluator.evaluate(data)
    assert result.case_names == data.GetCaseName()
    assert result.label.tolist() == data.GetLabel().tolist()
    assert len(result.prediction) == 18
    assert 18 not in RecordingBalance.fit_case_counts


def test_empty_grid_returns_default_parameters_without_inner_search():
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
    with patch.object(evaluator, '_score_candidate') as score_candidate:
        selected = evaluator.select_parameters(make_binary_data())
    assert selected == {}
    score_candidate.assert_not_called()
```

Add a `CountingClassifier` test double that stores `id(self)` in a class-level
set from `Fit`. Evaluate two candidates over three folds and assert more than
one distinct instance was fitted. Add a `DeterministicClassifier` whose
`choice='good'` predictions match labels and whose `choice='bad'` predictions
invert them; assert `select_parameters` returns `{'choice': 'good'}`.

- [ ] **Step 2: Run and verify failure**

```powershell
& 'C:\Users\SunsServer\miniconda3\envs\fae\python.exe' -m unittest tests.test_nested_validation -v
```

Expected: import failure because `NestedPipelineEvaluator` does not exist.

- [ ] **Step 3: Implement the evaluator**

Create `NestedValidation.py` with these public results and methods:

```python
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from BC.DataContainer.DataContainer import DataContainer
from BC.FeatureAnalysis.Classifier import HasEffectiveParameterGrid
from BC.FeatureAnalysis.CrossValidation import GetMaximumCvParts


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

    def _fit_processing(self, train_container):
        balance = deepcopy(self.balance)
        normalizer = deepcopy(self.normalizer)
        reducer = deepcopy(self.dimension_reducer)
        selector = deepcopy(self.feature_selector)
        balanced = balance.Run(train_container)
        normalized = normalizer.Run(balanced)
        reduced = reducer.Run(normalized) if reducer else normalized
        if selector:
            selector.SetSelectedFeatureNumber(self.feature_number)
            selected = selector.Run(reduced)
        else:
            selected = reduced
        return selected, normalizer, reducer, selector

    def _transform(self, container, normalizer, reducer, selector):
        transformed = normalizer.Transform(container)
        if reducer:
            transformed = reducer.Transform(transformed)
        if selector:
            transformed = selector.Transform(transformed)
        return transformed
```

Implement `_score_candidate` with `StratifiedKFold`, fold-local calls to
`_fit_processing`, a fresh deep-copied classifier with
`SetModelParameter(candidate)`, and `accuracy_score`. Implement
`select_parameters` with `ParameterGrid`, returning `{}` immediately for an
ineffective grid. Implement `evaluate` with an outer `StratifiedKFold`,
per-outer-fold parameter selection, fold-local fitting, and predictions written
back to their original indices.

Inner fold count must be:

```python
inner_parts = min(self.cv_parts, GetMaximumCvParts(outer_train.GetLabel()))
```

Raise a contextual `RuntimeError` if a candidate fails; do not replace failed
scores with a successful-looking default.

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2. Expected: all isolation, fresh-instance,
selection, and ordering tests pass.

- [ ] **Step 5: Commit**

```powershell
git add BC\FeatureAnalysis\NestedValidation.py tests\test_nested_validation.py
git commit -m "feat(bc): add fold-local nested validation" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 4: Integrate nested validation into model development

**Files:**
- Modify: `BC/FeatureAnalysis/Pipelines.py:221-310`
- Test: `tests/test_pipeline_nested_validation.py`

- [ ] **Step 1: Write failing integration tests**

Patch `NestedPipelineEvaluator` at the module boundary and assert:

```python
def test_pipeline_uses_original_training_data_for_nested_validation():
    train = make_binary_data()
    manager = make_manager(
        classifier_list=[SVM(kernel='linear', random_state=0)],
        feature_numbers=[1],
    )
    with patch('BC.FeatureAnalysis.Pipelines.NestedPipelineEvaluator') as evaluator:
        evaluator.return_value.evaluate.return_value = NestedValidationResult(
            prediction=np.linspace(0.1, 0.9, len(train.GetLabel())),
            label=train.GetLabel(),
            case_names=train.GetCaseName(),
        )
        evaluator.return_value.select_parameters.return_value = {'C': 1.0}
        list(manager.Run(train, store_folder=folder))
    evaluator.return_value.evaluate.assert_called_once_with(train)


def test_pipeline_saves_selected_full_data_parameters():
    train = make_binary_data()
    manager = make_manager(
        classifier_list=[SVM(kernel='linear', random_state=0)],
        feature_numbers=[1],
    )
    with patch('BC.FeatureAnalysis.Pipelines.NestedPipelineEvaluator') as evaluator:
        evaluator.return_value.evaluate.return_value = NestedValidationResult(
            prediction=np.linspace(0.1, 0.9, len(train.GetLabel())),
            label=train.GetLabel(),
            case_names=train.GetCaseName(),
        )
        evaluator.return_value.select_parameters.return_value = {'C': 0.3}
        list(manager.Run(train, store_folder=folder))
    with open(find_only_file(folder, 'used_hyper_param.json')) as source:
        used = json.load(source)
    assert used['C'] == 0.3
```

Define `make_manager`, `make_binary_data`, and `find_only_file` in the test
module using `NoneBalance`, `NormalizerNone`, no reducer, no selector, a
two-fold `ArbitratyCrossValidation`, and `tempfile.TemporaryDirectory`. Add a
second integration test that reads `CV_VAL_prediction.csv` and asserts its
index exactly equals `train.GetCaseName()`. Add a third test with two feature
counts and patch `SVM.Save` to record model object IDs; assert the two IDs
differ.

- [ ] **Step 2: Run and verify failure**

```powershell
& 'C:\Users\SunsServer\miniconda3\envs\fae\python.exe' -m unittest tests.test_pipeline_nested_validation -v
```

Expected: failure because `PipelinesManager.Run` still preprocesses globally
before validation and reuses its classifier object.

- [ ] **Step 3: Integrate the evaluator**

At the beginning of `Run`, snapshot unfitted classifier templates:

```python
classifier_templates = [deepcopy(classifier) for classifier in self.classifier_list]
```

For each pipeline combination, construct `NestedPipelineEvaluator` from the
original `train_container` and fresh component copies. Use
`evaluate(train_container)` for `CV_VAL`. Select full-data parameters with
`select_parameters(train_container)`.

For the final persisted model:

```python
final_classifier = deepcopy(classifier_templates[cls_index])
if selected_params:
    final_classifier.SetModelParameter(selected_params)
final_classifier.SetDataContainer(fs_balance_train_container)
final_classifier.Fit()
final_classifier.Save(cls_store_folder)
```

Use `final_classifier` for balanced training, original training, and test
predictions. Do not mutate the template stored in `classifier_templates`.

- [ ] **Step 4: Run unit and integration tests**

```powershell
& 'C:\Users\SunsServer\miniconda3\envs\fae\python.exe' -m unittest tests.test_nested_validation tests.test_pipeline_nested_validation -v
```

Expected: all tests pass and saved parameters match the selected candidate.

- [ ] **Step 5: Commit**

```powershell
git add BC\FeatureAnalysis\Pipelines.py tests\test_pipeline_nested_validation.py
git commit -m "fix(bc): use nested cross-validation for model evaluation" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 5: Verify the complete classifier matrix

**Files:**
- Modify only if a regression is exposed by verification.

- [ ] **Step 1: Run all added regression tests**

```powershell
& 'C:\Users\SunsServer\miniconda3\envs\fae\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass without unexpected warnings or tracebacks.

- [ ] **Step 2: Compile every changed Python module**

```powershell
& 'C:\Users\SunsServer\miniconda3\envs\fae\python.exe' -m py_compile BC\FeatureAnalysis\Classifier.py BC\FeatureAnalysis\CrossValidation.py BC\FeatureAnalysis\NestedValidation.py BC\FeatureAnalysis\Pipelines.py BC\GUI\ProcessForm.py
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Run a headless smoke matrix**

Use a deterministic binary `DataContainer` with at least 20 cases per class,
two features, two CV folds, no balancing, no dimensionality reduction, and no
feature selector. Run each classifier with Hyper-Parameters enabled into its
own temporary output directory. Assert that every run writes `model.pickle`,
`model_param.json`, `used_hyper_param.json`, and prediction CSV files. Assert
that SVM non-linear and AdaBoost use the existing selector-rank fallback when
SHAP output is unavailable.

- [ ] **Step 4: Check repository state and diff**

```powershell
git diff --check
git status --short
git log --oneline master..HEAD
```

Expected: no whitespace errors; only planned source, config, test, and design
files are changed. The pre-existing `.github/copilot-instructions.md` change
must remain uncommitted and untouched.

- [ ] **Step 5: Request final code review**

Review the complete `master..HEAD` change set against
`docs/superpowers/specs/2026-09-23-bc-hyperparameter-validation-design.md`.
Resolve all critical and important findings, rerun Steps 1-4, and commit any
review fixes with the required co-author trailer.
