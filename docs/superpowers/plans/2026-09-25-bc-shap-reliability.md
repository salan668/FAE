# BC Hyperparameter and SHAP Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible BC hyperparameter evaluation and SHAP explanations that cannot be silently reused for another fitted model or synthetic training cases.

**Architecture:** Add a focused explanation-artifact module that atomically writes, invalidates, and validates SHAP sidecars. Classifier persistence uses this module and supplies real transformed training cases for explanation. The pipeline uses fold-local nested evaluation for reported CV results, while the UI accepts only a verified metadata/value/feature-data triplet before rendering a conventional SHAP beeswarm.

**Tech Stack:** Python 3.11, PySide6, pandas, numpy, scikit-learn, imbalanced-learn, shap 0.49.1, unittest, `fae` Conda environment.

**Spec:** `docs/superpowers/specs/2026-09-25-bc-shap-reliability-design.md`

## Global Constraints

- Work only on `codex/bc-shap-reliability`; do not alter the dirty `master` checkout.
- Run every test with `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe` and `PYTHONDONTWRITEBYTECODE=1`.
- Persist no approximate explanation for an unsupported estimator in this release.
- An SHAP artifact is valid only when status is `available`, both CSV files validate, their shape and labels align, and metadata matches `model_param.json`.
- Final model fitting may use balanced data, but SHAP background and explained rows must be transformed non-resampled training cases.
- Nested model selection uses ROC-AUC from probability predictions and keeps outer validation cases out of candidate selection.
- Preserve legacy result folders without metadata by using the existing non-SHAP fallback rather than trusting standalone SHAP CSV files.

## Review Focus

- A former linear SVM result is overwritten by an RBF SVM: no SVM SHAP or coefficient artifact may remain; Task 2 adds this regression test.
- A valid SHAP CSV is paired with a missing, malformed, or differently indexed feature-data CSV: the UI must reject it; Task 3 adds these tests.
- SMOTE creates synthetic rows during final fitting: SHAP case names must remain the real transformed training case names; Task 2 adds this test.
- Equal ROC-AUC candidates occur in nested search: parameter selection must remain deterministic; Task 4 adds this test.
- The smallest class is too small to create an inner fold: model development must raise an actionable validation error; Task 4 adds this test.

---

### Task 1: Explanation artifact contract

**Files:**
- Create: `BC/FeatureAnalysis/ExplanationArtifacts.py`
- Create: `tests/test_explanation_artifacts.py`

**Interfaces:**
- Consumes: a classifier result directory, classifier name, fitted estimator parameters, a SHAP DataFrame, and a transformed-feature DataFrame.
- Produces: `write_available_explanation(folder, classifier_name, model_params, shap_df, feature_df, explained_data_kind) -> dict`, `write_unavailable_explanation(folder, classifier_name, model_params, status, reason) -> dict`, and `load_verified_explanation(folder, classifier_name) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None`.

- [ ] **Step 1: Write the failing artifact tests**

```python
def test_available_artifact_writes_aligned_values_features_and_metadata(tmp_path):
    shap_df = pd.DataFrame([[0.1, -0.2]], index=['case_1'], columns=['a', 'b'])
    feature_df = pd.DataFrame([[3.0, 4.0]], index=['case_1'], columns=['a', 'b'])
    metadata = write_available_explanation(tmp_path, 'SVM', {'kernel': 'linear'}, shap_df, feature_df, 'real_train')
    assert metadata['status'] == 'available'
    assert load_verified_explanation(tmp_path, 'SVM')[0].equals(shap_df)

def test_unavailable_artifact_removes_old_shap_and_coef(tmp_path):
    (tmp_path / 'SVM_shap.csv').write_text('old')
    (tmp_path / 'SVM_shap_features.csv').write_text('old')
    (tmp_path / 'SVM_coef.csv').write_text('old')
    metadata = write_unavailable_explanation(tmp_path, 'SVM', {'kernel': 'rbf'}, 'unsupported', 'non-linear kernel')
    assert metadata['status'] == 'unsupported'
    assert not (tmp_path / 'SVM_shap.csv').exists()
    assert not (tmp_path / 'SVM_coef.csv').exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_explanation_artifacts -v`

Expected: FAIL because `ExplanationArtifacts` and its public functions do not exist.

- [ ] **Step 3: Implement atomic artifact writing and verification**

```python
ARTIFACT_VERSION = 1

def write_available_explanation(folder, classifier_name, model_params, shap_df, feature_df, explained_data_kind):
    _validate_aligned_frames(shap_df, feature_df)
    _write_csv_atomically(shap_df, _value_path(folder, classifier_name))
    _write_csv_atomically(feature_df, _feature_path(folder, classifier_name))
    return _write_metadata(folder, classifier_name, model_params, 'available', '', explained_data_kind)

def load_verified_explanation(folder, classifier_name):
    metadata = _read_metadata(folder)
    if metadata is None or metadata.get('status') != 'available':
        return None
    shap_df, feature_df = _read_and_validate_frames(folder, classifier_name, metadata)
    return shap_df, feature_df, metadata
```

Implement SHA-256 signatures for serialized fitted parameters and aligned CSV content. Metadata must name both files, include `explained_data_kind`, and serialize with `default=str`. `load_verified_explanation` must compare metadata's parameter signature with the sibling `model_param.json` before returning frames. Unavailable status removes all classifier-specific SHAP and coefficient artifacts before atomically replacing `explanation.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_explanation_artifacts -v`

Expected: PASS; tests prove aligned persistence, content validation, and cleanup.

- [ ] **Step 5: Commit**

```bash
git add BC/FeatureAnalysis/ExplanationArtifacts.py tests/test_explanation_artifacts.py
git commit -m "feat(bc): add verified explanation artifacts"
```

### Task 2: Classifier persistence and real-case SHAP generation

**Files:**
- Modify: `BC/FeatureAnalysis/Classifier.py:136-191,237-408,473-553`
- Modify: `BC/FeatureAnalysis/Pipelines.py:265-305`
- Create: `tests/test_classifier_explanations.py`
- Modify: `tests/test_svm_classifier.py`, `tests/test_classifier_hyperparameters.py`

**Interfaces:**
- Consumes: Task 1 artifact functions and optional `explanation_container: DataContainer` passed to `Classifier.Save`.
- Produces: `Classifier.Save(store_folder, explanation_container=None)`, where every classifier writes one verified status record; `PipelinesManager.Run` passes `fs_train_container` to final save after fitting on `fs_balance_train_container`.

- [ ] **Step 1: Write failing classifier and pipeline tests**

```python
def test_rbf_svm_replaces_old_linear_explanation_with_unsupported_status(tmp_path, data):
    (tmp_path / 'SVM_shap.csv').write_text('stale')
    classifier = SVM(kernel='rbf')
    classifier.SetDataContainer(data)
    classifier.Fit()
    classifier.Save(tmp_path, explanation_container=data)
    assert _status(tmp_path)['status'] == 'unsupported'
    assert not (tmp_path / 'SVM_shap.csv').exists()

def test_final_shap_uses_real_cases_after_balanced_fit(tmp_path, real_data, balanced_data):
    classifier = LR()
    classifier.SetDataContainer(balanced_data)
    classifier.Fit()
    classifier.Save(tmp_path, explanation_container=real_data)
    shap_df, feature_df, metadata = load_verified_explanation(tmp_path, 'LR')
    assert list(shap_df.index) == real_data.GetCaseName()
    assert metadata['explained_data_kind'] == 'real_train'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_classifier_explanations tests.test_svm_classifier tests.test_classifier_hyperparameters -v`

Expected: FAIL because `Save` has no `explanation_container` argument and does not create verified status metadata.

- [ ] **Step 3: Implement explicit model support and explanation-container saving**

```python
def _SaveShap(self, store_folder, explainer_type, explanation_container):
    X = explanation_container.GetArray()
    feature_df = pd.DataFrame(X, index=explanation_container.GetCaseName(), columns=explanation_container.GetFeatureName())
    try:
        shap_df = _compute_shap_values(self.model, X, feature_df.columns, explainer_type)
        write_available_explanation(store_folder, self.GetName(), self.model.get_params(), shap_df, feature_df, 'real_train')
    except Exception as error:
        write_unavailable_explanation(store_folder, self.GetName(), self.model.get_params(), 'failed', str(error))
```

Pass the real transformed training container to supported save methods. Mark RBF/poly SVM, AdaBoost, AE, GP, and NB `unsupported` before an explainer is attempted. Keep coefficient export only for supported linear estimators. Update `PipelinesManager.Run` so final fitting stays on balanced features while `Save(..., explanation_container=fs_train_container)` explains real cases.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_classifier_explanations tests.test_svm_classifier tests.test_classifier_hyperparameters -v`

Expected: PASS; unsupported models invalidate old output and supported models emit aligned real-case explanations.

- [ ] **Step 5: Commit**

```bash
git add BC/FeatureAnalysis/Classifier.py BC/FeatureAnalysis/Pipelines.py tests/test_classifier_explanations.py tests/test_svm_classifier.py tests/test_classifier_hyperparameters.py
git commit -m "fix(bc): bind SHAP artifacts to fitted models"
```

### Task 3: Verified SHAP visualization and safe fallback

**Files:**
- Modify: `BC/Visualization/FeatureSort.py:209-274`
- Modify: `BC/GUI/VisualizationForm.py:471-545`
- Create: `tests/test_shap_visualization.py`

**Interfaces:**
- Consumes: `load_verified_explanation(folder, classifier_name)` from Task 1.
- Produces: `SHAPBeeswarmPlot(shap_df, feature_df, max_num=20, is_show=False, fig=None)` and a UI contribution path that uses SHAP only for a verified tuple.

- [ ] **Step 1: Write failing visualization tests**

```python
def test_beeswarm_uses_feature_values_for_dot_colors():
    ax = SHAPBeeswarmPlot(shap_df, feature_df, fig=Figure())
    assert ax.get_figure().axes[-1].get_ylabel() == 'Feature value'

def test_invalid_or_legacy_artifact_is_not_loaded(tmp_path):
    (tmp_path / 'SVM_shap.csv').write_text(',a\\ncase_1,0.1\\n')
    assert load_verified_explanation(tmp_path, 'SVM') is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_shap_visualization tests.test_explanation_artifacts -v`

Expected: FAIL because the plot does not accept feature data and legacy standalone SHAP CSVs are still treated as displayable by the UI path.

- [ ] **Step 3: Implement verified loading and conventional beeswarm color semantics**

```python
result = load_verified_explanation(cls_folder, classifier_name)
if result is not None:
    shap_df, feature_df, metadata = result
    SHAPBeeswarmPlot(shap_df, feature_df, max_num=max_num, fig=self.canvasFeature.getFigure())
else:
    self._show_contribution_fallback(cls_folder, fs_folder)
```

Map each plotted feature's values to a sequential low-to-high colormap, label the colorbar `Feature value`, and retain signed SHAP values on the x-axis. The fallback label states that verified SHAP is unavailable and legacy folders must be regenerated.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_shap_visualization tests.test_explanation_artifacts -v`

Expected: PASS; color semantics use feature values and missing or corrupt artifacts do not enter the SHAP path.

- [ ] **Step 5: Commit**

```bash
git add BC/Visualization/FeatureSort.py BC/GUI/VisualizationForm.py tests/test_shap_visualization.py
git commit -m "fix(bc): require verified SHAP visualization"
```

### Task 4: Fold-local nested hyperparameter evaluation and audit trail

**Files:**
- Create: `BC/FeatureAnalysis/NestedValidation.py`
- Modify: `BC/FeatureAnalysis/CrossValidation.py`, `BC/FeatureAnalysis/Pipelines.py`, `BC/GUI/ProcessForm.py`, `BC/FeatureAnalysis/Classifier.py`
- Modify: `BC/HyperParameters/Classifier/LDA.json`
- Create: `BC/HyperParameters/Classifier/NB.json`
- Create: `tests/test_nested_validation.py`, `tests/test_pipeline_nested_validation.py`
- Modify: `tests/test_classifier_hyperparameters.py`

**Interfaces:**
- Consumes: classifier templates, parameter grid, preprocessing stages, and outer training data.
- Produces: `NestedPipelineEvaluator.evaluate(data_container) -> NestedValidationResult(prediction, label, case_names, audit)`, `select_parameters(data_container) -> dict`, and a per-model `selected_hyper_parameters.json` containing `final` and `outer_folds`.

- [ ] **Step 1: Write failing nested-validation tests**

```python
def test_outer_validation_cases_are_not_seen_by_inner_parameter_search():
    result = evaluator.evaluate(data)
    assert result.case_names == data.GetCaseName()
    assert all(set(item['outer_validation_cases']).isdisjoint(item['inner_selection_cases']) for item in result.audit)

def test_nested_selection_uses_auc_and_is_deterministic_for_ties():
    assert evaluator.select_parameters(data) == {'C': 0.1}

def test_two_fold_outer_cv_with_two_cases_per_class_has_clear_inner_cv_error():
    with self.assertRaisesRegex(ValueError, 'inner cross-validation'):
        evaluator.evaluate(minimal_balanced_data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_nested_validation tests.test_pipeline_nested_validation -v`

Expected: FAIL because nested evaluation and selected-parameter audit records do not exist.

- [ ] **Step 3: Implement fold-local preprocessing, ROC-AUC selection, and persistence**

```python
score = roc_auc_score(validation.GetLabel(), classifier.Predict(processed_validation.GetArray()))
if score > best_score:
    best_candidate, best_score = candidate, score

if GetMaximumCvParts(outer_train.GetLabel()) < 2:
    raise ValueError('Nested inner cross-validation requires at least 2 cases in each class after an outer split.')
```

Deep-copy every processing stage and classifier template per fold. Replace global pre-CV grid fitting with nested outer predictions, then select final parameters on full real training data and fit the saved model on its balanced transformed equivalent. Use `HasEffectiveParameterGrid` to skip `[{}]`; retain valid LDA `svd` shrinkage `null`; add NB variance smoothing metadata. Write final and fold selections atomically.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_nested_validation tests.test_pipeline_nested_validation tests.test_classifier_hyperparameters -v`

Expected: PASS; folds are isolated, AUC drives selection, ties are deterministic, invalid inner folds raise clearly, and audit output is complete.

- [ ] **Step 5: Commit**

```bash
git add BC/FeatureAnalysis/NestedValidation.py BC/FeatureAnalysis/CrossValidation.py BC/FeatureAnalysis/Pipelines.py BC/GUI/ProcessForm.py BC/FeatureAnalysis/Classifier.py BC/HyperParameters/Classifier/LDA.json BC/HyperParameters/Classifier/NB.json tests/test_nested_validation.py tests/test_pipeline_nested_validation.py tests/test_classifier_hyperparameters.py
git commit -m "fix(bc): isolate hyperparameter validation"
```

### Task 5: Documentation and full regression verification

**Files:**
- Modify: `README.md`, `.github/copilot-instructions.md`
- Create: `tests/test_documentation_contract.py`
- Test: all `tests/test_*.py` files from Tasks 1-4.

**Interfaces:**
- Consumes: all public interfaces from Tasks 1-4.
- Produces: documented artifact format and a repeatable `fae` regression command.

- [ ] **Step 1: Write failing documentation-presence test**

```python
def test_documentation_describes_verified_shap_contract():
    readme = Path('README.md').read_text(encoding='utf-8')
    assert 'explanation.json' in readme
    assert 'nested cross-validation' in readme.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest tests.test_documentation_contract -v`

Expected: FAIL because documentation does not describe verified artifacts or nested validation.

- [ ] **Step 3: Add concise artifact and compatibility documentation**

```markdown
### Model explanations

BC writes `explanation.json` with verified SHAP values and transformed real-case feature data for supported estimators. Results created before this format require rerunning the model before SHAP can be displayed.
```

Document supported estimators, fallback behavior, and the exact `fae` test command. Add `tests/test_documentation_contract.py` using Step 1's assertion.

- [ ] **Step 4: Run complete regression suite**

Run: `C:\\Users\\SunsServer\\miniconda3\\envs\\fae\\python.exe -m unittest discover -s tests -p "test_*.py" -v`

Expected: PASS with all BC and documentation tests green.

- [ ] **Step 5: Commit**

```bash
git add README.md .github/copilot-instructions.md tests/test_documentation_contract.py
git commit -m "docs: explain verified BC model artifacts"
```

## Self-review

- Spec coverage: Task 1 owns status, atomic writes, signatures, and cleanup; Task 2 owns model support plus real-case explanation; Task 3 owns UI loading and feature-value colors; Task 4 owns nested ROC-AUC evaluation and audit; Task 5 owns operational documentation and full verification.
- Placeholder scan: every implementation step names its API, test, and command.
- Type consistency: Task 1 produces `load_verified_explanation`; Tasks 2 and 3 consume it. Task 2 introduces `explanation_container`; Task 4 preserves that call contract. Task 4 produces fold audit data consumed by pipeline persistence.
- Review focus: every listed failure mode has an explicit regression test in its owning task.
