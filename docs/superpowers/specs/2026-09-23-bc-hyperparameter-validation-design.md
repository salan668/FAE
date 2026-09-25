# BC Hyperparameter Validation Design

## Goal

Make BC model development safe and predictable when **Hyper-Parameters** is
enabled. The change must prevent preprocessing leakage during validation, keep
classifier searches valid across repeated pipeline combinations, reject invalid
cross-validation settings before execution, and avoid searches that contain no
actual parameter choices.

## Scope

The implementation covers the classifiers exposed by
`BC/GUI/ProcessForm.py`: SVM, LDA, MLP/AE, Random Forest, Logistic Regression,
L1 Logistic Regression, AdaBoost, Decision Tree, Gaussian Process, and Naive
Bayes. Existing result folders, metrics, prediction files, pipeline metadata,
and visualization fallbacks remain compatible.

## Design

### Nested cross-validation

`PipelinesManager` will evaluate each model configuration with explicit nested
stratified cross-validation:

1. Split the original training data into outer training and validation folds.
2. For each hyperparameter candidate, split the outer training fold into inner
   training and validation folds.
3. Deep-copy and fit balancing, normalization, dimensionality reduction, and
   feature selection only on each inner training fold.
4. Transform the matching inner validation fold with those fitted copies.
5. Fit a fresh classifier with the candidate parameters and score its
   predictions by accuracy.
6. Select the candidate with the highest mean inner score.
7. Refit fresh processing components and the selected classifier on the full
   outer training fold, then predict the untouched outer validation fold.

The out-of-fold predictions produced by the outer loop become `CV_VAL`.
Temporary fold processing must not write intermediate files.

For the production model, candidate parameters are selected by the same
fold-local evaluation over the complete original training set. The existing
full-data processing path then fits and saves the final transformations,
classifier, metrics, and visualization artifacts.

When hyperparameter search is disabled, outer validation must still fit all
supervised and data-derived processing inside each fold so that `CV_VAL`
remains out-of-sample.

### Component isolation

Every fold receives deep copies of the balancer, normalizer, dimension reducer,
feature selector, and classifier. No fitted estimator or selected feature set
may be reused between candidates, folds, or full-data pipeline combinations.
The implementation will use the existing `DataContainer`, `Run`, and
`Transform` interfaces rather than introducing sklearn adapters.

### Parameter handling

- Treat `{}`, `[{}]`, and missing classifier entries as no effective search.
- Normalize classifier settings to a list of parameter dictionaries.
- Use `sklearn.model_selection.ParameterGrid` for deterministic candidate
  expansion.
- Set every selected parameter explicitly on a fresh classifier.
- Update LDA's SVD candidate to set `shrinkage` to JSON `null`.
- Add a Naive Bayes `var_smoothing` grid.
- Keep Gaussian Process configured with no effective search unless meaningful
  candidates are added later.

### Cross-validation validation

After loading training data, set the fold spin box maximum to the smallest
class population rather than the total case count. Immediately before
execution, validate the selected fold count again and show a clear message if
it exceeds the smallest class population. Validation uses original class
counts because resampling occurs only within training folds.

Inner cross-validation uses the requested fold count when possible and reduces
it to the smallest class population in the current outer training fold. A fold
with fewer than two members in either class is rejected with a clear error
instead of allowing sklearn to fail deep inside the worker.

### Model artifacts and warnings

The existing SVM kernel-aware coefficient and SHAP behavior remains part of the
branch. Expected absence of unsupported optional explanation output must use
the visualization's existing selector-rank fallback rather than an error-shaped
log. Model serialization must continue even when optional explanation export
fails.

## Error handling

Invalid CV settings are rejected in the GUI before the worker starts. Invalid
parameter candidates fail with classifier and candidate context; they are not
silently converted into successful results. If every candidate fails, the
pipeline stops with an actionable error. Optional visualization artifacts may
fall back only where the current UI already supports that fallback.

## Compatibility

No serialized pipeline metadata fields or accepted-version rules change.
Existing result readers continue to consume the same metric, prediction,
model, coefficient, SHAP, and parameter filenames. The implementation does not
modify generated UI structure.

## Verification

Regression tests will cover:

- preprocessing and feature selection being fitted only on fold training data;
- nested selection and outer prediction using fresh classifier instances;
- LDA searches remaining valid across consecutive pipeline combinations;
- rejection and GUI limiting of excessive fold counts;
- skipping empty Gaussian Process searches;
- Naive Bayes parameter search;
- SVM linear and non-linear save behavior;
- successful model serialization and expected visualization fallback files.

The focused test suite and Python compilation checks must pass before the work
is considered complete.
