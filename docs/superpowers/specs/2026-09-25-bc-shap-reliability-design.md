# BC hyperparameter and SHAP reliability

## Goal

Make BC model-development results scientifically traceable: cross-validation must not use validation cases for hyperparameter selection, and the Feature Contribution panel must never display a SHAP artifact that does not belong to the saved final model and its explained cases.

## Scope

This change covers the BC pipeline, classifier persistence, SHAP artifacts, and the Feature Contribution panel. It does not add approximate explainers for models that are not supported by the pinned SHAP runtime.

## Result contract

Each classifier result folder will contain an `explanation.json` status record. It records the artifact format version, estimator name and fitted parameters, explanation status (`available`, `unsupported`, or `failed`), reason, explained-data kind, and names of the value and feature-data CSV files.

When status is `available`, the pipeline writes `Classifier_shap.csv` containing signed SHAP values and `Classifier_shap_features.csv` containing the transformed feature values for the same real training cases, in the same row and column order. Files are written through temporary paths and become visible only after validation succeeds. Failed or unsupported computation removes obsolete SHAP and coefficient artifacts for the classifier before writing status.

The final classifier may be trained on balanced data. Its explanation background and explained rows must instead be the transformed, non-resampled training container. This prevents synthetic rows from appearing as clinical cases in a SHAP plot.

Supported explainers are explicit: linear SVM, LDA, LR, and LRLasso use the linear path; Random Forest and Decision Tree use the tree path. Non-linear SVM, AdaBoost, AE, Gaussian Process, and Naive Bayes are recorded as unsupported in this release.

## UI behavior and backward safety

The Feature Contribution panel reads `explanation.json` before opening a SHAP CSV. It renders a standard beeswarm only for a valid, matching pair of SHAP and feature-data files. The dot x-position represents SHAP value and color represents the corresponding low-to-high feature value.

Existing result folders without the status record are treated as legacy and do not display their standalone SHAP CSV. The panel shows the existing coefficient or selector-rank fallback and states that rerunning the model is required for a verified SHAP explanation.

## Hyperparameter evaluation

The BC pipeline uses fold-local preprocessing, balancing, feature selection, and hyperparameter selection for outer-fold predictions. Parameter candidates are scored with ROC-AUC from predicted probabilities. The final saved model selects its parameters using the full training data, then trains on its balanced transformed form. The final selected parameters and per-fold selected parameters are persisted for audit.

The implementation reuses the existing `fix/bc-hyperparameter-validation` direction only where its behavior meets this contract. Cross-validation validation must account for class counts and raise a clear error when an outer fold cannot support its inner selection.

## Tests and acceptance criteria

Tests run with the `fae` Conda environment and cover:

1. RBF and polynomial SVM, AdaBoost, and any SHAP failure remove old SHAP artifacts and write an unavailable status.
2. A supported linear or tree model writes aligned SHAP values, feature data, and metadata; the explanation rows are real rather than synthetic training cases.
3. The visualization refuses incomplete, stale, malformed, or legacy SHAP artifacts and falls back safely.
4. Nested evaluation keeps every outer validation case outside its inner parameter search, uses ROC-AUC, and persists fold and final parameter choices.
5. The complete BC regression suite passes in the `fae` environment.
