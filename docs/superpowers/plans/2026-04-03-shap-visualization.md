# SHAP Feature Contribution Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace/augment the existing Coef bar chart in BC Visualization with SHAP-based feature contribution plots (bar + beeswarm), supporting linear and tree classifiers with graceful fallback for others.

**Architecture:** During training, each supported classifier computes SHAP values via `LinearExplainer` or `TreeExplainer` and saves a `{Name}_shap.csv` (samples × features). At visualization time, `UpdateContribution()` reads this file and renders either a mean-|SHAP| bar chart (red/blue by sign) or a beeswarm scatter plot. Unsupported classifiers (AE, GP, NB, SVM with non-linear kernel) fall back to the existing coef/selector display.

**Tech Stack:** `shap` library, `matplotlib`, `numpy`, `pandas`, PySide6

---

## File Map

| File | Change |
|------|--------|
| `BC/FeatureAnalysis/Classifier.py` | Add `_SaveShap()` to base class; call from SVM/LR/LRLasso/LDA/RF/DT/AdaBoost `Save()` |
| `BC/Visualization/FeatureSort.py` | Add `SHAPBarPlot()` and `SHAPBeeswarmPlot()` |
| `BC/GUI/VisualizationForm.py` | Add plot-type radio buttons; update `UpdateContribution()` |
| `install.bat` | Add `shap` to pip install line |
| `README.md` | Add `shap` to dependencies list |

---

## Task 1: Add `shap` Dependency

**Files:**
- Modify: `install.bat`
- Modify: `README.md`

- [ ] **Step 1: Add shap to install.bat**

Open `install.bat`. The current pip install line looks like:
```
pip install numpy scipy matplotlib pandas pillow pyside6 ...
```
Add `shap` to the end of that line:
```bat
pip install numpy scipy matplotlib pandas pillow pyside6 pyqtgraph pyradiomics seaborn reportlab imbalanced-learn pdfdocument statsmodels lifelines pyinstaller scikit-learn shap
```

- [ ] **Step 2: Add shap to README.md dependencies list**

In `README.md`, find the pre-install dependencies list (the bullet list under the install section). Add:
```markdown
- shap
```

- [ ] **Step 3: Install shap in the current environment**

Run:
```
pip install shap
```
Expected: shap installs without error. Verify with `python -c "import shap; print(shap.__version__)"`.

---

## Task 2: Add `_SaveShap()` Helper to Base Classifier

**Files:**
- Modify: `BC/FeatureAnalysis/Classifier.py`

The base `Classifier` class (around line 40) needs a `_SaveShap(store_folder, explainer_type)` method. Add it after the existing `Save()` method.

- [ ] **Step 1: Add import for shap at the top of Classifier.py**

Find the existing imports at the top of `BC/FeatureAnalysis/Classifier.py` and add:
```python
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
```
Place this after the existing `import` block (after `pandas`, `numpy`, etc.).

- [ ] **Step 2: Add `_SaveShap()` to the base Classifier class**

Find the base `Classifier` class's `Save()` method (around line 130). Insert the following method directly after it:

```python
def _SaveShap(self, store_folder, explainer_type='linear'):
    """Compute and save SHAP values for training data.

    Saves {ClassName}_shap.csv (samples x features, signed SHAP values).
    explainer_type: 'linear' | 'tree'
    Falls back silently on any error (e.g. non-linear SVM kernel).
    """
    if not _SHAP_AVAILABLE:
        return
    try:
        X = self._x
        feature_names = self._data_container.GetFeatureName()
        case_names = self._data_container.GetCaseName()

        if explainer_type == 'linear':
            explainer = shap.LinearExplainer(self.model, X)
            shap_values = explainer.shap_values(X)
        elif explainer_type == 'tree':
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            # sklearn tree models return list [class0, class1] for binary
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
        else:
            return

        shap_path = os.path.join(store_folder, self.GetName() + '_shap.csv')
        df = pd.DataFrame(shap_values, index=case_names, columns=feature_names)
        df.to_csv(shap_path)
    except Exception as e:
        self.logger.warning('SHAP computation skipped for {}: {}'.format(
            self.GetName(), str(e)))
```

- [ ] **Step 3: Verify syntax**

```
cd C:\Users\SunsServer\Project\FAE\FAE
python -m py_compile BC\FeatureAnalysis\Classifier.py && echo OK
```
Expected: `OK`

---

## Task 3: Call `_SaveShap()` from Linear Classifiers

**Files:**
- Modify: `BC/FeatureAnalysis/Classifier.py` — SVM, LR, LRLasso, LDA `Save()` methods

- [ ] **Step 1: SVM.Save() — add `_SaveShap` call**

Find `SVM.Save()` (around line 190). At the end of the method, before or after `super(SVM, self).Save(store_folder)`, add:
```python
        self._SaveShap(store_folder, explainer_type='linear')
```
The complete tail of `SVM.Save()` should look like:
```python
        # ... existing intercept save block ...
        self._SaveShap(store_folder, explainer_type='linear')
        super(SVM, self).Save(store_folder)
```

- [ ] **Step 2: LR.Save() — add `_SaveShap` call**

Find `LR.Save()`. At the end (before/after `super().Save()`), add:
```python
        self._SaveShap(store_folder, explainer_type='linear')
```

- [ ] **Step 3: LRLasso.Save() — add `_SaveShap` call**

Find `LRLasso.Save()`. Same pattern:
```python
        self._SaveShap(store_folder, explainer_type='linear')
```

- [ ] **Step 4: LDA.Save() — add `_SaveShap` call**

Find `LDA.Save()`. Same pattern:
```python
        self._SaveShap(store_folder, explainer_type='linear')
```

- [ ] **Step 5: Verify syntax**

```
python -m py_compile BC\FeatureAnalysis\Classifier.py && echo OK
```
Expected: `OK`

---

## Task 4: Call `_SaveShap()` from Tree Classifiers

**Files:**
- Modify: `BC/FeatureAnalysis/Classifier.py` — RandomForest, DecisionTree, AdaBoost `Save()` methods

Tree classifiers currently do **not** save any coef file. Each one calls `super().Save(store_folder)` which pickles the model. We add `_SaveShap` before that call.

- [ ] **Step 1: RandomForest.Save() — add `_SaveShap` call**

Find `RandomForest.Save()`. It currently just calls `super().Save()`. Replace it with:
```python
    def Save(self, store_folder):
        self._SaveShap(store_folder, explainer_type='tree')
        super(RandomForest, self).Save(store_folder)
```

- [ ] **Step 2: DecisionTree.Save() — add `_SaveShap` call**

Find `DecisionTree.Save()`. Same pattern:
```python
    def Save(self, store_folder):
        self._SaveShap(store_folder, explainer_type='tree')
        super(DecisionTree, self).Save(store_folder)
```

- [ ] **Step 3: AdaBoost.Save() — add `_SaveShap` call**

Find `AdaBoost.Save()`. Same pattern:
```python
    def Save(self, store_folder):
        self._SaveShap(store_folder, explainer_type='tree')
        super(AdaBoost, self).Save(store_folder)
```

- [ ] **Step 4: Verify syntax**

```
python -m py_compile BC\FeatureAnalysis\Classifier.py && echo OK
```
Expected: `OK`

- [ ] **Step 5: Quick smoke test for SHAP computation**

```python
# Run in python REPL from project root
import sys; sys.path.insert(0, '.')
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
X = np.random.randn(50, 5)
y = (X[:, 0] > 0).astype(int)
model = LogisticRegression().fit(X, y)
explainer = shap.LinearExplainer(model, X)
vals = explainer.shap_values(X)
print(vals.shape)  # Expected: (50, 5)
```

---

## Task 5: Add `SHAPBarPlot()` to FeatureSort.py

**Files:**
- Modify: `BC/Visualization/FeatureSort.py`

**Sorting & coloring design (updated 2026-04-03):**
- Sort by **signed mean SHAP** (descending): most-positive at top → most-negative at bottom.
- Color via `RdBu_r` colormap mapped to the signed range → natural red-to-blue gradient, no alternating.
- Bar length = `|mean SHAP|`. X-axis shows absolute magnitude; sign is carried by color.
- A colorbar on the right edge shows the value scale (red = positive, blue = negative).

- [ ] **Step 1: Add `SHAPBarPlot()` at the end of FeatureSort.py**

```python
def SHAPBarPlot(shap_df, max_num=20, is_show=False, fig=None):
    """Horizontal bar chart sorted by signed mean SHAP with RdBu_r gradient coloring.

    Features are ordered positive-max (top) → negative-min (bottom).
    Bar length = |mean SHAP|; color encodes sign and magnitude via RdBu_r colormap.

    Args:
        shap_df: pd.DataFrame (n_samples, n_features), signed SHAP values.
        max_num: Top N features to display (selected by |mean SHAP|, then sorted by sign).
        is_show: Whether to call fig.show().
        fig: matplotlib Figure object.

    Returns:
        matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    if fig is None:
        fig = plt.figure()

    mean_shap = shap_df.mean(axis=0)   # signed mean per feature

    # Step 1: Select top-N by |mean SHAP|
    if max_num > 0:
        top_idx = mean_shap.abs().nlargest(max_num).index
    else:
        top_idx = mean_shap.index

    # Step 2: Sort selected features by signed value descending
    #         → positive max at top, negative min at bottom (for barh: bottom = index 0)
    sorted_series = mean_shap[top_idx].sort_values(ascending=True)  # ascending=True for barh bottom-to-top
    labels = list(sorted_series.index)
    signed_vals = sorted_series.values                  # signed, for coloring
    bar_lengths = np.abs(signed_vals)                   # always positive, for bar width

    # Step 3: Map signed values → RdBu_r colors
    v_max = np.abs(signed_vals).max() if len(signed_vals) > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max)
    cmap = cm.get_cmap('RdBu_r')
    colors = [cmap(norm(v)) for v in signed_vals]

    # Step 4: Draw
    fig.clear()
    ax = fig.add_axes([0.42, 0.08, 0.50, 0.84])

    ax.barh(range(len(labels)), bar_lengths, color=colors, height=0.65,
            edgecolor='none')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=9, color='#555555')
    ax.set_title('Feature Contribution (SHAP)', fontsize=10, pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

    # Step 5: Colorbar (sign legend)
    cax = fig.add_axes([0.94, 0.08, 0.025, 0.84])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('mean SHAP', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if is_show:
        fig.show()
    return ax
```

- [ ] **Step 2: Verify syntax**

```
python -m py_compile BC\Visualization\FeatureSort.py && echo OK
```
Expected: `OK`

---

## Task 6: Add `SHAPBeeswarmPlot()` to FeatureSort.py

**Files:**
- Modify: `BC/Visualization/FeatureSort.py`

The beeswarm shows each training sample as a dot. X-axis = SHAP value. Y-axis = feature (jittered). Color = normalized feature value (blue=low, red=high via `RdBu_r` colormap).

- [ ] **Step 1: Add `SHAPBeeswarmPlot()` at the end of FeatureSort.py, after `SHAPBarPlot()`**

```python
def SHAPBeeswarmPlot(shap_df, feature_df=None, max_num=20, is_show=False, fig=None):
    """Beeswarm plot: each dot = one training sample, colored by feature value.

    Args:
        shap_df: pd.DataFrame (n_samples, n_features), signed SHAP values.
        feature_df: pd.DataFrame (n_samples, n_features), original feature values
                    used for dot coloring. If None, coloring is skipped (gray dots).
        max_num: Number of top features to show.
        is_show: Whether to call fig.show().
        fig: matplotlib Figure object.

    Returns:
        matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.colorbar import ColorbarBase

    if fig is None:
        fig = plt.figure()

    mean_abs = shap_df.abs().mean(axis=0)
    sorted_idx = mean_abs.sort_values(ascending=False).index
    if max_num > 0:
        sorted_idx = sorted_idx[:max_num]
    sorted_idx = sorted_idx[::-1]   # bottom-to-top

    fig.clear()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.45, right=0.88, top=0.92, bottom=0.08)

    cmap = cm.get_cmap('RdBu_r')
    rng = np.random.default_rng(42)

    for y_pos, feat in enumerate(sorted_idx):
        shap_vals = shap_df[feat].values
        n = len(shap_vals)

        if feature_df is not None and feat in feature_df.columns:
            feat_vals = feature_df[feat].values.astype(float)
            feat_min, feat_max = feat_vals.min(), feat_vals.max()
            norm_vals = (feat_vals - feat_min) / (feat_max - feat_min + 1e-8)
            dot_colors = cmap(norm_vals)
        else:
            dot_colors = np.full((n, 4), [0.5, 0.5, 0.5, 0.7])

        jitter = rng.uniform(-0.25, 0.25, size=n)
        ax.scatter(shap_vals, y_pos + jitter, c=dot_colors,
                   s=12, alpha=0.7, linewidths=0)

    ax.axvline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(list(sorted_idx), fontsize=9)
    ax.set_xlabel('SHAP value', fontsize=9, color='#555555')
    ax.set_title('Feature Contribution (SHAP Beeswarm)', fontsize=10, pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

    # Colorbar for feature value
    if feature_df is not None:
        cax = fig.add_axes([0.90, 0.2, 0.02, 0.6])
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('Feature value', fontsize=8)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Low', 'High'])

    if is_show:
        fig.show()
    return ax
```

- [ ] **Step 2: Verify syntax**

```
python -m py_compile BC\Visualization\FeatureSort.py && echo OK
```
Expected: `OK`

---

## Task 7: Update VisualizationForm — UI Controls + UpdateContribution Logic

**Files:**
- Modify: `BC/GUI/VisualizationForm.py`

**Changes:**
1. Add two radio buttons ("Bar" / "Beeswarm") programmatically into `verticalLayout_5` (which already contains `canvasFeature`).
2. Update `UpdateContribution()` to: check for `_shap.csv` first → render the selected plot type; fall back to existing logic if no shap file.

- [ ] **Step 1: Add imports at the top of VisualizationForm.py**

Find the import block. Add:
```python
from BC.Visualization.FeatureSort import GeneralFeatureSort, SHAPBarPlot, SHAPBeeswarmPlot
```
(Replace any existing `from BC.Visualization.FeatureSort import GeneralFeatureSort` line.)

Also ensure `Qt` is imported (already added earlier for `setCheckState`):
```python
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QRadioButton,
                                QButtonGroup, QLabel, *) # merge with existing *
```
Since the file already has `from PySide6.QtWidgets import *`, just ensure `Qt` is in the QtCore import. The widgets are already available via `*`.

- [ ] **Step 2: Add plot-type radio buttons in `__init__`**

Find the `__init__` method. After the line `self.spinContributeFeatureNumber.valueChanged.connect(self.UpdateContribution)` (last contribution signal connection), insert:

```python
        # --- SHAP plot type selector (added programmatically) ---
        self._shap_plot_widget = QWidget()
        _shap_layout = QHBoxLayout(self._shap_plot_widget)
        _shap_layout.setContentsMargins(4, 2, 4, 2)
        _shap_label = QLabel("Plot type:")
        _shap_label.setStyleSheet("font-size: 11px; color: #555;")
        self._radioSHAPBar = QRadioButton("Bar")
        self._radioSHAPBeeswarm = QRadioButton("Beeswarm")
        self._radioSHAPBar.setChecked(True)
        self._radioSHAPBar.setToolTip("Mean |SHAP| bar chart")
        self._radioSHAPBeeswarm.setToolTip("Per-sample SHAP scatter plot")
        _shap_layout.addWidget(_shap_label)
        _shap_layout.addWidget(self._radioSHAPBar)
        _shap_layout.addWidget(self._radioSHAPBeeswarm)
        _shap_layout.addStretch()
        # Insert above canvasFeature in verticalLayout_5
        _parent_layout = self.canvasFeature.parent().layout()
        _idx = _parent_layout.indexOf(self.canvasFeature)
        _parent_layout.insertWidget(_idx, self._shap_plot_widget)
        self._radioSHAPBar.toggled.connect(self.UpdateContribution)
        self._radioSHAPBeeswarm.toggled.connect(self.UpdateContribution)
```

- [ ] **Step 3: Replace `UpdateContribution()` method**

Find the entire `UpdateContribution` method and replace it with the new version below. The logic is:
1. Try to load `{Classifier}_shap.csv`
2. If found → render `SHAPBarPlot` or `SHAPBeeswarmPlot` based on radio selection
3. If not found → fall through to existing coef/selector logic (unchanged)

```python
    def UpdateContribution(self):
        if (not self.__is_ui_ready) or self.__is_clear:
            return

        try:
            pipeline_name = self._fae.GetStoreName(
                self.comboContributionNormalizor.currentText(),
                self.comboContributionDimension.currentText(),
                self.comboContributionFeatureSelector.currentText(),
                str(self.spinContributeFeatureNumber.value()),
                self.comboContributionClassifier.currentText())
            norm_folder, dr_folder, fs_folder, cls_folder = self._fae.SplitFolder(
                pipeline_name, self._root_folder)

            max_num = self.spinContributeFeatureNumber.value()

            # ── SHAP path (preferred) ──────────────────────────────────────
            shap_name = self.comboContributionClassifier.currentText() + '_shap.csv'
            shap_file_path = os.path.join(cls_folder, shap_name)

            if os.path.exists(shap_file_path):
                self.radioContributionClassifier.setEnabled(False)
                self.radioContributionFeatureSelector.setEnabled(False)
                self._shap_plot_widget.setVisible(True)

                shap_df = pd.read_csv(shap_file_path, index_col=0)

                if self._radioSHAPBeeswarm.isChecked():
                    # Attempt to load original feature values for coloring
                    # The training data CSV lives next to the shap file; try to
                    # reconstruct from the DataContainer stored in _fae.
                    feature_df = None
                    try:
                        feature_df = pd.DataFrame(
                            self._fae.GetTrainDataContainer().GetArray(),
                            columns=self._fae.GetTrainDataContainer().GetFeatureName())
                    except Exception:
                        pass
                    SHAPBeeswarmPlot(shap_df, feature_df=feature_df,
                                     max_num=max_num, is_show=False,
                                     fig=self.canvasFeature.getFigure())
                else:
                    SHAPBarPlot(shap_df, max_num=max_num, is_show=False,
                                fig=self.canvasFeature.getFigure())

            # ── Coef fallback (linear classifiers without SHAP) ───────────
            else:
                self._shap_plot_widget.setVisible(False)
                coef_name = self.comboContributionClassifier.currentText() + '_coef.csv'
                coef_file_path = os.path.join(cls_folder, coef_name)
                sort_name = (self.comboContributionFeatureSelector.currentText()
                             + '_sort.csv')
                sort_file_path = os.path.join(fs_folder, sort_name)

                if os.path.exists(coef_file_path):
                    self.radioContributionClassifier.setEnabled(True)
                    self.radioContributionFeatureSelector.setEnabled(False)
                    self.radioContributionClassifier.setChecked(True)
                    df = pd.read_csv(coef_file_path, index_col=0)
                    value = list(np.abs(df.iloc[:, 0]))
                    processed_feature_name = list(df.index)
                    original_value = list(df.iloc[:, 0])
                    for index in range(len(original_value)):
                        suffix = ' P' if original_value[index] > 0 else ' N'
                        processed_feature_name[index] += suffix
                    GeneralFeatureSort(processed_feature_name, value,
                                       is_show=False,
                                       fig=self.canvasFeature.getFigure())
                else:
                    self.radioContributionClassifier.setEnabled(False)
                    self.radioContributionFeatureSelector.setEnabled(True)
                    self.radioContributionFeatureSelector.setChecked(True)
                    df = pd.read_csv(sort_file_path, index_col=0)
                    value = list(df.iloc[:, 0])
                    sort_by = df.columns.values[0]
                    reverse = sort_by in ('F', 'weight')
                    processed_feature_name = list(df.index)
                    original_value = list(df.iloc[:, 0])
                    for index in range(len(original_value)):
                        fv = original_value[index]
                        processed_feature_name[index] += (
                            ' ' + str(fv) if isinstance(fv, int)
                            else ' %.2f' % fv)
                    GeneralFeatureSort(processed_feature_name, value,
                                       max_num=max_num, is_show=False,
                                       fig=self.canvasFeature.getFigure(),
                                       reverse=reverse)

            self.canvasFeature.draw()
        except Exception as e:
            content = 'UpdateContribution failed'
            QMessageBox.about(self, content, e.__str__())
```

**Note on `_fae.GetTrainDataContainer()`:** This call is speculative — check whether `self._fae` exposes a `GetTrainDataContainer()` method. If not, simply set `feature_df = None` (beeswarm will use gray dots, still fully functional).

- [ ] **Step 4: Verify syntax**

```
python -m py_compile BC\GUI\VisualizationForm.py && echo OK
```
Expected: `OK`

---

## Task 8: Check `_fae.GetTrainDataContainer()` Availability

**Files:**
- Possibly modify: `BC/GUI/VisualizationForm.py` (one line)

- [ ] **Step 1: Check if _fae exposes training data**

Search the project:
```
grep -rn "GetTrainDataContainer\|GetDataContainer\|train_data_container" BC\FeatureAnalysis\Pipelines.py BC\GUI\VisualizationForm.py
```

- If `GetTrainDataContainer()` exists → no change needed, Task 7 Step 3 already handles it.
- If it does **not** exist → find the line in `UpdateContribution`:
  ```python
  feature_df = pd.DataFrame(
      self._fae.GetTrainDataContainer().GetArray(), ...)
  ```
  Replace it with just:
  ```python
  pass  # feature values unavailable; beeswarm uses gray dots
  ```
  And remove the `try/except` wrapper (just keep `feature_df = None`).

- [ ] **Step 2: Verify syntax after any changes**

```
python -m py_compile BC\GUI\VisualizationForm.py && echo OK
```

---

## Task 9: End-to-End Manual Verification

No automated tests exist in this project. Verify manually:

- [ ] **Step 1: Re-run a BC pipeline with a linear classifier (e.g. LR)**

After training completes, check the output folder for:
```
<pipeline_folder>/LR_shap.csv
```
Open it — it should be a CSV with rows=case names, columns=feature names, values being signed floats.

- [ ] **Step 2: Open BC Visualization → Feature Contribution tab**

- Select the LR pipeline.
- Verify: the "Bar" / "Beeswarm" radio buttons appear above the chart.
- Verify: Bar chart shows red/blue bars (not gray).
- Switch to "Beeswarm" — verify scatter dots appear.

- [ ] **Step 3: Test with a tree classifier (e.g. RF)**

Same as Step 2 but with a Random Forest pipeline. Confirm `RF_shap.csv` is generated and visualized.

- [ ] **Step 4: Test fallback with a non-supported classifier (e.g. AE/MLP)**

Verify: no `AE_shap.csv` is generated. In Visualization, the chart falls back to the existing coef/selector display (gray bars, P/N suffix).

- [ ] **Step 5: Verify SVM with non-linear kernel falls back gracefully**

If SVM is configured with `kernel='rbf'`, `LinearExplainer` will raise an exception. Verify: the exception is caught silently (logged as warning), no `SVM_shap.csv` is written, and visualization falls back to the coef-based display.

---

## Notes

- **`_shap.csv` file size**: For large datasets (>1000 samples, >100 features), the CSV will be multi-MB. This is acceptable for research use.
- **AdaBoost**: `shap.TreeExplainer` supports `AdaBoostClassifier` with `algorithm='SAMME.R'` (default). If it raises, the exception is caught and falls back gracefully.
- **SVM with linear kernel**: `LinearExplainer` requires `coef_` attribute. SVM with linear kernel has it; other kernels do not. The `try/except` in `_SaveShap` handles this transparently.
- **Beeswarm gray dots**: When `feature_df=None`, dots are plotted in gray. The SHAP direction (sign, magnitude) is still fully visible.
