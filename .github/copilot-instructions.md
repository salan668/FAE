# FAE — Copilot Instructions

## What this project is

FAE (FeAture Explorer) is a **PySide6 desktop GUI application** for radiomics and medical imaging analysis. It is not a library or web app. There are no automated tests.

## Running the application

```bash
python MainFrameCall.py          # standard entry
python MainFrameCall_opt.py      # compatibility mode (disables native dialogs)
```

## Environment setup

```bash
conda create -n fae python=3.11
conda activate fae
pip install numpy scipy matplotlib pandas pillow pyside6 pyqtgraph pyradiomics \
    seaborn reportlab imbalanced-learn pdfdocument statsmodels lifelines \
    pyinstaller scikit-learn shap scikit-survival trimesh pingouin
```

## Building (PyInstaller)

```bash
pyinstaller --clean --noconfirm MainFrameCall.spec
```

`MainFrameCall.spec` builds two executables: `fae` (from `MainFrameCall.py`) and `fae_opt` (from `MainFrameCall_opt.py`). Do not use `Release.bat` — it is outdated (targets v0.6.6).

## Architecture

### Startup chain

```
MainFrameCall*.py
  └── HomeUI\HomePageForm.py      ← navigation hub, owns all submodule instances
        ├── Feature\GUI\FeatureExtractionForm.py
        ├── Feature\GUI\FeatureMergeForm.py
        ├── Feature\GUI\IccEstimationForm.py
        ├── BC\GUI\PrepareConnection
        ├── BC\GUI\ProcessConnection
        ├── BC\GUI\VisualizationConnection
        ├── BC\GUI\ModelPredictionForm
        ├── SA\GUI\ProcessForm
        ├── SA\GUI\VisualizationForm
        └── Plugin\PluginManager
```

`HomePageForm` instantiates all submodule windows at startup (not lazy-loaded). Window switching uses `show()` / `hide()`. Submodules signal return-to-home via `close_signal`.

### Three core domains

- **`Feature\`** — radiomics feature extraction, series/ROI file matching, ICC estimation. Consumes `SimpleITK` + `pyradiomics`. Produces feature CSV tables.
- **`BC\`** — binary classification pipeline: DataContainer → DataBalance → Normalizer → DimensionReduction → FeatureSelector → Classifier → CrossValidation → Visualization/Report. Key orchestrator: `BC\FeatureAnalysis\Pipelines.py`.
- **`SA\`** — survival analysis pipeline parallel to BC. Key orchestrator: `SA\PipelineManager.py`.

**Feature produces feature tables; BC and SA consume them.**

### Plugin system

Plugins live under `Plugin\`. Each plugin subdirectory needs a `config.json` with `name` and `path`. Plugins are launched via `os.system(...)` — there is no in-process plugin API.

## Key conventions

### PySide6 API rules (migration complete as of commit `2222cad`)

- Signals: `Signal` from `PySide6.QtCore`, **not** `pyqtSignal`
- Dialog exec: `dialog.exec()`, **not** `exec_()`
- All Qt enums use namespace form: `Qt.AlignmentFlag.AlignCenter`, `QSizePolicy.Policy.Expanding`, `QFont.Weight.Bold`, `Qt.CheckState.Checked`, `QFileDialog.FileMode.Directory`, `QFileDialog.Option.ShowDirsOnly`
- `QFileDialog` static methods take **positional** arguments only — `directory=` and `filter=` keyword args are not supported
- `QApplication` creation: `QApplication.instance() or QApplication(sys.argv)`
- matplotlib backend: `matplotlib.backends.backend_qtagg` (not `backend_qt5agg`)

### UI file pattern

Each GUI panel follows a three-file pattern:

| File | Role |
|------|------|
| `Xxx.ui` | Qt Designer source |
| `Xxx.py` | Generated Python (checked in, do not hand-edit structure) |
| `XxxForm.py` | Business logic and signal/slot wiring |

Generated `.py` files are checked in for runtime convenience. If you modify a `.ui` file, regenerate its `.py` using a PySide6-compatible `pyside6-uic` and verify the output imports remain `from PySide6 import QtCore, QtGui, QtWidgets`.

### SHAP output convention (BC, v0.8.0+)

Each BC classifier's `Save()` calls `_SaveShap()`, writing signed SHAP values to `{Name}_shap.csv` alongside the model. `BC\Visualization\FeatureSort.py` reads this file for Bar and Beeswarm plots. When the SHAP file is absent, the UI falls back to selector-rank visualization.

### SHAP layout constraint

The Feature Contribution panel's parent layout is `verticalLayout_5` (a `QVBoxLayout`). Do **not** use `canvasFeature.parent().layout()` — that resolves to `gridLayout_4`.

### Notable directory gotcha

`src\` contains **image assets** (PNG icons), not Python source code.

## Version management

Version is defined in `HomeUI\VersionConstant.py`:

```python
MAJOR = 0; MINOR = 8; PATCH = 0
VERSION = '0.8.0'
ACCEPT_VERSION = ['0.8.0', '0.7.0']
```

Accepted versions control whether saved pipeline files from older FAE releases can be loaded.

## Dependency source of truth

Dependency information is scattered. Cross-reference all three when setting up:

1. `install.bat` — canonical conda/pip install commands
2. `README.md` — package list (may be ahead of `install.bat`)
3. `MainFrameCall.spec` — what PyInstaller actually bundles (includes `scipy`, `pandas`, `matplotlib`, `imblearn`, `radiomics` data trees; excludes `torch`, `IPython`, `tkinter`)
