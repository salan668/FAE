# PySide6 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repository's PyQt5 dependency with PySide6 across runtime code, generated UI modules, plotting adapters, and packaging/docs so the desktop app can move toward broader distribution and commercial use.

**Architecture:** Migrate the codebase in four layers: core Qt runtime APIs, generated UI Python modules, plotting/widget integration, and dependency/build metadata. Verification relies on repository-wide search plus Python import smoke checks because the repo does not currently include an automated test suite.

**Tech Stack:** Python, PySide6, matplotlib, pyqtgraph, PyInstaller, ripgrep, existing `.ui`-generated Python modules

---

### Task 1: Baseline the Qt migration surface

**Files:**
- Modify: `C:\Users\SunsServer\.copilot\session-state\90092052-91f0-4dc2-93e8-31c82db13b73\plan.md`
- Reference: `C:\Users\SunsServer\Project\FAE\FAE\docs\superpowers\specs\2026-04-02-pyside6-migration-design.md`
- Reference: `C:\Users\SunsServer\Project\FAE\FAE\MainFrameCall.py`
- Reference: `C:\Users\SunsServer\Project\FAE\FAE\MainFrameCall_opt.py`
- Reference: `C:\Users\SunsServer\Project\FAE\FAE\HomeUI\HomePageForm.py`
- Reference: `C:\Users\SunsServer\Project\FAE\FAE\MatplotlibWidget.py`
- Reference: `C:\Users\SunsServer\Project\FAE\FAE\SA\GUI\MatplotlibWidget.py`

- [ ] **Step 1: Run a failing inventory search**

Run:

```powershell
rg -n "PyQt5|pyqtSignal|exec_\(|backend_qt4agg|backend_qt5agg" C:\Users\SunsServer\Project\FAE\FAE -g "*.py"
```

Expected: multiple matches across entrypoints, forms, generated UI files, and matplotlib adapters.

- [ ] **Step 2: Record the migration order in the session plan**

Add or confirm this summary in `C:\Users\SunsServer\.copilot\session-state\90092052-91f0-4dc2-93e8-31c82db13b73\plan.md`:

```markdown
- Migrate entrypoints and shared Qt helpers first.
- Migrate generated UI Python files second.
- Migrate form/controller files third.
- Repair matplotlib/pyqtgraph Qt adapters fourth.
- Update docs and packaging last.
```

- [ ] **Step 3: Re-run the search and save the output in the working notes**

Run:

```powershell
rg -n "PyQt5|pyqtSignal|exec_\(|backend_qt4agg|backend_qt5agg" C:\Users\SunsServer\Project\FAE\FAE -g "*.py" > C:\Users\SunsServer\.copilot\session-state\90092052-91f0-4dc2-93e8-31c82db13b73\files\pyside6-migration-surface.txt
```

Expected: a text artifact listing the current migration surface for later verification.

- [ ] **Step 4: Commit the planning artifact**

```bash
git add docs/superpowers/plans/2026-04-02-pyside6-migration.md
git commit -m "chore: add PySide6 migration plan

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Migrate the entrypoints and homepage runtime code

**Files:**
- Modify: `MainFrameCall.py`
- Modify: `MainFrameCall_opt.py`
- Modify: `HomeUI\HomePageForm.py`

- [ ] **Step 1: Run a targeted failing search for the runtime layer**

Run:

```powershell
rg -n "PyQt5|pyqtSignal|exec_\(" C:\Users\SunsServer\Project\FAE\FAE\MainFrameCall.py C:\Users\SunsServer\Project\FAE\FAE\MainFrameCall_opt.py C:\Users\SunsServer\Project\FAE\FAE\HomeUI\HomePageForm.py
```

Expected: PyQt5 imports, `exec_()`, and a direct `PyQt5.QtCore.Qt.KeepAspectRatio` reference.

- [ ] **Step 2: Replace PyQt5 imports and `exec_()` in `MainFrameCall.py`**

Update the import and shutdown call to this shape:

```python
from PySide6.QtWidgets import QApplication
from HomeUI.HomePageForm import HomePageForm

sys.exit(app.exec())
```

- [ ] **Step 3: Replace PyQt5 imports and `exec_()` in `MainFrameCall_opt.py`**

Update the import and runtime call to this shape:

```python
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from HomeUI.HomePageForm import HomePageForm

QApplication.setAttribute(Qt.AA_DontUseNativeDialogs)
sys.exit(app.exec())
```

- [ ] **Step 4: Replace Qt imports and enum usage in `HomeUI\HomePageForm.py`**

Refactor the imports and image scaling call to this shape:

```python
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QDialog

pixmap = pixmap.scaled(self.labelPluginFigure.size(), Qt.KeepAspectRatio)
```

Also replace the local `app.exec_()` call at the bottom with `app.exec()`.

- [ ] **Step 5: Run a smoke import check for the runtime layer**

Run:

```powershell
python -c "from MainFrameCall import HomePageForm; print('runtime-import-ok')"
```

Expected: `runtime-import-ok`

- [ ] **Step 6: Commit the runtime-layer migration**

```bash
git add MainFrameCall.py MainFrameCall_opt.py HomeUI/HomePageForm.py
git commit -m "refactor: migrate runtime entrypoints to PySide6

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Migrate checked-in generated UI modules

**Files:**
- Modify: `HomeUI\HomePage.py`
- Modify: `Feature\GUI\FeatureExtraction.py`
- Modify: `Feature\GUI\FeatureMerge.py`
- Modify: `Feature\GUI\IccEstimation.py`
- Modify: `BC\GUI\Prepare.py`
- Modify: `BC\GUI\Process.py`
- Modify: `BC\GUI\Visualization.py`
- Modify: `BC\GUI\ModelPrediction.py`
- Modify: `BC\GUI\FeatureExtraction.py`
- Modify: `BC\GUI\FeatureExtraction2.py`
- Modify: `SA\GUI\Prepare.py`
- Modify: `SA\GUI\Process.py`
- Modify: `SA\GUI\Visualization.py`

- [ ] **Step 1: Run a failing search limited to generated UI modules**

Run:

```powershell
rg -n "^from PyQt5 import QtCore, QtGui, QtWidgets$|Created by: PyQt5 UI code generator" C:\Users\SunsServer\Project\FAE\FAE\HomeUI C:\Users\SunsServer\Project\FAE\FAE\Feature\GUI C:\Users\SunsServer\Project\FAE\FAE\BC\GUI C:\Users\SunsServer\Project\FAE\FAE\SA\GUI -g "*.py"
```

Expected: generated UI files still referencing PyQt5.

- [ ] **Step 2: Update each generated module import line**

Replace the import line in each generated file with:

```python
from PySide6 import QtCore, QtGui, QtWidgets
```

Preserve the rest of the generated layout code unless a PySide6 enum or method syntax issue appears.

- [ ] **Step 3: Update the generator header comments where they claim PyQt5**

Normalize the header to this shape:

```python
# Form implementation generated from reading ui file '...'
# Updated for PySide6 compatibility.
```

This avoids leaving misleading PyQt5-specific provenance comments in files that are now maintained by hand.

- [ ] **Step 4: Run a search to verify the generated files no longer import PyQt5**

Run:

```powershell
rg -n "from PyQt5 import QtCore, QtGui, QtWidgets|Created by: PyQt5 UI code generator" C:\Users\SunsServer\Project\FAE\FAE\HomeUI C:\Users\SunsServer\Project\FAE\FAE\Feature\GUI C:\Users\SunsServer\Project\FAE\FAE\BC\GUI C:\Users\SunsServer\Project\FAE\FAE\SA\GUI -g "*.py"
```

Expected: no matches.

- [ ] **Step 5: Commit the generated-UI migration**

```bash
git add HomeUI/HomePage.py Feature/GUI/*.py BC/GUI/*.py SA/GUI/*.py
git commit -m "refactor: migrate generated UI modules to PySide6

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Migrate form and controller modules to PySide6 APIs

**Files:**
- Modify: `Feature\GUI\FeatureExtractionForm.py`
- Modify: `Feature\GUI\FeatureExtractionConfig.py`
- Modify: `Feature\GUI\FeatureMergeForm.py`
- Modify: `Feature\GUI\IccEstimationForm.py`
- Modify: `BC\GUI\PrepareForm.py`
- Modify: `BC\GUI\ProcessForm.py`
- Modify: `BC\GUI\VisualizationForm.py`
- Modify: `BC\GUI\ModelPredictionForm.py`
- Modify: `SA\GUI\ProcessForm.py`
- Modify: `SA\GUI\VisualizationForm.py`

- [ ] **Step 1: Run a failing search for Qt runtime APIs in form modules**

Run:

```powershell
rg -n "from PyQt5|import PyQt5|pyqtSignal|QtCore\.pyqtSignal|exec_\(" C:\Users\SunsServer\Project\FAE\FAE\Feature\GUI C:\Users\SunsServer\Project\FAE\FAE\BC\GUI C:\Users\SunsServer\Project\FAE\FAE\SA\GUI -g "*Form.py" -g "FeatureExtractionConfig.py"
```

Expected: multiple matches for PyQt5 imports, signal declarations, and modal dialog execution.

- [ ] **Step 2: Replace signal declarations with `Signal`**

Use this pattern anywhere a class currently defines PyQt signals:

```python
from PySide6.QtCore import QThread, Signal

progress_signal = Signal(int)
text_signal = Signal(str)
finish_signal = Signal(bool)
close_signal = Signal(bool)
```

If a file currently imports `QtCore` only for `QtCore.pyqtSignal`, remove that dependency unless it is still needed elsewhere.

- [ ] **Step 3: Replace `exec_()` calls with `exec()`**

Update dialog code to this shape:

```python
dlg = QFileDialog(self)
if dlg.exec():
    selected = dlg.selectedFiles()
```

Apply the same pattern to every modal dialog check in:

- `Feature\GUI\FeatureExtractionForm.py`
- `Feature\GUI\FeatureExtractionConfig.py`
- `BC\GUI\ProcessForm.py`
- `BC\GUI\VisualizationForm.py`
- `BC\GUI\ModelPredictionForm.py`
- `SA\GUI\ProcessForm.py`
- `SA\GUI\VisualizationForm.py`

- [ ] **Step 4: Replace wildcard imports where the PySide6 move makes them ambiguous**

Refactor broad imports to explicit imports in this shape:

```python
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from PySide6.QtCore import QObject, QThread, Signal
```

Do not refactor unrelated logic; only make import changes needed to keep the modules readable and PySide6-compatible.

- [ ] **Step 5: Run an import smoke check across the migrated form layer**

Run:

```powershell
python -c "from Feature.GUI.FeatureExtractionForm import FeatureExtractionForm; from BC.GUI.ProcessForm import ProcessConnection; from SA.GUI.ProcessForm import ProcessForm; print('forms-import-ok')"
```

Expected: `forms-import-ok`

- [ ] **Step 6: Commit the form-layer migration**

```bash
git add Feature/GUI/*Form.py Feature/GUI/FeatureExtractionConfig.py BC/GUI/*Form.py SA/GUI/*Form.py
git commit -m "refactor: migrate Qt form controllers to PySide6

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Repair shared matplotlib and pyqtgraph Qt adapters

**Files:**
- Modify: `MatplotlibWidget.py`
- Modify: `SA\GUI\MatplotlibWidget.py`

- [ ] **Step 1: Run a failing search for old Qt backend usage**

Run:

```powershell
rg -n "backend_qt4agg|backend_qt5agg|USE_PYSIDE|USE_PYQT5|PyQt5" C:\Users\SunsServer\Project\FAE\FAE\MatplotlibWidget.py C:\Users\SunsServer\Project\FAE\FAE\SA\GUI\MatplotlibWidget.py
```

Expected: Qt4/Qt5 backend imports and pyqtgraph compatibility flags.

- [ ] **Step 2: Simplify `MatplotlibWidget.py` to a direct PySide6-backed widget**

Refactor to this structure:

```python
from PySide6.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None, size=(5.0, 4.0), dpi=100):
        super().__init__(parent)
        self.fig = Figure(size, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
```

- [ ] **Step 3: Simplify `SA\GUI\MatplotlibWidget.py` the same way**

Refactor away from `pyqtgraph.Qt` and obsolete backend switching to this structure:

```python
from PySide6.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None, size=(5.0, 4.0), dpi=100):
        super().__init__(parent)
        self.fig = Figure(size, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
```

Only reintroduce `NavigationToolbar2QT` if a caller in the repo actually uses it.

- [ ] **Step 4: Run a targeted import smoke check for the plotting layer**

Run:

```powershell
python -c "from MatplotlibWidget import MatplotlibWidget; from SA.GUI.MatplotlibWidget import MatplotlibWidget as SAMatplotlibWidget; print('matplotlib-widget-import-ok')"
```

Expected: `matplotlib-widget-import-ok`

- [ ] **Step 5: Commit the plotting adapter migration**

```bash
git add MatplotlibWidget.py SA/GUI/MatplotlibWidget.py
git commit -m "refactor: update matplotlib Qt adapters for PySide6

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Update dependency, packaging, and contributor guidance

**Files:**
- Modify: `install.bat`
- Modify: `README.md`
- Modify: `MainFrameCall.spec`
- Modify: `AI_Relate\FAE_Project_Memory.md`

- [ ] **Step 1: Run a failing search for package metadata that still references PyQt5**

Run:

```powershell
rg -n "PyQt5|pyqt5|QtWebEngine|pyuic|backend_qt5agg" C:\Users\SunsServer\Project\FAE\FAE\README.md C:\Users\SunsServer\Project\FAE\FAE\install.bat C:\Users\SunsServer\Project\FAE\FAE\MainFrameCall.spec C:\Users\SunsServer\Project\FAE\FAE\AI_Relate\FAE_Project_Memory.md
```

Expected: matches in README and install/build guidance.

- [ ] **Step 2: Update `install.bat` to a PySide6-capable stack**

Change the install line to a PySide6-oriented dependency set in this shape:

```bat
conda create -n fae python=3.11
conda activate fae
pip install numpy scipy matplotlib pandas pillow pyside6 pyqtgraph pyradiomics seaborn reportlab imbalanced-learn pdfdocument statsmodels lifelines pyinstaller
```

Pin versions only where implementation validation proves a specific minimum is required.

- [ ] **Step 3: Update `README.md` contributor instructions**

Replace the Qt dependency and UI-generation guidance with text in this shape:

```markdown
- PySide6

The checked-in UI Python modules are maintained for runtime convenience. If regenerating them from `.ui` files, use a PySide6-compatible workflow and verify the generated imports before committing.
```

- [ ] **Step 4: Update `MainFrameCall.spec` for the new Qt runtime**

Remove or replace PyQt5-only excludes in this shape:

```python
shared_excludes = [
    'torch',
    'torchvision',
    'torchaudio',
    'tensorboard',
    'IPython',
    'jedi',
    'tkinter'
]
```

If implementation testing shows PySide6-specific packaging hooks are required, add them in the same edit rather than leaving the spec half-migrated.

- [ ] **Step 5: Update the repository memory file**

Add a short section to `AI_Relate\FAE_Project_Memory.md` noting that the Qt binding was migrated from PyQt5 to PySide6 and that future UI work should preserve PySide6 imports in checked-in generated modules.

- [ ] **Step 6: Commit the dependency and documentation updates**

```bash
git add install.bat README.md MainFrameCall.spec AI_Relate/FAE_Project_Memory.md
git commit -m "docs: align build and dependency guidance with PySide6

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Run final verification and clean up remaining PyQt5 traces

**Files:**
- Modify: `MainFrameCall.py`
- Modify: `MainFrameCall_opt.py`
- Modify: `HomeUI\HomePageForm.py`
- Modify: `HomeUI\HomePage.py`
- Modify: `Feature\GUI\*.py`
- Modify: `BC\GUI\*.py`
- Modify: `SA\GUI\*.py`
- Modify: `MatplotlibWidget.py`
- Modify: `install.bat`
- Modify: `README.md`
- Modify: `MainFrameCall.spec`
- Modify: `AI_Relate\FAE_Project_Memory.md`

- [ ] **Step 1: Run the final repository-wide search**

Run:

```powershell
rg -n "PyQt5|pyqtSignal|QtCore\.pyqtSignal|exec_\(|backend_qt4agg|backend_qt5agg" C:\Users\SunsServer\Project\FAE\FAE -g "*.py" -g "*.md" -g "*.bat" -g "*.spec"
```

Expected: no matches, except optional historical references inside archived design docs if those are intentionally retained.

- [ ] **Step 2: Run final smoke imports for the critical path**

Run:

```powershell
python -c "from HomeUI.HomePageForm import HomePageForm; from Feature.GUI.FeatureExtractionForm import FeatureExtractionForm; from BC.GUI.ProcessForm import ProcessConnection; from SA.GUI.ProcessForm import ProcessForm; print('pyside6-migration-smoke-ok')"
```

Expected: `pyside6-migration-smoke-ok`

- [ ] **Step 3: If smoke imports fail, fix the smallest blocking incompatibility and rerun**

Use this triage order:

```text
1. import path errors
2. Signal/exec API errors
3. matplotlib backend errors
4. generated UI syntax differences
```

Repeat Step 2 until the critical import path succeeds.

- [ ] **Step 4: Review the final diff before handoff**

Run:

```bash
git --no-pager diff --stat
git --no-pager diff
```

Expected: only Qt migration, dependency, packaging, and project-memory changes.

- [ ] **Step 5: Commit the final verification fixes**

```bash
git add MainFrameCall.py MainFrameCall_opt.py HomeUI/HomePageForm.py HomeUI/HomePage.py Feature/GUI/*.py BC/GUI/*.py SA/GUI/*.py MatplotlibWidget.py install.bat README.md MainFrameCall.spec AI_Relate/FAE_Project_Memory.md
git commit -m "refactor: finish PySide6 migration

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

