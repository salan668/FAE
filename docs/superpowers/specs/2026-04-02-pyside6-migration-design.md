# PySide6 Migration Design

## Problem

The project currently uses PyQt5 across its entrypoints, generated UI modules, form logic, and matplotlib/Qt integration code. This creates friction for wider distribution and commercial adoption. The goal is to migrate the application to PySide6 as the sole Qt binding.

## Scope

The migration will be considered complete when:

- source code no longer depends on `PyQt5`
- main windows and dialogs use `PySide6`
- checked-in generated UI Python files are updated to PySide6-style imports and APIs
- Qt-specific runtime APIs are corrected, including `Signal`, `exec()`, and Qt enum usage
- Qt-related packaging and dependency files are updated for the PySide6 stack

Out of scope:

- business-logic refactors unrelated to the Qt binding change
- UI redesign or layout changes
- maintaining a PyQt5/PySide6 compatibility layer

## Selected Approach

Use a staged migration with one target runtime: `PySide6`.

Why this approach:

- safer than a blind global replace because the repo has multiple Qt integration surfaces
- simpler than a dual-binding compatibility layer because the target is a full cutover
- easier to verify incrementally by layer: core Qt APIs, generated UI files, plotting adapters, then packaging

Alternatives considered:

1. One-pass bulk replacement of imports and APIs. Faster initially, but higher risk of missing signal, backend, and dialog execution differences.
2. Dual-binding compatibility layer. Lower migration risk, but adds ongoing maintenance overhead the user does not want.

## Design

### 1. Qt foundation layer

Update the entrypoints and all form/controller modules to import from `PySide6` instead of `PyQt5`.

The main API changes to apply are:

- `pyqtSignal` / `QtCore.pyqtSignal` -> `Signal`
- `app.exec_()` / `dialog.exec_()` -> `app.exec()` / `dialog.exec()`
- direct `PyQt5.QtCore.Qt...` enum references -> `Qt...` or `QtCore.Qt...` in PySide6-compatible form

This layer includes at least:

- `MainFrameCall.py`
- `MainFrameCall_opt.py`
- `HomeUI\HomePageForm.py`
- `Feature\GUI\*Form.py`
- `BC\GUI\*Form.py`
- `SA\GUI\*Form.py`
- `MatplotlibWidget.py`
- `SA\GUI\MatplotlibWidget.py`

### 2. Generated UI layer

The repo checks in generated Python modules produced from `.ui` files. These must be updated so they import from `PySide6` and remain compatible with the form classes that use them.

Tracked UI sources include:

- `HomeUI\HomePage.ui`
- `Feature\GUI\FeatureExtraction.ui`
- `Feature\GUI\FeatureMerge.ui`
- `Feature\GUI\IccEstimation.ui`
- `BC\GUI\Prepare.ui`
- `BC\GUI\Process.ui`
- `BC\GUI\Visualization.ui`
- `BC\GUI\ModelPrediction.ui`
- `SA\GUI\Prepare.ui`
- `SA\GUI\Process.ui`
- `SA\GUI\Visualization.ui`

Their checked-in generated `.py` counterparts will be updated in place.

Some generated files appear to lack checked-in `.ui` sources, notably:

- `BC\GUI\FeatureExtraction.py`
- `BC\GUI\FeatureExtraction2.py`

Per the approved scope, these files will be edited directly to preserve runtime compatibility rather than blocked on reconstructing missing `.ui` sources.

### 3. Plotting and widget integration layer

Qt + matplotlib integration is the highest-risk part of this migration.

Two files need focused handling:

- `MatplotlibWidget.py`
- `SA\GUI\MatplotlibWidget.py`

Expected changes:

- move away from PyQt5-specific backend imports
- remove obsolete `backend_qt4agg` usage
- ensure the widget backend matches the PySide6-supported matplotlib Qt backend
- keep any pyqtgraph bridge code aligned with the selected Qt binding

Because old pinned dependency versions are likely incompatible with PySide6, this layer may require upgrading:

- `matplotlib`
- `pyqtgraph`
- `pyinstaller`

### 4. Packaging and dependency layer

Update project metadata and build files so the supported Qt stack is clearly PySide6-based.

Files to update:

- `install.bat`
- `README.md`
- `MainFrameCall.spec`
- any other build references discovered during implementation

Expected outcomes:

- PyQt5 dependency references are removed or replaced
- README installation guidance reflects the new stack
- PyInstaller configuration no longer assumes PyQt5-only exclusions or hooks where that would break PySide6 packaging

## Error Handling and Migration Rules

- Keep changes targeted to the Qt migration; do not refactor unrelated business logic.
- Where generated UI files are missing source `.ui` files, edit the generated files directly.
- Prefer explicit API fixes over compatibility hacks.
- If a file needs dependency-version-driven changes, update the corresponding installation/build documentation in the same migration.

## Verification Strategy

Verification will be done in three layers:

### Static verification

- search the repository for remaining `PyQt5` imports or references
- search for outdated APIs such as `pyqtSignal` and `exec_()`

### Runtime verification

- verify the main entrypoint can construct `QApplication`
- verify homepage-related imports resolve under `PySide6`
- verify critical plotting/widget modules import successfully after backend changes

### Packaging/dependency verification

- confirm `install.bat` and `README.md` point to a PySide6-capable dependency set
- confirm `MainFrameCall.spec` no longer encodes PyQt5-only assumptions that would break the new runtime

## Risks

1. Old pinned dependency versions may not support PySide6 cleanly.
2. Checked-in generated UI files increase the migration surface area.
3. Missing `.ui` sources for some generated files prevent fully clean regeneration.
4. `SA\GUI\MatplotlibWidget.py` appears to contain the most fragile Qt backend code and may need deeper adjustments than the rest of the migration.

## Implementation Boundary

The implementation should proceed in this order:

1. migrate core Qt imports and APIs
2. update generated UI modules
3. fix plotting/backend integration
4. update dependency and packaging files
5. verify no PyQt5 remnants remain

