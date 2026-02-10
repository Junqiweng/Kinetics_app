# Repository Guidelines

## Project Structure & Module Organization
The app entrypoint is `main.py` (Streamlit UI shell). Core logic is in `modules/`, split by function: reactor/kinetics math (`reactors.py`, `kinetics.py`, `fitting.py`), UI tabs (`tab_model.py`, `tab_data.py`, `tab_fit.py`), and config/state helpers (`config_manager.py`, `bootstrap.py`, `contexts.py`).  
Documentation lives in `docs/` (`user_guide.md` and `help_*.md`).  
Validation data and generators are in `test_data/` (CSV/JSON fixtures + generation scripts).  
Quick health checks live in `scripts/` (`smoke_validate.py`).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` - install runtime dependencies.
- `streamlit run main.py` - start the local app (`http://localhost:8501`).
- `python scripts/smoke_validate.py` - run lightweight prediction/config smoke checks for PFR/CSTR/BSTR paths.
- `python scripts/regression_state_validate.py` - run session-state regression checks.
- `python test_data/generate_orthogonal_design.py` - generate baseline PFR sample data.
- `python test_data/generate_complex_data.py` - regenerate advanced validation datasets.

## Coding Style & Naming Conventions
Use 4-space indentation and follow PEP 8 style with clear type hints where practical.  
Prefer simple, procedural, engineering-focused code over unnecessary abstraction.  
Use descriptive `snake_case` for files, functions, and variables (for example, `fit_execution.py`, `reaction_order_matrix`).  
Keep unit-aware names/comments explicit (for example, `T_K`, `vdot_m3_s`, `ea_J_mol`).  
No formatter/linter config is committed currently; keep diffs small and consistent with neighboring code.

## Testing Guidelines
Current baseline testing is script-driven (no dedicated `pytest` suite yet).  
Before opening a PR, run `python scripts/smoke_validate.py` and verify app startup with `streamlit run main.py`.  
For model/data logic changes, add or update reproducible fixtures in `test_data/` and document expected behavior in the PR.

## Commit & Pull Request Guidelines
Recent history uses release-like subjects such as `V1.88 ...` with concise Chinese summaries. Follow this pattern for versioned milestones, or use short imperative summaries for regular commits.  
PRs should include: change purpose, impacted modules, validation steps/commands, and UI screenshots for visual changes.  
Link related issues/tasks and call out any backward-incompatible config or data format changes explicitly.
