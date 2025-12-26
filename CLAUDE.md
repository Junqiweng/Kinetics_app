# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kinetics_app is a **Streamlit-based web application** for chemical reaction kinetics parameter fitting. It supports three reactor types (PFR, CSTR, BSTR) and three kinetic models (Power-law, Langmuir-Hinshelwood, Reversible reactions). The primary users are chemical engineers who need to fit kinetic parameters (k₀, Eₐ, reaction orders) from experimental data.

**Key assumption**: This codebase follows a **procedural programming style** optimized for chemical engineers who may be Python beginners. Avoid introducing classes, decorators, or complex abstractions unless absolutely necessary.

## Common Commands

### Run the Application
```bash
streamlit run app.py
```
The app will open at http://localhost:8501

### Generate Test Data
```bash
# Generate simple PFR example (orthogonal design, 27 runs)
python test_data/generate_orthogonal_design.py

# Generate complex validation data (PFR/BSTR with multiple reactions)
python test_data/generate_complex_data.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Code Architecture

### Module Hierarchy and Responsibilities

The codebase follows a **layered architecture** where core scientific computation is separated from UI logic:

```
app.py (Streamlit UI)
    ↓
modules/
├── Core Scientific Layer
│   ├── kinetics.py        # Rate calculations: r_j = k(T) × f(C)
│   ├── reactors.py        # ODE/algebraic solvers for PFR/CSTR/BSTR
│   └── fitting.py         # Parameter packing/unpacking, residual calculation
│
├── Configuration & Data
│   ├── constants.py       # Centralized physical constants and defaults
│   ├── config_manager.py  # JSON import/export, auto-save/restore
│   └── upload_persistence.py  # CSV file persistence across page refreshes
│
└── UI & Application Logic
    ├── ui_components.py   # Reusable UI elements, export utilities
    ├── ui_help.py         # Tutorial/help rendering from docs/*.md
    ├── app_style.py       # CSS/Matplotlib styling
    ├── app_config_state.py    # Streamlit session state management
    ├── app_data_utils.py      # Data validation, column naming
    └── app_fitting_background.py  # Background fitting with threading
```

### Critical Architectural Patterns

1. **Kinetic Model Dispatch**: All three models (Power-law, L-H, Reversible) share a common interface:
   - Input: `(conc_mol_m3, temperature_K, k0, ea_J_mol, reaction_order_matrix, ...)`
   - Output: `rate_vector` (shape: `n_reactions`)
   - See [kinetics.py](modules/kinetics.py) for implementations

2. **Reactor Solvers**:
   - **PFR**: `solve_ivp` integrates `dF/dV = Σ ν_ij × r_j` over reactor volume
   - **CSTR**: `least_squares` solves steady-state algebraic equation
   - **BSTR**: `solve_ivp` integrates `dC/dt = Σ ν_ij × r_j` over time
   - All use scipy's `solve_ivp` with configurable methods (RK45/BDF/Radau)

3. **Parameter Fitting Flow**:
   ```
   User UI Input → pack_fitting_params() → scipy.least_squares
                                              ↓
                                          residual_function()
                                              ↓
   For each row: reactor solver → predicted output → residual
   ```
   - See [fitting.py](modules/fitting.py) for `pack_fitting_params`, `unpack_fitting_params`
   - Residuals are calculated per-row and concatenated into a 1D array

4. **Configuration System**:
   - All UI state can be serialized to JSON via `config_manager.collect_config_from_state()`
   - Auto-save happens after each fit to `{tempdir}/Kinetics_app_persist/{session_id}_config.json`
   - Browser LocalStorage used for cloud deployments (see [browser_storage.py](modules/browser_storage.py))

5. **Constants Centralization**:
   - ALL magic numbers live in [constants.py](modules/constants.py)
   - Physical constants: `R_GAS_J_MOL_K = 8.314462618`
   - Numerical safeguards: `EPSILON_CONCENTRATION = 1e-30` (prevents 0^negative)
   - Default parameter bounds: `DEFAULT_K0_MIN/MAX`, `DEFAULT_EA_MIN/MAX_J_MOL`
   - UI defaults: `UI_DATA_PREVIEW_ROWS`, `DEFAULT_MAX_NFEV`
   - **Never hardcode these values elsewhere**

### Data Flow: CSV Upload → Fitting → Results

1. **CSV Upload** ([upload_persistence.py](modules/upload_persistence.py)):
   - User uploads CSV → saved to temp directory
   - Columns must match reactor type (e.g., PFR needs `V_m3`, `T_K`, `vdot_m3_s`, `F0_<species>_mol_s`)
   - Measurement columns: `Fout_<species>_mol_s` or `Cout_<species>_mol_m3`

2. **Data Validation** ([app_data_utils.py](modules/app_data_utils.py)):
   - Selected measurement columns **cannot have NaN/empty values**
   - Input columns (T, V, etc.) are validated for numeric type and range

3. **Fitting Execution** ([app_fitting_background.py](modules/app_fitting_background.py)):
   - Runs in background thread with progress updates via `queue.Queue`
   - Uses `scipy.optimize.least_squares` with `method="trf"`
   - Multi-start capability: tries multiple random initial guesses

4. **Results Caching**:
   - Fitted parameters stored in `st.session_state["fit_result"]`
   - Results are "locked" to the configuration/data hash at fit time
   - Changing data or config invalidates cache

### Streamlit-Specific Patterns

- **Session State Keys**: Use descriptive names like `species_names_input`, `reactor_type`, `fit_result`
- **Caching**: Use `@st.cache_data` for CSV parsing (`_read_csv_bytes_cached`)
- **File Downloads**: Use `st.download_button` with in-memory buffers (see [ui_components.py](modules/ui_components.py))
- **Browser Storage**: For cloud deployments, use `modules.browser_storage` to persist config across sessions

## Important Constraints

### Numerical Stability
- **Negative reaction orders**: When `n < 0` and `C → 0`, code clamps concentration to `EPSILON_CONCENTRATION` to avoid division by zero
- **Stiff ODEs**: For systems with widely varying timescales, users can switch ODE method to `BDF` or `Radau`
- **Parameter scaling**: `least_squares` uses `x_scale="jac"` to handle parameters of different magnitudes

### Assumptions and Limitations
- **PFR**: Assumes constant volumetric flow rate (liquid-phase, no pressure drop)
- **CSTR**: Solves for steady-state only (no transient dynamics)
- **BSTR**: Assumes constant volume, no inflow/outflow
- **Missing measurements**: Selected target species **must have valid measurements** in every row

### Configuration Management
- Configuration JSON includes all UI inputs: species, stoichiometry, initial guesses, bounds, fitting options
- Auto-save triggers: after successful fit, manual config export
- Auto-restore: on app startup, loads last config from temp directory (if exists)
- **Cloud caveat**: Temp directory may be cleared; browser LocalStorage provides backup

## Testing and Validation

### Quick Validation Workflow
1. Generate test data: `python test_data/generate_orthogonal_design.py`
2. Run app: `streamlit run app.py`
3. Upload `test_data/orthogonal_design_data.csv`
4. Select target variable: `Fout (mol/s)`, target species: `A`
5. Click "开始拟合" (Start Fitting)
6. Expected result: k₀ ≈ 1e6 s⁻¹, Eₐ ≈ 5e4 J/mol

### Validation Data Sets
- `orthogonal_design_data.csv`: Simple PFR, single reaction A→B
- `validation_PFR_Mixed.csv`: Complex PFR with 4 reactions (mixed kinetics)
- `validation_Batch_Series.csv`: BSTR with series reactions A→B→C→D→E
- `validation_PFR_LH.csv`: PFR with Langmuir-Hinshelwood inhibition

Each complex validation dataset has a matching `validation_*.json` config file for import.

## Key Files to Understand First

When starting work on this codebase, read these files in order:

1. [README.md](README.md) - User-facing documentation
2. [modules/constants.py](modules/constants.py) - All numerical constants and defaults
3. [modules/kinetics.py](modules/kinetics.py) - Core rate equations
4. [modules/reactors.py](modules/reactors.py) - ODE/algebraic solvers
5. [modules/fitting.py](modules/fitting.py) - Parameter packing and residual calculation
6. [app.py](app.py) - Main UI logic (large file, skim structure first)

## Development Guidelines

### Code Style
- **Procedural over OOP**: Use simple functions, avoid classes unless managing complex state
- **Variable naming**: Use descriptive names with physical meaning
  - Good: `temperature_K`, `reactor_volume_m3`, `activation_energy_J_mol`
  - Bad: `T`, `V`, `Ea`, `df`, `arr`
- **Units in variable names**: Always include units (e.g., `_K`, `_m3`, `_mol_s`, `_J_mol`)
- **Comments**: Explain "why" not "what"; chemical engineering context is valuable

### When Adding Features
- **Check constants.py first**: Add new defaults/bounds there, not in UI code
- **Update config schema**: If adding new inputs, update `config_manager.collect_config_from_state()` and `apply_config_to_state()`
- **Validate input data**: Add checks in `app_data_utils.py` for new column requirements
- **Document in docs/**: Update relevant `docs/help_*.md` files for user-facing changes

### Numerical Considerations
- Always use constants from `constants.py` (e.g., `R_GAS_J_MOL_K`, `EPSILON_CONCENTRATION`)
- For new kinetic models: follow the interface in `kinetics.py` (input conc/T, output rate vector)
- For new reactors: follow the interface in `reactors.py` (return outlet composition + optional profile)
- Test with stiff systems: verify BDF/Radau solvers work correctly

### UI Patterns
- Use `st.expander` for advanced options to reduce clutter
- Display units in labels: "Temperature [K]", "Volume [m³]"
- Provide download buttons for templates/examples
- Show progress bars for long-running fits
- Lock results display to fitted config/data (prevent confusion from config changes)

## Common Pitfalls

1. **Hardcoding numerical values**: Always use `constants.py`
2. **Ignoring units**: Variable names must include units
3. **Breaking config serialization**: If you add a new widget, update `config_manager`
4. **Overly complex abstractions**: Keep code simple for non-expert Python users
5. **Forgetting numerical safeguards**: Check for division by zero, negative concentrations
6. **Not testing with validation data**: Always test with `test_data/validation_*.csv` before committing

## File Naming Conventions

- **Modules**: Lowercase with underscores (`kinetics.py`, `config_manager.py`)
- **Test data**: Descriptive names with reactor type (`orthogonal_design_data.csv`, `validation_PFR_LH.json`)
- **Documentation**: Prefix with `help_` or descriptive name (`help_quickstart.md`, `user_guide.md`)
- **Constants**: UPPER_CASE in `constants.py` (`DEFAULT_K0_MIN`, `EPSILON_CONCENTRATION`)

## Reference Documentation

- **User guide**: [docs/user_guide.md](docs/user_guide.md) - Comprehensive usage instructions
- **Tutorials**: [docs/help_*.md](docs/) - In-app help content
- **TODO**: [TODO.md](TODO.md) - Planned features and known limitations
- **Test data**: [test_data/README.md](test_data/README.md) - How to generate and use test data
