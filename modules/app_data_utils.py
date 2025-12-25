# 文件作用：数据处理与辅助函数（默认化学计量数表、输出列名/单位、拟合结果对比表等）。

from __future__ import annotations

import numpy as np
import pandas as pd

from . import fitting
from .constants import DEFAULT_MAX_STEP_FRACTION


def _clean_species_names(species_text: str) -> list[str]:
    parts = [p.strip() for p in species_text.split(",")]
    names = [p for p in parts if p]
    unique = []
    for n in names:
        if n not in unique:
            unique.append(n)
    return unique


def _build_default_nu_table(species_names: list[str], n_reactions: int) -> pd.DataFrame:
    nu_default = pd.DataFrame(
        data=np.zeros((len(species_names), n_reactions), dtype=float),
        index=species_names,
        columns=[f"R{j+1}" for j in range(n_reactions)],
    )
    if n_reactions >= 1:
        if "A" in species_names:
            nu_default.loc["A", "R1"] = -1.0
        if "B" in species_names:
            nu_default.loc["B", "R1"] = 1.0
    return nu_default


def _get_measurement_column_name(output_mode: str, species_name: str) -> str:
    output_mode = str(output_mode).strip()
    if output_mode.startswith("F"):
        return f"Fout_{species_name}_mol_s"
    if output_mode.startswith("C"):
        return f"Cout_{species_name}_mol_m3"
    if output_mode.startswith("x"):
        return f"xout_{species_name}"
    # 兼容旧版本：曾支持 X (conversion)
    return f"X_{species_name}"


def _get_output_unit_text(output_mode: str) -> str:
    output_mode = str(output_mode).strip()
    if output_mode.startswith("F"):
        return "mol/s"
    if output_mode.startswith("C"):
        return "mol/m^3"
    if output_mode.startswith("x"):
        return "-"
    return "-"


def _build_fit_comparison_long_table(
    data_df: pd.DataFrame,
    species_names: list[str],
    output_mode: str,
    output_species_list: list[str],
    stoich_matrix: np.ndarray,
    fitted_params: dict,
    solver_method: str,
    rtol: float,
    atol: float,
    reactor_type: str,
    kinetic_model: str,
    max_step_fraction: float = DEFAULT_MAX_STEP_FRACTION,
) -> pd.DataFrame:
    rows = []
    row_indices = data_df.index.to_numpy()
    output_column_names = [
        _get_measurement_column_name(output_mode, sp) for sp in output_species_list
    ]
    measured_matrix = np.zeros((len(data_df), len(output_column_names)), dtype=float)
    for col_index, column_name in enumerate(output_column_names):
        measured_matrix[:, col_index] = pd.to_numeric(
            data_df[column_name], errors="coerce"
        ).to_numpy(dtype=float)

    species_name_to_index = {name: i for i, name in enumerate(species_names)}
    output_species_indices = [
        species_name_to_index[name] for name in output_species_list
    ]
    if reactor_type == "PFR":
        inlet_column_names = [f"F0_{name}_mol_s" for name in species_names]
    else:
        inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]

    model_eval_cache: dict = {}
    data_rows = list(data_df.itertuples(index=False))
    for row_pos, row in enumerate(data_rows):
        row_index = row_indices[row_pos]
        pred_vals, ok, message = fitting._predict_outputs_for_row(
            row,
            species_names,
            output_mode,
            output_species_list,
            stoich_matrix,
            fitted_params["k0"],
            fitted_params["ea_J_mol"],
            fitted_params["reaction_order_matrix"],
            solver_method,
            rtol,
            atol,
            reactor_type,
            kinetic_model,
            fitted_params.get("K0_ads", None),
            fitted_params.get("Ea_K", None),
            fitted_params.get("m_inhibition", None),
            fitted_params.get("k0_rev", None),
            fitted_params.get("ea_rev", None),
            fitted_params.get("order_rev", None),
            max_step_fraction=max_step_fraction,
            name_to_index=species_name_to_index,
            output_species_indices=output_species_indices,
            inlet_column_names=inlet_column_names,
            model_eval_cache=model_eval_cache,
        )

        for out_i, species_name in enumerate(output_species_list):
            measured_value = float(measured_matrix[row_pos, out_i])
            predicted_value = float(pred_vals[out_i]) if ok else np.nan
            residual_value = predicted_value - measured_value
            rows.append(
                {
                    "row_index": row_index,
                    "species": species_name,
                    "measured": measured_value,
                    "predicted": predicted_value,
                    "residual": residual_value,
                    "ok": bool(ok),
                    "message": str(message),
                }
            )

    return pd.DataFrame(rows)
