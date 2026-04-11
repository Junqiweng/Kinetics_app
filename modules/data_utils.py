# 文件作用：数据处理与辅助函数（默认化学计量数表、输出列名/单位、拟合结果对比表等）。

from __future__ import annotations

import re as _re

import numpy as np
import pandas as pd
import streamlit as st

from . import fitting
from .constants import (
    DEFAULT_MAX_STEP_FRACTION,
    EPSILON_DENOMINATOR,
    OUTPUT_MODE_COUT,
    OUTPUT_MODE_FOUT,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
)


def _clean_species_names(species_text: str) -> list[str]:
    parts = [p.strip() for p in species_text.split(",")]
    names = [p for p in parts if p]
    unique = []
    for n in names:
        if n not in unique:
            unique.append(n)
    return unique


def _parse_reaction_equation(equation: str, species_names: list[str]) -> np.ndarray | None:
    """
    解析单条反应方程式，返回一维计量数向量（长度=len(species_names)）。

    支持格式：
      "A → 2B"           # → 或 -> 作为分隔
      "A + B → C + D"
      "2A + 3B → C"
      "A + 0.5B → 2C"    # 支持小数系数

    返回 None 表示解析失败。
    """
    equation = equation.strip()
    if not equation:
        return None

    # 分离反应物和产物
    if "→" in equation:
        parts = equation.split("→", 1)
    elif "->" in equation:
        parts = equation.split("->", 1)
    elif "=" in equation:
        parts = equation.split("=", 1)
    else:
        return None

    if len(parts) != 2:
        return None

    species_names_sorted = sorted(
        [str(name) for name in species_names], key=len, reverse=True
    )

    def _parse_side(side_str: str) -> dict[str, float] | None:
        """解析方程一侧（如 '2A + 3B'），返回 {species: coefficient} 字典。"""
        result: dict[str, float] = {}
        rest = side_str.strip()
        while rest:
            matched = False
            for name in species_names_sorted:
                pattern = (
                    r"^(?P<coeff>[+]?(?:\d+(?:\.\d*)?|\.\d+)?)\s*"
                    + _re.escape(name)
                )
                match = _re.match(pattern, rest)
                if match is None:
                    continue

                coeff_str = str(match.group("coeff") or "").strip()
                coeff = float(coeff_str) if coeff_str else 1.0
                result[name] = result.get(name, 0.0) + coeff
                rest = rest[match.end() :].lstrip()
                matched = True
                break

            if not matched:
                return None

            if not rest:
                break
            if not rest.startswith("+"):
                return None
            rest = rest[1:].lstrip()
        return result

    reactants = _parse_side(parts[0])
    products = _parse_side(parts[1])
    if reactants is None or products is None:
        return None

    # 构建计量数向量
    nu = np.zeros(len(species_names), dtype=float)
    species_set = set(species_names)

    for name, coeff in reactants.items():
        if name not in species_set:
            return None
        idx = species_names.index(name)
        nu[idx] -= coeff

    for name, coeff in products.items():
        if name not in species_set:
            return None
        idx = species_names.index(name)
        nu[idx] += coeff

    return nu


def _parse_reaction_equations(
    equations_text: str, species_names: list[str]
) -> tuple[np.ndarray | None, list[str]]:
    """
    解析多条反应方程式（每行一条），返回 (stoich_matrix, errors)。

    stoich_matrix: shape (n_species, n_reactions)，若有解析错误则返回 None。
    errors: 每条无法解析的行的错误信息列表。
    """
    lines = [line.strip() for line in equations_text.strip().splitlines() if line.strip()]
    if not lines:
        return None, ["未输入任何反应方程式"]

    errors: list[str] = []
    columns: list[np.ndarray] = []
    for i, line in enumerate(lines):
        nu = _parse_reaction_equation(line, species_names)
        if nu is None:
            errors.append(f"第 {i+1} 行 \"{line}\"：无法解析（请检查格式和物种名称）")
        else:
            columns.append(nu)

    if errors:
        return None, errors

    # shape: (n_species, n_reactions) — 每列是一个反应的计量数向量
    stoich_matrix = np.column_stack(columns) if columns else None
    return stoich_matrix, errors


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
    raise ValueError(f"未知 output_mode: {output_mode}")


def _get_output_unit_text(output_mode: str) -> str:
    output_mode = str(output_mode).strip()
    if output_mode.startswith("F"):
        return "mol/s"
    if output_mode.startswith("C"):
        return "mol/m^3"
    if output_mode.startswith("x"):
        return "-"
    return "-"


def _freeze_params(params: dict) -> tuple:
    """将 fitted_params dict 转为可哈希的 tuple 表示（供 @st.cache_data 使用）。"""

    def _to_hashable(obj):
        """递归地将 list 转为 tuple（保留嵌套结构 / 多维形状）。"""
        if isinstance(obj, list):
            return tuple(_to_hashable(x) for x in obj)
        return obj

    items = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, np.ndarray):
            # 使用 tolist() 保留多维结构（如 reaction_order_matrix 的 2D 形状）
            items.append((k, _to_hashable(v.tolist())))
        elif isinstance(v, (list, tuple)):
            items.append((k, _to_hashable(v)))
        elif v is None:
            items.append((k, None))
        else:
            items.append((k, v))
    return tuple(items)


@st.cache_data(show_spinner="正在计算对比数据…")
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
    reversible_enabled: bool = False,
    pfr_flow_model: str = PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    max_step_fraction: float = DEFAULT_MAX_STEP_FRACTION,
    validation_mode: str = "output",
) -> pd.DataFrame:
    """
    构建拟合后的 “预测 vs 实验” 长表（用于奇偶校验图 / 残差图 / 导出）。

    参数
    ----
    validation_mode:
      - "output"：对比原始测量量（由 output_mode 指定：Cout/Fout/xout）
      - "conversion"：对比转化率（默认：
          - PFR/CSTR：优先按摩尔流率计算 X_i = (F0_i - Fout_i) / F0_i
          - BSTR：无摩尔流率，按浓度计算 X_i = (C0_i - Cout_i) / C0_i
        ）
    """

    def _row_get_value(row_obj: object, col: str) -> float:
        try:
            return float(getattr(row_obj, col))
        except Exception:
            return float("nan")

    validation_mode_norm = str(validation_mode).strip().lower()
    if validation_mode_norm in (
        "conversion",
        "conv",
        "x_conv",
        "xconversion",
        "转化率",
    ):
        rows = []
        row_indices = data_df.index.to_numpy()
        columns = set(map(str, data_df.columns))

        species_name_to_index = {name: i for i, name in enumerate(species_names)}
        output_species_indices = [
            species_name_to_index[name] for name in output_species_list
        ]

        # --- 预测所需：入口列推断（主要用于液相 PFR：可能只有 C0 或只有 F0）---
        has_all_c0 = all(f"C0_{name}_mol_m3" in columns for name in species_names)
        has_all_f0 = all(f"F0_{name}_mol_s" in columns for name in species_names)

        if reactor_type == REACTOR_TYPE_BSTR:
            pred_output_mode = OUTPUT_MODE_COUT
            inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]
        elif reactor_type == REACTOR_TYPE_CSTR:
            pred_output_mode = OUTPUT_MODE_FOUT
            inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]
        elif (
            reactor_type == REACTOR_TYPE_PFR
            and str(pfr_flow_model).strip() == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P
        ):
            pred_output_mode = OUTPUT_MODE_FOUT
            inlet_column_names = [f"F0_{name}_mol_s" for name in species_names]
        elif reactor_type == REACTOR_TYPE_PFR:
            if has_all_c0:
                pred_output_mode = OUTPUT_MODE_COUT
                inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]
            elif has_all_f0:
                pred_output_mode = OUTPUT_MODE_FOUT
                inlet_column_names = [f"F0_{name}_mol_s" for name in species_names]
            else:
                # 兜底：让 fitting 自己决定 inlet 列（若缺列会返回 ok=False）
                pred_output_mode = OUTPUT_MODE_FOUT
                inlet_column_names = None
        else:
            pred_output_mode = OUTPUT_MODE_FOUT
            inlet_column_names = None

        model_eval_cache: dict = {}
        data_rows = list(data_df.itertuples(index=False))
        for row_pos, row in enumerate(data_rows):
            row_index = row_indices[row_pos]
            vdot_m3_s = _row_get_value(row, "vdot_m3_s")

            pred_vals, model_ok, model_message = fitting._predict_outputs_for_row(
                row,
                species_names,
                pred_output_mode,
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
                reversible_enabled,
                pfr_flow_model,
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
                ok = bool(model_ok)
                message = str(model_message)

                if reactor_type == REACTOR_TYPE_BSTR:
                    c0 = _row_get_value(row, f"C0_{species_name}_mol_m3")
                    cout_meas = _row_get_value(row, f"Cout_{species_name}_mol_m3")
                    if not (np.isfinite(c0) and (c0 > 0.0)):
                        ok = False
                        message = f"C0_{species_name}_mol_m3 无效（<=0/NaN）"
                    if not np.isfinite(cout_meas):
                        ok = False
                        message = f"Cout_{species_name}_mol_m3 缺失/无效"

                    cout_pred = float(pred_vals[out_i]) if ok else np.nan
                    measured_value = (c0 - cout_meas) / c0 if ok else float("nan")
                    predicted_value = (c0 - cout_pred) / c0 if ok else float("nan")

                else:
                    # PFR/CSTR：按摩尔流率计算转化率（必要时可由浓度+vdot换算）
                    f0 = _row_get_value(row, f"F0_{species_name}_mol_s")
                    c0 = _row_get_value(row, f"C0_{species_name}_mol_m3")

                    if not np.isfinite(f0):
                        if (
                            np.isfinite(c0)
                            and np.isfinite(vdot_m3_s)
                            and vdot_m3_s > 0.0
                        ):
                            f0 = float(c0) * float(vdot_m3_s)
                        else:
                            ok = False
                            message = f"入口缺少 F0_{species_name}_mol_s，且无法由 C0+vdot 换算"

                    fout_meas = _row_get_value(row, f"Fout_{species_name}_mol_s")
                    cout_meas = _row_get_value(row, f"Cout_{species_name}_mol_m3")

                    if not np.isfinite(fout_meas):
                        if (
                            np.isfinite(cout_meas)
                            and np.isfinite(vdot_m3_s)
                            and vdot_m3_s > 0.0
                        ):
                            fout_meas = float(cout_meas) * float(vdot_m3_s)
                        else:
                            ok = False
                            message = f"出口缺少 Fout_{species_name}_mol_s，且无法由 Cout+vdot 换算"

                    if ok and (not (np.isfinite(f0) and (f0 > 0.0))):
                        ok = False
                        message = (
                            f"入口摩尔流率 F0_{species_name}_mol_s 无效（<=0/NaN）"
                        )

                    if pred_output_mode == OUTPUT_MODE_FOUT:
                        fout_pred = float(pred_vals[out_i]) if ok else np.nan
                    else:
                        # pred_output_mode == Cout：转为 Fout = Cout * vdot
                        if np.isfinite(vdot_m3_s) and vdot_m3_s > 0.0:
                            fout_pred = (
                                float(pred_vals[out_i]) * float(vdot_m3_s)
                                if ok
                                else np.nan
                            )
                        else:
                            ok = False
                            message = "vdot_m3_s 无效：无法将 Cout 预测值换算为 Fout"
                            fout_pred = float("nan")

                    measured_value = (f0 - fout_meas) / f0 if ok else float("nan")
                    predicted_value = (f0 - fout_pred) / f0 if ok else float("nan")

                residual_value = float(predicted_value) - float(measured_value)
                if (
                    ok
                    and np.isfinite(measured_value)
                    and (abs(measured_value) > EPSILON_DENOMINATOR)
                ):
                    relative_residual = residual_value / float(measured_value)
                else:
                    relative_residual = float("nan")

                rows.append(
                    {
                        "row_index": row_index,
                        "species": species_name,
                        "measured": float(measured_value),
                        "predicted": float(predicted_value),
                        "residual": float(residual_value),
                        "relative_residual": float(relative_residual),
                        "ok": bool(ok)
                        and np.isfinite(measured_value)
                        and np.isfinite(predicted_value),
                        "message": str(message),
                    }
                )

        return pd.DataFrame(rows)

    # --- validation_mode = output：沿用原 output_mode 的对比逻辑 ---
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
    if reactor_type == REACTOR_TYPE_PFR:
        inlet_column_names = (
            [f"F0_{name}_mol_s" for name in species_names]
            if str(pfr_flow_model).strip() == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P
            else (
                [f"C0_{name}_mol_m3" for name in species_names]
                if str(output_mode).startswith("C")
                else [f"F0_{name}_mol_s" for name in species_names]
            )
        )
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
            reversible_enabled,
            pfr_flow_model,
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
            if (
                ok
                and np.isfinite(measured_value)
                and (abs(measured_value) > EPSILON_DENOMINATOR)
            ):
                relative_residual = residual_value / float(measured_value)
            else:
                relative_residual = float("nan")
            rows.append(
                {
                    "row_index": row_index,
                    "species": species_name,
                    "measured": measured_value,
                    "predicted": predicted_value,
                    "residual": residual_value,
                    "relative_residual": relative_residual,
                    "ok": bool(ok),
                    "message": str(message),
                }
            )

    return pd.DataFrame(rows)
