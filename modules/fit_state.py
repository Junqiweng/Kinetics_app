from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def _hash_dataframe(data_df: pd.DataFrame | None) -> str:
    if data_df is None or not isinstance(data_df, pd.DataFrame):
        return ""
    return hashlib.md5(data_df.to_csv(index=False).encode()).hexdigest()


def _hash_numeric_array(value: Any) -> str:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return ""
    return hashlib.md5(arr.tobytes()).hexdigest()


def _hash_state_parts(*values: Any) -> str:
    hasher = hashlib.md5()
    for value in values:
        if value is None:
            hasher.update(b"<none>")
            continue
        arr = np.asarray(value)
        hasher.update(str(arr.shape).encode())
        hasher.update(str(arr.dtype).encode())
        if arr.dtype.kind == "b":
            hasher.update(np.asarray(arr, dtype=np.uint8).tobytes())
        else:
            try:
                hasher.update(np.asarray(arr, dtype=float).tobytes())
            except (TypeError, ValueError):
                hasher.update(str(arr.tolist()).encode())
    return hasher.hexdigest()


def build_fit_state_snapshot(
    *,
    data_df: pd.DataFrame | None,
    species_names: list[str] | tuple[str, ...],
    output_mode: str,
    output_species_list: list[str] | tuple[str, ...],
    stoich_matrix,
    solver_method: str,
    rtol: float,
    atol: float,
    reactor_type: str,
    kinetic_model: str,
    reversible_enabled: bool,
    pfr_flow_model: str,
    max_step_fraction: float,
    residual_type: str,
    use_log_k0_fit: bool,
    use_log_k0_rev_fit: bool,
    use_log_K0_ads_fit: bool,
    fit_k0_flags,
    fit_ea_flags,
    fit_order_flags_matrix,
    fit_K0_ads_flags,
    fit_Ea_K_flags,
    fit_m_flags,
    fit_k0_rev_flags,
    fit_ea_rev_flags,
    fit_order_rev_flags_matrix,
    k0_min: float,
    k0_max: float,
    ea_min: float,
    ea_max: float,
    ord_min: float,
    ord_max: float,
    K0_ads_min: float,
    K0_ads_max: float,
    Ea_K_min: float,
    Ea_K_max: float,
    k0_rev_min: float,
    k0_rev_max: float,
    ea_rev_min_J_mol: float,
    ea_rev_max_J_mol: float,
    order_rev_min: float,
    order_rev_max: float,
) -> dict:
    bounds_parts: list[Any] = [
        k0_min,
        k0_max,
        ea_min,
        ea_max,
        ord_min,
        ord_max,
    ]
    if str(kinetic_model) == "langmuir_hinshelwood":
        bounds_parts.extend([K0_ads_min, K0_ads_max, Ea_K_min, Ea_K_max])
    if bool(reversible_enabled):
        bounds_parts.extend(
            [
                k0_rev_min,
                k0_rev_max,
                ea_rev_min_J_mol,
                ea_rev_max_J_mol,
                order_rev_min,
                order_rev_max,
            ]
        )

    return {
        "data_hash": _hash_dataframe(data_df),
        "species_names": [str(x) for x in species_names],
        "output_mode": str(output_mode),
        "output_species": [str(x) for x in output_species_list],
        "stoich_hash": _hash_numeric_array(stoich_matrix),
        "solver_method": str(solver_method),
        "rtol": float(rtol),
        "atol": float(atol),
        "reactor_type": str(reactor_type),
        "kinetic_model": str(kinetic_model),
        "reversible_enabled": bool(reversible_enabled),
        "pfr_flow_model": str(pfr_flow_model),
        "max_step_fraction": float(max_step_fraction),
        "residual_type": str(residual_type),
        "use_log_k0_fit": bool(use_log_k0_fit),
        "use_log_k0_rev_fit": bool(use_log_k0_rev_fit),
        "use_log_K0_ads_fit": bool(use_log_K0_ads_fit),
        "fit_flags_hash": _hash_state_parts(
            fit_k0_flags,
            fit_ea_flags,
            fit_order_flags_matrix,
            fit_K0_ads_flags,
            fit_Ea_K_flags,
            fit_m_flags,
            fit_k0_rev_flags,
            fit_ea_rev_flags,
            fit_order_rev_flags_matrix,
        ),
        "fit_bounds_hash": _hash_state_parts(
            *bounds_parts,
        ),
    }


def build_fit_result_state_snapshot(fit_result: dict | None) -> dict:
    if not isinstance(fit_result, dict):
        return {}
    return {
        "data_hash": str(fit_result.get("data_hash", "")),
        "species_names": [str(x) for x in fit_result.get("species_names", [])],
        "output_mode": str(fit_result.get("output_mode", "")),
        "output_species": [str(x) for x in fit_result.get("output_species", [])],
        "stoich_hash": _hash_numeric_array(fit_result.get("stoich_matrix", [])),
        "solver_method": str(fit_result.get("solver_method", "")),
        "rtol": float(fit_result.get("rtol", np.nan)),
        "atol": float(fit_result.get("atol", np.nan)),
        "reactor_type": str(fit_result.get("reactor_type", "")),
        "kinetic_model": str(fit_result.get("kinetic_model", "")),
        "reversible_enabled": bool(fit_result.get("reversible_enabled", False)),
        "pfr_flow_model": str(fit_result.get("pfr_flow_model", "")),
        "max_step_fraction": float(fit_result.get("max_step_fraction", np.nan)),
        "residual_type": str(fit_result.get("residual_type", "")),
        "use_log_k0_fit": bool(fit_result.get("use_log_k0_fit", False)),
        "use_log_k0_rev_fit": bool(fit_result.get("use_log_k0_rev_fit", False)),
        "use_log_K0_ads_fit": bool(fit_result.get("use_log_K0_ads_fit", False)),
        "fit_flags_hash": str(fit_result.get("fit_flags_hash", "")),
        "fit_bounds_hash": str(fit_result.get("fit_bounds_hash", "")),
    }


def serialize_params_snapshot(params: dict | None) -> dict:
    if not isinstance(params, dict):
        return {}

    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            out[str(key)] = None
            continue
        if isinstance(value, np.ndarray):
            out[str(key)] = value.tolist()
            continue
        if isinstance(value, (list, tuple)):
            out[str(key)] = list(value)
            continue
        out[str(key)] = value
    return out


def describe_fit_state_differences(current_state: dict, result_state: dict) -> list[str]:
    if not current_state or not result_state:
        return []

    reasons: list[str] = []

    if current_state.get("data_hash", "") != result_state.get("data_hash", ""):
        reasons.append("实验数据已变化")
    if current_state.get("reactor_type", "") != result_state.get("reactor_type", ""):
        reasons.append("反应器类型已变化")
    if current_state.get("pfr_flow_model", "") != result_state.get("pfr_flow_model", ""):
        reasons.append("PFR 流动模型已变化")
    if current_state.get("kinetic_model", "") != result_state.get("kinetic_model", ""):
        reasons.append("动力学模型已变化")
    if bool(current_state.get("reversible_enabled", False)) != bool(
        result_state.get("reversible_enabled", False)
    ):
        reasons.append("可逆反应开关已变化")
    if current_state.get("species_names", []) != result_state.get("species_names", []):
        reasons.append("物种列表已变化")
    if current_state.get("output_mode", "") != result_state.get("output_mode", ""):
        reasons.append("拟合目标变量已变化")
    if current_state.get("output_species", []) != result_state.get("output_species", []):
        reasons.append("目标物种选择已变化")
    if current_state.get("stoich_hash", "") != result_state.get("stoich_hash", ""):
        reasons.append("化学计量数矩阵已变化")
    if current_state.get("solver_method", "") != result_state.get("solver_method", ""):
        reasons.append("求解器已变化")

    for float_key, label in [
        ("rtol", "rtol"),
        ("atol", "atol"),
        ("max_step_fraction", "max_step_fraction"),
    ]:
        cur_val = float(current_state.get(float_key, np.nan))
        old_val = float(result_state.get(float_key, np.nan))
        if (not np.isfinite(cur_val)) or (not np.isfinite(old_val)):
            if cur_val != old_val:
                reasons.append(f"{label} 已变化")
            continue
        if not np.isclose(cur_val, old_val, rtol=1e-12, atol=1e-15):
            reasons.append(f"{label} 已变化")

    if current_state.get("residual_type", "") != result_state.get("residual_type", ""):
        reasons.append("残差类型已变化")
    if bool(current_state.get("use_log_k0_fit", False)) != bool(
        result_state.get("use_log_k0_fit", False)
    ):
        reasons.append("k₀ 对数拟合开关已变化")
    if bool(current_state.get("use_log_k0_rev_fit", False)) != bool(
        result_state.get("use_log_k0_rev_fit", False)
    ):
        reasons.append("k₀,rev 对数拟合开关已变化")
    if bool(current_state.get("use_log_K0_ads_fit", False)) != bool(
        result_state.get("use_log_K0_ads_fit", False)
    ):
        reasons.append("K₀,ads 对数拟合开关已变化")
    if current_state.get("fit_flags_hash", "") != result_state.get("fit_flags_hash", ""):
        reasons.append("待拟合参数选择已变化")
    if current_state.get("fit_bounds_hash", "") != result_state.get("fit_bounds_hash", ""):
        reasons.append("参数边界已变化")

    return reasons
