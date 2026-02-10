# 文件作用：将 UI 输入参数组织为拟合向量，并调用各反应器/动力学模型计算预测值与残差（供 least_squares 优化）。

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_EA_K_MAX_J_MOL,
    DEFAULT_EA_K_MIN_J_MOL,
    DEFAULT_EA_REV_MAX_J_MOL,
    DEFAULT_EA_REV_MIN_J_MOL,
    DEFAULT_K0_ADS_MAX,
    DEFAULT_K0_ADS_MIN,
    DEFAULT_K0_REV_MAX,
    DEFAULT_K0_REV_MIN,
    DEFAULT_M_INHIBITION_MAX,
    DEFAULT_M_INHIBITION_MIN,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_ORDER_REV_MAX,
    DEFAULT_ORDER_REV_MIN,
    EPSILON_CONCENTRATION,
    EPSILON_FLOW_RATE,
    OUTPUT_MODE_COUT,
    OUTPUT_MODE_FOUT,
    OUTPUT_MODE_XOUT,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
    R_GAS_J_MOL_K,
)
from .reactors import (
    integrate_batch_reactor,
    integrate_pfr_molar_flows,
    integrate_pfr_molar_flows_gas_ideal_const_p,
    solve_cstr_steady_state_concentrations,
)


def _to_float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _row_get_value(row: object, key: str, default) -> object:
    """
    兼容 pandas.Series / dict / itertuples(namedtuple) 的取值方式。
    """
    if isinstance(row, pd.Series):
        return row.get(key, default)
    if isinstance(row, dict):
        return row.get(key, default)
    if hasattr(row, key):
        return getattr(row, key)
    return default


def _pack_parameters(
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    # 朗缪尔-辛斯伍德（L-H）参数
    K0_ads_guess: np.ndarray | None = None,
    Ea_K_guess: np.ndarray | None = None,
    m_inhibition_guess: np.ndarray | None = None,
    fit_K0_ads_flags: np.ndarray | None = None,
    fit_Ea_K_flags: np.ndarray | None = None,
    fit_m_flags: np.ndarray | None = None,
    # 可逆反应参数
    k0_rev_guess: np.ndarray | None = None,
    ea_rev_guess: np.ndarray | None = None,
    order_rev_guess: np.ndarray | None = None,
    fit_k0_rev_flags: np.ndarray | None = None,
    fit_ea_rev_flags: np.ndarray | None = None,
    fit_order_rev_flags_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    将所有需要拟合的参数打包成一个向量。
    顺序：k0 -> Ea -> n -> K0_ads -> Ea_K -> m -> k0_rev -> Ea_rev -> n_rev
    """
    parts = []

    # 1. k0 (正反应)
    if np.any(fit_k0_flags):
        parts.append(k0_guess[fit_k0_flags])

    # 2. Ea (正反应)
    if np.any(fit_ea_flags):
        parts.append(ea_guess_J_mol[fit_ea_flags])

    # 3. 反应级数（正反应）
    order_mask_flat = np.asarray(fit_order_flags_matrix, dtype=bool).ravel()
    if np.any(order_mask_flat):
        parts.append(order_guess.ravel()[order_mask_flat])

    # 4. K0_ads（L-H）
    if fit_K0_ads_flags is not None and np.any(fit_K0_ads_flags):
        parts.append(K0_ads_guess[fit_K0_ads_flags])

    # 5. Ea_K（L-H）
    if fit_Ea_K_flags is not None and np.any(fit_Ea_K_flags):
        parts.append(Ea_K_guess[fit_Ea_K_flags])

    # 6. m_inhibition（L-H）
    if fit_m_flags is not None and np.any(fit_m_flags):
        parts.append(m_inhibition_guess[fit_m_flags])

    # 7. k0_rev (可逆)
    if fit_k0_rev_flags is not None and np.any(fit_k0_rev_flags):
        parts.append(k0_rev_guess[fit_k0_rev_flags])

    # 8. Ea_rev (可逆)
    if fit_ea_rev_flags is not None and np.any(fit_ea_rev_flags):
        parts.append(ea_rev_guess[fit_ea_rev_flags])

    # 9. 反应级数（可逆）
    if fit_order_rev_flags_matrix is not None:
        order_rev_mask_flat = np.asarray(fit_order_rev_flags_matrix, dtype=bool).ravel()
        if np.any(order_rev_mask_flat):
            parts.append(order_rev_guess.ravel()[order_rev_mask_flat])

    if len(parts) == 0:
        return np.array([], dtype=float)
    return np.concatenate(parts).astype(float)


def _unpack_parameters(
    parameter_vector: np.ndarray,
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    # 朗缪尔-辛斯伍德（L-H）参数
    K0_ads_guess: np.ndarray | None = None,
    Ea_K_guess: np.ndarray | None = None,
    m_inhibition_guess: np.ndarray | None = None,
    fit_K0_ads_flags: np.ndarray | None = None,
    fit_Ea_K_flags: np.ndarray | None = None,
    fit_m_flags: np.ndarray | None = None,
    # 可逆反应参数
    k0_rev_guess: np.ndarray | None = None,
    ea_rev_guess: np.ndarray | None = None,
    order_rev_guess: np.ndarray | None = None,
    fit_k0_rev_flags: np.ndarray | None = None,
    fit_ea_rev_flags: np.ndarray | None = None,
    fit_order_rev_flags_matrix: np.ndarray | None = None,
) -> dict:
    """
    从参数向量中解包所有参数。
    返回包含所有参数的字典。
    """
    index = 0

    # 初始化为 guesses
    k0 = k0_guess.copy().astype(float)
    ea_J_mol = ea_guess_J_mol.copy().astype(float)
    reaction_order_matrix = order_guess.copy().astype(float)
    K0_ads = K0_ads_guess.copy().astype(float) if K0_ads_guess is not None else None
    Ea_K = Ea_K_guess.copy().astype(float) if Ea_K_guess is not None else None
    m_inhibition = (
        m_inhibition_guess.copy().astype(float)
        if m_inhibition_guess is not None
        else None
    )
    k0_rev = k0_rev_guess.copy().astype(float) if k0_rev_guess is not None else None
    ea_rev = ea_rev_guess.copy().astype(float) if ea_rev_guess is not None else None
    order_rev = (
        order_rev_guess.copy().astype(float) if order_rev_guess is not None else None
    )

    # 1. k0
    n_fit_k0 = int(np.sum(fit_k0_flags))
    if n_fit_k0 > 0:
        k0[fit_k0_flags] = parameter_vector[index : index + n_fit_k0]
        index += n_fit_k0

    # 2. Ea
    n_fit_ea = int(np.sum(fit_ea_flags))
    if n_fit_ea > 0:
        ea_J_mol[fit_ea_flags] = parameter_vector[index : index + n_fit_ea]
        index += n_fit_ea

    # 3. 反应级数
    n_reactions, n_species = reaction_order_matrix.shape
    order_mask_flat = np.asarray(fit_order_flags_matrix, dtype=bool).ravel()
    n_fit_orders = int(np.sum(order_mask_flat))
    if n_fit_orders > 0:
        flat_orders = reaction_order_matrix.ravel()
        flat_orders[order_mask_flat] = parameter_vector[index : index + n_fit_orders]
        reaction_order_matrix = flat_orders.reshape((n_reactions, n_species))
        index += n_fit_orders

    # 4. K0_ads
    if fit_K0_ads_flags is not None:
        n_fit_K0_ads = int(np.sum(fit_K0_ads_flags))
        if n_fit_K0_ads > 0:
            K0_ads[fit_K0_ads_flags] = parameter_vector[index : index + n_fit_K0_ads]
            index += n_fit_K0_ads

    # 5. Ea_K
    if fit_Ea_K_flags is not None:
        n_fit_Ea_K = int(np.sum(fit_Ea_K_flags))
        if n_fit_Ea_K > 0:
            Ea_K[fit_Ea_K_flags] = parameter_vector[index : index + n_fit_Ea_K]
            index += n_fit_Ea_K

    # 6. m_inhibition
    if fit_m_flags is not None:
        n_fit_m = int(np.sum(fit_m_flags))
        if n_fit_m > 0:
            m_inhibition[fit_m_flags] = parameter_vector[index : index + n_fit_m]
            index += n_fit_m

    # 7. k0_rev
    if fit_k0_rev_flags is not None:
        n_fit_k0_rev = int(np.sum(fit_k0_rev_flags))
        if n_fit_k0_rev > 0:
            k0_rev[fit_k0_rev_flags] = parameter_vector[index : index + n_fit_k0_rev]
            index += n_fit_k0_rev

    # 8. Ea_rev
    if fit_ea_rev_flags is not None:
        n_fit_ea_rev = int(np.sum(fit_ea_rev_flags))
        if n_fit_ea_rev > 0:
            ea_rev[fit_ea_rev_flags] = parameter_vector[index : index + n_fit_ea_rev]
            index += n_fit_ea_rev

    # 9. order_rev
    if fit_order_rev_flags_matrix is not None and order_rev is not None:
        order_rev_mask_flat = np.asarray(fit_order_rev_flags_matrix, dtype=bool).ravel()
        n_fit_order_rev = int(np.sum(order_rev_mask_flat))
        if n_fit_order_rev > 0:
            flat_order_rev = order_rev.ravel()
            flat_order_rev[order_rev_mask_flat] = parameter_vector[
                index : index + n_fit_order_rev
            ]
            order_rev = flat_order_rev.reshape(order_rev_guess.shape)
            index += n_fit_order_rev

    return {
        "k0": k0,
        "ea_J_mol": ea_J_mol,
        "reaction_order_matrix": reaction_order_matrix,
        "K0_ads": K0_ads,
        "Ea_K": Ea_K,
        "m_inhibition": m_inhibition,
        "k0_rev": k0_rev,
        "ea_rev": ea_rev,
        "order_rev": order_rev,
    }


def _build_bounds(
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    k0_min: float,
    k0_max: float,
    ea_min_J_mol: float,
    ea_max_J_mol: float,
    order_min: float,
    order_max: float,
    # 朗缪尔-辛斯伍德（L-H）边界参数
    fit_K0_ads_flags: np.ndarray | None = None,
    fit_Ea_K_flags: np.ndarray | None = None,
    fit_m_flags: np.ndarray | None = None,
    K0_ads_min: float = DEFAULT_K0_ADS_MIN,
    K0_ads_max: float = DEFAULT_K0_ADS_MAX,
    Ea_K_min: float = DEFAULT_EA_K_MIN_J_MOL,  # 允许负值（放热吸附）
    Ea_K_max: float = DEFAULT_EA_K_MAX_J_MOL,
    m_min: float = DEFAULT_M_INHIBITION_MIN,
    m_max: float = DEFAULT_M_INHIBITION_MAX,
    # 可逆反应边界参数
    fit_k0_rev_flags: np.ndarray | None = None,
    fit_ea_rev_flags: np.ndarray | None = None,
    fit_order_rev_flags_matrix: np.ndarray | None = None,
    k0_rev_min: float = DEFAULT_K0_REV_MIN,
    k0_rev_max: float = DEFAULT_K0_REV_MAX,
    ea_rev_min: float = DEFAULT_EA_REV_MIN_J_MOL,
    ea_rev_max: float = DEFAULT_EA_REV_MAX_J_MOL,
    order_rev_min: float = DEFAULT_ORDER_REV_MIN,
    order_rev_max: float = DEFAULT_ORDER_REV_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建所有拟合参数的边界。
    顺序与 _pack_parameters 一致。
    """
    lower_parts = []
    upper_parts = []

    # 1. k0
    n_fit_k0 = int(np.sum(fit_k0_flags))
    if n_fit_k0 > 0:
        lower_parts.append(np.full(n_fit_k0, float(k0_min), dtype=float))
        upper_parts.append(np.full(n_fit_k0, float(k0_max), dtype=float))

    # 2. Ea
    n_fit_ea = int(np.sum(fit_ea_flags))
    if n_fit_ea > 0:
        lower_parts.append(np.full(n_fit_ea, ea_min_J_mol, dtype=float))
        upper_parts.append(np.full(n_fit_ea, ea_max_J_mol, dtype=float))

    # 3. 反应级数
    n_fit_orders_total = int(np.sum(np.asarray(fit_order_flags_matrix, dtype=bool)))
    if n_fit_orders_total > 0:
        lower_parts.append(np.full(n_fit_orders_total, order_min, dtype=float))
        upper_parts.append(np.full(n_fit_orders_total, order_max, dtype=float))

    # 4. K0_ads
    if fit_K0_ads_flags is not None:
        n_fit_K0_ads = int(np.sum(fit_K0_ads_flags))
        if n_fit_K0_ads > 0:
            lower_parts.append(np.full(n_fit_K0_ads, K0_ads_min, dtype=float))
            upper_parts.append(np.full(n_fit_K0_ads, K0_ads_max, dtype=float))

    # 5. Ea_K
    if fit_Ea_K_flags is not None:
        n_fit_Ea_K = int(np.sum(fit_Ea_K_flags))
        if n_fit_Ea_K > 0:
            lower_parts.append(np.full(n_fit_Ea_K, Ea_K_min, dtype=float))
            upper_parts.append(np.full(n_fit_Ea_K, Ea_K_max, dtype=float))

    # 6. m_inhibition
    if fit_m_flags is not None:
        n_fit_m = int(np.sum(fit_m_flags))
        if n_fit_m > 0:
            lower_parts.append(np.full(n_fit_m, m_min, dtype=float))
            upper_parts.append(np.full(n_fit_m, m_max, dtype=float))

    # 7. k0_rev
    if fit_k0_rev_flags is not None:
        n_fit_k0_rev = int(np.sum(fit_k0_rev_flags))
        if n_fit_k0_rev > 0:
            lower_parts.append(np.full(n_fit_k0_rev, float(k0_rev_min), dtype=float))
            upper_parts.append(np.full(n_fit_k0_rev, float(k0_rev_max), dtype=float))

    # 8. Ea_rev
    if fit_ea_rev_flags is not None:
        n_fit_ea_rev = int(np.sum(fit_ea_rev_flags))
        if n_fit_ea_rev > 0:
            lower_parts.append(np.full(n_fit_ea_rev, ea_rev_min, dtype=float))
            upper_parts.append(np.full(n_fit_ea_rev, ea_rev_max, dtype=float))

    # 9. order_rev
    if fit_order_rev_flags_matrix is not None:
        n_fit_order_rev = int(
            np.sum(np.asarray(fit_order_rev_flags_matrix, dtype=bool))
        )
        if n_fit_order_rev > 0:
            lower_parts.append(np.full(n_fit_order_rev, order_rev_min, dtype=float))
            upper_parts.append(np.full(n_fit_order_rev, order_rev_max, dtype=float))

    if len(lower_parts) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(lower_parts), np.concatenate(upper_parts)


def _predict_outputs_for_row(
    row: object,
    species_names: list[str],
    output_mode: str,
    output_species_list: list[str],
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    reactor_type: str = REACTOR_TYPE_PFR,
    kinetic_model: str = "power_law",
    reversible_enabled: bool = False,
    pfr_flow_model: str = PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    name_to_index: dict[str, int] | None = None,
    output_species_indices: list[int] | None = None,
    inlet_column_names: list[str] | None = None,
    model_eval_cache: dict | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """
    根据反应器类型和动力学模型预测输出值。
    """
    temperature_K = _to_float_or_nan(_row_get_value(row, "T_K", np.nan))
    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            "温度 T_K 无效（请检查 CSV 的 T_K 列）",
        )

    if name_to_index is None:
        name_to_index = {name: i for i, name in enumerate(species_names)}

    if output_species_indices is None:
        try:
            output_species_indices = [
                name_to_index[name] for name in output_species_list
            ]
        except Exception:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "输出物种不在物种列表中（请检查物种名是否匹配）",
            )

    if reactor_type == REACTOR_TYPE_PFR:
        # 流动反应器（PFR）：
        # - liquid_const_vdot: 需要 V_m3, vdot_m3_s, F0_*（或 Cout 模式下允许 C0_* 并由 vdot 换算）
        # - gas_ideal_const_p: 需要 V_m3, P_Pa, F0_*（入口强制用 F0，不使用 vdot）
        reactor_volume_m3 = _to_float_or_nan(_row_get_value(row, "V_m3", np.nan))
        if not np.isfinite(reactor_volume_m3):
            return np.zeros(len(output_species_list), dtype=float), False, "缺少 V_m3"
        if reactor_volume_m3 < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "V_m3 不能为负",
            )

        pfr_flow_model = str(pfr_flow_model).strip() or PFR_FLOW_MODEL_LIQUID_CONST_VDOT

        if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
            # --- 气相：理想气体、等温、恒压 P（不考虑压降）---
            pressure_Pa = _to_float_or_nan(_row_get_value(row, "P_Pa", np.nan))
            if (not np.isfinite(pressure_Pa)) or (pressure_Pa <= 0.0):
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "压力 P_Pa 无效（请检查 CSV 的 P_Pa 列）",
                )

            # 入口强制使用 F0_*_mol_s（不再随 output_mode 切换到 C0_*）
            inlet_column_names = [f"F0_{name}_mol_s" for name in species_names]

            molar_flow_inlet = np.zeros(len(species_names), dtype=float)
            for i, col in enumerate(inlet_column_names):
                value = _to_float_or_nan(_row_get_value(row, col, np.nan))
                if not np.isfinite(value):
                    return (
                        np.zeros(len(output_species_list), dtype=float),
                        False,
                        f"缺少 {col}",
                    )
                if value < 0.0:
                    return (
                        np.zeros(len(output_species_list), dtype=float),
                        False,
                        f"{col} 不能为负",
                    )
                molar_flow_inlet[i] = float(value)

            molar_flow_outlet = None
            ok = True
            message = "OK"
            if model_eval_cache is not None:
                cache_key = (
                    "PFR",
                    str(pfr_flow_model),
                    float(reactor_volume_m3),
                    float(temperature_K),
                    float(pressure_Pa),
                    tuple(molar_flow_inlet),
                )
                if cache_key in model_eval_cache:
                    cached = model_eval_cache[cache_key]
                    molar_flow_outlet, ok, message = cached

            if molar_flow_outlet is None:
                molar_flow_outlet, ok, message = (
                    integrate_pfr_molar_flows_gas_ideal_const_p(
                        reactor_volume_m3=reactor_volume_m3,
                        temperature_K=temperature_K,
                        pressure_Pa=pressure_Pa,
                        molar_flow_inlet_mol_s=molar_flow_inlet,
                        stoich_matrix=stoich_matrix,
                        k0=k0,
                        ea_J_mol=ea_J_mol,
                        reaction_order_matrix=reaction_order_matrix,
                        solver_method=solver_method,
                        rtol=rtol,
                        atol=atol,
                        kinetic_model=kinetic_model,
                        reversible_enabled=reversible_enabled,
                        max_step_fraction=max_step_fraction,
                        K0_ads=K0_ads,
                        Ea_K_J_mol=Ea_K_J_mol,
                        m_inhibition=m_inhibition,
                        k0_rev=k0_rev,
                        ea_rev_J_mol=ea_rev_J_mol,
                        order_rev_matrix=order_rev_matrix,
                        stop_event=stop_event,
                        max_wall_time_s=max_wall_time_s,
                    )
                )
                if model_eval_cache is not None:
                    model_eval_cache[cache_key] = (
                        molar_flow_outlet,
                        bool(ok),
                        str(message),
                    )
            if not ok:
                return np.zeros(len(output_species_list), dtype=float), False, message

            # 计算输出值
            conc_total_mol_m3 = float(pressure_Pa) / max(
                float(R_GAS_J_MOL_K) * float(temperature_K), EPSILON_CONCENTRATION
            )
            total_flow = float(np.sum(molar_flow_outlet))
            if (output_mode in (OUTPUT_MODE_XOUT, OUTPUT_MODE_COUT)) and (
                total_flow < EPSILON_FLOW_RATE
            ):
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "总摩尔流率过低，无法计算 xout/Cout（请检查入口流量与工况）",
                )

            output_values = np.zeros(len(output_species_list), dtype=float)
            for out_i, idx in enumerate(output_species_indices):
                if output_mode == OUTPUT_MODE_FOUT:
                    output_values[out_i] = molar_flow_outlet[idx]
                elif output_mode == OUTPUT_MODE_XOUT:
                    output_values[out_i] = molar_flow_outlet[idx] / total_flow
                elif output_mode == OUTPUT_MODE_COUT:
                    y_i = molar_flow_outlet[idx] / total_flow
                    output_values[out_i] = float(y_i) * float(conc_total_mol_m3)
                else:
                    return (
                        np.zeros(len(output_species_list), dtype=float),
                        False,
                        "未知输出模式",
                    )

        else:
            # --- 液相：体积流量 vdot 近似恒定 ---
            vdot_m3_s = _to_float_or_nan(_row_get_value(row, "vdot_m3_s", np.nan))
            if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "体积流量 vdot_m3_s 无效（请检查 CSV 的 vdot_m3_s 列）",
                )

            # 约定（仅液相 PFR）：当拟合目标为 Cout 时，入口也允许使用浓度 C0_*，
            # 并由 vdot 自动换算为 F0 参与计算。
            use_conc_inlet = str(output_mode).startswith("C")
            if inlet_column_names is None:
                inlet_column_names = (
                    [f"C0_{name}_mol_m3" for name in species_names]
                    if use_conc_inlet
                    else [f"F0_{name}_mol_s" for name in species_names]
                )

            molar_flow_inlet = np.zeros(len(species_names), dtype=float)
            for i, col in enumerate(inlet_column_names):
                value = _to_float_or_nan(_row_get_value(row, col, np.nan))
                if not np.isfinite(value):
                    return (
                        np.zeros(len(output_species_list), dtype=float),
                        False,
                        f"缺少 {col}",
                    )
                if value < 0.0:
                    return (
                        np.zeros(len(output_species_list), dtype=float),
                        False,
                        f"{col} 不能为负",
                    )
                if use_conc_inlet:
                    # C0 [mol/m^3] -> F0 [mol/s] = C0 * vdot
                    molar_flow_inlet[i] = float(value) * float(vdot_m3_s)
                else:
                    molar_flow_inlet[i] = float(value)

            molar_flow_outlet = None
            ok = True
            message = "OK"
            if model_eval_cache is not None:
                cache_key = (
                    "PFR",
                    str(PFR_FLOW_MODEL_LIQUID_CONST_VDOT),
                    float(reactor_volume_m3),
                    float(temperature_K),
                    float(vdot_m3_s),
                    tuple(molar_flow_inlet),
                )
                if cache_key in model_eval_cache:
                    cached = model_eval_cache[cache_key]
                    molar_flow_outlet, ok, message = cached

            if molar_flow_outlet is None:
                molar_flow_outlet, ok, message = integrate_pfr_molar_flows(
                    reactor_volume_m3=reactor_volume_m3,
                    temperature_K=temperature_K,
                    vdot_m3_s=vdot_m3_s,
                    molar_flow_inlet_mol_s=molar_flow_inlet,
                    stoich_matrix=stoich_matrix,
                    k0=k0,
                    ea_J_mol=ea_J_mol,
                    reaction_order_matrix=reaction_order_matrix,
                    solver_method=solver_method,
                    rtol=rtol,
                    atol=atol,
                    kinetic_model=kinetic_model,
                    reversible_enabled=reversible_enabled,
                    max_step_fraction=max_step_fraction,
                    K0_ads=K0_ads,
                    Ea_K_J_mol=Ea_K_J_mol,
                    m_inhibition=m_inhibition,
                    k0_rev=k0_rev,
                    ea_rev_J_mol=ea_rev_J_mol,
                    order_rev_matrix=order_rev_matrix,
                    stop_event=stop_event,
                    max_wall_time_s=max_wall_time_s,
                )
                if model_eval_cache is not None:
                    model_eval_cache[cache_key] = (
                        molar_flow_outlet,
                        bool(ok),
                        str(message),
                    )
            if not ok:
                return np.zeros(len(output_species_list), dtype=float), False, message

            # 计算输出值
            output_values = np.zeros(len(output_species_list), dtype=float)
            for out_i, idx in enumerate(output_species_indices):
                if output_mode == OUTPUT_MODE_FOUT:
                    output_values[out_i] = molar_flow_outlet[idx]
                elif output_mode == OUTPUT_MODE_COUT:
                    output_values[out_i] = molar_flow_outlet[idx] / max(
                        vdot_m3_s, EPSILON_FLOW_RATE
                    )
                elif output_mode == OUTPUT_MODE_XOUT:
                    # 摩尔组成 y_i = F_i / Σ F_j
                    total_flow = np.sum(molar_flow_outlet)
                    if total_flow < EPSILON_FLOW_RATE:
                        return (
                            np.zeros(len(output_species_list), dtype=float),
                            False,
                            "总摩尔流率过低，无法计算 xout（请检查入口流量与工况）",
                        )
                    output_values[out_i] = molar_flow_outlet[idx] / total_flow
                else:
                    return (
                        np.zeros(len(output_species_list), dtype=float),
                        False,
                        "未知输出模式",
                    )

    elif reactor_type == REACTOR_TYPE_CSTR:
        reactor_volume_m3 = _to_float_or_nan(_row_get_value(row, "V_m3", np.nan))
        if not np.isfinite(reactor_volume_m3):
            return np.zeros(len(output_species_list), dtype=float), False, "缺少 V_m3"
        if reactor_volume_m3 < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "V_m3 不能为负",
            )

        vdot_m3_s = _to_float_or_nan(_row_get_value(row, "vdot_m3_s", np.nan))
        if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "体积流量 vdot_m3_s 无效（请检查 CSV 的 vdot_m3_s 列）",
            )

        if inlet_column_names is None:
            inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]

        conc_inlet = np.zeros(len(species_names), dtype=float)
        for i, col in enumerate(inlet_column_names):
            value = _to_float_or_nan(_row_get_value(row, col, np.nan))
            if not np.isfinite(value):
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    f"缺少 {col}",
                )
            if value < 0.0:
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    f"{col} 不能为负",
                )
            conc_inlet[i] = float(value)

        conc_outlet = None
        ok = True
        message = "OK"
        if model_eval_cache is not None:
            cache_key = (
                "CSTR",
                float(reactor_volume_m3),
                float(vdot_m3_s),
                float(temperature_K),
                tuple(conc_inlet),
            )
            if cache_key in model_eval_cache:
                cached = model_eval_cache[cache_key]
                conc_outlet, ok, message = cached

        if conc_outlet is None:
            conc_outlet, ok, message = solve_cstr_steady_state_concentrations(
                reactor_volume_m3=reactor_volume_m3,
                temperature_K=temperature_K,
                vdot_m3_s=vdot_m3_s,
                conc_inlet_mol_m3=conc_inlet,
                stoich_matrix=stoich_matrix,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                kinetic_model=kinetic_model,
                reversible_enabled=reversible_enabled,
                K0_ads=K0_ads,
                Ea_K_J_mol=Ea_K_J_mol,
                m_inhibition=m_inhibition,
                k0_rev=k0_rev,
                ea_rev_J_mol=ea_rev_J_mol,
                order_rev_matrix=order_rev_matrix,
                stop_event=stop_event,
                max_wall_time_s=max_wall_time_s,
            )
            if model_eval_cache is not None:
                model_eval_cache[cache_key] = (conc_outlet, bool(ok), str(message))

        if not ok:
            return np.zeros(len(output_species_list), dtype=float), False, message

        output_values = np.zeros(len(output_species_list), dtype=float)
        for out_i, idx in enumerate(output_species_indices):
            if output_mode == OUTPUT_MODE_COUT:
                output_values[out_i] = conc_outlet[idx]
            elif output_mode == OUTPUT_MODE_FOUT:
                output_values[out_i] = float(vdot_m3_s) * conc_outlet[idx]
            elif output_mode == OUTPUT_MODE_XOUT:
                # 摩尔组成 y_i = F_i / Σ F_j = C_i / Σ C_j (体积流量可约掉)
                total_conc = np.sum(conc_outlet)
                if total_conc < EPSILON_CONCENTRATION:
                    return (
                        np.zeros(len(output_species_list), dtype=float),
                        False,
                        "总浓度过低，无法计算 xout（请检查入口浓度与工况）",
                    )
                output_values[out_i] = conc_outlet[idx] / total_conc
            else:
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "未知输出模式",
                )

    elif reactor_type in (REACTOR_TYPE_BSTR, "Batch"):
        # 间歇釜（BSTR）需要 t_s, C0_*
        reaction_time_s = _to_float_or_nan(_row_get_value(row, "t_s", np.nan))
        if not np.isfinite(reaction_time_s):
            return np.zeros(len(output_species_list), dtype=float), False, "缺少 t_s"
        if reaction_time_s < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "t_s 不能为负",
            )

        if inlet_column_names is None:
            inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]

        conc_initial = np.zeros(len(species_names), dtype=float)
        for i, col in enumerate(inlet_column_names):
            value = _to_float_or_nan(_row_get_value(row, col, np.nan))
            if not np.isfinite(value):
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    f"缺少 {col}",
                )
            if value < 0.0:
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    f"{col} 不能为负",
                )
            conc_initial[i] = float(value)

        conc_final = None
        ok = True
        message = "OK"
        if model_eval_cache is not None:
            cache_key = (
                "BSTR",
                float(reaction_time_s),
                float(temperature_K),
                tuple(conc_initial),
            )
            if cache_key in model_eval_cache:
                cached = model_eval_cache[cache_key]
                conc_final, ok, message = cached

        if conc_final is None:
            conc_final, ok, message = integrate_batch_reactor(
                reaction_time_s=reaction_time_s,
                temperature_K=temperature_K,
                conc_initial_mol_m3=conc_initial,
                stoich_matrix=stoich_matrix,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                solver_method=solver_method,
                rtol=rtol,
                atol=atol,
                kinetic_model=kinetic_model,
                reversible_enabled=reversible_enabled,
                max_step_fraction=max_step_fraction,
                K0_ads=K0_ads,
                Ea_K_J_mol=Ea_K_J_mol,
                m_inhibition=m_inhibition,
                k0_rev=k0_rev,
                ea_rev_J_mol=ea_rev_J_mol,
                order_rev_matrix=order_rev_matrix,
                stop_event=stop_event,
                max_wall_time_s=max_wall_time_s,
            )
            if model_eval_cache is not None:
                model_eval_cache[cache_key] = (conc_final, bool(ok), str(message))
        if not ok:
            return np.zeros(len(output_species_list), dtype=float), False, message

        # 计算输出值
        output_values = np.zeros(len(output_species_list), dtype=float)
        for out_i, idx in enumerate(output_species_indices):
            if output_mode == OUTPUT_MODE_COUT:
                output_values[out_i] = conc_final[idx]
            else:
                # 间歇釜（BSTR）不支持 Fout 模式
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "BSTR 反应器仅支持 Cout 输出模式",
                )

    else:
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            f"未知反应器类型: {reactor_type}",
        )

    if not bool(np.all(np.isfinite(output_values))):
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            "模型输出包含 NaN/Inf（请检查工况、输入数据与求解器设置）",
        )

    return output_values, True, "OK"
