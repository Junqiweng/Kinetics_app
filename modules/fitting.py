from __future__ import annotations

import numpy as np
import pandas as pd

from .reactors import integrate_batch_reactor, integrate_pfr_molar_flows


def _to_float_or_nan(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _pack_parameters(
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    # L-H 参数
    K0_ads_guess: np.ndarray = None,
    Ea_K_guess: np.ndarray = None,
    m_inhibition_guess: np.ndarray = None,
    fit_K0_ads_flags: np.ndarray = None,
    fit_Ea_K_flags: np.ndarray = None,
    fit_m_flags: np.ndarray = None,
    # 可逆反应参数
    k0_rev_guess: np.ndarray = None,
    ea_rev_guess: np.ndarray = None,
    order_rev_guess: np.ndarray = None,
    fit_k0_rev_flags: np.ndarray = None,
    fit_ea_rev_flags: np.ndarray = None,
    fit_order_rev_flags_matrix: np.ndarray = None,
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

    # 3. Orders (正反应)
    order_mask_flat = np.asarray(fit_order_flags_matrix, dtype=bool).ravel()
    if np.any(order_mask_flat):
        parts.append(order_guess.ravel()[order_mask_flat])

    # 4. K0_ads (L-H)
    if fit_K0_ads_flags is not None and np.any(fit_K0_ads_flags):
        parts.append(K0_ads_guess[fit_K0_ads_flags])

    # 5. Ea_K (L-H)
    if fit_Ea_K_flags is not None and np.any(fit_Ea_K_flags):
        parts.append(Ea_K_guess[fit_Ea_K_flags])

    # 6. m_inhibition (L-H)
    if fit_m_flags is not None and np.any(fit_m_flags):
        parts.append(m_inhibition_guess[fit_m_flags])

    # 7. k0_rev (可逆)
    if fit_k0_rev_flags is not None and np.any(fit_k0_rev_flags):
        parts.append(k0_rev_guess[fit_k0_rev_flags])

    # 8. Ea_rev (可逆)
    if fit_ea_rev_flags is not None and np.any(fit_ea_rev_flags):
        parts.append(ea_rev_guess[fit_ea_rev_flags])

    # 9. Orders_rev (可逆)
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
    # L-H 参数
    K0_ads_guess: np.ndarray = None,
    Ea_K_guess: np.ndarray = None,
    m_inhibition_guess: np.ndarray = None,
    fit_K0_ads_flags: np.ndarray = None,
    fit_Ea_K_flags: np.ndarray = None,
    fit_m_flags: np.ndarray = None,
    # 可逆反应参数
    k0_rev_guess: np.ndarray = None,
    ea_rev_guess: np.ndarray = None,
    order_rev_guess: np.ndarray = None,
    fit_k0_rev_flags: np.ndarray = None,
    fit_ea_rev_flags: np.ndarray = None,
    fit_order_rev_flags_matrix: np.ndarray = None,
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

    # 3. Orders
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
    # L-H 边界参数
    fit_K0_ads_flags: np.ndarray = None,
    fit_Ea_K_flags: np.ndarray = None,
    fit_m_flags: np.ndarray = None,
    K0_ads_min: float = 1e-10,
    K0_ads_max: float = 1e10,
    Ea_K_min: float = -2e5,  # 允许负值（放热吸附）
    Ea_K_max: float = 2e5,
    m_min: float = 0.0,
    m_max: float = 5.0,
    # 可逆反应边界参数
    fit_k0_rev_flags: np.ndarray = None,
    fit_ea_rev_flags: np.ndarray = None,
    fit_order_rev_flags_matrix: np.ndarray = None,
    k0_rev_min: float = 1e-10,
    k0_rev_max: float = 1e15,
    ea_rev_min: float = 0.0,
    ea_rev_max: float = 5e5,
    order_rev_min: float = -3.0,
    order_rev_max: float = 5.0,
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
        lower_parts.append(np.full(n_fit_k0, k0_min, dtype=float))
        upper_parts.append(np.full(n_fit_k0, k0_max, dtype=float))

    # 2. Ea
    n_fit_ea = int(np.sum(fit_ea_flags))
    if n_fit_ea > 0:
        lower_parts.append(np.full(n_fit_ea, ea_min_J_mol, dtype=float))
        upper_parts.append(np.full(n_fit_ea, ea_max_J_mol, dtype=float))

    # 3. Orders
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
            lower_parts.append(np.full(n_fit_k0_rev, k0_rev_min, dtype=float))
            upper_parts.append(np.full(n_fit_k0_rev, k0_rev_max, dtype=float))

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
    row: pd.Series,
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
    reactor_type: str = "PFR",
    kinetic_model: str = "power_law",
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, bool, str]:
    """
    根据反应器类型和动力学模型预测输出值。
    """
    temperature_K = _to_float_or_nan(row.get("T_K", np.nan))
    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            "温度 T_K 无效（请检查 CSV 的 T_K 列）",
        )

    name_to_index = {name: i for i, name in enumerate(species_names)}

    if reactor_type == "PFR":
        # PFR 需要 V_m3, vdot_m3_s, F0_*
        reactor_volume_m3 = _to_float_or_nan(row.get("V_m3", np.nan))
        if not np.isfinite(reactor_volume_m3):
            return np.zeros(len(output_species_list), dtype=float), False, "缺少 V_m3"
        if reactor_volume_m3 < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "V_m3 不能为负",
            )

        vdot_m3_s = _to_float_or_nan(row.get("vdot_m3_s", np.nan))
        if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "体积流量 vdot_m3_s 无效（请检查 CSV 的 vdot_m3_s 列）",
            )

        molar_flow_inlet = np.zeros(len(species_names), dtype=float)
        for i, name in enumerate(species_names):
            col = f"F0_{name}_mol_s"
            value = _to_float_or_nan(row.get(col, np.nan))
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
            K0_ads=K0_ads,
            Ea_K_J_mol=Ea_K_J_mol,
            m_inhibition=m_inhibition,
            k0_rev=k0_rev,
            ea_rev_J_mol=ea_rev_J_mol,
            order_rev_matrix=order_rev_matrix,
        )
        if not ok:
            return np.zeros(len(output_species_list), dtype=float), False, message

        # 计算输出值
        output_values = np.zeros(len(output_species_list), dtype=float)
        for out_i, species in enumerate(output_species_list):
            idx = name_to_index[species]
            if output_mode == "Fout (mol/s)":
                output_values[out_i] = molar_flow_outlet[idx]
            elif output_mode == "Cout (mol/m^3)":
                output_values[out_i] = molar_flow_outlet[idx] / max(vdot_m3_s, 1e-30)
            elif output_mode == "X (conversion)":
                f0 = molar_flow_inlet[idx]
                fout = molar_flow_outlet[idx]
                if f0 < 1e-30:
                    output_values[out_i] = np.nan
                else:
                    output_values[out_i] = (f0 - fout) / f0
            else:
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "未知输出模式",
                )

    elif reactor_type == "Batch":
        # Batch 需要 t_s, C0_*
        reaction_time_s = _to_float_or_nan(row.get("t_s", np.nan))
        if not np.isfinite(reaction_time_s):
            return np.zeros(len(output_species_list), dtype=float), False, "缺少 t_s"
        if reaction_time_s < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "t_s 不能为负",
            )

        conc_initial = np.zeros(len(species_names), dtype=float)
        for i, name in enumerate(species_names):
            col = f"C0_{name}_mol_m3"
            value = _to_float_or_nan(row.get(col, np.nan))
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
            K0_ads=K0_ads,
            Ea_K_J_mol=Ea_K_J_mol,
            m_inhibition=m_inhibition,
            k0_rev=k0_rev,
            ea_rev_J_mol=ea_rev_J_mol,
            order_rev_matrix=order_rev_matrix,
        )
        if not ok:
            return np.zeros(len(output_species_list), dtype=float), False, message

        # 计算输出值
        output_values = np.zeros(len(output_species_list), dtype=float)
        for out_i, species in enumerate(output_species_list):
            idx = name_to_index[species]
            if output_mode == "Cout (mol/m^3)":
                output_values[out_i] = conc_final[idx]
            elif output_mode == "X (conversion)":
                c0 = conc_initial[idx]
                c_final = conc_final[idx]
                if c0 < 1e-30:
                    output_values[out_i] = np.nan
                else:
                    output_values[out_i] = (c0 - c_final) / c0
            else:
                # Batch 不支持 Fout 模式
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "Batch 反应器不支持 Fout 输出模式，请选择 Cout 或 X",
                )

    else:
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            f"未知反应器类型: {reactor_type}",
        )

    return output_values, True, "OK"
