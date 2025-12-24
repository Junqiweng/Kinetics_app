from __future__ import annotations

import numpy as np

from .constants import (
    R_GAS_J_MOL_K,
    EPSILON_CONCENTRATION,
    EPSILON_DENOMINATOR,
    FLOAT_EQUALITY_TOLERANCE,
)


def safe_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def calc_rate_vector_power_law(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
) -> np.ndarray:
    """
    conc_mol_m3: shape (n_species,)
    k0: shape (n_reactions,)   pre-exponential factor
    ea_J_mol: shape (n_reactions,) activation energy [J/mol]
    reaction_order_matrix: shape (n_reactions, n_species)
    """
    conc_mol_m3 = safe_nonnegative(conc_mol_m3)
    k_T = k0 * np.exp(-ea_J_mol / (R_GAS_J_MOL_K * temperature_K))

    # rate_j = k_j(T) * Π_i C_i^(n_ij)
    n_reactions = k0.size
    rate_vector = np.zeros(n_reactions, dtype=float)
    for reaction_index in range(n_reactions):
        rate_value = k_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = reaction_order_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, EPSILON_CONCENTRATION)
            rate_value = rate_value * (conc_value**order_value)
        rate_vector[reaction_index] = rate_value
    return rate_vector


def calc_rate_vector_langmuir_hinshelwood(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    K0_ads: np.ndarray,
    Ea_K_J_mol: np.ndarray,
    m_inhibition: np.ndarray,
) -> np.ndarray:
    """
    Langmuir-Hinshelwood 动力学速率计算。

    r_j = k_j(T) * Π_i C_i^(n_ij) / (1 + Σ_i K_i(T) * C_i)^m_j

    其中吸附常数也遵循 Arrhenius 方程：
    K_i(T) = K_{0,i} * exp(-Ea_{K,i} / RT)

    参数:
        conc_mol_m3: 浓度向量 (n_species,) [mol/m³]
        temperature_K: 温度 [K]
        k0: 指前因子 (n_reactions,)
        ea_J_mol: 活化能 (n_reactions,) [J/mol]
        reaction_order_matrix: 反应级数矩阵 (n_reactions, n_species)
        K0_ads: 吸附常数指前因子 (n_species,)
        Ea_K_J_mol: 吸附热 (n_species,) [J/mol]，可为负值（放热吸附）
        m_inhibition: 抑制指数 (n_reactions,)

    返回:
        rate_vector: 反应速率向量 (n_reactions,)
    """
    conc_mol_m3 = safe_nonnegative(conc_mol_m3)
    k_T = k0 * np.exp(-ea_J_mol / (R_GAS_J_MOL_K * temperature_K))

    # 计算温度依赖的吸附常数 K_i(T)
    K_ads_T = K0_ads * np.exp(-Ea_K_J_mol / (R_GAS_J_MOL_K * temperature_K))

    # 计算分母：(1 + Σ_i K_i(T) * C_i)
    denominator_base = 1.0 + np.sum(K_ads_T * conc_mol_m3)

    n_reactions = k0.size
    rate_vector = np.zeros(n_reactions, dtype=float)
    for reaction_index in range(n_reactions):
        # 分子：k_j(T) * Π_i C_i^(n_ij)
        rate_numerator = k_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = reaction_order_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, EPSILON_CONCENTRATION)
            rate_numerator = rate_numerator * (conc_value**order_value)

        # 分母：(1 + Σ_i K_i(T) * C_i)^m_j
        m_j = m_inhibition[reaction_index]
        denominator = denominator_base**m_j if m_j != 0.0 else 1.0

        rate_vector[reaction_index] = rate_numerator / max(
            denominator, EPSILON_DENOMINATOR
        )

    return rate_vector


def calc_rate_vector_reversible(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    k0_fwd: np.ndarray,
    ea_fwd_J_mol: np.ndarray,
    order_fwd_matrix: np.ndarray,
    k0_rev: np.ndarray,
    ea_rev_J_mol: np.ndarray,
    order_rev_matrix: np.ndarray,
) -> np.ndarray:
    """
    可逆反应动力学速率计算。

    r_j = r_j^+ - r_j^-
        = k_j^+(T) * Π_i C_i^(n_ij^+) - k_j^-(T) * Π_i C_i^(n_ij^-)

    参数:
        conc_mol_m3: 浓度向量 (n_species,) [mol/m³]
        temperature_K: 温度 [K]
        k0_fwd, ea_fwd_J_mol: 正反应指前因子和活化能
        order_fwd_matrix: 正反应级数矩阵 (n_reactions, n_species)
        k0_rev, ea_rev_J_mol: 逆反应指前因子和活化能
        order_rev_matrix: 逆反应级数矩阵 (n_reactions, n_species)

    返回:
        rate_vector: 净反应速率向量 (n_reactions,)
    """
    conc_mol_m3 = safe_nonnegative(conc_mol_m3)
    k_fwd_T = k0_fwd * np.exp(-ea_fwd_J_mol / (R_GAS_J_MOL_K * temperature_K))
    k_rev_T = k0_rev * np.exp(-ea_rev_J_mol / (R_GAS_J_MOL_K * temperature_K))

    n_reactions = k0_fwd.size
    rate_vector = np.zeros(n_reactions, dtype=float)

    for reaction_index in range(n_reactions):
        # 正反应速率
        rate_fwd = k_fwd_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = order_fwd_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, EPSILON_CONCENTRATION)
            rate_fwd = rate_fwd * (conc_value**order_value)

        # 逆反应速率
        rate_rev = k_rev_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = order_rev_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, EPSILON_CONCENTRATION)
            rate_rev = rate_rev * (conc_value**order_value)

        # 净反应速率
        rate_vector[reaction_index] = rate_fwd - rate_rev

    return rate_vector
