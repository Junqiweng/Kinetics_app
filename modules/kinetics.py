# 文件作用：实现不同动力学模型（幂律、L-H、可逆反应）的速率矢量计算，并提供必要的数值保护函数。
# 性能说明：速率计算已向量化（NumPy 广播），避免在 ODE 热路径上使用 Python for 循环。

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


def _vectorized_concentration_product(
    conc_mol_m3: np.ndarray,
    order_matrix: np.ndarray,
    label: str = "",
) -> np.ndarray:
    """
    向量化计算 Π_i C_i^(n_ij)，返回 shape (n_reactions,)。

    步骤：
    1. 对负级数位置，将浓度下限保护到 EPSILON_CONCENTRATION
    2. 对零级数位置，跳过计算（贡献因子 = 1）
    3. 使用 np.prod 沿 species 轴求积

    参数:
        conc_mol_m3: 浓度向量 (n_species,)
        order_matrix: 反应级数矩阵 (n_reactions, n_species)
        label: 错误消息前缀（用于调试）
    """
    # conc_safe: 对负级数的物种用 EPSILON_CONCENTRATION 保护
    # 广播形状: conc_mol_m3 (n_species,) -> (1, n_species)
    conc_row = conc_mol_m3[np.newaxis, :]  # (1, n_species)
    neg_order_mask = order_matrix < 0.0
    conc_protected = np.where(
        neg_order_mask,
        np.maximum(conc_row, EPSILON_CONCENTRATION),
        conc_row,
    )  # (n_reactions, n_species)

    # 对零级数位置跳过（C^0 = 1）：将这些位置的浓度设为 1，级数设为 0
    zero_order_mask = np.abs(order_matrix) < FLOAT_EQUALITY_TOLERANCE
    conc_for_pow = np.where(zero_order_mask, 1.0, conc_protected)
    order_for_pow = np.where(zero_order_mask, 0.0, order_matrix)

    # 向量化幂运算 + 沿 species 轴求积
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        factors = conc_for_pow**order_for_pow  # (n_reactions, n_species)

    # 检查是否有非有限值
    if not np.all(np.isfinite(factors)):
        bad = np.argwhere(~np.isfinite(factors))
        r_idx, s_idx = int(bad[0, 0]), int(bad[0, 1])
        raise FloatingPointError(
            f"{label}: rate 非有限（reaction={r_idx}, species={s_idx}）"
        )

    product = np.prod(factors, axis=1)  # (n_reactions,)
    return product


def _assert_finite_rate_vector(rate_vector: np.ndarray, label: str) -> None:
    """检查速率向量是否全部为有限值，避免 NaN/Inf 进入 ODE/拟合流程。"""
    if not np.all(np.isfinite(rate_vector)):
        bad_idx = int(np.argwhere(~np.isfinite(rate_vector))[0, 0])
        raise FloatingPointError(f"{label}: rate 非有限（reaction={bad_idx}）")


def calc_rate_vector_power_law(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    *,
    k_T_precomputed: np.ndarray | None = None,
) -> np.ndarray:
    """
    参数说明：
        conc_mol_m3：浓度向量，形状 (n_species,) [mol/m³]
        k0：指前因子向量，形状 (n_reactions,)
        ea_J_mol：活化能向量，形状 (n_reactions,) [J/mol]
        reaction_order_matrix：反应级数矩阵，形状 (n_reactions, n_species)
        k_T_precomputed：可选，预计算的 k(T) 向量（等温优化，跳过 Arrhenius）
    """
    conc_mol_m3 = safe_nonnegative(conc_mol_m3)

    if k_T_precomputed is not None:
        k_T = k_T_precomputed
    else:
        k_T = k0 * np.exp(-ea_J_mol / (R_GAS_J_MOL_K * temperature_K))
        if not np.all(np.isfinite(k_T)):
            raise FloatingPointError("power_law: Arrhenius 计算得到 NaN/Inf（k_T）")

    # 速率表达式：rate_j = k_j(T) * Π_i C_i^(n_ij)  — 向量化计算
    conc_product = _vectorized_concentration_product(
        conc_mol_m3, reaction_order_matrix, label="power_law"
    )
    rate_vector = k_T * conc_product
    _assert_finite_rate_vector(rate_vector, "power_law")
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
    *,
    k_T_precomputed: np.ndarray | None = None,
    K_ads_T_precomputed: np.ndarray | None = None,
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
        k_T_precomputed: 可选，预计算的 k(T) 向量（等温优化）
        K_ads_T_precomputed: 可选，预计算的 K_ads(T) 向量（等温优化）

    返回:
        rate_vector: 反应速率向量 (n_reactions,)
    """
    conc_mol_m3 = safe_nonnegative(conc_mol_m3)

    if k_T_precomputed is not None:
        k_T = k_T_precomputed
    else:
        k_T = k0 * np.exp(-ea_J_mol / (R_GAS_J_MOL_K * temperature_K))
        if not np.all(np.isfinite(k_T)):
            raise FloatingPointError(
                "langmuir_hinshelwood: Arrhenius 计算得到 NaN/Inf（k_T）"
            )

    # 计算温度依赖的吸附常数 K_i(T)
    if K_ads_T_precomputed is not None:
        K_ads_T = K_ads_T_precomputed
    else:
        K_ads_T = K0_ads * np.exp(-Ea_K_J_mol / (R_GAS_J_MOL_K * temperature_K))
        if not np.all(np.isfinite(K_ads_T)):
            raise FloatingPointError(
                "langmuir_hinshelwood: Arrhenius 计算得到 NaN/Inf（K_ads_T）"
            )

    # 计算分母基底：(1 + Σ_i K_i(T) * C_i)
    denominator_base = 1.0 + np.sum(K_ads_T * conc_mol_m3)
    if not np.isfinite(denominator_base):
        raise FloatingPointError(
            "langmuir_hinshelwood: 分母基底得到 NaN/Inf（denominator_base）"
        )

    # 分子：k_j(T) * Π_i C_i^(n_ij)  — 向量化
    conc_product = _vectorized_concentration_product(
        conc_mol_m3, reaction_order_matrix, label="langmuir_hinshelwood"
    )
    rate_numerator = k_T * conc_product  # (n_reactions,)

    # 分母：(1 + Σ_i K_i(T) * C_i)^m_j  — 向量化
    m_active_mask = np.abs(m_inhibition) >= FLOAT_EQUALITY_TOLERANCE
    denominator = np.where(m_active_mask, denominator_base**m_inhibition, 1.0)
    if not np.all(np.isfinite(denominator)):
        bad_idx = int(np.argwhere(~np.isfinite(denominator))[0, 0])
        raise FloatingPointError(
            f"langmuir_hinshelwood: 分母非有限（reaction={bad_idx}）"
        )

    rate_vector = rate_numerator / np.maximum(denominator, EPSILON_DENOMINATOR)
    _assert_finite_rate_vector(rate_vector, "langmuir_hinshelwood")
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
    *,
    k_fwd_T_precomputed: np.ndarray | None = None,
    k_rev_T_precomputed: np.ndarray | None = None,
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
        k_fwd_T_precomputed: 可选，预计算的正反应 k(T)（等温优化）
        k_rev_T_precomputed: 可选，预计算的逆反应 k(T)（等温优化）

    返回:
        rate_vector: 净反应速率向量 (n_reactions,)
    """
    conc_mol_m3 = safe_nonnegative(conc_mol_m3)

    if k_fwd_T_precomputed is not None:
        k_fwd_T = k_fwd_T_precomputed
    else:
        k_fwd_T = k0_fwd * np.exp(-ea_fwd_J_mol / (R_GAS_J_MOL_K * temperature_K))
        if not np.all(np.isfinite(k_fwd_T)):
            raise FloatingPointError(
                "reversible: Arrhenius 计算得到 NaN/Inf（k_fwd_T）"
            )

    if k_rev_T_precomputed is not None:
        k_rev_T = k_rev_T_precomputed
    else:
        k_rev_T = k0_rev * np.exp(-ea_rev_J_mol / (R_GAS_J_MOL_K * temperature_K))
        if not np.all(np.isfinite(k_rev_T)):
            raise FloatingPointError(
                "reversible: Arrhenius 计算得到 NaN/Inf（k_rev_T）"
            )

    # 正反应速率（向量化）
    conc_product_fwd = _vectorized_concentration_product(
        conc_mol_m3, order_fwd_matrix, label="reversible_fwd"
    )
    rate_fwd = k_fwd_T * conc_product_fwd

    # 逆反应速率（向量化）
    conc_product_rev = _vectorized_concentration_product(
        conc_mol_m3, order_rev_matrix, label="reversible_rev"
    )
    rate_rev = k_rev_T * conc_product_rev

    # 净反应速率
    rate_vector = rate_fwd - rate_rev
    _assert_finite_rate_vector(rate_vector, "reversible")
    return rate_vector
