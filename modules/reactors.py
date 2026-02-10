# 文件作用：反应器模型求解（PFR、CSTR、BSTR），通过 `solve_ivp`/`least_squares` 计算出口组成或时间剖面。

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .constants import (
    DEFAULT_CSTR_STEADY_FTOL,
    DEFAULT_CSTR_STEADY_GTOL,
    DEFAULT_CSTR_STEADY_MAX_NFEV,
    DEFAULT_CSTR_STEADY_XTOL,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_PROFILE_N_POINTS,
    EPSILON_DENOMINATOR,
    EPSILON_FLOW_RATE,
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODEL_POWER_LAW,
    KINETIC_MODEL_REVERSIBLE,
    R_GAS_J_MOL_K,
)
from .kinetics import (
    calc_rate_vector_langmuir_hinshelwood,
    calc_rate_vector_power_law,
    calc_rate_vector_reversible,
    safe_nonnegative,
)


def _precompute_arrhenius(
    temperature_K: float,
    kinetic_model: str,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
) -> dict:
    """
    等温条件下预计算 Arrhenius 常数，避免在 ODE 每步重复计算 exp()。

    返回 dict，键名与 calc_rate_vector_* 的 `*_precomputed` 参数对应。
    """
    RT = R_GAS_J_MOL_K * temperature_K
    pre = {}
    k_T = k0 * np.exp(-ea_J_mol / RT)
    pre["k_T"] = k_T

    if kinetic_model == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
        if K0_ads is not None and Ea_K_J_mol is not None:
            pre["K_ads_T"] = K0_ads * np.exp(-Ea_K_J_mol / RT)

    if kinetic_model == KINETIC_MODEL_REVERSIBLE:
        if k0_rev is not None and ea_rev_J_mol is not None:
            pre["k_rev_T"] = k0_rev * np.exp(-ea_rev_J_mol / RT)

    return pre


def _dispatch_rate_vector(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    kinetic_model: str,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    K0_ads: np.ndarray | None,
    Ea_K_J_mol: np.ndarray | None,
    m_inhibition: np.ndarray | None,
    k0_rev: np.ndarray | None,
    ea_rev_J_mol: np.ndarray | None,
    order_rev_matrix: np.ndarray | None,
    precomputed: dict | None = None,
) -> np.ndarray:
    """
    统一分派不同动力学模型的速率计算，并传入预计算的 Arrhenius 常数。
    减少在每个 ODE 闭包中重复的 if/elif 分支。
    """
    pre = precomputed or {}

    if kinetic_model == KINETIC_MODEL_POWER_LAW:
        return calc_rate_vector_power_law(
            conc_mol_m3=conc_mol_m3,
            temperature_K=temperature_K,
            k0=k0,
            ea_J_mol=ea_J_mol,
            reaction_order_matrix=reaction_order_matrix,
            k_T_precomputed=pre.get("k_T"),
        )
    if kinetic_model == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
        return calc_rate_vector_langmuir_hinshelwood(
            conc_mol_m3=conc_mol_m3,
            temperature_K=temperature_K,
            k0=k0,
            ea_J_mol=ea_J_mol,
            reaction_order_matrix=reaction_order_matrix,
            K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_mol_m3.size),
            Ea_K_J_mol=(
                Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_mol_m3.size)
            ),
            m_inhibition=(
                m_inhibition if m_inhibition is not None else np.ones(k0.size)
            ),
            k_T_precomputed=pre.get("k_T"),
            K_ads_T_precomputed=pre.get("K_ads_T"),
        )
    if kinetic_model == KINETIC_MODEL_REVERSIBLE:
        return calc_rate_vector_reversible(
            conc_mol_m3=conc_mol_m3,
            temperature_K=temperature_K,
            k0_fwd=k0,
            ea_fwd_J_mol=ea_J_mol,
            order_fwd_matrix=reaction_order_matrix,
            k0_rev=k0_rev if k0_rev is not None else np.zeros(k0.size),
            ea_rev_J_mol=(
                ea_rev_J_mol if ea_rev_J_mol is not None else np.zeros(k0.size)
            ),
            order_rev_matrix=(
                order_rev_matrix
                if order_rev_matrix is not None
                else np.zeros_like(reaction_order_matrix)
            ),
            k_fwd_T_precomputed=pre.get("k_T"),
            k_rev_T_precomputed=pre.get("k_rev_T"),
        )
    # 默认回退到 power_law
    return calc_rate_vector_power_law(
        conc_mol_m3=conc_mol_m3,
        temperature_K=temperature_K,
        k0=k0,
        ea_J_mol=ea_J_mol,
        reaction_order_matrix=reaction_order_matrix,
        k_T_precomputed=pre.get("k_T"),
    )


def _compute_max_step(total_span: float, max_step_fraction: float | None) -> float:
    """
    将 “max_step_fraction” 转为 solve_ivp 的 max_step。

    例：max_step_fraction=0.1 表示 max_step = 0.1 * total_span。
    - 若为 0 或 None：不限制（solve_ivp 默认 np.inf）
    """
    if max_step_fraction is None:
        return np.inf
    try:
        fraction = float(max_step_fraction)
    except (ValueError, TypeError):
        return np.inf
    if (not np.isfinite(fraction)) or (fraction <= 0.0):
        return np.inf
    return float(total_span) * fraction


def integrate_pfr_molar_flows(
    reactor_volume_m3: float,
    temperature_K: float,
    vdot_m3_s: float,
    molar_flow_inlet_mol_s: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """
    PFR 设计方程（液相/体积流量近似恒定）：
      dF_i/dV = Σ_j ν_{i,j} r_j
      C_i = F_i / vdot
    """
    if not np.isfinite(reactor_volume_m3):
        return molar_flow_inlet_mol_s.copy(), False, "V_m3 无效（NaN/Inf）"
    if reactor_volume_m3 < 0.0:
        return molar_flow_inlet_mol_s.copy(), False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        return molar_flow_inlet_mol_s.copy(), True, "V=0"

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return molar_flow_inlet_mol_s.copy(), False, "温度 T_K 无效"
    if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
        return molar_flow_inlet_mol_s.copy(), False, "体积流量 vdot_m3_s 无效"

    if not np.all(np.isfinite(molar_flow_inlet_mol_s)):
        return molar_flow_inlet_mol_s.copy(), False, "入口摩尔流量包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return molar_flow_inlet_mol_s.copy(), False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return molar_flow_inlet_mol_s.copy(), False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return molar_flow_inlet_mol_s.copy(), False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return molar_flow_inlet_mol_s.copy(), False, "反应级数矩阵 n 包含 NaN/Inf"

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            not np.isfinite(max_wall_time_s) or max_wall_time_s <= 0.0
        ):
            max_wall_time_s = None

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_mol_m3 = safe_nonnegative(molar_flow_mol_s) / max(
            vdot_m3_s, EPSILON_FLOW_RATE
        )

        rate_vector = _dispatch_rate_vector(
            conc_mol_m3,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

        dF_dV = stoich_matrix @ rate_vector
        return dF_dV

    max_step_value = _compute_max_step(float(reactor_volume_m3), max_step_fraction)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reactor_volume_m3)),
            y0=molar_flow_inlet_mol_s.astype(float),
            method=solver_method,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )
    except (
        Exception
    ) as exc:  # scipy 可能抛出多种异常（ValueError, RuntimeError等），宽泛捕获确保稳定性
        return molar_flow_inlet_mol_s.copy(), False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return molar_flow_inlet_mol_s.copy(), False, str(message)

    molar_flow_outlet = solution.y[:, -1]
    if not np.all(np.isfinite(molar_flow_outlet)):
        return molar_flow_inlet_mol_s.copy(), False, "solve_ivp 输出包含 NaN/Inf"
    return molar_flow_outlet, True, "OK"


def integrate_pfr_molar_flows_gas_ideal_const_p(
    reactor_volume_m3: float,
    temperature_K: float,
    pressure_Pa: float,
    molar_flow_inlet_mol_s: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """
    气相 PFR 设计方程（理想气体、等温、恒压 P，不考虑压降）：

      dF_i/dV = Σ_j ν_{i,j} r_j

    动力学所需浓度由摩尔分率计算：

      y_i = F_i / Σ_k F_k
      C_tot = P / (R T)
      C_i = y_i * C_tot

    注意：
    - 这里的 “恒压” 指每一条实验数据行内，沿程 P 不变（不建压降方程）。
    - temperature_K 与 pressure_Pa 在积分过程中视为常数。
    """
    if not np.isfinite(reactor_volume_m3):
        return molar_flow_inlet_mol_s.copy(), False, "V_m3 无效（NaN/Inf）"
    if reactor_volume_m3 < 0.0:
        return molar_flow_inlet_mol_s.copy(), False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        return molar_flow_inlet_mol_s.copy(), True, "V=0"

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return molar_flow_inlet_mol_s.copy(), False, "温度 T_K 无效"
    if (not np.isfinite(pressure_Pa)) or (pressure_Pa <= 0.0):
        return molar_flow_inlet_mol_s.copy(), False, "压力 P_Pa 无效"

    if not np.all(np.isfinite(molar_flow_inlet_mol_s)):
        return molar_flow_inlet_mol_s.copy(), False, "入口摩尔流量包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return molar_flow_inlet_mol_s.copy(), False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return molar_flow_inlet_mol_s.copy(), False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return molar_flow_inlet_mol_s.copy(), False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return molar_flow_inlet_mol_s.copy(), False, "反应级数矩阵 n 包含 NaN/Inf"

    # C_tot = P/(R*T) 在该模型内为常数（等温、恒压）
    conc_total_mol_m3 = float(pressure_Pa) / max(
        float(R_GAS_J_MOL_K) * float(temperature_K), EPSILON_DENOMINATOR
    )

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            not np.isfinite(max_wall_time_s) or max_wall_time_s <= 0.0
        ):
            max_wall_time_s = None

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        flow_safe = safe_nonnegative(molar_flow_mol_s)
        total_flow = float(np.sum(flow_safe))
        y = flow_safe / max(total_flow, EPSILON_FLOW_RATE)
        conc_mol_m3 = y * float(conc_total_mol_m3)

        rate_vector = _dispatch_rate_vector(
            conc_mol_m3,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

        dF_dV = stoich_matrix @ rate_vector
        return dF_dV

    max_step_value = _compute_max_step(float(reactor_volume_m3), max_step_fraction)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reactor_volume_m3)),
            y0=molar_flow_inlet_mol_s.astype(float),
            method=solver_method,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )
    except Exception as exc:
        return molar_flow_inlet_mol_s.copy(), False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return molar_flow_inlet_mol_s.copy(), False, str(message)

    molar_flow_outlet = solution.y[:, -1]
    if not np.all(np.isfinite(molar_flow_outlet)):
        return molar_flow_inlet_mol_s.copy(), False, "solve_ivp 输出包含 NaN/Inf"
    return molar_flow_outlet, True, "OK"


def integrate_batch_reactor(
    reaction_time_s: float,
    temperature_K: float,
    conc_initial_mol_m3: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    """
    BSTR reactor ODE:
      dC_i/dt = Σ_j nu_{i,j} r_j
    """
    if not np.isfinite(reaction_time_s):
        return conc_initial_mol_m3.copy(), False, "t_s 无效（NaN/Inf）"
    if reaction_time_s < 0.0:
        return conc_initial_mol_m3.copy(), False, "t_s 不能为负"
    if reaction_time_s == 0.0:
        return conc_initial_mol_m3.copy(), True, "t=0"

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return conc_initial_mol_m3.copy(), False, "温度 T_K 无效"

    if not np.all(np.isfinite(conc_initial_mol_m3)):
        return conc_initial_mol_m3.copy(), False, "初始浓度包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return conc_initial_mol_m3.copy(), False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return conc_initial_mol_m3.copy(), False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return conc_initial_mol_m3.copy(), False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return conc_initial_mol_m3.copy(), False, "反应级数矩阵 n 包含 NaN/Inf"

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            (not np.isfinite(max_wall_time_s)) or (max_wall_time_s <= 0.0)
        ):
            max_wall_time_s = None

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_safe = safe_nonnegative(conc_mol_m3)

        rate_vector = _dispatch_rate_vector(
            conc_safe,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

        dC_dt = stoich_matrix @ rate_vector
        return dC_dt

    max_step_value = _compute_max_step(float(reaction_time_s), max_step_fraction)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reaction_time_s)),
            y0=conc_initial_mol_m3.astype(float),
            method=solver_method,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )
    except (
        Exception
    ) as exc:  # scipy 可能抛出多种异常（ValueError, RuntimeError等），宽泛捕获确保稳定性
        return conc_initial_mol_m3.copy(), False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return conc_initial_mol_m3.copy(), False, str(message)

    conc_final = solution.y[:, -1]
    if not np.all(np.isfinite(conc_final)):
        return conc_initial_mol_m3.copy(), False, "solve_ivp 输出包含 NaN/Inf"
    return conc_final, True, "OK"


def solve_cstr_steady_state_concentrations(
    reactor_volume_m3: float,
    temperature_K: float,
    vdot_m3_s: float,
    conc_inlet_mol_m3: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    max_nfev: int = DEFAULT_CSTR_STEADY_MAX_NFEV,
    xtol: float = DEFAULT_CSTR_STEADY_XTOL,
    ftol: float = DEFAULT_CSTR_STEADY_FTOL,
    gtol: float = DEFAULT_CSTR_STEADY_GTOL,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, bool, str]:
    r"""
    CSTR 稳态物料衡算（液相/体积流量恒定）：

    $0 = v \cdot C_{0,i} - v \cdot C_i + V \sum_j \nu_{i,j} r_j(C,T)$

    等价为：
    $C_i = C_{0,i} + \tau \sum_j \nu_{i,j} r_j(C,T)$，其中 $\tau = V/v$。

    返回：
        conc_outlet_mol_m3: shape (n_species,)
    """
    if not np.isfinite(reactor_volume_m3):
        return conc_inlet_mol_m3.copy(), False, "V_m3 无效（NaN/Inf）"
    if reactor_volume_m3 < 0.0:
        return conc_inlet_mol_m3.copy(), False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        return conc_inlet_mol_m3.copy(), True, "V=0"

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return conc_inlet_mol_m3.copy(), False, "温度 T_K 无效"
    if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
        return conc_inlet_mol_m3.copy(), False, "体积流量 vdot_m3_s 无效"

    if not np.all(np.isfinite(conc_inlet_mol_m3)):
        return conc_inlet_mol_m3.copy(), False, "入口浓度包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return conc_inlet_mol_m3.copy(), False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return conc_inlet_mol_m3.copy(), False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return conc_inlet_mol_m3.copy(), False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return conc_inlet_mol_m3.copy(), False, "反应级数矩阵 n 包含 NaN/Inf"

    try:
        max_nfev = int(max_nfev)
    except (ValueError, TypeError):
        max_nfev = DEFAULT_CSTR_STEADY_MAX_NFEV
    if max_nfev <= 0:
        max_nfev = DEFAULT_CSTR_STEADY_MAX_NFEV

    tau_s = float(reactor_volume_m3) / max(
        float(vdot_m3_s), EPSILON_FLOW_RATE
    )  # Residence time [s]
    if (not np.isfinite(tau_s)) or (tau_s < 0.0):
        return conc_inlet_mol_m3.copy(), False, "停留时间 tau_s 无效"
    if tau_s == 0.0:
        return conc_inlet_mol_m3.copy(), True, "tau=0"

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            (not np.isfinite(max_wall_time_s)) or (max_wall_time_s <= 0.0)
        ):
            max_wall_time_s = None

    conc_inlet_mol_m3 = conc_inlet_mol_m3.astype(float)

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def _rate_vector(conc_mol_m3: np.ndarray) -> np.ndarray:
        conc_safe = safe_nonnegative(conc_mol_m3)
        return _dispatch_rate_vector(
            conc_safe,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

    def residual_fun(conc_guess_mol_m3: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"稳态求解超时（>{max_wall_time_s:.1f} s）")

        rate_vector = _rate_vector(conc_guess_mol_m3)
        source_term_mol_m3_s = stoich_matrix @ rate_vector
        residual = conc_guess_mol_m3 - conc_inlet_mol_m3 - tau_s * source_term_mol_m3_s
        return residual

    x0 = safe_nonnegative(conc_inlet_mol_m3).copy()
    x0 = np.maximum(x0, 0.0)
    try:
        result = least_squares(
            residual_fun,
            x0=x0,
            bounds=(0.0, np.inf),
            method="trf",
            max_nfev=max_nfev,
            xtol=float(xtol),
            ftol=float(ftol),
            gtol=float(gtol),
        )
    except (
        Exception
    ) as exc:  # scipy 可能抛出多种异常（ValueError, RuntimeError等），宽泛捕获确保稳定性
        return conc_inlet_mol_m3.copy(), False, f"CSTR 稳态求解异常: {exc}"

    conc_outlet_mol_m3 = result.x.astype(float)
    if not np.all(np.isfinite(conc_outlet_mol_m3)):
        return conc_inlet_mol_m3.copy(), False, "CSTR 稳态求解得到 NaN/Inf"

    if not bool(getattr(result, "success", False)):
        message = str(getattr(result, "message", "CSTR 稳态求解失败"))
        return conc_outlet_mol_m3, False, message

    return conc_outlet_mol_m3, True, "OK"


def integrate_cstr_profile(
    simulation_time_s: float,
    temperature_K: float,
    reactor_volume_m3: float,
    vdot_m3_s: float,
    conc_inlet_mol_m3: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    n_points: int = DEFAULT_PROFILE_N_POINTS,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    r"""
    CSTR 动态方程（用于画随时间剖面）：

    $dC_i/dt = (C_{0,i} - C_i)/\tau + \sum_j \nu_{i,j} r_j(C,T)$

    其中 $\tau = V/v$。

    返回：
        time_grid_s: shape (n_points,)
        conc_profile_mol_m3: shape (n_species, n_points)
    """
    n_points = int(n_points)
    if n_points < 2:
        n_points = 2

    if not np.isfinite(simulation_time_s):
        return (
            np.array([0.0]),
            conc_inlet_mol_m3[:, None],
            False,
            "t_s 无效（NaN/Inf）",
        )
    if simulation_time_s < 0.0:
        return np.array([0.0]), conc_inlet_mol_m3[:, None], False, "t_s 不能为负"
    if simulation_time_s == 0.0:
        return (
            np.array([0.0], dtype=float),
            conc_inlet_mol_m3.astype(float)[:, None],
            True,
            "t=0",
        )

    if not np.isfinite(reactor_volume_m3):
        return (
            np.array([0.0]),
            conc_inlet_mol_m3[:, None],
            False,
            "V_m3 无效（NaN/Inf）",
        )
    if reactor_volume_m3 < 0.0:
        return np.array([0.0]), conc_inlet_mol_m3[:, None], False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        time_grid_s = np.linspace(0.0, float(simulation_time_s), n_points, dtype=float)
        conc_profile = np.repeat(
            conc_inlet_mol_m3.astype(float)[:, None], n_points, axis=1
        )
        return time_grid_s, conc_profile, True, "V=0"

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return np.array([0.0]), conc_inlet_mol_m3[:, None], False, "温度 T_K 无效"
    if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
        return (
            np.array([0.0]),
            conc_inlet_mol_m3[:, None],
            False,
            "体积流量 vdot_m3_s 无效",
        )

    if not np.all(np.isfinite(conc_inlet_mol_m3)):
        return (
            np.array([0.0]),
            conc_inlet_mol_m3[:, None],
            False,
            "入口浓度包含 NaN/Inf",
        )
    if not np.all(np.isfinite(stoich_matrix)):
        return (
            np.array([0.0]),
            conc_inlet_mol_m3[:, None],
            False,
            "化学计量数矩阵 ν 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(k0)):
        return np.array([0.0]), conc_inlet_mol_m3[:, None], False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return np.array([0.0]), conc_inlet_mol_m3[:, None], False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return (
            np.array([0.0]),
            conc_inlet_mol_m3[:, None],
            False,
            "反应级数矩阵 n 包含 NaN/Inf",
        )

    tau_s = float(reactor_volume_m3) / max(float(vdot_m3_s), EPSILON_FLOW_RATE)
    if (not np.isfinite(tau_s)) or (tau_s <= 0.0):
        return np.array([0.0]), conc_inlet_mol_m3[:, None], False, "停留时间 tau_s 无效"

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            not np.isfinite(max_wall_time_s) or max_wall_time_s <= 0.0
        ):
            max_wall_time_s = None

    conc_inlet_mol_m3 = conc_inlet_mol_m3.astype(float)

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_safe = safe_nonnegative(conc_mol_m3)

        rate_vector = _dispatch_rate_vector(
            conc_safe,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

        reaction_source_mol_m3_s = stoich_matrix @ rate_vector
        mixing_term_mol_m3_s = (conc_inlet_mol_m3 - conc_safe) / max(
            tau_s, EPSILON_DENOMINATOR
        )
        dC_dt = mixing_term_mol_m3_s + reaction_source_mol_m3_s
        return dC_dt

    time_grid_s = np.linspace(0.0, float(simulation_time_s), n_points, dtype=float)
    max_step_value = _compute_max_step(float(simulation_time_s), max_step_fraction)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(simulation_time_s)),
            y0=conc_inlet_mol_m3.astype(float),
            method=solver_method,
            t_eval=time_grid_s,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )
    except (
        Exception
    ) as exc:  # scipy 可能抛出多种异常（ValueError, RuntimeError等），宽泛捕获确保稳定性
        return (
            time_grid_s,
            conc_inlet_mol_m3.astype(float)[:, None],
            False,
            f"solve_ivp异常: {exc}",
        )

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return (
            time_grid_s,
            conc_inlet_mol_m3.astype(float)[:, None],
            False,
            str(message),
        )

    if (not np.all(np.isfinite(solution.t))) or (not np.all(np.isfinite(solution.y))):
        return (
            time_grid_s,
            conc_inlet_mol_m3.astype(float)[:, None],
            False,
            "solve_ivp 输出包含 NaN/Inf",
        )

    return solution.t.astype(float), solution.y.astype(float), True, "OK"


def integrate_pfr_profile(
    reactor_volume_m3: float,
    temperature_K: float,
    vdot_m3_s: float,
    molar_flow_inlet_mol_s: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    n_points: int = DEFAULT_PROFILE_N_POINTS,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    返回 PFR 沿程剖面：
      volume_grid_m3: shape (n_points,)
      molar_flow_profile_mol_s: shape (n_species, n_points)
    """
    n_points = int(n_points)
    if n_points < 2:
        n_points = 2

    if not np.isfinite(reactor_volume_m3):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "V_m3 无效（NaN/Inf）",
        )
    if reactor_volume_m3 < 0.0:
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        return (
            np.array([0.0], dtype=float),
            molar_flow_inlet_mol_s.astype(float)[:, None],
            True,
            "V=0",
        )

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "温度 T_K 无效"
    if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "体积流量 vdot_m3_s 无效",
        )

    if not np.all(np.isfinite(molar_flow_inlet_mol_s)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "入口摩尔流量包含 NaN/Inf",
        )
    if not np.all(np.isfinite(stoich_matrix)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "化学计量数矩阵 ν 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(k0)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "k0 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(ea_J_mol)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "Ea 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(reaction_order_matrix)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "反应级数矩阵 n 包含 NaN/Inf",
        )

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            (not np.isfinite(max_wall_time_s)) or (max_wall_time_s <= 0.0)
        ):
            max_wall_time_s = None

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_mol_m3 = safe_nonnegative(molar_flow_mol_s) / max(
            vdot_m3_s, EPSILON_FLOW_RATE
        )

        rate_vector = _dispatch_rate_vector(
            conc_mol_m3,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

        dF_dV = stoich_matrix @ rate_vector
        return dF_dV

    volume_grid_m3 = np.linspace(0.0, float(reactor_volume_m3), n_points, dtype=float)

    max_step_value = _compute_max_step(float(reactor_volume_m3), max_step_fraction)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reactor_volume_m3)),
            y0=molar_flow_inlet_mol_s.astype(float),
            method=solver_method,
            t_eval=volume_grid_m3,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )
    except (
        Exception
    ) as exc:  # scipy 可能抛出多种异常（ValueError, RuntimeError等），宽泛捕获确保稳定性
        return (
            volume_grid_m3,
            molar_flow_inlet_mol_s.astype(float)[:, None],
            False,
            f"solve_ivp异常: {exc}",
        )

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return (
            volume_grid_m3,
            molar_flow_inlet_mol_s.astype(float)[:, None],
            False,
            str(message),
        )

    if (not np.all(np.isfinite(solution.t))) or (not np.all(np.isfinite(solution.y))):
        return (
            volume_grid_m3,
            molar_flow_inlet_mol_s.astype(float)[:, None],
            False,
            "solve_ivp 输出包含 NaN/Inf",
        )

    return solution.t.astype(float), solution.y.astype(float), True, "OK"


def integrate_pfr_profile_gas_ideal_const_p(
    reactor_volume_m3: float,
    temperature_K: float,
    pressure_Pa: float,
    molar_flow_inlet_mol_s: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    n_points: int = DEFAULT_PROFILE_N_POINTS,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    返回气相 PFR（理想气体、等温、恒压 P、无压降）沿程剖面：
      volume_grid_m3: shape (n_points,)
      molar_flow_profile_mol_s: shape (n_species, n_points)

    其中动力学使用的浓度按：
      y_i = F_i / ΣF
      C_tot = P/(R T)
      C_i = y_i · C_tot
    """
    n_points = int(n_points)
    if n_points < 2:
        n_points = 2

    if not np.isfinite(reactor_volume_m3):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "V_m3 无效（NaN/Inf）",
        )
    if reactor_volume_m3 < 0.0:
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        return (
            np.array([0.0], dtype=float),
            molar_flow_inlet_mol_s.astype(float)[:, None],
            True,
            "V=0",
        )

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "温度 T_K 无效"
    if (not np.isfinite(pressure_Pa)) or (pressure_Pa <= 0.0):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "压力 P_Pa 无效",
        )

    if not np.all(np.isfinite(molar_flow_inlet_mol_s)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "入口摩尔流量包含 NaN/Inf",
        )
    if not np.all(np.isfinite(stoich_matrix)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "化学计量数矩阵 ν 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(k0)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "k0 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(ea_J_mol)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "Ea 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(reaction_order_matrix)):
        return (
            np.array([0.0]),
            molar_flow_inlet_mol_s[:, None],
            False,
            "反应级数矩阵 n 包含 NaN/Inf",
        )

    conc_total_mol_m3 = float(pressure_Pa) / max(
        float(R_GAS_J_MOL_K) * float(temperature_K), EPSILON_DENOMINATOR
    )

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            (not np.isfinite(max_wall_time_s)) or (max_wall_time_s <= 0.0)
        ):
            max_wall_time_s = None

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        flow_safe = safe_nonnegative(molar_flow_mol_s)
        total_flow = float(np.sum(flow_safe))
        y = flow_safe / max(total_flow, EPSILON_FLOW_RATE)
        conc_mol_m3 = y * float(conc_total_mol_m3)

        rate_vector = _dispatch_rate_vector(
            conc_mol_m3,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

        dF_dV = stoich_matrix @ rate_vector
        return dF_dV

    volume_grid_m3 = np.linspace(0.0, float(reactor_volume_m3), n_points, dtype=float)

    max_step_value = _compute_max_step(float(reactor_volume_m3), max_step_fraction)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reactor_volume_m3)),
            y0=molar_flow_inlet_mol_s.astype(float),
            method=solver_method,
            t_eval=volume_grid_m3,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )
    except Exception as exc:
        return (
            volume_grid_m3,
            molar_flow_inlet_mol_s.astype(float)[:, None],
            False,
            f"solve_ivp异常: {exc}",
        )

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return (
            volume_grid_m3,
            molar_flow_inlet_mol_s.astype(float)[:, None],
            False,
            str(message),
        )

    if (not np.all(np.isfinite(solution.t))) or (not np.all(np.isfinite(solution.y))):
        return (
            volume_grid_m3,
            molar_flow_inlet_mol_s.astype(float)[:, None],
            False,
            "solve_ivp 输出包含 NaN/Inf",
        )

    return solution.t.astype(float), solution.y.astype(float), True, "OK"


def integrate_batch_profile(
    reaction_time_s: float,
    temperature_K: float,
    conc_initial_mol_m3: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    n_points: int = DEFAULT_PROFILE_N_POINTS,
    kinetic_model: str = KINETIC_MODEL_POWER_LAW,
    max_step_fraction: float | None = DEFAULT_MAX_STEP_FRACTION,
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    order_rev_matrix: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    max_wall_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    返回 BSTR 随时间剖面：
      time_grid_s: shape (n_points,)
      conc_profile_mol_m3: shape (n_species, n_points)
    """
    n_points = int(n_points)
    if n_points < 2:
        n_points = 2

    if not np.isfinite(reaction_time_s):
        return (
            np.array([0.0]),
            conc_initial_mol_m3[:, None],
            False,
            "t_s 无效（NaN/Inf）",
        )
    if reaction_time_s < 0.0:
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "t_s 不能为负"
    if reaction_time_s == 0.0:
        return (
            np.array([0.0], dtype=float),
            conc_initial_mol_m3.astype(float)[:, None],
            True,
            "t=0",
        )

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "温度 T_K 无效"

    if not np.all(np.isfinite(conc_initial_mol_m3)):
        return (
            np.array([0.0]),
            conc_initial_mol_m3[:, None],
            False,
            "初始浓度包含 NaN/Inf",
        )
    if not np.all(np.isfinite(stoich_matrix)):
        return (
            np.array([0.0]),
            conc_initial_mol_m3[:, None],
            False,
            "化学计量数矩阵 ν 包含 NaN/Inf",
        )
    if not np.all(np.isfinite(k0)):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return (
            np.array([0.0]),
            conc_initial_mol_m3[:, None],
            False,
            "反应级数矩阵 n 包含 NaN/Inf",
        )

    start_time_s = time.monotonic()
    if max_wall_time_s is not None:
        try:
            max_wall_time_s = float(max_wall_time_s)
        except (ValueError, TypeError):
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            not np.isfinite(max_wall_time_s) or max_wall_time_s <= 0.0
        ):
            max_wall_time_s = None

    # 等温预计算 Arrhenius 常数
    _pre = _precompute_arrhenius(
        temperature_K,
        kinetic_model,
        k0,
        ea_J_mol,
        K0_ads,
        Ea_K_J_mol,
        k0_rev,
        ea_rev_J_mol,
    )

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_safe = safe_nonnegative(conc_mol_m3)

        rate_vector = _dispatch_rate_vector(
            conc_safe,
            temperature_K,
            kinetic_model,
            k0,
            ea_J_mol,
            reaction_order_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            k0_rev,
            ea_rev_J_mol,
            order_rev_matrix,
            precomputed=_pre,
        )

        dC_dt = stoich_matrix @ rate_vector
        return dC_dt

    time_grid_s = np.linspace(0.0, float(reaction_time_s), n_points, dtype=float)
    max_step_value = _compute_max_step(float(reaction_time_s), max_step_fraction)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reaction_time_s)),
            y0=conc_initial_mol_m3.astype(float),
            method=solver_method,
            t_eval=time_grid_s,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )
    except (
        Exception
    ) as exc:  # scipy 可能抛出多种异常（ValueError, RuntimeError等），宽泛捕获确保稳定性
        return (
            time_grid_s,
            conc_initial_mol_m3.astype(float)[:, None],
            False,
            f"solve_ivp异常: {exc}",
        )

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return (
            time_grid_s,
            conc_initial_mol_m3.astype(float)[:, None],
            False,
            str(message),
        )

    if (not np.all(np.isfinite(solution.t))) or (not np.all(np.isfinite(solution.y))):
        return (
            time_grid_s,
            conc_initial_mol_m3.astype(float)[:, None],
            False,
            "solve_ivp 输出包含 NaN/Inf",
        )

    return solution.t.astype(float), solution.y.astype(float), True, "OK"
