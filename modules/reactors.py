# 文件作用：反应器模型求解（PFR、CSTR、BSTR），通过 `solve_ivp`/`least_squares` 计算出口组成或时间剖面。

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .constants import EPSILON_FLOW_RATE, EPSILON_DENOMINATOR
from .kinetics import (
    calc_rate_vector_langmuir_hinshelwood,
    calc_rate_vector_power_law,
    calc_rate_vector_reversible,
    safe_nonnegative,
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
    except Exception:
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
    kinetic_model: str = "power_law",
    max_step_fraction: float | None = 0.1,
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
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
        except Exception:
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            not np.isfinite(max_wall_time_s) or max_wall_time_s <= 0.0
        ):
            max_wall_time_s = None

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

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
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
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
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
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
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
    kinetic_model: str = "power_law",
    max_step_fraction: float | None = 0.1,
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
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
        except Exception:
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            (not np.isfinite(max_wall_time_s)) or (max_wall_time_s <= 0.0)
        ):
            max_wall_time_s = None

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_safe = safe_nonnegative(conc_mol_m3)

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_safe.size),
                Ea_K_J_mol=(
                    Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_safe.size)
                ),
                m_inhibition=(
                    m_inhibition if m_inhibition is not None else np.ones(k0.size)
                ),
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
                conc_mol_m3=conc_safe,
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
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
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
    except Exception as exc:
        return conc_initial_mol_m3.copy(), False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return conc_initial_mol_m3.copy(), False, str(message)

    conc_final = solution.y[:, -1]
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
    kinetic_model: str = "power_law",
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
    max_nfev: int = 200,
    xtol: float = 1e-10,
    ftol: float = 1e-10,
    gtol: float = 1e-10,
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
    except Exception:
        max_nfev = 200
    if max_nfev <= 0:
        max_nfev = 200

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
        except Exception:
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            (not np.isfinite(max_wall_time_s)) or (max_wall_time_s <= 0.0)
        ):
            max_wall_time_s = None

    conc_inlet_mol_m3 = conc_inlet_mol_m3.astype(float)

    def _rate_vector(conc_mol_m3: np.ndarray) -> np.ndarray:
        conc_safe = safe_nonnegative(conc_mol_m3)
        if kinetic_model == "power_law":
            return calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        if kinetic_model == "langmuir_hinshelwood":
            return calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_safe.size),
                Ea_K_J_mol=(
                    Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_safe.size)
                ),
                m_inhibition=(
                    m_inhibition if m_inhibition is not None else np.ones(k0.size)
                ),
            )
        if kinetic_model == "reversible":
            return calc_rate_vector_reversible(
                conc_mol_m3=conc_safe,
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
            )
        return calc_rate_vector_power_law(
            conc_mol_m3=conc_safe,
            temperature_K=temperature_K,
            k0=k0,
            ea_J_mol=ea_J_mol,
            reaction_order_matrix=reaction_order_matrix,
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
    except Exception as exc:
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
    n_points: int = 200,
    kinetic_model: str = "power_law",
    max_step_fraction: float | None = 0.1,
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
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
        except Exception:
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            not np.isfinite(max_wall_time_s) or max_wall_time_s <= 0.0
        ):
            max_wall_time_s = None

    conc_inlet_mol_m3 = conc_inlet_mol_m3.astype(float)

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_safe = safe_nonnegative(conc_mol_m3)

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_safe.size),
                Ea_K_J_mol=(
                    Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_safe.size)
                ),
                m_inhibition=(
                    m_inhibition if m_inhibition is not None else np.ones(k0.size)
                ),
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
                conc_mol_m3=conc_safe,
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
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
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
    except Exception as exc:
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
    n_points: int = 200,
    kinetic_model: str = "power_law",
    max_step_fraction: float | None = 0.1,
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
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
        except Exception:
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            (not np.isfinite(max_wall_time_s)) or (max_wall_time_s <= 0.0)
        ):
            max_wall_time_s = None

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

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
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
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
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
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
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
    n_points: int = 200,
    kinetic_model: str = "power_law",
    max_step_fraction: float | None = 0.1,
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
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
        except Exception:
            max_wall_time_s = None
        if (max_wall_time_s is not None) and (
            not np.isfinite(max_wall_time_s) or max_wall_time_s <= 0.0
        ):
            max_wall_time_s = None

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户终止（stop_event）")
        if (max_wall_time_s is not None) and (
            (time.monotonic() - start_time_s) > max_wall_time_s
        ):
            raise RuntimeError(f"ODE 求解超时（>{max_wall_time_s:.1f} s）")

        conc_safe = safe_nonnegative(conc_mol_m3)

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_safe.size),
                Ea_K_J_mol=(
                    Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_safe.size)
                ),
                m_inhibition=(
                    m_inhibition if m_inhibition is not None else np.ones(k0.size)
                ),
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
                conc_mol_m3=conc_safe,
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
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
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
    except Exception as exc:
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

    return solution.t.astype(float), solution.y.astype(float), True, "OK"
