from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .kinetics import (
    calc_rate_vector_langmuir_hinshelwood,
    calc_rate_vector_power_law,
    calc_rate_vector_reversible,
    safe_nonnegative,
)


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
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, bool, str]:
    """
    PFR design equation (liquid / constant volumetric flow):
      dF_i/dV = Σ_j nu_{i,j} r_j
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

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        conc_mol_m3 = safe_nonnegative(molar_flow_mol_s) / max(vdot_m3_s, 1e-30)

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

    # Limit max_step to prevent infinite step refinement for stiff/extreme parameters
    max_step_value = (
        float(reactor_volume_m3) / 10.0 if reactor_volume_m3 > 0 else np.inf
    )
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
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, bool, str]:
    """
    Batch reactor ODE:
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

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
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

    # Limit max_step to prevent infinite step refinement for stiff/extreme parameters
    max_step_value = float(reaction_time_s) / 10.0 if reaction_time_s > 0 else np.inf
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
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
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

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        conc_mol_m3 = safe_nonnegative(molar_flow_mol_s) / max(vdot_m3_s, 1e-30)

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

    # Limit max_step to prevent infinite step refinement
    max_step_value = (
        float(reactor_volume_m3) / 10.0 if reactor_volume_m3 > 0 else np.inf
    )
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
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    返回 Batch 随时间剖面：
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

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
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
    # Limit max_step to prevent infinite step refinement
    max_step_value = float(reaction_time_s) / 10.0 if reaction_time_s > 0 else np.inf
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
