"""
轻量自检脚本（不依赖 pytest/ruff）：

目标：
1) 快速跑通 PFR / CSTR / BSTR 的一次预测流程（调用核心求解/动力学）
2) 验证基础配置校验逻辑正常

使用方法（项目根目录）：
  python scripts/smoke_validate.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules import fitting
from modules import config_manager
from modules import reactors
from modules.kinetics import calc_rate_vector_reversible
from modules.constants import (
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODEL_POWER_LAW,
    OUTPUT_MODE_COUT,
    OUTPUT_MODE_FOUT,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
)


def _assert_finite_array(x: np.ndarray, name: str) -> None:
    if x is None:
        raise AssertionError(f"{name} is None")
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise AssertionError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"{name} contains NaN/Inf: {arr}")


def smoke_predict_pfr() -> None:
    species_names = ["A", "B"]
    # A -> B
    stoich_matrix = np.array([[-1.0], [1.0]], dtype=float)  # (n_species, n_reactions)

    k0 = np.array([1.0e6], dtype=float)  # [SI]
    ea_J_mol = np.array([5.0e4], dtype=float)  # [J/mol]
    reaction_order_matrix = np.array([[1.0, 0.0]], dtype=float)  # (n_reactions, n_species)

    row = {
        "V_m3": 1.0e-3,
        "T_K": 350.0,
        "vdot_m3_s": 1.0e-4,
        "F0_A_mol_s": 0.01,
        "F0_B_mol_s": 0.0,
    }

    y, ok, msg = fitting._predict_outputs_for_row(
        row=row,
        species_names=species_names,
        output_mode=OUTPUT_MODE_FOUT,
        output_species_list=["A"],
        stoich_matrix=stoich_matrix,
        k0=k0,
        ea_J_mol=ea_J_mol,
        reaction_order_matrix=reaction_order_matrix,
        solver_method="RK45",
        rtol=1e-8,
        atol=1e-12,
        reactor_type=REACTOR_TYPE_PFR,
        kinetic_model=KINETIC_MODEL_POWER_LAW,
        reversible_enabled=False,
    )
    if not ok:
        raise AssertionError(f"PFR predict failed: {msg}")
    _assert_finite_array(y, "PFR y")


def smoke_predict_cstr() -> None:
    species_names = ["A", "B"]
    stoich_matrix = np.array([[-1.0], [1.0]], dtype=float)

    k0 = np.array([1.0e6], dtype=float)
    ea_J_mol = np.array([5.0e4], dtype=float)
    reaction_order_matrix = np.array([[1.0, 0.0]], dtype=float)

    row = {
        "V_m3": 2.0e-3,
        "T_K": 350.0,
        "vdot_m3_s": 1.0e-4,
        "C0_A_mol_m3": 2000.0,
        "C0_B_mol_m3": 0.0,
    }

    y, ok, msg = fitting._predict_outputs_for_row(
        row=row,
        species_names=species_names,
        output_mode=OUTPUT_MODE_COUT,
        output_species_list=["A", "B"],
        stoich_matrix=stoich_matrix,
        k0=k0,
        ea_J_mol=ea_J_mol,
        reaction_order_matrix=reaction_order_matrix,
        solver_method="RK45",
        rtol=1e-8,
        atol=1e-12,
        reactor_type=REACTOR_TYPE_CSTR,
        kinetic_model=KINETIC_MODEL_POWER_LAW,
        reversible_enabled=False,
    )
    if not ok:
        raise AssertionError(f"CSTR predict failed: {msg}")
    _assert_finite_array(y, "CSTR y")


def smoke_predict_bstr() -> None:
    species_names = ["A", "B"]
    stoich_matrix = np.array([[-1.0], [1.0]], dtype=float)

    k0 = np.array([1.0e6], dtype=float)
    ea_J_mol = np.array([5.0e4], dtype=float)
    reaction_order_matrix = np.array([[1.0, 0.0]], dtype=float)

    row = {
        "t_s": 120.0,
        "T_K": 350.0,
        "C0_A_mol_m3": 2000.0,
        "C0_B_mol_m3": 0.0,
    }

    y, ok, msg = fitting._predict_outputs_for_row(
        row=row,
        species_names=species_names,
        output_mode=OUTPUT_MODE_COUT,
        output_species_list=["A", "B"],
        stoich_matrix=stoich_matrix,
        k0=k0,
        ea_J_mol=ea_J_mol,
        reaction_order_matrix=reaction_order_matrix,
        solver_method="RK45",
        rtol=1e-8,
        atol=1e-12,
        reactor_type=REACTOR_TYPE_BSTR,
        kinetic_model=KINETIC_MODEL_POWER_LAW,
        reversible_enabled=False,
    )
    if not ok:
        raise AssertionError(f"BSTR predict failed: {msg}")
    _assert_finite_array(y, "BSTR y")


def smoke_validate_default_config() -> None:
    cfg = config_manager.get_default_config()
    ok, msg = config_manager.validate_config(cfg)
    if not ok:
        raise AssertionError(f"default config invalid: {msg}")


def smoke_predict_pfr_power_reversible() -> None:
    species_names = ["A", "B"]
    stoich_matrix = np.array([[-1.0], [1.0]], dtype=float)

    row = {
        "V_m3": 1.0e-3,
        "T_K": 350.0,
        "vdot_m3_s": 1.0e-4,
        "F0_A_mol_s": 0.01,
        "F0_B_mol_s": 0.0,
    }
    y, ok, msg = fitting._predict_outputs_for_row(
        row=row,
        species_names=species_names,
        output_mode=OUTPUT_MODE_FOUT,
        output_species_list=["A", "B"],
        stoich_matrix=stoich_matrix,
        k0=np.array([1.0e6], dtype=float),
        ea_J_mol=np.array([5.0e4], dtype=float),
        reaction_order_matrix=np.array([[1.0, 0.0]], dtype=float),
        solver_method="RK45",
        rtol=1e-8,
        atol=1e-12,
        reactor_type=REACTOR_TYPE_PFR,
        kinetic_model=KINETIC_MODEL_POWER_LAW,
        reversible_enabled=True,
        k0_rev=np.array([2.0e5], dtype=float),
        ea_rev_J_mol=np.array([4.5e4], dtype=float),
        order_rev_matrix=np.array([[0.0, 1.0]], dtype=float),
    )
    if not ok:
        raise AssertionError(f"PFR power-law+reversible predict failed: {msg}")
    _assert_finite_array(y, "PFR power-law+reversible y")


def smoke_predict_pfr_lh() -> None:
    species_names = ["A", "B"]
    stoich_matrix = np.array([[-1.0], [1.0]], dtype=float)

    row = {
        "V_m3": 1.0e-3,
        "T_K": 350.0,
        "vdot_m3_s": 1.0e-4,
        "F0_A_mol_s": 0.01,
        "F0_B_mol_s": 0.0,
    }
    y, ok, msg = fitting._predict_outputs_for_row(
        row=row,
        species_names=species_names,
        output_mode=OUTPUT_MODE_FOUT,
        output_species_list=["A", "B"],
        stoich_matrix=stoich_matrix,
        k0=np.array([1.0e6], dtype=float),
        ea_J_mol=np.array([5.0e4], dtype=float),
        reaction_order_matrix=np.array([[1.0, 0.0]], dtype=float),
        solver_method="RK45",
        rtol=1e-8,
        atol=1e-12,
        reactor_type=REACTOR_TYPE_PFR,
        kinetic_model=KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
        reversible_enabled=False,
        K0_ads=np.array([1.0e-3, 0.0], dtype=float),
        Ea_K_J_mol=np.array([-1.0e4, 0.0], dtype=float),
        m_inhibition=np.array([1.0], dtype=float),
    )
    if not ok:
        raise AssertionError(f"PFR L-H predict failed: {msg}")
    _assert_finite_array(y, "PFR L-H y")


def smoke_predict_pfr_lh_reversible() -> None:
    species_names = ["A", "B"]
    stoich_matrix = np.array([[-1.0], [1.0]], dtype=float)

    row = {
        "V_m3": 1.0e-3,
        "T_K": 350.0,
        "vdot_m3_s": 1.0e-4,
        "F0_A_mol_s": 0.01,
        "F0_B_mol_s": 0.0,
    }
    y, ok, msg = fitting._predict_outputs_for_row(
        row=row,
        species_names=species_names,
        output_mode=OUTPUT_MODE_FOUT,
        output_species_list=["A", "B"],
        stoich_matrix=stoich_matrix,
        k0=np.array([1.0e6], dtype=float),
        ea_J_mol=np.array([5.0e4], dtype=float),
        reaction_order_matrix=np.array([[1.0, 0.0]], dtype=float),
        solver_method="RK45",
        rtol=1e-8,
        atol=1e-12,
        reactor_type=REACTOR_TYPE_PFR,
        kinetic_model=KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
        reversible_enabled=True,
        K0_ads=np.array([1.0e-3, 0.0], dtype=float),
        Ea_K_J_mol=np.array([-1.0e4, 0.0], dtype=float),
        m_inhibition=np.array([1.0], dtype=float),
        k0_rev=np.array([2.0e5], dtype=float),
        ea_rev_J_mol=np.array([4.5e4], dtype=float),
        order_rev_matrix=np.array([[0.0, 1.0]], dtype=float),
    )
    if not ok:
        raise AssertionError(f"PFR L-H+reversible predict failed: {msg}")
    _assert_finite_array(y, "PFR L-H+reversible y")


def smoke_power_reversible_formula_compatibility() -> None:
    conc = np.array([120.0, 35.0], dtype=float)
    temperature_K = 360.0
    k0_fwd = np.array([2.0e6], dtype=float)
    ea_fwd = np.array([5.8e4], dtype=float)
    order_fwd = np.array([[1.0, 0.0]], dtype=float)
    k0_rev = np.array([6.0e5], dtype=float)
    ea_rev = np.array([5.2e4], dtype=float)
    order_rev = np.array([[0.0, 1.0]], dtype=float)

    expected = calc_rate_vector_reversible(
        conc_mol_m3=conc,
        temperature_K=temperature_K,
        k0_fwd=k0_fwd,
        ea_fwd_J_mol=ea_fwd,
        order_fwd_matrix=order_fwd,
        k0_rev=k0_rev,
        ea_rev_J_mol=ea_rev,
        order_rev_matrix=order_rev,
    )
    actual = reactors._dispatch_rate_vector(  # noqa: SLF001 - smoke 脚本允许校验内部等价性
        conc_mol_m3=conc,
        temperature_K=temperature_K,
        kinetic_model=KINETIC_MODEL_POWER_LAW,
        reversible_enabled=True,
        k0=k0_fwd,
        ea_J_mol=ea_fwd,
        reaction_order_matrix=order_fwd,
        K0_ads=None,
        Ea_K_J_mol=None,
        m_inhibition=None,
        k0_rev=k0_rev,
        ea_rev_J_mol=ea_rev,
        order_rev_matrix=order_rev,
        precomputed=None,
    )
    if not np.allclose(expected, actual, rtol=1e-12, atol=1e-15):
        raise AssertionError(
            f"power-law+reversible 与旧可逆公式不一致: expected={expected}, actual={actual}"
        )


def main() -> None:
    smoke_validate_default_config()
    smoke_predict_pfr()
    smoke_predict_pfr_power_reversible()
    smoke_predict_pfr_lh()
    smoke_predict_pfr_lh_reversible()
    smoke_power_reversible_formula_compatibility()
    smoke_predict_cstr()
    smoke_predict_bstr()

    # 给一个更直观的成功标记（兼容某些终端不显示异常堆栈的场景）
    if not math.isfinite(1.0):
        raise RuntimeError("unreachable")
    print("SMOKE_VALIDATE: OK")


if __name__ == "__main__":
    main()
