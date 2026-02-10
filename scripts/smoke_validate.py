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
from modules.constants import (
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
    )
    if not ok:
        raise AssertionError(f"BSTR predict failed: {msg}")
    _assert_finite_array(y, "BSTR y")


def smoke_validate_default_config() -> None:
    cfg = config_manager.get_default_config()
    ok, msg = config_manager.validate_config(cfg)
    if not ok:
        raise AssertionError(f"default config invalid: {msg}")


def main() -> None:
    smoke_validate_default_config()
    smoke_predict_pfr()
    smoke_predict_cstr()
    smoke_predict_bstr()

    # 给一个更直观的成功标记（兼容某些终端不显示异常堆栈的场景）
    if not math.isfinite(1.0):
        raise RuntimeError("unreachable")
    print("SMOKE_VALIDATE: OK")


if __name__ == "__main__":
    main()
