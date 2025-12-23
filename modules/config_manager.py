"""
配置管理模块：导入/导出用户设置，自动恢复上次配置
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd

# 持久化目录（跨刷新可用；避免写入代码目录导致多用户互相覆盖）
_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "Kinetics_app_persist")
os.makedirs(_PERSIST_DIR, exist_ok=True)

_AUTO_SAVE_FILE = os.path.join(_PERSIST_DIR, "last_config.json")


def _atomic_write_text(file_path: str, text: str, encoding: str = "utf-8") -> None:
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="tmp_", suffix=".txt", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(temp_path, file_path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _convert_to_serializable(obj: Any) -> Any:
    """
    将 NumPy/Pandas 对象转换为可 JSON 序列化的 Python 原生类型。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "data": obj.to_dict(orient="split"),
        }
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    return obj


def _convert_from_serializable(obj: Any) -> Any:
    """
    将 JSON 反序列化后的对象还原为 NumPy/Pandas 对象。
    """
    if isinstance(obj, dict):
        if obj.get("__type__") == "DataFrame":
            return pd.DataFrame(**obj["data"])
        return {k: _convert_from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_from_serializable(v) for v in obj]
    return obj


def collect_config(
    # --- 基础设置 ---
    reactor_type: str,
    kinetic_model: str,
    solver_method: str,
    rtol: float,
    atol: float,
    max_step_fraction: float,
    # --- 物种与反应 ---
    species_text: str,
    n_reactions: int,
    # --- 化学计量数和反应级数 ---
    stoich_matrix: np.ndarray,
    order_guess: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    # --- 动力学参数 ---
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    # --- L-H 参数 ---
    K0_ads: np.ndarray | None = None,
    Ea_K_J_mol: np.ndarray | None = None,
    fit_K0_ads_flags: np.ndarray | None = None,
    fit_Ea_K_flags: np.ndarray | None = None,
    m_inhibition: np.ndarray | None = None,
    fit_m_flags: np.ndarray | None = None,
    # --- 可逆反应参数 ---
    k0_rev: np.ndarray | None = None,
    ea_rev_J_mol: np.ndarray | None = None,
    fit_k0_rev_flags: np.ndarray | None = None,
    fit_ea_rev_flags: np.ndarray | None = None,
    order_rev: np.ndarray | None = None,
    fit_order_rev_flags_matrix: np.ndarray | None = None,
    # --- 拟合目标 ---
    output_mode: str = "Fout (mol/s)",
    output_species_list: list[str] | None = None,
    # --- 参数边界 ---
    k0_min: float = 1e-15,
    k0_max: float = 1e15,
    ea_min_J_mol: float = 1e4,
    ea_max_J_mol: float = 3e5,
    order_min: float = -2.0,
    order_max: float = 5.0,
    # L-H 边界
    K0_ads_min: float = 0.0,
    K0_ads_max: float = 1e10,
    Ea_K_min: float = -2e5,
    Ea_K_max: float = 2e5,
    m_min: float = 0.0,
    m_max: float = 5.0,
    # 可逆反应边界
    k0_rev_min: float = 1e-15,
    k0_rev_max: float = 1e15,
    ea_rev_min_J_mol: float = 1e4,
    ea_rev_max_J_mol: float = 3e5,
    order_rev_min: float = -2.0,
    order_rev_max: float = 5.0,
    # --- 残差类型与高级设置 ---
    residual_type: str = "绝对残差",
    diff_step_rel: float = 1e-2,
    max_nfev: int = 3000,
    use_x_scale_jac: bool = True,
    use_multi_start: bool = True,
    n_starts: int = 10,
    max_nfev_coarse: int = 300,
    random_seed: int = 42,
    # --- 显示格式 ---
    table_number_style: str = "科学计数",
    table_decimal_places: int = 3,
    plot_tick_auto: bool = True,
    plot_number_style: str = "科学计数",
    plot_decimal_places: int = 3,
) -> dict:
    """
    收集所有用户配置到一个字典中，便于导出。
    """
    config = {
        "version": "1.0",
        # 基础设置
        "reactor_type": reactor_type,
        "kinetic_model": kinetic_model,
        "solver_method": solver_method,
        "rtol": rtol,
        "atol": atol,
        "max_step_fraction": max_step_fraction,
        # 物种与反应
        "species_text": species_text,
        "n_reactions": n_reactions,
        # 化学计量数和反应级数
        "stoich_matrix": stoich_matrix,
        "order_guess": order_guess,
        "fit_order_flags_matrix": fit_order_flags_matrix,
        # 动力学参数
        "k0_guess": k0_guess,
        "ea_guess_J_mol": ea_guess_J_mol,
        "fit_k0_flags": fit_k0_flags,
        "fit_ea_flags": fit_ea_flags,
        # L-H 参数
        "K0_ads": K0_ads,
        "Ea_K_J_mol": Ea_K_J_mol,
        "fit_K0_ads_flags": fit_K0_ads_flags,
        "fit_Ea_K_flags": fit_Ea_K_flags,
        "m_inhibition": m_inhibition,
        "fit_m_flags": fit_m_flags,
        # 可逆反应参数
        "k0_rev": k0_rev,
        "ea_rev_J_mol": ea_rev_J_mol,
        "fit_k0_rev_flags": fit_k0_rev_flags,
        "fit_ea_rev_flags": fit_ea_rev_flags,
        "order_rev": order_rev,
        "fit_order_rev_flags_matrix": fit_order_rev_flags_matrix,
        # 拟合目标
        "output_mode": output_mode,
        "output_species_list": output_species_list or [],
        # 参数边界
        "k0_min": k0_min,
        "k0_max": k0_max,
        "ea_min_J_mol": ea_min_J_mol,
        "ea_max_J_mol": ea_max_J_mol,
        "order_min": order_min,
        "order_max": order_max,
        # L-H 边界
        "K0_ads_min": K0_ads_min,
        "K0_ads_max": K0_ads_max,
        "Ea_K_min": Ea_K_min,
        "Ea_K_max": Ea_K_max,
        "m_min": m_min,
        "m_max": m_max,
        # 可逆反应边界
        "k0_rev_min": k0_rev_min,
        "k0_rev_max": k0_rev_max,
        "ea_rev_min_J_mol": ea_rev_min_J_mol,
        "ea_rev_max_J_mol": ea_rev_max_J_mol,
        "order_rev_min": order_rev_min,
        "order_rev_max": order_rev_max,
        # 残差类型与高级设置
        "residual_type": residual_type,
        "diff_step_rel": diff_step_rel,
        "max_nfev": max_nfev,
        "use_x_scale_jac": use_x_scale_jac,
        "use_multi_start": use_multi_start,
        "n_starts": n_starts,
        "max_nfev_coarse": max_nfev_coarse,
        "random_seed": random_seed,
        # 显示格式
        "table_number_style": table_number_style,
        "table_decimal_places": table_decimal_places,
        "plot_tick_auto": plot_tick_auto,
        "plot_number_style": plot_number_style,
        "plot_decimal_places": plot_decimal_places,
    }
    return _convert_to_serializable(config)


def export_config_to_json(config: dict) -> str:
    """
    将配置字典导出为 JSON 字符串。
    """
    return json.dumps(config, indent=2, ensure_ascii=False)


def import_config_from_json(json_str: str) -> dict:
    """
    从 JSON 字符串导入配置字典。
    """
    config = json.loads(json_str)
    return _convert_from_serializable(config)


def auto_save_config(config: dict, client_id: str | None = None) -> tuple[bool, str]:
    """
    自动保存配置到本地文件。

    Returns:
        (ok, message)
    """
    _ = client_id  # 保留参数以兼容旧调用；当前实现只保存“一个”配置缓存
    file_path = _AUTO_SAVE_FILE
    try:
        text = json.dumps(config, indent=2, ensure_ascii=False)
        _atomic_write_text(file_path, text, encoding="utf-8")
        return True, "OK"
    except Exception as exc:
        return False, f"自动保存失败: {exc}"


def auto_load_config(client_id: str | None = None) -> tuple[dict | None, str]:
    """
    从本地文件加载上次保存的配置。

    Returns:
        (config, message)
    """
    _ = client_id  # 保留参数以兼容旧调用；当前实现只加载“一个”配置缓存
    file_path = _AUTO_SAVE_FILE
    if not os.path.exists(file_path):
        return None, "未找到自动保存配置"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return _convert_from_serializable(config), "OK"
    except Exception as exc:
        return None, f"自动加载失败: {exc}"


def clear_auto_saved_config(client_id: str | None = None) -> tuple[bool, str]:
    """
    删除自动保存的配置文件（重置为默认）。

    Returns:
        (ok, message)
    """
    _ = client_id  # 保留参数以兼容旧调用；当前实现只清理“一个”配置缓存
    file_path = _AUTO_SAVE_FILE
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True, "OK"
        except Exception as exc:
            return False, f"删除失败: {exc}"
    return True, "文件不存在"


def get_default_config() -> dict:
    """
    返回默认配置值。
    """
    return {
        "version": "1.0",
        # 基础设置
        "reactor_type": "PFR",
        "kinetic_model": "power_law",
        "solver_method": "RK45",
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_step_fraction": 0.1,
        # 物种与反应
        "species_text": "A,B,C",
        "n_reactions": 1,
        # 显示格式
        "table_number_style": "科学计数",
        "table_decimal_places": 3,
        "plot_tick_auto": True,
        "plot_number_style": "科学计数",
        "plot_decimal_places": 3,
        # 拟合目标
        "output_mode": "Fout (mol/s)",
        "output_species_list": ["A"],
        # 参数边界
        "k0_min": 1e-15,
        "k0_max": 1e15,
        "ea_min_J_mol": 1e4,
        "ea_max_J_mol": 3e5,
        "order_min": -2.0,
        "order_max": 5.0,
        # 残差类型与高级设置
        "residual_type": "绝对残差",
        "diff_step_rel": 1e-2,
        "max_nfev": 3000,
        "use_x_scale_jac": True,
        "use_multi_start": True,
        "n_starts": 10,
        "max_nfev_coarse": 300,
        "random_seed": 42,
    }


def validate_config(config: dict) -> tuple[bool, str]:
    """
    验证配置是否有效。

    Returns:
        (is_valid, error_message)
    """

    def _parse_species_text(species_text: str) -> list[str]:
        parts = [p.strip() for p in str(species_text).split(",")]
        names = [p for p in parts if p]
        unique_names: list[str] = []
        for name in names:
            if name not in unique_names:
                unique_names.append(name)
        return unique_names

    def _check_array_shape(
        key: str, expected_shape: tuple[int, ...], dtype: type
    ) -> tuple[bool, str]:
        if key not in config:
            return True, ""
        try:
            arr = np.asarray(config[key], dtype=dtype)
        except Exception:
            return False, f"配置项 {key} 无法转换为数组"
        if arr.shape != expected_shape:
            return (
                False,
                f"配置项 {key} 维度不匹配：期望 {expected_shape}，实际 {arr.shape}",
            )
        return True, ""

    required_keys = ["reactor_type", "kinetic_model", "species_text", "n_reactions"]
    for key in required_keys:
        if key not in config:
            return False, f"缺少必需的配置项：{key}"

    reactor_type = str(config.get("reactor_type", "")).strip()
    if reactor_type == "Batch":
        reactor_type = "BSTR"
        config["reactor_type"] = reactor_type

    if reactor_type not in ["PFR", "CSTR", "BSTR"]:
        return False, f"无效的反应器类型：{reactor_type}"

    if config["kinetic_model"] not in [
        "power_law",
        "langmuir_hinshelwood",
        "reversible",
    ]:
        return False, f"无效的动力学模型：{config['kinetic_model']}"

    if config.get("version", "1.0") != "1.0":
        return False, f"配置文件版本不兼容：{config.get('version')}"

    # 可选项：求解器/容限
    if "solver_method" in config:
        if config["solver_method"] not in ["RK45", "BDF", "Radau"]:
            return False, f"无效的 solver_method：{config['solver_method']}"
    if "rtol" in config:
        try:
            rtol = float(config["rtol"])
        except Exception:
            return False, "rtol 无法转换为数值"
        if (not np.isfinite(rtol)) or (rtol <= 0.0):
            return False, "rtol 必须为正且有限"
    if "atol" in config:
        try:
            atol = float(config["atol"])
        except Exception:
            return False, "atol 无法转换为数值"
        if (not np.isfinite(atol)) or (atol <= 0.0):
            return False, "atol 必须为正且有限"

    if "max_step_fraction" in config:
        try:
            value = float(config["max_step_fraction"])
        except Exception:
            return False, "max_step_fraction 无法转换为数值"
        if (not np.isfinite(value)) or (value < 0.0):
            return False, "max_step_fraction 必须为非负且有限（0 表示不限制）"

    # 说明：历史版本曾导出过 residual_penalty_* 字段；当前版本不再使用，导入时忽略即可。

    # 可选项：输出模式与目标物种
    if "output_mode" in config:
        output_mode = str(config["output_mode"])
        if output_mode == "X (conversion)":
            return False, "当前版本已移除 X (conversion) 作为拟合目标变量"

        if reactor_type == "BSTR":
            allowed_output_modes = ["Cout (mol/m^3)"]
        else:
            allowed_output_modes = ["Fout (mol/s)", "Cout (mol/m^3)"]

        if output_mode not in allowed_output_modes:
            return False, f"无效的 output_mode：{output_mode}"

    species_names = _parse_species_text(config["species_text"])
    if len(species_names) < 1:
        return False, "species_text 解析后为空，请检查物种名"

    try:
        n_reactions = int(config["n_reactions"])
    except Exception:
        return False, "n_reactions 无法转换为整数"
    if n_reactions < 1:
        return False, "n_reactions 必须 >= 1"

    if "output_species_list" in config:
        if not isinstance(config["output_species_list"], list):
            return False, "output_species_list 必须为列表"
        if len(config["output_species_list"]) < 1:
            return False, "output_species_list 不能为空（至少选择一个物种）"
        for s in config["output_species_list"]:
            if not isinstance(s, str):
                return False, "output_species_list 中存在非字符串物种名"
            if s not in species_names:
                return False, f"output_species_list 包含未知物种：{s}"

    # 结构一致性：矩阵/向量尺寸（仅在配置中存在时校验）
    n_species = len(species_names)
    ok, msg = _check_array_shape("stoich_matrix", (n_species, n_reactions), float)
    if not ok:
        return ok, msg
    ok, msg = _check_array_shape("order_guess", (n_reactions, n_species), float)
    if not ok:
        return ok, msg
    ok, msg = _check_array_shape(
        "fit_order_flags_matrix", (n_reactions, n_species), bool
    )
    if not ok:
        return ok, msg

    ok, msg = _check_array_shape("k0_guess", (n_reactions,), float)
    if not ok:
        return ok, msg
    ok, msg = _check_array_shape("ea_guess_J_mol", (n_reactions,), float)
    if not ok:
        return ok, msg
    ok, msg = _check_array_shape("fit_k0_flags", (n_reactions,), bool)
    if not ok:
        return ok, msg
    ok, msg = _check_array_shape("fit_ea_flags", (n_reactions,), bool)
    if not ok:
        return ok, msg

    if config["kinetic_model"] == "langmuir_hinshelwood":
        ok, msg = _check_array_shape("K0_ads", (n_species,), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("Ea_K_J_mol", (n_species,), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("fit_K0_ads_flags", (n_species,), bool)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("fit_Ea_K_flags", (n_species,), bool)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("m_inhibition", (n_reactions,), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("fit_m_flags", (n_reactions,), bool)
        if not ok:
            return ok, msg

    if config["kinetic_model"] == "reversible":
        ok, msg = _check_array_shape("k0_rev", (n_reactions,), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("ea_rev_J_mol", (n_reactions,), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("fit_k0_rev_flags", (n_reactions,), bool)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("fit_ea_rev_flags", (n_reactions,), bool)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("order_rev", (n_reactions, n_species), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape(
            "fit_order_rev_flags_matrix", (n_reactions, n_species), bool
        )
        if not ok:
            return ok, msg

    return True, ""
