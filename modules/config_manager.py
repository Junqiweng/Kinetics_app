# 文件作用：配置管理（收集当前 UI 配置、导入/导出 JSON、自动保存/恢复上次配置）。

"""
配置管理模块：导入/导出用户设置，自动恢复上次配置
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_ATOL,
    DEFAULT_DIFF_STEP_REL,
    DEFAULT_EA_K_MAX_J_MOL,
    DEFAULT_EA_K_MIN_J_MOL,
    DEFAULT_EA_MAX_J_MOL,
    DEFAULT_EA_MIN_J_MOL,
    DEFAULT_EA_REV_MAX_J_MOL,
    DEFAULT_EA_REV_MIN_J_MOL,
    DEFAULT_K0_ADS_MAX,
    DEFAULT_K0_ADS_MIN,
    DEFAULT_K0_MAX,
    DEFAULT_K0_MIN,
    DEFAULT_K0_REV_MAX,
    DEFAULT_K0_REV_MIN,
    DEFAULT_M_INHIBITION_MAX,
    DEFAULT_M_INHIBITION_MIN,
    DEFAULT_MAX_NFEV,
    DEFAULT_MAX_NFEV_COARSE,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_N_STARTS,
    DEFAULT_ORDER_MAX,
    DEFAULT_ORDER_MIN,
    DEFAULT_ORDER_REV_MAX,
    DEFAULT_ORDER_REV_MIN,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RTOL,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    PERSIST_DIR_NAME,
    KINETIC_MODELS,
    KINETIC_MODEL_POWER_LAW,
    OUTPUT_MODE_FOUT,
    OUTPUT_MODES_BATCH,
    OUTPUT_MODES_FLOW,
    REACTOR_TYPES,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_PFR,
)
from .file_utils import atomic_write_text

# 持久化基础目录（从 constants 统一管理目录名）
_PERSIST_BASE_DIR = os.path.join(tempfile.gettempdir(), PERSIST_DIR_NAME)


def _get_auto_save_file(session_id: str | None = None) -> str:
    """
    返回自动保存配置文件的路径。

    参数:
        session_id: 会话 ID，若提供则使用独立目录

    返回:
        配置文件路径
    """
    normalized_sid = None
    if session_id:
        try:
            normalized_sid = str(uuid.UUID(str(session_id).strip()))
        except Exception:
            normalized_sid = None

    if normalized_sid:
        persist_dir = os.path.join(_PERSIST_BASE_DIR, normalized_sid)
    else:
        persist_dir = _PERSIST_BASE_DIR
    os.makedirs(persist_dir, exist_ok=True)
    return os.path.join(persist_dir, "last_config.json")



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
    pfr_flow_model: str,
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
    # --- 朗缪尔-辛斯伍德（L-H）参数 ---
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
    output_mode: str = OUTPUT_MODE_FOUT,
    output_species_list: list[str] | None = None,
    # --- 参数边界 ---
    k0_min: float = DEFAULT_K0_MIN,
    k0_max: float = DEFAULT_K0_MAX,
    ea_min_J_mol: float = DEFAULT_EA_MIN_J_MOL,
    ea_max_J_mol: float = DEFAULT_EA_MAX_J_MOL,
    order_min: float = DEFAULT_ORDER_MIN,
    order_max: float = DEFAULT_ORDER_MAX,
    # 朗缪尔-辛斯伍德（L-H）边界
    K0_ads_min: float = DEFAULT_K0_ADS_MIN,
    K0_ads_max: float = DEFAULT_K0_ADS_MAX,
    Ea_K_min: float = DEFAULT_EA_K_MIN_J_MOL,
    Ea_K_max: float = DEFAULT_EA_K_MAX_J_MOL,
    m_min: float = DEFAULT_M_INHIBITION_MIN,
    m_max: float = DEFAULT_M_INHIBITION_MAX,
    # 可逆反应边界
    k0_rev_min: float = DEFAULT_K0_REV_MIN,
    k0_rev_max: float = DEFAULT_K0_REV_MAX,
    ea_rev_min_J_mol: float = DEFAULT_EA_REV_MIN_J_MOL,
    ea_rev_max_J_mol: float = DEFAULT_EA_REV_MAX_J_MOL,
    order_rev_min: float = DEFAULT_ORDER_REV_MIN,
    order_rev_max: float = DEFAULT_ORDER_REV_MAX,
    # --- 残差类型与高级设置 ---
    residual_type: str = "绝对残差",
    diff_step_rel: float = DEFAULT_DIFF_STEP_REL,
    max_nfev: int = DEFAULT_MAX_NFEV,
    use_x_scale_jac: bool = True,
    use_multi_start: bool = True,
    n_starts: int = DEFAULT_N_STARTS,
    max_nfev_coarse: int = DEFAULT_MAX_NFEV_COARSE,
    random_seed: int = DEFAULT_RANDOM_SEED,
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
        "pfr_flow_model": str(pfr_flow_model).strip() or PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
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
        # 朗缪尔-辛斯伍德（L-H）参数
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
        # 朗缪尔-辛斯伍德（L-H）边界
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


def auto_save_config(config: dict, session_id: str | None = None) -> tuple[bool, str]:
    """
    自动保存配置到本地文件。

    返回:
        (ok, message)
    """
    file_path = _get_auto_save_file(session_id)
    try:
        text = json.dumps(config, indent=2, ensure_ascii=False)
        atomic_write_text(file_path, text, encoding="utf-8")
        return True, "OK"
    except Exception as exc:
        return False, f"自动保存失败: {exc}"


def auto_load_config(session_id: str | None = None) -> tuple[dict | None, str]:
    """
    从本地文件加载上次保存的配置。

    返回:
        (config, message)
    """
    file_path = _get_auto_save_file(session_id)
    if not os.path.exists(file_path):
        return None, "未找到自动保存配置"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return _convert_from_serializable(config), "OK"
    except Exception as exc:
        return None, f"自动加载失败: {exc}"


def clear_auto_saved_config(session_id: str | None = None) -> tuple[bool, str]:
    """
    删除自动保存的配置文件（重置为默认）。

    返回:
        (ok, message)
    """
    file_path = _get_auto_save_file(session_id)
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
        "reactor_type": REACTOR_TYPE_PFR,
        "pfr_flow_model": PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
        "kinetic_model": KINETIC_MODEL_POWER_LAW,
        "solver_method": "RK45",
        "rtol": DEFAULT_RTOL,
        "atol": DEFAULT_ATOL,
        "max_step_fraction": DEFAULT_MAX_STEP_FRACTION,
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
        "output_mode": OUTPUT_MODE_FOUT,
        "output_species_list": ["A"],
        # 参数边界
        "k0_min": DEFAULT_K0_MIN,
        "k0_max": DEFAULT_K0_MAX,
        "ea_min_J_mol": DEFAULT_EA_MIN_J_MOL,
        "ea_max_J_mol": DEFAULT_EA_MAX_J_MOL,
        "order_min": DEFAULT_ORDER_MIN,
        "order_max": DEFAULT_ORDER_MAX,
        # 残差类型与高级设置
        "residual_type": "绝对残差",
        "diff_step_rel": DEFAULT_DIFF_STEP_REL,
        "max_nfev": DEFAULT_MAX_NFEV,
        "use_x_scale_jac": True,
        "use_multi_start": True,
        "n_starts": DEFAULT_N_STARTS,
        "max_nfev_coarse": DEFAULT_MAX_NFEV_COARSE,
        "random_seed": DEFAULT_RANDOM_SEED,
    }


def validate_config(config: dict) -> tuple[bool, str]:
    """
    验证配置是否有效。

    返回:
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
        except (ValueError, TypeError):
            return False, f"配置项 {key} 无法转换为数组"
        if arr.shape != expected_shape:
            return (
                False,
                f"配置项 {key} 维度不匹配：期望 {expected_shape}，实际 {arr.shape}",
            )
        return True, ""

    def _check_array_values(
        key: str, must_be_positive: bool = False, allow_nan: bool = False
    ) -> tuple[bool, str]:
        """检查数组数值的有效性"""
        if key not in config:
            return True, ""
        try:
            arr = np.asarray(config[key], dtype=float)
        except (ValueError, TypeError):
            return False, f"配置项 {key} 无法转换为数值数组"

        # 检查是否包含无效值（NaN, Inf）
        if not allow_nan and not np.all(np.isfinite(arr)):
            return False, f"配置项 {key} 包含无效值（NaN 或 Inf）"

        # 检查是否必须为正值
        if must_be_positive and np.any(arr <= 0):
            return False, f"配置项 {key} 必须全部为正值"

        return True, ""

    def _check_positive_int(key: str, min_value: int = 1) -> tuple[bool, str]:
        if key not in config:
            return True, ""
        try:
            value = int(config[key])
        except (ValueError, TypeError):
            return False, f"{key} 无法转换为整数"
        if value < int(min_value):
            return False, f"{key} 必须 >= {int(min_value)}"
        return True, ""

    def _check_nonnegative_int(key: str) -> tuple[bool, str]:
        if key not in config:
            return True, ""
        try:
            value = int(config[key])
        except (ValueError, TypeError):
            return False, f"{key} 无法转换为整数"
        if value < 0:
            return False, f"{key} 必须为非负整数"
        return True, ""

    def _check_positive_float(key: str) -> tuple[bool, str]:
        if key not in config:
            return True, ""
        try:
            value = float(config[key])
        except (ValueError, TypeError):
            return False, f"{key} 无法转换为数值"
        if (not np.isfinite(value)) or (value <= 0.0):
            return False, f"{key} 必须为正且有限"
        return True, ""

    required_keys = ["reactor_type", "kinetic_model", "species_text", "n_reactions"]
    for key in required_keys:
        if key not in config:
            return False, f"缺少必需的配置项：{key}"

    reactor_type = str(config.get("reactor_type", "")).strip()
    if reactor_type == "Batch":
        reactor_type = REACTOR_TYPE_BSTR
        config["reactor_type"] = reactor_type

    if reactor_type not in REACTOR_TYPES:
        return False, f"无效的反应器类型：{reactor_type}"

    # 可选项：PFR 流动模型（仅对 PFR 生效；但为了配置一致性，允许在任意 reactor_type 下保存/校验）
    if "pfr_flow_model" in config:
        pfr_flow_model = str(config.get("pfr_flow_model", "")).strip()
        if pfr_flow_model not in (
            PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
            PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
        ):
            return False, f"无效的 pfr_flow_model：{pfr_flow_model}"

    if config["kinetic_model"] not in KINETIC_MODELS:
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
        except (ValueError, TypeError):
            return False, "rtol 无法转换为数值"
        if (not np.isfinite(rtol)) or (rtol <= 0.0):
            return False, "rtol 必须为正且有限"
    if "atol" in config:
        try:
            atol = float(config["atol"])
        except (ValueError, TypeError):
            return False, "atol 无法转换为数值"
        if (not np.isfinite(atol)) or (atol <= 0.0):
            return False, "atol 必须为正且有限"

    if "max_step_fraction" in config:
        try:
            value = float(config["max_step_fraction"])
        except (ValueError, TypeError):
            return False, "max_step_fraction 无法转换为数值"
        if (not np.isfinite(value)) or (value < 0.0):
            return False, "max_step_fraction 必须为非负且有限（0 表示不限制）"

    ok, msg = _check_positive_float("diff_step_rel")
    if not ok:
        return ok, msg

    ok, msg = _check_positive_int("max_nfev", min_value=1)
    if not ok:
        return ok, msg

    ok, msg = _check_positive_int("max_nfev_coarse", min_value=1)
    if not ok:
        return ok, msg

    ok, msg = _check_positive_int("n_starts", min_value=1)
    if not ok:
        return ok, msg

    ok, msg = _check_nonnegative_int("random_seed")
    if not ok:
        return ok, msg

    for bool_key in ["use_x_scale_jac", "use_multi_start"]:
        if bool_key in config and not isinstance(config[bool_key], (bool, np.bool_)):
            return False, f"{bool_key} 必须为布尔值（true/false）"

    # 说明：历史版本曾导出过 residual_penalty_* 字段；当前版本不再使用，导入时忽略即可。

    # 可选项：输出模式与目标物种
    if "output_mode" in config:
        output_mode = str(config["output_mode"])

        allowed_output_modes = (
            OUTPUT_MODES_BATCH if reactor_type == REACTOR_TYPE_BSTR else OUTPUT_MODES_FLOW
        )

        if output_mode not in allowed_output_modes:
            return False, f"无效的 output_mode：{output_mode}"

    species_names = _parse_species_text(config["species_text"])
    if len(species_names) < 1:
        return False, "species_text 解析后为空，请检查物种名"

    try:
        n_reactions = int(config["n_reactions"])
    except (ValueError, TypeError):
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
    ok, msg = _check_array_values("k0_guess", must_be_positive=True)
    if not ok:
        return ok, msg

    ok, msg = _check_array_shape("ea_guess_J_mol", (n_reactions,), float)
    if not ok:
        return ok, msg
    ok, msg = _check_array_values("ea_guess_J_mol", must_be_positive=False)
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
        # K0_ads 允许为0（不参与吸附的物种），只需检查非负即可
        if "K0_ads" in config:
            try:
                K0_ads_arr = np.asarray(config["K0_ads"], dtype=float)
            except (ValueError, TypeError):
                return False, "配置项 K0_ads 无法转换为数值数组"
            if not np.all(np.isfinite(K0_ads_arr)):
                return False, "配置项 K0_ads 包含无效值（NaN 或 Inf）"
            if np.any(K0_ads_arr < 0):
                return False, "配置项 K0_ads 必须全部为非负值"

        ok, msg = _check_array_shape("Ea_K_J_mol", (n_species,), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_values("Ea_K_J_mol", must_be_positive=False)
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
        ok, msg = _check_array_values("m_inhibition", must_be_positive=False)
        if not ok:
            return ok, msg

        ok, msg = _check_array_shape("fit_m_flags", (n_reactions,), bool)
        if not ok:
            return ok, msg

    if config["kinetic_model"] == "reversible":
        ok, msg = _check_array_shape("k0_rev", (n_reactions,), float)
        if not ok:
            return ok, msg
        # 说明：
        # - k0_rev 作为“逆反应初值”允许为 0（表示逆向贡献为 0 或暂不启用逆向）。
        # - 但若用户选择“拟合 k0_rev”，则对应反应的 k0_rev 必须严格为正，
        #   否则优化器在边界/比例上更容易出现不可解释行为。
        if "k0_rev" in config:
            try:
                k0_rev_arr = np.asarray(config["k0_rev"], dtype=float)
            except (ValueError, TypeError):
                return False, "配置项 k0_rev 无法转换为数值数组"
            if not np.all(np.isfinite(k0_rev_arr)):
                return False, "配置项 k0_rev 包含无效值（NaN 或 Inf）"
            if np.any(k0_rev_arr < 0):
                return False, "配置项 k0_rev 必须全部为非负值（允许为 0）"

        ok, msg = _check_array_shape("ea_rev_J_mol", (n_reactions,), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_values("ea_rev_J_mol", must_be_positive=False)
        if not ok:
            return ok, msg

        ok, msg = _check_array_shape("fit_k0_rev_flags", (n_reactions,), bool)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape("fit_ea_rev_flags", (n_reactions,), bool)
        if not ok:
            return ok, msg

        # 若拟合 k0_rev，则对应初值必须 > 0
        if ("fit_k0_rev_flags" in config) and np.any(
            np.asarray(config["fit_k0_rev_flags"], dtype=bool)
        ):
            if "k0_rev" not in config:
                return False, "拟合 k0_rev 时必须提供 k0_rev 初值"
            k0_rev_arr = np.asarray(config["k0_rev"], dtype=float)
            fit_mask = np.asarray(config["fit_k0_rev_flags"], dtype=bool)
            if np.any(k0_rev_arr[fit_mask] <= 0.0):
                return False, "配置项 k0_rev：被选中拟合的位置必须为正值（>0）"
        ok, msg = _check_array_shape("order_rev", (n_reactions, n_species), float)
        if not ok:
            return ok, msg
        ok, msg = _check_array_shape(
            "fit_order_rev_flags_matrix", (n_reactions, n_species), bool
        )
        if not ok:
            return ok, msg

    # 验证参数边界的合理性
    if "k0_min" in config and "k0_max" in config:
        try:
            k0_min = float(config["k0_min"])
            k0_max = float(config["k0_max"])
        except (ValueError, TypeError):
            return False, "k0_min 或 k0_max 无法转换为数值"
        if not (np.isfinite(k0_min) and np.isfinite(k0_max)):
            return False, "k0_min 或 k0_max 包含无效值（NaN 或 Inf）"
        if k0_min <= 0:
            return False, "k0_min 必须为正值"
        if k0_max <= 0:
            return False, "k0_max 必须为正值"
        if k0_min >= k0_max:
            return False, f"k0_min ({k0_min}) 必须小于 k0_max ({k0_max})"

    if "ea_min_J_mol" in config and "ea_max_J_mol" in config:
        try:
            ea_min = float(config["ea_min_J_mol"])
            ea_max = float(config["ea_max_J_mol"])
        except (ValueError, TypeError):
            return False, "ea_min_J_mol 或 ea_max_J_mol 无法转换为数值"
        if not (np.isfinite(ea_min) and np.isfinite(ea_max)):
            return False, "ea_min_J_mol 或 ea_max_J_mol 包含无效值（NaN 或 Inf）"
        if ea_min >= ea_max:
            return False, f"ea_min_J_mol ({ea_min}) 必须小于 ea_max_J_mol ({ea_max})"

    # 验证拟合参数边界（针对可逆反应模型）
    if config["kinetic_model"] == "reversible":
        if "k0_rev_min" in config and "k0_rev_max" in config:
            try:
                k0_rev_min = float(config["k0_rev_min"])
                k0_rev_max = float(config["k0_rev_max"])
            except (ValueError, TypeError):
                return False, "k0_rev_min 或 k0_rev_max 无法转换为数值"
            if not (np.isfinite(k0_rev_min) and np.isfinite(k0_rev_max)):
                return False, "k0_rev_min 或 k0_rev_max 包含无效值（NaN 或 Inf）"
            if k0_rev_min <= 0:
                return False, "k0_rev_min 必须为正值"
            if k0_rev_max <= 0:
                return False, "k0_rev_max 必须为正值"
            if k0_rev_min >= k0_rev_max:
                return False, f"k0_rev_min ({k0_rev_min}) 必须小于 k0_rev_max ({k0_rev_max})"

        if "ea_rev_min_J_mol" in config and "ea_rev_max_J_mol" in config:
            try:
                ea_rev_min = float(config["ea_rev_min_J_mol"])
                ea_rev_max = float(config["ea_rev_max_J_mol"])
            except (ValueError, TypeError):
                return False, "ea_rev_min_J_mol 或 ea_rev_max_J_mol 无法转换为数值"
            if not (np.isfinite(ea_rev_min) and np.isfinite(ea_rev_max)):
                return False, "ea_rev_min_J_mol 或 ea_rev_max_J_mol 包含无效值（NaN 或 Inf）"
            if ea_rev_min >= ea_rev_max:
                return False, f"ea_rev_min_J_mol ({ea_rev_min}) 必须小于 ea_rev_max_J_mol ({ea_rev_max})"

    return True, ""
