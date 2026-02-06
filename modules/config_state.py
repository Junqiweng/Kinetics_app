# 文件作用：处理配置导入/重置时的 `st.session_state` 写入与提示逻辑，避免在控件创建后修改状态导致报错。

from __future__ import annotations

import streamlit as st

from .constants import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    KINETIC_MODELS,
    OUTPUT_MODE_COUT,
    OUTPUT_MODE_FOUT,
    OUTPUT_MODES_BATCH,
    OUTPUT_MODES_FLOW,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPES,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_PFR,
)


def _apply_imported_config_to_widget_state(config: dict) -> None:
    """
    将导入配置写入 widget 对应的 session_state。

    关键点：必须在 widget 创建之前写入，否则会触发
    “cannot be modified after the widget ... is instantiated”。
    """
    reactor_type_cfg = str(config.get("reactor_type", "")).strip()
    kinetic_model_cfg = str(config.get("kinetic_model", "")).strip()
    solver_method_cfg = str(config.get("solver_method", "")).strip()

    if reactor_type_cfg == "Batch":
        reactor_type_cfg = REACTOR_TYPE_BSTR
    if reactor_type_cfg in REACTOR_TYPES:
        st.session_state["cfg_reactor_type"] = reactor_type_cfg

    # PFR 流动模型（仅 PFR 时显示控件；但可以提前写入 session_state 以便用户切回 PFR）
    pfr_flow_model_cfg = str(config.get("pfr_flow_model", "")).strip()
    if pfr_flow_model_cfg in (
        PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
        PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    ):
        st.session_state["cfg_pfr_flow_model"] = pfr_flow_model_cfg
    elif reactor_type_cfg == REACTOR_TYPE_PFR:
        # 若导入配置未提供该键，则回退到默认（确保控件有稳定值）
        st.session_state["cfg_pfr_flow_model"] = PFR_FLOW_MODEL_LIQUID_CONST_VDOT
    if kinetic_model_cfg in KINETIC_MODELS:
        st.session_state["cfg_kinetic_model"] = kinetic_model_cfg
    if solver_method_cfg in ["RK45", "BDF", "Radau"]:
        st.session_state["cfg_solver_method"] = solver_method_cfg

    if "rtol" in config:
        st.session_state["cfg_rtol"] = float(config.get("rtol", DEFAULT_RTOL))
    if "atol" in config:
        st.session_state["cfg_atol"] = float(config.get("atol", DEFAULT_ATOL))

    if "species_text" in config:
        st.session_state["cfg_species_text"] = str(
            config.get("species_text", "")
        ).strip()
    if "n_reactions" in config:
        st.session_state["cfg_n_reactions"] = int(config.get("n_reactions", 1))

    output_mode_cfg = str(config.get("output_mode", "")).strip()
    allowed_output_modes = (
        OUTPUT_MODES_BATCH
        if reactor_type_cfg == REACTOR_TYPE_BSTR
        else OUTPUT_MODES_FLOW
    )
    if output_mode_cfg in allowed_output_modes:
        st.session_state["cfg_output_mode"] = output_mode_cfg
    elif allowed_output_modes:
        # 兼容旧配置：若未提供 output_mode 或提供了非法值，则回退到历史默认值
        # - 流动反应器（PFR/CSTR）：默认 Fout
        # - 间歇釜（BSTR）：默认 Cout
        st.session_state["cfg_output_mode"] = (
            OUTPUT_MODE_COUT
            if reactor_type_cfg == REACTOR_TYPE_BSTR
            else OUTPUT_MODE_FOUT
        )

    output_species_list_cfg = config.get("output_species_list", None)
    if isinstance(output_species_list_cfg, list):
        st.session_state["cfg_output_species_list"] = [
            str(x) for x in output_species_list_cfg
        ]

    for key_name in [
        "k0_min",
        "k0_max",
        "ea_min_J_mol",
        "ea_max_J_mol",
        "order_min",
        "order_max",
        "K0_ads_min",
        "K0_ads_max",
        "Ea_K_min",
        "Ea_K_max",
        "diff_step_rel",
        "max_nfev",
        "use_x_scale_jac",
        "use_multi_start",
        "n_starts",
        "max_nfev_coarse",
        "random_seed",
        "max_step_fraction",
    ]:
        if key_name in config:
            st.session_state[f"cfg_{key_name}"] = config[key_name]


def _warn_once(flag_key: str, message: str) -> None:
    """
    避免同一条 warning 在每次 rerun 都重复刷屏。
    """
    if not bool(st.session_state.get(flag_key, False)):
        st.session_state[flag_key] = True
        st.warning(message)


def _clear_config_related_state(
    *,
    keep_csv_data: bool,
    clear_csv_uploader_widget: bool,
) -> None:
    """
    清理“配置相关”的 session_state。

    参数:
        keep_csv_data:
            - True: 保留当前 CSV 缓存（用于“导入配置”场景）
            - False: 清空当前 CSV 缓存（用于“重置默认”场景）
        clear_csv_uploader_widget:
            - True: 切换 CSV uploader key，强制清空前端已选文件显示
            - False: 保留当前 uploader 显示状态
    """
    # 配置 JSON uploader 总是重建，避免导入后仍显示旧文件
    st.session_state["uploader_ver_config_json"] = (
        int(st.session_state.get("uploader_ver_config_json", 0)) + 1
    )
    if clear_csv_uploader_widget:
        st.session_state["uploader_ver_csv"] = (
            int(st.session_state.get("uploader_ver_csv", 0)) + 1
        )

    keys_to_delete: list[str] = []

    # 1) 所有 cfg_* 控件值（全局设置 + 高级设置）
    for key in list(st.session_state.keys()):
        if str(key).startswith("cfg_"):
            keys_to_delete.append(str(key))

    # 2) data_editor / expander 动态 key（否则“回到默认尺寸”时仍可能记住旧表格）
    for key in list(st.session_state.keys()):
        key_str = str(key)
        if key_str.startswith("nu_") or key_str.startswith("lh_m_"):
            keys_to_delete.append(key_str)

    # 3) 导入/导出与初始化标记
    keys_to_delete.extend(
        [
            "imported_config",
            "imported_config_digest",
            "pending_imported_config",
            "config_initialized",
        ]
    )

    # 4) 配置上传控件状态
    for key in list(st.session_state.keys()):
        key_str = str(key)
        if key_str.startswith("uploaded_config_json_"):
            keys_to_delete.append(key_str)
        if clear_csv_uploader_widget and key_str.startswith("uploaded_csv_"):
            keys_to_delete.append(key_str)

    # 5) CSV 数据缓存（仅 reset 场景清空）
    if not keep_csv_data:
        keys_to_delete.extend(["uploaded_csv_bytes", "uploaded_csv_name", "data_df_cached"])

    # 6) 拟合结果缓存（避免“配置已变但结果仍是旧的”造成误解）
    keys_to_delete.extend(
        [
            "fit_results",
            "fit_compare_cache_key",
            "fit_compare_long_df",
            "fit_compare_long_df_all",
            "fitting_timeline",
            "fitting_metrics",
            "fitting_ms_summary",
            "fitting_final_summary",
            "fitting_job_summary",
        ]
    )

    for key in sorted(set(keys_to_delete)):
        if key in st.session_state:
            del st.session_state[key]


def _clear_state_for_imported_config() -> None:
    """
    导入配置专用清理：
    - 清理旧控件状态与拟合结果
    - 保留当前 CSV 数据与 uploader 状态
    """
    _clear_config_related_state(
        keep_csv_data=True,
        clear_csv_uploader_widget=False,
    )


def _clear_state_for_reset_default() -> None:
    """
    重置默认专用清理：
    - 清理旧控件状态与拟合结果
    - 清空当前 CSV 数据与 uploader 状态
    """
    _clear_config_related_state(
        keep_csv_data=False,
        clear_csv_uploader_widget=True,
    )
