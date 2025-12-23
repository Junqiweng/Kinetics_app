from __future__ import annotations

import streamlit as st


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
        reactor_type_cfg = "BSTR"
    if reactor_type_cfg in ["PFR", "CSTR", "BSTR"]:
        st.session_state["cfg_reactor_type"] = reactor_type_cfg
    if kinetic_model_cfg in ["power_law", "langmuir_hinshelwood", "reversible"]:
        st.session_state["cfg_kinetic_model"] = kinetic_model_cfg
    if solver_method_cfg in ["RK45", "BDF", "Radau"]:
        st.session_state["cfg_solver_method"] = solver_method_cfg

    if "rtol" in config:
        st.session_state["cfg_rtol"] = float(config.get("rtol", 1e-6))
    if "atol" in config:
        st.session_state["cfg_atol"] = float(config.get("atol", 1e-9))

    if "species_text" in config:
        st.session_state["cfg_species_text"] = str(config.get("species_text", "")).strip()
    if "n_reactions" in config:
        st.session_state["cfg_n_reactions"] = int(config.get("n_reactions", 1))

    output_mode_cfg = str(config.get("output_mode", "")).strip()
    if reactor_type_cfg == "BSTR":
        allowed_output_modes = ["Cout (mol/m^3)", "X (conversion)"]
    else:
        allowed_output_modes = ["Fout (mol/s)", "Cout (mol/m^3)", "X (conversion)"]
    if output_mode_cfg in allowed_output_modes:
        st.session_state["cfg_output_mode"] = output_mode_cfg
    elif allowed_output_modes:
        st.session_state["cfg_output_mode"] = allowed_output_modes[0]

    output_species_list_cfg = config.get("output_species_list", None)
    if isinstance(output_species_list_cfg, list):
        st.session_state["cfg_output_species_list"] = [str(x) for x in output_species_list_cfg]

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


def _clear_config_related_state() -> None:
    """
    清理“配置相关”的 session_state，使 UI 真正回到默认值。

    说明：必须在 widgets 创建之前调用，否则会触发 Streamlit 的
    “cannot be modified after the widget ... is instantiated” 报错。
    """
    # 先“切换”上传控件的 key：Streamlit 的 file_uploader 在某些情况下即使删掉 session_state，
    # 前端仍可能显示旧文件；通过更换 key 强制创建一个全新的 uploader，从而清空显示。
    st.session_state["uploader_ver_config_json"] = (
        int(st.session_state.get("uploader_ver_config_json", 0)) + 1
    )
    st.session_state["uploader_ver_csv"] = int(st.session_state.get("uploader_ver_csv", 0)) + 1

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

    # 3.1) 上传控件 / 上传缓存（配置 JSON + 实验数据 CSV）
    for key in list(st.session_state.keys()):
        key_str = str(key)
        if key_str.startswith("uploaded_config_json_") or key_str.startswith("uploaded_csv_"):
            keys_to_delete.append(key_str)
    keys_to_delete.extend(["data_df_cached"])

    # 4) 拟合结果缓存（避免“配置已变但结果仍是旧的”造成误解）
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
