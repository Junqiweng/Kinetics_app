from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import modules.browser_storage as browser_storage
import modules.config_manager as config_manager
import modules.ui_text as ui_text
from modules.constants import (
    DEFAULT_DIFF_STEP_REL,
    DEFAULT_EA_K_MAX_J_MOL,
    DEFAULT_EA_K_MIN_J_MOL,
    DEFAULT_EA_MAX_J_MOL,
    DEFAULT_EA_MIN_J_MOL,
    DEFAULT_K0_ADS_MAX,
    DEFAULT_K0_ADS_MIN,
    DEFAULT_K0_MAX,
    DEFAULT_K0_MIN,
    DEFAULT_MAX_NFEV,
    DEFAULT_MAX_NFEV_COARSE,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_N_STARTS,
    DEFAULT_ORDER_MAX,
    DEFAULT_ORDER_MIN,
    DEFAULT_RANDOM_SEED,
    OUTPUT_MODE_COUT,
    OUTPUT_MODES_BATCH,
    OUTPUT_MODES_FLOW,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
    UI_DATA_PREVIEW_HEIGHT_PX,
    UI_DATA_PREVIEW_ROWS,
)
from modules.upload_persistence import (
    _delete_persisted_upload,
    _read_csv_bytes_cached,
    _save_persisted_upload,
)


def render_data_tab(tab_data, ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    reactor_type = ctx["reactor_type"]
    pfr_flow_model = ctx["pfr_flow_model"]
    species_names = ctx["species_names"]
    session_id = ctx["session_id"]
    export_config_placeholder = ctx["export_config_placeholder"]
    kinetic_model = ctx["kinetic_model"]
    solver_method = ctx["solver_method"]
    rtol = ctx["rtol"]
    atol = ctx["atol"]
    species_text = ctx["species_text"]
    n_reactions = ctx["n_reactions"]
    stoich_matrix = ctx["stoich_matrix"]
    order_guess = ctx["order_guess"]
    fit_order_flags_matrix = ctx["fit_order_flags_matrix"]
    k0_guess = ctx["k0_guess"]
    ea_guess_J_mol = ctx["ea_guess_J_mol"]
    fit_k0_flags = ctx["fit_k0_flags"]
    fit_ea_flags = ctx["fit_ea_flags"]
    K0_ads = ctx["K0_ads"]
    Ea_K_J_mol = ctx["Ea_K_J_mol"]
    fit_K0_ads_flags = ctx["fit_K0_ads_flags"]
    fit_Ea_K_flags = ctx["fit_Ea_K_flags"]
    m_inhibition = ctx["m_inhibition"]
    fit_m_flags = ctx["fit_m_flags"]
    k0_rev = ctx["k0_rev"]
    ea_rev_J_mol = ctx["ea_rev_J_mol"]
    fit_k0_rev_flags = ctx["fit_k0_rev_flags"]
    fit_ea_rev_flags = ctx["fit_ea_rev_flags"]
    order_rev = ctx["order_rev"]
    fit_order_rev_flags_matrix = ctx["fit_order_rev_flags_matrix"]
    # ---------------- 选项卡 2：数据 ----------------
    data_df = st.session_state.get("data_df_cached", None)
    output_mode = OUTPUT_MODE_COUT
    output_species_list = []

    with tab_data:
        # --- 拟合目标变量 + 物种选择（同一行）---
        col_target1, col_target2 = st.columns(2)
        with col_target1:
            opts = (
                OUTPUT_MODES_FLOW
                if reactor_type in (REACTOR_TYPE_PFR, REACTOR_TYPE_CSTR)
                else OUTPUT_MODES_BATCH
            )
            if ("cfg_output_mode" in st.session_state) and (
                str(st.session_state["cfg_output_mode"]) not in opts
            ):
                st.session_state["cfg_output_mode"] = opts[0]
            output_mode = st.selectbox(
                "拟合目标变量",
                opts,
                index=(
                    opts.index(get_cfg("output_mode", opts[0]))
                    if get_cfg("output_mode", opts[0]) in opts
                    else 0
                ),
                format_func=lambda x: ui_text.map_label(
                    ui_text.OUTPUT_MODE_LABELS, str(x)
                ),
                key="cfg_output_mode",
                help="选择用于拟合的测量量类型：Cout（出口浓度）、Fout（出口摩尔流率）、xout（出口摩尔分率）。",
            )

        with col_target2:
            # 读取配置中保存的物种列表，并清理无效物种
            saved_species_list = get_cfg("output_species_list", None)
            if saved_species_list is not None and isinstance(saved_species_list, list):
                # 从配置中读取，过滤掉不在当前物种列表中的物种
                valid_species = [
                    str(x) for x in saved_species_list if str(x) in species_names
                ]
                if valid_species:
                    default_species = valid_species
                else:
                    default_species = list(species_names)
            else:
                default_species = list(species_names)

            # 重要：不要同时给 multiselect 的 default=... 并且又写 session_state[key]，
            # 否则会触发 Streamlit 警告：
            # "The widget with key ... was created with a default value but also had its value set via the Session State API."
            # 这里统一以 session_state 作为单一数据源。
            if "cfg_output_species_list" not in st.session_state:
                st.session_state["cfg_output_species_list"] = default_species
            else:
                current_list = st.session_state.get("cfg_output_species_list", [])
                if not isinstance(current_list, list):
                    current_list = []
                cleaned_list = [str(x) for x in current_list if str(x) in species_names]
                st.session_state["cfg_output_species_list"] = (
                    cleaned_list if cleaned_list else default_species
                )

            fit_mask = st.multiselect(
                "选择进入目标函数的物种",
                species_names,
                key="cfg_output_species_list",
                help="选择哪些物种的测量值用于计算拟合残差",
            )
            output_species_list = fit_mask

        st.divider()
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.markdown("#### 1. 下载模板")
            # 根据 output_mode 决定测量列
            if output_mode.startswith("F"):
                meas_cols = [f"Fout_{s}_mol_s" for s in species_names]
            elif output_mode.startswith("x"):
                meas_cols = [f"xout_{s}" for s in species_names]
            else:
                meas_cols = [f"Cout_{s}_mol_m3" for s in species_names]

            # 根据反应器类型决定输入条件列
            if reactor_type == REACTOR_TYPE_PFR:
                if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
                    # 气相 PFR：理想气体、等温、恒压 P（不考虑压降）
                    # 入口强制使用 F0_*，不再允许用 C0_* + vdot 换算。
                    inlet_cols = [f"F0_{s}_mol_s" for s in species_names]
                    cols = ["V_m3", "T_K", "P_Pa"] + inlet_cols + meas_cols
                else:
                    # 液相 PFR：体积流量 vdot 近似恒定
                    # 约定：当拟合目标为 Cout 时，入口也允许使用浓度 C0_*（并由 vdot 自动换算为 F0 参与计算）
                    inlet_cols = (
                        [f"C0_{s}_mol_m3" for s in species_names]
                        if output_mode.startswith("C")
                        else [f"F0_{s}_mol_s" for s in species_names]
                    )
                    cols = ["V_m3", "T_K", "vdot_m3_s"] + inlet_cols + meas_cols
            elif reactor_type == REACTOR_TYPE_CSTR:
                cols = (
                    ["V_m3", "T_K", "vdot_m3_s"]
                    + [f"C0_{s}_mol_m3" for s in species_names]
                    + meas_cols
                )
            else:
                # BSTR
                cols = (
                    ["t_s", "T_K"]
                    + [f"C0_{s}_mol_m3" for s in species_names]
                    + meas_cols
                )

            template_csv = (
                pd.DataFrame(columns=cols).to_csv(index=False).encode("utf-8")
            )
            # 动态生成模板文件名，包含反应器类型和测量类型
            template_filename = f"template_{reactor_type}_{output_mode.split()[0]}.csv"
            st.download_button(
                "📥 下载 CSV 模板", template_csv, template_filename, "text/csv"
            )
            st.caption(
                f"模板包含 {len(cols)} 列：输入条件 + {output_mode.split()[0]} 测量值"
            )

        with col_d2:
            st.markdown("#### 2. 上传数据")
            csv_uploader_key = (
                f"uploaded_csv_{int(st.session_state.get('uploader_ver_csv', 0))}"
            )
            if (
                "uploaded_csv_bytes" in st.session_state
                and st.session_state["uploaded_csv_bytes"]
            ):
                cached_name = str(st.session_state.get("uploaded_csv_name", "")).strip()
                cached_text = (
                    f"已缓存文件：{cached_name}" if cached_name else "已缓存上传文件"
                )
                st.caption(cached_text + "（页面刷新/切换不会丢失，除非手动删除）")
                if st.button("🗑️ 删除已上传文件", key="delete_uploaded_csv"):
                    for k in ["uploaded_csv_bytes", "uploaded_csv_name"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    if "data_df_cached" in st.session_state:
                        del st.session_state["data_df_cached"]
                    ok, message = _delete_persisted_upload(session_id)
                    if not ok:
                        st.warning(message)
                    if csv_uploader_key in st.session_state:
                        del st.session_state[csv_uploader_key]
                    st.session_state["uploader_ver_csv"] = (
                        int(st.session_state.get("uploader_ver_csv", 0)) + 1
                    )
                    st.rerun()

            uploaded_file = st.file_uploader(
                "上传 CSV",
                type=["csv"],
                label_visibility="collapsed",
                key=csv_uploader_key,
            )

        if uploaded_file:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_name = str(getattr(uploaded_file, "name", "")).strip()
                st.session_state["uploaded_csv_bytes"] = uploaded_bytes
                st.session_state["uploaded_csv_name"] = uploaded_name
                ok, message = _save_persisted_upload(
                    uploaded_bytes, uploaded_name, session_id
                )
                if not ok:
                    st.warning(message)
            except Exception as exc:
                st.error(f"读取上传文件失败: {exc}")

        if uploaded_file or (
            "uploaded_csv_bytes" in st.session_state
            and st.session_state["uploaded_csv_bytes"]
        ):
            try:
                if uploaded_file:
                    csv_bytes = uploaded_file.getvalue()
                else:
                    csv_bytes = st.session_state["uploaded_csv_bytes"]
                data_df = _read_csv_bytes_cached(csv_bytes)
                st.session_state["data_df_cached"] = data_df
                st.markdown("#### 数据预览")
                st.dataframe(
                    data_df.head(UI_DATA_PREVIEW_ROWS),
                    use_container_width=True,
                    height=UI_DATA_PREVIEW_HEIGHT_PX,
                )
            except Exception as exc:
                st.error(f"CSV 读取失败: {exc}")
                data_df = None

    # --- 构建导出配置（基础版；若拟合页启用高级设置，会在拟合页再次更新）---
    if export_config_placeholder is not None:
        export_k0_min = float(get_cfg("k0_min", DEFAULT_K0_MIN))
        export_k0_max = float(get_cfg("k0_max", DEFAULT_K0_MAX))
        export_ea_min = float(get_cfg("ea_min_J_mol", DEFAULT_EA_MIN_J_MOL))
        export_ea_max = float(get_cfg("ea_max_J_mol", DEFAULT_EA_MAX_J_MOL))
        export_ord_min = float(get_cfg("order_min", DEFAULT_ORDER_MIN))
        export_ord_max = float(get_cfg("order_max", DEFAULT_ORDER_MAX))
        export_K0_ads_min = float(get_cfg("K0_ads_min", DEFAULT_K0_ADS_MIN))
        export_K0_ads_max = float(get_cfg("K0_ads_max", DEFAULT_K0_ADS_MAX))
        export_Ea_K_min = float(get_cfg("Ea_K_min", DEFAULT_EA_K_MIN_J_MOL))
        export_Ea_K_max = float(get_cfg("Ea_K_max", DEFAULT_EA_K_MAX_J_MOL))

        export_diff_step_rel = float(get_cfg("diff_step_rel", DEFAULT_DIFF_STEP_REL))
        export_max_nfev = int(get_cfg("max_nfev", DEFAULT_MAX_NFEV))
        export_use_x_scale_jac = bool(get_cfg("use_x_scale_jac", True))
        export_use_ms = bool(get_cfg("use_multi_start", True))
        export_n_starts = int(get_cfg("n_starts", DEFAULT_N_STARTS))
        export_max_nfev_coarse = int(
            get_cfg("max_nfev_coarse", DEFAULT_MAX_NFEV_COARSE)
        )
        export_random_seed = int(get_cfg("random_seed", DEFAULT_RANDOM_SEED))
        export_max_step_fraction = float(
            get_cfg("max_step_fraction", DEFAULT_MAX_STEP_FRACTION)
        )

        export_cfg = config_manager.collect_config(
            reactor_type=reactor_type,
            pfr_flow_model=str(pfr_flow_model),
            kinetic_model=kinetic_model,
            solver_method=solver_method,
            rtol=float(rtol),
            atol=float(atol),
            max_step_fraction=export_max_step_fraction,
            species_text=str(species_text),
            n_reactions=int(n_reactions),
            stoich_matrix=np.asarray(stoich_matrix, dtype=float),
            order_guess=np.asarray(order_guess, dtype=float),
            fit_order_flags_matrix=np.asarray(fit_order_flags_matrix, dtype=bool),
            k0_guess=np.asarray(k0_guess, dtype=float),
            ea_guess_J_mol=np.asarray(ea_guess_J_mol, dtype=float),
            fit_k0_flags=np.asarray(fit_k0_flags, dtype=bool),
            fit_ea_flags=np.asarray(fit_ea_flags, dtype=bool),
            K0_ads=None if K0_ads is None else np.asarray(K0_ads, dtype=float),
            Ea_K_J_mol=(
                None if Ea_K_J_mol is None else np.asarray(Ea_K_J_mol, dtype=float)
            ),
            fit_K0_ads_flags=(
                None
                if fit_K0_ads_flags is None
                else np.asarray(fit_K0_ads_flags, dtype=bool)
            ),
            fit_Ea_K_flags=(
                None
                if fit_Ea_K_flags is None
                else np.asarray(fit_Ea_K_flags, dtype=bool)
            ),
            m_inhibition=(
                None if m_inhibition is None else np.asarray(m_inhibition, dtype=float)
            ),
            fit_m_flags=(
                None if fit_m_flags is None else np.asarray(fit_m_flags, dtype=bool)
            ),
            k0_rev=None if k0_rev is None else np.asarray(k0_rev, dtype=float),
            ea_rev_J_mol=(
                None if ea_rev_J_mol is None else np.asarray(ea_rev_J_mol, dtype=float)
            ),
            fit_k0_rev_flags=(
                None
                if fit_k0_rev_flags is None
                else np.asarray(fit_k0_rev_flags, dtype=bool)
            ),
            fit_ea_rev_flags=(
                None
                if fit_ea_rev_flags is None
                else np.asarray(fit_ea_rev_flags, dtype=bool)
            ),
            order_rev=None if order_rev is None else np.asarray(order_rev, dtype=float),
            fit_order_rev_flags_matrix=(
                None
                if fit_order_rev_flags_matrix is None
                else np.asarray(fit_order_rev_flags_matrix, dtype=bool)
            ),
            output_mode=str(output_mode),
            output_species_list=list(output_species_list),
            k0_min=export_k0_min,
            k0_max=export_k0_max,
            ea_min_J_mol=export_ea_min,
            ea_max_J_mol=export_ea_max,
            order_min=export_ord_min,
            order_max=export_ord_max,
            K0_ads_min=export_K0_ads_min,
            K0_ads_max=export_K0_ads_max,
            Ea_K_min=export_Ea_K_min,
            Ea_K_max=export_Ea_K_max,
            diff_step_rel=export_diff_step_rel,
            max_nfev=export_max_nfev,
            use_x_scale_jac=export_use_x_scale_jac,
            use_multi_start=export_use_ms,
            n_starts=export_n_starts,
            max_nfev_coarse=export_max_nfev_coarse,
            random_seed=export_random_seed,
        )
        is_valid_cfg, _ = config_manager.validate_config(export_cfg)
        if is_valid_cfg:
            # 本地文件系统保存（用于本地运行）
            ok, message = config_manager.auto_save_config(export_cfg, session_id)
            if not ok:
                st.warning(message)
            # 浏览器 LocalStorage 保存（用于 Streamlit Cloud 等云环境）
            browser_storage.save_config_to_browser(export_cfg)
        export_config_bytes = config_manager.export_config_to_json(export_cfg).encode(
            "utf-8"
        )
        export_config_placeholder.download_button(
            "📥 导出当前配置 (JSON)",
            export_config_bytes,
            file_name="kinetics_config.json",
            mime="application/json",
            use_container_width=True,
            key="export_config_download_basic",
        )
    return {
        "data_df": data_df,
        "output_mode": output_mode,
        "output_species_list": output_species_list,
    }
