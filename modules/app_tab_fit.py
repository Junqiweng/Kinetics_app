from __future__ import annotations

import hashlib
import json
import queue
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import modules.browser_storage as browser_storage
import modules.config_manager as config_manager
import modules.reactors as reactors
import modules.ui_components as ui_comp
import modules.ui_text as ui_text
from modules.app_config_state import _warn_once
from modules.app_data_utils import (
    _build_fit_comparison_long_table,
    _get_measurement_column_name,
    _get_output_unit_text,
)
from modules.app_fitting_background import (
    _count_fitted_parameters,
    _drain_fitting_progress_queue,
    _get_fitting_executor,
    _render_fitting_live_progress,
    _render_fitting_progress_panel,
    _reset_fitting_progress_ui_state,
    _run_fitting_job,
)
from modules.app_plot_helpers import (
    _fit_plot_color,
    _plot_reference_series,
    _style_fit_axis,
    _style_fit_legend,
)
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
    DEFAULT_EA_REV_MAX_J_MOL,
    DEFAULT_EA_REV_MIN_J_MOL,
    DEFAULT_K0_REV_MAX,
    DEFAULT_K0_REV_MIN,
    DEFAULT_ORDER_REV_MAX,
    DEFAULT_ORDER_REV_MIN,
    EPSILON_CONCENTRATION,
    EPSILON_FLOW_RATE,
    R_GAS_J_MOL_K,
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODEL_REVERSIBLE,
    OUTPUT_MODE_COUT,
    OUTPUT_MODE_FOUT,
    OUTPUT_MODE_XOUT,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
    UI_COMPARE_TABLE_HEIGHT_PX,
    UI_MAX_NFEV_STEP,
    UI_MAX_STEP_FRACTION_STEP,
    UI_METRICS_TABLE_HEIGHT_PX,
    UI_PARAM_TABLE_HEIGHT_PX,
    UI_PROFILE_POINTS_DEFAULT,
    UI_PROFILE_POINTS_MAX,
    UI_PROFILE_POINTS_MIN,
    UI_PROFILE_POINTS_STEP,
)


def render_fit_tab(tab_fit, ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    export_config_placeholder = ctx["export_config_placeholder"]
    session_id = ctx["session_id"]
    reactor_type = ctx["reactor_type"]
    pfr_flow_model = ctx["pfr_flow_model"]
    kinetic_model = ctx["kinetic_model"]
    solver_method = ctx["solver_method"]
    rtol = ctx["rtol"]
    atol = ctx["atol"]
    species_text = ctx["species_text"]
    n_reactions = ctx["n_reactions"]
    species_names = ctx["species_names"]
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
    data_df = ctx["data_df"]
    output_mode = ctx["output_mode"]
    output_species_list = ctx["output_species_list"]
    _request_start_fitting = ctx["request_start_fitting"]
    _request_stop_fitting = ctx["request_stop_fitting"]
    # ---------------- 选项卡 3：拟合 ----------------
    with tab_fit:
        fit_results_cached = st.session_state.get("fit_results", None)

        # 允许“无当前数据”时仍能查看历史拟合结果（避免被 st.stop 截断或 len(None) 报错）
        if (data_df is None) and isinstance(fit_results_cached, dict):
            data_df = fit_results_cached.get("data", None)
        if (not output_species_list) and isinstance(fit_results_cached, dict):
            output_species_list = list(fit_results_cached.get("output_species", []))

        if data_df is None:
            st.info("请先在「实验数据」页面上传 CSV 文件（或恢复已缓存的文件）。")
            if fit_results_cached is None:
                st.stop()
        if not output_species_list:
            st.error("请选择至少一个目标物种。")
            if fit_results_cached is None:
                st.stop()

        # --- 高级设置（展开）---
        with st.expander("高级设置与边界 (点击展开)", expanded=False):

            st.markdown("**1. 基础边界设置**")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                k0_min = ui_comp.smart_number_input(
                    "k₀ 下限（k0_min）",
                    value=float(get_cfg("k0_min", DEFAULT_K0_MIN)),
                    key="cfg_k0_min",
                )
                k0_max = ui_comp.smart_number_input(
                    "k₀ 上限（k0_max）",
                    value=float(get_cfg("k0_max", DEFAULT_K0_MAX)),
                    key="cfg_k0_max",
                )
            with col_b2:
                ea_min = ui_comp.smart_number_input(
                    "Eₐ 下限（ea_min_J_mol）",
                    value=float(get_cfg("ea_min_J_mol", DEFAULT_EA_MIN_J_MOL)),
                    key="cfg_ea_min_J_mol",
                )
                ea_max = ui_comp.smart_number_input(
                    "Eₐ 上限（ea_max_J_mol）",
                    value=float(get_cfg("ea_max_J_mol", DEFAULT_EA_MAX_J_MOL)),
                    key="cfg_ea_max_J_mol",
                )
            with col_b3:
                ord_min = ui_comp.smart_number_input(
                    "反应级数下限（order_min）",
                    value=float(get_cfg("order_min", DEFAULT_ORDER_MIN)),
                    key="cfg_order_min",
                )
                ord_max = ui_comp.smart_number_input(
                    "反应级数上限（order_max）",
                    value=float(get_cfg("order_max", DEFAULT_ORDER_MAX)),
                    key="cfg_order_max",
                )

            K0_ads_min = float(get_cfg("K0_ads_min", DEFAULT_K0_ADS_MIN))
            K0_ads_max = float(get_cfg("K0_ads_max", DEFAULT_K0_ADS_MAX))
            Ea_K_min = float(get_cfg("Ea_K_min", DEFAULT_EA_K_MIN_J_MOL))
            Ea_K_max = float(get_cfg("Ea_K_max", DEFAULT_EA_K_MAX_J_MOL))
            if kinetic_model == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
                st.markdown("**1.2 L-H 边界设置**")
                col_lh_b1, col_lh_b2 = st.columns(2)
                with col_lh_b1:
                    K0_ads_min = ui_comp.smart_number_input(
                        "K₀,ads 下限（K0_ads_min）",
                        value=K0_ads_min,
                        key="cfg_K0_ads_min",
                    )
                    K0_ads_max = ui_comp.smart_number_input(
                        "K₀,ads 上限（K0_ads_max）",
                        value=K0_ads_max,
                        key="cfg_K0_ads_max",
                    )
                with col_lh_b2:
                    Ea_K_min = ui_comp.smart_number_input(
                        "Eₐ,K 下限（Ea_K_min）",
                        value=Ea_K_min,
                        key="cfg_Ea_K_min",
                    )
                    Ea_K_max = ui_comp.smart_number_input(
                        "Eₐ,K 上限（Ea_K_max）",
                        value=Ea_K_max,
                        key="cfg_Ea_K_max",
                    )

            # 可逆反应边界（逆反应）
            k0_rev_min = float(get_cfg("k0_rev_min", DEFAULT_K0_REV_MIN))
            k0_rev_max = float(get_cfg("k0_rev_max", DEFAULT_K0_REV_MAX))
            ea_rev_min_J_mol = float(
                get_cfg("ea_rev_min_J_mol", DEFAULT_EA_REV_MIN_J_MOL)
            )
            ea_rev_max_J_mol = float(
                get_cfg("ea_rev_max_J_mol", DEFAULT_EA_REV_MAX_J_MOL)
            )
            order_rev_min = float(get_cfg("order_rev_min", DEFAULT_ORDER_REV_MIN))
            order_rev_max = float(get_cfg("order_rev_max", DEFAULT_ORDER_REV_MAX))
            if kinetic_model == KINETIC_MODEL_REVERSIBLE:
                st.markdown("**1.3 可逆反应边界设置（逆反应）**")
                col_rev_b1, col_rev_b2, col_rev_b3 = st.columns(3)
                with col_rev_b1:
                    k0_rev_min = ui_comp.smart_number_input(
                        "k₀,rev 下限（k0_rev_min）",
                        value=k0_rev_min,
                        key="cfg_k0_rev_min",
                    )
                    k0_rev_max = ui_comp.smart_number_input(
                        "k₀,rev 上限（k0_rev_max）",
                        value=k0_rev_max,
                        key="cfg_k0_rev_max",
                    )
                with col_rev_b2:
                    ea_rev_min_J_mol = ui_comp.smart_number_input(
                        "Eₐ,rev 下限（ea_rev_min_J_mol）",
                        value=ea_rev_min_J_mol,
                        key="cfg_ea_rev_min_J_mol",
                    )
                    ea_rev_max_J_mol = ui_comp.smart_number_input(
                        "Eₐ,rev 上限（ea_rev_max_J_mol）",
                        value=ea_rev_max_J_mol,
                        key="cfg_ea_rev_max_J_mol",
                    )
                with col_rev_b3:
                    order_rev_min = ui_comp.smart_number_input(
                        "逆反应级数下限（order_rev_min）",
                        value=order_rev_min,
                        key="cfg_order_rev_min",
                    )
                    order_rev_max = ui_comp.smart_number_input(
                        "逆反应级数上限（order_rev_max）",
                        value=order_rev_max,
                        key="cfg_order_rev_max",
                    )

            st.divider()
            st.markdown("**2. 算法与鲁棒性**")

            # 第一行：主要迭代参数
            col_iter1, col_iter2, col_iter3 = st.columns(3)
            with col_iter1:
                max_nfev = int(
                    st.number_input(
                        "最大迭代次数（max_nfev）",
                        value=int(get_cfg("max_nfev", DEFAULT_MAX_NFEV)),
                        step=UI_MAX_NFEV_STEP,
                        key="cfg_max_nfev",
                        help="提示：每次外层迭代中，数值差分 Jacobian 需要多次模型调用，因此显示的总调用次数通常大于该值。",
                    )
                )
            with col_iter2:
                diff_step_rel = ui_comp.smart_number_input(
                    "差分步长（diff_step）",
                    value=get_cfg("diff_step_rel", DEFAULT_DIFF_STEP_REL),
                    key="cfg_diff_step_rel",
                    help="用于 least_squares 的数值差分 Jacobian 相对步长；拟合停滞时可适当调大，拟合过粗时可适当调小。",
                )
            with col_iter3:
                max_step_fraction = ui_comp.smart_number_input(
                    "最大步长比例（max_step_fraction）",
                    value=float(
                        get_cfg("max_step_fraction", DEFAULT_MAX_STEP_FRACTION)
                    ),
                    min_value=0.0,
                    max_value=10.0,
                    step=UI_MAX_STEP_FRACTION_STEP,
                    key="cfg_max_step_fraction",
                    help="用于 solve_ivp 的积分步长上限：max_step = fraction × 总时间/总体积；0 表示不限制。",
                )

            # 第二行：Multi-start 相关选项
            col_ms1, col_ms2, col_ms3 = st.columns(3)
            with col_ms1:
                use_ms = st.checkbox(
                    "多起点搜索（Multi-start）",
                    value=bool(get_cfg("use_multi_start", True)),
                    key="cfg_use_multi_start",
                )
            with col_ms2:
                n_starts = int(
                    st.number_input(
                        "起点数量（n_starts）",
                        value=get_cfg("n_starts", DEFAULT_N_STARTS),
                        min_value=1,
                        step=1,
                        key="cfg_n_starts",
                        help="仅在启用多起点搜索且 n_starts > 1 时生效。",
                    )
                )
            with col_ms3:
                max_nfev_coarse = int(
                    st.number_input(
                        "粗拟合迭代上限（max_nfev_coarse）",
                        value=get_cfg("max_nfev_coarse", DEFAULT_MAX_NFEV_COARSE),
                        step=50,
                        key="cfg_max_nfev_coarse",
                        help="仅在启用多起点搜索时，用于每个起点的粗拟合阶段。",
                    )
                )

            # 第三行：其他选项
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                use_x_scale_jac = st.checkbox(
                    "启用雅可比尺度归一（x_scale='jac'）",
                    value=get_cfg("use_x_scale_jac", True),
                    key="cfg_use_x_scale_jac",
                )
            with col_opt2:
                random_seed = int(
                    st.number_input(
                        "随机种子（random_seed）",
                        value=get_cfg("random_seed", DEFAULT_RANDOM_SEED),
                        step=1,
                        key="cfg_random_seed",
                    )
                )

            st.divider()
            st.markdown("**3. 目标函数设置**")
            st.caption("目标函数定义残差的计算方式，不同类型适用于不同数据特征：")

            residual_type_options = [
                "绝对残差",
                "相对残差",
                "百分比残差",
            ]
            residual_type_default = str(get_cfg("residual_type", "绝对残差"))
            if residual_type_default not in residual_type_options:
                residual_type_default = "绝对残差"
            residual_type_index = residual_type_options.index(residual_type_default)

            residual_type = st.selectbox(
                "残差类型",
                options=residual_type_options,
                index=residual_type_index,
                key="cfg_residual_type",
                help="选择用于构建目标函数的残差计算方式",
            )

            # 显示当前残差类型的公式说明
            residual_formula_info = {
                "绝对残差": (
                    "**绝对残差（Absolute Residual）**\n\n"
                    r"$r_i = y_i^{pred} - y_i^{meas}$"
                    "\n\n适用于：测量值数量级相近的数据。当测量值范围差异大时，大值主导拟合。"
                ),
                "相对残差": (
                    "**相对残差（Relative Residual）**\n\n"
                    r"$r_i = \frac{y_i^{pred} - y_i^{meas}}{y_i^{meas}}$"
                    "\n\n适用于：测量值跨越多个数量级的数据。对所有数据点给予相近权重。\n\n"
                    r"⚠️ 注意：若 $y_i^{meas}$ 接近零，残差会变得非常大，可能导致数值不稳定。"
                ),
                "百分比残差": (
                    "**百分比残差（Percentage Residual with offset）**\n\n"
                    r"$r_i = 100 \times \frac{y_i^{pred} - y_i^{meas}}{|y_i^{meas}| + \epsilon}$"
                    "\n\n"
                    r"其中 $\epsilon$ 为小正数（典型值的 1%），避免除零；$r_i$ 的单位为 %。"
                    "\n\n适用于：测量值可能接近零的数据。兼顾相对误差与数值稳定性。"
                ),
            }
            with st.container(border=True):
                st.markdown(
                    '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
                )
                st.markdown(residual_formula_info.get(residual_type, ""))

            st.divider()
            st.caption(
                "说明：当模型计算失败（如 solve_ivp 失败）时，残差会使用系统默认罚项（不在 UI 中提供调节）。"
            )

        # 使用高级设置更新导出配置（仅当拟合页激活且控件已创建时）
        if export_config_placeholder is not None:
            export_config_placeholder.empty()
            export_cfg = config_manager.collect_config(
                reactor_type=reactor_type,
                pfr_flow_model=str(pfr_flow_model),
                kinetic_model=kinetic_model,
                solver_method=solver_method,
                rtol=float(rtol),
                atol=float(atol),
                max_step_fraction=float(max_step_fraction),
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
                    None
                    if m_inhibition is None
                    else np.asarray(m_inhibition, dtype=float)
                ),
                fit_m_flags=(
                    None if fit_m_flags is None else np.asarray(fit_m_flags, dtype=bool)
                ),
                k0_rev=None if k0_rev is None else np.asarray(k0_rev, dtype=float),
                ea_rev_J_mol=(
                    None
                    if ea_rev_J_mol is None
                    else np.asarray(ea_rev_J_mol, dtype=float)
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
                order_rev=(
                    None if order_rev is None else np.asarray(order_rev, dtype=float)
                ),
                fit_order_rev_flags_matrix=(
                    None
                    if fit_order_rev_flags_matrix is None
                    else np.asarray(fit_order_rev_flags_matrix, dtype=bool)
                ),
                output_mode=str(output_mode),
                output_species_list=list(output_species_list),
                k0_min=float(k0_min),
                k0_max=float(k0_max),
                ea_min_J_mol=float(ea_min),
                ea_max_J_mol=float(ea_max),
                order_min=float(ord_min),
                order_max=float(ord_max),
                K0_ads_min=float(K0_ads_min),
                K0_ads_max=float(K0_ads_max),
                Ea_K_min=float(Ea_K_min),
                Ea_K_max=float(Ea_K_max),
                k0_rev_min=float(k0_rev_min),
                k0_rev_max=float(k0_rev_max),
                ea_rev_min_J_mol=float(ea_rev_min_J_mol),
                ea_rev_max_J_mol=float(ea_rev_max_J_mol),
                order_rev_min=float(order_rev_min),
                order_rev_max=float(order_rev_max),
                residual_type=str(residual_type),
                diff_step_rel=float(diff_step_rel),
                max_nfev=int(max_nfev),
                use_x_scale_jac=bool(use_x_scale_jac),
                use_multi_start=bool(use_ms),
                n_starts=int(n_starts),
                max_nfev_coarse=int(max_nfev_coarse),
                random_seed=int(random_seed),
            )
            is_valid_cfg, _ = config_manager.validate_config(export_cfg)
            if is_valid_cfg:
                # 本地文件系统保存（用于本地运行）
                ok, message = config_manager.auto_save_config(export_cfg, session_id)
                if not ok:
                    st.warning(message)
                # 浏览器 LocalStorage 保存（用于 Streamlit Cloud 等云环境）
                browser_storage.save_config_to_browser(export_cfg)
            export_config_bytes = config_manager.export_config_to_json(
                export_cfg
            ).encode("utf-8")
            export_config_placeholder.download_button(
                "📥 导出当前配置 (JSON)",
                export_config_bytes,
                file_name="kinetics_config.json",
                mime="application/json",
                use_container_width=True,
                key="export_config_download_advanced",
            )

        # --- 操作按钮 ---

        _drain_fitting_progress_queue()

        fitting_stop_event = st.session_state.get("fitting_stop_event", None)
        if fitting_stop_event is None:
            fitting_stop_event = threading.Event()
            st.session_state["fitting_stop_event"] = fitting_stop_event

        fitting_future = st.session_state.get("fitting_future", None)
        fitting_running = bool(st.session_state.get("fitting_running", False))

        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            col_act1, col_act2, col_act3, col_act4, col_act5 = st.columns(
                [3, 1, 1, 1, 1], vertical_alignment="center"
            )
            col_act1.button(
                "🚀 开始拟合",
                type="primary",
                disabled=fitting_running,
                use_container_width=True,
                on_click=_request_start_fitting,
            )
            col_act2.button(
                "⏹️ 终止",
                type="secondary",
                disabled=not fitting_running,
                use_container_width=True,
                on_click=_request_stop_fitting,
            )
            auto_refresh = col_act3.checkbox(
                "自动刷新",
                value=bool(st.session_state.get("fitting_auto_refresh", True)),
                disabled=not fitting_running,
                help="开启后，页面会按设定间隔自动刷新，以持续更新拟合进度与阶段信息；关闭可降低页面刷新负载与 CPU 占用。",
            )
            col_interval_label, col_interval_input = col_act5.columns(
                [1.1, 1.4], vertical_alignment="center"
            )
            col_interval_label.markdown(
                '<div class="kinetics-inline-label">间隔(s)</div>',
                unsafe_allow_html=True,
            )
            refresh_interval_s = float(
                ui_comp.smart_number_input(
                    "间隔(s)",
                    value=float(
                        st.session_state.get("fitting_refresh_interval_s", 2.0)
                    ),
                    min_value=0.5,
                    max_value=10.0,
                    step=0.5,
                    key="cfg_refresh_interval_s_ui",
                    disabled=(not fitting_running) or (not auto_refresh),
                    help="自动刷新间隔 [s]",
                    label_visibility="collapsed",
                    container=col_interval_input,
                )
            )
            clear_btn = col_act4.button(
                "🧹 清除结果",
                type="secondary",
                disabled=fitting_running,
                use_container_width=True,
                help="清除上一次拟合的结果、对比表缓存与时间线（不影响当前输入配置）。",
            )
        st.session_state["fitting_auto_refresh"] = bool(auto_refresh)
        st.session_state["fitting_refresh_interval_s"] = float(refresh_interval_s)

        # --- 显示拟合相关的通知（在 tab 内部显示，避免 tabs 状态重置）---
        fit_notice = st.session_state.pop("fit_notice", None)
        if isinstance(fit_notice, dict):
            notice_kind = str(fit_notice.get("kind", "")).strip().lower()
            notice_text = str(fit_notice.get("text", "")).strip()
            if notice_text:
                if notice_kind == "success":
                    st.success(notice_text)
                elif notice_kind == "warning":
                    st.warning(notice_text)
                elif notice_kind == "error":
                    st.error(notice_text)

        if clear_btn and (not fitting_running):
            for key in [
                "fit_results",
                "fit_compare_cache_key",
                "fit_compare_long_df",
                "fit_compare_long_df_all",
                "fitting_timeline",
                "fitting_metrics",
                "fitting_ms_summary",
                "fitting_final_summary",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # --- 处理“开始拟合”请求（回调触发）---
        if bool(st.session_state.pop("start_fit_requested", False)) and (
            not fitting_future
        ):
            if data_df is None:
                st.session_state["fitting_running"] = False
                st.session_state["fit_notice"] = {
                    "kind": "error",
                    "text": "当前没有可用的 CSV 数据，请先在「实验数据」页面上传或恢复已缓存的文件。",
                }
            elif not output_species_list:
                st.session_state["fitting_running"] = False
                st.session_state["fit_notice"] = {
                    "kind": "error",
                    "text": "请选择至少一个目标物种。",
                }
            else:
                # 每次拟合都使用新的线程池（避免上次任务残留导致排队/卡住）
                old_executor = st.session_state.get("fitting_executor", None)
                if old_executor is not None:
                    try:
                        old_executor.shutdown(wait=False, cancel_futures=True)
                    except Exception as exc:
                        _warn_once(
                            "warn_executor_shutdown",
                            f"关闭旧的拟合线程池失败（可忽略）：{exc}",
                        )
                    st.session_state["fitting_executor"] = None

                st.session_state["fitting_stopped"] = False
                st.session_state["fitting_progress"] = 0.0
                st.session_state["fitting_status"] = "准备启动..."

                stop_event = threading.Event()
                st.session_state["fitting_stop_event"] = stop_event

                progress_queue = queue.Queue()
                st.session_state["fitting_progress_queue"] = progress_queue

                _reset_fitting_progress_ui_state()
                n_fit_params = _count_fitted_parameters(
                    fit_k0_flags,
                    fit_ea_flags,
                    fit_order_flags_matrix,
                    fit_K0_ads_flags,
                    fit_Ea_K_flags,
                    fit_m_flags,
                    fit_k0_rev_flags,
                    fit_ea_rev_flags,
                    fit_order_rev_flags_matrix,
                )
                st.session_state["fitting_job_summary"] = {
                    "title": "拟合任务概览",
                    "lines": [
                        f"数据点数量: {int(len(data_df))} 行",
                        f"待拟合参数: {int(n_fit_params)} 个",
                        f"反应器类型: {ui_text.map_label(ui_text.REACTOR_TYPE_LABELS, str(reactor_type))}",
                        f"动力学模型: {ui_text.map_label(ui_text.KINETIC_MODEL_LABELS, str(kinetic_model))}",
                        f"残差类型: {residual_type}",
                        "优化算法: Trust Region Reflective (trf)",
                        f"最大函数评估次数: {int(max_nfev)}",
                        (
                            f"多起点拟合: {int(n_starts)} 个起点"
                            if (use_ms and int(n_starts) > 1)
                            else "多起点拟合: 关闭"
                        ),
                    ],
                }

                job_inputs = {
                    "data_df": data_df,
                    "species_names": species_names,
                    "output_mode": output_mode,
                    "output_species_list": output_species_list,
                    "stoich_matrix": stoich_matrix,
                    "k0_guess": k0_guess,
                    "ea_guess_J_mol": ea_guess_J_mol,
                    "order_guess": order_guess,
                    "fit_k0_flags": fit_k0_flags,
                    "fit_ea_flags": fit_ea_flags,
                    "fit_order_flags_matrix": fit_order_flags_matrix,
                    "K0_ads": K0_ads,
                    "Ea_K_J_mol": Ea_K_J_mol,
                    "m_inhibition": m_inhibition,
                    "fit_K0_ads_flags": fit_K0_ads_flags,
                    "fit_Ea_K_flags": fit_Ea_K_flags,
                    "fit_m_flags": fit_m_flags,
                    "k0_rev": k0_rev,
                    "ea_rev_J_mol": ea_rev_J_mol,
                    "order_rev": order_rev,
                    "fit_k0_rev_flags": fit_k0_rev_flags,
                    "fit_ea_rev_flags": fit_ea_rev_flags,
                    "fit_order_rev_flags_matrix": fit_order_rev_flags_matrix,
                    "solver_method": solver_method,
                    "rtol": rtol,
                    "atol": atol,
                    "reactor_type": reactor_type,
                    "kinetic_model": kinetic_model,
                    "pfr_flow_model": str(pfr_flow_model),
                    "use_ms": use_ms,
                    "n_starts": n_starts,
                    "random_seed": random_seed,
                    "max_nfev": max_nfev,
                    "max_nfev_coarse": max_nfev_coarse,
                    "diff_step_rel": diff_step_rel,
                    "use_x_scale_jac": use_x_scale_jac,
                    "k0_min": k0_min,
                    "k0_max": k0_max,
                    "ea_min": ea_min,
                    "ea_max": ea_max,
                    "ord_min": ord_min,
                    "ord_max": ord_max,
                    "k0_rev_min": float(k0_rev_min),
                    "k0_rev_max": float(k0_rev_max),
                    "ea_rev_min_J_mol": float(ea_rev_min_J_mol),
                    "ea_rev_max_J_mol": float(ea_rev_max_J_mol),
                    "order_rev_min": float(order_rev_min),
                    "order_rev_max": float(order_rev_max),
                    "K0_ads_min": float(K0_ads_min),
                    "K0_ads_max": float(K0_ads_max),
                    "Ea_K_min": float(Ea_K_min),
                    "Ea_K_max": float(Ea_K_max),
                    "max_step_fraction": float(max_step_fraction),
                    "residual_type": str(residual_type),
                }

                executor = _get_fitting_executor()
                st.session_state["fitting_future"] = executor.submit(
                    _run_fitting_job, job_inputs, stop_event, progress_queue
                )

        if fitting_running:
            st.caption("“自动刷新”：仅刷新进度区域（避免整页闪烁）；如需降低页面刷新负载可关闭。")
            refresh_interval_s = float(
                st.session_state.get("fitting_refresh_interval_s", 2.0)
            )
            _render_fitting_live_progress()
            if bool(st.session_state.get("fitting_auto_refresh", True)):
                # 稳定性优先：避免 st.fragment(run_every=...) 在连接关闭瞬间留下异步写任务。
                # 这里改为常规轮询刷新（整页 rerun），代价是页面刷新频率更高。
                refresh_interval_s = float(np.clip(refresh_interval_s, 0.2, 30.0))
                time.sleep(refresh_interval_s)
                st.rerun()
        elif st.session_state.get("fitting_timeline", []):
            _render_fitting_progress_panel()

        # 在拟合页底部创建结果容器
        tab_fit_results_container = st.container()

    # --- 结果展示（优化版）---
    if "fit_results" in st.session_state:
        res = st.session_state["fit_results"]
        tab_fit_results_container.divider()
        phi_value = float(res.get("phi_final", res.get("cost", 0.0)))
        phi_text = ui_comp.smart_float_to_str(phi_value)
        tab_fit_results_container.markdown(f"### 拟合结果 (目标函数 Φ: {phi_text})")
        tab_fit_results_container.latex(
            r"\Phi(\theta)=\frac{1}{2}\sum_{i=1}^{N} r_i(\theta)^2,\quad r_i=y_i^{\mathrm{pred}}-y_i^{\mathrm{meas}}"
        )

        fitted_params = res["params"]
        df_fit = res["data"]
        species_names_fit = res.get("species_names", species_names)
        stoich_matrix_fit = res.get("stoich_matrix", stoich_matrix)
        solver_method_fit = res.get("solver_method", solver_method)
        rtol_fit = float(res.get("rtol", rtol))
        atol_fit = float(res.get("atol", atol))
        max_step_fraction_fit = float(
            res.get(
                "max_step_fraction",
                get_cfg("max_step_fraction", DEFAULT_MAX_STEP_FRACTION),
            )
        )
        reactor_type_fit = res.get("reactor_type", reactor_type)
        kinetic_model_fit = res.get("kinetic_model", kinetic_model)
        output_mode_fit = res.get("output_mode", output_mode)
        # 奇偶校验图的候选物种会在 tab_parity 中根据“验证量（浓度/转化率）”动态判定
        parity_species_candidates = list(species_names_fit)
        parity_species_unavailable = []

        tab_param, tab_parity, tab_profile, tab_export = tab_fit_results_container.tabs(
            ["参数", "奇偶校验图", "沿程/随时间剖面", "导出"]
        )

        with tab_param:
            st.markdown("#### 拟合参数")
            col_p1, col_p2 = st.columns([1, 1])

            with col_p1:
                reaction_names = [f"R{i+1}" for i in range(len(fitted_params["k0"]))]
                df_k0_ea = pd.DataFrame(
                    {
                        "k₀ [SI]": fitted_params["k0"],
                        "Eₐ [J/mol]": fitted_params["ea_J_mol"],
                    },
                    index=reaction_names,
                )
                st.markdown("**k₀ 与 Eₐ**")
                st.dataframe(
                    ui_comp.format_dataframe_for_display(df_k0_ea),
                    use_container_width=True,
                    height=UI_PARAM_TABLE_HEIGHT_PX,
                )

            with col_p2:
                st.markdown("**反应级数矩阵 $n$**")
                df_orders = pd.DataFrame(
                    fitted_params["reaction_order_matrix"],
                    index=reaction_names,
                    columns=species_names_fit,
                )
                st.dataframe(
                    ui_comp.format_dataframe_for_display(df_orders),
                    use_container_width=True,
                    height=UI_PARAM_TABLE_HEIGHT_PX,
                )

            if kinetic_model_fit == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
                st.markdown("#### Langmuir-Hinshelwood 参数")
                col_lh1, col_lh2 = st.columns([1, 1])
                with col_lh1:
                    if (
                        fitted_params.get("K0_ads", None) is not None
                        and fitted_params.get("Ea_K", None) is not None
                    ):
                        df_ads = pd.DataFrame(
                            {
                                "K₀,ads [1/(mol/m^3)]": fitted_params["K0_ads"],
                                "Eₐ,K [J/mol]": fitted_params["Ea_K"],
                            },
                            index=species_names_fit,
                        )
                        st.dataframe(
                            ui_comp.format_dataframe_for_display(df_ads),
                            use_container_width=True,
                            height=UI_PARAM_TABLE_HEIGHT_PX,
                        )
                with col_lh2:
                    if fitted_params.get("m_inhibition", None) is not None:
                        df_m = pd.DataFrame(
                            {"m_inhibition [-]": fitted_params["m_inhibition"]},
                            index=reaction_names,
                        )
                        st.dataframe(
                            ui_comp.format_dataframe_for_display(df_m),
                            use_container_width=True,
                            height=UI_PARAM_TABLE_HEIGHT_PX,
                        )

            if kinetic_model_fit == KINETIC_MODEL_REVERSIBLE:
                st.markdown("#### 可逆反应参数（逆反应）")
                if (
                    fitted_params.get("k0_rev", None) is not None
                    and fitted_params.get("ea_rev", None) is not None
                ):
                    df_rev = pd.DataFrame(
                        {
                            "k₀,rev [SI]": fitted_params["k0_rev"],
                            "Eₐ,rev [J/mol]": fitted_params["ea_rev"],
                        },
                        index=reaction_names,
                    )
                    st.dataframe(
                        ui_comp.format_dataframe_for_display(df_rev),
                        use_container_width=True,
                        height=UI_PARAM_TABLE_HEIGHT_PX,
                    )
                if fitted_params.get("order_rev", None) is not None:
                    st.markdown("**逆反应级数矩阵 $n^-$**")
                    df_order_rev = pd.DataFrame(
                        fitted_params["order_rev"],
                        index=reaction_names,
                        columns=species_names_fit,
                    )
                    st.dataframe(
                        ui_comp.format_dataframe_for_display(df_order_rev),
                        use_container_width=True,
                        height=UI_PARAM_TABLE_HEIGHT_PX,
                    )

        with tab_parity:
            st.markdown("#### 分物种奇偶校验图（实验值 vs 预测值）")
            output_mode_fit_str = str(output_mode_fit).strip()
            output_label_map = {
                OUTPUT_MODE_COUT: "出口浓度 (Cout)",
                OUTPUT_MODE_FOUT: "出口摩尔流率 (Fout)",
                OUTPUT_MODE_XOUT: "出口摩尔分率 (xout)",
            }
            default_output_label = output_label_map.get(
                output_mode_fit_str, f"输出量（{output_mode_fit_str}）"
            )

            parity_validation_choice = st.radio(
                "验证量",
                [default_output_label, "转化率 (X)"],
                index=0,
                horizontal=True,
                key="parity_validation_choice",
                help="选择奇偶校验图的对比量：当前输出量（与拟合输出模式一致）或转化率。转化率优先按摩尔流率计算（BSTR 无摩尔流率则用浓度）。",
            )

            pfr_flow_model_fit = str(
                res.get("pfr_flow_model", PFR_FLOW_MODEL_LIQUID_CONST_VDOT)
            ).strip()

            # 根据验证量动态确定：对比方式、单位、以及可用物种
            if parity_validation_choice == default_output_label:
                compare_output_mode = output_mode_fit_str
                compare_validation_mode = "output"
                unit_text_parity = _get_output_unit_text(compare_output_mode)
            else:
                compare_output_mode = output_mode_fit_str  # 占位：conversion 模式下不会使用测量列名映射
                compare_validation_mode = "conversion"
                unit_text_parity = "-"

            parity_species_candidates = []
            parity_species_unavailable = []
            df_cols = set(map(str, df_fit.columns))

            for sp_name in species_names_fit:
                if compare_validation_mode == "output":
                    meas_col = _get_measurement_column_name(compare_output_mode, sp_name)
                    if meas_col not in df_cols:
                        parity_species_unavailable.append(f"{sp_name}（缺少列 {meas_col}）")
                        continue
                    numeric_series = pd.to_numeric(df_fit[meas_col], errors="coerce")
                    if bool(np.any(np.isfinite(numeric_series.to_numpy()))):
                        parity_species_candidates.append(sp_name)
                    else:
                        parity_species_unavailable.append(
                            f"{sp_name}（列 {meas_col} 全为 NaN/非数字）"
                        )
                    continue

                # --- conversion 模式：按反应器/流动模型检查必要列 ---
                if reactor_type_fit == REACTOR_TYPE_BSTR:
                    required_cols = [
                        f"C0_{sp_name}_mol_m3",
                        f"Cout_{sp_name}_mol_m3",
                    ]
                    missing = [c for c in required_cols if c not in df_cols]
                    if missing:
                        parity_species_unavailable.append(
                            f"{sp_name}（缺少列: {', '.join(missing)}）"
                        )
                        continue
                    series_list = [
                        pd.to_numeric(df_fit[required_cols[0]], errors="coerce"),
                        pd.to_numeric(df_fit[required_cols[1]], errors="coerce"),
                    ]
                    if any(bool(np.any(np.isfinite(s.to_numpy()))) for s in series_list):
                        parity_species_candidates.append(sp_name)
                    else:
                        parity_species_unavailable.append(
                            f"{sp_name}（C0/Cout 全为 NaN/非数字）"
                        )
                    continue

                if (
                    reactor_type_fit == REACTOR_TYPE_PFR
                    and str(pfr_flow_model_fit) == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P
                ):
                    required_cols = [
                        f"F0_{sp_name}_mol_s",
                        f"Fout_{sp_name}_mol_s",
                    ]
                    missing = [c for c in required_cols if c not in df_cols]
                    if missing:
                        parity_species_unavailable.append(
                            f"{sp_name}（气相 PFR 转化率需要列: {', '.join(missing)}）"
                        )
                        continue
                    series_list = [
                        pd.to_numeric(df_fit[required_cols[0]], errors="coerce"),
                        pd.to_numeric(df_fit[required_cols[1]], errors="coerce"),
                    ]
                    if any(bool(np.any(np.isfinite(s.to_numpy()))) for s in series_list):
                        parity_species_candidates.append(sp_name)
                    else:
                        parity_species_unavailable.append(
                            f"{sp_name}（F0/Fout 全为 NaN/非数字）"
                        )
                    continue

                # 其他（液相 PFR / CSTR）：允许 F0/Fout；若缺则用 C0/Cout + vdot 换算
                need_vdot = "vdot_m3_s" in df_cols
                has_inlet = (
                    (f"F0_{sp_name}_mol_s" in df_cols)
                    or (need_vdot and (f"C0_{sp_name}_mol_m3" in df_cols))
                )
                has_outlet = (
                    (f"Fout_{sp_name}_mol_s" in df_cols)
                    or (need_vdot and (f"Cout_{sp_name}_mol_m3" in df_cols))
                )
                if not has_inlet or not has_outlet:
                    parts = []
                    if not has_inlet:
                        parts.append("入口缺少 F0 或 C0+vdot")
                    if not has_outlet:
                        parts.append("出口缺少 Fout 或 Cout+vdot")
                    parity_species_unavailable.append(f"{sp_name}（{'；'.join(parts)}）")
                    continue

                inlet_col = (
                    f"F0_{sp_name}_mol_s"
                    if f"F0_{sp_name}_mol_s" in df_cols
                    else f"C0_{sp_name}_mol_m3"
                )
                outlet_col = (
                    f"Fout_{sp_name}_mol_s"
                    if f"Fout_{sp_name}_mol_s" in df_cols
                    else f"Cout_{sp_name}_mol_m3"
                )
                numeric_in = pd.to_numeric(df_fit[inlet_col], errors="coerce")
                numeric_out = pd.to_numeric(df_fit[outlet_col], errors="coerce")
                if bool(np.any(np.isfinite(numeric_in.to_numpy()))) and bool(
                    np.any(np.isfinite(numeric_out.to_numpy()))
                ):
                    parity_species_candidates.append(sp_name)
                else:
                    parity_species_unavailable.append(
                        f"{sp_name}（入口/出口列全为 NaN/非数字）"
                    )

            if parity_species_unavailable:
                show_missing = st.checkbox("显示无法绘图的物种原因", value=False)
                if show_missing:
                    st.caption(
                        "无法绘制奇偶校验图的物种： " + "，".join(parity_species_unavailable)
                    )

            cache_key = (
                float(res.get("phi_final", res.get("cost", 0.0))),
                str(compare_validation_mode),
                str(compare_output_mode),
                tuple(parity_species_candidates),
                float(rtol_fit),
                float(atol_fit),
                str(solver_method_fit),
                str(reactor_type_fit),
                str(pfr_flow_model_fit),
                str(kinetic_model_fit),
                float(max_step_fraction_fit),
            )
            if (
                st.session_state.get("fit_compare_cache_key", None) != cache_key
                or "fit_compare_long_df" not in st.session_state
            ):
                try:
                    st.session_state["fit_compare_cache_key"] = cache_key
                    st.session_state["fit_compare_long_df"] = (
                        _build_fit_comparison_long_table(
                            data_df=df_fit,
                            species_names=species_names_fit,
                            output_mode=str(compare_output_mode),
                            output_species_list=parity_species_candidates,
                            stoich_matrix=stoich_matrix_fit,
                            fitted_params=fitted_params,
                            solver_method=solver_method_fit,
                            rtol=float(rtol_fit),
                            atol=float(atol_fit),
                            reactor_type=reactor_type_fit,
                            kinetic_model=kinetic_model_fit,
                            pfr_flow_model=str(pfr_flow_model_fit),
                            max_step_fraction=float(max_step_fraction_fit),
                            validation_mode=str(compare_validation_mode),
                        )
                    )
                except Exception as exc:
                    st.error(f"生成对比数据失败: {exc}")
                    st.session_state["fit_compare_long_df"] = pd.DataFrame()

            df_long = st.session_state["fit_compare_long_df"]
            if df_long.empty:
                st.warning("对比数据为空：无法生成奇偶校验图。")
            else:
                # --- 布局：左侧选择数据/子图布局，右侧绘图附加选项 ---
                col_sel, col_opt = st.columns([1.35, 1.0])
                with col_sel:
                    species_selected = st.multiselect(
                        "选择要显示的物种",
                        list(parity_species_candidates),
                        default=list(parity_species_candidates),
                        help="仅对当前奇偶校验图与残差图生效。",
                    )
                    n_cols = int(
                        st.number_input(
                            "每行子图数",
                            min_value=1,
                            max_value=4,
                            value=2,
                            step=1,
                            help="仅影响子图排版；不改变拟合/预测结果。",
                        )
                    )

                with col_opt:
                    show_residual_plot = st.checkbox("显示残差图", value=True)
                    show_error_lines = st.checkbox("显示±误差线", value=True)
                    error_band_percent = float(
                        st.slider(
                            "相对误差带（%）",
                            min_value=0.0,
                            max_value=50.0,
                            value=10.0,
                            step=0.5,
                            key="parity_error_band_percent",
                            help="在 y=x 两侧绘制 y=(1±e)x 参考线，用于直观判断预测偏差范围。",
                            disabled=(not show_error_lines),
                        )
                    )

                st.divider()

                df_ok = df_long[df_long["ok"]].copy()
                df_ok = df_ok[
                    np.isfinite(df_ok["measured"]) & np.isfinite(df_ok["predicted"])
                ]
                if df_ok.empty:
                    st.error(
                        "所有实验点都无法成功预测（solve_ivp 失败或输入不合法）。\n"
                        "建议：尝试把求解器切换为 `BDF` 或 `Radau`，并适当放宽 `rtol/atol`。"
                    )
                else:
                    df_ok = df_ok[df_ok["species"].isin(species_selected)]
                    if df_ok.empty:
                        st.warning("所选物种没有可用数据点。")
                    else:
                        vals_all = np.concatenate(
                            [
                                df_ok["measured"].to_numpy(dtype=float),
                                df_ok["predicted"].to_numpy(dtype=float),
                            ]
                        )
                        vmin_auto = float(np.nanmin(vals_all))
                        vmax_auto = float(np.nanmax(vals_all))
                        if (not np.isfinite(vmin_auto)) or (not np.isfinite(vmax_auto)):
                            vmin_auto, vmax_auto = 0.0, 1.0
                        if vmax_auto <= vmin_auto:
                            vmax_auto = vmin_auto + 1.0
                        pad = 0.05 * float(vmax_auto - vmin_auto)
                        axis_min_auto = float(vmin_auto - pad)
                        axis_max_auto = float(vmax_auto + pad)

                        species_list_plot = list(
                            dict.fromkeys(df_ok["species"].tolist())
                        )
                        n_plots = len(species_list_plot)
                        n_rows = int(np.ceil(n_plots / max(n_cols, 1)))

                        axis_ranges_by_species = None
                        with st.expander("坐标范围设置（横纵一致 + 等比例）", expanded=False):
                            st.caption(
                                "默认强制 x/y 等比例，以避免因为坐标拉伸导致对拟合优劣的误判。"
                            )
                            axis_scope = st.radio(
                                "坐标范围作用域",
                                ["所有子图一致（推荐）", "每个子图独立"],
                                index=0,
                                horizontal=True,
                                key="parity_axis_scope",
                                help="所有子图一致：便于不同物种之间直接比较拟合质量；每个子图独立：可单独放大细节，但不同子图的点云“紧密程度”不可直接横向比较。",
                            )
                            axis_range_mode = st.radio(
                                "范围来源",
                                ["自动（按数据）", "自定义"],
                                index=0,
                                horizontal=True,
                                key="parity_axis_range_mode",
                                help="自动：按数据最小/最大值（含 5% padding）确定坐标范围；自定义：手动输入 min/max。若选择“每个子图独立”，则可分别为每个子图设置 min/max。",
                            )
                            st.caption(
                                f"全局自动范围（用于统一坐标时的默认值）：[{axis_min_auto:.6g}, {axis_max_auto:.6g}]"
                            )

                            if axis_scope == "所有子图一致（推荐）":
                                if axis_range_mode == "自定义":
                                    col_ax1, col_ax2 = st.columns([1, 1])
                                    axis_min_user = float(
                                        col_ax1.number_input(
                                            "坐标最小值",
                                            value=float(axis_min_auto),
                                            key="parity_axis_min",
                                        )
                                    )
                                    axis_max_user = float(
                                        col_ax2.number_input(
                                            "坐标最大值",
                                            value=float(axis_max_auto),
                                            key="parity_axis_max",
                                        )
                                    )
                                    if axis_max_user <= axis_min_user:
                                        st.warning(
                                            "坐标范围无效：需要满足 max > min。将回退到自动范围。"
                                        )
                                        axis_min_plot, axis_max_plot = (
                                            axis_min_auto,
                                            axis_max_auto,
                                        )
                                    else:
                                        axis_min_plot, axis_max_plot = (
                                            axis_min_user,
                                            axis_max_user,
                                        )
                                else:
                                    axis_min_plot, axis_max_plot = (
                                        axis_min_auto,
                                        axis_max_auto,
                                    )
                            else:
                                # 每个子图独立：先计算每个物种的自动范围；如选择自定义，则逐图覆盖
                                auto_ranges = {}
                                for species_name in species_list_plot:
                                    df_sp = df_ok[df_ok["species"] == species_name]
                                    vals_sp = np.concatenate(
                                        [
                                            df_sp["measured"].to_numpy(dtype=float),
                                            df_sp["predicted"].to_numpy(dtype=float),
                                        ]
                                    )
                                    vmin_sp = float(np.nanmin(vals_sp))
                                    vmax_sp = float(np.nanmax(vals_sp))
                                    if (not np.isfinite(vmin_sp)) or (
                                        not np.isfinite(vmax_sp)
                                    ):
                                        vmin_sp, vmax_sp = 0.0, 1.0
                                    if vmax_sp <= vmin_sp:
                                        vmax_sp = vmin_sp + 1.0
                                    pad_sp = 0.05 * float(vmax_sp - vmin_sp)
                                    auto_ranges[species_name] = (
                                        float(vmin_sp - pad_sp),
                                        float(vmax_sp + pad_sp),
                                    )

                                axis_ranges_by_species = dict(auto_ranges)
                                if axis_range_mode == "自定义":
                                    st.markdown("**逐图自定义**")
                                    st.caption(
                                        "每个子图的 x/y 使用相同 min/max，并保持等比例；若输入无效（max ≤ min），该子图会回退到自动范围。"
                                    )
                                    h1, h2, h3 = st.columns([1.2, 1, 1])
                                    h1.markdown("**物种**")
                                    h2.markdown("**min**")
                                    h3.markdown("**max**")
                                    invalid_species = []
                                    for idx, species_name in enumerate(species_list_plot):
                                        c1, c2, c3 = st.columns([1.2, 1, 1])
                                        c1.write(species_name)
                                        auto_min, auto_max = auto_ranges[species_name]
                                        key_hash = hashlib.md5(
                                            str(species_name).encode("utf-8")
                                        ).hexdigest()[:12]
                                        axis_min_user = float(
                                            c2.number_input(
                                                "min",
                                                value=float(auto_min),
                                                key=f"parity_axis_min_{idx}_{key_hash}",
                                                label_visibility="collapsed",
                                            )
                                        )
                                        axis_max_user = float(
                                            c3.number_input(
                                                "max",
                                                value=float(auto_max),
                                                key=f"parity_axis_max_{idx}_{key_hash}",
                                                label_visibility="collapsed",
                                            )
                                        )
                                        if axis_max_user <= axis_min_user:
                                            invalid_species.append(species_name)
                                            axis_ranges_by_species[species_name] = (
                                                float(auto_min),
                                                float(auto_max),
                                            )
                                        else:
                                            axis_ranges_by_species[species_name] = (
                                                float(axis_min_user),
                                                float(axis_max_user),
                                            )
                                    if invalid_species:
                                        st.warning(
                                            "以下物种的坐标范围无效（max ≤ min），已回退到自动范围："
                                            + "，".join(map(str, invalid_species))
                                        )
                                else:
                                    # 自动范围：axis_ranges_by_species 已包含逐物种自动范围
                                    pass

                        fig, axes = plt.subplots(
                            n_rows,
                            n_cols,
                            figsize=(5.2 * n_cols, 4.3 * n_rows),
                            squeeze=False,
                        )

                        for i, species_name in enumerate(species_list_plot):
                            ax = axes[i // n_cols][i % n_cols]
                            df_sp = df_ok[df_ok["species"] == species_name]
                            series_color = _fit_plot_color(i)
                            ax.scatter(
                                df_sp["measured"].to_numpy(dtype=float),
                                df_sp["predicted"].to_numpy(dtype=float),
                                s=44,
                                alpha=0.9,
                                facecolors=series_color,
                                edgecolors="#ffffff",
                                linewidths=0.9,
                                label=species_name,
                                zorder=3,
                            )
                            min_v = float(
                                np.nanmin(
                                    np.concatenate(
                                        [
                                            df_sp["measured"].to_numpy(),
                                            df_sp["predicted"].to_numpy(),
                                        ]
                                    )
                                )
                            )
                            max_v = float(
                                np.nanmax(
                                    np.concatenate(
                                        [
                                            df_sp["measured"].to_numpy(),
                                            df_sp["predicted"].to_numpy(),
                                        ]
                                    )
                                )
                            )
                            # x/y 坐标范围 + 等比例（可全局统一，也可逐图独立）
                            if axis_ranges_by_species is None:
                                axis_min_i, axis_max_i = axis_min_plot, axis_max_plot
                            else:
                                axis_min_i, axis_max_i = axis_ranges_by_species.get(
                                    species_name,
                                    (axis_min_auto, axis_max_auto),
                                )
                            ax.set_xlim(axis_min_i, axis_max_i)
                            ax.set_ylim(axis_min_i, axis_max_i)
                            ax.set_aspect("equal", adjustable="box")

                            if (
                                np.isfinite(min_v)
                                and np.isfinite(max_v)
                                and max_v > min_v
                            ):
                                ax.plot(
                                    [axis_min_i, axis_max_i],
                                    [axis_min_i, axis_max_i],
                                    color="#000000",
                                    linestyle="--",
                                    linewidth=1.2,
                                    label="Ideal y = x",
                                )
                                if show_error_lines and (error_band_percent > 0.0):
                                    e = float(error_band_percent) / 100.0
                                    error_label = f"± {error_band_percent:.1f}% band"
                                    ax.plot(
                                        [axis_min_i, axis_max_i],
                                        [
                                            (1.0 - e) * axis_min_i,
                                            (1.0 - e) * axis_max_i,
                                        ],
                                        color="tab:gray",
                                        linestyle="--",
                                        linewidth=1.0,
                                        label=error_label,
                                    )
                                    ax.plot(
                                        [axis_min_i, axis_max_i],
                                        [
                                            (1.0 + e) * axis_min_i,
                                            (1.0 + e) * axis_max_i,
                                        ],
                                        color="tab:gray",
                                        linestyle="--",
                                        linewidth=1.0,
                                        label="_nolegend_",
                                    )
                            ax.set_title(f"Species: {species_name}")
                            ax.set_xlabel(
                                ui_text.axis_label_with_unit(
                                    ui_text.AXIS_LABEL_MEASURED, unit_text_parity
                                )
                            )
                            ax.set_ylabel(
                                ui_text.axis_label_with_unit(
                                    ui_text.AXIS_LABEL_PREDICTED, unit_text_parity
                                )
                            )
                            _style_fit_axis(ax, show_grid=False)
                            _style_fit_legend(ax)

                        for j in range(n_plots, n_rows * n_cols):
                            axes[j // n_cols][j % n_cols].axis("off")

                        fig.tight_layout()
                        st.pyplot(fig)

                        image_format = st.selectbox(
                            "图像格式",
                            ["png", "svg"],
                            index=0,
                            key="parity_image_format",
                        )
                        st.download_button(
                            "📥 下载奇偶校验图",
                            ui_comp.figure_to_image_bytes(fig, image_format),
                            file_name=f"parity_plot.{image_format}",
                            mime=(
                                "image/png"
                                if image_format == "png"
                                else "image/svg+xml"
                            ),
                        )
                        plt.close(fig)

                if show_residual_plot:
                    st.markdown("#### 残差图（预测值 - 实验值）")
                    df_res = df_long[df_long["ok"]].copy()
                    df_res = df_res[df_res["species"].isin(species_selected)]
                    df_res = df_res[
                        np.isfinite(df_res["residual"]) & np.isfinite(df_res["measured"])
                    ]
                    if df_res.empty:
                        st.warning("所选物种没有可用残差数据。")
                    else:
                        species_list_residual = [
                            sp for sp in species_selected if sp in set(df_res["species"])
                        ]
                        n_residual_plots = len(species_list_residual)
                        n_residual_rows = int(
                            np.ceil(n_residual_plots / max(int(n_cols), 1))
                        )

                        fig_r, axes_r = plt.subplots(
                            n_residual_rows,
                            n_cols,
                            figsize=(5.2 * n_cols, 4.0 * n_residual_rows),
                            squeeze=False,
                        )

                        for i, species_name in enumerate(species_list_residual):
                            ax_r = axes_r[i // n_cols][i % n_cols]
                            df_sp = df_res[df_res["species"] == species_name]
                            series_color = _fit_plot_color(i)
                            ax_r.scatter(
                                df_sp["measured"].to_numpy(dtype=float),
                                df_sp["residual"].to_numpy(dtype=float),
                                s=42,
                                alpha=0.9,
                                facecolors=series_color,
                                edgecolors="#ffffff",
                                linewidths=0.9,
                                label=species_name,
                                zorder=3,
                            )
                            ax_r.axhline(
                                0.0,
                                color="#000000",
                                linestyle="--",
                                linewidth=1.2,
                                label="Zero residual",
                            )
                            ax_r.set_title(f"Species: {species_name}")
                            ax_r.set_xlabel(
                                ui_text.axis_label_with_unit(
                                    ui_text.AXIS_LABEL_MEASURED, unit_text_parity
                                )
                            )
                            ax_r.set_ylabel(
                                ui_text.axis_label_with_unit(
                                    ui_text.AXIS_LABEL_RESIDUAL, unit_text_parity
                                )
                            )
                            _style_fit_axis(ax_r, show_grid=False)
                            _style_fit_legend(ax_r)

                        for j in range(n_residual_plots, n_residual_rows * n_cols):
                            axes_r[j // n_cols][j % n_cols].axis("off")

                        fig_r.tight_layout()
                        st.pyplot(fig_r)
                        residual_image_format = st.selectbox(
                            "残差图像格式",
                            ["png", "svg"],
                            index=0,
                            key="residual_image_format",
                        )
                        st.download_button(
                            "📥 下载残差图",
                            ui_comp.figure_to_image_bytes(fig_r, residual_image_format),
                            file_name=f"residual_plot.{residual_image_format}",
                            mime=(
                                "image/png"
                                if residual_image_format == "png"
                                else "image/svg+xml"
                            ),
                        )
                        plt.close(fig_r)

                show_compare_table = st.checkbox("显示预测 vs 实验对比表", value=False)
                if show_compare_table:
                    st.markdown("#### 预测 vs 实验对比表（含相对残差）")
                    df_show = df_long.copy()
                    df_show = df_show[df_show["species"].isin(species_selected)]
                    # 按用户需求：不显示 ok/message；新增 relative_residual（在构表阶段已计算）
                    drop_cols = [c for c in ["ok", "message"] if c in df_show.columns]
                    if drop_cols:
                        df_show = df_show.drop(columns=drop_cols)

                    preferred_order = [
                        "row_index",
                        "species",
                        "measured",
                        "predicted",
                        "residual",
                        "relative_residual",
                    ]
                    existing_preferred = [c for c in preferred_order if c in df_show.columns]
                    remaining_cols = [c for c in df_show.columns if c not in existing_preferred]
                    df_show = df_show[existing_preferred + remaining_cols]
                    st.dataframe(
                        df_show,
                        use_container_width=True,
                        height=UI_COMPARE_TABLE_HEIGHT_PX,
                    )

                st.markdown("#### 拟合误差指标（按物种）")
                rows_metric = []
                for species_name in species_selected:
                    df_sp = df_long[
                        (df_long["species"] == species_name) & (df_long["ok"])
                    ].copy()
                    df_sp = df_sp[
                        np.isfinite(df_sp["measured"]) & np.isfinite(df_sp["predicted"])
                    ]
                    if df_sp.empty:
                        continue
                    resid = df_sp["predicted"].to_numpy(dtype=float) - df_sp[
                        "measured"
                    ].to_numpy(dtype=float)
                    rmse = float(np.sqrt(np.mean(resid**2)))
                    mae = float(np.mean(np.abs(resid)))
                    rows_metric.append(
                        {
                            "species": species_name,
                            "N": int(df_sp.shape[0]),
                            "RMSE": rmse,
                            "MAE": mae,
                        }
                    )
                if rows_metric:
                    st.dataframe(
                        pd.DataFrame(rows_metric),
                        use_container_width=True,
                        height=UI_METRICS_TABLE_HEIGHT_PX,
                    )

        with tab_profile:
            st.markdown("#### 沿程/随时间剖面")
            st.caption("说明：本页剖面为模型**预测**数据（不是实验测量值）。")
            if df_fit.empty:
                st.warning("数据为空：无法生成剖面。")
            else:
                row_indices = df_fit.index.tolist()
                selected_row_index = st.selectbox(
                    "选择一个实验点（按 DataFrame index）",
                    row_indices,
                    index=0,
                )
                profile_points = int(
                    st.number_input(
                        "剖面点数",
                        min_value=UI_PROFILE_POINTS_MIN,
                        max_value=UI_PROFILE_POINTS_MAX,
                        value=UI_PROFILE_POINTS_DEFAULT,
                        step=UI_PROFILE_POINTS_STEP,
                    )
                )
                profile_species = st.multiselect(
                    "选择要画剖面的物种（可多选）",
                    list(species_names_fit),
                    default=list(species_names_fit[: min(3, len(species_names_fit))]),
                )

                row_sel = df_fit.loc[selected_row_index]
                if reactor_type_fit == REACTOR_TYPE_PFR:
                    profile_kind_options = ["F (mol/s)", "C (mol/m^3)"]
                    profile_kind = st.radio(
                        "剖面变量",
                        profile_kind_options,
                        index=0,
                        horizontal=True,
                        format_func=lambda x: ui_text.map_label(
                            ui_text.PROFILE_KIND_LABELS, str(x)
                        ),
                    )
                    reactor_volume_m3 = float(row_sel.get("V_m3", np.nan))
                    temperature_K = float(row_sel.get("T_K", np.nan))
                    pfr_flow_model_fit = str(
                        res.get("pfr_flow_model", PFR_FLOW_MODEL_LIQUID_CONST_VDOT)
                    ).strip()

                    molar_flow_inlet = np.zeros(len(species_names_fit), dtype=float)
                    if pfr_flow_model_fit == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
                        # 气相：入口强制用 F0_*
                        pressure_Pa = float(row_sel.get("P_Pa", np.nan))
                        for i, sp_name in enumerate(species_names_fit):
                            molar_flow_inlet[i] = float(
                                row_sel.get(f"F0_{sp_name}_mol_s", np.nan)
                            )

                        volume_grid_m3, molar_flow_profile, ok, message = (
                            reactors.integrate_pfr_profile_gas_ideal_const_p(
                                reactor_volume_m3=reactor_volume_m3,
                                temperature_K=temperature_K,
                                pressure_Pa=pressure_Pa,
                                molar_flow_inlet_mol_s=molar_flow_inlet,
                                stoich_matrix=stoich_matrix_fit,
                                k0=fitted_params["k0"],
                                ea_J_mol=fitted_params["ea_J_mol"],
                                reaction_order_matrix=fitted_params[
                                    "reaction_order_matrix"
                                ],
                                solver_method=solver_method_fit,
                                rtol=rtol_fit,
                                atol=atol_fit,
                                n_points=profile_points,
                                kinetic_model=kinetic_model_fit,
                                max_step_fraction=max_step_fraction_fit,
                                K0_ads=fitted_params.get("K0_ads", None),
                                Ea_K_J_mol=fitted_params.get("Ea_K", None),
                                m_inhibition=fitted_params.get("m_inhibition", None),
                                k0_rev=fitted_params.get("k0_rev", None),
                                ea_rev_J_mol=fitted_params.get("ea_rev", None),
                                order_rev_matrix=fitted_params.get("order_rev", None),
                            )
                        )
                    else:
                        # 液相：vdot 恒定（C=F/vdot）；Cout 拟合时允许入口用 C0_* 并由 vdot 换算
                        vdot_m3_s = float(row_sel.get("vdot_m3_s", np.nan))
                        use_conc_inlet = str(output_mode_fit).strip().startswith("C")
                        for i, sp_name in enumerate(species_names_fit):
                            if use_conc_inlet:
                                c0 = float(row_sel.get(f"C0_{sp_name}_mol_m3", np.nan))
                                molar_flow_inlet[i] = c0 * float(vdot_m3_s)
                            else:
                                molar_flow_inlet[i] = float(
                                    row_sel.get(f"F0_{sp_name}_mol_s", np.nan)
                                )

                        volume_grid_m3, molar_flow_profile, ok, message = (
                            reactors.integrate_pfr_profile(
                                reactor_volume_m3=reactor_volume_m3,
                                temperature_K=temperature_K,
                                vdot_m3_s=vdot_m3_s,
                                molar_flow_inlet_mol_s=molar_flow_inlet,
                                stoich_matrix=stoich_matrix_fit,
                                k0=fitted_params["k0"],
                                ea_J_mol=fitted_params["ea_J_mol"],
                                reaction_order_matrix=fitted_params[
                                    "reaction_order_matrix"
                                ],
                                solver_method=solver_method_fit,
                                rtol=rtol_fit,
                                atol=atol_fit,
                                n_points=profile_points,
                                kinetic_model=kinetic_model_fit,
                                max_step_fraction=max_step_fraction_fit,
                                K0_ads=fitted_params.get("K0_ads", None),
                                Ea_K_J_mol=fitted_params.get("Ea_K", None),
                                m_inhibition=fitted_params.get("m_inhibition", None),
                                k0_rev=fitted_params.get("k0_rev", None),
                                ea_rev_J_mol=fitted_params.get("ea_rev", None),
                                order_rev_matrix=fitted_params.get("order_rev", None),
                            )
                        )
                    if not ok:
                        st.error(
                            f"PFR 剖面计算失败: {message}\n"
                            "建议：尝试将求解器切换为 `BDF` 或 `Radau`，并适当放宽 `rtol/atol`。"
                        )
                    else:
                        fig_pf, ax_pf = plt.subplots(figsize=(7, 4.5))
                        name_to_index = {
                            name: i for i, name in enumerate(species_names_fit)
                        }

                        profile_df = pd.DataFrame({"V_m3": volume_grid_m3})
                        for i, species_name in enumerate(profile_species):
                            idx = name_to_index[species_name]
                            series_color = _fit_plot_color(i)
                            if profile_kind.startswith("F"):
                                y = molar_flow_profile[idx, :]
                                _plot_reference_series(
                                    ax_pf,
                                    volume_grid_m3,
                                    y,
                                    label=species_name,
                                    color=series_color,
                                )
                                profile_df[f"F_{species_name}_mol_s"] = y
                            else:
                                if pfr_flow_model_fit == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
                                    # C_i = y_i · P/(R·T)
                                    pressure_Pa = float(row_sel.get("P_Pa", np.nan))
                                    conc_total = float(pressure_Pa) / max(
                                        float(R_GAS_J_MOL_K) * float(temperature_K),
                                        EPSILON_CONCENTRATION,
                                    )
                                    total_flow = np.sum(molar_flow_profile, axis=0)
                                    conc = (
                                        molar_flow_profile[idx, :]
                                        / np.maximum(total_flow, EPSILON_FLOW_RATE)
                                        * float(conc_total)
                                    )
                                else:
                                    conc = molar_flow_profile[idx, :] / max(
                                        vdot_m3_s, EPSILON_FLOW_RATE
                                    )
                                _plot_reference_series(
                                    ax_pf,
                                    volume_grid_m3,
                                    conc,
                                    label=species_name,
                                    color=series_color,
                                )
                                profile_df[f"C_{species_name}_mol_m3"] = conc

                        ax_pf.set_xlabel(ui_text.AXIS_LABEL_REACTOR_VOLUME)
                        ax_pf.set_ylabel(
                            ui_text.AXIS_LABEL_FLOW_RATE
                            if profile_kind.startswith("F")
                            else ui_text.AXIS_LABEL_CONCENTRATION
                        )
                        _style_fit_axis(ax_pf, show_grid=False)
                        _style_fit_legend(ax_pf)
                        st.pyplot(fig_pf)

                        st.download_button(
                            "📥 下载剖面数据 CSV",
                            profile_df.to_csv(index=False).encode("utf-8"),
                            file_name="profile_data.csv",
                            mime="text/csv",
                        )
                        image_format_pf = st.selectbox(
                            "剖面图格式",
                            ["png", "svg"],
                            index=0,
                            key="profile_image_format",
                        )
                        st.download_button(
                            "📥 下载剖面图",
                            ui_comp.figure_to_image_bytes(fig_pf, image_format_pf),
                            file_name=f"profile_plot.{image_format_pf}",
                            mime=(
                                "image/png"
                                if image_format_pf == "png"
                                else "image/svg+xml"
                            ),
                        )
                        plt.close(fig_pf)

                elif reactor_type_fit == REACTOR_TYPE_CSTR:
                    profile_kind = "C (mol/m^3)"
                    reactor_volume_m3 = float(row_sel.get("V_m3", np.nan))
                    temperature_K = float(row_sel.get("T_K", np.nan))
                    vdot_m3_s = float(row_sel.get("vdot_m3_s", np.nan))

                    conc_inlet = np.zeros(len(species_names_fit), dtype=float)
                    for i, sp_name in enumerate(species_names_fit):
                        conc_inlet[i] = float(
                            row_sel.get(f"C0_{sp_name}_mol_m3", np.nan)
                        )

                    tau_s = reactor_volume_m3 / max(vdot_m3_s, EPSILON_FLOW_RATE)
                    simulation_time_s = float(5.0 * tau_s)

                    time_grid_s, conc_profile, ok, message = (
                        reactors.integrate_cstr_profile(
                            simulation_time_s=simulation_time_s,
                            temperature_K=temperature_K,
                            reactor_volume_m3=reactor_volume_m3,
                            vdot_m3_s=vdot_m3_s,
                            conc_inlet_mol_m3=conc_inlet,
                            stoich_matrix=stoich_matrix_fit,
                            k0=fitted_params["k0"],
                            ea_J_mol=fitted_params["ea_J_mol"],
                            reaction_order_matrix=fitted_params[
                                "reaction_order_matrix"
                            ],
                            solver_method=solver_method_fit,
                            rtol=rtol_fit,
                            atol=atol_fit,
                            n_points=profile_points,
                            kinetic_model=kinetic_model_fit,
                            max_step_fraction=max_step_fraction_fit,
                            K0_ads=fitted_params.get("K0_ads", None),
                            Ea_K_J_mol=fitted_params.get("Ea_K", None),
                            m_inhibition=fitted_params.get("m_inhibition", None),
                            k0_rev=fitted_params.get("k0_rev", None),
                            ea_rev_J_mol=fitted_params.get("ea_rev", None),
                            order_rev_matrix=fitted_params.get("order_rev", None),
                        )
                    )

                    if not ok:
                        st.error(
                            f"CSTR 剖面计算失败: {message}\n"
                            "建议：尝试将求解器切换为 `BDF` 或 `Radau`，并适当放宽 `rtol/atol`。"
                        )
                    else:
                        fig_cs, ax_cs = plt.subplots(figsize=(7, 4.5))
                        name_to_index = {
                            name: i for i, name in enumerate(species_names_fit)
                        }
                        profile_df = pd.DataFrame({"t_s": time_grid_s})
                        for i, species_name in enumerate(profile_species):
                            idx = name_to_index[species_name]
                            y = conc_profile[idx, :]
                            _plot_reference_series(
                                ax_cs,
                                time_grid_s,
                                y,
                                label=species_name,
                                color=_fit_plot_color(i),
                            )
                            profile_df[f"C_{species_name}_mol_m3"] = y

                        ax_cs.set_xlabel(ui_text.AXIS_LABEL_TIME)
                        ax_cs.set_ylabel(ui_text.AXIS_LABEL_CONCENTRATION)
                        _style_fit_axis(ax_cs, show_grid=False)
                        _style_fit_legend(ax_cs)
                        st.pyplot(fig_cs)

                        st.download_button(
                            "📥 下载剖面数据 CSV",
                            profile_df.to_csv(index=False).encode("utf-8"),
                            file_name="profile_data.csv",
                            mime="text/csv",
                        )
                        image_format_cs = st.selectbox(
                            "剖面图格式",
                            ["png", "svg"],
                            index=0,
                            key="cstr_profile_image_format",
                        )
                        st.download_button(
                            "📥 下载剖面图",
                            ui_comp.figure_to_image_bytes(fig_cs, image_format_cs),
                            file_name=f"profile_plot.{image_format_cs}",
                            mime=(
                                "image/png"
                                if image_format_cs == "png"
                                else "image/svg+xml"
                            ),
                        )
                        plt.close(fig_cs)

                else:
                    profile_kind = "C (mol/m^3)"
                    reaction_time_s = float(row_sel.get("t_s", np.nan))
                    temperature_K = float(row_sel.get("T_K", np.nan))
                    conc_initial = np.zeros(len(species_names_fit), dtype=float)
                    for i, sp_name in enumerate(species_names_fit):
                        conc_initial[i] = float(
                            row_sel.get(f"C0_{sp_name}_mol_m3", np.nan)
                        )

                    time_grid_s, conc_profile, ok, message = (
                        reactors.integrate_batch_profile(
                            reaction_time_s=reaction_time_s,
                            temperature_K=temperature_K,
                            conc_initial_mol_m3=conc_initial,
                            stoich_matrix=stoich_matrix_fit,
                            k0=fitted_params["k0"],
                            ea_J_mol=fitted_params["ea_J_mol"],
                            reaction_order_matrix=fitted_params[
                                "reaction_order_matrix"
                            ],
                            solver_method=solver_method_fit,
                            rtol=rtol_fit,
                            atol=atol_fit,
                            n_points=profile_points,
                            kinetic_model=kinetic_model_fit,
                            max_step_fraction=max_step_fraction_fit,
                            K0_ads=fitted_params.get("K0_ads", None),
                            Ea_K_J_mol=fitted_params.get("Ea_K", None),
                            m_inhibition=fitted_params.get("m_inhibition", None),
                            k0_rev=fitted_params.get("k0_rev", None),
                            ea_rev_J_mol=fitted_params.get("ea_rev", None),
                            order_rev_matrix=fitted_params.get("order_rev", None),
                        )
                    )
                    if not ok:
                        st.error(
                            f"BSTR 剖面计算失败: {message}\n"
                            "建议：尝试将求解器切换为 `BDF` 或 `Radau`，并适当放宽 `rtol/atol`。"
                        )
                    else:
                        fig_bt, ax_bt = plt.subplots(figsize=(7, 4.5))
                        name_to_index = {
                            name: i for i, name in enumerate(species_names_fit)
                        }
                        profile_df = pd.DataFrame({"t_s": time_grid_s})
                        for i, species_name in enumerate(profile_species):
                            idx = name_to_index[species_name]
                            y = conc_profile[idx, :]
                            _plot_reference_series(
                                ax_bt,
                                time_grid_s,
                                y,
                                label=species_name,
                                color=_fit_plot_color(i),
                            )
                            profile_df[f"C_{species_name}_mol_m3"] = y

                        ax_bt.set_xlabel(ui_text.AXIS_LABEL_TIME)
                        ax_bt.set_ylabel(ui_text.AXIS_LABEL_CONCENTRATION)
                        _style_fit_axis(ax_bt, show_grid=False)
                        _style_fit_legend(ax_bt)
                        st.pyplot(fig_bt)

                        st.download_button(
                            "📥 下载剖面数据 CSV",
                            profile_df.to_csv(index=False).encode("utf-8"),
                            file_name="profile_data.csv",
                            mime="text/csv",
                        )
                        image_format_bt = st.selectbox(
                            "剖面图格式",
                            ["png", "svg"],
                            index=0,
                            key="batch_profile_image_format",
                        )
                        st.download_button(
                            "📥 下载剖面图",
                            ui_comp.figure_to_image_bytes(fig_bt, image_format_bt),
                            file_name=f"profile_plot.{image_format_bt}",
                            mime=(
                                "image/png"
                                if image_format_bt == "png"
                                else "image/svg+xml"
                            ),
                        )
                        plt.close(fig_bt)

        with tab_export:
            st.markdown("#### 导出拟合结果与对比数据")

            df_param_export = pd.DataFrame(
                {
                    "reaction": [f"R{i+1}" for i in range(len(fitted_params["k0"]))],
                    "k0_SI": fitted_params["k0"],
                    "Ea_J_mol": fitted_params["ea_J_mol"],
                }
            )
            st.download_button(
                "📥 导出参数（k₀, Eₐ）CSV",
                df_param_export.to_csv(index=False).encode("utf-8"),
                file_name="fit_params_k0_ea.csv",
                mime="text/csv",
            )

            fitted_params_json = json.dumps(
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in fitted_params.items()
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            st.download_button(
                "📥 导出全部拟合参数 JSON",
                fitted_params_json,
                file_name="fit_params_all.json",
                mime="application/json",
            )

            df_long = st.session_state.get("fit_compare_long_df", pd.DataFrame())
            if not df_long.empty:
                df_export = df_long.copy()
                drop_cols = [c for c in ["ok", "message"] if c in df_export.columns]
                if drop_cols:
                    df_export = df_export.drop(columns=drop_cols)
                st.download_button(
                    "📥 导出预测 vs 实验对比（长表）CSV",
                    df_export.to_csv(index=False).encode("utf-8"),
                    file_name="pred_vs_meas_long.csv",
                    mime="text/csv",
                )
            else:
                st.info("先在「奇偶校验图」页生成对比数据后，再导出对比表。")
    return {
        "data_df": data_df,
        "output_species_list": output_species_list,
    }
