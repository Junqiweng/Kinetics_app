from __future__ import annotations

import queue
import threading

import streamlit as st

import modules.ui_text as ui_text
from modules.config_state import _warn_once
from modules.fit_setup import derive_effective_fit_flags
from modules.fit_state import (
    build_fit_result_state_snapshot,
    build_fit_state_snapshot,
    describe_fit_state_differences,
)
from modules.fitting_background import (
    _count_fitted_parameters,
    _drain_fitting_progress_queue,
    _get_fitting_executor,
    _render_fitting_live_progress,
    _render_fitting_progress_panel,
    _reset_fitting_progress_ui_state,
    _run_fitting_job,
)


def render_fit_actions(ctx: dict, fit_advanced_state: dict) -> dict:
    data_df = ctx["data_df"]
    output_mode = ctx["output_mode"]
    output_species_list = ctx["output_species_list"]
    species_names = ctx["species_names"]
    stoich_matrix = ctx["stoich_matrix"]
    k0_guess = ctx["k0_guess"]
    ea_guess_J_mol = ctx["ea_guess_J_mol"]
    order_guess = ctx["order_guess"]
    fit_k0_flags = ctx["fit_k0_flags"]
    fit_ea_flags = ctx["fit_ea_flags"]
    fit_order_flags_matrix = ctx["fit_order_flags_matrix"]
    K0_ads = ctx["K0_ads"]
    Ea_K_J_mol = ctx["Ea_K_J_mol"]
    m_inhibition = ctx["m_inhibition"]
    fit_K0_ads_flags = ctx["fit_K0_ads_flags"]
    fit_Ea_K_flags = ctx["fit_Ea_K_flags"]
    fit_m_flags = ctx["fit_m_flags"]
    k0_rev = ctx["k0_rev"]
    ea_rev_J_mol = ctx["ea_rev_J_mol"]
    order_rev = ctx["order_rev"]
    fit_k0_rev_flags = ctx["fit_k0_rev_flags"]
    fit_ea_rev_flags = ctx["fit_ea_rev_flags"]
    fit_order_rev_flags_matrix = ctx["fit_order_rev_flags_matrix"]
    solver_method = ctx["solver_method"]
    rtol = ctx["rtol"]
    atol = ctx["atol"]
    reactor_type = ctx["reactor_type"]
    kinetic_model = ctx["kinetic_model"]
    reversible_enabled = bool(ctx.get("reversible_enabled", False))
    pfr_flow_model = ctx["pfr_flow_model"]
    _request_start_fitting = ctx["request_start_fitting"]
    _request_stop_fitting = ctx["request_stop_fitting"]

    effective_fit_flags = derive_effective_fit_flags(
        ctx,
        str(kinetic_model),
        bool(reversible_enabled),
    )
    fit_K0_ads_flags = effective_fit_flags["fit_K0_ads_flags"]
    fit_Ea_K_flags = effective_fit_flags["fit_Ea_K_flags"]
    fit_m_flags = effective_fit_flags["fit_m_flags"]
    fit_k0_rev_flags = effective_fit_flags["fit_k0_rev_flags"]
    fit_ea_rev_flags = effective_fit_flags["fit_ea_rev_flags"]
    fit_order_rev_flags_matrix = effective_fit_flags["fit_order_rev_flags_matrix"]

    k0_min = float(fit_advanced_state["k0_min"])
    k0_max = float(fit_advanced_state["k0_max"])
    ea_min = float(fit_advanced_state["ea_min"])
    ea_max = float(fit_advanced_state["ea_max"])
    ord_min = float(fit_advanced_state["ord_min"])
    ord_max = float(fit_advanced_state["ord_max"])
    K0_ads_min = float(fit_advanced_state["K0_ads_min"])
    K0_ads_max = float(fit_advanced_state["K0_ads_max"])
    Ea_K_min = float(fit_advanced_state["Ea_K_min"])
    Ea_K_max = float(fit_advanced_state["Ea_K_max"])
    k0_rev_min = float(fit_advanced_state["k0_rev_min"])
    k0_rev_max = float(fit_advanced_state["k0_rev_max"])
    ea_rev_min_J_mol = float(fit_advanced_state["ea_rev_min_J_mol"])
    ea_rev_max_J_mol = float(fit_advanced_state["ea_rev_max_J_mol"])
    order_rev_min = float(fit_advanced_state["order_rev_min"])
    order_rev_max = float(fit_advanced_state["order_rev_max"])
    max_nfev = int(fit_advanced_state["max_nfev"])
    diff_step_rel = float(fit_advanced_state["diff_step_rel"])
    max_step_fraction = float(fit_advanced_state["max_step_fraction"])
    use_ms = bool(fit_advanced_state["use_ms"])
    n_starts = int(fit_advanced_state["n_starts"])
    max_nfev_coarse = int(fit_advanced_state["max_nfev_coarse"])
    use_x_scale_jac = bool(fit_advanced_state["use_x_scale_jac"])
    use_log_k0_fit = bool(fit_advanced_state["use_log_k0_fit"])
    use_log_k0_rev_fit = bool(fit_advanced_state["use_log_k0_rev_fit"])
    use_log_K0_ads_fit = bool(fit_advanced_state["use_log_K0_ads_fit"])
    random_seed = int(fit_advanced_state["random_seed"])
    residual_type = str(fit_advanced_state["residual_type"])
    _drain_fitting_progress_queue()

    fitting_stop_event = st.session_state.get("fitting_stop_event", None)
    if fitting_stop_event is None:
        fitting_stop_event = threading.Event()
        st.session_state["fitting_stop_event"] = fitting_stop_event

    fitting_future = st.session_state.get("fitting_future", None)
    fitting_running = bool(st.session_state.get("fitting_running", False))

    with st.container(border=True):
        st.markdown('<div class="kinetics-card-marker"></div>', unsafe_allow_html=True)
        col_act1, col_act2, col_act3, col_act4, col_act5 = st.columns(
            [3, 1, 1, 1, 1], vertical_alignment="center"
        )
        col_act1.button(
            "🚀 开始拟合",
            type="primary",
            disabled=fitting_running,
            width="stretch",
            on_click=_request_start_fitting,
        )
        col_act2.button(
            "⏹️ 终止",
            type="secondary",
            disabled=not fitting_running,
            width="stretch",
            on_click=_request_stop_fitting,
        )
        auto_refresh = col_act3.checkbox(
            "自动刷新",
            value=bool(st.session_state.get("fitting_auto_refresh", True)),
            help="开启后，页面会按设定间隔自动刷新，以持续更新拟合进度与阶段信息；关闭可降低页面刷新负载与 CPU 占用。",
        )
        col_interval_label, col_interval_input = col_act5.columns(
            [1.1, 1.4], vertical_alignment="center"
        )
        col_interval_label.markdown(
            '<div class="kinetics-inline-label">间隔(s)</div>',
            unsafe_allow_html=True,
        )
        refresh_interval_s = round(
            float(
                col_interval_input.number_input(
                    "间隔(s)",
                    value=float(
                        st.session_state.get("fitting_refresh_interval_s", 2.0)
                    ),
                    min_value=0.5,
                    max_value=10.0,
                    step=0.5,
                    format="%.1f",
                    key="cfg_refresh_interval_s_ui",
                    disabled=(not auto_refresh),
                    help="自动刷新间隔 [s]（可在拟合前预设）",
                    label_visibility="collapsed",
                )
            ),
            1,
        )
        clear_btn = col_act4.button(
            "🧹 清除结果",
            type="secondary",
            disabled=fitting_running,
            width="stretch",
            help="清除上一次拟合的结果、对比表缓存与时间线（不影响当前输入配置）。",
        )
    st.session_state["fitting_auto_refresh"] = bool(auto_refresh)
    st.session_state["fitting_refresh_interval_s"] = round(float(refresh_interval_s), 1)

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

    fit_results_cached = st.session_state.get("fit_results", None)
    stale_reasons: list[str] = []
    if isinstance(fit_results_cached, dict):
        active_data_df = st.session_state.get("data_df_cached", None)
        if active_data_df is None:
            active_data_df = data_df
        current_state = build_fit_state_snapshot(
            data_df=active_data_df,
            species_names=species_names,
            output_mode=str(output_mode),
            output_species_list=list(output_species_list),
            stoich_matrix=stoich_matrix,
            solver_method=str(solver_method),
            rtol=float(rtol),
            atol=float(atol),
            reactor_type=str(reactor_type),
            kinetic_model=str(kinetic_model),
            reversible_enabled=bool(reversible_enabled),
            pfr_flow_model=str(pfr_flow_model),
            max_step_fraction=float(max_step_fraction),
            residual_type=str(residual_type),
            use_log_k0_fit=bool(use_log_k0_fit),
            use_log_k0_rev_fit=bool(use_log_k0_rev_fit),
            use_log_K0_ads_fit=bool(use_log_K0_ads_fit),
            fit_k0_flags=fit_k0_flags,
            fit_ea_flags=fit_ea_flags,
            fit_order_flags_matrix=fit_order_flags_matrix,
            fit_K0_ads_flags=fit_K0_ads_flags,
            fit_Ea_K_flags=fit_Ea_K_flags,
            fit_m_flags=fit_m_flags,
            fit_k0_rev_flags=fit_k0_rev_flags,
            fit_ea_rev_flags=fit_ea_rev_flags,
            fit_order_rev_flags_matrix=fit_order_rev_flags_matrix,
            k0_min=float(k0_min),
            k0_max=float(k0_max),
            ea_min=float(ea_min),
            ea_max=float(ea_max),
            ord_min=float(ord_min),
            ord_max=float(ord_max),
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
        )
        stale_reasons = describe_fit_state_differences(
            current_state,
            build_fit_result_state_snapshot(fit_results_cached),
        )
    st.session_state["fit_results_stale_reasons"] = list(stale_reasons)
    if stale_reasons:
        st.warning(
            "当前缓存的拟合结果已过期，建议重新拟合后再判断参数或误差表现。"
        )
        for reason in stale_reasons:
            st.markdown(f"- {reason}")

    if clear_btn and (not fitting_running):
        for key in [
            "fit_results",
            "fit_compare_long_df",
            "fit_results_stale_reasons",
            "fitting_timeline",
            "fitting_metrics",
            "fitting_ms_summary",
            "fitting_ms_results_log",
            "fitting_final_summary",
            "fitting_history",
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
            reactor_label = ui_text.map_label(
                ui_text.REACTOR_TYPE_LABELS, str(reactor_type)
            )
            if str(reactor_type) == "PFR":
                reactor_label = (
                    f"{reactor_label} / "
                    f"{ui_text.map_label(ui_text.PFR_FLOW_MODEL_LABELS, str(pfr_flow_model))}"
                )
            solver_label = ui_text.map_label(
                ui_text.SOLVER_METHOD_LABELS, str(solver_method)
            )
            target_species_text = (
                "、".join([str(x) for x in output_species_list])
                if output_species_list
                else "未选择"
            )
            if use_ms and int(n_starts) > 1:
                ms_text = (
                    f"开启（n_starts={int(n_starts)}, coarse_max_nfev={int(max_nfev_coarse)}, "
                    f"seed={int(random_seed)}）"
                )
            else:
                ms_text = "关闭"

            st.session_state["fitting_job_summary"] = {
                "title": "拟合任务概览",
                "lines": [
                    f"数据规模: {int(len(data_df))} 行 × {int(len(output_species_list))} 个目标物种",
                    f"待拟合参数: {int(n_fit_params)} 个",
                    f"目标物种: {target_species_text}",
                    f"反应器/流动模型: {reactor_label}",
                    f"动力学模型: {ui_text.kinetic_model_display_text(str(kinetic_model), bool(reversible_enabled))}",
                    f"残差定义: {residual_type}",
                    (
                        f"数值求解: {solver_label}, "
                        f"rtol={float(rtol):.1e}, atol={float(atol):.1e}"
                    ),
                    (
                        "优化设置: least_squares(trf), "
                        f"max_nfev={int(max_nfev)}, diff_step={float(diff_step_rel):.1e}, "
                        f"x_scale={'jac' if bool(use_x_scale_jac) else '1.0'}"
                    ),
                    (
                        "对数拟合: "
                        f"k₀={'开' if bool(use_log_k0_fit) else '关'}, "
                        f"k₀,rev={'开' if bool(use_log_k0_rev_fit) else '关'}, "
                        f"K₀,ads={'开' if bool(use_log_K0_ads_fit) else '关'}"
                    ),
                    f"多起点策略: {ms_text}",
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
                "reversible_enabled": bool(reversible_enabled),
                "pfr_flow_model": str(pfr_flow_model),
                "use_ms": use_ms,
                "n_starts": n_starts,
                "random_seed": random_seed,
                "max_nfev": max_nfev,
                "max_nfev_coarse": max_nfev_coarse,
                "diff_step_rel": diff_step_rel,
                "use_x_scale_jac": use_x_scale_jac,
                "use_log_k0_fit": use_log_k0_fit,
                "use_log_k0_rev_fit": use_log_k0_rev_fit,
                "use_log_K0_ads_fit": use_log_K0_ads_fit,
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
        st.caption(
            "“自动刷新”：仅刷新进度区域（避免整页闪烁）；如需降低页面刷新负载可关闭。"
        )
        refresh_interval_s = float(
            st.session_state.get("fitting_refresh_interval_s", 2.0)
        )
        refresh_interval_s = float(max(0.2, min(30.0, refresh_interval_s)))
        if bool(st.session_state.get("fitting_auto_refresh", True)):

            @st.fragment(run_every=refresh_interval_s)
            def _fit_live_progress_fragment() -> None:
                _render_fitting_live_progress()

            _fit_live_progress_fragment()
        else:
            _render_fitting_live_progress()
    elif st.session_state.get("fitting_timeline", []):
        _render_fitting_progress_panel()

    # 在拟合页底部创建结果容器
    tab_fit_results_container = st.container()

    return {
        "tab_fit_results_container": tab_fit_results_container,
        "fitting_running": bool(st.session_state.get("fitting_running", False)),
    }
