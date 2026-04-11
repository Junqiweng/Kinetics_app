# 文件作用：后台拟合任务的线程调度、进度队列、UI 进度展示与结果回传。

from __future__ import annotations

import datetime as _dt
import difflib
import hashlib
import html as html_lib
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import least_squares

from . import fitting
from .fit_state import (
    build_fit_state_snapshot,
    build_fit_result_state_snapshot,
    serialize_params_snapshot,
)
from .constants import (
    CSV_CLOSE_MATCHES_CUTOFF,
    CSV_CLOSE_MATCHES_MAX,
    CSV_COLUMN_PREVIEW_COUNT,
    CSV_INVALID_INDEX_PREVIEW_COUNT,
    DEFAULT_EA_K_MAX_J_MOL,
    DEFAULT_EA_K_MIN_J_MOL,
    DEFAULT_EA_REV_MAX_J_MOL,
    DEFAULT_EA_REV_MIN_J_MOL,
    DEFAULT_K0_ADS_MAX,
    DEFAULT_K0_REV_MAX,
    DEFAULT_K0_REV_MIN,
    DEFAULT_M_INHIBITION_MAX,
    DEFAULT_M_INHIBITION_MIN,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_ORDER_REV_MAX,
    DEFAULT_ORDER_REV_MIN,
    DEFAULT_RESIDUAL_PENALTY_MULTIPLIER,
    DEFAULT_RESIDUAL_PENALTY_MIN_ABS,
    EPSILON_RELATIVE,
    FITTING_EPSILON_NORM,
    FITTING_EPSILON_PHI_RATIO,
    FITTING_UI_UPDATE_INTERVAL_S,
    PERCENTAGE_RESIDUAL_EPSILON_FACTOR,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    KINETIC_MODEL_POWER_LAW,
    KINETIC_MODEL_REVERSIBLE,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
)


class FittingStoppedError(Exception):
    pass


def _append_fit_history_entry(fit_results: dict | None) -> None:
    """将一次完成的拟合结果写入当前会话历史。"""
    if not isinstance(fit_results, dict):
        return

    if "fitting_history" not in st.session_state:
        st.session_state["fitting_history"] = []

    fit_history = st.session_state["fitting_history"]
    fit_id = int(len(fit_history) + 1)
    phi_value = float(fit_results.get("phi_final", fit_results.get("cost", 0.0)))
    history_entry = {
        "fit_id": fit_id,
        "time": _dt.datetime.now().strftime("%H:%M:%S"),
        "phi": phi_value,
        "kinetic_model": str(fit_results.get("kinetic_model", "")),
        "reactor_type": str(fit_results.get("reactor_type", "")),
        "residual_type": str(fit_results.get("residual_type", "")),
        "n_params": int(fit_results.get("n_fit_params", 0)),
        "skipped": bool(fit_results.get("fit_skipped", False)),
        "output_mode": str(fit_results.get("output_mode", "")),
        "output_species": [str(x) for x in fit_results.get("output_species", [])],
        "species_names": [str(x) for x in fit_results.get("species_names", [])],
        "state_snapshot": build_fit_result_state_snapshot(fit_results),
        "params_snapshot": serialize_params_snapshot(fit_results.get("params", {})),
    }
    fit_history.append(history_entry)


def _build_actionable_error_message(error_str: str) -> str:
    """将技术性错误信息转换为包含可操作建议的用户友好提示。"""
    msg = f"拟合失败: {error_str}"
    lower = error_str.lower()

    suggestions: list[str] = []

    # ODE 积分相关
    if any(kw in lower for kw in ["ode", "solve_ivp", "integration", "积分", "step size", "too small", "max_step"]):
        suggestions.append("切换求解器为 BDF 或 Radau（适合刚性问题）")
        suggestions.append("减小 rtol/atol（如 1e-8 / 1e-10）")
        suggestions.append("减小 max_step_fraction（如 0.01）限制步长")
        suggestions.append("检查初始参数 k₀、Eₐ 的量级是否合理")

    # 优化未收敛
    if any(kw in lower for kw in ["max_nfev", "maximum number of function", "未收敛", "not converge"]):
        suggestions.append("增大 max_nfev（如 5000 或 10000）")
        suggestions.append("开启多起点拟合（Multi-start）")
        suggestions.append("检查参数边界是否过窄或过宽")

    # 多起点失败
    if any(kw in lower for kw in ["multi-start", "多起点", "未获得有效结果"]):
        suggestions.append("增加起点数量（n_starts）")
        suggestions.append("放宽参数边界范围")
        suggestions.append("检查数据是否有异常值或 NaN")

    # 矩阵/数值问题
    if any(kw in lower for kw in ["singular", "nan", "inf", "overflow", "underflow", "finite"]):
        suggestions.append("检查初始参数量级（k₀ 过大/过小均可导致数值溢出）")
        suggestions.append("尝试使用参数缩放（勾选 x_scale='jac'）")
        suggestions.append("检查化学计量数矩阵和反应级数是否正确")

    # 边界相关
    if any(kw in lower for kw in ["bound", "边界", "lower", "upper", "infeasible"]):
        suggestions.append("检查参数边界设置：上界应大于下界")
        suggestions.append("确保初始猜测值在边界范围内")

    # 数据相关
    if any(kw in lower for kw in ["column", "列", "csv", "dataframe", "key error"]):
        suggestions.append("检查 CSV 文件列名是否与模板一致")
        suggestions.append("确保所选目标物种在数据中有对应的测量列")

    if suggestions:
        advice = "\n\n**建议操作：**\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(suggestions))
        msg += advice

    return msg


def _get_fitting_executor() -> ThreadPoolExecutor:
    """
    每个会话单独的线程池（避免跨会话共享导致“任务占用/卡住”）。
    """
    executor = st.session_state.get("fitting_executor", None)
    is_shutdown = (
        bool(getattr(executor, "_shutdown", False)) if executor is not None else True
    )
    if executor is None or is_shutdown:
        executor = ThreadPoolExecutor(max_workers=1)
        st.session_state["fitting_executor"] = executor
    return executor


def _drain_fitting_progress_queue() -> None:
    if "fitting_progress_queue" not in st.session_state:
        return

    progress_queue = st.session_state["fitting_progress_queue"]
    while True:
        try:
            msg_type, msg_value = progress_queue.get_nowait()
        except queue.Empty:
            break

        if msg_type == "progress":
            st.session_state["fitting_progress"] = float(msg_value)
        elif msg_type == "status":
            st.session_state["fitting_status"] = str(msg_value)
        elif msg_type == "timeline_add":
            if "fitting_timeline" not in st.session_state:
                st.session_state["fitting_timeline"] = []
            icon = str(msg_value.get("icon", "")).strip()
            text = str(msg_value.get("text", "")).strip()
            if icon or text:
                st.session_state["fitting_timeline"].append((icon, text))
        elif msg_type == "metric":
            if "fitting_metrics" not in st.session_state:
                st.session_state["fitting_metrics"] = {}
            metric_name = str(msg_value.get("name", "")).strip()
            metric_value = msg_value.get("value", None)
            if metric_name:
                st.session_state["fitting_metrics"][metric_name] = metric_value
        elif msg_type == "ms_summary":
            st.session_state["fitting_ms_summary"] = str(msg_value)
        elif msg_type == "ms_results_log":
            st.session_state["fitting_ms_results_log"] = msg_value
        elif msg_type == "final_summary":
            st.session_state["fitting_final_summary"] = str(msg_value)


def _count_fitted_parameters(
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    fit_K0_ads_flags: np.ndarray,
    fit_Ea_K_flags: np.ndarray,
    fit_m_flags: np.ndarray,
    fit_k0_rev_flags: np.ndarray,
    fit_ea_rev_flags: np.ndarray,
    fit_order_rev_flags_matrix: np.ndarray,
) -> int:
    n_fit_params = 0
    for flags in [
        fit_k0_flags,
        fit_ea_flags,
        fit_order_flags_matrix,
        fit_K0_ads_flags,
        fit_Ea_K_flags,
        fit_m_flags,
        fit_k0_rev_flags,
        fit_ea_rev_flags,
        fit_order_rev_flags_matrix,
    ]:
        if flags is None:
            continue
        n_fit_params += int(np.count_nonzero(np.asarray(flags, dtype=bool)))
    return int(n_fit_params)


def _reset_fitting_progress_ui_state() -> None:
    st.session_state["fitting_timeline"] = []
    st.session_state["fitting_metrics"] = {}
    st.session_state["fitting_ms_summary"] = ""
    st.session_state["fitting_ms_results_log"] = []
    st.session_state["fitting_final_summary"] = ""


def _render_fitting_overview_box(job_summary: dict) -> None:
    lines = job_summary.get("lines", [])
    if not lines:
        return

    title = html_lib.escape(str(job_summary.get("title", "拟合任务概览")))
    bullet_html = "\n".join([f"<li>{html_lib.escape(str(x))}</li>" for x in lines])
    st.markdown(
        f"""
        <div class="kinetics-overview-box">
          <div class="kinetics-overview-head">
            <div class="kinetics-overview-dot"></div>
            <div class="kinetics-overview-title">{title}</div>
          </div>
          <ul class="kinetics-overview-list">
            {bullet_html}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_lines(summary_text: str) -> None:
    lines = [line.strip() for line in str(summary_text).splitlines() if line.strip()]
    if not lines:
        return
    for line in lines:
        st.write(f"• {line}")


def _render_fitting_progress_panel() -> None:
    job_summary = st.session_state.get("fitting_job_summary", {})
    timeline = st.session_state.get("fitting_timeline", [])
    ms_summary = str(st.session_state.get("fitting_ms_summary", "")).strip()
    final_summary = str(st.session_state.get("fitting_final_summary", "")).strip()
    fitting_history = st.session_state.get("fitting_history", [])

    if not (job_summary or timeline or ms_summary or final_summary or fitting_history):
        return

    with st.expander("拟合任务概览与摘要（点击展开）", expanded=False):
        if job_summary:
            _render_fitting_overview_box(job_summary)

        if timeline:
            st.write("")
            with st.container(border=True):
                st.markdown(
                    '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
                )
                st.markdown("#### 进度日志")
                for icon, text in timeline:
                    text = str(text).strip()
                    if text:
                        st.markdown(f"- {icon} {text}")

        if ms_summary:
            st.write("")
            with st.container(border=True):
                st.markdown(
                    '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
                )
                st.markdown("#### 多起点（Multi-start）摘要")
                _render_summary_lines(ms_summary)

                # 多起点对比表
                ms_log = st.session_state.get("fitting_ms_results_log", None)
                if isinstance(ms_log, list) and ms_log:
                    st.markdown("**各起点结果对比**")
                    df_ms = pd.DataFrame(ms_log)
                    df_ms = df_ms.rename(columns={
                        "start": "起点",
                        "cost": "Φ (cost)",
                        "nfev": "函数评估次数",
                        "status": "收敛状态",
                        "is_best": "最佳",
                    })
                    df_ms = df_ms.sort_values("Φ (cost)")
                    st.dataframe(df_ms, width="stretch", hide_index=True)

        if final_summary:
            st.write("")
            with st.container(border=True):
                st.markdown(
                    '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
                )
                st.markdown("#### 拟合摘要")
                _render_summary_lines(final_summary)

        if fitting_history and len(fitting_history) > 1:
            st.write("")
            with st.container(border=True):
                st.markdown(
                    '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
                )
                st.markdown("#### 拟合历史（本次会话）")
                df_hist = pd.DataFrame(fitting_history)
                df_hist = df_hist.drop(
                    columns=["state_snapshot", "params_snapshot"],
                    errors="ignore",
                )
                df_hist = df_hist.rename(columns={
                    "fit_id": "拟合 #",
                    "time": "时间",
                    "phi": "Φ (cost)",
                    "kinetic_model": "动力学模型",
                    "reactor_type": "反应器",
                    "residual_type": "残差类型",
                    "n_params": "拟合参数数",
                    "skipped": "跳过",
                })
                st.dataframe(df_hist, width="stretch", hide_index=True)


def _finalize_finished_fitting_future() -> None:
    """
    在 fragment 刷新上下文中完成 future 收尾。

    说明：
    - 仅在这里回收结果并写入 session_state；
    - 是否触发整页渲染由 _render_fitting_live_progress 统一控制。
    """
    fitting_future = st.session_state.get("fitting_future", None)
    if fitting_future is None:
        return
    if not bool(fitting_future.done()):
        return

    st.session_state["fitting_running"] = False
    st.session_state["fitting_future"] = None
    st.session_state["fitting_needs_app_rerun"] = True
    if "fitting_timeline" not in st.session_state:
        st.session_state["fitting_timeline"] = []

    try:
        fit_results = fitting_future.result()
        _drain_fitting_progress_queue()
        st.session_state["fit_results"] = fit_results
        st.session_state["fit_results_version"] = (
            int(st.session_state.get("fit_results_version", 0)) + 1
        )
        st.session_state["fitting_status"] = "拟合完成。"

        phi_value = float(fit_results.get("phi_final", fit_results.get("cost", 0.0)))
        phi_text = f"{phi_value:.3e}"

        _append_fit_history_entry(fit_results)

        st.session_state["fitting_timeline"].append(
            ("✅", f"拟合完成，最终 Φ: {phi_text}")
        )
        st.session_state["fit_notice"] = {
            "kind": "success",
            "text": "拟合完成！结果已缓存（结果展示将锁定为本次拟合的配置与数据）。"
            f" 目标函数 Φ: {phi_text}",
        }
    except FittingStoppedError:
        st.session_state["fitting_status"] = "用户终止。"
        st.session_state["fitting_timeline"].append(("⚠️", "拟合已终止。"))
        st.session_state["fit_notice"] = {"kind": "warning", "text": "拟合已终止。"}
    except Exception as exc:
        st.session_state["fitting_status"] = "拟合失败。"
        st.session_state["fitting_timeline"].append(("❌", f"拟合失败: {exc}"))
        error_text = _build_actionable_error_message(str(exc))
        st.session_state["fit_notice"] = {
            "kind": "error",
            "text": error_text,
        }


def _render_fitting_live_progress() -> None:
    """
    只刷新“进度显示”区域，避免整页闪烁。

    说明：当后台拟合完成时，会触发一次全局 rerun 以渲染最终结果。
    """
    _drain_fitting_progress_queue()

    fitting_future = st.session_state.get("fitting_future", None)
    fitting_running = bool(st.session_state.get("fitting_running", False))

    # 重要：st.fragment(run_every=...) 只会重跑 fragment 本身，不会执行整页脚本。
    # 因此当后台任务结束后，需要触发一次 app 级 rerun 才能立即展示完整结果。
    if fitting_running and (fitting_future is not None) and bool(fitting_future.done()):
        _finalize_finished_fitting_future()
        fitting_running = bool(st.session_state.get("fitting_running", False))

    if bool(st.session_state.get("fitting_needs_app_rerun", False)):
        st.session_state["fitting_needs_app_rerun"] = False
        st.rerun(scope="app")
        return

    if not fitting_running:
        return

    with st.container(border=True):
        st.markdown('<div class="kinetics-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("#### 后台拟合中")
        st.caption("拟合正在后台运行中（页面可继续操作）。")
        st.progress(float(st.session_state.get("fitting_progress", 0.0)))
        status_text = str(st.session_state.get("fitting_status", "")).strip()
        if status_text:
            st.caption(status_text)

    _render_fitting_progress_panel()

    if not bool(st.session_state.get("fitting_auto_refresh", True)):
        st.button("🔄 刷新进度", width="stretch", key="fit_manual_refresh_progress")


def _run_fitting_job(
    job_inputs: dict, stop_event: threading.Event, progress_queue: queue.Queue
) -> dict:
    def set_status(status_text: str) -> None:
        progress_queue.put(("status", status_text))

    def set_progress(progress_value: float) -> None:
        progress_queue.put(("progress", float(progress_value)))

    def timeline_add(icon: str, text: str) -> None:
        progress_queue.put(("timeline_add", {"icon": icon, "text": text}))

    def set_metric(name: str, value) -> None:
        progress_queue.put(("metric", {"name": name, "value": value}))

    def set_ms_summary(text: str) -> None:
        progress_queue.put(("ms_summary", str(text)))

    def set_final_summary(text: str) -> None:
        progress_queue.put(("final_summary", str(text)))

    data_df = job_inputs["data_df"]
    species_names = job_inputs["species_names"]
    output_mode = job_inputs["output_mode"]
    output_species_list = job_inputs["output_species_list"]
    stoich_matrix = job_inputs["stoich_matrix"]

    k0_guess = job_inputs["k0_guess"]
    ea_guess_J_mol = job_inputs["ea_guess_J_mol"]
    order_guess = job_inputs["order_guess"]
    fit_k0_flags = job_inputs["fit_k0_flags"]
    fit_ea_flags = job_inputs["fit_ea_flags"]
    fit_order_flags_matrix = job_inputs["fit_order_flags_matrix"]

    K0_ads = job_inputs["K0_ads"]
    Ea_K_J_mol = job_inputs["Ea_K_J_mol"]
    m_inhibition = job_inputs["m_inhibition"]
    fit_K0_ads_flags = job_inputs["fit_K0_ads_flags"]
    fit_Ea_K_flags = job_inputs["fit_Ea_K_flags"]
    fit_m_flags = job_inputs["fit_m_flags"]

    k0_rev = job_inputs["k0_rev"]
    ea_rev_J_mol = job_inputs["ea_rev_J_mol"]
    order_rev = job_inputs["order_rev"]
    fit_k0_rev_flags = job_inputs["fit_k0_rev_flags"]
    fit_ea_rev_flags = job_inputs["fit_ea_rev_flags"]
    fit_order_rev_flags_matrix = job_inputs["fit_order_rev_flags_matrix"]

    solver_method = job_inputs["solver_method"]
    rtol = job_inputs["rtol"]
    atol = job_inputs["atol"]
    reactor_type = job_inputs["reactor_type"]
    kinetic_model = job_inputs["kinetic_model"]
    reversible_enabled = bool(job_inputs.get("reversible_enabled", False))
    if str(kinetic_model).strip() == KINETIC_MODEL_REVERSIBLE:
        kinetic_model = KINETIC_MODEL_POWER_LAW
        reversible_enabled = True
    pfr_flow_model = str(
        job_inputs.get("pfr_flow_model", PFR_FLOW_MODEL_LIQUID_CONST_VDOT)
    ).strip()
    if pfr_flow_model not in (
        PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
        PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    ):
        pfr_flow_model = PFR_FLOW_MODEL_LIQUID_CONST_VDOT

    use_ms = bool(job_inputs["use_ms"])
    n_starts = int(job_inputs["n_starts"])
    random_seed = int(job_inputs["random_seed"])
    max_nfev = int(job_inputs["max_nfev"])
    max_nfev_coarse = int(job_inputs["max_nfev_coarse"])
    diff_step_rel = float(job_inputs["diff_step_rel"])
    use_x_scale_jac = bool(job_inputs["use_x_scale_jac"])
    use_log_k0_fit = bool(job_inputs.get("use_log_k0_fit", False))
    use_log_k0_rev_fit = bool(job_inputs.get("use_log_k0_rev_fit", False))
    use_log_K0_ads_fit = bool(job_inputs.get("use_log_K0_ads_fit", False))

    k0_min = job_inputs["k0_min"]
    k0_max = job_inputs["k0_max"]
    ea_min = job_inputs["ea_min"]
    ea_max = job_inputs["ea_max"]
    ord_min = job_inputs["ord_min"]
    ord_max = job_inputs["ord_max"]
    k0_rev_min = float(job_inputs.get("k0_rev_min", DEFAULT_K0_REV_MIN))
    k0_rev_max = float(job_inputs.get("k0_rev_max", DEFAULT_K0_REV_MAX))
    ea_rev_min_J_mol = float(
        job_inputs.get("ea_rev_min_J_mol", DEFAULT_EA_REV_MIN_J_MOL)
    )
    ea_rev_max_J_mol = float(
        job_inputs.get("ea_rev_max_J_mol", DEFAULT_EA_REV_MAX_J_MOL)
    )
    ord_rev_min = float(job_inputs.get("order_rev_min", DEFAULT_ORDER_REV_MIN))
    ord_rev_max = float(job_inputs.get("order_rev_max", DEFAULT_ORDER_REV_MAX))
    K0_ads_min = job_inputs.get("K0_ads_min", 0.0)
    K0_ads_max = job_inputs.get("K0_ads_max", DEFAULT_K0_ADS_MAX)
    Ea_K_min = job_inputs.get("Ea_K_min", DEFAULT_EA_K_MIN_J_MOL)
    Ea_K_max = job_inputs.get("Ea_K_max", DEFAULT_EA_K_MAX_J_MOL)
    max_step_fraction = float(
        job_inputs.get("max_step_fraction", DEFAULT_MAX_STEP_FRACTION)
    )
    residual_type = str(job_inputs.get("residual_type", "绝对残差"))

    # 拟合前硬校验：避免无效数值设置进入后台线程后“失败罚项看似拟合成功”。
    if (not np.isfinite(float(rtol))) or (float(rtol) <= 0.0):
        raise ValueError("rtol 必须为正且有限。")
    if (not np.isfinite(float(atol))) or (float(atol) <= 0.0):
        raise ValueError("atol 必须为正且有限。")
    if (not np.isfinite(float(diff_step_rel))) or (float(diff_step_rel) <= 0.0):
        raise ValueError("diff_step_rel 必须为正且有限。")
    if int(max_nfev) < 1:
        raise ValueError("max_nfev 必须 >= 1。")
    if int(n_starts) < 1:
        raise ValueError("n_starts 必须 >= 1。")
    if int(use_ms) and int(n_starts) > 1:
        if int(max_nfev_coarse) < 1:
            raise ValueError("max_nfev_coarse 必须 >= 1（多起点模式）。")
        if int(random_seed) < 0:
            raise ValueError("random_seed 必须为非负整数（多起点模式）。")

    if stop_event.is_set():
        raise FittingStoppedError("Stopped by user")

    set_status("打包参数并构建边界...")
    set_progress(0.01)

    param_vector = fitting._pack_parameters(
        k0_guess,
        ea_guess_J_mol,
        order_guess,
        fit_k0_flags,
        fit_ea_flags,
        fit_order_flags_matrix,
        K0_ads,
        Ea_K_J_mol,
        m_inhibition,
        fit_K0_ads_flags,
        fit_Ea_K_flags,
        fit_m_flags,
        k0_rev,
        ea_rev_J_mol,
        order_rev,
        fit_k0_rev_flags,
        fit_ea_rev_flags,
        fit_order_rev_flags_matrix,
        use_log_k0_fit=use_log_k0_fit,
        use_log_k0_rev_fit=use_log_k0_rev_fit,
        use_log_K0_ads_fit=use_log_K0_ads_fit,
    )
    n_fit_params_total = int(np.asarray(param_vector).size)

    lb, ub = fitting._build_bounds(
        k0_guess,
        ea_guess_J_mol,
        order_guess,
        fit_k0_flags,
        fit_ea_flags,
        fit_order_flags_matrix,
        k0_min,
        k0_max,
        ea_min,
        ea_max,
        ord_min,
        ord_max,
        fit_K0_ads_flags,
        fit_Ea_K_flags,
        fit_m_flags,
        K0_ads_min,
        K0_ads_max,
        Ea_K_min,
        Ea_K_max,
        DEFAULT_M_INHIBITION_MIN,
        DEFAULT_M_INHIBITION_MAX,
        fit_k0_rev_flags,
        fit_ea_rev_flags,
        fit_order_rev_flags_matrix,
        k0_rev_min,
        k0_rev_max,
        ea_rev_min_J_mol,
        ea_rev_max_J_mol,
        ord_rev_min,
        ord_rev_max,
        use_log_k0_fit=use_log_k0_fit,
        use_log_k0_rev_fit=use_log_k0_rev_fit,
        use_log_K0_ads_fit=use_log_K0_ads_fit,
    )

    output_column_names = []
    for species_name in output_species_list:
        if output_mode.startswith("F"):
            column_name = f"Fout_{species_name}_mol_s"
        elif output_mode.startswith("C"):
            column_name = f"Cout_{species_name}_mol_m3"
        elif output_mode.startswith("x"):
            column_name = f"xout_{species_name}"
        else:
            raise ValueError(
                "未知输出模式；当前支持：Cout、Fout、xout（出口摩尔组成）。"
            )
        output_column_names.append(column_name)

    missing_output_columns = []
    for column_name in output_column_names:
        if column_name not in data_df.columns:
            missing_output_columns.append(column_name)

    if missing_output_columns:
        missing_columns_text = ", ".join(missing_output_columns)
        available_cols = [str(c) for c in list(data_df.columns)]
        available_cols_text = ", ".join(available_cols[:CSV_COLUMN_PREVIEW_COUNT])
        suggestions = []
        for missing_name in missing_output_columns:
            matches = difflib.get_close_matches(
                missing_name,
                available_cols,
                n=CSV_CLOSE_MATCHES_MAX,
                cutoff=CSV_CLOSE_MATCHES_CUTOFF,
            )
            if matches:
                suggestions.append(f"- `{missing_name}` 可能对应: {', '.join(matches)}")
        suggestion_text = ("\n" + "\n".join(suggestions)) if suggestions else ""
        raise ValueError(
            "数据表缺少所选输出测量列，无法构建残差并进行拟合。\n"
            f"- 当前输出模式: {output_mode}\n"
            f"- 需要的列名: {missing_columns_text}\n"
            f"- 当前 CSV 列名（前 {CSV_COLUMN_PREVIEW_COUNT} 个）: {available_cols_text}\n"
            "提示：系统会自动去掉列名首尾空格；请优先使用「下载 CSV 模板」生成的表头。\n"
            "请检查：输出模式/物种选择是否正确，以及数据文件表头是否匹配。"
            f"{suggestion_text}"
        )

    # --- 必要输入列检查（避免所有行都“失败罚项”，看起来像卡住）---
    if reactor_type == REACTOR_TYPE_PFR:
        if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
            # 气相 PFR（理想气体、恒压 P、无压降）：入口强制使用 F0_*，并要求 P_Pa
            inlet_cols = [f"F0_{name}_mol_s" for name in species_names]
            required_input_columns = ["V_m3", "T_K", "P_Pa"] + inlet_cols
        else:
            # 液相 PFR（体积流量 vdot 近似恒定）：
            # 约定：当拟合目标为 Cout 时，入口也使用浓度 C0_*（并由 vdot 自动换算为 F0 参与计算）
            inlet_cols = (
                [f"C0_{name}_mol_m3" for name in species_names]
                if str(output_mode).startswith("C")
                else [f"F0_{name}_mol_s" for name in species_names]
            )
            required_input_columns = ["V_m3", "T_K", "vdot_m3_s"] + inlet_cols
    elif reactor_type == REACTOR_TYPE_CSTR:
        required_input_columns = ["V_m3", "T_K", "vdot_m3_s"] + [
            f"C0_{name}_mol_m3" for name in species_names
        ]
    else:
        required_input_columns = ["t_s", "T_K"] + [
            f"C0_{name}_mol_m3" for name in species_names
        ]

    missing_input_columns = [
        c for c in required_input_columns if c not in data_df.columns
    ]
    if missing_input_columns:
        missing_text = ", ".join(missing_input_columns)
        available_cols = [str(c) for c in list(data_df.columns)]
        available_cols_text = ", ".join(available_cols[:CSV_COLUMN_PREVIEW_COUNT])
        suggestions = []
        for missing_name in missing_input_columns:
            matches = difflib.get_close_matches(
                missing_name,
                available_cols,
                n=CSV_CLOSE_MATCHES_MAX,
                cutoff=CSV_CLOSE_MATCHES_CUTOFF,
            )
            if matches:
                suggestions.append(f"- `{missing_name}` 可能对应: {', '.join(matches)}")
        suggestion_text = ("\n" + "\n".join(suggestions)) if suggestions else ""
        raise ValueError(
            "数据表缺少必要输入列，无法进行模型计算与拟合。\n"
            f"- 反应器类型: {reactor_type}\n"
            f"- 缺少列名: {missing_text}\n"
            f"- 当前 CSV 列名（前 {CSV_COLUMN_PREVIEW_COUNT} 个）: {available_cols_text}\n"
            "请使用「下载 CSV 模板」生成的表头，或检查列名是否拼写一致（含单位后缀）。"
            f"{suggestion_text}"
        )

    n_data_rows = int(len(data_df))
    n_outputs = int(len(output_column_names))
    measured_matrix = np.zeros((n_data_rows, n_outputs), dtype=float)

    invalid_input_messages = []
    for column_name in required_input_columns:
        numeric_series = pd.to_numeric(data_df[column_name], errors="coerce")
        numeric_values = numeric_series.to_numpy(dtype=float)

        invalid_mask = ~np.isfinite(numeric_values)
        if bool(np.any(invalid_mask)):
            invalid_row_indices = data_df.index[invalid_mask].tolist()
            sample_indices_text = ", ".join(
                [str(i) for i in invalid_row_indices[:CSV_INVALID_INDEX_PREVIEW_COUNT]]
            )
            invalid_input_messages.append(
                f"- 列 `{column_name}` 含 NaN/非数字/无穷大：共 {len(invalid_row_indices)} 行"
                + (
                    f"（示例 index: {sample_indices_text}）"
                    if sample_indices_text
                    else ""
                )
            )
            continue

        # 基本物理约束（单位见模板列名）
        if column_name == "T_K":
            bad_mask = numeric_values <= 0.0
            bad_desc = "必须 > 0"
        elif column_name == "vdot_m3_s":
            bad_mask = numeric_values <= 0.0
            bad_desc = "必须 > 0"
        elif column_name == "P_Pa":
            bad_mask = numeric_values <= 0.0
            bad_desc = "必须 > 0"
        elif column_name == "V_m3":
            bad_mask = numeric_values < 0.0
            bad_desc = "不能为负"
        elif column_name == "t_s":
            bad_mask = numeric_values < 0.0
            bad_desc = "不能为负"
        else:
            # 入口变量：F0_* 或 C0_*
            bad_mask = numeric_values < 0.0
            bad_desc = "不能为负"

        if bool(np.any(bad_mask)):
            bad_row_indices = data_df.index[bad_mask].tolist()
            sample_indices_text = ", ".join(
                [str(i) for i in bad_row_indices[:CSV_INVALID_INDEX_PREVIEW_COUNT]]
            )
            invalid_input_messages.append(
                f"- 列 `{column_name}` {bad_desc}：共 {len(bad_row_indices)} 行"
                + (
                    f"（示例 index: {sample_indices_text}）"
                    if sample_indices_text
                    else ""
                )
            )

    invalid_value_messages = []
    for col_index, column_name in enumerate(output_column_names):
        numeric_series = pd.to_numeric(data_df[column_name], errors="coerce")
        numeric_values = numeric_series.to_numpy(dtype=float)
        measured_matrix[:, col_index] = numeric_values
        invalid_mask = ~np.isfinite(numeric_values)
        if bool(np.any(invalid_mask)):
            invalid_row_indices = data_df.index[invalid_mask].tolist()
            sample_indices_text = ", ".join([str(i) for i in invalid_row_indices[:10]])
            invalid_value_messages.append(
                f"- 列 `{column_name}` 含 NaN/非数字/无穷大：共 {len(invalid_row_indices)} 行"
                + (
                    f"（示例 index: {sample_indices_text}）"
                    if sample_indices_text
                    else ""
                )
            )

    if invalid_value_messages:
        raise ValueError(
            "所选输出测量列中存在 NaN/非数字值，拟合已停止（避免残差被静默当作 0）。\n"
            + "\n".join(invalid_value_messages)
            + "\n请清理数据（删除/填补缺失值，或取消选择对应输出物种/输出模式）后再拟合。"
        )

    if invalid_input_messages:
        if reactor_type == REACTOR_TYPE_PFR:
            pfr_hint = (
                "PFR(气相): V_m3, T_K, P_Pa, F0_*"
                if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P
                else "PFR(液相): V_m3, T_K, vdot_m3_s, F0_* 或（Cout 模式）C0_*"
            )
        else:
            pfr_hint = "PFR: V_m3, T_K, vdot_m3_s, F0_* 或（Cout 模式）C0_*"
        raise ValueError(
            "输入条件列存在 NaN/非数字/不合理值，拟合已停止。\n"
            + "\n".join(invalid_input_messages)
            + "\n请先修正输入条件列（"
            + pfr_hint
            + "；BSTR: t_s, T_K, C0_*；CSTR: V_m3, T_K, vdot_m3_s, C0_*），再开始拟合。"
        )

    typical_measured_scale = (
        float(np.nanmedian(np.abs(measured_matrix)))
        if measured_matrix.size > 0
        else 1.0
    )
    if (not np.isfinite(typical_measured_scale)) or (typical_measured_scale <= 0.0):
        typical_measured_scale = 1.0

    if (not np.isfinite(max_step_fraction)) or (max_step_fraction < 0.0):
        max_step_fraction = DEFAULT_MAX_STEP_FRACTION

    residual_penalty_value = float(
        max(
            DEFAULT_RESIDUAL_PENALTY_MULTIPLIER * typical_measured_scale,
            DEFAULT_RESIDUAL_PENALTY_MIN_ABS,
        )
    )
    set_metric("typical_measured_scale", typical_measured_scale)
    set_metric("residual_penalty_value", residual_penalty_value)
    timeline_add(
        "ℹ️",
        f"失败罚项：typical_scale≈{typical_measured_scale:.3e}，penalty={residual_penalty_value:.3e}",
    )
    timeline_add(
        "ℹ️", f"ODE 步长限制：max_step_fraction={max_step_fraction:.3g}（0 表示不限制）"
    )

    # 计算 epsilon（用于百分比残差，避免除零）
    residual_epsilon = float(
        typical_measured_scale * PERCENTAGE_RESIDUAL_EPSILON_FACTOR
    )  # 典型值的百分比
    if residual_epsilon < EPSILON_RELATIVE:
        residual_epsilon = EPSILON_RELATIVE

    # 残差类型信息
    residual_type_names = {
        "绝对残差": "Absolute: r = y_pred - y_meas",
        "相对残差": "Relative: r = (y_pred - y_meas) / sign(y_meas)·max(|y_meas|, ε)",
        "百分比残差": f"Percentage: r = 100 * (y_pred - y_meas) / (|y_meas| + ε), ε≈{residual_epsilon:.2e}",
    }
    residual_formula_for_summary = residual_type_names.get(
        residual_type, residual_type_names["绝对残差"]
    )
    timeline_add(
        "ℹ️",
        f"残差类型：{residual_type} — {residual_type_names.get(residual_type, '')}",
    )

    def _build_skipped_fit_result(skip_reason: str, phi_value: float) -> dict:
        phi_value = float(phi_value)
        set_metric("initial_cost", phi_value)
        set_metric("final_phi", phi_value)
        timeline_add("ℹ️", skip_reason)
        set_status("跳过拟合（直接使用初值计算）。")
        set_progress(1.0)
        set_final_summary(
            f"目标函数定义：Φ(θ)=1/2·∑ r_i(θ)^2\n"
            f"残差定义：{residual_formula_for_summary}\n"
            f"执行结果：未执行 least_squares（{skip_reason}）\n"
            f"目标函数值：Φ={phi_value:.3e}\n"
            f"失败罚项：typical_scale≈{typical_measured_scale:.3e}，penalty={residual_penalty_value:.3e}\n"
            f"ODE 步长限制：max_step_fraction={max_step_fraction:.3g}（0 表示不限制）"
        )

        fitted_params = fitting._unpack_parameters(
            param_vector,
            k0_guess,
            ea_guess_J_mol,
            order_guess,
            fit_k0_flags,
            fit_ea_flags,
            fit_order_flags_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            fit_K0_ads_flags,
            fit_Ea_K_flags,
            fit_m_flags,
            k0_rev,
            ea_rev_J_mol,
            order_rev,
            fit_k0_rev_flags,
            fit_ea_rev_flags,
            fit_order_rev_flags_matrix,
            use_log_k0_fit=use_log_k0_fit,
            use_log_k0_rev_fit=use_log_k0_rev_fit,
            use_log_K0_ads_fit=use_log_K0_ads_fit,
        )
        fit_state_snapshot = build_fit_state_snapshot(
            data_df=data_df,
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
            order_rev_min=float(ord_rev_min),
            order_rev_max=float(ord_rev_max),
        )

        return {
            "params": fitted_params,
            "data": data_df.copy(deep=True),
            "data_hash": hashlib.md5(data_df.to_csv(index=False).encode()).hexdigest(),
            "species_names": species_names,
            "output_mode": output_mode,
            "output_species": output_species_list,
            "stoich_matrix": stoich_matrix,
            "solver_method": solver_method,
            "rtol": float(rtol),
            "atol": float(atol),
            "max_step_fraction": float(max_step_fraction),
            "reactor_type": reactor_type,
            "kinetic_model": kinetic_model,
            "reversible_enabled": bool(reversible_enabled),
            "pfr_flow_model": str(pfr_flow_model),
            "use_log_k0_fit": bool(use_log_k0_fit),
            "use_log_k0_rev_fit": bool(use_log_k0_rev_fit),
            "use_log_K0_ads_fit": bool(use_log_K0_ads_fit),
            "fit_flags_hash": str(fit_state_snapshot.get("fit_flags_hash", "")),
            "fit_bounds_hash": str(fit_state_snapshot.get("fit_bounds_hash", "")),
            "initial_cost": phi_value,
            "cost": phi_value,
            "phi_initial": phi_value,
            "phi_final": phi_value,
            "residual_type": str(residual_type),
            "n_fit_params": int(n_fit_params_total),
            "fit_skipped": True,
            "fit_skipped_reason": str(skip_reason),
        }

    if n_data_rows <= 0:
        return _build_skipped_fit_result(
            "未执行拟合：当前数据表没有任何数据行（N=0），已按初值计算。",
            0.0,
        )

    if n_outputs <= 0:
        return _build_skipped_fit_result(
            "未执行拟合：未选择任何输出变量（N_outputs=0），已按初值计算。",
            0.0,
        )

    data_rows = list(data_df.itertuples(index=False))
    species_name_to_index = {name: i for i, name in enumerate(species_names)}
    try:
        output_species_indices = [
            species_name_to_index[name] for name in output_species_list
        ]
    except Exception:
        raise ValueError("输出物种不在物种列表中（请检查物种名是否匹配）")

    if reactor_type == REACTOR_TYPE_PFR:
        inlet_column_names = (
            [f"F0_{name}_mol_s" for name in species_names]
            if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P
            else (
                [f"C0_{name}_mol_m3" for name in species_names]
                if str(output_mode).startswith("C")
                else [f"F0_{name}_mol_s" for name in species_names]
            )
        )
    else:
        inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]

    # 行级并行：持久化线程池（避免每次残差调用重复创建线程）
    # scipy.integrate.solve_ivp 在 C 层释放 GIL，多线程可真正并行
    _n_row_workers = min(n_data_rows, os.cpu_count() or 4)
    _row_executor = ThreadPoolExecutor(max_workers=_n_row_workers)

    # 可变容差容器：允许粗拟合阶段使用放宽容差，精细拟合恢复原始容差
    _active_tol = {"rtol": float(rtol), "atol": float(atol)}

    # --- 基于函数评估次数（nfev）的进度跟踪 ---
    stage_label = "初始化"
    stage_base_progress = 0.0
    stage_span_progress = 0.05
    stage_max_nfev = 1
    stage_nfev = 0
    best_cost_so_far = float("inf")
    last_ui_update_s = time.time()
    n_params_fit: int | None = None
    valid_points_last_eval = 0
    max_valid_points_seen = 0

    def emit_stage_progress(row_index: int | None = None) -> None:
        nonlocal last_ui_update_s
        calls_per_iteration_est = int(max(int(n_params_fit or 0) + 1, 1))
        call_budget_est = int(max(int(stage_max_nfev), 1)) * calls_per_iteration_est

        if row_index is None:
            call_progress_est = float(stage_nfev)
            call_progress_text = str(int(stage_nfev))
            row_progress_text = ""
        else:
            row_frac = float(row_index + 1) / float(max(int(n_data_rows), 1))
            call_progress_est = float(max(int(stage_nfev) - 1, 0)) + row_frac
            call_progress_text = f"{call_progress_est:.1f}"
            row_progress_text = f" | 数据行 {int(row_index) + 1}/{int(n_data_rows)}"

        frac = float(call_progress_est) / float(max(call_budget_est, 1))
        frac = float(np.clip(frac, 0.0, 1.0))
        set_progress(stage_base_progress + stage_span_progress * frac)

        if np.isfinite(best_cost_so_far):
            set_status(
                f"{stage_label} | 调用≈{call_progress_text}/{int(call_budget_est)} "
                f"(max_iter={int(stage_max_nfev)}, n={int(n_params_fit or 0)})"
                f"{row_progress_text} | best Φ≈{best_cost_so_far:.3e}"
            )
        else:
            set_status(
                f"{stage_label} | 调用≈{call_progress_text}/{int(call_budget_est)} "
                f"(max_iter={int(stage_max_nfev)}, n={int(n_params_fit or 0)})"
                f"{row_progress_text}"
            )
        last_ui_update_s = time.time()

    def residual_func_wrapper(x: np.ndarray) -> np.ndarray:
        nonlocal stage_nfev, best_cost_so_far, last_ui_update_s, n_params_fit
        nonlocal valid_points_last_eval, max_valid_points_seen
        if stop_event.is_set():
            raise FittingStoppedError("Stopped by user")

        if n_params_fit is None:
            n_params_fit = int(np.asarray(x).size)

        stage_nfev += 1
        p = fitting._unpack_parameters(
            x,
            k0_guess,
            ea_guess_J_mol,
            order_guess,
            fit_k0_flags,
            fit_ea_flags,
            fit_order_flags_matrix,
            K0_ads,
            Ea_K_J_mol,
            m_inhibition,
            fit_K0_ads_flags,
            fit_Ea_K_flags,
            fit_m_flags,
            k0_rev,
            ea_rev_J_mol,
            order_rev,
            fit_k0_rev_flags,
            fit_ea_rev_flags,
            fit_order_rev_flags_matrix,
            use_log_k0_fit=use_log_k0_fit,
            use_log_k0_rev_fit=use_log_k0_rev_fit,
            use_log_K0_ads_fit=use_log_K0_ads_fit,
        )

        residual_array = np.zeros(n_data_rows * n_outputs, dtype=float)
        valid_points_this_eval = 0

        # 并行评估各数据行（solve_ivp 释放 GIL，_n_row_workers 线程真正并行执行）
        # 每个线程持有独立的 local_cache，避免共享字典的竞争条件
        def _eval_single_row(args: tuple) -> tuple:
            row_index, row = args
            if stop_event.is_set():
                return row_index, None, False
            local_cache: dict = {}
            pred, ok, _ = fitting._predict_outputs_for_row(
                row,
                species_names,
                output_mode,
                output_species_list,
                stoich_matrix,
                p["k0"],
                p["ea_J_mol"],
                p["reaction_order_matrix"],
                solver_method,
                _active_tol["rtol"],
                _active_tol["atol"],
                reactor_type,
                kinetic_model,
                reversible_enabled,
                pfr_flow_model,
                p["K0_ads"],
                p["Ea_K"],
                p["m_inhibition"],
                p["k0_rev"],
                p["ea_rev"],
                p["order_rev"],
                max_step_fraction=max_step_fraction,
                name_to_index=species_name_to_index,
                output_species_indices=output_species_indices,
                inlet_column_names=inlet_column_names,
                model_eval_cache=local_cache,
                stop_event=stop_event,
            )
            return row_index, pred, ok

        row_results = list(_row_executor.map(_eval_single_row, enumerate(data_rows)))

        if stop_event.is_set():
            raise FittingStoppedError("Stopped by user")

        for row_index, pred, ok in row_results:
            base = row_index * n_outputs
            if not ok or pred is None:
                residual_array[base : base + n_outputs] = residual_penalty_value
            else:
                measured_row = measured_matrix[row_index, :]
                diff = pred - measured_row

                # 根据残差类型计算残差
                if residual_type == "相对残差":
                    # 相对残差: r = (y_pred - y_meas) / sign(y_meas)·max(|y_meas|, ε)
                    # 说明：自动使用 epsilon，避免 y_meas≈0 时的除零问题。
                    sign_measured = np.where(measured_row < 0.0, -1.0, 1.0)
                    denominator = sign_measured * np.maximum(
                        np.abs(measured_row), residual_epsilon
                    )
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        rel_residual = diff / denominator
                    valid_mask = np.isfinite(rel_residual)
                    valid_points_this_eval += int(np.count_nonzero(valid_mask))
                    invalid_mask = ~np.isfinite(rel_residual)
                    if np.any(invalid_mask):
                        rel_residual[invalid_mask] = residual_penalty_value
                    residual_array[base : base + n_outputs] = rel_residual
                elif residual_type == "百分比残差":
                    # 百分比残差: r = 100 * (y_pred - y_meas) / (|y_meas| + epsilon)
                    denominator = np.abs(measured_row) + residual_epsilon
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        pct_residual = 100.0 * (diff / denominator)
                    valid_mask = np.isfinite(pct_residual)
                    valid_points_this_eval += int(np.count_nonzero(valid_mask))
                    invalid_mask = ~np.isfinite(pct_residual)
                    if np.any(invalid_mask):
                        pct_residual[invalid_mask] = residual_penalty_value
                    residual_array[base : base + n_outputs] = pct_residual
                else:
                    # 默认：绝对残差 r = y_pred - y_meas
                    valid_mask = np.isfinite(diff)
                    valid_points_this_eval += int(np.count_nonzero(valid_mask))
                    if not bool(np.all(valid_mask)):
                        residual_array[base : base + n_outputs] = residual_penalty_value
                    else:
                        residual_array[base : base + n_outputs] = diff

        if residual_array.size > 0:
            cost_now = float(0.5 * np.sum(residual_array**2))
            if np.isfinite(cost_now) and (cost_now < best_cost_so_far):
                best_cost_so_far = cost_now
        valid_points_last_eval = int(valid_points_this_eval)
        if valid_points_last_eval > int(max_valid_points_seen):
            max_valid_points_seen = int(valid_points_last_eval)

        now_s = time.time()
        if (now_s - last_ui_update_s) >= FITTING_UI_UPDATE_INTERVAL_S:
            emit_stage_progress(row_index=None)

        return residual_array

    timeline_add("⏳", "阶段 1: 计算初始残差...")
    stage_label = "初始残差"
    stage_base_progress = 0.0
    stage_span_progress = 0.05
    stage_max_nfev = 1
    stage_nfev = 0
    initial_residuals = residual_func_wrapper(param_vector)
    initial_cost = float(0.5 * np.sum(initial_residuals**2))
    set_metric("initial_cost", initial_cost)
    timeline_add("✅", f"初始目标函数值 Φ: {initial_cost:.4e}")

    if int(valid_points_last_eval) <= 0:
        raise ValueError(
            "没有任何有效预测点（N_valid=0），已阻断拟合。"
            " 请检查：求解器容限（rtol/atol）、入口流量/浓度、输出模式与工况是否匹配。"
        )

    if n_fit_params_total <= 0:
        _row_executor.shutdown(wait=False)
        return _build_skipped_fit_result(
            "未执行拟合：未勾选任何待拟合参数（n_fit_params=0），已按初值计算。",
            initial_cost,
        )

    set_status("开始 least_squares 拟合...")
    set_progress(0.05)

    best_res = None
    best_start_index = 1

    if use_ms and n_starts > 1:
        timeline_add("⏳", f"阶段 2: 多起点粗拟合 ({n_starts} 个起点)...")
        # 粗拟合使用放宽 10 倍的 ODE 容差：目的仅为筛选最优起点，无需高精度
        _active_tol["rtol"] = float(rtol) * 10.0
        _active_tol["atol"] = float(atol) * 10.0

        # Latin Hypercube Sampling（LHS）替代纯随机：在高维空间中更均匀地覆盖参数区间
        from scipy.stats.qmc import LatinHypercube

        lhs_sampler = LatinHypercube(d=len(lb), seed=random_seed)
        lhs_samples = lhs_sampler.random(n=n_starts - 1)  # shape (n_starts-1, d)
        lhs_scaled = lb + (ub - lb) * lhs_samples  # 缩放到 [lb, ub]

        starts = [param_vector]
        for i in range(n_starts - 1):
            rand_vec = np.clip(lhs_scaled[i], lb, ub)
            starts.append(rand_vec)

        ms_results_log: list[dict] = []  # 记录每个起点的结果

        for start_index, x0 in enumerate(starts):
            if stop_event.is_set():
                raise FittingStoppedError("Stopped by user")

            set_status(f"多起点：第 {start_index+1}/{n_starts} 个起点...")
            stage_label = f"多起点粗拟合 {start_index+1}/{n_starts}"
            stage_base_progress = 0.05 + 0.75 * (start_index / max(int(n_starts), 1))
            stage_span_progress = 0.75 / max(int(n_starts), 1)
            stage_max_nfev = int(max_nfev_coarse)
            stage_nfev = 0
            set_progress(stage_base_progress)

            res = least_squares(
                residual_func_wrapper,
                x0,
                bounds=(lb, ub),
                method="trf",
                diff_step=diff_step_rel,
                max_nfev=max_nfev_coarse,
                x_scale="jac" if use_x_scale_jac else 1.0,
            )
            ms_results_log.append({
                "start": start_index + 1,
                "cost": float(res.cost),
                "nfev": int(res.nfev),
                "status": str(res.message),
                "is_best": False,
            })
            if best_res is None or res.cost < best_res.cost:
                best_res = res
                best_start_index = int(start_index + 1)

        if best_res is None:
            raise RuntimeError("多起点（Multi-start）执行失败：未获得有效结果。")

        # 标记最佳起点
        for entry in ms_results_log:
            if entry["start"] == best_start_index:
                entry["is_best"] = True

        # 保存多起点对比数据到 session_state
        progress_queue.put(("ms_results_log", ms_results_log))

        if stop_event.is_set():
            raise FittingStoppedError("Stopped by user")

        coarse_best_phi = float(best_res.cost)
        set_metric("coarse_best_phi", coarse_best_phi)
        timeline_add(
            "✅",
            f"粗拟合完成，最佳起点: {best_start_index}/{n_starts}，最佳 Φ: {coarse_best_phi:.4e}",
        )
        set_ms_summary(
            f"多起点数量：{int(n_starts)}（seed={int(random_seed)}）\n"
            f"粗拟合预算：max_nfev={int(max_nfev_coarse)}\n"
            f"最佳起点：{int(best_start_index)}/{int(n_starts)}\n"
            f"粗拟合最佳 Φ：{coarse_best_phi:.3e}"
        )

        # 精细拟合恢复原始 ODE 容差，保证收敛精度
        _active_tol["rtol"] = float(rtol)
        _active_tol["atol"] = float(atol)

        set_status("使用最优起点做精细拟合...")
        set_progress(0.85)
        timeline_add(
            "⏳",
            f"阶段 3: 精细拟合 (从最佳起点 {best_start_index}/{n_starts} 开始, 初始 Φ: {float(best_res.cost):.4e})...",
        )
        stage_label = "精细拟合"
        stage_base_progress = 0.85
        stage_span_progress = 0.15
        stage_max_nfev = int(max_nfev)
        stage_nfev = 0
        final_res = least_squares(
            residual_func_wrapper,
            best_res.x,
            bounds=(lb, ub),
            method="trf",
            diff_step=diff_step_rel,
            max_nfev=max_nfev,
            x_scale="jac" if use_x_scale_jac else 1.0,
        )
    else:
        set_status("单起点拟合中...")
        set_progress(0.3)
        timeline_add("⏳", "阶段 2: 单起点拟合...")
        stage_label = "单起点拟合"
        stage_base_progress = 0.05
        stage_span_progress = 0.95
        stage_max_nfev = int(max_nfev)
        stage_nfev = 0
        final_res = least_squares(
            residual_func_wrapper,
            param_vector,
            bounds=(lb, ub),
            method="trf",
            diff_step=diff_step_rel,
            max_nfev=max_nfev,
            x_scale="jac" if use_x_scale_jac else 1.0,
        )

    if stop_event.is_set():
        raise FittingStoppedError("Stopped by user")

    if int(getattr(final_res, "status", 0)) == 0:
        timeline_add(
            "⚠️", "达到最大迭代次数上限，拟合提前停止（可增大最大迭代次数 max_nfev）。"
        )

    if int(max_valid_points_seen) <= 0:
        raise ValueError(
            "拟合全过程没有任何有效预测点（N_valid=0），结果无效并已阻断。"
        )

    final_phi = float(final_res.cost)
    set_metric("final_phi", final_phi)
    timeline_add("✅", f"精细拟合完成，最终 Φ: {final_phi:.4e}")

    phi_ratio = float(final_phi / max(initial_cost, FITTING_EPSILON_PHI_RATIO))
    param_relative_change = float(
        np.linalg.norm(final_res.x - param_vector)
        / (np.linalg.norm(param_vector) + FITTING_EPSILON_NORM)
    )
    set_metric("phi_ratio", phi_ratio)
    set_metric("param_relative_change", param_relative_change)
    set_final_summary(
        f"目标函数定义：Φ(θ)=1/2·∑ r_i(θ)^2\n"
        f"残差定义：{residual_formula_for_summary}\n"
        f"Φ 变化：初始 {initial_cost:.3e} -> 拟合 {final_phi:.3e}（比例 {phi_ratio:.3e}）\n"
        f"参数相对变化：{param_relative_change:.3e}\n"
        f"失败罚项：typical_scale≈{typical_measured_scale:.3e}，penalty={residual_penalty_value:.3e}\n"
        f"ODE 步长限制：max_step_fraction={max_step_fraction:.3g}（0 表示不限制）"
    )

    set_status("解包并保存拟合结果...")
    set_progress(0.95)

    fitted_params = fitting._unpack_parameters(
        final_res.x,
        k0_guess,
        ea_guess_J_mol,
        order_guess,
        fit_k0_flags,
        fit_ea_flags,
        fit_order_flags_matrix,
        K0_ads,
        Ea_K_J_mol,
        m_inhibition,
        fit_K0_ads_flags,
        fit_Ea_K_flags,
        fit_m_flags,
        k0_rev,
        ea_rev_J_mol,
        order_rev,
        fit_k0_rev_flags,
        fit_ea_rev_flags,
        fit_order_rev_flags_matrix,
        use_log_k0_fit=use_log_k0_fit,
        use_log_k0_rev_fit=use_log_k0_rev_fit,
        use_log_K0_ads_fit=use_log_K0_ads_fit,
    )
    fit_state_snapshot = build_fit_state_snapshot(
        data_df=data_df,
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
        order_rev_min=float(ord_rev_min),
        order_rev_max=float(ord_rev_max),
    )

    set_status("拟合完成。")
    set_progress(1.0)
    _row_executor.shutdown(wait=False)

    return {
        "params": fitted_params,
        "data": data_df.copy(deep=True),
        "data_hash": hashlib.md5(data_df.to_csv(index=False).encode()).hexdigest(),
        "species_names": species_names,
        "output_mode": output_mode,
        "output_species": output_species_list,
        "stoich_matrix": stoich_matrix,
        "solver_method": solver_method,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_step_fraction": float(max_step_fraction),
        "reactor_type": reactor_type,
        "kinetic_model": kinetic_model,
        "reversible_enabled": bool(reversible_enabled),
        "pfr_flow_model": str(pfr_flow_model),
        "use_log_k0_fit": bool(use_log_k0_fit),
        "use_log_k0_rev_fit": bool(use_log_k0_rev_fit),
        "use_log_K0_ads_fit": bool(use_log_K0_ads_fit),
        "fit_flags_hash": str(fit_state_snapshot.get("fit_flags_hash", "")),
        "fit_bounds_hash": str(fit_state_snapshot.get("fit_bounds_hash", "")),
        # 向后兼容的键名
        "initial_cost": float(initial_cost),
        "cost": float(final_res.cost),
        # 推荐使用的目标函数字段名
        "phi_initial": float(initial_cost),
        "phi_final": float(final_res.cost),
        "residual_type": str(residual_type),
        "n_fit_params": int(n_fit_params_total),
    }
