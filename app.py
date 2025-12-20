from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import html as html_lib
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

import modules.fitting as fitting
import modules.reactors as reactors
import modules.ui_help as ui_help
import modules.config_manager as config_manager
import modules.ui_components as ui_comp  # New module


class FittingStoppedError(Exception):
    pass


# 固定默认值（不在 UI 中暴露给用户）
DEFAULT_RESIDUAL_PENALTY_MULTIPLIER = 1e3
DEFAULT_RESIDUAL_PENALTY_MIN_ABS = 1e3


# ========== Helper functions ==========
def _read_csv_bytes_cached(uploaded_bytes: bytes) -> pd.DataFrame:
    """
    从 bytes 读取 CSV。
    """
    return pd.read_csv(io.BytesIO(uploaded_bytes))


def _apply_imported_config_to_widget_state(config: dict) -> None:
    """
    将导入配置写入 widget 对应的 session_state。

    关键点：必须在 widget 创建之前写入，否则会触发
    “cannot be modified after the widget ... is instantiated”。
    """
    reactor_type_cfg = str(config.get("reactor_type", "")).strip()
    kinetic_model_cfg = str(config.get("kinetic_model", "")).strip()
    solver_method_cfg = str(config.get("solver_method", "")).strip()

    if reactor_type_cfg in ["PFR", "Batch"]:
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
    if reactor_type_cfg == "Batch":
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


def _get_persist_dir() -> str:
    """
    本地持久化目录：用于跨刷新恢复（只保留一份缓存，新内容覆盖旧内容）。
    """
    persist_dir = os.path.join(tempfile.gettempdir(), "Kinetics_app_persist")
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def _atomic_write_bytes(file_path: str, data: bytes) -> None:
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="tmp_", suffix=".bin", dir=dir_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temp_path, file_path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


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


def _get_upload_file_paths() -> tuple[str, str]:
    """
    Returns:
        (csv_bytes_path, meta_json_path)

    说明：只保留“一份”上传缓存，新内容覆盖旧内容。
    """
    persist_dir = _get_persist_dir()
    csv_path = os.path.join(persist_dir, "uploaded.csv")
    meta_path = os.path.join(persist_dir, "uploaded.meta.json")
    return csv_path, meta_path


def _load_persisted_upload() -> tuple[bytes | None, str | None, str]:
    """
    Returns:
        (uploaded_csv_bytes, uploaded_csv_name, message)
    """
    csv_path, meta_path = _get_upload_file_paths()
    if not os.path.exists(csv_path):
        return None, None, "未找到已缓存上传文件"

    try:
        uploaded_bytes = open(csv_path, "rb").read()
    except Exception as exc:
        return None, None, f"读取缓存 CSV 失败: {exc}"

    uploaded_name = ""
    if os.path.exists(meta_path):
        try:
            meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
            uploaded_name = str(meta.get("name", "")).strip()
        except Exception:
            uploaded_name = ""

    if not uploaded_bytes:
        return None, None, "缓存 CSV 为空"
    return uploaded_bytes, uploaded_name, "OK"


def _save_persisted_upload(uploaded_bytes: bytes, uploaded_name: str) -> tuple[bool, str]:
    csv_path, meta_path = _get_upload_file_paths()
    try:
        _atomic_write_bytes(csv_path, uploaded_bytes)
        meta = {"name": str(uploaded_name).strip(), "saved_at_unix_s": float(time.time())}
        _atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, "OK"
    except Exception as exc:
        return False, f"缓存上传文件失败: {exc}"


def _delete_persisted_upload() -> tuple[bool, str]:
    csv_path, meta_path = _get_upload_file_paths()
    try:
        for path in [csv_path, meta_path]:
            if os.path.exists(path):
                os.remove(path)
        return True, "OK"
    except Exception as exc:
        return False, f"删除缓存上传文件失败: {exc}"


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
            # 清空 file_uploader 状态，避免“重置后又自动把同一个 JSON 重新导入”
            "uploaded_config_json",
        ]
    )

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


def _get_fitting_executor() -> ThreadPoolExecutor:
    """
    每个会话单独的线程池（避免跨会话共享导致“任务占用/卡住”）。
    """
    executor = st.session_state.get("fitting_executor", None)
    is_shutdown = bool(getattr(executor, "_shutdown", False)) if executor is not None else True
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
    st.session_state["fitting_final_summary"] = ""


def _render_fitting_overview_box(job_summary: dict) -> None:
    lines = job_summary.get("lines", [])
    if not lines:
        return

    title = html_lib.escape(str(job_summary.get("title", "📊 拟合任务概览")))
    bullet_html = "\n".join([f"<li>{html_lib.escape(str(x))}</li>" for x in lines])
    st.markdown(
        f"""
        <div style="background:#eff6ff; border:1px solid #dbeafe; padding:16px 18px; border-radius:14px;">
          <div style="font-weight:700; color:#1e40af; margin-bottom:8px;">{title}</div>
          <ul style="margin:0 0 0 18px; color:#0f172a; line-height:1.7;">
            {bullet_html}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_fitting_progress_panel() -> None:
    job_summary = st.session_state.get("fitting_job_summary", {})
    timeline = st.session_state.get("fitting_timeline", [])
    ms_summary = str(st.session_state.get("fitting_ms_summary", "")).strip()
    final_summary = str(st.session_state.get("fitting_final_summary", "")).strip()

    if job_summary:
        _render_fitting_overview_box(job_summary)

    if timeline:
        st.write("")
        for icon, text in timeline:
            if str(text).strip():
                st.markdown(f"{icon} {text}")

    if ms_summary:
        st.write("")
        with st.container(border=True):
            st.code(ms_summary, language="text")

    if final_summary:
        st.caption(final_summary)


def _render_fitting_live_progress() -> None:
    """
    只刷新“进度显示”区域，避免整页闪烁。

    说明：当后台拟合完成时，会触发一次全局 rerun 以渲染最终结果。
    """
    _drain_fitting_progress_queue()

    fitting_future = st.session_state.get("fitting_future", None)
    fitting_running = bool(st.session_state.get("fitting_running", False))

    if fitting_running and (fitting_future is not None) and fitting_future.done():
        st.rerun(scope="app")

    if not fitting_running:
        return

    st.info("拟合正在后台运行中（页面可继续操作）。")
    st.progress(float(st.session_state.get("fitting_progress", 0.0)))
    status_text = str(st.session_state.get("fitting_status", "")).strip()
    if status_text:
        st.caption(status_text)

    _render_fitting_progress_panel()

    if not bool(st.session_state.get("fitting_auto_refresh", True)):
        st.button("🔄 刷新进度", use_container_width=True, key="fit_manual_refresh_progress")


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

    use_ms = job_inputs["use_ms"]
    n_starts = job_inputs["n_starts"]
    random_seed = job_inputs["random_seed"]
    max_nfev = job_inputs["max_nfev"]
    max_nfev_coarse = job_inputs["max_nfev_coarse"]
    diff_step_rel = job_inputs["diff_step_rel"]
    use_x_scale_jac = job_inputs["use_x_scale_jac"]

    k0_min = job_inputs["k0_min"]
    k0_max = job_inputs["k0_max"]
    ea_min = job_inputs["ea_min"]
    ea_max = job_inputs["ea_max"]
    ord_min = job_inputs["ord_min"]
    ord_max = job_inputs["ord_max"]
    max_step_fraction = float(job_inputs.get("max_step_fraction", 0.1))

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
    )

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
        0,
        1e10,
        -2e5,
        2e5,
        0,
        5,
        fit_k0_rev_flags,
        fit_ea_rev_flags,
        fit_order_rev_flags_matrix,
        k0_min,
        k0_max,
        ea_min,
        ea_max,
        ord_min,
        ord_max,
    )

    output_column_names = []
    for species_name in output_species_list:
        if output_mode.startswith("F"):
            column_name = f"Fout_{species_name}_mol_s"
        elif output_mode.startswith("C"):
            column_name = f"Cout_{species_name}_mol_m3"
        else:
            column_name = f"X_{species_name}"
        output_column_names.append(column_name)

    missing_output_columns = []
    for column_name in output_column_names:
        if column_name not in data_df.columns:
            missing_output_columns.append(column_name)

    if missing_output_columns:
        missing_columns_text = ", ".join(missing_output_columns)
        raise ValueError(
            "数据表缺少所选输出测量列，无法构建残差并进行拟合。\n"
            f"- 当前输出模式: {output_mode}\n"
            f"- 需要的列名: {missing_columns_text}\n"
            "请检查：输出模式/物种选择是否正确，以及数据文件表头是否匹配。"
        )

    n_data_rows = int(len(data_df))
    n_outputs = int(len(output_column_names))
    measured_matrix = np.zeros((n_data_rows, n_outputs), dtype=float)

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
                + (f"（示例 index: {sample_indices_text}）" if sample_indices_text else "")
            )

    if invalid_value_messages:
        raise ValueError(
            "所选输出测量列中存在 NaN/非数字值，拟合已停止（避免残差被静默当作 0）。\n"
            + "\n".join(invalid_value_messages)
            + "\n请清理数据（删除/填补缺失值，或取消选择对应输出物种/输出模式）后再拟合。"
        )

    typical_measured_scale = float(np.nanmedian(np.abs(measured_matrix))) if measured_matrix.size > 0 else 1.0
    if (not np.isfinite(typical_measured_scale)) or (typical_measured_scale <= 0.0):
        typical_measured_scale = 1.0

    if (not np.isfinite(max_step_fraction)) or (max_step_fraction < 0.0):
        max_step_fraction = 0.1

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
    timeline_add("ℹ️", f"ODE 步长限制：max_step_fraction={max_step_fraction:.3g}（0 表示不限制）")

    data_rows = list(data_df.itertuples(index=False))
    species_name_to_index = {name: i for i, name in enumerate(species_names)}
    try:
        output_species_indices = [species_name_to_index[name] for name in output_species_list]
    except Exception:
        raise ValueError("输出物种不在物种列表中（请检查物种名是否匹配）")

    if reactor_type == "PFR":
        inlet_column_names = [f"F0_{name}_mol_s" for name in species_names]
    else:
        inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]

    # --- Progress tracking based on function evaluations (nfev) ---
    stage_label = "初始化"
    stage_base_progress = 0.0
    stage_span_progress = 0.05
    stage_max_nfev = 1
    stage_nfev = 0
    best_cost_so_far = float("inf")
    last_ui_update_s = time.time()
    n_params_fit: int | None = None

    def residual_func_wrapper(x: np.ndarray) -> np.ndarray:
        nonlocal stage_nfev, best_cost_so_far, last_ui_update_s, n_params_fit
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
        )

        residual_array = np.zeros(n_data_rows * n_outputs, dtype=float)
        model_eval_cache: dict = {}
        for row_index, row in enumerate(data_rows):
            if stop_event.is_set():
                raise FittingStoppedError("Stopped by user")

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
                rtol,
                atol,
                reactor_type,
                kinetic_model,
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
                model_eval_cache=model_eval_cache,
            )

            base = row_index * n_outputs
            if not ok:
                residual_array[base : base + n_outputs] = residual_penalty_value
            else:
                diff = pred - measured_matrix[row_index, :]
                if not bool(np.all(np.isfinite(diff))):
                    residual_array[base : base + n_outputs] = residual_penalty_value
                else:
                    residual_array[base : base + n_outputs] = diff
        if residual_array.size > 0:
            cost_now = float(0.5 * np.sum(residual_array**2))
            if np.isfinite(cost_now) and (cost_now < best_cost_so_far):
                best_cost_so_far = cost_now

        now_s = time.time()
        if (now_s - last_ui_update_s) >= 0.6:
            calls_per_iteration_est = int(max(int(n_params_fit or 0) + 1, 1))
            call_budget_est = int(max(int(stage_max_nfev), 1)) * calls_per_iteration_est
            frac = float(stage_nfev) / float(max(call_budget_est, 1))
            frac = float(np.clip(frac, 0.0, 1.0))
            set_progress(stage_base_progress + stage_span_progress * frac)
            if np.isfinite(best_cost_so_far):
                set_status(
                    f"{stage_label} | 调用≈{int(stage_nfev)}/{int(call_budget_est)} "
                    f"(max_iter={int(stage_max_nfev)}, n={int(n_params_fit)}) | best Φ≈{best_cost_so_far:.3e}"
                )
            else:
                set_status(
                    f"{stage_label} | 调用≈{int(stage_nfev)}/{int(call_budget_est)} "
                    f"(max_iter={int(stage_max_nfev)}, n={int(n_params_fit)})"
                )
            last_ui_update_s = now_s

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

    set_status("开始 least_squares 拟合...")
    set_progress(0.05)

    best_res = None
    best_start_index = 1

    if use_ms and n_starts > 1:
        timeline_add("⏳", f"阶段 2: 多起点粗拟合 ({n_starts} 个起点)...")
        rng = np.random.default_rng(random_seed)
        starts = [param_vector]
        for _ in range(n_starts - 1):
            rand_vec = lb + (ub - lb) * rng.random(len(lb))
            rand_vec = np.clip(rand_vec, lb, ub)
            starts.append(rand_vec)

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
            if best_res is None or res.cost < best_res.cost:
                best_res = res
                best_start_index = int(start_index + 1)

        if best_res is None:
            raise RuntimeError("Multi-start failed: no valid result.")

        if stop_event.is_set():
            raise FittingStoppedError("Stopped by user")

        coarse_best_phi = float(best_res.cost)
        set_metric("coarse_best_phi", coarse_best_phi)
        timeline_add(
            "✅",
            f"粗拟合完成，最佳起点: {best_start_index}/{n_starts}，最佳 Φ: {coarse_best_phi:.4e}",
        )
        set_ms_summary(
            f"multi-start: n_starts={n_starts}, seed={random_seed}, coarse max_nfev={max_nfev_coarse}, "
            f"best_start={best_start_index}/{n_starts}"
        )

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
        timeline_add("⚠️", "达到最大迭代次数上限，拟合提前停止（可增大 Max Iterations）。")

    final_phi = float(final_res.cost)
    set_metric("final_phi", final_phi)
    timeline_add("✅", f"精细拟合完成，最终 Φ: {final_phi:.4e}")

    phi_ratio = float(final_phi / max(initial_cost, 1e-300))
    param_relative_change = float(
        np.linalg.norm(final_res.x - param_vector) / (np.linalg.norm(param_vector) + 1e-12)
    )
    set_metric("phi_ratio", phi_ratio)
    set_metric("param_relative_change", param_relative_change)
    set_final_summary(
        "目标函数：Φ(θ)=1/2·∑ r_i(θ)^2，r_i=y_i^pred−y_i^meas。\n"
        f"Φ：初始 {initial_cost:.3e} -> 拟合 {final_phi:.3e} (比例 {phi_ratio:.3e}); "
        f"参数相对变化 {param_relative_change:.3e}\n"
        f"失败罚项：typical_scale≈{typical_measured_scale:.3e}, penalty={residual_penalty_value:.3e}\n"
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
    )

    set_status("拟合完成。")
    set_progress(1.0)

    return {
        "params": fitted_params,
        "data": data_df,
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
        # Backward compatible keys
        "initial_cost": float(initial_cost),
        "cost": float(final_res.cost),
        # Preferred objective naming
        "phi_initial": float(initial_cost),
        "phi_final": float(final_res.cost),
    }
def _clean_species_names(species_text: str) -> list[str]:
    parts = [p.strip() for p in species_text.split(",")]
    names = [p for p in parts if p]
    unique = []
    for n in names:
        if n not in unique:
            unique.append(n)
    return unique


def _build_default_nu_table(species_names: list[str], n_reactions: int) -> pd.DataFrame:
    nu_default = pd.DataFrame(
        data=np.zeros((len(species_names), n_reactions), dtype=float),
        index=species_names,
        columns=[f"R{j+1}" for j in range(n_reactions)],
    )
    if n_reactions >= 1:
        if "A" in species_names:
            nu_default.loc["A", "R1"] = -1.0
        if "B" in species_names:
            nu_default.loc["B", "R1"] = 1.0
    return nu_default


def _get_measurement_column_name(output_mode: str, species_name: str) -> str:
    output_mode = str(output_mode).strip()
    if output_mode.startswith("F"):
        return f"Fout_{species_name}_mol_s"
    if output_mode.startswith("C"):
        return f"Cout_{species_name}_mol_m3"
    return f"X_{species_name}"


def _get_output_unit_text(output_mode: str) -> str:
    output_mode = str(output_mode).strip()
    if output_mode.startswith("F"):
        return "mol/s"
    if output_mode.startswith("C"):
        return "mol/m^3"
    return "-"


def _build_fit_comparison_long_table(
    data_df: pd.DataFrame,
    species_names: list[str],
    output_mode: str,
    output_species_list: list[str],
    stoich_matrix: np.ndarray,
    fitted_params: dict,
    solver_method: str,
    rtol: float,
    atol: float,
    reactor_type: str,
    kinetic_model: str,
    max_step_fraction: float = 0.1,
) -> pd.DataFrame:
    rows = []
    row_indices = data_df.index.to_numpy()
    output_column_names = [_get_measurement_column_name(output_mode, sp) for sp in output_species_list]
    measured_matrix = np.zeros((len(data_df), len(output_column_names)), dtype=float)
    for col_index, column_name in enumerate(output_column_names):
        measured_matrix[:, col_index] = pd.to_numeric(data_df[column_name], errors="coerce").to_numpy(dtype=float)

    species_name_to_index = {name: i for i, name in enumerate(species_names)}
    output_species_indices = [species_name_to_index[name] for name in output_species_list]
    if reactor_type == "PFR":
        inlet_column_names = [f"F0_{name}_mol_s" for name in species_names]
    else:
        inlet_column_names = [f"C0_{name}_mol_m3" for name in species_names]

    model_eval_cache: dict = {}
    data_rows = list(data_df.itertuples(index=False))
    for row_pos, row in enumerate(data_rows):
        row_index = row_indices[row_pos]
        pred_vals, ok, message = fitting._predict_outputs_for_row(
            row,
            species_names,
            output_mode,
            output_species_list,
            stoich_matrix,
            fitted_params["k0"],
            fitted_params["ea_J_mol"],
            fitted_params["reaction_order_matrix"],
            solver_method,
            rtol,
            atol,
            reactor_type,
            kinetic_model,
            fitted_params.get("K0_ads", None),
            fitted_params.get("Ea_K", None),
            fitted_params.get("m_inhibition", None),
            fitted_params.get("k0_rev", None),
            fitted_params.get("ea_rev", None),
            fitted_params.get("order_rev", None),
            max_step_fraction=max_step_fraction,
            name_to_index=species_name_to_index,
            output_species_indices=output_species_indices,
            inlet_column_names=inlet_column_names,
            model_eval_cache=model_eval_cache,
        )

        for out_i, species_name in enumerate(output_species_list):
            measured_value = float(measured_matrix[row_pos, out_i])
            predicted_value = float(pred_vals[out_i]) if ok else np.nan
            residual_value = predicted_value - measured_value
            rows.append(
                {
                    "row_index": row_index,
                    "species": species_name,
                    "measured": measured_value,
                    "predicted": predicted_value,
                    "residual": residual_value,
                    "ok": bool(ok),
                    "message": str(message),
                }
            )

    return pd.DataFrame(rows)


# ========== Main App ==========
def main():
    st.set_page_config(
        page_title="Kinetics_app | 反应动力学拟合", layout="wide", page_icon="⚗️"
    )

    # --- 重置为默认：延迟到“下一次 rerun”再清理（避免修改已创建 widget 的 session_state）---
    if bool(st.session_state.pop("pending_reset_to_default", False)):
        ok, message = config_manager.clear_auto_saved_config()
        if not ok:
            st.warning(message)
        _clear_config_related_state()
        st.success("已重置为默认配置。")

    # --- 手动导入配置：延迟到“下一次 rerun”再应用（避免修改已创建 widget 的 session_state）---
    if "pending_imported_config" in st.session_state:
        pending_cfg = st.session_state.pop("pending_imported_config")
        is_valid, error_message = config_manager.validate_config(pending_cfg)
        if not is_valid:
            st.error(f"导入配置失败（配置校验未通过）：{error_message}")
        else:
            st.session_state["imported_config"] = pending_cfg
            _apply_imported_config_to_widget_state(pending_cfg)
            ok, message = config_manager.auto_save_config(pending_cfg)
            if not ok:
                st.warning(message)

    # --- Auto Load Config ---
    if "config_initialized" not in st.session_state:
        st.session_state["config_initialized"] = True
        saved_config, load_message = config_manager.auto_load_config()
        if saved_config is not None:
            is_valid, error_message = config_manager.validate_config(saved_config)
            if is_valid:
                st.session_state["imported_config"] = saved_config
            else:
                st.warning(f"自动恢复配置无效，已忽略：{error_message}")
        else:
            if str(load_message).startswith("自动加载失败"):
                st.warning(load_message)

    def get_cfg(key, default):
        if "imported_config" in st.session_state:
            return st.session_state["imported_config"].get(key, default)
        return default

    # Initialize fitting stopped flag
    if "fitting_stopped" not in st.session_state:
        st.session_state.fitting_stopped = False
    if "fitting_running" not in st.session_state:
        st.session_state["fitting_running"] = False
    if "fitting_future" not in st.session_state:
        st.session_state["fitting_future"] = None
    if "fitting_stop_event" not in st.session_state:
        st.session_state["fitting_stop_event"] = threading.Event()
    if "fitting_progress_queue" not in st.session_state:
        st.session_state["fitting_progress_queue"] = queue.Queue()
    if "fitting_progress" not in st.session_state:
        st.session_state["fitting_progress"] = 0.0
    if "fitting_status" not in st.session_state:
        st.session_state["fitting_status"] = ""
    if "fitting_timeline" not in st.session_state:
        st.session_state["fitting_timeline"] = []
    if "fitting_metrics" not in st.session_state:
        st.session_state["fitting_metrics"] = {}
    if "fitting_job_summary" not in st.session_state:
        st.session_state["fitting_job_summary"] = {}
    if "fitting_ms_summary" not in st.session_state:
        st.session_state["fitting_ms_summary"] = ""
    if "fitting_final_summary" not in st.session_state:
        st.session_state["fitting_final_summary"] = ""
    if "fitting_auto_refresh" not in st.session_state:
        st.session_state["fitting_auto_refresh"] = True
    if "fitting_executor" not in st.session_state:
        st.session_state["fitting_executor"] = None

    # --- Restore cached uploaded CSV (persist across browser refresh) ---
    if "uploaded_csv_bytes" not in st.session_state:
        uploaded_bytes, uploaded_name, message = _load_persisted_upload()
        if uploaded_bytes is not None:
            st.session_state["uploaded_csv_bytes"] = uploaded_bytes
            st.session_state["uploaded_csv_name"] = uploaded_name or ""
        else:
            if (message != "未找到已缓存上传文件") and (
                "upload_restore_warned" not in st.session_state
            ):
                st.session_state["upload_restore_warned"] = True
                st.warning(message)
    if "uploaded_csv_name" not in st.session_state:
        st.session_state["uploaded_csv_name"] = ""

    if "data_df_cached" not in st.session_state:
        try:
            if "uploaded_csv_bytes" in st.session_state and st.session_state["uploaded_csv_bytes"]:
                st.session_state["data_df_cached"] = _read_csv_bytes_cached(
                    st.session_state["uploaded_csv_bytes"]
                )
        except Exception as exc:
            if "data_restore_warned" not in st.session_state:
                st.session_state["data_restore_warned"] = True
                st.warning(f"恢复缓存 CSV 失败（请重新上传）：{exc}")

    # --- CSS Styles ---
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
          font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Noto Sans", "Liberation Sans", sans-serif;
          color: #1e293b;
          font-size: 15px;
        }
        .block-container { padding-top: 2rem; padding-bottom: 5rem; max-width: 1400px; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #f1f5f9; }
        h1, h2, h3 { background: linear-gradient(120deg, #0f172a, #334155); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid #e2e8f0; }
        .stTabs [data-baseweb="tab"] { font-weight: 600; color: #64748b; }
        .stTabs [aria-selected="true"] { color: #4f46e5; border-bottom: 2px solid #4f46e5; }
        .stMarkdown p, .stMarkdown li { font-size: 0.98rem; line-height: 1.65; }
        .stCaption, .stAlert p { font-size: 0.92rem; }
        .stButton > button, .stDownloadButton > button { font-size: 0.95rem; }
        div[role="dialog"][aria-modal="true"] { width: 92vw !important; max-width: 1200px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Plot Style ---
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "axes.edgecolor": "#e2e8f0",
            "axes.labelcolor": "#475569",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "text.color": "#1e293b",
            "grid.color": "#f1f5f9",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )

    @st.dialog("教程/帮助")
    def _show_help_dialog() -> None:
        ui_help.render_help_page()

    # ========== Sidebar ==========
    export_config_placeholder = None
    with st.sidebar:
        st.markdown("### 全局设置")
        global_disabled = bool(st.session_state.get("fitting_running", False))
        with st.container(border=True):
            help_btn = st.button("📖 教程/帮助", use_container_width=True)
            if help_btn:
                _show_help_dialog()

        with st.container(border=True):
            st.caption("修改任意选项会自动应用（Streamlit 会自动 rerun）。")
            st.markdown("#### 核心模型")
            reactor_type = st.selectbox(
                "反应器",
                ["PFR", "Batch"],
                index=0 if get_cfg("reactor_type", "PFR") == "PFR" else 1,
                key="cfg_reactor_type",
                disabled=global_disabled,
            )
            kinetic_model = st.selectbox(
                "动力学",
                ["power_law", "langmuir_hinshelwood", "reversible"],
                index=["power_law", "langmuir_hinshelwood", "reversible"].index(
                    get_cfg("kinetic_model", "power_law")
                ),
                key="cfg_kinetic_model",
                disabled=global_disabled,
            )

            st.markdown("#### 求解器")
            solver_method = st.selectbox(
                "Method",
                ["RK45", "BDF", "Radau"],
                index=["RK45", "BDF", "Radau"].index(get_cfg("solver_method", "RK45")),
                key="cfg_solver_method",
                disabled=global_disabled,
            )
            col_tol1, col_tol2 = st.columns(2)
            rtol = col_tol1.number_input(
                "rtol",
                value=get_cfg("rtol", 1e-6),
                format="%.1e",
                key="cfg_rtol",
                disabled=global_disabled,
            )
            atol = col_tol2.number_input(
                "atol",
                value=get_cfg("atol", 1e-9),
                format="%.1e",
                key="cfg_atol",
                disabled=global_disabled,
            )

        # Config Managment
        with st.expander("⚙️ 配置管理 (导入/导出/重置)"):
            uploaded_config = st.file_uploader(
                "导入配置", type=["json"], key="uploaded_config_json", disabled=global_disabled
            )
            if uploaded_config:
                try:
                    uploaded_bytes = uploaded_config.getvalue()
                    file_digest = hashlib.sha256(uploaded_bytes).hexdigest()
                    if st.session_state.get("imported_config_digest", None) == file_digest:
                        pass
                    else:
                        cfg_text = uploaded_bytes.decode("utf-8")
                        cfg = config_manager.import_config_from_json(cfg_text)
                        is_valid, error_message = config_manager.validate_config(cfg)
                        if not is_valid:
                            st.error(f"导入配置失败（配置校验未通过）：{error_message}")
                        else:
                            st.session_state["imported_config_digest"] = file_digest
                            st.session_state["pending_imported_config"] = cfg
                            st.success("导入成功！正在应用配置并刷新页面...")
                            st.rerun()
                except Exception as exc:
                    st.error(f"导入配置失败（JSON/编码错误）：{exc}")

            export_config_placeholder = st.empty()

            if st.button("重置为默认", disabled=global_disabled):
                st.session_state["pending_reset_to_default"] = True
                st.rerun()

    # ========== Main Content ==========
    st.title(f"⚗️ {reactor_type} 反应动力学参数拟合")
    if reactor_type == "PFR":
        st.caption("模型：PFR (solve_ivp) + least_squares")
    else:
        st.caption("模型：Batch (solve_ivp) + least_squares")

    tab_model, tab_data, tab_fit = st.tabs(
        ["① 反应与模型", "② 实验数据", "③ 拟合与结果"]
    )
    tab_fit_results_container = tab_fit.container()

    # ---------------- TAB 1: MODEL ----------------
    with tab_model:
        col_def1, col_def2 = st.columns([2, 1])
        with col_def1:
            species_text = st.text_input(
                "物种列表 (逗号分隔)",
                value=get_cfg("species_text", "A,B,C"),
                key="cfg_species_text",
            )
        with col_def2:
            n_reactions = int(
                st.number_input(
                    "反应数",
                    value=get_cfg("n_reactions", 1),
                    min_value=1,
                    key="cfg_n_reactions",
                )
            )

        species_names = _clean_species_names(species_text)
        if not species_names:
            st.stop()

        # Stoichiometry
        st.markdown("**化学计量数矩阵 ν** (行=物种, 列=反应)")
        nu_default = _build_default_nu_table(species_names, n_reactions)
        # Apply imported stoich if exists
        imp_stoich = get_cfg("stoich_matrix", None)
        if imp_stoich:
            try:
                arr = np.array(imp_stoich)
                if arr.shape == nu_default.shape:
                    nu_default = pd.DataFrame(
                        arr, index=nu_default.index, columns=nu_default.columns
                    )
                else:
                    _warn_once(
                        f"warn_stoich_shape_{len(species_names)}_{n_reactions}",
                        f"导入配置中的化学计量数矩阵尺寸不匹配，已忽略：期望 {nu_default.shape}，实际 {arr.shape}",
                    )
            except Exception as exc:
                _warn_once(
                    f"warn_stoich_parse_{len(species_names)}_{n_reactions}",
                    f"导入配置中的化学计量数矩阵无法解析，已忽略：{exc}",
                )

        nu_table = st.data_editor(
            nu_default,
            use_container_width=True,
            key=f"nu_{len(species_names)}_{n_reactions}",
        )
        stoich_matrix = nu_table.to_numpy(dtype=float)

        st.markdown("---")
        st.markdown("#### 动力学参数初值")

        # --- Base Parameters (k0, Ea, n) ---
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.caption("速率常数 k0 & 活化能 Ea")
            # Logic to get defaults
            k0_def = (
                np.array(get_cfg("k0_guess", [1e3] * n_reactions))
                if get_cfg("k0_guess", None)
                else np.full(n_reactions, 1e3)
            )
            ea_def = (
                np.array(get_cfg("ea_guess_J_mol", [8e4] * n_reactions))
                if get_cfg("ea_guess_J_mol", None)
                else np.full(n_reactions, 8e4)
            )
            fit_k0_def = (
                np.array(get_cfg("fit_k0_flags", [True] * n_reactions))
                if get_cfg("fit_k0_flags", None)
                else np.full(n_reactions, True)
            )
            fit_ea_def = (
                np.array(get_cfg("fit_ea_flags", [True] * n_reactions))
                if get_cfg("fit_ea_flags", None)
                else np.full(n_reactions, True)
            )

            k0_guess, ea_guess_J_mol, fit_k0_flags, fit_ea_flags = (
                ui_comp.render_param_table(
                    f"base_params_{n_reactions}",
                    [f"R{i+1}" for i in range(n_reactions)],
                    "k0_guess",
                    k0_def,
                    "指前因子",
                    "Ea_guess_J_mol",
                    ea_def,
                    "活化能 [J/mol]",
                    fit_k0_def,
                    fit_ea_def,
                )
            )

        with col_p2:
            st.caption("反应级数 n")
            # Logic for order defaults
            order_data = (
                np.array(
                    get_cfg("order_guess", np.zeros((n_reactions, len(species_names))))
                )
                if get_cfg("order_guess", None)
                else None
            )
            if order_data is None:
                # Simple default logic
                order_data = np.zeros((n_reactions, len(species_names)))
                if len(species_names) > 0:
                    order_data[:, 0] = 1.0

            fit_order_def = (
                np.array(
                    get_cfg(
                        "fit_order_flags_matrix",
                        np.zeros((n_reactions, len(species_names)), dtype=bool),
                    )
                )
                if get_cfg("fit_order_flags_matrix", None)
                else None
            )

            order_guess, fit_order_flags_matrix = ui_comp.render_order_table(
                f"base_orders_{n_reactions}",
                [f"R{i+1}" for i in range(n_reactions)],
                species_names,
                order_data,
                fit_order_def,
            )

        # --- L-H Parameters ---
        K0_ads, Ea_K_J_mol, fit_K0_ads_flags, fit_Ea_K_flags = None, None, None, None
        m_inhibition, fit_m_flags = None, None

        if kinetic_model == "langmuir_hinshelwood":
            with st.expander("Langmuir-Hinshelwood (L-H) 参数", expanded=True):
                col_lh1, col_lh2 = st.columns(2)
                with col_lh1:
                    st.caption("吸附常数 K (每物种)")
                    # Defaults
                    K0_def = (
                        np.array(get_cfg("K0_ads", [1.0] * len(species_names)))
                        if get_cfg("K0_ads", None)
                        else np.ones(len(species_names))
                    )
                    EaK_def = (
                        np.array(get_cfg("Ea_K_J_mol", [-2e4] * len(species_names)))
                        if get_cfg("Ea_K_J_mol", None)
                        else np.full(len(species_names), -2e4)
                    )
                    fit_K0_def = (
                        np.array(
                            get_cfg("fit_K0_ads_flags", [False] * len(species_names))
                        )
                        if get_cfg("fit_K0_ads_flags", None)
                        else np.full(len(species_names), False)
                    )
                    fit_EaK_def = (
                        np.array(
                            get_cfg("fit_Ea_K_flags", [False] * len(species_names))
                        )
                        if get_cfg("fit_Ea_K_flags", None)
                        else np.full(len(species_names), False)
                    )

                    K0_ads, Ea_K_J_mol, fit_K0_ads_flags, fit_Ea_K_flags = (
                        ui_comp.render_param_table(
                            f"lh_ads_{len(species_names)}",
                            species_names,
                            "K0_ads",
                            K0_def,
                            "吸附常数指前因子",
                            "Ea_K_J_mol",
                            EaK_def,
                            "吸附热",
                            fit_K0_def,
                            fit_EaK_def,
                        )
                    )
                with col_lh2:
                    st.caption("抑制指数 m (每反应)")
                    m_def = (
                        np.array(get_cfg("m_inhibition", [1.0] * n_reactions))
                        if get_cfg("m_inhibition", None)
                        else np.ones(n_reactions)
                    )
                    fit_m_def = (
                        np.array(get_cfg("fit_m_flags", [False] * n_reactions))
                        if get_cfg("fit_m_flags", None)
                        else np.full(n_reactions, False)
                    )

                    m_df = pd.DataFrame(
                        {"m": m_def, "Fit_m": fit_m_def},
                        index=[f"R{i+1}" for i in range(n_reactions)],
                    )
                    m_editor = st.data_editor(
                        m_df,
                        key=f"lh_m_{n_reactions}",
                        column_config={
                            "Fit_m": st.column_config.CheckboxColumn(default=False)
                        },
                    )
                    m_inhibition = m_editor["m"].to_numpy(dtype=float)
                    fit_m_flags = m_editor["Fit_m"].to_numpy(dtype=bool)
        else:
            # Init empty for compatibility
            K0_ads = np.zeros(len(species_names))
            Ea_K_J_mol = np.zeros(len(species_names))
            fit_K0_ads_flags = np.zeros(len(species_names), dtype=bool)
            fit_Ea_K_flags = np.zeros(len(species_names), dtype=bool)
            m_inhibition = np.ones(n_reactions)
            fit_m_flags = np.zeros(n_reactions, dtype=bool)

        # --- Reversible Parameters ---
        k0_rev, ea_rev_J_mol, fit_k0_rev_flags, fit_ea_rev_flags = (
            None,
            None,
            None,
            None,
        )
        order_rev, fit_order_rev_flags_matrix = None, None

        if kinetic_model == "reversible":
            with st.expander("可逆反应 (逆反应) 参数", expanded=True):
                col_rev1, col_rev2 = st.columns(2)
                with col_rev1:
                    st.caption("逆反应 k0⁻ & Ea⁻")
                    k0r_def = (
                        np.array(get_cfg("k0_rev", [1e2] * n_reactions))
                        if get_cfg("k0_rev", None)
                        else np.full(n_reactions, 1e2)
                    )
                    ear_def = (
                        np.array(get_cfg("ea_rev_J_mol", [9e4] * n_reactions))
                        if get_cfg("ea_rev_J_mol", None)
                        else np.full(n_reactions, 9e4)
                    )
                    fit_k0r_def = (
                        np.array(get_cfg("fit_k0_rev_flags", [False] * n_reactions))
                        if get_cfg("fit_k0_rev_flags", None)
                        else np.full(n_reactions, False)
                    )
                    fit_ear_def = (
                        np.array(get_cfg("fit_ea_rev_flags", [False] * n_reactions))
                        if get_cfg("fit_ea_rev_flags", None)
                        else np.full(n_reactions, False)
                    )

                    k0_rev, ea_rev_J_mol, fit_k0_rev_flags, fit_ea_rev_flags = (
                        ui_comp.render_param_table(
                            f"rev_params_{n_reactions}",
                            [f"R{i+1}" for i in range(n_reactions)],
                            "k0_rev",
                            k0r_def,
                            "逆反应 k0",
                            "Ea_rev_J_mol",
                            ear_def,
                            "逆反应 Ea",
                            fit_k0r_def,
                            fit_ear_def,
                        )
                    )
                with col_rev2:
                    st.caption("逆反应级数 n⁻")
                    ordr_def = (
                        np.array(
                            get_cfg(
                                "order_rev", np.zeros((n_reactions, len(species_names)))
                            )
                        )
                        if get_cfg("order_rev", None)
                        else None
                    )
                    fit_ordr_def = (
                        np.array(
                            get_cfg(
                                "fit_order_rev_flags_matrix",
                                np.zeros((n_reactions, len(species_names)), dtype=bool),
                            )
                        )
                        if get_cfg("fit_order_rev_flags_matrix", None)
                        else None
                    )

                    order_rev, fit_order_rev_flags_matrix = ui_comp.render_order_table(
                        f"rev_orders_{n_reactions}",
                        [f"R{i+1}" for i in range(n_reactions)],
                        species_names,
                        ordr_def,
                        fit_ordr_def,
                    )
        else:
            k0_rev = np.zeros(n_reactions)
            ea_rev_J_mol = np.zeros(n_reactions)
            fit_k0_rev_flags = np.zeros(n_reactions, dtype=bool)
            fit_ea_rev_flags = np.zeros(n_reactions, dtype=bool)
            order_rev = np.zeros((n_reactions, len(species_names)))
            fit_order_rev_flags_matrix = np.zeros(
                (n_reactions, len(species_names)), dtype=bool
            )

    # ---------------- TAB 2: DATA ----------------
    data_df = st.session_state.get("data_df_cached", None)
    output_mode = "Cout (mol/m^3)"
    output_species_list = []

    with tab_data:
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.markdown("#### 1. 下载模板")
            if reactor_type == "PFR":
                meas_cols = (
                    [f"Cout_{s}_mol_m3" for s in species_names]
                    + [f"Fout_{s}_mol_s" for s in species_names]
                    + [f"X_{s}" for s in species_names]
                )
                cols = (
                    ["V_m3", "T_K", "vdot_m3_s"]
                    + [f"F0_{s}_mol_s" for s in species_names]
                    + meas_cols
                )
            else:
                meas_cols = (
                    [f"Cout_{s}_mol_m3" for s in species_names]
                    + [f"X_{s}" for s in species_names]
                )
                cols = (
                    ["t_s", "T_K"]
                    + [f"C0_{s}_mol_m3" for s in species_names]
                    + meas_cols
                )

            template_csv = (
                pd.DataFrame(columns=cols).to_csv(index=False).encode("utf-8")
            )
            st.download_button(
                "📥 下载 CSV 模板", template_csv, "template.csv", "text/csv"
            )

        with col_d2:
            st.markdown("#### 2. 上传数据")
            if "uploaded_csv_bytes" in st.session_state and st.session_state["uploaded_csv_bytes"]:
                cached_name = str(st.session_state.get("uploaded_csv_name", "")).strip()
                cached_text = f"已缓存文件：{cached_name}" if cached_name else "已缓存上传文件"
                st.caption(cached_text + "（页面刷新/切换不会丢失，除非手动删除）")
                if st.button("🗑️ 删除已上传文件", key="delete_uploaded_csv"):
                    for k in ["uploaded_csv_bytes", "uploaded_csv_name"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    if "data_df_cached" in st.session_state:
                        del st.session_state["data_df_cached"]
                    ok, message = _delete_persisted_upload()
                    if not ok:
                        st.warning(message)
                    if "uploaded_csv" in st.session_state:
                        del st.session_state["uploaded_csv"]
                    st.rerun()

            uploaded_file = st.file_uploader(
                "上传 CSV", type=["csv"], label_visibility="collapsed", key="uploaded_csv"
            )

        st.divider()
        col_mz1, col_mz2 = st.columns(2)
        with col_mz1:
            opts = (
                ["Fout (mol/s)", "Cout (mol/m^3)", "X (conversion)"]
                if reactor_type == "PFR"
                else ["Cout (mol/m^3)", "X (conversion)"]
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
                key="cfg_output_mode",
            )

        with col_mz2:
            if "cfg_output_species_list" in st.session_state:
                current_list = st.session_state.get("cfg_output_species_list", [])
                if not isinstance(current_list, list):
                    current_list = []
                cleaned_list = [str(x) for x in current_list if str(x) in species_names]
                if not cleaned_list:
                    cleaned_list = list(species_names)
                st.session_state["cfg_output_species_list"] = cleaned_list
            fit_mask = st.multiselect(
                "选择进入目标函数的物种",
                species_names,
                default=list(species_names),
                key="cfg_output_species_list",
            )
            output_species_list = fit_mask

        if uploaded_file:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_name = str(getattr(uploaded_file, "name", "")).strip()
                st.session_state["uploaded_csv_bytes"] = uploaded_bytes
                st.session_state["uploaded_csv_name"] = uploaded_name
                ok, message = _save_persisted_upload(uploaded_bytes, uploaded_name)
                if not ok:
                    st.warning(message)
            except Exception as exc:
                st.error(f"读取上传文件失败: {exc}")

        if uploaded_file or ("uploaded_csv_bytes" in st.session_state and st.session_state["uploaded_csv_bytes"]):
            try:
                if uploaded_file:
                    csv_bytes = uploaded_file.getvalue()
                else:
                    csv_bytes = st.session_state["uploaded_csv_bytes"]
                data_df = _read_csv_bytes_cached(csv_bytes)
                st.session_state["data_df_cached"] = data_df
                st.markdown("#### 数据预览")
                st.dataframe(data_df.head(50), use_container_width=True, height=200)
            except Exception as exc:
                st.error(f"CSV 读取失败: {exc}")
                data_df = None

    # --- Build export config (basic; will be updated again in tab_fit if advanced settings exist) ---
    if export_config_placeholder is not None:
        export_k0_min = float(get_cfg("k0_min", 1e-15))
        export_k0_max = float(get_cfg("k0_max", 1e15))
        export_ea_min = float(get_cfg("ea_min_J_mol", 1e4))
        export_ea_max = float(get_cfg("ea_max_J_mol", 3e5))
        export_ord_min = float(get_cfg("order_min", -2.0))
        export_ord_max = float(get_cfg("order_max", 5.0))

        export_diff_step_rel = float(get_cfg("diff_step_rel", 1e-2))
        export_max_nfev = int(get_cfg("max_nfev", 3000))
        export_use_x_scale_jac = bool(get_cfg("use_x_scale_jac", True))
        export_use_ms = bool(get_cfg("use_multi_start", True))
        export_n_starts = int(get_cfg("n_starts", 10))
        export_max_nfev_coarse = int(get_cfg("max_nfev_coarse", 300))
        export_random_seed = int(get_cfg("random_seed", 42))
        export_max_step_fraction = float(get_cfg("max_step_fraction", 0.1))

        export_cfg = config_manager.collect_config(
            reactor_type=reactor_type,
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
            Ea_K_J_mol=None if Ea_K_J_mol is None else np.asarray(Ea_K_J_mol, dtype=float),
            fit_K0_ads_flags=(
                None if fit_K0_ads_flags is None else np.asarray(fit_K0_ads_flags, dtype=bool)
            ),
            fit_Ea_K_flags=(
                None if fit_Ea_K_flags is None else np.asarray(fit_Ea_K_flags, dtype=bool)
            ),
            m_inhibition=None if m_inhibition is None else np.asarray(m_inhibition, dtype=float),
            fit_m_flags=None if fit_m_flags is None else np.asarray(fit_m_flags, dtype=bool),
            k0_rev=None if k0_rev is None else np.asarray(k0_rev, dtype=float),
            ea_rev_J_mol=None if ea_rev_J_mol is None else np.asarray(ea_rev_J_mol, dtype=float),
            fit_k0_rev_flags=(
                None if fit_k0_rev_flags is None else np.asarray(fit_k0_rev_flags, dtype=bool)
            ),
            fit_ea_rev_flags=(
                None if fit_ea_rev_flags is None else np.asarray(fit_ea_rev_flags, dtype=bool)
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
            ok, message = config_manager.auto_save_config(export_cfg)
            if not ok:
                st.warning(message)
        export_config_bytes = config_manager.export_config_to_json(export_cfg).encode("utf-8")
        export_config_placeholder.download_button(
            "📥 导出当前配置 (JSON)",
            export_config_bytes,
            file_name="kinetics_config.json",
            mime="application/json",
            use_container_width=True,
            key="export_config_download_basic",
        )

    # ---------------- TAB 3: FITTING ----------------
    with tab_fit:
        if data_df is None:
            st.info("请先在「实验数据」页面上传 CSV 文件（或恢复已缓存的文件）。")
            st.stop()
        if not output_species_list:
            st.error("请选择至少一个目标物种。")
            st.stop()

        data_len = len(data_df)

        # --- Advanced Settings (Expanded) ---
        with st.expander("高级设置与边界 (点击展开)", expanded=False):
            st.caption("修改任意参数会自动应用（Streamlit 会自动 rerun）。")

            st.markdown("**1. 基础边界设置**")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                k0_min = st.number_input(
                    "k0 Min",
                    value=float(get_cfg("k0_min", 1e-15)),
                    format="%.1e",
                    key="cfg_k0_min",
                )
                k0_max = st.number_input(
                    "k0 Max",
                    value=float(get_cfg("k0_max", 1e15)),
                    format="%.1e",
                    key="cfg_k0_max",
                )
            with col_b2:
                ea_min = st.number_input(
                    "Ea Min",
                    value=float(get_cfg("ea_min_J_mol", 1e4)),
                    format="%.1e",
                    key="cfg_ea_min_J_mol",
                )
                ea_max = st.number_input(
                    "Ea Max",
                    value=float(get_cfg("ea_max_J_mol", 3e5)),
                    format="%.1e",
                    key="cfg_ea_max_J_mol",
                )
            with col_b3:
                ord_min = st.number_input(
                    "Order Min",
                    value=float(get_cfg("order_min", -2.0)),
                    key="cfg_order_min",
                )
                ord_max = st.number_input(
                    "Order Max",
                    value=float(get_cfg("order_max", 5.0)),
                    key="cfg_order_max",
                )

            st.divider()
            st.markdown("**2. 算法与鲁棒性**")
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            with col_adv1:
                max_nfev = int(
                    st.number_input(
                        "Max Iterations (外层迭代)",
                        value=int(get_cfg("max_nfev", 3000)),
                        step=500,
                        key="cfg_max_nfev",
                        help="提示：每次迭代内部会为数值差分 Jacobian 额外调用模型多次，所以看到的“调用次数”通常会大于该值。",
                    )
                )
                diff_step_rel = st.number_input(
                    "diff_step (Finite Diff)",
                    value=get_cfg("diff_step_rel", 1e-2),
                    format="%.1e",
                    key="cfg_diff_step_rel",
                )
                max_step_fraction = st.number_input(
                    "max_step_fraction (ODE)",
                    value=float(get_cfg("max_step_fraction", 0.1)),
                    min_value=0.0,
                    max_value=10.0,
                    step=0.05,
                    key="cfg_max_step_fraction",
                    help="用于 solve_ivp 的 max_step：max_step = fraction × 总时间/总体积；0 表示不限制。",
                )
            with col_adv2:
                use_ms = st.checkbox(
                    "Multi-start (多起点)",
                    value=bool(get_cfg("use_multi_start", True)),
                    key="cfg_use_multi_start",
                )
                n_starts = int(
                    st.number_input(
                        "Start Points",
                        value=get_cfg("n_starts", 10),
                        min_value=1,
                        step=1,
                        key="cfg_n_starts",
                        help="仅当 Multi-start 勾选且 Start Points>1 时才生效。",
                    )
                )
            with col_adv3:
                use_x_scale_jac = st.checkbox(
                    "Use x_scale='jac'",
                    value=get_cfg("use_x_scale_jac", True),
                    key="cfg_use_x_scale_jac",
                )
                random_seed = int(
                    st.number_input(
                        "Random Seed",
                        value=get_cfg("random_seed", 42),
                        step=1,
                        key="cfg_random_seed",
                    )
                )
                max_nfev_coarse = int(
                    st.number_input(
                        "Coarse check iters",
                        value=get_cfg("max_nfev_coarse", 300),
                        step=50,
                        key="cfg_max_nfev_coarse",
                        help="仅当 Multi-start 生效时用于粗拟合阶段。",
                    )
                )

            st.divider()
            st.caption(
                "说明：当模型计算失败（如 solve_ivp 失败）时，残差会使用系统默认罚项（不在 UI 中提供调节）。"
            )

        # Update export config with advanced settings (when tab_fit is active and widgets are available)
        if export_config_placeholder is not None:
            export_config_placeholder.empty()
            export_cfg = config_manager.collect_config(
                reactor_type=reactor_type,
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
                Ea_K_J_mol=None if Ea_K_J_mol is None else np.asarray(Ea_K_J_mol, dtype=float),
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
                m_inhibition=None if m_inhibition is None else np.asarray(m_inhibition, dtype=float),
                fit_m_flags=None if fit_m_flags is None else np.asarray(fit_m_flags, dtype=bool),
                k0_rev=None if k0_rev is None else np.asarray(k0_rev, dtype=float),
                ea_rev_J_mol=None if ea_rev_J_mol is None else np.asarray(ea_rev_J_mol, dtype=float),
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
                k0_min=float(k0_min),
                k0_max=float(k0_max),
                ea_min_J_mol=float(ea_min),
                ea_max_J_mol=float(ea_max),
                order_min=float(ord_min),
                order_max=float(ord_max),
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
                ok, message = config_manager.auto_save_config(export_cfg)
                if not ok:
                    st.warning(message)
            export_config_bytes = config_manager.export_config_to_json(export_cfg).encode("utf-8")
            export_config_placeholder.download_button(
                "📥 导出当前配置 (JSON)",
                export_config_bytes,
                file_name="kinetics_config.json",
                mime="application/json",
                use_container_width=True,
                key="export_config_download_advanced",
            )

        # --- Action Buttons ---
        _drain_fitting_progress_queue()

        fitting_future = st.session_state.get("fitting_future", None)
        fitting_running = bool(st.session_state.get("fitting_running", False))
        fitting_stop_event = st.session_state.get("fitting_stop_event", None)
        if fitting_stop_event is None:
            fitting_stop_event = threading.Event()
            st.session_state["fitting_stop_event"] = fitting_stop_event

        # Self-heal: if session refreshed and Future object is lost, stop showing "running".
        if fitting_running and (fitting_future is None):
            st.session_state["fitting_running"] = False
            st.session_state["fitting_status"] = "后台任务已丢失（可能是页面刷新导致）。请重新开始拟合。"
            fitting_running = False

        if fitting_future is not None and fitting_future.done():
            st.session_state["fitting_running"] = False
            st.session_state["fitting_future"] = None

            try:
                fit_results = fitting_future.result()
                _drain_fitting_progress_queue()
                st.session_state["fit_results"] = fit_results
                st.session_state["fitting_status"] = "拟合完成。"
                phi_value = float(fit_results.get("phi_final", fit_results.get("cost", 0.0)))
                st.session_state["fitting_timeline"].append(("✅", f"拟合完成，最终 Φ: {phi_value:.4e}"))
                st.success(
                    "拟合完成！结果已缓存（结果展示将锁定为本次拟合的配置与数据）。"
                    f" 目标函数 Φ: {phi_value:.4e}"
                )
            except FittingStoppedError:
                st.session_state["fitting_status"] = "用户终止。"
                st.session_state["fitting_timeline"].append(("⚠️", "拟合已终止。"))
                st.warning("拟合已终止。")
            except Exception as exc:
                st.session_state["fitting_status"] = "拟合失败。"
                st.session_state["fitting_timeline"].append(("❌", f"拟合失败: {exc}"))
                st.error(f"Fitting Error: {exc}")

        fitting_future = st.session_state.get("fitting_future", None)
        fitting_running = bool(st.session_state.get("fitting_running", False))

        col_act1, col_act2, col_act3, col_act4, col_act5 = st.columns([3, 1, 1, 1, 1])
        start_btn = col_act1.button(
            "🚀 开始拟合", type="primary", disabled=fitting_running, use_container_width=True
        )
        stop_btn = col_act2.button(
            "⏹️ 终止", type="secondary", disabled=not fitting_running, use_container_width=True
        )
        auto_refresh = col_act3.checkbox(
            "自动刷新",
            value=bool(st.session_state.get("fitting_auto_refresh", True)),
            disabled=not fitting_running,
            help="开启后，页面会每隔约 1 秒自动刷新一次，用于持续更新拟合进度与阶段信息；关闭可减少卡顿/CPU 占用。",
        )
        refresh_interval_s = float(
            col_act5.number_input(
                "间隔(s)",
                value=float(st.session_state.get("fitting_refresh_interval_s", 2.0)),
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                disabled=(not fitting_running) or (not auto_refresh),
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

        if stop_btn and fitting_running:
            st.session_state.fitting_stopped = True
            st.session_state["fitting_status"] = "已发送终止请求，等待后台停止..."
            fitting_stop_event.set()
            st.session_state["fitting_timeline"].append(("⏹️", "用户请求终止拟合..."))
            st.rerun()

        if start_btn and (not fitting_running):
            if data_df is None:
                st.error("当前没有可用的 CSV 数据，请先在「实验数据」页面上传或恢复已缓存的文件。")
                st.stop()

            # Ensure a fresh executor for each fitting run (avoid queued/stuck tasks from prior sessions).
            old_executor = st.session_state.get("fitting_executor", None)
            if old_executor is not None:
                try:
                    old_executor.shutdown(wait=False, cancel_futures=True)
                except Exception as exc:
                    _warn_once("warn_executor_shutdown", f"关闭旧的拟合线程池失败（可忽略）：{exc}")
                st.session_state["fitting_executor"] = None

            st.session_state.fitting_stopped = False
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
                "title": "📊 拟合任务概览",
                "lines": [
                    f"数据点数量: {int(len(data_df))} 行",
                    f"待拟合参数: {int(n_fit_params)} 个",
                    f"反应器类型: {reactor_type}",
                    f"动力学模型: {kinetic_model}",
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
                "max_step_fraction": float(max_step_fraction),
            }

            st.session_state["fitting_running"] = True
            executor = _get_fitting_executor()
            st.session_state["fitting_future"] = executor.submit(
                _run_fitting_job, job_inputs, stop_event, progress_queue
            )
            st.rerun()

        if fitting_running:
            st.caption("“自动刷新”：仅刷新进度区域（避免整页闪烁）；若觉得卡顿可关闭。")
            refresh_interval_s = float(st.session_state.get("fitting_refresh_interval_s", 2.0))
            if bool(st.session_state.get("fitting_auto_refresh", True)):
                st.fragment(_render_fitting_live_progress, run_every=refresh_interval_s)()
            else:
                st.fragment(_render_fitting_live_progress)()
        elif st.session_state.get("fitting_timeline", []):
            _render_fitting_progress_panel()

    # --- Results Display (Optimized) ---
    if "fit_results" in st.session_state:
        res = st.session_state["fit_results"]
        tab_fit_results_container.divider()
        phi_value = float(res.get("phi_final", res.get("cost", 0.0)))
        tab_fit_results_container.markdown(f"### 拟合结果 (目标函数 Φ: {phi_value:.4e})")
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
        max_step_fraction_fit = float(res.get("max_step_fraction", get_cfg("max_step_fraction", 0.1)))
        reactor_type_fit = res.get("reactor_type", reactor_type)
        kinetic_model_fit = res.get("kinetic_model", kinetic_model)
        output_mode_fit = res.get("output_mode", output_mode)
        output_species_fit = res.get("output_species", [])
        unit_text = _get_output_unit_text(output_mode_fit)

        parity_species_candidates = []
        parity_species_unavailable = []
        for sp_name in species_names_fit:
            meas_col = _get_measurement_column_name(output_mode_fit, sp_name)
            if meas_col not in df_fit.columns:
                parity_species_unavailable.append(f"{sp_name}（缺少列 {meas_col}）")
                continue
            numeric_series = pd.to_numeric(df_fit[meas_col], errors="coerce")
            if bool(np.any(np.isfinite(numeric_series.to_numpy()))):
                parity_species_candidates.append(sp_name)
            else:
                parity_species_unavailable.append(f"{sp_name}（列 {meas_col} 全为 NaN/非数字）")

        tab_param, tab_parity, tab_profile, tab_export = tab_fit_results_container.tabs(
            ["参数", "奇偶校验图", "沿程/随时间剖面", "导出"]
        )

        with tab_param:
            st.markdown("#### 拟合参数")
            col_p1, col_p2 = st.columns([1, 1])

            with col_p1:
                reaction_names = [f"R{i+1}" for i in range(len(fitted_params["k0"]))]
                df_k0_ea = pd.DataFrame(
                    {"k0 [SI]": fitted_params["k0"], "Ea [J/mol]": fitted_params["ea_J_mol"]},
                    index=reaction_names,
                )
                st.markdown("**k0 与 Ea**")
                st.dataframe(df_k0_ea, use_container_width=True, height=250)

            with col_p2:
                st.markdown("**反应级数矩阵 $n$**")
                df_orders = pd.DataFrame(
                    fitted_params["reaction_order_matrix"],
                    index=reaction_names,
                    columns=species_names_fit,
                )
                st.dataframe(df_orders, use_container_width=True, height=250)

            if kinetic_model_fit == "langmuir_hinshelwood":
                st.markdown("#### Langmuir-Hinshelwood 参数")
                col_lh1, col_lh2 = st.columns([1, 1])
                with col_lh1:
                    if fitted_params.get("K0_ads", None) is not None and fitted_params.get(
                        "Ea_K", None
                    ) is not None:
                        df_ads = pd.DataFrame(
                            {
                                "K0_ads [1/(mol/m^3)]": fitted_params["K0_ads"],
                                "Ea_K [J/mol]": fitted_params["Ea_K"],
                            },
                            index=species_names_fit,
                        )
                        st.dataframe(df_ads, use_container_width=True, height=250)
                with col_lh2:
                    if fitted_params.get("m_inhibition", None) is not None:
                        df_m = pd.DataFrame(
                            {"m_inhibition [-]": fitted_params["m_inhibition"]},
                            index=reaction_names,
                        )
                        st.dataframe(df_m, use_container_width=True, height=250)

            if kinetic_model_fit == "reversible":
                st.markdown("#### 可逆反应参数（逆反应）")
                if fitted_params.get("k0_rev", None) is not None and fitted_params.get(
                    "ea_rev", None
                ) is not None:
                    df_rev = pd.DataFrame(
                        {"k0_rev [SI]": fitted_params["k0_rev"], "Ea_rev [J/mol]": fitted_params["ea_rev"]},
                        index=reaction_names,
                    )
                    st.dataframe(df_rev, use_container_width=True, height=250)
                if fitted_params.get("order_rev", None) is not None:
                    st.markdown("**逆反应级数矩阵 $n^-$**")
                    df_order_rev = pd.DataFrame(
                        fitted_params["order_rev"],
                        index=reaction_names,
                        columns=species_names_fit,
                    )
                    st.dataframe(df_order_rev, use_container_width=True, height=250)

        with tab_parity:
            st.markdown("#### 不同物种的奇偶校验图 (Measured vs Predicted)")
            if parity_species_unavailable:
                show_missing = st.checkbox("显示无法绘图的物种原因", value=False)
                if show_missing:
                    st.caption("无法绘制奇偶校验图的物种： " + "，".join(parity_species_unavailable))

            cache_key = (
                float(res.get("phi_final", res.get("cost", 0.0))),
                str(output_mode_fit),
                tuple(parity_species_candidates),
                float(rtol_fit),
                float(atol_fit),
                str(solver_method_fit),
                str(reactor_type_fit),
                str(kinetic_model_fit),
                float(max_step_fraction_fit),
            )
            if (
                st.session_state.get("fit_compare_cache_key", None) != cache_key
                or "fit_compare_long_df" not in st.session_state
            ):
                try:
                    st.session_state["fit_compare_cache_key"] = cache_key
                    st.session_state["fit_compare_long_df"] = _build_fit_comparison_long_table(
                        data_df=df_fit,
                        species_names=species_names_fit,
                        output_mode=output_mode_fit,
                        output_species_list=parity_species_candidates,
                        stoich_matrix=stoich_matrix_fit,
                        fitted_params=fitted_params,
                        solver_method=solver_method_fit,
                        rtol=float(rtol_fit),
                        atol=float(atol_fit),
                        reactor_type=reactor_type_fit,
                        kinetic_model=kinetic_model_fit,
                        max_step_fraction=float(max_step_fraction_fit),
                    )
                except Exception as exc:
                    st.error(f"生成对比数据失败: {exc}")
                    st.session_state["fit_compare_long_df"] = pd.DataFrame()

            df_long = st.session_state["fit_compare_long_df"]
            if df_long.empty:
                st.warning("对比数据为空：无法生成奇偶校验图。")
            else:
                species_selected = st.multiselect(
                    "选择要显示的物种",
                    list(parity_species_candidates),
                    default=list(parity_species_candidates),
                )
                show_residual_plot = st.checkbox("显示残差图", value=True)
                n_cols = int(st.number_input("每行子图数", min_value=1, max_value=4, value=2, step=1))

                df_ok = df_long[df_long["ok"]].copy()
                df_ok = df_ok[np.isfinite(df_ok["measured"]) & np.isfinite(df_ok["predicted"])]
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
                        species_list_plot = list(dict.fromkeys(df_ok["species"].tolist()))
                        n_plots = len(species_list_plot)
                        n_rows = int(np.ceil(n_plots / max(n_cols, 1)))
                        fig, axes = plt.subplots(
                            n_rows,
                            n_cols,
                            figsize=(5.2 * n_cols, 4.3 * n_rows),
                            squeeze=False,
                        )

                        for i, species_name in enumerate(species_list_plot):
                            ax = axes[i // n_cols][i % n_cols]
                            df_sp = df_ok[df_ok["species"] == species_name]
                            ax.scatter(
                                df_sp["measured"].to_numpy(dtype=float),
                                df_sp["predicted"].to_numpy(dtype=float),
                                alpha=0.65,
                                label=species_name,
                            )
                            min_v = float(
                                np.nanmin(
                                    np.concatenate(
                                        [df_sp["measured"].to_numpy(), df_sp["predicted"].to_numpy()]
                                    )
                                )
                            )
                            max_v = float(
                                np.nanmax(
                                    np.concatenate(
                                        [df_sp["measured"].to_numpy(), df_sp["predicted"].to_numpy()]
                                    )
                                )
                            )
                            if np.isfinite(min_v) and np.isfinite(max_v) and max_v > min_v:
                                ax.plot([min_v, max_v], [min_v, max_v], "k--", label="y=x")
                            ax.set_title(f"{species_name}")
                            ax.set_xlabel(f"Measured [{unit_text}]")
                            ax.set_ylabel(f"Predicted [{unit_text}]")
                            ax.grid(True)
                            ax.legend()

                        for j in range(n_plots, n_rows * n_cols):
                            axes[j // n_cols][j % n_cols].axis("off")

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
                            mime="image/png" if image_format == "png" else "image/svg+xml",
                        )
                        plt.close(fig)

                if show_residual_plot:
                    st.markdown("#### 残差图 (Predicted - Measured)")
                    fig_r, ax_r = plt.subplots(figsize=(7.5, 4.8))
                    ax_r.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
                    for species_name in species_selected:
                        df_sp = df_long[df_long["species"] == species_name]
                        df_sp = df_sp[df_sp["ok"]]
                        df_sp = df_sp[np.isfinite(df_sp["residual"]) & np.isfinite(df_sp["measured"])]
                        if not df_sp.empty:
                            ax_r.scatter(
                                df_sp["measured"].to_numpy(dtype=float),
                                df_sp["residual"].to_numpy(dtype=float),
                                alpha=0.6,
                                label=species_name,
                            )
                    ax_r.set_xlabel(f"Measured [{unit_text}]")
                    ax_r.set_ylabel(f"Residual (Pred - Meas) [{unit_text}]")
                    ax_r.grid(True)
                    ax_r.legend()
                    fig_r.tight_layout()
                    st.pyplot(fig_r, use_container_width=True)
                    plt.close(fig_r)

                show_compare_table = st.checkbox("显示预测 vs 实验对比表", value=False)
                if show_compare_table:
                    st.markdown("#### 预测 vs 实验对比表")
                    df_show = df_long.copy()
                    df_show = df_show[df_show["species"].isin(species_selected)]
                    st.dataframe(df_show, use_container_width=True, height=260)

                st.markdown("#### 拟合误差指标（按物种）")
                rows_metric = []
                for species_name in species_selected:
                    df_sp = df_long[(df_long["species"] == species_name) & (df_long["ok"])].copy()
                    df_sp = df_sp[np.isfinite(df_sp["measured"]) & np.isfinite(df_sp["predicted"])]
                    if df_sp.empty:
                        continue
                    resid = df_sp["predicted"].to_numpy(dtype=float) - df_sp["measured"].to_numpy(dtype=float)
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
                    st.dataframe(pd.DataFrame(rows_metric), use_container_width=True, height=220)

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
                    st.number_input("剖面点数", min_value=20, max_value=2000, value=200, step=20)
                )
                profile_species = st.multiselect(
                    "选择要画剖面的物种（可多选）",
                    list(species_names_fit),
                    default=list(species_names_fit[: min(3, len(species_names_fit))]),
                )

                row_sel = df_fit.loc[selected_row_index]
                if reactor_type_fit == "PFR":
                    profile_kind = st.radio(
                        "剖面变量",
                        ["F (mol/s)", "C (mol/m^3)"],
                        index=0,
                        horizontal=True,
                    )
                    reactor_volume_m3 = float(row_sel.get("V_m3", np.nan))
                    temperature_K = float(row_sel.get("T_K", np.nan))
                    vdot_m3_s = float(row_sel.get("vdot_m3_s", np.nan))

                    molar_flow_inlet = np.zeros(len(species_names_fit), dtype=float)
                    for i, sp_name in enumerate(species_names_fit):
                        molar_flow_inlet[i] = float(row_sel.get(f"F0_{sp_name}_mol_s", np.nan))

                    volume_grid_m3, molar_flow_profile, ok, message = reactors.integrate_pfr_profile(
                        reactor_volume_m3=reactor_volume_m3,
                        temperature_K=temperature_K,
                        vdot_m3_s=vdot_m3_s,
                        molar_flow_inlet_mol_s=molar_flow_inlet,
                        stoich_matrix=stoich_matrix_fit,
                        k0=fitted_params["k0"],
                        ea_J_mol=fitted_params["ea_J_mol"],
                        reaction_order_matrix=fitted_params["reaction_order_matrix"],
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
                    if not ok:
                        st.error(
                            f"PFR 剖面计算失败: {message}\n"
                            "建议：尝试将求解器切换为 `BDF` 或 `Radau`，并适当放宽 `rtol/atol`。"
                        )
                    else:
                        fig_pf, ax_pf = plt.subplots(figsize=(7, 4.5))
                        name_to_index = {name: i for i, name in enumerate(species_names_fit)}

                        profile_df = pd.DataFrame({"V_m3": volume_grid_m3})
                        for species_name in profile_species:
                            idx = name_to_index[species_name]
                            if profile_kind.startswith("F"):
                                y = molar_flow_profile[idx, :]
                                ax_pf.plot(volume_grid_m3, y, linewidth=2, label=species_name)
                                profile_df[f"F_{species_name}_mol_s"] = y
                            else:
                                conc = molar_flow_profile[idx, :] / max(vdot_m3_s, 1e-30)
                                ax_pf.plot(volume_grid_m3, conc, linewidth=2, label=species_name)
                                profile_df[f"C_{species_name}_mol_m3"] = conc

                        ax_pf.set_xlabel("Reactor volume V [m^3]")
                        ax_pf.set_ylabel(f"{profile_kind} [{('mol/s' if profile_kind.startswith('F') else 'mol/m^3')}]")
                        ax_pf.grid(True)
                        ax_pf.legend()
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
                            mime="image/png" if image_format_pf == "png" else "image/svg+xml",
                        )
                        plt.close(fig_pf)

                else:
                    profile_kind = st.radio(
                        "剖面变量",
                        ["C (mol/m^3)", "X (conversion)"],
                        index=0,
                        horizontal=True,
                    )
                    reaction_time_s = float(row_sel.get("t_s", np.nan))
                    temperature_K = float(row_sel.get("T_K", np.nan))
                    conc_initial = np.zeros(len(species_names_fit), dtype=float)
                    for i, sp_name in enumerate(species_names_fit):
                        conc_initial[i] = float(row_sel.get(f"C0_{sp_name}_mol_m3", np.nan))

                    time_grid_s, conc_profile, ok, message = reactors.integrate_batch_profile(
                        reaction_time_s=reaction_time_s,
                        temperature_K=temperature_K,
                        conc_initial_mol_m3=conc_initial,
                        stoich_matrix=stoich_matrix_fit,
                        k0=fitted_params["k0"],
                        ea_J_mol=fitted_params["ea_J_mol"],
                        reaction_order_matrix=fitted_params["reaction_order_matrix"],
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
                    if not ok:
                        st.error(
                            f"Batch 剖面计算失败: {message}\n"
                            "建议：尝试将求解器切换为 `BDF` 或 `Radau`，并适当放宽 `rtol/atol`。"
                        )
                    else:
                        fig_bt, ax_bt = plt.subplots(figsize=(7, 4.5))
                        name_to_index = {name: i for i, name in enumerate(species_names_fit)}
                        profile_df = pd.DataFrame({"t_s": time_grid_s})
                        for species_name in profile_species:
                            idx = name_to_index[species_name]
                            if profile_kind.startswith("C"):
                                y = conc_profile[idx, :]
                                ax_bt.plot(time_grid_s, y, linewidth=2, label=species_name)
                                profile_df[f"C_{species_name}_mol_m3"] = y
                            else:
                                c0 = float(conc_initial[idx])
                                if c0 < 1e-30:
                                    x = np.full_like(time_grid_s, np.nan, dtype=float)
                                else:
                                    x = (c0 - conc_profile[idx, :]) / c0
                                ax_bt.plot(time_grid_s, x, linewidth=2, label=species_name)
                                profile_df[f"X_{species_name}"] = x

                        ax_bt.set_xlabel("Time t [s]")
                        ax_bt.set_ylabel(
                            f"{profile_kind} [{('mol/m^3' if profile_kind.startswith('C') else '-')}]"
                        )
                        ax_bt.grid(True)
                        ax_bt.legend()
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
                            mime="image/png" if image_format_bt == "png" else "image/svg+xml",
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
                "📥 导出参数 (k0, Ea) CSV",
                df_param_export.to_csv(index=False).encode("utf-8"),
                file_name="fit_params_k0_ea.csv",
                mime="text/csv",
            )

            fitted_params_json = json.dumps(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fitted_params.items()},
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
                st.download_button(
                    "📥 导出预测 vs 实验对比（长表）CSV",
                    df_long.to_csv(index=False).encode("utf-8"),
                    file_name="pred_vs_meas_long.csv",
                    mime="text/csv",
                )
            else:
                st.info("先在「奇偶校验图」页生成对比数据后，再导出对比表。")

if __name__ == "__main__":
    main()
