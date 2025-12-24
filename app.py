# 文件作用：Streamlit 主应用入口，提供反应动力学拟合的交互式界面（模型设置、数据上传、拟合与结果展示）。

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

import modules.fitting as fitting
import modules.reactors as reactors
import modules.ui_help as ui_help
import modules.config_manager as config_manager
import modules.ui_components as ui_comp  # UI 组件工具函数
import modules.browser_storage as browser_storage  # 浏览器 LocalStorage 持久化
import modules.app_style as app_style

from modules.app_config_state import (
    _apply_imported_config_to_widget_state,
    _clear_config_related_state,
    _warn_once,
)
from modules.app_data_utils import (
    _build_default_nu_table,
    _build_fit_comparison_long_table,
    _clean_species_names,
    _get_measurement_column_name,
    _get_output_unit_text,
)
from modules.app_fitting_background import (
    FittingStoppedError,
    _count_fitted_parameters,
    _drain_fitting_progress_queue,
    _get_fitting_executor,
    _render_fitting_live_progress,
    _render_fitting_progress_panel,
    _reset_fitting_progress_ui_state,
    _run_fitting_job,
)
from modules.upload_persistence import (
    _delete_persisted_upload,
    _load_persisted_upload,
    _read_csv_bytes_cached,
    _save_persisted_upload,
)


# ========= 主应用入口 =========
def main():
    st.set_page_config(
        page_title="Kinetics_app | 反应动力学拟合", layout="wide", page_icon="⚗️"
    )

    MAIN_TAB_LABELS = ["反应与模型", "实验数据", "拟合与结果"]

    def _set_active_main_tab(tab_label: str) -> None:
        tab_label = str(tab_label).strip()
        if tab_label:
            st.session_state["active_main_tab_label"] = tab_label

    def _restore_active_main_tab() -> None:
        """
        Streamlit 原生 st.tabs 在 rerun 后可能会回到第一个 tab。

        关键改动：只允许“用户手动点击 Tab”改变当前页面。

        做法：在前端监听 Tab 按钮的 click 事件，将当前 Tab 文本写入
        sessionStorage；每次 rerun 时从 sessionStorage 读取并恢复。

        这样可以保证：重置默认/改反应数/拟合状态更新等任何 rerun 都不会导致
        “自动跳到别的页面”，除非用户自己点了 Tab。
        """
        tab_labels_json = json.dumps(MAIN_TAB_LABELS, ensure_ascii=False)
        default_label_json = json.dumps(MAIN_TAB_LABELS[0], ensure_ascii=False)
        components.html(
            f"""
            <script>
              const TAB_LABELS = {tab_labels_json};
              const DEFAULT_LABEL = {default_label_json};
              const STORAGE_KEY = "kinetics_active_main_tab_label_v1";
              function norm(s) {{ return (s || '').replace(/\\s+/g, ' ').trim(); }}

              function getButtons() {{
                // 页面里可能还有其它 st.tabs（例如帮助页的 tab），这里需要只锁定“主 Tabs”。
                const tabLists = window.parent.document.querySelectorAll('div[data-baseweb="tab-list"]');
                for (const tabList of tabLists) {{
                  const buttons = tabList.querySelectorAll('button[data-baseweb="tab"]');
                  if (!buttons || buttons.length === 0) continue;

                  const texts = Array.from(buttons).map(b => norm(b.innerText));
                  let isMain = true;
                  for (const label of TAB_LABELS) {{
                    if (texts.indexOf(norm(label)) < 0) {{
                      isMain = false;
                      break;
                    }}
                  }}
                  if (isMain) return buttons;
                }}

                // 兜底：若找不到明确的主 tabList，则返回全部（避免完全失效）。
                return window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
              }}

              function getActiveLabelFromDom() {{
                const buttons = getButtons();
                for (const btn of buttons) {{
                  const selected = btn.getAttribute("aria-selected");
                  if (selected === "true") {{
                    return norm(btn.innerText);
                  }}
                }}
                return "";
              }}

              function loadStoredLabel() {{
                try {{
                  const stored = window.parent.sessionStorage.getItem(STORAGE_KEY);
                  return norm(stored);
                }} catch (e) {{
                  return "";
                }}
              }}

              function saveStoredLabel(label) {{
                const labelNorm = norm(label);
                if (!labelNorm) return;
                try {{
                  window.parent.sessionStorage.setItem(STORAGE_KEY, labelNorm);
                }} catch (e) {{}}
              }}

              function installClickListeners() {{
                const buttons = getButtons();
                for (const btn of buttons) {{
                  if (btn.dataset.kineticsTabListenerInstalled === "1") continue;
                  btn.dataset.kineticsTabListenerInstalled = "1";
                  btn.addEventListener("click", () => {{
                    saveStoredLabel(btn.innerText);
                  }});
                }}
              }}

              function tryRestore() {{
                installClickListeners();

                // 第一次进入页面时，先把“当前激活 Tab”写入 storage，避免空值。
                const currentActive = getActiveLabelFromDom();
                if (currentActive) {{
                  const stored = loadStoredLabel();
                  if (!stored) saveStoredLabel(currentActive);
                }}

                let target = loadStoredLabel();
                if (!target) target = norm(DEFAULT_LABEL);
                if (TAB_LABELS.map(norm).indexOf(target) < 0) target = norm(DEFAULT_LABEL);

                // 如果当前已在目标 Tab，则不做任何动作（避免抖动）。
                const nowActive = getActiveLabelFromDom();
                if (nowActive && norm(nowActive) === norm(target)) return true;

                const buttons = getButtons();
                for (const btn of buttons) {{
                  if (norm(btn.innerText) === norm(target)) {{
                    btn.click();
                    return true;
                  }}
                }}
                return false;
              }}

              setTimeout(tryRestore, 30);
              setTimeout(tryRestore, 150);
              setTimeout(tryRestore, 600);
            </script>
            """,
            height=0,
        )

    def _request_start_fitting() -> None:
        """
        Start 按钮回调：先锁定全局设置（sidebar），再在本次 rerun 中启动后台任务。

        说明：回调会在脚本执行前触发，因此可以避免“点开始拟合后需要额外 st.rerun 导致跳 Tab”的问题。
        """
        data_df_cached = st.session_state.get("data_df_cached", None)
        output_species_list_cached = st.session_state.get("cfg_output_species_list", [])

        if data_df_cached is None:
            st.session_state["fit_notice"] = {
                "kind": "error",
                "text": "当前没有可用的 CSV 数据，请先在「实验数据」页面上传或恢复已缓存的文件。",
            }
            return

        if not output_species_list_cached:
            st.session_state["fit_notice"] = {
                "kind": "error",
                "text": "请选择至少一个目标物种。",
            }
            return

        if bool(st.session_state.get("fitting_running", False)):
            return

        st.session_state["start_fit_requested"] = True
        st.session_state["fitting_running"] = True
        st.session_state["fitting_stopped"] = False

    def _request_stop_fitting() -> None:
        """
        Stop 按钮回调：请求后台停止（不额外触发 st.rerun，避免跳 Tab）。
        """
        if not bool(st.session_state.get("fitting_running", False)):
            return
        st.session_state["stop_fit_requested"] = True
        stop_event = st.session_state.get("fitting_stop_event", None)
        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass

    # --- 在渲染 sidebar 之前处理“后台拟合完成/异常/丢失”等状态，确保全局设置能及时解锁 ---
    _drain_fitting_progress_queue()
    fitting_future = st.session_state.get("fitting_future", None)
    fitting_running = bool(st.session_state.get("fitting_running", False))

    # --- 优先处理停止请求，以便在本次 rerun 中立即生效 ---
    if bool(st.session_state.pop("stop_fit_requested", False)) and fitting_running:
        st.session_state["fitting_stopped"] = True
        st.session_state["fitting_status"] = "已发送终止请求，等待后台停止..."

        fitting_stop_event = st.session_state.get("fitting_stop_event", None)
        try:
            if fitting_stop_event:
                fitting_stop_event.set()
        except Exception:
            pass
        st.session_state["fitting_timeline"].append(("⏹️", "用户请求终止拟合..."))

        # 等待一小会儿，让后台任务有机会退出
        # 这样下面的 fitting_future.done() 就能立即检测到，无需第二次 rerun
        import time

        for _ in range(10):  # 最多等待 0.5 秒
            if fitting_future and fitting_future.done():
                break
            time.sleep(0.05)

    # 检查后台任务是否丢失（但排除“刚刚请求启动”的情况）
    start_fit_requested = bool(st.session_state.get("start_fit_requested", False))
    if fitting_running and (fitting_future is None) and (not start_fit_requested):
        st.session_state["fitting_running"] = False
        st.session_state["fitting_status"] = (
            "后台任务已丢失（可能是页面刷新导致）。请重新开始拟合。"
        )
        fitting_running = False

    if fitting_future is not None and fitting_future.done():
        st.session_state["fitting_running"] = False
        st.session_state["fitting_future"] = None

        try:
            fit_results = fitting_future.result()
            _drain_fitting_progress_queue()
            st.session_state["fit_results"] = fit_results
            st.session_state["fitting_status"] = "拟合完成。"
            phi_value = float(
                fit_results.get("phi_final", fit_results.get("cost", 0.0))
            )
            st.session_state["fitting_timeline"].append(
                ("✅", f"拟合完成，最终 Φ: {phi_value:.4e}")
            )
            st.session_state["fit_notice"] = {
                "kind": "success",
                "text": "拟合完成！结果已缓存（结果展示将锁定为本次拟合的配置与数据）。"
                f" 目标函数 Φ: {phi_value:.4e}",
            }
        except FittingStoppedError:
            st.session_state["fitting_status"] = "用户终止。"
            st.session_state["fitting_timeline"].append(("⚠️", "拟合已终止。"))
            st.session_state["fit_notice"] = {"kind": "warning", "text": "拟合已终止。"}
        except Exception as exc:
            st.session_state["fitting_status"] = "拟合失败。"
            st.session_state["fitting_timeline"].append(("❌", f"拟合失败: {exc}"))
            st.session_state["fit_notice"] = {
                "kind": "error",
                "text": f"Fitting Error: {exc}",
            }

    # --- 一次性提示（用于"拟合完成/失败/终止"/按钮回调报错等消息）---
    # 注意：fit_notice 会在 tab_fit 内部显示，而不是在这里
    # 这是为了避免在 tabs 之前动态添加元素导致 tabs 状态重置

    # --- 重置为默认：延迟到“下一次 rerun”再清理（避免修改已创建 widget 的 session_state）---
    if bool(st.session_state.pop("pending_reset_to_default", False)):
        ok, message = config_manager.clear_auto_saved_config()
        if not ok:
            st.warning(message)
        # 同时清除浏览器 LocalStorage 中的配置
        browser_storage.clear_browser_config()
        # 同时删除“实验数据”中已缓存的上传文件（否则下次启动会被自动恢复）
        ok, message = _delete_persisted_upload()
        if not ok:
            st.warning(message)
        _clear_config_related_state()
        # 标记已重置，阻止本次 rerun 时从浏览器加载旧配置
        st.session_state["_reset_just_happened"] = True
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
            # 保存到本地文件系统
            ok, message = config_manager.auto_save_config(pending_cfg)
            if not ok:
                st.warning(message)
            # 同时保存到浏览器 LocalStorage
            browser_storage.save_config_to_browser(pending_cfg)

    # --- 自动加载配置 ---
    # 如果刚刚执行了重置，则跳过配置加载（避免重新加载浏览器中的旧配置）
    just_reset = bool(st.session_state.pop("_reset_just_happened", False))

    if "config_initialized" not in st.session_state:
        st.session_state["config_initialized"] = True

        if not just_reset:
            # 方案1：尝试从本地文件系统加载（用于本地运行）
            saved_config, load_message = config_manager.auto_load_config()
            if saved_config is not None:
                is_valid, error_message = config_manager.validate_config(saved_config)
                if is_valid:
                    st.session_state["imported_config"] = saved_config
                else:
                    st.warning(f"自动恢复配置无效，已忽略：{error_message}")
            else:
                # 方案2：尝试从浏览器 LocalStorage 加载（用于 Streamlit Cloud）
                browser_config = browser_storage.get_browser_loaded_config()
                if browser_config is not None:
                    is_valid, error_message = config_manager.validate_config(
                        browser_config
                    )
                    if is_valid:
                        st.session_state["imported_config"] = browser_config
                    else:
                        pass  # 静默忽略无效的浏览器缓存
                elif str(load_message).startswith("自动加载失败"):
                    st.warning(load_message)

    # 注入浏览器配置加载脚本（用于 Streamlit Cloud）
    # 如果刚刚重置，则不注入脚本，避免加载旧配置
    if not just_reset:
        browser_storage.inject_config_loader_script()

    def get_cfg(key, default):
        if "imported_config" in st.session_state:
            return st.session_state["imported_config"].get(key, default)
        return default

    # 初始化“停止拟合”标志与拟合状态
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

    # --- 恢复缓存的已上传 CSV（浏览器刷新后仍保留）---
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
            if (
                "uploaded_csv_bytes" in st.session_state
                and st.session_state["uploaded_csv_bytes"]
            ):
                st.session_state["data_df_cached"] = _read_csv_bytes_cached(
                    st.session_state["uploaded_csv_bytes"]
                )
        except Exception as exc:
            if "data_restore_warned" not in st.session_state:
                st.session_state["data_restore_warned"] = True
                st.warning(f"恢复缓存 CSV 失败（请重新上传）：{exc}")

    # --- 样式 ---
    app_style.apply_app_css()
    app_style.apply_plot_style()

    @st.dialog("教程/帮助")
    def _show_help_dialog() -> None:
        ui_help.render_help_page()

    # ========= 侧边栏 =========
    export_config_placeholder = None
    with st.sidebar:
        st.markdown("### 全局设置")
        global_disabled = bool(st.session_state.get("fitting_running", False))
        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            help_btn = st.button("教程/帮助", use_container_width=True)
            if help_btn:
                _show_help_dialog()

        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            st.markdown("#### 核心模型")
            reactor_type_default = str(get_cfg("reactor_type", "PFR")).strip()
            if reactor_type_default == "Batch":
                reactor_type_default = "BSTR"
            reactor_type_options = ["PFR", "CSTR", "BSTR"]
            if reactor_type_default not in reactor_type_options:
                reactor_type_default = "PFR"
            reactor_type = st.selectbox(
                "反应器",
                reactor_type_options,
                index=reactor_type_options.index(reactor_type_default),
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

        # 配置管理
        with st.expander("⚙️ 配置管理 (导入/导出/重置)"):
            config_uploader_key = f"uploaded_config_json_{int(st.session_state.get('uploader_ver_config_json', 0))}"
            uploaded_config = st.file_uploader(
                "导入配置",
                type=["json"],
                key=config_uploader_key,
                disabled=global_disabled,
            )
            if uploaded_config:
                try:
                    uploaded_bytes = uploaded_config.getvalue()
                    file_digest = hashlib.sha256(uploaded_bytes).hexdigest()
                    if (
                        st.session_state.get("imported_config_digest", None)
                        == file_digest
                    ):
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

    # ========= 主内容区 =========
    st.title(f"{reactor_type} 反应动力学参数拟合")
    if reactor_type == "PFR":
        st.caption("模型：PFR (solve_ivp) + least_squares")
    elif reactor_type == "CSTR":
        st.caption("模型：CSTR (steady-state) + least_squares")
    else:
        st.caption("模型：BSTR (solve_ivp) + least_squares")

    tab_model, tab_data, tab_fit = st.tabs(MAIN_TAB_LABELS)
    _restore_active_main_tab()

    # ---------------- 选项卡 1：模型 ----------------
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

        # 化学计量数
        st.markdown("**化学计量数矩阵 ν** (行=物种, 列=反应)")
        nu_default = _build_default_nu_table(species_names, n_reactions)
        # 若已导入配置，则优先应用其中的化学计量数
        imp_stoich = get_cfg("stoich_matrix", None)
        if imp_stoich:
            try:
                arr = np.asarray(imp_stoich, dtype=float)
                if arr.ndim != 2:
                    raise ValueError(f"需要二维矩阵，实际维度={arr.ndim}")

                # 尺寸不匹配时：自动补齐/截断，并用 0 填充空缺（不提示警告）
                if arr.shape != nu_default.shape:
                    fixed = np.zeros(nu_default.shape, dtype=float)
                    n_rows = min(fixed.shape[0], arr.shape[0])
                    n_cols = min(fixed.shape[1], arr.shape[1])
                    fixed[:n_rows, :n_cols] = arr[:n_rows, :n_cols]
                    arr = fixed

                nu_default = pd.DataFrame(
                    arr, index=nu_default.index, columns=nu_default.columns
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

        # --- 基础参数（k0, Ea, n）---
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.caption("速率常数 k0 & 活化能 Ea")
            # 获取默认值的逻辑
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
                    "k₀",
                    k0_def,
                    "指前因子",
                    "Eₐ [J/mol]",
                    ea_def,
                    "活化能",
                    fit_k0_def,
                    fit_ea_def,
                )
            )

        with col_p2:
            st.caption("反应级数 n")
            # 反应级数默认值逻辑
            order_data = (
                np.array(
                    get_cfg("order_guess", np.zeros((n_reactions, len(species_names))))
                )
                if get_cfg("order_guess", None)
                else None
            )
            if order_data is None:
                # 简单默认规则
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

        # --- L-H 参数 ---
        K0_ads, Ea_K_J_mol, fit_K0_ads_flags, fit_Ea_K_flags = None, None, None, None
        m_inhibition, fit_m_flags = None, None

        if kinetic_model == "langmuir_hinshelwood":
            with st.expander("Langmuir-Hinshelwood (L-H) 参数", expanded=True):
                col_lh1, col_lh2 = st.columns(2)
                with col_lh1:
                    st.caption("吸附常数 K (每物种)")
                    # 默认值
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
                            "K₀,ads",
                            K0_def,
                            "吸附常数指前因子",
                            "Eₐ,K [J/mol]",
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
            # 为兼容旧版本，初始化为空
            K0_ads = np.zeros(len(species_names))
            Ea_K_J_mol = np.zeros(len(species_names))
            fit_K0_ads_flags = np.zeros(len(species_names), dtype=bool)
            fit_Ea_K_flags = np.zeros(len(species_names), dtype=bool)
            m_inhibition = np.ones(n_reactions)
            fit_m_flags = np.zeros(n_reactions, dtype=bool)

        # --- 可逆反应参数 ---
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
                            "k₀⁻",
                            k0r_def,
                            "逆反应指前因子",
                            "Eₐ⁻ [J/mol]",
                            ear_def,
                            "逆反应活化能",
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

    # ---------------- 选项卡 2：数据 ----------------
    data_df = st.session_state.get("data_df_cached", None)
    output_mode = "Cout (mol/m^3)"
    output_species_list = []

    with tab_data:
        # --- 拟合目标变量 + 物种选择（同一行）---
        col_target1, col_target2 = st.columns(2)
        with col_target1:
            opts = (
                ["Cout (mol/m^3)", "Fout (mol/s)"]
                if reactor_type in ("PFR", "CSTR")
                else ["Cout (mol/m^3)"]
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
                help="选择用于拟合的测量数据类型（Cout=出口浓度，Fout=出口摩尔流量）",
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
            else:
                meas_cols = [f"Cout_{s}_mol_m3" for s in species_names]

            # 根据反应器类型决定输入条件列
            if reactor_type == "PFR":
                cols = (
                    ["V_m3", "T_K", "vdot_m3_s"]
                    + [f"F0_{s}_mol_s" for s in species_names]
                    + meas_cols
                )
            elif reactor_type == "CSTR":
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
                    ok, message = _delete_persisted_upload()
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
                ok, message = _save_persisted_upload(uploaded_bytes, uploaded_name)
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
                st.dataframe(data_df.head(50), use_container_width=True, height=200)
            except Exception as exc:
                st.error(f"CSV 读取失败: {exc}")
                data_df = None

    # --- 构建导出配置（基础版；若拟合页启用高级设置，会在拟合页再次更新）---
    if export_config_placeholder is not None:
        export_k0_min = float(get_cfg("k0_min", 1e-15))
        export_k0_max = float(get_cfg("k0_max", 1e15))
        export_ea_min = float(get_cfg("ea_min_J_mol", 1e4))
        export_ea_max = float(get_cfg("ea_max_J_mol", 3e5))
        export_ord_min = float(get_cfg("order_min", -2.0))
        export_ord_max = float(get_cfg("order_max", 5.0))
        export_K0_ads_min = float(get_cfg("K0_ads_min", 0.0))
        export_K0_ads_max = float(get_cfg("K0_ads_max", 1e10))
        export_Ea_K_min = float(get_cfg("Ea_K_min", -2e5))
        export_Ea_K_max = float(get_cfg("Ea_K_max", 2e5))

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
            ok, message = config_manager.auto_save_config(export_cfg)
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

        data_len = len(data_df) if data_df is not None else 0

        # --- 高级设置（展开）---
        with st.expander("高级设置与边界 (点击展开)", expanded=False):

            st.markdown("**1. 基础边界设置**")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                k0_min = ui_comp.smart_number_input(
                    "k0 Min",
                    value=float(get_cfg("k0_min", 1e-15)),
                    key="cfg_k0_min",
                )
                k0_max = ui_comp.smart_number_input(
                    "k0 Max",
                    value=float(get_cfg("k0_max", 1e15)),
                    key="cfg_k0_max",
                )
            with col_b2:
                ea_min = ui_comp.smart_number_input(
                    "Ea Min",
                    value=float(get_cfg("ea_min_J_mol", 1e4)),
                    key="cfg_ea_min_J_mol",
                )
                ea_max = ui_comp.smart_number_input(
                    "Ea Max",
                    value=float(get_cfg("ea_max_J_mol", 3e5)),
                    key="cfg_ea_max_J_mol",
                )
            with col_b3:
                ord_min = ui_comp.smart_number_input(
                    "Order Min",
                    value=float(get_cfg("order_min", -2.0)),
                    key="cfg_order_min",
                )
                ord_max = ui_comp.smart_number_input(
                    "Order Max",
                    value=float(get_cfg("order_max", 5.0)),
                    key="cfg_order_max",
                )

            K0_ads_min = float(get_cfg("K0_ads_min", 0.0))
            K0_ads_max = float(get_cfg("K0_ads_max", 1e10))
            Ea_K_min = float(get_cfg("Ea_K_min", -2e5))
            Ea_K_max = float(get_cfg("Ea_K_max", 2e5))
            if kinetic_model == "langmuir_hinshelwood":
                st.markdown("**1.2 L-H 边界设置**")
                col_lh_b1, col_lh_b2 = st.columns(2)
                with col_lh_b1:
                    K0_ads_min = ui_comp.smart_number_input(
                        "K0_ads Min",
                        value=K0_ads_min,
                        key="cfg_K0_ads_min",
                    )
                    K0_ads_max = ui_comp.smart_number_input(
                        "K0_ads Max",
                        value=K0_ads_max,
                        key="cfg_K0_ads_max",
                    )
                with col_lh_b2:
                    Ea_K_min = ui_comp.smart_number_input(
                        "Ea_K Min",
                        value=Ea_K_min,
                        key="cfg_Ea_K_min",
                    )
                    Ea_K_max = ui_comp.smart_number_input(
                        "Ea_K Max",
                        value=Ea_K_max,
                        key="cfg_Ea_K_max",
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
                diff_step_rel = ui_comp.smart_number_input(
                    "diff_step (Finite Diff)",
                    value=get_cfg("diff_step_rel", 1e-2),
                    key="cfg_diff_step_rel",
                    help="提示：用于 least_squares 计算数值差分 Jacobian 的相对步长，找不到解时可尝试调大该值。",
                )
                max_step_fraction = ui_comp.smart_number_input(
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
                    "**绝对残差** (Absolute Residual)\n\n"
                    r"$r_i = y_i^{pred} - y_i^{meas}$"
                    "\n\n适用于：测量值数量级相近的数据。当测量值范围差异大时，大值主导拟合。"
                ),
                "相对残差": (
                    "**相对残差** (Relative Residual)\n\n"
                    r"$r_i = \frac{y_i^{pred} - y_i^{meas}}{y_i^{meas}}$"
                    "\n\n适用于：测量值跨越多个数量级的数据。对所有数据点给予相近权重。\n\n"
                    r"⚠️ 注意：若 $y_i^{meas}$ 接近零，残差会变得非常大，可能导致数值不稳定。"
                ),
                "百分比残差": (
                    "**百分比残差** (Percentage Residual with offset)\n\n"
                    r"$r_i = \frac{y_i^{pred} - y_i^{meas}}{|y_i^{meas}| + \epsilon}$"
                    "\n\n"
                    r"其中 $\epsilon$ 为小正数（典型值的 1%），避免除零。"
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
                ok, message = config_manager.auto_save_config(export_cfg)
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
            start_btn = col_act1.button(
                "🚀 开始拟合",
                type="primary",
                disabled=fitting_running,
                use_container_width=True,
                on_click=_request_start_fitting,
            )
            stop_btn = col_act2.button(
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
                help="开启后，页面会每隔约 2 秒自动刷新一次，用于持续更新拟合进度与阶段信息；关闭可减少卡顿/CPU 占用。",
            )
            col_interval_label, col_interval_input = col_act5.columns(
                [1.1, 1.4], vertical_alignment="center"
            )
            col_interval_label.markdown(
                '<div style="font-size:0.8rem; color:var(--muted,#6E6E73); white-space:nowrap;">间隔(s)</div>',
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
                    "title": "📊 拟合任务概览",
                    "lines": [
                        f"数据点数量: {int(len(data_df))} 行",
                        f"待拟合参数: {int(n_fit_params)} 个",
                        f"反应器类型: {reactor_type}",
                        f"动力学模型: {kinetic_model}",
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
            st.caption("“自动刷新”：仅刷新进度区域（避免整页闪烁）；若觉得卡顿可关闭。")
            refresh_interval_s = float(
                st.session_state.get("fitting_refresh_interval_s", 2.0)
            )
            if bool(st.session_state.get("fitting_auto_refresh", True)):
                st.fragment(
                    _render_fitting_live_progress, run_every=refresh_interval_s
                )()
            else:
                st.fragment(_render_fitting_live_progress)()
        elif st.session_state.get("fitting_timeline", []):
            _render_fitting_progress_panel()

        # 在拟合页底部创建结果容器
        tab_fit_results_container = st.container()

    # --- 结果展示（优化版）---
    if "fit_results" in st.session_state:
        res = st.session_state["fit_results"]
        tab_fit_results_container.divider()
        phi_value = float(res.get("phi_final", res.get("cost", 0.0)))
        tab_fit_results_container.markdown(
            f"### 拟合结果 (目标函数 Φ: {phi_value:.4e})"
        )
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
            res.get("max_step_fraction", get_cfg("max_step_fraction", 0.1))
        )
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
                parity_species_unavailable.append(
                    f"{sp_name}（列 {meas_col} 全为 NaN/非数字）"
                )

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
                        "k0 [SI]": fitted_params["k0"],
                        "Ea [J/mol]": fitted_params["ea_J_mol"],
                    },
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
                    if (
                        fitted_params.get("K0_ads", None) is not None
                        and fitted_params.get("Ea_K", None) is not None
                    ):
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
                if (
                    fitted_params.get("k0_rev", None) is not None
                    and fitted_params.get("ea_rev", None) is not None
                ):
                    df_rev = pd.DataFrame(
                        {
                            "k0_rev [SI]": fitted_params["k0_rev"],
                            "Ea_rev [J/mol]": fitted_params["ea_rev"],
                        },
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
                    st.caption(
                        "无法绘制奇偶校验图的物种： "
                        + "，".join(parity_species_unavailable)
                    )

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
                    st.session_state["fit_compare_long_df"] = (
                        _build_fit_comparison_long_table(
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
                n_cols = int(
                    st.number_input(
                        "每行子图数", min_value=1, max_value=4, value=2, step=1
                    )
                )

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
                        species_list_plot = list(
                            dict.fromkeys(df_ok["species"].tolist())
                        )
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
                            if (
                                np.isfinite(min_v)
                                and np.isfinite(max_v)
                                and max_v > min_v
                            ):
                                ax.plot(
                                    [min_v, max_v], [min_v, max_v], "k--", label="y=x"
                                )
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
                            mime=(
                                "image/png"
                                if image_format == "png"
                                else "image/svg+xml"
                            ),
                        )
                        plt.close(fig)

                if show_residual_plot:
                    st.markdown("#### 残差图 (Predicted - Measured)")
                    fig_r, ax_r = plt.subplots(figsize=(7.5, 4.8))
                    ax_r.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
                    for species_name in species_selected:
                        df_sp = df_long[df_long["species"] == species_name]
                        df_sp = df_sp[df_sp["ok"]]
                        df_sp = df_sp[
                            np.isfinite(df_sp["residual"])
                            & np.isfinite(df_sp["measured"])
                        ]
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
                        pd.DataFrame(rows_metric), use_container_width=True, height=220
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
                        "剖面点数", min_value=20, max_value=2000, value=200, step=20
                    )
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
                        for species_name in profile_species:
                            idx = name_to_index[species_name]
                            if profile_kind.startswith("F"):
                                y = molar_flow_profile[idx, :]
                                ax_pf.plot(
                                    volume_grid_m3, y, linewidth=2, label=species_name
                                )
                                profile_df[f"F_{species_name}_mol_s"] = y
                            else:
                                conc = molar_flow_profile[idx, :] / max(
                                    vdot_m3_s, 1e-30
                                )
                                ax_pf.plot(
                                    volume_grid_m3,
                                    conc,
                                    linewidth=2,
                                    label=species_name,
                                )
                                profile_df[f"C_{species_name}_mol_m3"] = conc

                        ax_pf.set_xlabel("Reactor volume V [m^3]")
                        ax_pf.set_ylabel(
                            f"{profile_kind} [{('mol/s' if profile_kind.startswith('F') else 'mol/m^3')}]"
                        )
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
                            mime=(
                                "image/png"
                                if image_format_pf == "png"
                                else "image/svg+xml"
                            ),
                        )
                        plt.close(fig_pf)

                elif reactor_type_fit == "CSTR":
                    profile_kind = st.radio(
                        "剖面变量",
                        ["C (mol/m^3)", "X (conversion)"],
                        index=0,
                        horizontal=True,
                    )
                    reactor_volume_m3 = float(row_sel.get("V_m3", np.nan))
                    temperature_K = float(row_sel.get("T_K", np.nan))
                    vdot_m3_s = float(row_sel.get("vdot_m3_s", np.nan))

                    conc_inlet = np.zeros(len(species_names_fit), dtype=float)
                    for i, sp_name in enumerate(species_names_fit):
                        conc_inlet[i] = float(
                            row_sel.get(f"C0_{sp_name}_mol_m3", np.nan)
                        )

                    tau_s = reactor_volume_m3 / max(vdot_m3_s, 1e-30)
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
                        for species_name in profile_species:
                            idx = name_to_index[species_name]
                            if profile_kind.startswith("C"):
                                y = conc_profile[idx, :]
                                ax_cs.plot(
                                    time_grid_s, y, linewidth=2, label=species_name
                                )
                                profile_df[f"C_{species_name}_mol_m3"] = y
                            else:
                                c0 = float(conc_inlet[idx])
                                if c0 < 1e-30:
                                    x = np.full_like(time_grid_s, np.nan, dtype=float)
                                else:
                                    x = (c0 - conc_profile[idx, :]) / c0
                                ax_cs.plot(
                                    time_grid_s, x, linewidth=2, label=species_name
                                )
                                profile_df[f"X_{species_name}"] = x

                        ax_cs.set_xlabel("Time t [s]")
                        ax_cs.set_ylabel(
                            f"{profile_kind} [{('mol/m^3' if profile_kind.startswith('C') else '-')}]"
                        )
                        ax_cs.grid(True)
                        ax_cs.legend()
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
                        for species_name in profile_species:
                            idx = name_to_index[species_name]
                            if profile_kind.startswith("C"):
                                y = conc_profile[idx, :]
                                ax_bt.plot(
                                    time_grid_s, y, linewidth=2, label=species_name
                                )
                                profile_df[f"C_{species_name}_mol_m3"] = y
                            else:
                                c0 = float(conc_initial[idx])
                                if c0 < 1e-30:
                                    x = np.full_like(time_grid_s, np.nan, dtype=float)
                                else:
                                    x = (c0 - conc_profile[idx, :]) / c0
                                ax_bt.plot(
                                    time_grid_s, x, linewidth=2, label=species_name
                                )
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
                "📥 导出参数 (k0, Ea) CSV",
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
