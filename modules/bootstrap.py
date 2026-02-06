from __future__ import annotations

import json
import queue
import threading
import time

import streamlit as st
import streamlit.components.v1 as components

import modules.style as app_style
import modules.browser_storage as browser_storage
import modules.config_manager as config_manager
import modules.session_cleanup as session_cleanup
import modules.ui_components as ui_comp
import modules.ui_help as ui_help
from modules.config_state import (
    _apply_imported_config_to_widget_state,
    _clear_state_for_imported_config,
    _clear_state_for_reset_default,
)
from modules.fitting_background import FittingStoppedError, _drain_fitting_progress_queue
from modules.plot_helpers import _configure_matplotlib_chinese_font
from modules.constants import (
    DEFAULT_SESSION_MAX_AGE_HOURS,
    FITTING_STOP_WAIT_SLEEP_S,
    FITTING_STOP_WAIT_TRIES,
    SESSION_CLEANUP_EVERY_N_PAGE_LOADS,
)
from modules.upload_persistence import (
    _delete_persisted_upload,
    _load_persisted_upload,
    _read_csv_bytes_cached,
)


def bootstrap_app_state() -> dict:
    # ========= 会话 ID 初始化（多用户隔离） =========
    # 必须在其他操作之前初始化会话 ID
    browser_storage.inject_session_id_script()
    session_id = browser_storage.get_current_session_id()
    # 将 session_id 存储到 session_state 供后续使用
    if session_id:
        st.session_state["_current_session_id"] = session_id
        # 更新会话活跃标记，避免长时间打开页面的活跃会话被清理策略误删
        session_cleanup.update_session_activity(session_id)

    # ========= 定期清理过期会话 =========
    # 每 20 次页面加载执行一次清理（避免频繁 IO）
    cleanup_counter = st.session_state.get("_cleanup_counter", 0) + 1
    st.session_state["_cleanup_counter"] = cleanup_counter
    if cleanup_counter % SESSION_CLEANUP_EVERY_N_PAGE_LOADS == 1:
        session_cleanup.cleanup_old_sessions(
            max_age_hours=DEFAULT_SESSION_MAX_AGE_HOURS
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
        st.session_state["fitting_needs_app_rerun"] = False

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
        for _ in range(FITTING_STOP_WAIT_TRIES):
            if fitting_future and fitting_future.done():
                break
            time.sleep(FITTING_STOP_WAIT_SLEEP_S)

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
        st.session_state["fitting_needs_app_rerun"] = False
        # 同步局部状态，避免本轮脚本误入“仍在运行”分支并触发额外 sleep/rerun。
        fitting_running = False

        try:
            fit_results = fitting_future.result()
            _drain_fitting_progress_queue()
            st.session_state["fit_results"] = fit_results
            st.session_state["fitting_status"] = "拟合完成。"
            phi_value = float(
                fit_results.get("phi_final", fit_results.get("cost", 0.0))
            )
            phi_text = ui_comp.smart_float_to_str(phi_value)
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
            st.session_state["fit_notice"] = {
                "kind": "error",
                "text": f"Fitting Error: {exc}",
            }

    # --- 一次性提示（用于"拟合完成/失败/终止"/按钮回调报错等消息）---
    # 注意：fit_notice 会在 tab_fit 内部显示，而不是在这里
    # 这是为了避免在 tabs 之前动态添加元素导致 tabs 状态重置

    # --- 重置为默认：延迟到“下一次 rerun”再清理（避免修改已创建 widget 的 session_state）---
    if bool(st.session_state.pop("pending_reset_to_default", False)):
        ok, message = config_manager.clear_auto_saved_config(session_id)
        if not ok:
            st.warning(message)
        # 同时清除浏览器 LocalStorage 中的配置
        browser_storage.clear_browser_config()
        # 同时删除“实验数据”中已缓存的上传文件（否则下次启动会被自动恢复）
        ok, message = _delete_persisted_upload(session_id)
        if not ok:
            st.warning(message)
        _clear_state_for_reset_default()
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
            # 先清空旧的控件状态，防止与新导入的配置冲突导致 Streamlit 告警
            _clear_state_for_imported_config()
            # 将导入配置写入 cfg_*，确保 widgets 首次创建时就使用导入值
            _apply_imported_config_to_widget_state(pending_cfg)
            st.session_state["imported_config"] = pending_cfg
            # 保存到本地文件系统
            ok, message = config_manager.auto_save_config(pending_cfg, session_id)
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
            saved_config, load_message = config_manager.auto_load_config(session_id)
            if saved_config is not None:
                is_valid, error_message = config_manager.validate_config(saved_config)
                if is_valid:
                    _apply_imported_config_to_widget_state(saved_config)
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
                        _apply_imported_config_to_widget_state(browser_config)
                        st.session_state["imported_config"] = browser_config
                    else:
                        pass  # 静默忽略无效的浏览器缓存
                elif str(load_message).startswith("自动加载失败"):
                    st.warning(load_message)

    # 注入浏览器配置加载脚本（用于 Streamlit Cloud）
    # 如果刚刚重置，则不注入脚本，避免加载旧配置
    if not just_reset:
        browser_storage.inject_config_loader_script()

    # 若浏览器配置恢复失败，给出一次性提示（避免无休止刷新时用户无从下手）
    browser_cfg_error = str(
        st.session_state.get("_browser_config_load_error", "")
    ).strip()
    if browser_cfg_error and (not bool(st.session_state.get("_warned_browser_cfg_error", False))):
        st.session_state["_warned_browser_cfg_error"] = True
        st.warning(browser_cfg_error + "（已停止自动恢复；如需重试可在侧边栏执行“重置为默认”或清除浏览器缓存配置。）")

    def get_cfg(key, default):
        """
        读取当前配置的统一入口（优先读取当前 UI 的 cfg_* 状态，其次读取 imported_config）。

        说明：
        - Streamlit widgets 使用 key="cfg_xxx" 持久化当前值；
        - imported_config 仅用于“初始默认值/导入后恢复”。
        """
        cfg_key = f"cfg_{str(key).strip()}"
        if cfg_key in st.session_state:
            return st.session_state.get(cfg_key, default)
        imported_cfg = st.session_state.get("imported_config", None)
        if isinstance(imported_cfg, dict):
            return imported_cfg.get(key, default)
        return default

    # 初始化“停止拟合”标志与拟合状态
    if "fitting_stopped" not in st.session_state:
        st.session_state["fitting_stopped"] = False
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
    if "fitting_needs_app_rerun" not in st.session_state:
        st.session_state["fitting_needs_app_rerun"] = False

    # --- 恢复缓存的已上传 CSV（浏览器刷新后仍保留）---
    if "uploaded_csv_bytes" not in st.session_state:
        uploaded_bytes, uploaded_name, message = _load_persisted_upload(session_id)
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
    # Matplotlib 的 style 可能会覆盖字体设置，这里再强制一次以确保图中中文可显示
    _configure_matplotlib_chinese_font()

    @st.dialog("使用指南与帮助")
    def _show_help_dialog() -> None:
        ui_help.render_help_page()
    return {
        "session_id": session_id,
        "main_tab_labels": MAIN_TAB_LABELS,
        "set_active_main_tab": _set_active_main_tab,
        "restore_active_main_tab": _restore_active_main_tab,
        "request_start_fitting": _request_start_fitting,
        "request_stop_fitting": _request_stop_fitting,
        "get_cfg": get_cfg,
        "show_help_dialog": _show_help_dialog,
    }

