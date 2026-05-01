from __future__ import annotations

import hashlib
import os

import pandas as pd
import streamlit as st

import modules.config_manager as config_manager
from modules.constants import OUTPUT_MODE_COUT, REACTOR_TYPE_BSTR
from modules.ai_session import (
    build_ai_snapshot,
    build_codex_prompt,
    build_config_diff_rows,
    call_codex_agent,
    extract_action_block,
    find_codex_cli,
    prepare_config_action,
)
from modules.ui_help import _build_example_batch_csv_bytes
from modules.upload_persistence import _read_csv_bytes_cached, _save_persisted_upload


def _build_bstr_example_config() -> dict:
    """
    构建 BSTR 示例配置。

    示例数据来自一阶幂律反应 A -> B：
    - 目标测量列为 Cout_A_mol_m3；
    - 固定 Ea 与反应级数，只拟合 k0，避免单一温度下 k0/Ea 强相关导致参数不唯一。
    """
    cfg = config_manager.get_default_config()
    cfg.update(
        {
            "reactor_type": REACTOR_TYPE_BSTR,
            "kinetic_model": "power_law",
            "reversible_enabled": False,
            "solver_method": "LSODA",
            "species_text": "A,B",
            "n_reactions": 1,
            "stoich_matrix": [[-1.0], [1.0]],
            "order_guess": [[1.0, 0.0]],
            "fit_order_flags_matrix": [[False, False]],
            "k0_guess": [1.0e5],
            "ea_guess_J_mol": [5.0e4],
            "fit_k0_flags": [True],
            "fit_ea_flags": [False],
            "output_mode": OUTPUT_MODE_COUT,
            "output_species_list": ["A"],
            "residual_type": "相对残差",
            "use_multi_start": True,
            "n_starts": 3,
            "max_nfev_coarse": 300,
            "max_nfev": 1200,
            "random_seed": 42,
            "use_log_k0_fit": True,
        }
    )
    return cfg


def _queue_bstr_example_fit(ctx: dict) -> tuple[bool, str]:
    uploaded_bytes = _build_example_batch_csv_bytes()
    uploaded_name = "bstr_example_first_order.csv"
    data_df = _read_csv_bytes_cached(uploaded_bytes)
    csv_hash = hashlib.sha256(uploaded_bytes).hexdigest()

    st.session_state["uploaded_csv_bytes"] = uploaded_bytes
    st.session_state["uploaded_csv_name"] = uploaded_name
    st.session_state["data_df_cached"] = data_df.copy()
    st.session_state["_data_df_source_hash"] = csv_hash
    st.session_state["_data_df_upload_token"] = "ai_bstr_example_" + csv_hash[:12]

    ok, message = _save_persisted_upload(
        uploaded_bytes,
        uploaded_name,
        ctx.get("session_id", None),
    )
    if not ok:
        return False, message

    cfg = _build_bstr_example_config()
    is_valid, error_message = config_manager.validate_config(cfg)
    if not is_valid:
        return False, "示例配置校验失败：" + str(error_message)

    st.session_state["pending_imported_config"] = cfg
    st.session_state["ai_auto_start_fit_after_import"] = True
    st.session_state["ai_session_notice"] = "已载入 BSTR 示例数据与配置，并准备启动示例拟合。"
    return True, "OK"


def _init_ai_session_state() -> None:
    if "ai_chat_messages" not in st.session_state:
        st.session_state["ai_chat_messages"] = []
    if "ai_pending_config_action" not in st.session_state:
        st.session_state["ai_pending_config_action"] = None


def _append_message(role: str, content: str) -> None:
    messages = st.session_state.get("ai_chat_messages", [])
    if not isinstance(messages, list):
        messages = []
    messages.append({"role": str(role), "content": str(content)})
    st.session_state["ai_chat_messages"] = messages


def _render_pending_action() -> None:
    pending = st.session_state.get("ai_pending_config_action", None)
    if not isinstance(pending, dict):
        return

    st.markdown("#### 待应用的智能修改")
    summary = str(pending.get("summary", "")).strip()
    if summary:
        st.info(summary)

    diff_rows = pending.get("diff_rows", [])
    if isinstance(diff_rows, list) and diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)

    can_run_fit = bool(pending.get("can_run_fit", False))
    col_apply, col_apply_run, col_discard = st.columns([1, 1, 1])
    with col_apply:
        if st.button("应用修改", width="stretch", key="ai_apply_config_only"):
            st.session_state["pending_imported_config"] = pending["new_config"]
            st.session_state["ai_session_notice"] = "已应用智能会话给出的配置修改。"
            st.session_state["ai_pending_config_action"] = None
            st.rerun()
    with col_apply_run:
        if st.button(
            "应用并重新拟合",
            width="stretch",
            key="ai_apply_config_and_fit",
            disabled=not can_run_fit,
            help="需要已有 CSV 数据和目标物种，才能自动启动拟合。",
        ):
            st.session_state["pending_imported_config"] = pending["new_config"]
            st.session_state["ai_auto_start_fit_after_import"] = True
            st.session_state["ai_session_notice"] = "已应用智能会话给出的配置修改，并准备重新拟合。"
            st.session_state["ai_pending_config_action"] = None
            st.rerun()
    with col_discard:
        if st.button("放弃修改", width="stretch", key="ai_discard_config_action"):
            st.session_state["ai_pending_config_action"] = None
            st.rerun()


def _handle_user_message(user_message: str, ctx: dict) -> None:
    snapshot, current_config, config_error = build_ai_snapshot(ctx)
    _append_message("user", user_message)

    prompt = build_codex_prompt(
        user_message=user_message,
        chat_history=st.session_state.get("ai_chat_messages", []),
        snapshot=snapshot,
    )
    ok, codex_message = call_codex_agent(
        prompt=prompt,
        cwd=os.getcwd(),
    )
    if not ok:
        _append_message("assistant", "Codex CLI 调用失败：\n\n" + codex_message)
        return

    clean_message, action, parse_error = extract_action_block(codex_message)
    if parse_error:
        clean_message = clean_message + "\n\n> " + parse_error
    _append_message("assistant", clean_message)

    if not isinstance(action, dict):
        return

    new_config, action_error = prepare_config_action(
        current_config=current_config,
        action=action,
    )
    if action_error:
        _append_message("assistant", "智能修改未进入待应用状态：" + action_error)
        return

    config_patch = action.get("config_patch", {})
    patch_keys = list(config_patch.keys()) if isinstance(config_patch, dict) else []
    can_run_fit = (
        ctx.get("data_df", None) is not None
        and bool(new_config.get("output_species_list", []))
    )
    auto_apply_and_run = (
        bool(st.session_state.get("ai_auto_apply_and_run", False))
        and bool(action.get("run_fit_after_apply", False))
        and bool(can_run_fit)
    )
    if auto_apply_and_run:
        st.session_state["pending_imported_config"] = new_config
        st.session_state["ai_auto_start_fit_after_import"] = True
        st.session_state["ai_session_notice"] = "已根据智能会话建议自动应用配置，并准备重新拟合。"
        _append_message("assistant", "已根据当前开关自动提交配置修改，并将在页面刷新后启动重新拟合。")
        return

    st.session_state["ai_pending_config_action"] = {
        "summary": str(action.get("summary", "智能会话建议应用以下配置修改。")).strip(),
        "new_config": new_config,
        "run_fit_after_apply": bool(action.get("run_fit_after_apply", False)),
        "can_run_fit": bool(can_run_fit),
        "diff_rows": build_config_diff_rows(
            old_config=current_config or {},
            new_config=new_config,
            patch_keys=patch_keys,
        ),
    }


def render_ai_session_tab(tab_ai, ctx: dict) -> None:
    """
    渲染“智能会话”页面。

    第一阶段：通过本机 Codex CLI 完成自然语言诊断和建模建议。
    第二阶段：当 Codex 给出合法 config_patch 时，允许用户一键应用配置并重新拟合。
    """
    _init_ai_session_state()

    with tab_ai:
        notice = str(st.session_state.pop("ai_session_notice", "")).strip()
        if notice:
            st.success(notice)

        st.markdown("#### 智能建模会话")
        st.caption(
            "本功能调用本机 Codex CLI，不在网页代码中配置 OpenAI API key。当前阶段只允许自动应用配置修改，不允许自动改代码。"
        )

        codex_path = find_codex_cli()
        if codex_path:
            st.caption(f"Codex CLI：`{codex_path}`")
        else:
            st.warning("未找到本机 codex 命令。请先在终端确认 `codex --help` 可正常运行。")

        col_example, col_auto = st.columns([1, 2])
        with col_example:
            if st.button("载入示例并拟合", width="stretch", key="ai_run_bstr_example"):
                ok, message = _queue_bstr_example_fit(ctx)
                if not ok:
                    st.error(message)
                else:
                    st.rerun()
        with col_auto:
            st.checkbox(
                "收到合法建议后自动应用并重新拟合",
                value=bool(st.session_state.get("ai_auto_apply_and_run", False)),
                key="ai_auto_apply_and_run",
                help="仅当 Codex 返回合法 config_patch、当前已有数据、且建议明确要求重新拟合时生效。",
            )

        snapshot, current_config, config_error = build_ai_snapshot(ctx)
        col_state_1, col_state_2, col_state_3, col_state_4 = st.columns(4)
        col_state_1.metric("模型配置", "可用" if current_config else "未就绪")
        col_state_2.metric(
            "实验数据",
            "已加载" if snapshot.get("data_summary", {}).get("available") else "未加载",
        )
        col_state_3.metric(
            "拟合结果",
            "已有" if snapshot.get("fit_results_summary", {}).get("available") else "暂无",
        )
        col_state_4.metric(
            "待应用修改",
            "有" if isinstance(st.session_state.get("ai_pending_config_action"), dict) else "无",
        )
        if config_error:
            st.warning("当前配置快照存在问题：" + str(config_error))

        _render_pending_action()

        st.divider()
        for message in st.session_state.get("ai_chat_messages", []):
            role = str(message.get("role", "assistant"))
            content = str(message.get("content", ""))
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(content)

        with st.form("ai_session_input_form", clear_on_submit=True):
            user_message = st.text_area(
                "输入你的建模或拟合问题",
                placeholder="例如：请检查当前拟合为什么不收敛，并给出可以自动应用的配置修改。",
                height=110,
                key="ai_session_user_message",
            )
            col_send, col_clear = st.columns([1, 1])
            send_clicked = col_send.form_submit_button(
                "发送给 Codex",
                width="stretch",
                disabled=(not bool(codex_path)),
            )
            clear_clicked = col_clear.form_submit_button("清空本次会话", width="stretch")

        if clear_clicked:
            st.session_state["ai_chat_messages"] = []
            st.session_state["ai_pending_config_action"] = None
            st.rerun()

        if send_clicked:
            user_message = str(user_message).strip()
            if not user_message:
                st.warning("请输入问题后再发送。")
            else:
                with st.spinner("正在调用本机 Codex 进行诊断..."):
                    _handle_user_message(user_message, ctx)
                st.rerun()
