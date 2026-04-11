from __future__ import annotations

import hashlib
import math

import streamlit as st

import modules.config_manager as config_manager
import modules.ui_text as ui_text
from modules.constants import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODELS,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPES,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
    UI_TOLERANCE_FORMAT_STREAMLIT,
)


def render_sidebar(ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    show_help_dialog = ctx["show_help_dialog"]

    def _safe_positive_tolerance(value, fallback: float, lower: float = 1e-15) -> float:
        try:
            x = float(value)
        except (TypeError, ValueError, OverflowError):
            x = float(fallback)
        if not math.isfinite(x):
            x = float(fallback)
        return float(max(float(lower), x))

    # ========= 侧边栏 =========
    export_config_placeholder = None
    with st.sidebar:
        st.markdown("### 全局设置")
        global_disabled = bool(st.session_state.get("fitting_running", False))
        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            col_help, col_new = st.columns(2)
            with col_help:
                help_btn = st.button("使用帮助", width="stretch")
                if help_btn:
                    show_help_dialog()
            with col_new:
                if st.button("新建项目", width="stretch", disabled=global_disabled):
                    st.session_state["pending_reset_to_default"] = True
                    st.rerun()

        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            st.markdown("#### 核心模型")
            reactor_type_default = str(
                get_cfg("reactor_type", REACTOR_TYPE_PFR)
            ).strip()
            if reactor_type_default == "Batch":
                reactor_type_default = REACTOR_TYPE_BSTR
            reactor_type_options = list(REACTOR_TYPES)
            if reactor_type_default not in reactor_type_options:
                reactor_type_default = REACTOR_TYPE_PFR
            reactor_type = st.selectbox(
                "反应器",
                reactor_type_options,
                index=reactor_type_options.index(reactor_type_default),
                format_func=lambda x: ui_text.map_label(
                    ui_text.REACTOR_TYPE_LABELS, str(x)
                ),
                key="cfg_reactor_type",
                disabled=global_disabled,
            )

            # PFR 流动模型/相态（仅 PFR 需要）
            pfr_flow_model_default = str(
                get_cfg("pfr_flow_model", PFR_FLOW_MODEL_LIQUID_CONST_VDOT)
            ).strip()
            if pfr_flow_model_default not in (
                PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
                PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
            ):
                pfr_flow_model_default = PFR_FLOW_MODEL_LIQUID_CONST_VDOT
            pfr_flow_model = pfr_flow_model_default
            if reactor_type == REACTOR_TYPE_PFR:
                pfr_flow_model_options = [
                    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
                    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
                ]
                pfr_flow_model = st.selectbox(
                    "PFR 流动模型",
                    pfr_flow_model_options,
                    index=pfr_flow_model_options.index(pfr_flow_model_default),
                    format_func=lambda x: ui_text.map_label(
                        ui_text.PFR_FLOW_MODEL_LABELS, str(x)
                    ),
                    key="cfg_pfr_flow_model",
                    disabled=global_disabled,
                    help=(
                        "液相：vdot 近似恒定，C=F/vdot。\n"
                        "气相：理想气体、等温、恒压 P（不考虑压降），C=y·P/(R·T)，入口强制 F0_*。"
                    ),
                )
            kinetic_model_default = str(get_cfg("kinetic_model", KINETIC_MODELS[0]))
            if kinetic_model_default not in KINETIC_MODELS:
                kinetic_model_default = KINETIC_MODELS[0]
            kinetic_model = st.selectbox(
                "动力学",
                KINETIC_MODELS,
                index=KINETIC_MODELS.index(kinetic_model_default),
                format_func=lambda x: ui_text.map_label(
                    ui_text.KINETIC_MODEL_LABELS, str(x)
                ),
                key="cfg_kinetic_model",
                disabled=global_disabled,
            )
            reversible_enabled = st.checkbox(
                "启用可逆反应",
                value=bool(get_cfg("reversible_enabled", False)),
                key="cfg_reversible_enabled",
                disabled=global_disabled,
                help="可逆反应作为动力学扩展选项，可与幂律或 Langmuir-Hinshelwood 联合使用。",
            )

        with st.expander("求解器设置", expanded=False):
            solver_method_options = ["LSODA", "RK45", "BDF", "Radau"]
            solver_method_default = str(get_cfg("solver_method", "LSODA")).strip()
            if solver_method_default not in solver_method_options:
                solver_method_default = "LSODA"
            solver_method = st.selectbox(
                "求解方法（Method）",
                solver_method_options,
                index=solver_method_options.index(solver_method_default),
                format_func=lambda x: ui_text.map_label(
                    ui_text.SOLVER_METHOD_LABELS, str(x)
                ),
                key="cfg_solver_method",
                disabled=global_disabled,
            )
            # 刚性系统智能提示：LHHW 模型或 CSTR 反应器通常产生刚性 ODE
            _is_stiff_scenario = (
                kinetic_model == KINETIC_MODEL_LANGMUIR_HINSHELWOOD
                or reactor_type == REACTOR_TYPE_CSTR
            )
            if _is_stiff_scenario and solver_method in ("RK45", "RK23"):
                st.caption(
                    "建议：当前模型/反应器可能产生刚性 ODE，"
                    "切换到 **LSODA** 或 **BDF** 可显著提升求解速度。"
                )
            col_tol1, col_tol2 = st.columns(2)
            rtol = col_tol1.number_input(
                "rtol",
                value=_safe_positive_tolerance(get_cfg("rtol", DEFAULT_RTOL), DEFAULT_RTOL),
                min_value=1e-15,
                format=UI_TOLERANCE_FORMAT_STREAMLIT,
                key="cfg_rtol",
                disabled=global_disabled,
            )
            atol = col_tol2.number_input(
                "atol",
                value=_safe_positive_tolerance(get_cfg("atol", DEFAULT_ATOL), DEFAULT_ATOL),
                min_value=1e-15,
                format=UI_TOLERANCE_FORMAT_STREAMLIT,
                key="cfg_atol",
                disabled=global_disabled,
            )

        # 配置管理
        with st.expander("配置管理（导入、导出、重置）"):
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
                            # 生成变更摘要
                            old_cfg = st.session_state.get("imported_config", {})
                            if isinstance(old_cfg, dict):
                                diff_keys = []
                                compare_keys = [
                                    "reactor_type", "kinetic_model", "reversible_enabled",
                                    "solver_method", "rtol", "atol",
                                    "species_text", "n_reactions", "output_mode",
                                ]
                                for ck in compare_keys:
                                    old_val = old_cfg.get(ck, None)
                                    new_val = cfg.get(ck, None)
                                    if old_val != new_val and new_val is not None:
                                        diff_keys.append(f"**{ck}**: {old_val} → {new_val}")
                                # 数组类型的 key 只报"已变更"
                                for ak in ["k0_guess", "ea_guess_J_mol", "stoich_matrix", "order_guess"]:
                                    old_v = old_cfg.get(ak, None)
                                    new_v = cfg.get(ak, None)
                                    if new_v is not None and str(old_v) != str(new_v):
                                        diff_keys.append(f"**{ak}**: 已变更")
                                if diff_keys:
                                    st.session_state["config_import_diff"] = diff_keys
                                else:
                                    st.session_state["config_import_diff"] = ["无明显差异"]
                            st.session_state["imported_config_digest"] = file_digest
                            st.session_state["pending_imported_config"] = cfg
                            st.success("导入成功！正在应用配置并刷新页面...")
                            st.rerun()
                except Exception as exc:
                    st.error(f"导入配置失败（JSON/编码错误）：{exc}")

            # 显示上次导入的变更摘要
            config_diff = st.session_state.pop("config_import_diff", None)
            if isinstance(config_diff, list) and config_diff:
                st.markdown("**导入变更摘要：**")
                for item in config_diff:
                    st.markdown(f"- {item}")

            export_config_placeholder = st.empty()

            if st.button("重置为默认", disabled=global_disabled):
                st.session_state["pending_reset_to_default"] = True
                st.rerun()
    return {
        "export_config_placeholder": export_config_placeholder,
        "reactor_type": reactor_type,
        "pfr_flow_model": pfr_flow_model,
        "kinetic_model": kinetic_model,
        "reversible_enabled": bool(reversible_enabled),
        "solver_method": solver_method,
        "rtol": rtol,
        "atol": atol,
    }
