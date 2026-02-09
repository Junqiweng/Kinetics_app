from __future__ import annotations

import hashlib

import streamlit as st

import modules.config_manager as config_manager
import modules.ui_text as ui_text
from modules.constants import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    KINETIC_MODELS,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPES,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_PFR,
    UI_TOLERANCE_FORMAT_STREAMLIT,
)


def render_sidebar(ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    show_help_dialog = ctx["show_help_dialog"]
    # ========= 侧边栏 =========
    export_config_placeholder = None
    with st.sidebar:
        st.markdown("### 全局设置")
        global_disabled = bool(st.session_state.get("fitting_running", False))
        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            help_btn = st.button("使用帮助", width="stretch")
            if help_btn:
                show_help_dialog()

        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            st.markdown("#### 核心模型")
            reactor_type_default = str(get_cfg("reactor_type", REACTOR_TYPE_PFR)).strip()
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

            st.markdown("#### 求解器")
            solver_method = st.selectbox(
                "求解方法（Method）",
                ["RK45", "BDF", "Radau"],
                index=["RK45", "BDF", "Radau"].index(get_cfg("solver_method", "RK45")),
                format_func=lambda x: ui_text.map_label(
                    ui_text.SOLVER_METHOD_LABELS, str(x)
                ),
                key="cfg_solver_method",
                disabled=global_disabled,
            )
            col_tol1, col_tol2 = st.columns(2)
            rtol = col_tol1.number_input(
                "rtol",
                value=get_cfg("rtol", DEFAULT_RTOL),
                format=UI_TOLERANCE_FORMAT_STREAMLIT,
                key="cfg_rtol",
                disabled=global_disabled,
            )
            atol = col_tol2.number_input(
                "atol",
                value=get_cfg("atol", DEFAULT_ATOL),
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
    return {
        "export_config_placeholder": export_config_placeholder,
        "reactor_type": reactor_type,
        "pfr_flow_model": pfr_flow_model,
        "kinetic_model": kinetic_model,
        "solver_method": solver_method,
        "rtol": rtol,
        "atol": atol,
    }
