# 文件作用：Streamlit 主应用入口，提供反应动力学拟合的交互式界面（模型设置、数据上传、拟合与结果展示）。

from __future__ import annotations

import streamlit as st

from modules.app_bootstrap import bootstrap_app_state
from modules.app_plot_helpers import apply_runtime_patches
from modules.app_sidebar import render_sidebar
from modules.app_tab_data import render_data_tab
from modules.app_tab_fit import render_fit_tab
from modules.app_tab_model import render_model_tab
from modules.constants import REACTOR_TYPE_CSTR, REACTOR_TYPE_PFR


def main() -> None:
    st.set_page_config(
        page_title="Kinetics_app | 反应动力学拟合", layout="wide", page_icon="⚗️"
    )
    apply_runtime_patches()

    bootstrap_state = bootstrap_app_state()
    sidebar_state = render_sidebar(
        {
            "get_cfg": bootstrap_state["get_cfg"],
            "show_help_dialog": bootstrap_state["show_help_dialog"],
        }
    )

    reactor_type = sidebar_state["reactor_type"]
    st.title(f"{reactor_type} 反应动力学参数拟合")
    if reactor_type == REACTOR_TYPE_PFR:
        st.caption(r"模型：$\frac{dF_i}{dV} = \sum_j \nu_{ij} r_j$")
    elif reactor_type == REACTOR_TYPE_CSTR:
        st.caption(r"模型：$F_{i,0} - F_i + V \sum_j \nu_{ij} r_j = 0$")
    else:
        st.caption(r"模型：$\frac{dC_i}{dt} = \sum_j \nu_{ij} r_j$")

    tab_model, tab_data, tab_fit = st.tabs(bootstrap_state["main_tab_labels"])
    bootstrap_state["restore_active_main_tab"]()

    base_ctx = {**bootstrap_state, **sidebar_state}
    model_state = render_model_tab(tab_model, base_ctx)
    data_state = render_data_tab(tab_data, {**base_ctx, **model_state})
    render_fit_tab(tab_fit, {**base_ctx, **model_state, **data_state})


if __name__ == "__main__":
    main()
