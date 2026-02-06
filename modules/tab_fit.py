from __future__ import annotations

import streamlit as st

from modules.fit_advanced import render_fit_advanced
from modules.fit_execution import render_fit_actions
from modules.fit_results import render_fit_results


def render_fit_tab(tab_fit, ctx: dict) -> dict:
    data_df = ctx["data_df"]
    output_species_list = ctx["output_species_list"]

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

        fit_ctx = dict(ctx)
        fit_ctx["data_df"] = data_df
        fit_ctx["output_species_list"] = output_species_list

        fit_advanced_state = render_fit_advanced(fit_ctx)
        runtime_state = render_fit_actions(fit_ctx, fit_advanced_state)

    render_fit_results(
        runtime_state["tab_fit_results_container"],
        fit_ctx,
        fit_advanced_state,
        runtime_state,
    )
    return {
        "data_df": data_df,
        "output_species_list": output_species_list,
    }

