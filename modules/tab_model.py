from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules.config_state import _warn_once
from modules.data_utils import _build_default_nu_table, _clean_species_names, _parse_reaction_equations
from modules.fit_setup import resolve_fit_parameter_state


def _build_stoich_widget_reset_prefixes(species_count: int) -> list[str]:
    return [f"nu_{int(species_count)}_"]


def _apply_pending_model_widget_updates() -> None:
    pending = st.session_state.pop("pending_model_widget_updates", None)
    if not isinstance(pending, dict):
        return

    cfg_updates = pending.get("cfg_updates", {})
    if not isinstance(cfg_updates, dict):
        cfg_updates = {}

    config_updates = pending.get("config_updates", {})
    if not isinstance(config_updates, dict):
        config_updates = {}

    reset_prefixes = pending.get("reset_prefixes", [])
    if not isinstance(reset_prefixes, list):
        reset_prefixes = []

    imported_cfg = st.session_state.get("imported_config", {})
    if not isinstance(imported_cfg, dict):
        imported_cfg = {}
    merged_cfg = dict(imported_cfg)
    merged_cfg.update(config_updates)
    st.session_state["imported_config"] = merged_cfg

    for key, value in config_updates.items():
        st.session_state[f"cfg_{str(key)}"] = value

    for key, value in cfg_updates.items():
        st.session_state[str(key)] = value

    for key in list(st.session_state.keys()):
        key_str = str(key)
        if any(key_str.startswith(prefix) for prefix in reset_prefixes):
            del st.session_state[key]

    notice_text = str(pending.get("notice", "")).strip()
    if notice_text:
        st.session_state["model_tab_notice"] = notice_text


def render_model_tab(tab_model, ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    kinetic_model = ctx["kinetic_model"]
    reversible_enabled = bool(ctx.get("reversible_enabled", False))
    # ---------------- 选项卡 1：模型 ----------------
    with tab_model:
        _apply_pending_model_widget_updates()

        model_tab_notice = str(st.session_state.pop("model_tab_notice", "")).strip()
        if model_tab_notice:
            st.success(model_tab_notice)

        col_def1, col_def2 = st.columns([2, 1])
        with col_def1:
            species_text = st.text_input(
                "物种列表 (逗号分隔)",
                value=get_cfg("species_text", "A,B,C"),
                key="cfg_species_text",
                help="输入参与反应的所有物种名称，用逗号分隔。名称将用于 CSV 列名映射（如 C0_A_mol_m3）。",
            )
        with col_def2:
            n_reactions = int(
                st.number_input(
                    "反应数",
                    value=get_cfg("n_reactions", 1),
                    min_value=1,
                    key="cfg_n_reactions",
                    help="定义反应体系中的独立反应数量。每个反应对应计量数矩阵中的一列。",
                )
            )

        species_names = _clean_species_names(species_text)
        if not species_names:
            st.stop()

        # 化学计量数
        st.markdown(
            "**化学计量数矩阵 ν** (行=物种, 列=反应)",
            help="反应物为负数，产物为正数。例如 A → 2B 中，A 的计量数为 -1，B 为 +2。",
        )

        # --- 反应方程式快捷输入 ---
        with st.expander("从反应方程式生成（可选）", expanded=False):
            st.caption("每行一条反应，例如：`A → 2B` 或 `A + B -> C + D`。点击「生成」后自动填充下方矩阵。")
            eq_text = st.text_area(
                "反应方程式",
                height=80,
                placeholder="A → 2B\nA + C -> 3D",
                key="reaction_equations_input",
                label_visibility="collapsed",
            )
            if st.button("生成计量数矩阵", disabled=(not eq_text.strip())):
                parsed_matrix, parse_errors = _parse_reaction_equations(eq_text, species_names)
                if parse_errors:
                    for err in parse_errors:
                        st.error(err)
                elif parsed_matrix is not None:
                    n_parsed_reactions = parsed_matrix.shape[1]
                    st.session_state["pending_model_widget_updates"] = {
                        "config_updates": {
                            "stoich_matrix": parsed_matrix.tolist(),
                            "n_reactions": n_parsed_reactions,
                        },
                        "cfg_updates": {
                            "cfg_n_reactions": int(n_parsed_reactions),
                        },
                        "reset_prefixes": _build_stoich_widget_reset_prefixes(
                            len(species_names)
                        ),
                        "notice": f"已从 {n_parsed_reactions} 条方程式生成计量数矩阵。",
                    }
                    st.rerun()

        nu_default = _build_default_nu_table(species_names, n_reactions)
        # 若已导入配置，则优先应用其中的化学计量数
        imp_stoich = get_cfg("stoich_matrix", None)
        if imp_stoich is not None:
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
            width="stretch",
            key=f"nu_{len(species_names)}_{n_reactions}",
        )
        stoich_matrix = nu_table.to_numpy(dtype=float)
    param_state = resolve_fit_parameter_state(
        get_cfg,
        species_names,
        n_reactions,
        kinetic_model,
        reversible_enabled,
    )
    return {
        "reversible_enabled": bool(reversible_enabled),
        "species_text": species_text,
        "n_reactions": n_reactions,
        "species_names": species_names,
        "stoich_matrix": stoich_matrix,
        **param_state,
    }
