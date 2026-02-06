from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import modules.ui_components as ui_comp
from modules.app_config_state import _warn_once
from modules.app_data_utils import _build_default_nu_table, _clean_species_names
from modules.constants import (
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODEL_REVERSIBLE,
)


def render_model_tab(tab_model, ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    kinetic_model = ctx["kinetic_model"]
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
            use_container_width=True,
            key=f"nu_{len(species_names)}_{n_reactions}",
        )
        stoich_matrix = nu_table.to_numpy(dtype=float)

        st.markdown("---")
        st.markdown("#### 动力学参数初值")

        # --- 基础参数（k₀, Eₐ, n）---
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.caption("速率常数 k₀ 与活化能 Eₐ")
            # 获取默认值的逻辑
            k0_guess_cfg = get_cfg("k0_guess", None)
            if k0_guess_cfg is None:
                k0_def = np.full(n_reactions, 1e3, dtype=float)
            else:
                k0_def = np.asarray(k0_guess_cfg, dtype=float)

            ea_guess_cfg = get_cfg("ea_guess_J_mol", None)
            if ea_guess_cfg is None:
                ea_def = np.full(n_reactions, 8e4, dtype=float)
            else:
                ea_def = np.asarray(ea_guess_cfg, dtype=float)

            fit_k0_cfg = get_cfg("fit_k0_flags", None)
            if fit_k0_cfg is None:
                fit_k0_def = np.full(n_reactions, True, dtype=bool)
            else:
                fit_k0_def = np.asarray(fit_k0_cfg, dtype=bool)

            fit_ea_cfg = get_cfg("fit_ea_flags", None)
            if fit_ea_cfg is None:
                fit_ea_def = np.full(n_reactions, True, dtype=bool)
            else:
                fit_ea_def = np.asarray(fit_ea_cfg, dtype=bool)

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
            order_guess_cfg = get_cfg("order_guess", None)
            order_data = (
                None if order_guess_cfg is None else np.asarray(order_guess_cfg, dtype=float)
            )
            if order_data is None:
                # 简单默认规则
                order_data = np.zeros((n_reactions, len(species_names)))
                if len(species_names) > 0:
                    order_data[:, 0] = 1.0

            fit_order_cfg = get_cfg("fit_order_flags_matrix", None)
            fit_order_def = (
                None if fit_order_cfg is None else np.asarray(fit_order_cfg, dtype=bool)
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

        if kinetic_model == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
            with st.expander("Langmuir-Hinshelwood (L-H) 参数", expanded=True):
                col_lh1, col_lh2 = st.columns(2)
                with col_lh1:
                    st.caption("吸附常数 K (每物种)")
                    # 默认值
                    K0_ads_cfg = get_cfg("K0_ads", None)
                    if K0_ads_cfg is None:
                        K0_def = np.ones(len(species_names), dtype=float)
                    else:
                        K0_def = np.asarray(K0_ads_cfg, dtype=float)

                    Ea_K_cfg = get_cfg("Ea_K_J_mol", None)
                    if Ea_K_cfg is None:
                        EaK_def = np.full(len(species_names), -2e4, dtype=float)
                    else:
                        EaK_def = np.asarray(Ea_K_cfg, dtype=float)

                    fit_K0_ads_cfg = get_cfg("fit_K0_ads_flags", None)
                    if fit_K0_ads_cfg is None:
                        fit_K0_def = np.full(len(species_names), False, dtype=bool)
                    else:
                        fit_K0_def = np.asarray(fit_K0_ads_cfg, dtype=bool)

                    fit_Ea_K_cfg = get_cfg("fit_Ea_K_flags", None)
                    if fit_Ea_K_cfg is None:
                        fit_EaK_def = np.full(len(species_names), False, dtype=bool)
                    else:
                        fit_EaK_def = np.asarray(fit_Ea_K_cfg, dtype=bool)

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
                    m_cfg = get_cfg("m_inhibition", None)
                    if m_cfg is None:
                        m_def = np.ones(n_reactions, dtype=float)
                    else:
                        m_def = np.asarray(m_cfg, dtype=float)

                    fit_m_cfg = get_cfg("fit_m_flags", None)
                    if fit_m_cfg is None:
                        fit_m_def = np.full(n_reactions, False, dtype=bool)
                    else:
                        fit_m_def = np.asarray(fit_m_cfg, dtype=bool)

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

        if kinetic_model == KINETIC_MODEL_REVERSIBLE:
            with st.expander("可逆反应 (逆反应) 参数", expanded=True):
                col_rev1, col_rev2 = st.columns(2)
                with col_rev1:
                    st.caption("逆反应 k₀⁻ 与 Eₐ⁻")
                    k0_rev_cfg = get_cfg("k0_rev", None)
                    if k0_rev_cfg is None:
                        k0r_def = np.full(n_reactions, 1e2, dtype=float)
                    else:
                        k0r_def = np.asarray(k0_rev_cfg, dtype=float)

                    ea_rev_cfg = get_cfg("ea_rev_J_mol", None)
                    if ea_rev_cfg is None:
                        ear_def = np.full(n_reactions, 9e4, dtype=float)
                    else:
                        ear_def = np.asarray(ea_rev_cfg, dtype=float)

                    fit_k0_rev_cfg = get_cfg("fit_k0_rev_flags", None)
                    if fit_k0_rev_cfg is None:
                        fit_k0r_def = np.full(n_reactions, False, dtype=bool)
                    else:
                        fit_k0r_def = np.asarray(fit_k0_rev_cfg, dtype=bool)

                    fit_ea_rev_cfg = get_cfg("fit_ea_rev_flags", None)
                    if fit_ea_rev_cfg is None:
                        fit_ear_def = np.full(n_reactions, False, dtype=bool)
                    else:
                        fit_ear_def = np.asarray(fit_ea_rev_cfg, dtype=bool)

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
                    order_rev_cfg = get_cfg("order_rev", None)
                    ordr_def = (
                        None
                        if order_rev_cfg is None
                        else np.asarray(order_rev_cfg, dtype=float)
                    )

                    fit_order_rev_cfg = get_cfg("fit_order_rev_flags_matrix", None)
                    fit_ordr_def = (
                        None
                        if fit_order_rev_cfg is None
                        else np.asarray(fit_order_rev_cfg, dtype=bool)
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
    return {
        "species_text": species_text,
        "n_reactions": n_reactions,
        "species_names": species_names,
        "stoich_matrix": stoich_matrix,
        "k0_guess": k0_guess,
        "ea_guess_J_mol": ea_guess_J_mol,
        "fit_k0_flags": fit_k0_flags,
        "fit_ea_flags": fit_ea_flags,
        "order_guess": order_guess,
        "fit_order_flags_matrix": fit_order_flags_matrix,
        "K0_ads": K0_ads,
        "Ea_K_J_mol": Ea_K_J_mol,
        "fit_K0_ads_flags": fit_K0_ads_flags,
        "fit_Ea_K_flags": fit_Ea_K_flags,
        "m_inhibition": m_inhibition,
        "fit_m_flags": fit_m_flags,
        "k0_rev": k0_rev,
        "ea_rev_J_mol": ea_rev_J_mol,
        "fit_k0_rev_flags": fit_k0_rev_flags,
        "fit_ea_rev_flags": fit_ea_rev_flags,
        "order_rev": order_rev,
        "fit_order_rev_flags_matrix": fit_order_rev_flags_matrix,
    }
