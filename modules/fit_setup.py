from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import modules.ui_components as ui_comp
from modules.constants import (
    DEFAULT_EA_K_MAX_J_MOL,
    DEFAULT_EA_K_MIN_J_MOL,
    DEFAULT_EA_MAX_J_MOL,
    DEFAULT_EA_MIN_J_MOL,
    DEFAULT_K0_ADS_MAX,
    DEFAULT_K0_ADS_MIN,
    DEFAULT_K0_MAX,
    DEFAULT_K0_MIN,
    DEFAULT_ORDER_MAX,
    DEFAULT_ORDER_MIN,
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
)


def _ensure_1d_length(
    values,
    target_length: int,
    fill_value: float | bool,
    *,
    dtype,
    extend_with_last: bool = True,
) -> np.ndarray:
    if target_length <= 0:
        return np.array([], dtype=dtype)

    if values is None:
        return np.full(target_length, fill_value, dtype=dtype)

    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        arr = np.array([arr.item()], dtype=dtype)
    else:
        arr = arr.reshape(-1).astype(dtype)

    out = np.full(target_length, fill_value, dtype=dtype)
    n_copy = min(target_length, int(arr.size))
    if n_copy > 0:
        out[:n_copy] = arr[:n_copy]
        if extend_with_last and arr.size < target_length:
            out[n_copy:] = arr[-1]
    return out


def _ensure_2d_shape(
    values,
    target_rows: int,
    target_cols: int,
    fill_value: float | bool,
    *,
    dtype,
) -> np.ndarray:
    if target_rows <= 0 or target_cols <= 0:
        return np.zeros((max(0, target_rows), max(0, target_cols)), dtype=dtype)

    if values is None:
        return np.full((target_rows, target_cols), fill_value, dtype=dtype)

    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr).astype(dtype)

    out = np.full((target_rows, target_cols), fill_value, dtype=dtype)
    n_rows = min(target_rows, int(arr.shape[0]))
    n_cols = min(target_cols, int(arr.shape[1]))
    out[:n_rows, :n_cols] = arr[:n_rows, :n_cols]
    return out


def _set_cfg_array(key: str, values) -> None:
    if isinstance(values, np.ndarray):
        st.session_state[key] = values.tolist()
    else:
        st.session_state[key] = values


def derive_effective_fit_flags(
    state: dict,
    kinetic_model: str,
    reversible_enabled: bool,
) -> dict[str, np.ndarray]:
    effective_flags = {
        "fit_k0_flags": np.asarray(state["fit_k0_flags"], dtype=bool).copy(),
        "fit_ea_flags": np.asarray(state["fit_ea_flags"], dtype=bool).copy(),
        "fit_order_flags_matrix": np.asarray(
            state["fit_order_flags_matrix"], dtype=bool
        ).copy(),
        "fit_K0_ads_flags": np.asarray(state["fit_K0_ads_flags"], dtype=bool).copy(),
        "fit_Ea_K_flags": np.asarray(state["fit_Ea_K_flags"], dtype=bool).copy(),
        "fit_m_flags": np.asarray(state["fit_m_flags"], dtype=bool).copy(),
        "fit_k0_rev_flags": np.asarray(
            state["fit_k0_rev_flags"], dtype=bool
        ).copy(),
        "fit_ea_rev_flags": np.asarray(
            state["fit_ea_rev_flags"], dtype=bool
        ).copy(),
        "fit_order_rev_flags_matrix": np.asarray(
            state["fit_order_rev_flags_matrix"], dtype=bool
        ).copy(),
    }

    if kinetic_model != KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
        effective_flags["fit_K0_ads_flags"] = np.zeros_like(
            effective_flags["fit_K0_ads_flags"], dtype=bool
        )
        effective_flags["fit_Ea_K_flags"] = np.zeros_like(
            effective_flags["fit_Ea_K_flags"], dtype=bool
        )
        effective_flags["fit_m_flags"] = np.zeros_like(
            effective_flags["fit_m_flags"], dtype=bool
        )

    if not reversible_enabled:
        effective_flags["fit_k0_rev_flags"] = np.zeros_like(
            effective_flags["fit_k0_rev_flags"], dtype=bool
        )
        effective_flags["fit_ea_rev_flags"] = np.zeros_like(
            effective_flags["fit_ea_rev_flags"], dtype=bool
        )
        effective_flags["fit_order_rev_flags_matrix"] = np.zeros_like(
            effective_flags["fit_order_rev_flags_matrix"], dtype=bool
        )

    return effective_flags


def resolve_fit_parameter_state(
    get_cfg,
    species_names: list[str],
    n_reactions: int,
    kinetic_model: str,
    reversible_enabled: bool,
) -> dict:
    n_species = len(species_names)

    k0_guess = _ensure_1d_length(
        get_cfg("k0_guess", None), n_reactions, 1e3, dtype=float
    )
    ea_guess_J_mol = _ensure_1d_length(
        get_cfg("ea_guess_J_mol", None), n_reactions, 8e4, dtype=float
    )
    fit_k0_flags = _ensure_1d_length(
        get_cfg("fit_k0_flags", None), n_reactions, True, dtype=bool
    )
    fit_ea_flags = _ensure_1d_length(
        get_cfg("fit_ea_flags", None), n_reactions, True, dtype=bool
    )

    order_guess_cfg = get_cfg("order_guess", None)
    order_guess = _ensure_2d_shape(
        order_guess_cfg, n_reactions, n_species, 0.0, dtype=float
    )
    if order_guess_cfg is None and n_species > 0:
        order_guess[:, 0] = 1.0
    fit_order_flags_matrix = _ensure_2d_shape(
        get_cfg("fit_order_flags_matrix", None),
        n_reactions,
        n_species,
        False,
        dtype=bool,
    )

    K0_ads = _ensure_1d_length(get_cfg("K0_ads", None), n_species, 1.0, dtype=float)
    Ea_K_J_mol = _ensure_1d_length(
        get_cfg("Ea_K_J_mol", None), n_species, -2e4, dtype=float
    )
    fit_K0_ads_flags = _ensure_1d_length(
        get_cfg("fit_K0_ads_flags", None),
        n_species,
        False,
        dtype=bool,
        extend_with_last=False,
    )
    fit_Ea_K_flags = _ensure_1d_length(
        get_cfg("fit_Ea_K_flags", None),
        n_species,
        False,
        dtype=bool,
        extend_with_last=False,
    )
    m_inhibition = _ensure_1d_length(
        get_cfg("m_inhibition", None), n_reactions, 1.0, dtype=float
    )
    fit_m_flags = _ensure_1d_length(
        get_cfg("fit_m_flags", None),
        n_reactions,
        False,
        dtype=bool,
        extend_with_last=False,
    )

    k0_rev = _ensure_1d_length(get_cfg("k0_rev", None), n_reactions, 1e2, dtype=float)
    ea_rev_J_mol = _ensure_1d_length(
        get_cfg("ea_rev_J_mol", None), n_reactions, 9e4, dtype=float
    )
    fit_k0_rev_flags = _ensure_1d_length(
        get_cfg("fit_k0_rev_flags", None),
        n_reactions,
        False,
        dtype=bool,
        extend_with_last=False,
    )
    fit_ea_rev_flags = _ensure_1d_length(
        get_cfg("fit_ea_rev_flags", None),
        n_reactions,
        False,
        dtype=bool,
        extend_with_last=False,
    )
    order_rev = _ensure_2d_shape(
        get_cfg("order_rev", None), n_reactions, n_species, 0.0, dtype=float
    )
    fit_order_rev_flags_matrix = _ensure_2d_shape(
        get_cfg("fit_order_rev_flags_matrix", None),
        n_reactions,
        n_species,
        False,
        dtype=bool,
    )

    return {
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


def render_fit_setup(ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    species_names = ctx["species_names"]
    n_reactions = int(ctx["n_reactions"])
    kinetic_model = str(ctx["kinetic_model"])
    reversible_enabled = bool(ctx.get("reversible_enabled", False))

    state = resolve_fit_parameter_state(
        get_cfg,
        list(species_names),
        n_reactions,
        kinetic_model,
        reversible_enabled,
    )

    with st.container(border=True):
        st.markdown("#### 初值与基础边界")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.caption("速率常数 k₀ [SI 单位，取决于反应级数] 与活化能 Eₐ [J/mol]")
            (
                state["k0_guess"],
                state["ea_guess_J_mol"],
                state["fit_k0_flags"],
                state["fit_ea_flags"],
            ) = ui_comp.render_param_table(
                f"base_params_{n_reactions}",
                [f"R{i+1}" for i in range(n_reactions)],
                "k₀",
                state["k0_guess"],
                "指前因子",
                "Eₐ [J/mol]",
                state["ea_guess_J_mol"],
                "活化能",
                state["fit_k0_flags"],
                state["fit_ea_flags"],
            )

        with col_p2:
            st.caption("反应级数 n [-]（无量纲）")
            (
                state["order_guess"],
                state["fit_order_flags_matrix"],
            ) = ui_comp.render_order_table(
                f"base_orders_{n_reactions}",
                [f"R{i+1}" for i in range(n_reactions)],
                list(species_names),
                state["order_guess"],
                state["fit_order_flags_matrix"],
            )

        st.markdown("**基础边界设置**")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            k0_min = ui_comp.smart_number_input(
                "k₀ 下限（k0_min）",
                value=float(get_cfg("k0_min", DEFAULT_K0_MIN)),
                key="cfg_k0_min",
            )
            k0_max = ui_comp.smart_number_input(
                "k₀ 上限（k0_max）",
                value=float(get_cfg("k0_max", DEFAULT_K0_MAX)),
                key="cfg_k0_max",
            )
        with col_b2:
            ea_min_J_mol = ui_comp.smart_number_input(
                "Eₐ 下限（ea_min_J_mol）",
                value=float(get_cfg("ea_min_J_mol", DEFAULT_EA_MIN_J_MOL)),
                key="cfg_ea_min_J_mol",
            )
            ea_max_J_mol = ui_comp.smart_number_input(
                "Eₐ 上限（ea_max_J_mol）",
                value=float(get_cfg("ea_max_J_mol", DEFAULT_EA_MAX_J_MOL)),
                key="cfg_ea_max_J_mol",
            )
        with col_b3:
            order_min = ui_comp.smart_number_input(
                "反应级数下限（order_min）",
                value=float(get_cfg("order_min", DEFAULT_ORDER_MIN)),
                key="cfg_order_min",
            )
            order_max = ui_comp.smart_number_input(
                "反应级数上限（order_max）",
                value=float(get_cfg("order_max", DEFAULT_ORDER_MAX)),
                key="cfg_order_max",
            )

        if kinetic_model == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
            with st.expander("Langmuir-Hinshelwood (L-H) 参数", expanded=True):
                col_lh1, col_lh2 = st.columns(2)
                with col_lh1:
                    st.caption("吸附常数 K₀,ads [1/(mol/m³)] 与吸附热 Eₐ,K [J/mol]")
                    (
                        state["K0_ads"],
                        state["Ea_K_J_mol"],
                        state["fit_K0_ads_flags"],
                        state["fit_Ea_K_flags"],
                    ) = ui_comp.render_param_table(
                        f"lh_ads_{len(species_names)}",
                        list(species_names),
                        "K₀,ads",
                        state["K0_ads"],
                        "吸附常数指前因子",
                        "Eₐ,K [J/mol]",
                        state["Ea_K_J_mol"],
                        "吸附热",
                        state["fit_K0_ads_flags"],
                        state["fit_Ea_K_flags"],
                    )
                with col_lh2:
                    st.caption("抑制指数 m [-]（无量纲，每反应）")
                    m_df = pd.DataFrame(
                        {"m": state["m_inhibition"], "Fit_m": state["fit_m_flags"]},
                        index=[f"R{i+1}" for i in range(n_reactions)],
                    )
                    m_editor = st.data_editor(
                        m_df,
                        key=f"lh_m_{n_reactions}",
                        column_config={
                            "Fit_m": st.column_config.CheckboxColumn(default=False)
                        },
                    )
                    state["m_inhibition"] = m_editor["m"].to_numpy(dtype=float)
                    state["fit_m_flags"] = m_editor["Fit_m"].to_numpy(dtype=bool)

                st.markdown("**L-H 边界设置**")
                col_lh_b1, col_lh_b2 = st.columns(2)
                with col_lh_b1:
                    K0_ads_min = ui_comp.smart_number_input(
                        "K₀,ads 下限（K0_ads_min）",
                        value=float(get_cfg("K0_ads_min", DEFAULT_K0_ADS_MIN)),
                        key="cfg_K0_ads_min",
                    )
                    K0_ads_max = ui_comp.smart_number_input(
                        "K₀,ads 上限（K0_ads_max）",
                        value=float(get_cfg("K0_ads_max", DEFAULT_K0_ADS_MAX)),
                        key="cfg_K0_ads_max",
                    )
                with col_lh_b2:
                    Ea_K_min = ui_comp.smart_number_input(
                        "Eₐ,K 下限（Ea_K_min）",
                        value=float(get_cfg("Ea_K_min", DEFAULT_EA_K_MIN_J_MOL)),
                        key="cfg_Ea_K_min",
                    )
                    Ea_K_max = ui_comp.smart_number_input(
                        "Eₐ,K 上限（Ea_K_max）",
                        value=float(get_cfg("Ea_K_max", DEFAULT_EA_K_MAX_J_MOL)),
                        key="cfg_Ea_K_max",
                    )
        else:
            K0_ads_min = float(get_cfg("K0_ads_min", DEFAULT_K0_ADS_MIN))
            K0_ads_max = float(get_cfg("K0_ads_max", DEFAULT_K0_ADS_MAX))
            Ea_K_min = float(get_cfg("Ea_K_min", DEFAULT_EA_K_MIN_J_MOL))
            Ea_K_max = float(get_cfg("Ea_K_max", DEFAULT_EA_K_MAX_J_MOL))

        if reversible_enabled:
            with st.expander("可逆反应 (逆反应) 参数", expanded=True):
                col_rev1, col_rev2 = st.columns(2)
                with col_rev1:
                    st.caption("逆反应 k₀⁻ [SI] 与 Eₐ⁻ [J/mol]")
                    (
                        state["k0_rev"],
                        state["ea_rev_J_mol"],
                        state["fit_k0_rev_flags"],
                        state["fit_ea_rev_flags"],
                    ) = ui_comp.render_param_table(
                        f"rev_params_{n_reactions}",
                        [f"R{i+1}" for i in range(n_reactions)],
                        "k₀⁻",
                        state["k0_rev"],
                        "逆反应指前因子",
                        "Eₐ⁻ [J/mol]",
                        state["ea_rev_J_mol"],
                        "逆反应活化能",
                        state["fit_k0_rev_flags"],
                        state["fit_ea_rev_flags"],
                    )
                with col_rev2:
                    st.caption("逆反应级数 n⁻ [-]（无量纲）")
                    (
                        state["order_rev"],
                        state["fit_order_rev_flags_matrix"],
                    ) = ui_comp.render_order_table(
                        f"rev_orders_{n_reactions}",
                        [f"R{i+1}" for i in range(n_reactions)],
                        list(species_names),
                        state["order_rev"],
                        state["fit_order_rev_flags_matrix"],
                    )

    _set_cfg_array("cfg_k0_guess", state["k0_guess"])
    _set_cfg_array("cfg_ea_guess_J_mol", state["ea_guess_J_mol"])
    _set_cfg_array("cfg_fit_k0_flags", state["fit_k0_flags"])
    _set_cfg_array("cfg_fit_ea_flags", state["fit_ea_flags"])
    _set_cfg_array("cfg_order_guess", state["order_guess"])
    _set_cfg_array("cfg_fit_order_flags_matrix", state["fit_order_flags_matrix"])
    _set_cfg_array("cfg_K0_ads", state["K0_ads"])
    _set_cfg_array("cfg_Ea_K_J_mol", state["Ea_K_J_mol"])
    _set_cfg_array("cfg_fit_K0_ads_flags", state["fit_K0_ads_flags"])
    _set_cfg_array("cfg_fit_Ea_K_flags", state["fit_Ea_K_flags"])
    _set_cfg_array("cfg_m_inhibition", state["m_inhibition"])
    _set_cfg_array("cfg_fit_m_flags", state["fit_m_flags"])
    _set_cfg_array("cfg_k0_rev", state["k0_rev"])
    _set_cfg_array("cfg_ea_rev_J_mol", state["ea_rev_J_mol"])
    _set_cfg_array("cfg_fit_k0_rev_flags", state["fit_k0_rev_flags"])
    _set_cfg_array("cfg_fit_ea_rev_flags", state["fit_ea_rev_flags"])
    _set_cfg_array("cfg_order_rev", state["order_rev"])
    _set_cfg_array(
        "cfg_fit_order_rev_flags_matrix", state["fit_order_rev_flags_matrix"]
    )

    return {
        **state,
        "k0_min": float(k0_min),
        "k0_max": float(k0_max),
        "ea_min_J_mol": float(ea_min_J_mol),
        "ea_max_J_mol": float(ea_max_J_mol),
        "order_min": float(order_min),
        "order_max": float(order_max),
        "K0_ads_min": float(K0_ads_min),
        "K0_ads_max": float(K0_ads_max),
        "Ea_K_min": float(Ea_K_min),
        "Ea_K_max": float(Ea_K_max),
    }
