from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

import modules.reactors as reactors
import modules.ui_components as ui_comp
import modules.ui_text as ui_text
from modules.data_utils import (
    _build_fit_comparison_long_table,
    _freeze_params,
    _get_measurement_column_name,
    _get_output_unit_text,
)
from modules.fit_setup import derive_effective_fit_flags
from modules.fitting_background import _format_rmse
from modules.fit_state import (
    build_fit_result_state_snapshot,
    build_fit_state_snapshot,
    describe_fit_state_differences,
)
from modules.plot_helpers import (
    _fit_plot_color,
    _plot_reference_series,
    _style_fit_axis,
    _style_fit_legend,
)
from modules.constants import (
    DEFAULT_MAX_STEP_FRACTION,
    EPSILON_CONCENTRATION,
    EPSILON_FLOW_RATE,
    R_GAS_J_MOL_K,
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODEL_POWER_LAW,
    KINETIC_MODEL_REVERSIBLE,
    OUTPUT_MODE_COUT,
    OUTPUT_MODE_FOUT,
    OUTPUT_MODE_XOUT,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
    UI_COMPARE_TABLE_HEIGHT_PX,
    UI_METRICS_TABLE_HEIGHT_PX,
    UI_PARAM_TABLE_HEIGHT_PX,
    UI_PROFILE_POINTS_DEFAULT,
    UI_PROFILE_POINTS_MAX,
    UI_PROFILE_POINTS_MIN,
    UI_PROFILE_POINTS_STEP,
)


@st.cache_data(show_spinner="正在计算剖面…")
def _compute_reactor_profile(
    reactor_type: str,
    kinetic_model: str,
    reversible_enabled: bool,
    pfr_flow_model: str,
    output_mode: str,
    row_data_dict: dict,
    species_names: tuple,
    fitted_params_frozen: tuple,
    stoich_matrix_tuple: tuple,
    solver_method: str,
    rtol: float,
    atol: float,
    max_step_fraction: float,
    n_points: int,
) -> dict:
    """
    提取剖面计算为独立纯函数（可被 @st.cache_data 缓存）。

    返回 dict 包含 x_grid, profiles(ndarray), ok, message, x_label_key, profile_kind 等。
    """
    # 还原参数
    fitted_params = {}
    for k, v in fitted_params_frozen:
        if isinstance(v, (tuple, list)):
            fitted_params[k] = np.array(v)
        else:
            fitted_params[k] = v
    stoich_matrix = np.array(stoich_matrix_tuple)
    species_names_list = list(species_names)

    temperature_K = float(row_data_dict.get("T_K", float("nan")))

    if reactor_type == REACTOR_TYPE_PFR:
        reactor_volume_m3 = float(row_data_dict.get("V_m3", float("nan")))
        molar_flow_inlet = np.zeros(len(species_names_list), dtype=float)

        if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
            pressure_Pa = float(row_data_dict.get("P_Pa", float("nan")))
            for i, sp_name in enumerate(species_names_list):
                molar_flow_inlet[i] = float(
                    row_data_dict.get(f"F0_{sp_name}_mol_s", float("nan"))
                )

            x_grid, profiles, ok, message = (
                reactors.integrate_pfr_profile_gas_ideal_const_p(
                    reactor_volume_m3=reactor_volume_m3,
                    temperature_K=temperature_K,
                    pressure_Pa=pressure_Pa,
                    molar_flow_inlet_mol_s=molar_flow_inlet,
                    stoich_matrix=stoich_matrix,
                    k0=fitted_params["k0"],
                    ea_J_mol=fitted_params["ea_J_mol"],
                    reaction_order_matrix=fitted_params["reaction_order_matrix"],
                    solver_method=solver_method,
                    rtol=rtol,
                    atol=atol,
                    n_points=n_points,
                    kinetic_model=kinetic_model,
                    reversible_enabled=bool(reversible_enabled),
                    max_step_fraction=max_step_fraction,
                    K0_ads=fitted_params.get("K0_ads", None),
                    Ea_K_J_mol=fitted_params.get("Ea_K", None),
                    m_inhibition=fitted_params.get("m_inhibition", None),
                    k0_rev=fitted_params.get("k0_rev", None),
                    ea_rev_J_mol=fitted_params.get("ea_rev", None),
                    order_rev_matrix=fitted_params.get("order_rev", None),
                )
            )
        else:
            vdot_m3_s = float(row_data_dict.get("vdot_m3_s", float("nan")))
            use_conc_inlet = str(output_mode).strip().startswith("C")
            for i, sp_name in enumerate(species_names_list):
                if use_conc_inlet:
                    c0 = float(row_data_dict.get(f"C0_{sp_name}_mol_m3", float("nan")))
                    molar_flow_inlet[i] = c0 * float(vdot_m3_s)
                else:
                    molar_flow_inlet[i] = float(
                        row_data_dict.get(f"F0_{sp_name}_mol_s", float("nan"))
                    )

            x_grid, profiles, ok, message = reactors.integrate_pfr_profile(
                reactor_volume_m3=reactor_volume_m3,
                temperature_K=temperature_K,
                vdot_m3_s=vdot_m3_s,
                molar_flow_inlet_mol_s=molar_flow_inlet,
                stoich_matrix=stoich_matrix,
                k0=fitted_params["k0"],
                ea_J_mol=fitted_params["ea_J_mol"],
                reaction_order_matrix=fitted_params["reaction_order_matrix"],
                solver_method=solver_method,
                rtol=rtol,
                atol=atol,
                n_points=n_points,
                kinetic_model=kinetic_model,
                reversible_enabled=bool(reversible_enabled),
                max_step_fraction=max_step_fraction,
                K0_ads=fitted_params.get("K0_ads", None),
                Ea_K_J_mol=fitted_params.get("Ea_K", None),
                m_inhibition=fitted_params.get("m_inhibition", None),
                k0_rev=fitted_params.get("k0_rev", None),
                ea_rev_J_mol=fitted_params.get("ea_rev", None),
                order_rev_matrix=fitted_params.get("order_rev", None),
            )

        return {
            "x_grid": x_grid.tolist() if ok else [],
            "profiles": profiles.tolist() if ok else [],
            "ok": ok,
            "message": message,
            "x_label": "V_m3",
        }

    elif reactor_type == REACTOR_TYPE_CSTR:
        reactor_volume_m3 = float(row_data_dict.get("V_m3", float("nan")))
        vdot_m3_s = float(row_data_dict.get("vdot_m3_s", float("nan")))
        conc_inlet = np.zeros(len(species_names_list), dtype=float)
        for i, sp_name in enumerate(species_names_list):
            conc_inlet[i] = float(
                row_data_dict.get(f"C0_{sp_name}_mol_m3", float("nan"))
            )

        from modules.constants import EPSILON_FLOW_RATE

        tau_s = reactor_volume_m3 / max(vdot_m3_s, EPSILON_FLOW_RATE)
        simulation_time_s = float(5.0 * tau_s)

        x_grid, profiles, ok, message = reactors.integrate_cstr_profile(
            simulation_time_s=simulation_time_s,
            temperature_K=temperature_K,
            reactor_volume_m3=reactor_volume_m3,
            vdot_m3_s=vdot_m3_s,
            conc_inlet_mol_m3=conc_inlet,
            stoich_matrix=stoich_matrix,
            k0=fitted_params["k0"],
            ea_J_mol=fitted_params["ea_J_mol"],
            reaction_order_matrix=fitted_params["reaction_order_matrix"],
            solver_method=solver_method,
            rtol=rtol,
            atol=atol,
            n_points=n_points,
            kinetic_model=kinetic_model,
            reversible_enabled=bool(reversible_enabled),
            max_step_fraction=max_step_fraction,
            K0_ads=fitted_params.get("K0_ads", None),
            Ea_K_J_mol=fitted_params.get("Ea_K", None),
            m_inhibition=fitted_params.get("m_inhibition", None),
            k0_rev=fitted_params.get("k0_rev", None),
            ea_rev_J_mol=fitted_params.get("ea_rev", None),
            order_rev_matrix=fitted_params.get("order_rev", None),
        )
        return {
            "x_grid": x_grid.tolist() if ok else [],
            "profiles": profiles.tolist() if ok else [],
            "ok": ok,
            "message": message,
            "x_label": "t_s",
        }

    else:  # BSTR
        reaction_time_s = float(row_data_dict.get("t_s", float("nan")))
        conc_initial = np.zeros(len(species_names_list), dtype=float)
        for i, sp_name in enumerate(species_names_list):
            conc_initial[i] = float(
                row_data_dict.get(f"C0_{sp_name}_mol_m3", float("nan"))
            )

        x_grid, profiles, ok, message = reactors.integrate_batch_profile(
            reaction_time_s=reaction_time_s,
            temperature_K=temperature_K,
            conc_initial_mol_m3=conc_initial,
            stoich_matrix=stoich_matrix,
            k0=fitted_params["k0"],
            ea_J_mol=fitted_params["ea_J_mol"],
            reaction_order_matrix=fitted_params["reaction_order_matrix"],
            solver_method=solver_method,
            rtol=rtol,
            atol=atol,
            n_points=n_points,
            kinetic_model=kinetic_model,
            reversible_enabled=bool(reversible_enabled),
            max_step_fraction=max_step_fraction,
            K0_ads=fitted_params.get("K0_ads", None),
            Ea_K_J_mol=fitted_params.get("Ea_K", None),
            m_inhibition=fitted_params.get("m_inhibition", None),
            k0_rev=fitted_params.get("k0_rev", None),
            ea_rev_J_mol=fitted_params.get("ea_rev", None),
            order_rev_matrix=fitted_params.get("order_rev", None),
        )
        return {
            "x_grid": x_grid.tolist() if ok else [],
            "profiles": profiles.tolist() if ok else [],
            "ok": ok,
            "message": message,
            "x_label": "t_s",
        }


def _render_centered_pyplot(fig) -> None:
    left_col, center_col, right_col = st.columns([1, 2, 1])
    del left_col, right_col
    with center_col:
        st.pyplot(fig, width="content")


def _flatten_params_snapshot(
    params_snapshot: dict | None,
    species_names: list[str],
) -> pd.DataFrame:
    if not isinstance(params_snapshot, dict):
        return pd.DataFrame(columns=["parameter", "value"])

    rows: list[dict[str, float | str]] = []

    def add_vector(values, prefix: str, labels: list[str]) -> None:
        if values is None:
            return
        arr = np.asarray(values, dtype=float).reshape(-1)
        for idx, value in enumerate(arr):
            label = labels[idx] if idx < len(labels) else str(idx + 1)
            rows.append({"parameter": f"{prefix}[{label}]", "value": float(value)})

    def add_matrix(values, prefix: str, row_labels: list[str], col_labels: list[str]) -> None:
        if values is None:
            return
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)
        for row_idx in range(arr.shape[0]):
            row_label = row_labels[row_idx] if row_idx < len(row_labels) else str(row_idx + 1)
            for col_idx in range(arr.shape[1]):
                col_label = (
                    col_labels[col_idx] if col_idx < len(col_labels) else str(col_idx + 1)
                )
                rows.append(
                    {
                        "parameter": f"{prefix}[{row_label},{col_label}]",
                        "value": float(arr[row_idx, col_idx]),
                    }
                )

    reaction_count = int(np.asarray(params_snapshot.get("k0", [])).size)
    reaction_labels = [f"R{i+1}" for i in range(reaction_count)]

    add_vector(params_snapshot.get("k0", None), "k0", reaction_labels)
    add_vector(params_snapshot.get("ea_J_mol", None), "Ea", reaction_labels)
    add_matrix(
        params_snapshot.get("reaction_order_matrix", None),
        "n",
        reaction_labels,
        list(species_names),
    )
    add_vector(params_snapshot.get("K0_ads", None), "K0_ads", list(species_names))
    add_vector(params_snapshot.get("Ea_K", None), "Ea_K", list(species_names))
    add_vector(params_snapshot.get("m_inhibition", None), "m", reaction_labels)
    add_vector(params_snapshot.get("k0_rev", None), "k0_rev", reaction_labels)
    add_vector(params_snapshot.get("ea_rev", None), "Ea_rev", reaction_labels)
    add_matrix(
        params_snapshot.get("order_rev", None),
        "n_rev",
        reaction_labels,
        list(species_names),
    )

    if not rows:
        return pd.DataFrame(columns=["parameter", "value"])
    return pd.DataFrame(rows)


def _format_history_option(history_entry: dict) -> str:
    fit_id = int(history_entry.get("fit_id", 0))
    time_text = str(history_entry.get("time", "")).strip()
    phi_value = ui_comp.smart_float_to_str(float(history_entry.get("phi", np.nan)))
    return f"#{fit_id} | {time_text} | Φ={phi_value}"


def _map_parity_temperatures_by_row_index(
    data_df: pd.DataFrame,
    row_indices,
) -> pd.Series:
    if "T_K" not in data_df.columns:
        return pd.Series(np.nan, index=pd.Index(row_indices))
    t_k_by_label = pd.to_numeric(data_df["T_K"], errors="coerce")
    row_index_labels = pd.Index(pd.Series(row_indices).tolist())
    return t_k_by_label.reindex(row_index_labels)


def _apply_equal_parity_ticks(
    ax: plt.Axes,
    axis_min: float,
    axis_max: float,
    nbins: int = 5,
) -> None:
    """
    让奇偶校验图的 x/y 主刻度和次刻度完全一致。

    说明：
    - 奇偶校验图用于判断 y = x 附近的偏离，因此不仅 xlim/ylim 要一致，
      刻度位置也应一致，否则视觉上仍可能产生误判。
    - 先用 MaxNLocator 生成一组“好看”的主刻度，再同时写入 x/y。
    - AutoMinorLocator(2) 表示每两个主刻度之间放 1 个次刻度；由于主刻度
      已经相同，x/y 次刻度也会一致。
    """
    axis_min = float(axis_min)
    axis_max = float(axis_max)
    if (
        (not np.isfinite(axis_min))
        or (not np.isfinite(axis_max))
        or axis_max <= axis_min
    ):
        axis_min, axis_max = 0.0, 1.0

    locator = mticker.MaxNLocator(nbins=int(max(2, nbins)))
    tick_values = np.asarray(locator.tick_values(axis_min, axis_max), dtype=float)
    tick_values = tick_values[np.isfinite(tick_values)]
    tick_values = tick_values[
        (tick_values >= axis_min - 1e-12 * max(1.0, abs(axis_min)))
        & (tick_values <= axis_max + 1e-12 * max(1.0, abs(axis_max)))
    ]
    if tick_values.size < 2:
        tick_values = np.linspace(axis_min, axis_max, int(max(2, nbins)), dtype=float)

    ax.xaxis.set_major_locator(mticker.FixedLocator(tick_values))
    ax.yaxis.set_major_locator(mticker.FixedLocator(tick_values))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)


def _remap_species_vector(
    values,
    species_names_fit: list[str],
    species_names_current: list[str],
) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != len(species_names_fit):
        raise ValueError("历史结果中的物种参数长度与物种列表不一致。")
    fit_name_to_index = {name: i for i, name in enumerate(species_names_fit)}
    return [float(arr[fit_name_to_index[name]]) for name in species_names_current]


def _remap_species_matrix(
    values,
    species_names_fit: list[str],
    species_names_current: list[str],
    n_reactions_current: int,
) -> list[list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_reactions_current, len(species_names_fit)):
        raise ValueError("历史结果中的级数矩阵形状与当前模型不一致。")
    fit_name_to_index = {name: i for i, name in enumerate(species_names_fit)}
    remapped = arr[:, [fit_name_to_index[name] for name in species_names_current]]
    return remapped.tolist()


def _build_initial_guess_updates_from_fit_result(
    fitted_params: dict | None,
    *,
    species_names_current: list[str],
    species_names_fit: list[str],
    stoich_matrix_current,
    stoich_matrix_fit,
    n_reactions_current: int,
    kinetic_model_current: str,
    kinetic_model_fit: str,
    reversible_enabled_current: bool,
    reversible_enabled_fit: bool,
) -> tuple[dict | None, str]:
    if not isinstance(fitted_params, dict):
        return None, "当前拟合结果缺少参数快照。"

    if str(kinetic_model_current) != str(kinetic_model_fit):
        return None, "当前动力学模型与结果来源不一致。"
    if bool(reversible_enabled_current) != bool(reversible_enabled_fit):
        return None, "当前可逆反应设置与结果来源不一致。"

    fit_species_names = [str(x) for x in species_names_fit]
    current_species_names = [str(x) for x in species_names_current]
    if len(fit_species_names) != len(current_species_names) or set(
        fit_species_names
    ) != set(current_species_names):
        return None, "当前物种列表与结果来源不一致。"

    stoich_current = np.asarray(stoich_matrix_current, dtype=float)
    stoich_fit = np.asarray(stoich_matrix_fit, dtype=float)
    if stoich_current.shape != stoich_fit.shape:
        return None, "当前化学计量数矩阵与结果来源不一致。"

    fit_name_to_index = {name: i for i, name in enumerate(fit_species_names)}
    stoich_fit_aligned = stoich_fit[
        [fit_name_to_index[name] for name in current_species_names], :
    ]
    if not np.allclose(stoich_current, stoich_fit_aligned, rtol=0.0, atol=1e-12):
        return None, "当前化学计量数矩阵与结果来源不一致。"

    k0_values = np.asarray(fitted_params.get("k0", []), dtype=float).reshape(-1)
    ea_values = np.asarray(fitted_params.get("ea_J_mol", []), dtype=float).reshape(-1)
    order_values = np.asarray(
        fitted_params.get("reaction_order_matrix", []), dtype=float
    )
    if k0_values.size != n_reactions_current or ea_values.size != n_reactions_current:
        return None, "当前反应数与结果来源不一致。"
    if order_values.shape != (n_reactions_current, len(fit_species_names)):
        return None, "历史结果中的反应级数矩阵与当前模型不一致。"

    try:
        config_updates = {
            "k0_guess": k0_values.tolist(),
            "ea_guess_J_mol": ea_values.tolist(),
            "order_guess": _remap_species_matrix(
                order_values,
                fit_species_names,
                current_species_names,
                n_reactions_current,
            ),
        }

        if fitted_params.get("K0_ads") is not None:
            config_updates["K0_ads"] = _remap_species_vector(
                fitted_params["K0_ads"],
                fit_species_names,
                current_species_names,
            )
        if fitted_params.get("Ea_K") is not None:
            config_updates["Ea_K_J_mol"] = _remap_species_vector(
                fitted_params["Ea_K"],
                fit_species_names,
                current_species_names,
            )
        if fitted_params.get("m_inhibition") is not None:
            m_values = np.asarray(fitted_params["m_inhibition"], dtype=float).reshape(-1)
            if m_values.size != n_reactions_current:
                return None, "历史结果中的抑制指数与当前反应数不一致。"
            config_updates["m_inhibition"] = m_values.tolist()
        if fitted_params.get("k0_rev") is not None:
            k0_rev_values = np.asarray(fitted_params["k0_rev"], dtype=float).reshape(-1)
            if k0_rev_values.size != n_reactions_current:
                return None, "历史结果中的逆反应 k0 与当前反应数不一致。"
            config_updates["k0_rev"] = k0_rev_values.tolist()
        if fitted_params.get("ea_rev") is not None:
            ea_rev_values = np.asarray(fitted_params["ea_rev"], dtype=float).reshape(-1)
            if ea_rev_values.size != n_reactions_current:
                return None, "历史结果中的逆反应 Ea 与当前反应数不一致。"
            config_updates["ea_rev_J_mol"] = ea_rev_values.tolist()
        if fitted_params.get("order_rev") is not None:
            config_updates["order_rev"] = _remap_species_matrix(
                fitted_params["order_rev"],
                fit_species_names,
                current_species_names,
                n_reactions_current,
            )
    except ValueError as exc:
        return None, str(exc)

    return config_updates, ""


def render_fit_results(
    tab_fit_results_container, ctx: dict, fit_advanced_state: dict, runtime_state: dict
) -> dict:
    get_cfg = ctx["get_cfg"]
    species_names = ctx["species_names"]
    stoich_matrix = ctx["stoich_matrix"]
    solver_method = ctx["solver_method"]
    rtol = ctx["rtol"]
    atol = ctx["atol"]
    reactor_type = ctx["reactor_type"]
    kinetic_model = ctx["kinetic_model"]
    reversible_enabled = bool(ctx.get("reversible_enabled", False))
    output_mode = ctx["output_mode"]
    n_reactions = int(ctx["n_reactions"])
    residual_type_current = str(fit_advanced_state.get("residual_type", "绝对残差"))
    effective_fit_flags = derive_effective_fit_flags(
        ctx,
        str(kinetic_model),
        bool(reversible_enabled),
    )
    max_step_fraction_current = float(
        fit_advanced_state.get(
            "max_step_fraction",
            get_cfg("max_step_fraction", DEFAULT_MAX_STEP_FRACTION),
        )
    )
    use_log_k0_fit_current = bool(
        fit_advanced_state.get("use_log_k0_fit", get_cfg("use_log_k0_fit", True))
    )
    use_log_k0_rev_fit_current = bool(
        fit_advanced_state.get(
            "use_log_k0_rev_fit", get_cfg("use_log_k0_rev_fit", True)
        )
    )
    use_log_K0_ads_fit_current = bool(
        fit_advanced_state.get(
            "use_log_K0_ads_fit", get_cfg("use_log_K0_ads_fit", True)
        )
    )
    fitting_running = bool(runtime_state.get("fitting_running", False))

    # 拟合进行中时不重绘上一轮完整结果，避免旧结果区的重型组件阻塞进度条刷新。
    if fitting_running:
        if "fit_results" in st.session_state:
            tab_fit_results_container.info(
                "当前正在重新拟合。为保证进度条刷新流畅，已暂时隐藏上一轮结果；拟合完成后会自动展示新结果。"
            )
        return {}

    # --- 结果展示（优化版）---
    if "fit_results" in st.session_state:
        res = st.session_state["fit_results"]
        tab_fit_results_container.divider()
        phi_value = float(res.get("phi_final", res.get("cost", 0.0)))
        phi_text = ui_comp.smart_float_to_str(phi_value)
        residual_type_used = str(res.get("residual_type", residual_type_current))
        residual_latex_map = {
            "绝对残差": r"r_i=y_i^{\mathrm{pred}}-y_i^{\mathrm{meas}}",
            "相对残差": (
                r"r_i=\dfrac{y_i^{\mathrm{pred}}-y_i^{\mathrm{meas}}}"
                r"{\mathrm{sign}(y_i^{\mathrm{meas}})\cdot\max(|y_i^{\mathrm{meas}}|,\varepsilon)}"
            ),
            "百分比残差": (
                r"r_i=100\times\dfrac{y_i^{\mathrm{pred}}-y_i^{\mathrm{meas}}}"
                r"{|y_i^{\mathrm{meas}}|+\varepsilon}"
            ),
        }
        residual_latex = residual_latex_map.get(
            residual_type_used, residual_latex_map["绝对残差"]
        )
        n_valid_points = int(res.get("n_valid_points", 0))
        rmse_text = _format_rmse(phi_value, n_valid_points, residual_type_used)
        header_line = f"### 拟合结果 · 目标函数 Φ={phi_text}"
        if rmse_text:
            header_line += f" · {rmse_text}"
        tab_fit_results_container.markdown(header_line)
        tab_fit_results_container.caption(f"残差类型：{residual_type_used}")
        tab_fit_results_container.latex(
            r"\Phi(\theta)=\frac{1}{2}\sum_{i=1}^{N} r_i(\theta)^2,\quad "
            + residual_latex
        )

        fitted_params = res["params"]
        # 优先使用拟合时的数据快照，避免上传/删除新 CSV 后结果页基于错误数据重算
        if "data" in res and isinstance(res["data"], pd.DataFrame):
            df_fit = res["data"]
        else:
            df_fit = st.session_state.get("data_df_cached", pd.DataFrame())

        active_data_df = st.session_state.get("data_df_cached", None)
        if active_data_df is None and isinstance(df_fit, pd.DataFrame):
            active_data_df = df_fit

        stale_reasons = describe_fit_state_differences(
            build_fit_state_snapshot(
                data_df=active_data_df,
                species_names=species_names,
                output_mode=str(output_mode),
                output_species_list=list(ctx.get("output_species_list", [])),
                stoich_matrix=stoich_matrix,
                solver_method=str(solver_method),
                rtol=float(rtol),
                atol=float(atol),
                reactor_type=str(reactor_type),
                kinetic_model=str(kinetic_model),
                reversible_enabled=bool(reversible_enabled),
                pfr_flow_model=str(ctx.get("pfr_flow_model", "")),
                max_step_fraction=float(max_step_fraction_current),
                residual_type=str(residual_type_current),
                use_log_k0_fit=bool(use_log_k0_fit_current),
                use_log_k0_rev_fit=bool(use_log_k0_rev_fit_current),
                use_log_K0_ads_fit=bool(use_log_K0_ads_fit_current),
                fit_k0_flags=ctx["fit_k0_flags"],
                fit_ea_flags=ctx["fit_ea_flags"],
                fit_order_flags_matrix=ctx["fit_order_flags_matrix"],
                fit_K0_ads_flags=effective_fit_flags["fit_K0_ads_flags"],
                fit_Ea_K_flags=effective_fit_flags["fit_Ea_K_flags"],
                fit_m_flags=effective_fit_flags["fit_m_flags"],
                fit_k0_rev_flags=effective_fit_flags["fit_k0_rev_flags"],
                fit_ea_rev_flags=effective_fit_flags["fit_ea_rev_flags"],
                fit_order_rev_flags_matrix=effective_fit_flags["fit_order_rev_flags_matrix"],
                k0_min=float(fit_advanced_state["k0_min"]),
                k0_max=float(fit_advanced_state["k0_max"]),
                ea_min=float(fit_advanced_state["ea_min"]),
                ea_max=float(fit_advanced_state["ea_max"]),
                ord_min=float(fit_advanced_state["ord_min"]),
                ord_max=float(fit_advanced_state["ord_max"]),
                K0_ads_min=float(fit_advanced_state["K0_ads_min"]),
                K0_ads_max=float(fit_advanced_state["K0_ads_max"]),
                Ea_K_min=float(fit_advanced_state["Ea_K_min"]),
                Ea_K_max=float(fit_advanced_state["Ea_K_max"]),
                k0_rev_min=float(fit_advanced_state["k0_rev_min"]),
                k0_rev_max=float(fit_advanced_state["k0_rev_max"]),
                ea_rev_min_J_mol=float(fit_advanced_state["ea_rev_min_J_mol"]),
                ea_rev_max_J_mol=float(fit_advanced_state["ea_rev_max_J_mol"]),
                order_rev_min=float(fit_advanced_state["order_rev_min"]),
                order_rev_max=float(fit_advanced_state["order_rev_max"]),
            ),
            build_fit_result_state_snapshot(res),
        )
        if stale_reasons:
            tab_fit_results_container.warning(
                "当前显示的是历史拟合结果，和最新输入不一致。"
            )
            for reason in stale_reasons:
                tab_fit_results_container.markdown(f"- {reason}")
        species_names_fit = res.get("species_names", species_names)
        stoich_matrix_fit = res.get("stoich_matrix", stoich_matrix)
        solver_method_fit = res.get("solver_method", solver_method)
        rtol_fit = float(res.get("rtol", rtol))
        atol_fit = float(res.get("atol", atol))
        max_step_fraction_fit = float(
            res.get(
                "max_step_fraction",
                get_cfg("max_step_fraction", DEFAULT_MAX_STEP_FRACTION),
            )
        )
        reactor_type_fit = res.get("reactor_type", reactor_type)
        kinetic_model_fit = res.get("kinetic_model", kinetic_model)
        reversible_enabled_fit = bool(
            res.get("reversible_enabled", reversible_enabled)
        )
        if str(kinetic_model_fit).strip() == KINETIC_MODEL_REVERSIBLE:
            kinetic_model_fit = KINETIC_MODEL_POWER_LAW
            reversible_enabled_fit = True
        output_mode_fit = res.get("output_mode", output_mode)
        # 奇偶校验图的候选物种会在 tab_parity 中根据“验证量（浓度/转化率）”动态判定
        parity_species_candidates = list(species_names_fit)
        parity_species_unavailable = []

        result_tab_labels = ["参数", "奇偶校验图", "沿程/随时间剖面", "导出"]
        fit_result_tab_default = result_tab_labels[0]

        fit_result_tab_idx = st.query_params.get("fit_result_tab_i", None)
        if isinstance(fit_result_tab_idx, (list, tuple)):
            fit_result_tab_idx = fit_result_tab_idx[0] if fit_result_tab_idx else None
        if fit_result_tab_idx is not None:
            try:
                fit_result_tab_idx_int = int(str(fit_result_tab_idx).strip())
            except Exception:
                fit_result_tab_idx_int = -1
            if 0 <= fit_result_tab_idx_int < len(result_tab_labels):
                fit_result_tab_default = result_tab_labels[fit_result_tab_idx_int]
        else:
            fit_result_tab_legacy = st.query_params.get("fit_result_tab", None)
            if isinstance(fit_result_tab_legacy, (list, tuple)):
                fit_result_tab_legacy = (
                    fit_result_tab_legacy[0] if fit_result_tab_legacy else None
                )
            fit_result_tab_legacy = (
                str(fit_result_tab_legacy).strip()
                if fit_result_tab_legacy is not None
                else ""
            )
            if fit_result_tab_legacy in result_tab_labels:
                fit_result_tab_default = fit_result_tab_legacy

        if (
            fit_result_tab_default == result_tab_labels[0]
            and "parity_validation_choice" in st.session_state
        ):
            fit_result_tab_default = "奇偶校验图"
        tab_param, tab_parity, tab_profile, tab_export = tab_fit_results_container.tabs(
            result_tab_labels,
            default=fit_result_tab_default,
        )

        with tab_param:
            st.markdown("#### 拟合参数")
            col_p1, col_p2 = st.columns([1, 1])

            with col_p1:
                reaction_names = [f"R{i+1}" for i in range(len(fitted_params["k0"]))]
                df_k0_ea = pd.DataFrame(
                    {
                        "k₀ [SI]": fitted_params["k0"],
                        "Eₐ [J/mol]": fitted_params["ea_J_mol"],
                    },
                    index=reaction_names,
                )
                st.markdown("**k₀ 与 Eₐ**")
                st.dataframe(
                    ui_comp.format_dataframe_for_display(df_k0_ea),
                    width="stretch",
                    height=UI_PARAM_TABLE_HEIGHT_PX,
                )

            with col_p2:
                st.markdown("**反应级数矩阵 $n$**")
                df_orders = pd.DataFrame(
                    fitted_params["reaction_order_matrix"],
                    index=reaction_names,
                    columns=species_names_fit,
                )
                st.dataframe(
                    ui_comp.format_dataframe_for_display(df_orders),
                    width="stretch",
                    height=UI_PARAM_TABLE_HEIGHT_PX,
                )

            if kinetic_model_fit == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
                st.markdown("#### Langmuir-Hinshelwood 参数")
                col_lh1, col_lh2 = st.columns([1, 1])
                with col_lh1:
                    if (
                        fitted_params.get("K0_ads", None) is not None
                        and fitted_params.get("Ea_K", None) is not None
                    ):
                        df_ads = pd.DataFrame(
                            {
                                "K₀,ads [1/(mol/m^3)]": fitted_params["K0_ads"],
                                "Eₐ,K [J/mol]": fitted_params["Ea_K"],
                            },
                            index=species_names_fit,
                        )
                        st.dataframe(
                            ui_comp.format_dataframe_for_display(df_ads),
                            width="stretch",
                            height=UI_PARAM_TABLE_HEIGHT_PX,
                        )
                with col_lh2:
                    if fitted_params.get("m_inhibition", None) is not None:
                        df_m = pd.DataFrame(
                            {"m_inhibition [-]": fitted_params["m_inhibition"]},
                            index=reaction_names,
                        )
                        st.dataframe(
                            ui_comp.format_dataframe_for_display(df_m),
                            width="stretch",
                            height=UI_PARAM_TABLE_HEIGHT_PX,
                        )

            if bool(reversible_enabled_fit):
                st.markdown("#### 可逆反应参数（逆反应）")
                if (
                    fitted_params.get("k0_rev", None) is not None
                    and fitted_params.get("ea_rev", None) is not None
                ):
                    df_rev = pd.DataFrame(
                        {
                            "k₀,rev [SI]": fitted_params["k0_rev"],
                            "Eₐ,rev [J/mol]": fitted_params["ea_rev"],
                        },
                        index=reaction_names,
                    )
                    st.dataframe(
                        ui_comp.format_dataframe_for_display(df_rev),
                        width="stretch",
                        height=UI_PARAM_TABLE_HEIGHT_PX,
                    )
                if fitted_params.get("order_rev", None) is not None:
                    st.markdown("**逆反应级数矩阵 $n^-$**")
                    df_order_rev = pd.DataFrame(
                        fitted_params["order_rev"],
                        index=reaction_names,
                        columns=species_names_fit,
                    )
                    st.dataframe(
                        ui_comp.format_dataframe_for_display(df_order_rev),
                        width="stretch",
                        height=UI_PARAM_TABLE_HEIGHT_PX,
                    )

            # --- 「将拟合结果作为新初值」按钮 ---
            st.divider()
            reuse_updates, reuse_disabled_reason = _build_initial_guess_updates_from_fit_result(
                fitted_params,
                species_names_current=list(species_names),
                species_names_fit=list(species_names_fit),
                stoich_matrix_current=stoich_matrix,
                stoich_matrix_fit=stoich_matrix_fit,
                n_reactions_current=n_reactions,
                kinetic_model_current=str(kinetic_model),
                kinetic_model_fit=str(kinetic_model_fit),
                reversible_enabled_current=bool(reversible_enabled),
                reversible_enabled_fit=bool(reversible_enabled_fit),
            )
            reuse_button_help = (
                "将上方拟合得到的参数填回当前拟合页的初值面板，作为下次拟合的初始猜测值。"
            )
            if reuse_disabled_reason:
                reuse_button_help += f" 当前不可用：{reuse_disabled_reason}"
            if st.button(
                "📥 将拟合结果作为新初值",
                help=reuse_button_help,
                use_container_width=True,
                disabled=(reuse_updates is None),
            ):
                st.session_state["pending_model_widget_updates"] = {
                    "config_updates": reuse_updates,
                    "reset_prefixes": [
                        "base_params_",
                        "base_orders_",
                        "lh_ads_",
                        "lh_m_",
                        "rev_params_",
                        "rev_orders_",
                    ],
                    "notice": "",
                }
                st.session_state["fit_notice"] = {
                    "kind": "success",
                    "text": "已将拟合结果写入当前拟合页的初值面板。",
                }
                st.rerun()
            elif reuse_disabled_reason:
                st.info(f"当前结果不能直接回填为新初值：{reuse_disabled_reason}")

            fitting_history = st.session_state.get("fitting_history", [])
            valid_history = [
                entry for entry in fitting_history if isinstance(entry, dict)
            ]
            if len(valid_history) > 1:
                st.divider()
                st.markdown("#### 与历史拟合比较")

                option_labels = [_format_history_option(entry) for entry in valid_history]
                label_to_entry = {
                    _format_history_option(entry): entry for entry in valid_history
                }
                default_base_idx = max(len(option_labels) - 2, 0)
                default_compare_idx = len(option_labels) - 1
                col_cmp1, col_cmp2 = st.columns(2)
                base_label = col_cmp1.selectbox(
                    "基准拟合",
                    option_labels,
                    index=default_base_idx,
                    key="fit_history_compare_base",
                )
                compare_label = col_cmp2.selectbox(
                    "对比拟合",
                    option_labels,
                    index=default_compare_idx,
                    key="fit_history_compare_target",
                )
                base_entry = label_to_entry.get(base_label, {})
                compare_entry = label_to_entry.get(compare_label, {})

                if base_label == compare_label:
                    st.info("请选择两次不同的拟合记录进行比较。")
                else:
                    state_diff = describe_fit_state_differences(
                        compare_entry.get("state_snapshot", {}),
                        base_entry.get("state_snapshot", {}),
                    )
                    if state_diff:
                        st.warning("两次拟合的输入设置不完全一致，参数变化需要结合设置变化一起判断。")
                        for reason in state_diff:
                            st.markdown(f"- {reason}")

                    phi_base = float(base_entry.get("phi", np.nan))
                    phi_compare = float(compare_entry.get("phi", np.nan))
                    if np.isfinite(phi_base) and abs(phi_base) > 1e-15:
                        phi_delta_pct = (phi_compare - phi_base) / phi_base * 100.0
                    else:
                        phi_delta_pct = np.nan
                    df_compare_summary = pd.DataFrame(
                        [
                            {
                                "指标": "Φ (cost)",
                                "基准": phi_base,
                                "对比": phi_compare,
                                "变化(%)": phi_delta_pct,
                            },
                            {
                                "指标": "拟合参数数",
                                "基准": float(base_entry.get("n_params", np.nan)),
                                "对比": float(compare_entry.get("n_params", np.nan)),
                                "变化(%)": np.nan,
                            },
                        ]
                    )
                    st.dataframe(
                        ui_comp.format_dataframe_for_display(df_compare_summary),
                        width="stretch",
                        hide_index=True,
                    )

                    base_species_names = [
                        str(x) for x in base_entry.get("species_names", [])
                    ]
                    compare_species_names = [
                        str(x) for x in compare_entry.get("species_names", [])
                    ]
                    base_params_df = _flatten_params_snapshot(
                        base_entry.get("params_snapshot", {}),
                        base_species_names,
                    )
                    compare_params_df = _flatten_params_snapshot(
                        compare_entry.get("params_snapshot", {}),
                        compare_species_names,
                    )

                    if base_params_df.empty or compare_params_df.empty:
                        st.info("历史记录缺少参数快照，当前无法比较参数变化。")
                    else:
                        merged_params = pd.merge(
                            base_params_df,
                            compare_params_df,
                            on="parameter",
                            how="outer",
                            suffixes=("_base", "_compare"),
                        )
                        merged_params["delta_abs"] = (
                            merged_params["value_compare"] - merged_params["value_base"]
                        )
                        merged_params["delta_pct"] = np.where(
                            np.isfinite(merged_params["value_base"])
                            & (np.abs(merged_params["value_base"]) > 1e-15),
                            merged_params["delta_abs"]
                            / merged_params["value_base"]
                            * 100.0,
                            np.nan,
                        )
                        show_changed_only = st.checkbox(
                            "仅显示变化参数",
                            value=True,
                            key="fit_history_compare_changed_only",
                        )
                        if show_changed_only:
                            same_mask = (
                                np.isfinite(merged_params["value_base"])
                                & np.isfinite(merged_params["value_compare"])
                                & np.isclose(
                                    merged_params["value_base"],
                                    merged_params["value_compare"],
                                    rtol=1e-10,
                                    atol=1e-12,
                                )
                            )
                            merged_params = merged_params[~same_mask].copy()
                        if merged_params.empty:
                            st.success("两次拟合的参数快照没有数值变化。")
                        else:
                            merged_params = merged_params.rename(
                                columns={
                                    "parameter": "参数",
                                    "value_base": "基准值",
                                    "value_compare": "对比值",
                                    "delta_abs": "绝对变化",
                                    "delta_pct": "相对变化(%)",
                                }
                            )
                            st.dataframe(
                                ui_comp.format_dataframe_for_display(merged_params),
                                width="stretch",
                                hide_index=True,
                                height=min(
                                    max(280, 36 * (len(merged_params) + 1)),
                                    560,
                                ),
                            )

        @st.fragment
        def _parity_tab_fragment():
            st.markdown("#### 分物种奇偶校验图（实验值 vs 预测值）")
            output_mode_fit_str = str(output_mode_fit).strip()
            output_label_map = {
                OUTPUT_MODE_COUT: "出口浓度 (Cout)",
                OUTPUT_MODE_FOUT: "出口摩尔流率 (Fout)",
                OUTPUT_MODE_XOUT: "出口摩尔分率 (xout)",
            }
            default_output_label = output_label_map.get(
                output_mode_fit_str, f"输出量（{output_mode_fit_str}）"
            )

            parity_validation_choice = st.radio(
                "验证量",
                [default_output_label, "转化率 (X)"],
                index=0,
                horizontal=True,
                key="parity_validation_choice",
                help="选择奇偶校验图的对比量：当前输出量（与拟合输出模式一致）或转化率。转化率优先按摩尔流率计算（BSTR 无摩尔流率则用浓度）。",
            )

            pfr_flow_model_fit = str(
                res.get("pfr_flow_model", PFR_FLOW_MODEL_LIQUID_CONST_VDOT)
            ).strip()

            # 根据验证量动态确定：对比方式、单位、以及可用物种
            if parity_validation_choice == default_output_label:
                compare_output_mode = output_mode_fit_str
                compare_validation_mode = "output"
                unit_text_parity = _get_output_unit_text(compare_output_mode)
            else:
                compare_output_mode = (
                    output_mode_fit_str  # 占位：conversion 模式下不会使用测量列名映射
                )
                compare_validation_mode = "conversion"
                unit_text_parity = "-"

            parity_species_candidates = []
            parity_species_unavailable = []
            df_cols = set(map(str, df_fit.columns))

            for sp_name in species_names_fit:
                if compare_validation_mode == "output":
                    meas_col = _get_measurement_column_name(
                        compare_output_mode, sp_name
                    )
                    if meas_col not in df_cols:
                        parity_species_unavailable.append(
                            f"{sp_name}（缺少列 {meas_col}）"
                        )
                        continue
                    numeric_series = pd.to_numeric(df_fit[meas_col], errors="coerce")
                    if bool(np.any(np.isfinite(numeric_series.to_numpy()))):
                        parity_species_candidates.append(sp_name)
                    else:
                        parity_species_unavailable.append(
                            f"{sp_name}（列 {meas_col} 全为 NaN/非数字）"
                        )
                    continue

                # --- conversion 模式：按反应器/流动模型检查必要列 ---
                if reactor_type_fit == REACTOR_TYPE_BSTR:
                    required_cols = [
                        f"C0_{sp_name}_mol_m3",
                        f"Cout_{sp_name}_mol_m3",
                    ]
                    missing = [c for c in required_cols if c not in df_cols]
                    if missing:
                        parity_species_unavailable.append(
                            f"{sp_name}（缺少列: {', '.join(missing)}）"
                        )
                        continue
                    series_list = [
                        pd.to_numeric(df_fit[required_cols[0]], errors="coerce"),
                        pd.to_numeric(df_fit[required_cols[1]], errors="coerce"),
                    ]
                    if any(
                        bool(np.any(np.isfinite(s.to_numpy()))) for s in series_list
                    ):
                        parity_species_candidates.append(sp_name)
                    else:
                        parity_species_unavailable.append(
                            f"{sp_name}（C0/Cout 全为 NaN/非数字）"
                        )
                    continue

                if (
                    reactor_type_fit == REACTOR_TYPE_PFR
                    and str(pfr_flow_model_fit) == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P
                ):
                    required_cols = [
                        f"F0_{sp_name}_mol_s",
                        f"Fout_{sp_name}_mol_s",
                    ]
                    missing = [c for c in required_cols if c not in df_cols]
                    if missing:
                        parity_species_unavailable.append(
                            f"{sp_name}（气相 PFR 转化率需要列: {', '.join(missing)}）"
                        )
                        continue
                    series_list = [
                        pd.to_numeric(df_fit[required_cols[0]], errors="coerce"),
                        pd.to_numeric(df_fit[required_cols[1]], errors="coerce"),
                    ]
                    if any(
                        bool(np.any(np.isfinite(s.to_numpy()))) for s in series_list
                    ):
                        parity_species_candidates.append(sp_name)
                    else:
                        parity_species_unavailable.append(
                            f"{sp_name}（F0/Fout 全为 NaN/非数字）"
                        )
                    continue

                # 其他（液相 PFR / CSTR）：允许 F0/Fout；若缺则用 C0/Cout + vdot 换算
                need_vdot = "vdot_m3_s" in df_cols
                has_inlet = (f"F0_{sp_name}_mol_s" in df_cols) or (
                    need_vdot and (f"C0_{sp_name}_mol_m3" in df_cols)
                )
                has_outlet = (f"Fout_{sp_name}_mol_s" in df_cols) or (
                    need_vdot and (f"Cout_{sp_name}_mol_m3" in df_cols)
                )
                if not has_inlet or not has_outlet:
                    parts = []
                    if not has_inlet:
                        parts.append("入口缺少 F0 或 C0+vdot")
                    if not has_outlet:
                        parts.append("出口缺少 Fout 或 Cout+vdot")
                    parity_species_unavailable.append(
                        f"{sp_name}（{'；'.join(parts)}）"
                    )
                    continue

                inlet_col = (
                    f"F0_{sp_name}_mol_s"
                    if f"F0_{sp_name}_mol_s" in df_cols
                    else f"C0_{sp_name}_mol_m3"
                )
                outlet_col = (
                    f"Fout_{sp_name}_mol_s"
                    if f"Fout_{sp_name}_mol_s" in df_cols
                    else f"Cout_{sp_name}_mol_m3"
                )
                numeric_in = pd.to_numeric(df_fit[inlet_col], errors="coerce")
                numeric_out = pd.to_numeric(df_fit[outlet_col], errors="coerce")
                if bool(np.any(np.isfinite(numeric_in.to_numpy()))) and bool(
                    np.any(np.isfinite(numeric_out.to_numpy()))
                ):
                    parity_species_candidates.append(sp_name)
                else:
                    parity_species_unavailable.append(
                        f"{sp_name}（入口/出口列全为 NaN/非数字）"
                    )

            if parity_species_unavailable:
                show_missing = st.checkbox("显示无法绘图的物种原因", value=False)
                if show_missing:
                    st.caption(
                        "无法绘制奇偶校验图的物种： "
                        + "，".join(parity_species_unavailable)
                    )

            try:
                df_long = _build_fit_comparison_long_table(
                    data_df=df_fit,
                    species_names=species_names_fit,
                    output_mode=str(compare_output_mode),
                    output_species_list=parity_species_candidates,
                    stoich_matrix=stoich_matrix_fit,
                    fitted_params=fitted_params,
                    solver_method=solver_method_fit,
                    rtol=float(rtol_fit),
                    atol=float(atol_fit),
                    reactor_type=reactor_type_fit,
                    kinetic_model=kinetic_model_fit,
                    reversible_enabled=bool(reversible_enabled_fit),
                    pfr_flow_model=str(pfr_flow_model_fit),
                    max_step_fraction=float(max_step_fraction_fit),
                    validation_mode=str(compare_validation_mode),
                )
            except Exception as exc:
                st.error(f"生成对比数据失败: {exc}")
                df_long = pd.DataFrame()

            # 保存到 session_state 供导出 tab 使用
            st.session_state["fit_compare_long_df"] = df_long
            if df_long.empty:
                st.warning("对比数据为空：无法生成奇偶校验图。")
            else:
                # --- 布局：左侧选择数据/子图布局，右侧绘图附加选项 ---
                col_sel, col_opt = st.columns([1.35, 1.0])
                with col_sel:
                    species_selected = st.multiselect(
                        "选择要显示的物种",
                        list(parity_species_candidates),
                        default=list(parity_species_candidates),
                        help="仅对当前奇偶校验图与残差图生效。",
                    )
                    n_cols = int(
                        st.number_input(
                            "每行子图数",
                            min_value=1,
                            max_value=4,
                            value=2,
                            step=1,
                            help="仅影响子图排版；不改变拟合/预测结果。",
                        )
                    )

                with col_opt:
                    show_residual_plot = st.checkbox("显示残差图", value=True)
                    show_error_lines = st.checkbox("显示±误差线", value=True)
                    error_band_percent = float(
                        st.slider(
                            "相对误差带（%）",
                            min_value=0.0,
                            max_value=50.0,
                            value=10.0,
                            step=0.5,
                            key="parity_error_band_percent",
                            help="在 y=x 两侧绘制 y=(1±e)x 参考线，用于直观判断预测偏差范围。",
                            disabled=(not show_error_lines),
                        )
                    )
                    # 按温度着色选项
                    color_by_temp = False
                    if "T_K" in df_fit.columns:
                        color_by_temp = st.checkbox(
                            "按温度着色",
                            value=False,
                            key="parity_color_by_temp",
                            help="使用色温映射显示不同温度下的实验点，帮助识别温度相关的系统偏差。",
                        )

                st.divider()

                df_ok = df_long[df_long["ok"]].copy()
                df_ok = df_ok[
                    np.isfinite(df_ok["measured"]) & np.isfinite(df_ok["predicted"])
                ]
                # 合并温度列用于着色
                if color_by_temp and "T_K" in df_fit.columns and "row_index" in df_ok.columns:
                    df_ok["T_K"] = _map_parity_temperatures_by_row_index(
                        df_fit,
                        df_ok["row_index"],
                    ).to_numpy(dtype=float)
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
                        vals_all = np.concatenate(
                            [
                                df_ok["measured"].to_numpy(dtype=float),
                                df_ok["predicted"].to_numpy(dtype=float),
                            ]
                        )
                        vmin_auto = float(np.nanmin(vals_all))
                        vmax_auto = float(np.nanmax(vals_all))
                        if (not np.isfinite(vmin_auto)) or (not np.isfinite(vmax_auto)):
                            vmin_auto, vmax_auto = 0.0, 1.0
                        if vmax_auto <= vmin_auto:
                            vmax_auto = vmin_auto + 1.0
                        pad = 0.05 * float(vmax_auto - vmin_auto)
                        axis_min_auto = float(vmin_auto - pad)
                        axis_max_auto = float(vmax_auto + pad)

                        species_list_plot = list(
                            dict.fromkeys(df_ok["species"].tolist())
                        )
                        n_plots = len(species_list_plot)
                        n_rows = int(np.ceil(n_plots / max(n_cols, 1)))

                        axis_ranges_by_species = None
                        with st.expander(
                            "坐标范围设置（横纵一致 + 等比例）", expanded=False
                        ):
                            st.caption(
                                "默认按每个物种的数据范围自适应，同时保持该子图 x/y 横纵一致。"
                            )
                            axis_scope = st.radio(
                                "坐标范围作用域",
                                ["每个子图独立（x/y 相同）", "所有子图一致"],
                                index=0,
                                horizontal=True,
                                key="parity_axis_scope_v3",
                                help="每个子图独立：按当前物种的实验值和预测值共同确定一个范围，并同时用于 x/y；所有子图一致：所有物种共用同一个 x/y 范围。",
                            )
                            axis_range_mode = st.radio(
                                "范围来源",
                                ["自动（按数据）", "自定义"],
                                index=0,
                                horizontal=True,
                                key="parity_axis_range_mode",
                                help="自动：按数据最小/最大值（含 5% padding）确定坐标范围；自定义：手动输入 min/max。若选择“每个子图独立”，则可分别为每个子图设置 min/max。",
                            )
                            st.caption(
                                f"全局自动范围（用于统一坐标时的默认值）：[{axis_min_auto:.6g}, {axis_max_auto:.6g}]"
                            )

                            if axis_scope == "所有子图一致":
                                if axis_range_mode == "自定义":
                                    col_ax1, col_ax2 = st.columns([1, 1])
                                    axis_min_user = float(
                                        col_ax1.number_input(
                                            "坐标最小值",
                                            value=float(axis_min_auto),
                                            key="parity_axis_min",
                                        )
                                    )
                                    axis_max_user = float(
                                        col_ax2.number_input(
                                            "坐标最大值",
                                            value=float(axis_max_auto),
                                            key="parity_axis_max",
                                        )
                                    )
                                    if axis_max_user <= axis_min_user:
                                        st.warning(
                                            "坐标范围无效：需要满足 max > min。将回退到自动范围。"
                                        )
                                        axis_min_plot, axis_max_plot = (
                                            axis_min_auto,
                                            axis_max_auto,
                                        )
                                    else:
                                        axis_min_plot, axis_max_plot = (
                                            axis_min_user,
                                            axis_max_user,
                                        )
                                else:
                                    axis_min_plot, axis_max_plot = (
                                        axis_min_auto,
                                        axis_max_auto,
                                    )
                            else:
                                # 每个子图独立：先计算每个物种的自动范围；如选择自定义，则逐图覆盖
                                auto_ranges = {}
                                for species_name in species_list_plot:
                                    df_sp = df_ok[df_ok["species"] == species_name]
                                    vals_sp = np.concatenate(
                                        [
                                            df_sp["measured"].to_numpy(dtype=float),
                                            df_sp["predicted"].to_numpy(dtype=float),
                                        ]
                                    )
                                    vmin_sp = float(np.nanmin(vals_sp))
                                    vmax_sp = float(np.nanmax(vals_sp))
                                    if (not np.isfinite(vmin_sp)) or (
                                        not np.isfinite(vmax_sp)
                                    ):
                                        vmin_sp, vmax_sp = 0.0, 1.0
                                    if vmax_sp <= vmin_sp:
                                        vmax_sp = vmin_sp + 1.0
                                    pad_sp = 0.05 * float(vmax_sp - vmin_sp)
                                    auto_ranges[species_name] = (
                                        float(vmin_sp - pad_sp),
                                        float(vmax_sp + pad_sp),
                                    )

                                axis_ranges_by_species = dict(auto_ranges)
                                if axis_range_mode == "自定义":
                                    st.markdown("**逐图自定义**")
                                    st.caption(
                                        "每个子图的 x/y 使用相同 min/max，并保持等比例；若输入无效（max ≤ min），该子图会回退到自动范围。"
                                    )
                                    h1, h2, h3 = st.columns([1.2, 1, 1])
                                    h1.markdown("**物种**")
                                    h2.markdown("**min**")
                                    h3.markdown("**max**")
                                    invalid_species = []
                                    for idx, species_name in enumerate(
                                        species_list_plot
                                    ):
                                        c1, c2, c3 = st.columns([1.2, 1, 1])
                                        c1.write(species_name)
                                        auto_min, auto_max = auto_ranges[species_name]
                                        key_hash = hashlib.md5(
                                            str(species_name).encode("utf-8")
                                        ).hexdigest()[:12]
                                        axis_min_user = float(
                                            c2.number_input(
                                                "min",
                                                value=float(auto_min),
                                                key=f"parity_axis_min_{idx}_{key_hash}",
                                                label_visibility="collapsed",
                                            )
                                        )
                                        axis_max_user = float(
                                            c3.number_input(
                                                "max",
                                                value=float(auto_max),
                                                key=f"parity_axis_max_{idx}_{key_hash}",
                                                label_visibility="collapsed",
                                            )
                                        )
                                        if axis_max_user <= axis_min_user:
                                            invalid_species.append(species_name)
                                            axis_ranges_by_species[species_name] = (
                                                float(auto_min),
                                                float(auto_max),
                                            )
                                        else:
                                            axis_ranges_by_species[species_name] = (
                                                float(axis_min_user),
                                                float(axis_max_user),
                                            )
                                    if invalid_species:
                                        st.warning(
                                            "以下物种的坐标范围无效（max ≤ min），已回退到自动范围："
                                            + "，".join(map(str, invalid_species))
                                        )
                                else:
                                    # 自动范围：axis_ranges_by_species 已包含逐物种自动范围
                                    pass

                        fig, axes = plt.subplots(
                            n_rows,
                            n_cols,
                            figsize=(5.2 * n_cols, 4.3 * n_rows),
                            squeeze=False,
                        )

                        try:
                            for i, species_name in enumerate(species_list_plot):
                                ax = axes[i // n_cols][i % n_cols]
                                df_sp = df_ok[df_ok["species"] == species_name]
                                series_color = _fit_plot_color(i)
                                if color_by_temp and "T_K" in df_sp.columns:
                                    t_vals = df_sp["T_K"].to_numpy(dtype=float)
                                    sc = ax.scatter(
                                        df_sp["measured"].to_numpy(dtype=float),
                                        df_sp["predicted"].to_numpy(dtype=float),
                                        s=44,
                                        alpha=0.9,
                                        c=t_vals,
                                        cmap="coolwarm",
                                        edgecolors="#ffffff",
                                        linewidths=0.9,
                                        label=species_name,
                                        zorder=3,
                                    )
                                    plt.colorbar(sc, ax=ax, label="T [K]", shrink=0.8)
                                else:
                                    ax.scatter(
                                        df_sp["measured"].to_numpy(dtype=float),
                                        df_sp["predicted"].to_numpy(dtype=float),
                                        s=44,
                                        alpha=0.9,
                                        facecolors=series_color,
                                        edgecolors="#ffffff",
                                        linewidths=0.9,
                                        label=species_name,
                                        zorder=3,
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
                                # x/y 坐标范围 + 等比例（可全局统一，也可逐图独立）
                                if axis_ranges_by_species is None:
                                    axis_min_i, axis_max_i = (
                                        axis_min_plot,
                                        axis_max_plot,
                                    )
                                else:
                                    axis_min_i, axis_max_i = axis_ranges_by_species.get(
                                        species_name,
                                        (axis_min_auto, axis_max_auto),
                                    )
                                ax.set_xlim(axis_min_i, axis_max_i)
                                ax.set_ylim(axis_min_i, axis_max_i)
                                ax.set_aspect("equal", adjustable="box")

                                if (
                                    np.isfinite(min_v)
                                    and np.isfinite(max_v)
                                    and max_v > min_v
                                ):
                                    ax.plot(
                                        [axis_min_i, axis_max_i],
                                        [axis_min_i, axis_max_i],
                                        color="#000000",
                                        linestyle="--",
                                        linewidth=1.2,
                                        label="Ideal y = x",
                                    )
                                    if show_error_lines and (error_band_percent > 0.0):
                                        e = float(error_band_percent) / 100.0
                                        error_label = (
                                            f"± {error_band_percent:.1f}% band"
                                        )
                                        ax.plot(
                                            [axis_min_i, axis_max_i],
                                            [
                                                (1.0 - e) * axis_min_i,
                                                (1.0 - e) * axis_max_i,
                                            ],
                                            color="tab:gray",
                                            linestyle="--",
                                            linewidth=1.0,
                                            label=error_label,
                                        )
                                        ax.plot(
                                            [axis_min_i, axis_max_i],
                                            [
                                                (1.0 + e) * axis_min_i,
                                                (1.0 + e) * axis_max_i,
                                            ],
                                            color="tab:gray",
                                            linestyle="--",
                                            linewidth=1.0,
                                            label="_nolegend_",
                                        )
                                ax.set_title(f"Species: {species_name}")
                                ax.set_xlabel(
                                    ui_text.axis_label_with_unit(
                                        ui_text.AXIS_LABEL_MEASURED, unit_text_parity
                                    )
                                )
                                ax.set_ylabel(
                                    ui_text.axis_label_with_unit(
                                        ui_text.AXIS_LABEL_PREDICTED, unit_text_parity
                                    )
                                )
                                _style_fit_axis(ax, show_grid=False)
                                _apply_equal_parity_ticks(
                                    ax, axis_min_i, axis_max_i
                                )
                                _style_fit_legend(ax)

                            for j in range(n_plots, n_rows * n_cols):
                                axes[j // n_cols][j % n_cols].axis("off")

                            fig.tight_layout()
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
                        finally:
                            plt.close(fig)

                if show_residual_plot:
                    st.markdown("#### 残差图（预测值 - 实验值）")
                    df_res = df_long[df_long["ok"]].copy()
                    df_res = df_res[df_res["species"].isin(species_selected)]
                    df_res = df_res[
                        np.isfinite(df_res["residual"])
                        & np.isfinite(df_res["measured"])
                    ]
                    if df_res.empty:
                        st.warning("所选物种没有可用残差数据。")
                    else:
                        species_list_residual = [
                            sp
                            for sp in species_selected
                            if sp in set(df_res["species"])
                        ]
                        n_residual_plots = len(species_list_residual)
                        n_residual_rows = int(
                            np.ceil(n_residual_plots / max(int(n_cols), 1))
                        )

                        fig_r, axes_r = plt.subplots(
                            n_residual_rows,
                            n_cols,
                            figsize=(5.2 * n_cols, 4.0 * n_residual_rows),
                            squeeze=False,
                        )

                        try:
                            for i, species_name in enumerate(species_list_residual):
                                ax_r = axes_r[i // n_cols][i % n_cols]
                                df_sp = df_res[df_res["species"] == species_name]
                                series_color = _fit_plot_color(i)
                                ax_r.scatter(
                                    df_sp["measured"].to_numpy(dtype=float),
                                    df_sp["residual"].to_numpy(dtype=float),
                                    s=42,
                                    alpha=0.9,
                                    facecolors=series_color,
                                    edgecolors="#ffffff",
                                    linewidths=0.9,
                                    label=species_name,
                                    zorder=3,
                                )
                                ax_r.axhline(
                                    0.0,
                                    color="#000000",
                                    linestyle="--",
                                    linewidth=1.2,
                                    label="Zero residual",
                                )
                                ax_r.set_title(f"Species: {species_name}")
                                ax_r.set_xlabel(
                                    ui_text.axis_label_with_unit(
                                        ui_text.AXIS_LABEL_MEASURED, unit_text_parity
                                    )
                                )
                                ax_r.set_ylabel(
                                    ui_text.axis_label_with_unit(
                                        ui_text.AXIS_LABEL_RESIDUAL, unit_text_parity
                                    )
                                )
                                _style_fit_axis(ax_r, show_grid=False)
                                _style_fit_legend(ax_r)

                            for j in range(n_residual_plots, n_residual_rows * n_cols):
                                axes_r[j // n_cols][j % n_cols].axis("off")

                            fig_r.tight_layout()
                            st.pyplot(fig_r)
                            residual_image_format = st.selectbox(
                                "残差图像格式",
                                ["png", "svg"],
                                index=0,
                                key="residual_image_format",
                            )
                            st.download_button(
                                "📥 下载残差图",
                                ui_comp.figure_to_image_bytes(
                                    fig_r, residual_image_format
                                ),
                                file_name=f"residual_plot.{residual_image_format}",
                                mime=(
                                    "image/png"
                                    if residual_image_format == "png"
                                    else "image/svg+xml"
                                ),
                            )
                        finally:
                            plt.close(fig_r)

                show_compare_table = st.checkbox("显示预测 vs 实验对比表", value=False)
                if show_compare_table:
                    st.markdown("#### 预测 vs 实验对比表（含相对残差）")
                    df_show = df_long.copy()
                    df_show = df_show[df_show["species"].isin(species_selected)]
                    # 按用户需求：不显示 ok/message；新增 relative_residual（在构表阶段已计算）
                    drop_cols = [c for c in ["ok", "message"] if c in df_show.columns]
                    if drop_cols:
                        df_show = df_show.drop(columns=drop_cols)

                    preferred_order = [
                        "row_index",
                        "species",
                        "measured",
                        "predicted",
                        "residual",
                        "relative_residual",
                    ]
                    existing_preferred = [
                        c for c in preferred_order if c in df_show.columns
                    ]
                    remaining_cols = [
                        c for c in df_show.columns if c not in existing_preferred
                    ]
                    df_show = df_show[existing_preferred + remaining_cols]
                    st.dataframe(
                        df_show,
                        width="stretch",
                        height=UI_COMPARE_TABLE_HEIGHT_PX,
                    )

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
                        pd.DataFrame(rows_metric),
                        width="stretch",
                        height=UI_METRICS_TABLE_HEIGHT_PX,
                    )

        with tab_parity:
            _parity_tab_fragment()

        @st.fragment
        def _profile_tab_fragment():
            st.markdown("#### 沿程/随时间剖面")
            st.caption("说明：本页剖面为模型**预测**数据（不是实验测量值）。")
            if df_fit.empty:
                st.warning("数据为空：无法生成剖面。")
            else:
                row_indices = df_fit.index.tolist()
                selected_row_indices = st.multiselect(
                    "选择实验点（可多选以叠加对比）",
                    row_indices,
                    default=[row_indices[0]] if row_indices else [],
                    key="profile_selected_row_indices",
                    help="选择多行可在同一张图上叠加不同工况的剖面曲线。",
                )
                if not selected_row_indices:
                    st.info("请至少选择一个实验点。")
                    return
                selected_row_index = selected_row_indices[0]
                profile_points = int(
                    st.number_input(
                        "剖面点数",
                        min_value=UI_PROFILE_POINTS_MIN,
                        max_value=UI_PROFILE_POINTS_MAX,
                        value=UI_PROFILE_POINTS_DEFAULT,
                        step=UI_PROFILE_POINTS_STEP,
                        key="profile_points",
                    )
                )
                profile_species = st.multiselect(
                    "选择要画剖面的物种（可多选）",
                    list(species_names_fit),
                    default=list(species_names_fit[: min(3, len(species_names_fit))]),
                    key="profile_species",
                )

                multi_row_mode = len(selected_row_indices) > 1
                row_sel = df_fit.loc[selected_row_index]
                if reactor_type_fit == REACTOR_TYPE_PFR:
                    profile_kind_options = ["F (mol/s)", "C (mol/m^3)"]
                    profile_kind = st.radio(
                        "剖面变量",
                        profile_kind_options,
                        index=0,
                        horizontal=True,
                        format_func=lambda x: ui_text.map_label(
                            ui_text.PROFILE_KIND_LABELS, str(x)
                        ),
                        key="profile_kind",
                    )
                    pfr_flow_model_fit = str(
                        res.get("pfr_flow_model", PFR_FLOW_MODEL_LIQUID_CONST_VDOT)
                    ).strip()
                else:
                    profile_kind = "C (mol/m^3)"
                    pfr_flow_model_fit = PFR_FLOW_MODEL_LIQUID_CONST_VDOT

                # 序列化行数据为可哈希的 dict（供缓存函数使用）
                # 兼容 object 列中的数字字符串（如 "300"、"1e-3"）
                row_data_dict: dict[str, float] = {}
                for col in row_sel.index:
                    raw_value = row_sel.get(col, float("nan"))
                    try:
                        row_data_dict[str(col)] = float(raw_value)
                    except (TypeError, ValueError):
                        continue

                profile_result = _compute_reactor_profile(
                    reactor_type=reactor_type_fit,
                    kinetic_model=kinetic_model_fit,
                    reversible_enabled=bool(reversible_enabled_fit),
                    pfr_flow_model=pfr_flow_model_fit,
                    output_mode=output_mode_fit,
                    row_data_dict=row_data_dict,
                    species_names=tuple(species_names_fit),
                    fitted_params_frozen=_freeze_params(fitted_params),
                    stoich_matrix_tuple=tuple(tuple(r) for r in stoich_matrix_fit),
                    solver_method=solver_method_fit,
                    rtol=rtol_fit,
                    atol=atol_fit,
                    max_step_fraction=max_step_fraction_fit,
                    n_points=profile_points,
                )

                ok = profile_result["ok"]
                message = profile_result["message"]

                if not ok:
                    st.error(
                        f"{reactor_type_fit} 剖面计算失败: {message}\n"
                        "建议：尝试将求解器切换为 `BDF` 或 `Radau`，并适当放宽 `rtol/atol`。"
                    )
                else:
                    x_grid = np.array(profile_result["x_grid"])
                    profiles = np.array(profile_result["profiles"])
                    x_label_key = profile_result["x_label"]

                    name_to_index = {
                        name: i for i, name in enumerate(species_names_fit)
                    }

                    # --- 多行叠加模式：为每个选中的行计算剖面并叠加 ---
                    all_row_profiles: list[tuple[int, np.ndarray, np.ndarray, dict]] = []
                    # 第一行已经计算过
                    all_row_profiles.append((selected_row_index, x_grid, profiles, row_data_dict))

                    if multi_row_mode:
                        for extra_row_idx in selected_row_indices[1:]:
                            extra_row_sel = df_fit.loc[extra_row_idx]
                            extra_row_data: dict[str, float] = {}
                            for col in extra_row_sel.index:
                                try:
                                    extra_row_data[str(col)] = float(extra_row_sel.get(col, float("nan")))
                                except (TypeError, ValueError):
                                    continue
                            extra_result = _compute_reactor_profile(
                                reactor_type=reactor_type_fit,
                                kinetic_model=kinetic_model_fit,
                                reversible_enabled=bool(reversible_enabled_fit),
                                pfr_flow_model=pfr_flow_model_fit,
                                output_mode=output_mode_fit,
                                row_data_dict=extra_row_data,
                                species_names=tuple(species_names_fit),
                                fitted_params_frozen=_freeze_params(fitted_params),
                                stoich_matrix_tuple=tuple(tuple(r) for r in stoich_matrix_fit),
                                solver_method=solver_method_fit,
                                rtol=rtol_fit,
                                atol=atol_fit,
                                max_step_fraction=max_step_fraction_fit,
                                n_points=profile_points,
                            )
                            if extra_result["ok"]:
                                all_row_profiles.append((
                                    extra_row_idx,
                                    np.array(extra_result["x_grid"]),
                                    np.array(extra_result["profiles"]),
                                    extra_row_data,
                                ))

                    fig_prof = None
                    try:
                        fig_prof, ax_prof = plt.subplots(figsize=(5.5 if multi_row_mode else 4.6, 3.5 if multi_row_mode else 3.0))

                        profile_rows_for_export: list[dict[str, float | int | str]] = []
                        if x_label_key == "V_m3":
                            profile_df = pd.DataFrame({"V_m3": x_grid})
                            x_axis_label = ui_text.AXIS_LABEL_REACTOR_VOLUME
                        else:
                            profile_df = pd.DataFrame({"t_s": x_grid})
                            x_axis_label = ui_text.AXIS_LABEL_TIME

                        linestyles = ["-", "--", "-.", ":"]
                        for row_plot_idx, (rid, xg, profs, rd) in enumerate(all_row_profiles):
                            ls = linestyles[row_plot_idx % len(linestyles)]
                            row_label_suffix = f" (row {rid})" if multi_row_mode else ""
                            t_k_val = rd.get("T_K", None)
                            if multi_row_mode and t_k_val is not None:
                                row_label_suffix = f" ({t_k_val:.0f} K)"

                            for i, species_name in enumerate(profile_species):
                                idx = name_to_index[species_name]
                                series_color = _fit_plot_color(i)

                                if (
                                    reactor_type_fit == REACTOR_TYPE_PFR
                                    and profile_kind.startswith("F")
                                ):
                                    y = profs[idx, :]
                                    if row_plot_idx == 0:
                                        profile_df[f"F_{species_name}_mol_s"] = y
                                elif (
                                    reactor_type_fit == REACTOR_TYPE_PFR
                                    and not profile_kind.startswith("F")
                                ):
                                    cur_row_sel = df_fit.loc[rid]
                                    if (
                                        pfr_flow_model_fit
                                        == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P
                                    ):
                                        pressure_Pa = float(cur_row_sel.get("P_Pa", np.nan))
                                        temperature_K = float(cur_row_sel.get("T_K", np.nan))
                                        conc_total = float(pressure_Pa) / max(
                                            float(R_GAS_J_MOL_K) * float(temperature_K),
                                            EPSILON_CONCENTRATION,
                                        )
                                        total_flow = np.sum(profs, axis=0)
                                        y = (
                                            profs[idx, :]
                                            / np.maximum(total_flow, EPSILON_FLOW_RATE)
                                            * float(conc_total)
                                        )
                                    else:
                                        vdot_m3_s = float(cur_row_sel.get("vdot_m3_s", np.nan))
                                        y = profs[idx, :] / max(
                                            vdot_m3_s, EPSILON_FLOW_RATE
                                        )
                                    if row_plot_idx == 0:
                                        profile_df[f"C_{species_name}_mol_m3"] = y
                                else:
                                    # CSTR / BSTR: 直接就是浓度剖面
                                    y = profs[idx, :]
                                    if row_plot_idx == 0:
                                        profile_df[f"C_{species_name}_mol_m3"] = y

                                _plot_reference_series(
                                    ax_prof,
                                    xg,
                                    y,
                                    label=f"{species_name}{row_label_suffix}",
                                    color=series_color,
                                    linestyle=ls,
                                )

                                if multi_row_mode:
                                    for x_value, y_value in zip(xg, y):
                                        profile_rows_for_export.append(
                                            {
                                                "row_index": int(rid),
                                                x_label_key: float(x_value),
                                                "species": str(species_name),
                                                "profile_kind": str(profile_kind),
                                                "value": float(y_value),
                                            }
                                        )

                        if reactor_type_fit == REACTOR_TYPE_PFR:
                            y_axis_label = (
                                ui_text.AXIS_LABEL_FLOW_RATE
                                if profile_kind.startswith("F")
                                else ui_text.AXIS_LABEL_CONCENTRATION
                            )
                        else:
                            y_axis_label = ui_text.AXIS_LABEL_CONCENTRATION

                        ax_prof.set_xlabel(x_axis_label)
                        ax_prof.set_ylabel(y_axis_label)
                        _style_fit_axis(ax_prof, show_grid=False)
                        _style_fit_legend(ax_prof)
                        _render_centered_pyplot(fig_prof)

                        export_profile_df = (
                            pd.DataFrame(profile_rows_for_export)
                            if multi_row_mode
                            else profile_df
                        )
                        st.download_button(
                            "📥 下载剖面数据 CSV",
                            export_profile_df.to_csv(index=False).encode("utf-8"),
                            file_name="profile_data.csv",
                            mime="text/csv",
                        )
                        image_format_key = (
                            f"{reactor_type_fit.lower()}_profile_image_format"
                        )
                        image_format_prof = st.selectbox(
                            "剖面图格式",
                            ["png", "svg"],
                            index=0,
                            key=image_format_key,
                        )
                        st.download_button(
                            "📥 下载剖面图",
                            ui_comp.figure_to_image_bytes(fig_prof, image_format_prof),
                            file_name=f"profile_plot.{image_format_prof}",
                            mime=(
                                "image/png"
                                if image_format_prof == "png"
                                else "image/svg+xml"
                            ),
                        )
                    finally:
                        if fig_prof is not None:
                            plt.close(fig_prof)

        with tab_profile:
            _profile_tab_fragment()

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
                "📥 导出参数（k₀, Eₐ）CSV",
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
                df_export = df_long.copy()
                drop_cols = [c for c in ["ok", "message"] if c in df_export.columns]
                if drop_cols:
                    df_export = df_export.drop(columns=drop_cols)
                st.download_button(
                    "📥 导出预测 vs 实验对比（长表）CSV",
                    df_export.to_csv(index=False).encode("utf-8"),
                    file_name="pred_vs_meas_long.csv",
                    mime="text/csv",
                )
            else:
                st.info("先在「奇偶校验图」页生成对比数据后，再导出对比表。")

            # --- 一键导出完整报告（HTML）---
            st.divider()
            st.markdown("#### 完整拟合报告")
            if st.button("生成 HTML 报告", use_container_width=True):
                report_lines = [
                    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
                    "<title>Kinetics Fitting Report</title>",
                    "<style>body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px}",
                    "table{border-collapse:collapse;width:100%;margin:10px 0}",
                    "th,td{border:1px solid #ddd;padding:6px 10px;text-align:right}",
                    "th{background:#f5f5f7}</style></head><body>",
                    f"<h1>{reactor_type_fit} 反应动力学拟合报告</h1>",
                    f"<p><b>动力学模型</b>: {kinetic_model_fit}",
                    f"{'（可逆）' if reversible_enabled_fit else ''}</p>",
                    f"<p><b>目标函数 Φ</b>: {phi_text}（残差类型：{residual_type_used}）</p>",
                    "<h2>拟合参数</h2>",
                    df_k0_ea.to_html(),
                ]
                if kinetic_model_fit == KINETIC_MODEL_LANGMUIR_HINSHELWOOD:
                    if fitted_params.get("K0_ads") is not None:
                        df_ads_export = pd.DataFrame({
                            "K₀,ads": fitted_params["K0_ads"],
                            "Eₐ,K [J/mol]": fitted_params.get("Ea_K", []),
                        }, index=species_names_fit)
                        report_lines.append("<h3>L-H 参数</h3>")
                        report_lines.append(df_ads_export.to_html())
                if reversible_enabled_fit and fitted_params.get("k0_rev") is not None:
                    df_rev_export = pd.DataFrame({
                        "k₀,rev": fitted_params["k0_rev"],
                        "Eₐ,rev [J/mol]": fitted_params.get("ea_rev", []),
                    }, index=reaction_names)
                    report_lines.append("<h3>可逆反应参数</h3>")
                    report_lines.append(df_rev_export.to_html())
                report_lines.append("<h2>反应级数矩阵</h2>")
                report_lines.append(df_orders.to_html())
                if not df_long.empty:
                    report_lines.append("<h2>预测 vs 实验对比</h2>")
                    df_show = df_long.drop(columns=[c for c in ["ok", "message"] if c in df_long.columns], errors="ignore")
                    report_lines.append(df_show.to_html(index=False))
                report_lines.append("</body></html>")
                report_html = "\n".join(report_lines)
                st.download_button(
                    "📥 下载完整报告 (HTML)",
                    report_html.encode("utf-8"),
                    file_name="fitting_report.html",
                    mime="text/html",
                )
    return {}
