from __future__ import annotations

import math

import streamlit as st

import modules.ui_components as ui_comp
from modules.export_config import (
    build_export_config_from_ctx,
    persist_export_config,
    render_export_config_button,
)
from modules.constants import (
    DEFAULT_DIFF_STEP_REL,
    DEFAULT_EA_K_MAX_J_MOL,
    DEFAULT_EA_K_MIN_J_MOL,
    DEFAULT_EA_MAX_J_MOL,
    DEFAULT_EA_MIN_J_MOL,
    DEFAULT_K0_ADS_MAX,
    DEFAULT_K0_ADS_MIN,
    DEFAULT_K0_MAX,
    DEFAULT_K0_MIN,
    DEFAULT_MAX_NFEV,
    DEFAULT_MAX_NFEV_COARSE,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_N_STARTS,
    DEFAULT_ORDER_MAX,
    DEFAULT_ORDER_MIN,
    DEFAULT_RANDOM_SEED,
    DEFAULT_EA_REV_MAX_J_MOL,
    DEFAULT_EA_REV_MIN_J_MOL,
    DEFAULT_K0_REV_MAX,
    DEFAULT_K0_REV_MIN,
    DEFAULT_ORDER_REV_MAX,
    DEFAULT_ORDER_REV_MIN,
    UI_MAX_NFEV_STEP,
    UI_MAX_STEP_FRACTION_STEP,
    DEFAULT_USE_LOG_K0_ADS_FIT,
    DEFAULT_USE_LOG_K0_FIT,
    DEFAULT_USE_LOG_K0_REV_FIT,
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
)


def render_fit_advanced(ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    export_config_placeholder = ctx["export_config_placeholder"]
    kinetic_model = ctx["kinetic_model"]
    reversible_enabled = bool(ctx.get("reversible_enabled", False))
    output_mode = ctx["output_mode"]
    output_species_list = ctx["output_species_list"]

    def _safe_float(value, fallback: float) -> float:
        try:
            x = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(fallback)
        if not math.isfinite(x):
            return float(fallback)
        return float(x)

    def _safe_float_min(value, fallback: float, lower: float) -> float:
        x = _safe_float(value, fallback)
        return float(max(float(lower), x))

    def _safe_int_min(value, fallback: int, lower: int) -> int:
        try:
            x = int(value)
        except (TypeError, ValueError, OverflowError):
            x = int(fallback)
        return int(max(int(lower), int(x)))

    k0_min = float(get_cfg("k0_min", DEFAULT_K0_MIN))
    k0_max = float(get_cfg("k0_max", DEFAULT_K0_MAX))
    ea_min = float(get_cfg("ea_min_J_mol", DEFAULT_EA_MIN_J_MOL))
    ea_max = float(get_cfg("ea_max_J_mol", DEFAULT_EA_MAX_J_MOL))
    ord_min = float(get_cfg("order_min", DEFAULT_ORDER_MIN))
    ord_max = float(get_cfg("order_max", DEFAULT_ORDER_MAX))
    K0_ads_min = float(get_cfg("K0_ads_min", DEFAULT_K0_ADS_MIN))
    K0_ads_max = float(get_cfg("K0_ads_max", DEFAULT_K0_ADS_MAX))
    Ea_K_min = float(get_cfg("Ea_K_min", DEFAULT_EA_K_MIN_J_MOL))
    Ea_K_max = float(get_cfg("Ea_K_max", DEFAULT_EA_K_MAX_J_MOL))
    k0_rev_min = float(ctx.get("k0_rev_min", get_cfg("k0_rev_min", DEFAULT_K0_REV_MIN)))
    k0_rev_max = float(ctx.get("k0_rev_max", get_cfg("k0_rev_max", DEFAULT_K0_REV_MAX)))
    ea_rev_min_J_mol = float(
        ctx.get(
            "ea_rev_min_J_mol",
            get_cfg("ea_rev_min_J_mol", DEFAULT_EA_REV_MIN_J_MOL),
        )
    )
    ea_rev_max_J_mol = float(
        ctx.get(
            "ea_rev_max_J_mol",
            get_cfg("ea_rev_max_J_mol", DEFAULT_EA_REV_MAX_J_MOL),
        )
    )
    order_rev_min = float(
        ctx.get("order_rev_min", get_cfg("order_rev_min", DEFAULT_ORDER_REV_MIN))
    )
    order_rev_max = float(
        ctx.get("order_rev_max", get_cfg("order_rev_max", DEFAULT_ORDER_REV_MAX))
    )

    with st.expander("高级设置 (点击展开)", expanded=False):
        st.markdown("**1. 算法与鲁棒性**")

        with st.container(border=True):
            st.caption("求解器与差分参数")
            col_iter1, col_iter2, col_iter3 = st.columns(3)
            with col_iter1:
                max_nfev = int(
                    st.number_input(
                        "最大迭代次数（max_nfev）",
                        value=_safe_int_min(
                            get_cfg("max_nfev", DEFAULT_MAX_NFEV), DEFAULT_MAX_NFEV, 1
                        ),
                        min_value=1,
                        step=UI_MAX_NFEV_STEP,
                        key="cfg_max_nfev",
                        help="提示：每次外层迭代中，数值差分 Jacobian 需要多次模型调用，因此显示的总调用次数通常大于该值。",
                    )
                )
            with col_iter2:
                diff_step_rel = ui_comp.smart_number_input(
                    "差分步长（diff_step）",
                    value=_safe_float_min(
                        get_cfg("diff_step_rel", DEFAULT_DIFF_STEP_REL),
                        DEFAULT_DIFF_STEP_REL,
                        1e-15,
                    ),
                    min_value=1e-15,
                    key="cfg_diff_step_rel",
                    help="用于 least_squares 的数值差分 Jacobian 相对步长；拟合停滞时可适当调大，拟合过粗时可适当调小。",
                )
            with col_iter3:
                max_step_fraction = ui_comp.smart_number_input(
                    "最大步长比例（max_step_fraction）",
                    value=_safe_float_min(
                        get_cfg("max_step_fraction", DEFAULT_MAX_STEP_FRACTION),
                        DEFAULT_MAX_STEP_FRACTION,
                        0.0,
                    ),
                    min_value=0.0,
                    max_value=10.0,
                    step=UI_MAX_STEP_FRACTION_STEP,
                    key="cfg_max_step_fraction",
                    help="用于 solve_ivp 的积分步长上限：max_step = fraction × 总时间/总体积；0 表示不限制。",
                )
            col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
            use_x_scale_jac = col_opt1.checkbox(
                "启用雅可比尺度归一（x_scale='jac'）",
                value=get_cfg("use_x_scale_jac", True),
                key="cfg_use_x_scale_jac",
            )
            use_log_k0_fit = col_opt2.checkbox(
                "k₀ 用 log 拟合",
                value=bool(get_cfg("use_log_k0_fit", DEFAULT_USE_LOG_K0_FIT)),
                key="cfg_use_log_k0_fit",
                help="启用后，优化器在 log 空间中拟合 k₀，但模型计算仍使用线性空间的 k₀。",
            )
            use_log_k0_rev_fit = col_opt3.checkbox(
                "k₀,rev 用 log 拟合",
                value=bool(
                    get_cfg("use_log_k0_rev_fit", DEFAULT_USE_LOG_K0_REV_FIT)
                ),
                key="cfg_use_log_k0_rev_fit",
                disabled=(not reversible_enabled),
                help="仅对启用可逆反应后参与拟合的 k₀,rev 生效。",
            )
            use_log_K0_ads_fit = col_opt4.checkbox(
                "K₀,ads 用 log 拟合",
                value=bool(
                    get_cfg("use_log_K0_ads_fit", DEFAULT_USE_LOG_K0_ADS_FIT)
                ),
                key="cfg_use_log_K0_ads_fit",
                disabled=(kinetic_model != KINETIC_MODEL_LANGMUIR_HINSHELWOOD),
                help="仅对 L-H 模型下参与拟合的 K₀,ads 生效；启用时对应初值和下界必须 > 0。",
            )

        with st.container(border=True):
            st.caption("多起点搜索（Multi-start）参数")
            use_ms = st.checkbox(
                "启用多起点搜索（Multi-start）",
                value=bool(get_cfg("use_multi_start", True)),
                key="cfg_use_multi_start",
            )
            disable_ms_fields = not bool(use_ms)
            col_ms1, col_ms2, col_ms3 = st.columns(3)
            with col_ms1:
                n_starts = int(
                    st.number_input(
                        "起点数量（n_starts）",
                        value=_safe_int_min(
                            get_cfg("n_starts", DEFAULT_N_STARTS), DEFAULT_N_STARTS, 1
                        ),
                        min_value=1,
                        step=1,
                        key="cfg_n_starts",
                        help="仅在启用多起点搜索且 n_starts > 1 时生效。",
                        disabled=disable_ms_fields,
                    )
                )
            with col_ms2:
                max_nfev_coarse = int(
                    st.number_input(
                        "粗拟合迭代上限（max_nfev_coarse）",
                        value=_safe_int_min(
                            get_cfg("max_nfev_coarse", DEFAULT_MAX_NFEV_COARSE),
                            DEFAULT_MAX_NFEV_COARSE,
                            1,
                        ),
                        min_value=1,
                        step=50,
                        key="cfg_max_nfev_coarse",
                        help="仅在启用多起点搜索时，用于每个起点的粗拟合阶段。",
                        disabled=disable_ms_fields,
                    )
                )
            with col_ms3:
                random_seed = int(
                    st.number_input(
                        "随机种子（random_seed）",
                        value=_safe_int_min(
                            get_cfg("random_seed", DEFAULT_RANDOM_SEED),
                            DEFAULT_RANDOM_SEED,
                            0,
                        ),
                        min_value=0,
                        step=1,
                        key="cfg_random_seed",
                        disabled=disable_ms_fields,
                    )
                )

        st.divider()
        st.markdown("**2. 目标函数设置**")
        st.caption("目标函数定义残差的计算方式，不同类型适用于不同数据特征：")

        residual_type_options = [
            "绝对残差",
            "相对残差",
            "百分比残差",
        ]
        residual_type_default = str(get_cfg("residual_type", "绝对残差"))
        if residual_type_default not in residual_type_options:
            residual_type_default = "绝对残差"
        residual_type_index = residual_type_options.index(residual_type_default)

        residual_type = st.selectbox(
            "残差类型",
            options=residual_type_options,
            index=residual_type_index,
            key="cfg_residual_type",
            help="选择用于构建目标函数的残差计算方式",
        )

        # 显示当前残差类型的公式说明
        residual_formula_info = {
            "绝对残差": (
                "**绝对残差（Absolute Residual）**\n\n"
                r"$r_i = y_i^{pred} - y_i^{meas}$"
                "\n\n适用于：测量值数量级相近的数据。当测量值范围差异大时，大值主导拟合。"
            ),
            "相对残差": (
                "**相对残差（Relative Residual）**\n\n"
                r"$r_i = \frac{y_i^{pred} - y_i^{meas}}{\mathrm{sign}(y_i^{meas})\cdot\max(|y_i^{meas}|,\epsilon)}$"
                "\n\n适用于：测量值跨越多个数量级的数据。对所有数据点给予相近权重。\n\n"
                r"其中 $\epsilon$ 为小正数（同百分比残差使用的数值尺度），用于避免 $y_i^{meas}\approx 0$ 时的除零问题。"
            ),
            "百分比残差": (
                "**百分比残差（Percentage Residual with offset）**\n\n"
                r"$r_i = 100 \times \frac{y_i^{pred} - y_i^{meas}}{|y_i^{meas}| + \epsilon}$"
                "\n\n"
                r"其中 $\epsilon$ 为小正数（典型值的 1%），避免除零；$r_i$ 的单位为 %。"
                "\n\n适用于：测量值可能接近零的数据。兼顾相对误差与数值稳定性。"
            ),
        }
        with st.container(border=True):
            st.markdown(
                '<div class="kinetics-card-marker"></div>', unsafe_allow_html=True
            )
            st.markdown(residual_formula_info.get(residual_type, ""))

        st.divider()
        st.caption(
            "说明：当模型计算失败（如 solve_ivp 失败）时，残差会使用系统默认罚项（不在 UI 中提供调节）。"
        )

    if export_config_placeholder is not None:
        export_config_placeholder.empty()
        export_cfg = build_export_config_from_ctx(
            ctx,
            output_mode,
            output_species_list,
            advanced_overrides={
                "k0_min": float(k0_min),
                "k0_max": float(k0_max),
                "ea_min_J_mol": float(ea_min),
                "ea_max_J_mol": float(ea_max),
                "order_min": float(ord_min),
                "order_max": float(ord_max),
                "K0_ads_min": float(K0_ads_min),
                "K0_ads_max": float(K0_ads_max),
                "Ea_K_min": float(Ea_K_min),
                "Ea_K_max": float(Ea_K_max),
                "k0_rev_min": float(k0_rev_min),
                "k0_rev_max": float(k0_rev_max),
                "ea_rev_min_J_mol": float(ea_rev_min_J_mol),
                "ea_rev_max_J_mol": float(ea_rev_max_J_mol),
                "order_rev_min": float(order_rev_min),
                "order_rev_max": float(order_rev_max),
                "residual_type": str(residual_type),
                "diff_step_rel": float(diff_step_rel),
                "max_nfev": int(max_nfev),
                "use_x_scale_jac": bool(use_x_scale_jac),
                "use_log_k0_fit": bool(use_log_k0_fit),
                "use_log_k0_rev_fit": bool(use_log_k0_rev_fit),
                "use_log_K0_ads_fit": bool(use_log_K0_ads_fit),
                "use_multi_start": bool(use_ms),
                "n_starts": int(n_starts),
                "max_nfev_coarse": int(max_nfev_coarse),
                "random_seed": int(random_seed),
                "max_step_fraction": float(max_step_fraction),
            },
        )
        ok, message = persist_export_config(export_cfg, ctx["session_id"])
        if not ok and message:
            st.warning(message)
        render_export_config_button(
            export_config_placeholder,
            export_cfg,
            button_key="export_config_download_advanced",
        )

    return {
        "k0_min": float(k0_min),
        "k0_max": float(k0_max),
        "ea_min": float(ea_min),
        "ea_max": float(ea_max),
        "ord_min": float(ord_min),
        "ord_max": float(ord_max),
        "K0_ads_min": float(K0_ads_min),
        "K0_ads_max": float(K0_ads_max),
        "Ea_K_min": float(Ea_K_min),
        "Ea_K_max": float(Ea_K_max),
        "k0_rev_min": float(k0_rev_min),
        "k0_rev_max": float(k0_rev_max),
        "ea_rev_min_J_mol": float(ea_rev_min_J_mol),
        "ea_rev_max_J_mol": float(ea_rev_max_J_mol),
        "order_rev_min": float(order_rev_min),
        "order_rev_max": float(order_rev_max),
        "max_nfev": int(max_nfev),
        "diff_step_rel": float(diff_step_rel),
        "max_step_fraction": float(max_step_fraction),
        "use_ms": bool(use_ms),
        "n_starts": int(n_starts),
        "max_nfev_coarse": int(max_nfev_coarse),
        "use_x_scale_jac": bool(use_x_scale_jac),
        "use_log_k0_fit": bool(use_log_k0_fit),
        "use_log_k0_rev_fit": bool(use_log_k0_rev_fit),
        "use_log_K0_ads_fit": bool(use_log_K0_ads_fit),
        "random_seed": int(random_seed),
        "residual_type": str(residual_type),
    }
