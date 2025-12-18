from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


R_GAS_J_MOL_K = 8.314462618  # Gas constant [J/(mol*K)]


def _clean_species_names(species_text: str) -> list[str]:
    parts = [p.strip() for p in species_text.split(",")]
    names = [p for p in parts if p]
    unique_names = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)
    return unique_names


def _safe_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _to_float_or_nan(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _make_number_format_string(number_style: str, decimal_places: int) -> str:
    decimal_places = int(decimal_places)
    if number_style == "科学计数":
        return f"%.{decimal_places}e"
    return f"%.{decimal_places}f"


def _build_table_column_config(data_df: pd.DataFrame, number_format: str) -> dict:
    column_config: dict = {}
    for col in data_df.columns:
        if pd.api.types.is_numeric_dtype(data_df[col]):
            column_config[col] = st.column_config.NumberColumn(
                col, format=number_format
            )
        else:
            column_config[col] = st.column_config.TextColumn(col)
    return column_config


def _apply_plot_tick_format(
    ax: plt.Axes, number_style: str, decimal_places: int, use_auto: bool
) -> None:
    if use_auto:
        return

    decimal_places = int(decimal_places)
    if number_style == "科学计数":
        formatter = FuncFormatter(
            lambda x, pos: (
                "" if (not np.isfinite(x)) else f"{float(x):.{decimal_places}e}"
            )
        )
    else:
        formatter = FuncFormatter(
            lambda x, pos: (
                "" if (not np.isfinite(x)) else f"{float(x):.{decimal_places}f}"
            )
        )

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def _build_default_nu_table(species_names: list[str], n_reactions: int) -> pd.DataFrame:
    nu_default = pd.DataFrame(
        data=np.zeros((len(species_names), n_reactions), dtype=float),
        index=species_names,
        columns=[f"R{j+1}" for j in range(n_reactions)],
    )

    # 默认示例：A -> B（仅对第一个反应 R1 生效；若 A/B 不存在则跳过）
    if n_reactions >= 1:
        if "A" in nu_default.index:
            nu_default.loc["A", "R1"] = -1.0
        if "B" in nu_default.index:
            nu_default.loc["B", "R1"] = 1.0
    return nu_default


def _build_default_order_table(
    species_names: list[str], n_reactions: int
) -> pd.DataFrame:
    order_default = pd.DataFrame(
        data=np.zeros((n_reactions, len(species_names)), dtype=float),
        index=[f"R{j+1}" for j in range(n_reactions)],
        columns=species_names,
    )

    # 默认示例：r = k * C_A^1（仅对第一个反应 R1 生效；若 A 不存在则跳过）
    if n_reactions >= 1 and "A" in order_default.columns:
        order_default.loc["R1", "A"] = 1.0
    return order_default


def calc_rate_vector_power_law(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
) -> np.ndarray:
    """
    conc_mol_m3: shape (n_species,)
    k0: shape (n_reactions,)   pre-exponential factor
    ea_J_mol: shape (n_reactions,) activation energy [J/mol]
    reaction_order_matrix: shape (n_reactions, n_species)
    """
    conc_mol_m3 = _safe_nonnegative(conc_mol_m3)
    k_T = k0 * np.exp(-ea_J_mol / (R_GAS_J_MOL_K * temperature_K))

    # rate_j = k_j(T) * Π_i C_i^(n_ij)
    n_reactions = k0.size
    rate_vector = np.zeros(n_reactions, dtype=float)
    for reaction_index in range(n_reactions):
        rate_value = k_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = reaction_order_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            rate_value = rate_value * (conc_mol_m3[species_index] ** order_value)
        rate_vector[reaction_index] = rate_value
    return rate_vector


def integrate_pfr_molar_flows(
    reactor_volume_m3: float,
    temperature_K: float,
    vdot_m3_s: float,
    molar_flow_inlet_mol_s: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, bool, str]:
    """
    PFR design equation (liquid / constant volumetric flow):
      dF_i/dV = Σ_j nu_{i,j} r_j
      C_i = F_i / vdot
    """
    if not np.isfinite(reactor_volume_m3):
        return molar_flow_inlet_mol_s.copy(), False, "V_m3 无效（NaN/Inf）"
    if reactor_volume_m3 < 0.0:
        return molar_flow_inlet_mol_s.copy(), False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        return molar_flow_inlet_mol_s.copy(), True, "V=0"

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return molar_flow_inlet_mol_s.copy(), False, "温度 T_K 无效"
    if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
        return molar_flow_inlet_mol_s.copy(), False, "体积流量 vdot_m3_s 无效"

    if not np.all(np.isfinite(molar_flow_inlet_mol_s)):
        return molar_flow_inlet_mol_s.copy(), False, "入口摩尔流量包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return molar_flow_inlet_mol_s.copy(), False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return molar_flow_inlet_mol_s.copy(), False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return molar_flow_inlet_mol_s.copy(), False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return molar_flow_inlet_mol_s.copy(), False, "反应级数矩阵 n 包含 NaN/Inf"

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        conc_mol_m3 = _safe_nonnegative(molar_flow_mol_s) / max(vdot_m3_s, 1e-30)
        rate_vector = calc_rate_vector_power_law(
            conc_mol_m3=conc_mol_m3,
            temperature_K=temperature_K,
            k0=k0,
            ea_J_mol=ea_J_mol,
            reaction_order_matrix=reaction_order_matrix,
        )
        dF_dV = stoich_matrix @ rate_vector
        return dF_dV

    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reactor_volume_m3)),
            y0=molar_flow_inlet_mol_s.astype(float),
            method=solver_method,
            rtol=rtol,
            atol=atol,
        )
    except Exception as exc:
        return molar_flow_inlet_mol_s.copy(), False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return molar_flow_inlet_mol_s.copy(), False, str(message)

    molar_flow_outlet = solution.y[:, -1]
    return molar_flow_outlet, True, "OK"


def _pack_parameters(
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
) -> np.ndarray:
    parts = []
    # k0
    if np.any(fit_k0_flags):
        k0_to_fit = k0_guess[fit_k0_flags]
        parts.append(k0_to_fit)
    # Ea
    if np.any(fit_ea_flags):
        parts.append(ea_guess_J_mol[fit_ea_flags])
    # Orders (matrix flattened)
    # fit_order_flags_matrix: shape (n_reactions, n_species)
    order_mask_flat = np.asarray(fit_order_flags_matrix, dtype=bool).ravel()
    if np.any(order_mask_flat):
        parts.append(order_guess.ravel()[order_mask_flat])

    if len(parts) == 0:
        return np.array([], dtype=float)
    return np.concatenate(parts).astype(float)


def _unpack_parameters(
    parameter_vector: np.ndarray,
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = 0

    # Start with guesses (defaults)
    k0 = k0_guess.copy().astype(float)
    ea_J_mol = ea_guess_J_mol.copy().astype(float)
    reaction_order_matrix = order_guess.copy().astype(float)

    # 1. k0
    n_fit_k0 = np.sum(fit_k0_flags)
    if n_fit_k0 > 0:
        k0_part = parameter_vector[index : index + n_fit_k0]
        # Update only fitted indices
        k0[fit_k0_flags] = k0_part
        index += n_fit_k0

    # 2. Ea
    n_fit_ea = np.sum(fit_ea_flags)
    if n_fit_ea > 0:
        ea_part = parameter_vector[index : index + n_fit_ea]
        ea_J_mol[fit_ea_flags] = ea_part
        index += n_fit_ea

    # 3. Orders
    n_reactions, n_species = reaction_order_matrix.shape
    order_mask_flat = np.asarray(fit_order_flags_matrix, dtype=bool).ravel()
    n_fit_orders = np.sum(order_mask_flat)

    if n_fit_orders > 0:
        order_part = parameter_vector[index : index + n_fit_orders]
        # Current full flattened array
        flat_orders = reaction_order_matrix.ravel()
        # Update fitted slots
        flat_orders[order_mask_flat] = order_part
        # Reshape back
        reaction_order_matrix = flat_orders.reshape((n_reactions, n_species))
        index += n_fit_orders

    return k0, ea_J_mol, reaction_order_matrix


def _build_bounds(
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    k0_min: float,
    k0_max: float,
    ea_min_J_mol: float,
    ea_max_J_mol: float,
    order_min: float,
    order_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower_parts = []
    upper_parts = []

    # 1. k0
    n_fit_k0 = np.sum(fit_k0_flags)
    if n_fit_k0 > 0:
        # We need to construct bounds just for the fitted elements
        # But here min/max are scalars. so we make array of size n_fit_k0
        lower_parts.append(np.full(n_fit_k0, k0_min, dtype=float))
        upper_parts.append(np.full(n_fit_k0, k0_max, dtype=float))

    # 2. Ea
    n_fit_ea = np.sum(fit_ea_flags)
    if n_fit_ea > 0:
        lower_parts.append(np.full(n_fit_ea, ea_min_J_mol, dtype=float))
        upper_parts.append(np.full(n_fit_ea, ea_max_J_mol, dtype=float))

    # 3. Orders
    # Total fitted order parameters = number of True cells
    n_fit_orders_total = int(np.sum(np.asarray(fit_order_flags_matrix, dtype=bool)))
    if n_fit_orders_total > 0:
        lower_parts.append(np.full(n_fit_orders_total, order_min, dtype=float))
        upper_parts.append(np.full(n_fit_orders_total, order_max, dtype=float))

    if len(lower_parts) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(lower_parts), np.concatenate(upper_parts)


def _predict_outputs_for_row(
    row: pd.Series,
    species_names: list[str],
    output_mode: str,
    output_species_list: list[str],
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, bool, str]:
    reactor_volume_m3 = _to_float_or_nan(row.get("V_m3", np.nan))
    if not np.isfinite(reactor_volume_m3):
        return np.zeros(len(output_species_list), dtype=float), False, "缺少 V_m3"
    if reactor_volume_m3 < 0.0:
        return np.zeros(len(output_species_list), dtype=float), False, "V_m3 不能为负"

    temperature_K = _to_float_or_nan(row.get("T_K", np.nan))
    vdot_m3_s = _to_float_or_nan(row.get("vdot_m3_s", np.nan))

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            "温度 T_K 无效（请检查 CSV 的 T_K 列）",
        )
    if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            "体积流量 vdot_m3_s 无效（请检查 CSV 的 vdot_m3_s 列）",
        )

    molar_flow_inlet = np.zeros(len(species_names), dtype=float)
    for i, name in enumerate(species_names):
        col = f"F0_{name}_mol_s"
        value = _to_float_or_nan(row.get(col, np.nan))
        if not np.isfinite(value):
            return np.zeros(len(output_species_list), dtype=float), False, f"缺少 {col}"
        if value < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                f"{col} 不能为负",
            )
        molar_flow_inlet[i] = float(value)

    molar_flow_outlet, ok, message = integrate_pfr_molar_flows(
        reactor_volume_m3=reactor_volume_m3,
        temperature_K=temperature_K,
        vdot_m3_s=vdot_m3_s,
        molar_flow_inlet_mol_s=molar_flow_inlet,
        stoich_matrix=stoich_matrix,
        k0=k0,
        ea_J_mol=ea_J_mol,
        reaction_order_matrix=reaction_order_matrix,
        solver_method=solver_method,
        rtol=rtol,
        atol=atol,
    )
    if not ok:
        return np.zeros(len(output_species_list), dtype=float), False, message

    name_to_index = {name: i for i, name in enumerate(species_names)}
    output_values = np.zeros(len(output_species_list), dtype=float)
    for out_i, species in enumerate(output_species_list):
        idx = name_to_index[species]
        if output_mode == "Fout (mol/s)":
            output_values[out_i] = molar_flow_outlet[idx]
        elif output_mode == "Cout (mol/m^3)":
            output_values[out_i] = molar_flow_outlet[idx] / max(vdot_m3_s, 1e-30)
        elif output_mode == "X (conversion)":
            f0 = molar_flow_inlet[idx]
            fout = molar_flow_outlet[idx]
            # 对于入口流量为零的物种，转化率无意义，返回 NaN
            if f0 < 1e-30:
                output_values[out_i] = np.nan
            else:
                output_values[out_i] = (f0 - fout) / f0
        else:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "未知输出模式",
            )

    return output_values, True, "OK"


def main() -> None:
    st.set_page_config(
        page_title="Kinetics_app | PFR 动力学拟合",
        layout="wide",
        page_icon="⚗️",
    )

    # --- UI styles (main theme in `.streamlit/config.toml`) ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(15, 23, 42, 0.12);
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Plot Style ---
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    # Custom Plot Styling to match UI
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Matplotlib 显示中文需要指定可用的中文字体作为后备字体
            # Windows 常见：Microsoft YaHei / SimHei；macOS：PingFang SC；Linux：Noto Sans CJK SC / WenQuanYi Zen Hei
            "font.sans-serif": [
                "Inter",
                "Microsoft YaHei",
                "SimHei",
                "PingFang SC",
                "Noto Sans CJK SC",
                "WenQuanYi Zen Hei",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#e2e8f0",
            "axes.labelcolor": "#475569",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "text.color": "#1e293b",
            "grid.color": "#f1f5f9",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.titleweight": "bold",
        }
    )

    st.title("⚗️ PFR 反应动力学参数拟合")
    st.caption("PFR 数值积分（solve_ivp）+ 最小二乘拟合（least_squares）")

    with st.container(border=True):
        st.markdown(
            "**快速流程：**\n"
            "1) 在 **① 反应定义** 中输入物种/反应与初值；\n"
            "2) 在 **② 实验数据** 中上传 CSV，并勾选进入目标函数的物种；\n"
            "3) 在 **③ 参数拟合** 中设置边界并开始拟合。"
        )

    with st.expander("📖 查看详细理论模型与符号说明", expanded=False):
        st.markdown("#### 1. 反应动力学模型 (Reaction Kinetics)")
        st.markdown(
            "对于第 $j$ 个反应，其反应速率 $r_j$ 采用 **幂函数定律 (Power Law)** 结合 **Arrhenius 方程** 描述："
        )
        st.latex(
            r"""
            r_j = k_j(T) \prod_{i} C_i^{n_{i,j}} 
            """
        )
        st.latex(
            r"""
            k_j(T) = k_{0,j} \exp\left(-\frac{E_{a,j}}{R T}\right)
            """
        )

        st.markdown("#### 2. 反应器设计方程 (Reactor Model)")
        st.markdown(
            "采用 **平推流反应器 (PFR)** 模型，假设稳态、恒定体积流量（液相）："
        )
        st.latex(
            r"""
            \frac{dF_i}{dV} = \sum_{j=1}^{N_{rxn}} \nu_{i,j} r_j 
            """
        )
        st.markdown("其中浓度 $C_i$ 与摩尔流量 $F_i$ 的关系为：")
        st.latex(r"C_i = \frac{F_i}{\dot{v}}")

        st.markdown("#### 3. 参数拟合目标 (Optimization Objective)")
        st.markdown("通过调整参数 $\theta$ (即 $k_0, E_a, n$) 最小化加权残差平方和：")
        st.latex(
            r"""
            \min_{\theta} \sum_{k} \left[ w_k \cdot \left( y_{\text{pred}, k}(\theta) - y_{\text{meas}, k} \right) \right]^2
            """
        )

        st.markdown("#### 4. 符号说明 (Nomenclature)")
        st.markdown(
            r"""
| 符号 (Symbol) | 含义 (Description) | 标准单位 (SI Unit) |
| :--- | :--- | :--- |
| $r_j$ | 第 $j$ 个反应的反应速率 | $\text{mol} \cdot \text{m}^{-3} \cdot \text{s}^{-1}$ |
| $k_j(T)$ | 第 $j$ 个反应的速率常数 | 取决于反应级数 |
| $k_{0,j}$ | 指前因子 (Pre-exponential factor) | 取决于反应级数 |
| $E_{a,j}$ | 活化能 (Activation Energy) | $\text{J} \cdot \text{mol}^{-1}$ |
| $R$ | 通用气体常数 | $8.314 \text{ J} \cdot \text{mol}^{-1} \cdot \text{K}^{-1}$ |
| $T$ | 反应温度 | $\text{K}$ |
| $C_i$ | 物种 $i$ 的摩尔浓度 | $\text{mol} \cdot \text{m}^{-3}$ |
| $n_{i,j}$ | 反应 $j$ 中物种 $i$ 的反应级数 | 无量纲 (-) |
| $F_i$ | 物种 $i$ 的摩尔流量 | $\text{mol} \cdot \text{s}^{-1}$ |
| $V$ | 反应体积 (自变量) | $\text{m}^3$ |
| $\nu_{i,j}$ | 化学计量系数 (Stoichiometric coeff) | (-), 反应物为负, 生成物为正 |
| $\dot{v}$ | 体积流量 (Volumetric flow rate) | $\text{m}^3 \cdot \text{s}^{-1}$ |
| $y$ | 拟合目标变量 ($F_{out}, C_{out}, X$) | 取决于选择模式 |
| $w_k$ | 权重系数 | - |
            """
        )

    with st.sidebar:
        st.markdown("### 全局设置")

        with st.container(border=True):
            st.markdown("#### 显示格式")
            table_number_style = st.selectbox(
                "表格数值显示",
                options=["科学计数", "常规小数"],
                index=0,
                help="控制数据预览与结果表格的显示方式（不影响计算）。",
                key="table_number_style",
            )
            table_decimal_places = st.number_input(
                "表格小数位数",
                value=3,
                min_value=0,
                max_value=12,
                step=1,
                help="科学计数：表示尾数的小数位；常规小数：表示小数点后位数。",
                key="table_decimal_places",
            )
            plot_tick_auto = st.checkbox(
                "图轴数字自动（更美观）",
                value=True,
                help="推荐开启。关闭后可强制图轴使用科学计数/常规小数格式。",
                key="plot_tick_auto",
            )
            plot_number_style = st.selectbox(
                "图轴数字格式（关闭自动后生效）",
                options=["科学计数", "常规小数"],
                index=0,
                key="plot_number_style",
            )
            plot_decimal_places = st.number_input(
                "图轴小数位数（关闭自动后生效）",
                value=3,
                min_value=0,
                max_value=12,
                step=1,
                key="plot_decimal_places",
            )

        with st.container(border=True):
            st.markdown("#### ODE 求解器")
            solver_method = st.selectbox(
                "求解方法",
                options=["RK45", "BDF", "Radau"],
                index=0,
                help="若方程刚性明显，推荐使用 BDF 或 Radau。",
            )
            rtol = st.number_input(
                "rtol（相对误差容限）", value=1e-6, min_value=1e-12, format="%.2e"
            )
            atol = st.number_input(
                "atol（绝对误差容限）", value=1e-9, min_value=1e-15, format="%.2e"
            )

    st.subheader("① 反应定义")

    with st.container(border=True):
        st.markdown("#### 物种与反应数")
        col_input1, col_input2 = st.columns([2, 1])
        with col_input1:
            species_text = st.text_input("物种名（逗号分隔，如 A,B,C）", value="A,B,C")
        with col_input2:
            n_reactions = int(st.number_input("反应数", value=1, min_value=1, step=1))

    species_names = _clean_species_names(species_text)
    if len(species_names) < 1:
        st.warning("请至少输入一个物种。")
        st.stop()

    # 反应设置区域
    with st.container(border=True):
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.markdown(
                "**化学计量数矩阵 ν**\n\n"
                "<small>行=物种，列=反应（反应物为负，生成物为正）</small>",
                unsafe_allow_html=True,
            )
            table_number_format = _make_number_format_string(
                table_number_style, int(table_decimal_places)
            )
            nu_default = _build_default_nu_table(species_names, n_reactions)
            nu_column_config = {
                col: st.column_config.NumberColumn(col, format=table_number_format)
                for col in nu_default.columns
            }
            nu_table = st.data_editor(
                nu_default,
                use_container_width=True,
                num_rows="fixed",
                height=200,
                column_config=nu_column_config,
            )
            nu_table_numeric = nu_table.copy()
            for col in nu_table_numeric.columns:
                nu_table_numeric[col] = pd.to_numeric(
                    nu_table_numeric[col], errors="coerce"
                )
            if nu_table_numeric.isna().any().any():
                st.error("化学计量数矩阵 ν 中包含空值/非数值，请修正后再继续。")
                st.stop()
            stoich_matrix = nu_table_numeric.to_numpy(dtype=float)

        with col_right:
            st.markdown(
                "**初值猜测 & 拟合开关**\n\n"
                "<small>勾选 Fit? 列以对特定参数进行拟合。</small>",
                unsafe_allow_html=True,
            )
            param_default = pd.DataFrame(
                {
                    "k0_guess": np.full(n_reactions, f"{1.0e3:.2e}", dtype=object),
                    "Fit_k0": np.full(n_reactions, True, dtype=bool),
                    "Ea_guess_J_mol": np.full(
                        n_reactions, f"{8.0e4:.2e}", dtype=object
                    ),
                    "Fit_Ea": np.full(n_reactions, True, dtype=bool),
                },
                index=[f"R{j+1}" for j in range(n_reactions)],
            )
            # Use column configuration for better UX
            column_config = {
                "Fit_k0": st.column_config.CheckboxColumn("拟合 k0", default=True),
                "Fit_Ea": st.column_config.CheckboxColumn("拟合 Ea", default=True),
                "k0_guess": st.column_config.TextColumn(
                    "k0 初值",
                    help="支持科学计数法输入，如 1e5、2.3e-4；单位取决于反应级数（幂律模型）。",
                ),
                "Ea_guess_J_mol": st.column_config.TextColumn(
                    "Ea 初值 [J/mol]",
                    help="支持科学计数法输入，如 8e4、1.2e5",
                ),
            }

            param_table = st.data_editor(
                param_default,
                use_container_width=True,
                num_rows="fixed",
                height=250,
                column_config=column_config,
            )
            k0_guess = pd.to_numeric(param_table["k0_guess"], errors="coerce").to_numpy(
                dtype=float
            )
            ea_guess_J_mol = pd.to_numeric(
                param_table["Ea_guess_J_mol"], errors="coerce"
            ).to_numpy(dtype=float)

            if not np.all(np.isfinite(k0_guess)):
                st.error("k0_guess 列包含空值/非数值，请检查输入。")
                st.stop()
            if not np.all(np.isfinite(ea_guess_J_mol)):
                st.error("Ea_guess_J_mol 列包含空值/非数值，请检查输入。")
                st.stop()
            if np.any(k0_guess < 0.0):
                st.error("k0_guess 不能为负。")
                st.stop()

            # Extract boolean flags
            fit_k0_flags = param_table["Fit_k0"].to_numpy(dtype=bool)
            fit_ea_flags = param_table["Fit_Ea"].to_numpy(dtype=bool)

    with st.container(border=True):
        st.markdown("#### 反应级数矩阵 n（行=反应，列=物种）")
        order_default = _build_default_order_table(species_names, n_reactions)
        order_column_config = {
            col: st.column_config.NumberColumn(col, format=table_number_format)
            for col in order_default.columns
        }
        order_table = st.data_editor(
            order_default,
            use_container_width=True,
            num_rows="fixed",
            column_config=order_column_config,
        )
        order_table_numeric = order_table.copy()
        for col in order_table_numeric.columns:
            order_table_numeric[col] = pd.to_numeric(
                order_table_numeric[col], errors="coerce"
            )
        if order_table_numeric.isna().any().any():
            st.error("反应级数矩阵 n 中包含空值/非数值，请修正后再继续。")
            st.stop()
        order_guess = order_table_numeric.to_numpy(dtype=float)

        st.markdown("**拟合 n（逐格勾选）**")
        fit_order_default = pd.DataFrame(
            data=np.full((n_reactions, len(species_names)), False, dtype=bool),
            index=[f"R{j+1}" for j in range(n_reactions)],
            columns=species_names,
        )
        fit_order_column_config = {
            name: st.column_config.CheckboxColumn(name, default=False)
            for name in species_names
        }
        fit_order_table = st.data_editor(
            fit_order_default,
            use_container_width=True,
            num_rows="fixed",
            key=f"fit_order_table_{n_reactions}_{len(species_names)}",
            column_config=fit_order_column_config,
        )
        fit_order_flags_matrix = fit_order_table.to_numpy(dtype=bool)

    st.divider()
    st.subheader("② 实验数据")

    with st.container(border=True):
        col_up1, col_up2 = st.columns([1.2, 1])
        with col_up1:
            st.markdown(
                "**数据要求：**\n"
                "- 每行一个实验点（Reactor）\n"
                "- 必须列：`V_m3` (体积), `T_K` (温度), `vdot_m3_s` (体积流量), 入口摩尔流量 (如 `F0_A_mol_s`)\n"
                "- 说明：本 App 以每行数据自己的 `T_K` 与 `vdot_m3_s` 进行计算"
            )

            # 生成模板
            template_measured_mode_options = [
                "Fout (mol/s)",
                "Cout (mol/m^3)",
                "X (conversion)",
                "全部",
            ]
            template_measured_mode_display = {
                "Fout (mol/s)": "Fout：出口摩尔流量 [mol/s]",
                "Cout (mol/m^3)": "Cout：出口浓度 [mol/m³]",
                "X (conversion)": "X：转化率 [-]",
                "全部": "全部（同时生成 Fout/Cout/X）",
            }
            template_measured_mode = st.selectbox(
                "模板中包含的测量列类型",
                options=template_measured_mode_options,
                index=0,
                help="你计划用哪一种测量值做拟合，就在模板里生成相应列；也可以选“全部”。",
                format_func=lambda x: template_measured_mode_display.get(x, x),
            )
            template_columns = ["V_m3", "T_K", "vdot_m3_s"]
            for name in species_names:
                template_columns.append(f"F0_{name}_mol_s")

            if template_measured_mode in ["Fout (mol/s)", "全部"]:
                for name in species_names:
                    template_columns.append(f"Fout_{name}_mol_s")
            if template_measured_mode in ["Cout (mol/m^3)", "全部"]:
                for name in species_names:
                    template_columns.append(f"Cout_{name}_mol_m3")
            if template_measured_mode in ["X (conversion)", "全部"]:
                for name in species_names:
                    template_columns.append(f"X_{name}")

            template_df = pd.DataFrame(columns=template_columns)
            template_csv = template_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 下载 CSV 数据模板",
                data=template_csv,
                file_name="pfr_template.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_up2:
            st.markdown("**上传数据文件**")
            uploaded_file = st.file_uploader(
                "上传 CSV 文件", type=["csv"], label_visibility="collapsed"
            )

    with st.container(border=True):
        st.markdown("#### 目标函数：选择变量与物种")
        st.caption("提示：窗口较窄时两列会自动上下排列，复选框可能出现在下方。")

        col_out1, col_out2 = st.columns(2)
        with col_out1:
            output_mode_display = {
                "Fout (mol/s)": "Fout：出口摩尔流量 [mol/s]",
                "Cout (mol/m^3)": "Cout：出口浓度 [mol/m³]",
                "X (conversion)": "X：转化率 [-]",
            }
            output_mode = st.selectbox(
                "拟合目标变量",
                options=["Fout (mol/s)", "Cout (mol/m^3)", "X (conversion)"],
                index=0,
                format_func=lambda x: output_mode_display.get(x, x),
            )
        with col_out2:
            st.markdown("**选择进入目标函数的物种（复选框）**")

            fit_key_prefix = "fit_species__"
            for i, name in enumerate(species_names):
                key = f"{fit_key_prefix}{name}"
                if key not in st.session_state:
                    st.session_state[key] = i == 0

            col_btn1, col_btn2, col_btn3 = st.columns(3)
            if col_btn1.button("全选", use_container_width=True, key="fit_species_all"):
                for name in species_names:
                    st.session_state[f"{fit_key_prefix}{name}"] = True
            if col_btn2.button(
                "全不选", use_container_width=True, key="fit_species_none"
            ):
                for name in species_names:
                    st.session_state[f"{fit_key_prefix}{name}"] = False
            if col_btn3.button(
                "只选第一个", use_container_width=True, key="fit_species_first_only"
            ):
                for i, name in enumerate(species_names):
                    st.session_state[f"{fit_key_prefix}{name}"] = i == 0

            output_species_list = []
            for name in species_names:
                key = f"{fit_key_prefix}{name}"
                if st.checkbox(name, key=key):
                    output_species_list.append(name)

            st.caption(
                f"已选择 {len(output_species_list)} / {len(species_names)} 个物种进入目标函数。"
            )

    if len(output_species_list) == 0:
        st.error("请至少选择一个物种进行拟合。")
        st.stop()

    if uploaded_file is None:
        st.info("请先下载模板，填入数据后上传。")
        st.stop()

    data_df = pd.read_csv(uploaded_file)
    if data_df.empty:
        st.error("CSV 文件为空。")
        st.stop()

    # 简单的列检查 + 缺失值处理：空单元格按 0 处理（便于快速填表）
    required_cols_hint = ["V_m3", "T_K", "vdot_m3_s"] + [
        f"F0_{n}_mol_s" for n in species_names
    ]
    missing = [c for c in required_cols_hint if c not in data_df.columns]
    if missing:
        st.warning(
            f"注意：CSV 中缺少以下标准列（已按 0 自动补列，可能影响计算）：{missing}"
        )
        for col in missing:
            data_df[col] = 0.0

    # 对常用数值列：强制转为数值，无法解析的填 NaN，再统一用 0 填充
    numeric_cols_to_fill = list(required_cols_hint)
    for name in species_names:
        numeric_cols_to_fill.extend(
            [
                f"Fout_{name}_mol_s",
                f"Cout_{name}_mol_m3",
                f"X_{name}",
            ]
        )
    for col in numeric_cols_to_fill:
        if col not in data_df.columns:
            data_df[col] = 0.0
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

    data_df[numeric_cols_to_fill] = data_df[numeric_cols_to_fill].fillna(0.0)

    st.success(f"成功加载 {len(data_df)} 条实验数据。")

    with st.container(border=True):
        st.markdown("#### 数据预览（前 50 行）")
        st.caption("提示：双击单元格可查看/复制真实数值。")
        preview_df = data_df.head(50).copy()
        st.data_editor(
            preview_df,
            column_config=_build_table_column_config(preview_df, table_number_format),
            num_rows="fixed",
            key="preview_data_editor",
            use_container_width=True,
            height=260,
        )

    # 检查“所选目标物种”的测量列是否存在（允许你把全部物种都填上，只拟合其中一部分）
    if output_mode == "Fout (mol/s)":
        required_measured_cols = [f"Fout_{n}_mol_s" for n in output_species_list]
    elif output_mode == "Cout (mol/m^3)":
        required_measured_cols = [f"Cout_{n}_mol_m3" for n in output_species_list]
    else:
        required_measured_cols = [f"X_{n}" for n in output_species_list]

    missing_measured_cols = [
        c for c in required_measured_cols if c not in data_df.columns
    ]
    if missing_measured_cols:
        st.warning(
            "注意：你选择进入目标函数的物种，在 CSV 中缺少以下测量列："
            f"{missing_measured_cols}。缺少测量值的行会被赋予较大惩罚残差。"
        )

    st.divider()
    st.subheader("③ 参数拟合")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("物种数", len(species_names))
    col_m2.metric("反应数", int(n_reactions))
    col_m3.metric("目标物种数", len(output_species_list))
    col_m4.metric("实验点数", len(data_df))
    st.caption(f"目标物种：{', '.join(output_species_list)}")

    with st.container(border=True):
        st.markdown("#### 参数边界与加权设置")
        col_bounds1, col_bounds2, col_bounds3 = st.columns(3)
        with col_bounds1:
            st.markdown("**k0 范围**")
            k0_min = st.number_input(
                "Min",
                value=1e-15,
                min_value=1e-15,
                max_value=1e15,
                format="%.1e",
                key="k0min",
            )
            k0_max = st.number_input(
                "Max",
                value=1e15,
                min_value=1e-15,
                max_value=1e15,
                format="%.1e",
                key="k0max",
            )
        with col_bounds2:
            st.markdown("**Ea 范围 [J/mol]**")
            ea_min_J_mol = st.number_input(
                "Min",
                value=1.0e4,
                min_value=1.0e4,
                max_value=3.0e5,
                format="%.1e",
                key="eamin",
            )
            ea_max_J_mol = st.number_input(
                "Max",
                value=3.0e5,
                min_value=1.0e4,
                max_value=3.0e5,
                format="%.1e",
                key="eamax",
            )
        with col_bounds3:
            st.markdown("**级数 n 范围**")
            order_min = st.number_input("Min", value=-2.0, format="%.1f", key="nmin")
            order_max = st.number_input("Max", value=5.0, format="%.1f", key="nmax")

        weight_mode = st.selectbox(
            "残差加权策略", options=["不加权", "按测量值相对误差(1/|y|)"], index=0
        )

        with st.expander("高级拟合设置（提高鲁棒性）", expanded=False):
            st.caption(
                "当初始值离真值较远、拟合结果停在初值时，通常是数值灵敏度过低（数值 Jacobian≈0）导致。"
            )
            diff_step_rel = st.number_input(
                "diff_step：有限差分相对步长",
                value=1e-2,
                min_value=1e-6,
                max_value=1e-1,
                format="%.1e",
                help="SciPy 默认步长非常小，k0/Ea 初值不合理时容易“看不到”梯度；建议 1e-2 ~ 1e-3。",
            )
            max_nfev = int(
                st.number_input(
                    "max_nfev：最大函数评估次数",
                    value=2000,
                    min_value=200,
                    max_value=20000,
                    step=200,
                    help="越大越稳健但越慢（每次评估都要做多次 solve_ivp）。",
                )
            )
            use_x_scale_jac = st.checkbox(
                "启用参数缩放 x_scale='jac'",
                value=True,
                help="推荐开启，可改善不同量纲参数（k0、Ea、n）混合拟合时的收敛性。",
            )
            use_multi_start = st.checkbox(
                "多起点拟合（multi-start）",
                value=True,
                help="初值不准时更稳健，但会更慢（会从多个随机起点重复拟合）。",
            )
            n_starts = int(
                st.number_input(
                    "起点数量",
                    value=8,
                    min_value=1,
                    max_value=30,
                    step=1,
                    disabled=not use_multi_start,
                )
            )
            random_seed = int(
                st.number_input(
                    "随机种子",
                    value=0,
                    min_value=0,
                    max_value=999999,
                    step=1,
                    disabled=not use_multi_start,
                )
            )

    # 准备拟合函数...
    def residual_function(parameter_vector: np.ndarray) -> np.ndarray:
        k0, ea_J_mol, reaction_order_matrix = _unpack_parameters(
            parameter_vector=parameter_vector,
            k0_guess=k0_guess,
            ea_guess_J_mol=ea_guess_J_mol,
            order_guess=order_guess,
            fit_k0_flags=fit_k0_flags,
            fit_ea_flags=fit_ea_flags,
            fit_order_flags_matrix=fit_order_flags_matrix,
        )

        residuals = []
        for _, row in data_df.iterrows():
            pred_values, ok, _ = _predict_outputs_for_row(
                row=row,
                species_names=species_names,
                output_mode=output_mode,
                output_species_list=output_species_list,
                stoich_matrix=stoich_matrix,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                solver_method=solver_method,
                rtol=rtol,
                atol=atol,
            )
            if not ok:
                residuals.extend([1e6] * len(output_species_list))
                continue

            meas_values = np.zeros(len(output_species_list), dtype=float)
            has_missing_measurement = False
            for out_i, species in enumerate(output_species_list):
                if output_mode == "Fout (mol/s)":
                    col = f"Fout_{species}_mol_s"
                elif output_mode == "Cout (mol/m^3)":
                    col = f"Cout_{species}_mol_m3"
                else:
                    col = f"X_{species}"

                value = _to_float_or_nan(row.get(col, np.nan))
                if not np.isfinite(value):
                    has_missing_measurement = True
                    break
                meas_values[out_i] = float(value)

            if has_missing_measurement:
                residuals.extend([1e6] * len(output_species_list))
                continue

            # 处理预测值中的 NaN（例如转化率计算时入口流量为零的情况）
            nan_mask = np.isnan(pred_values) | np.isnan(meas_values)
            diff = pred_values - meas_values
            diff[nan_mask] = 0.0  # NaN 不贡献残差
            if weight_mode == "按测量值相对误差(1/|y|)":
                weight = 1.0 / np.maximum(np.abs(meas_values), 1e-12)
                weight[nan_mask] = 0.0  # NaN 对应权重也为零
                diff = diff * weight
            residuals.extend(diff.tolist())

        return np.array(residuals, dtype=float)

    initial_parameter_vector = _pack_parameters(
        k0_guess=k0_guess,
        ea_guess_J_mol=ea_guess_J_mol,
        order_guess=order_guess,
        fit_k0_flags=fit_k0_flags,
        fit_ea_flags=fit_ea_flags,
        fit_order_flags_matrix=fit_order_flags_matrix,
    )

    lower_bound, upper_bound = _build_bounds(
        k0_guess=k0_guess,
        ea_guess_J_mol=ea_guess_J_mol,
        order_guess=order_guess,
        fit_k0_flags=fit_k0_flags,
        fit_ea_flags=fit_ea_flags,
        fit_order_flags_matrix=fit_order_flags_matrix,
        k0_min=max(k0_min, 1e-15),
        k0_max=min(max(k0_max, k0_min * 1.0001), 1e15),
        ea_min_J_mol=max(ea_min_J_mol, 1.0e4),
        ea_max_J_mol=min(max(ea_max_J_mol, ea_min_J_mol + 1.0), 3.0e5),
        order_min=order_min,
        order_max=max(order_max, order_min + 1e-6),
    )

    if initial_parameter_vector.size > 0:
        if not np.all(np.isfinite(initial_parameter_vector)):
            st.error("初值向量包含 NaN/Inf，请检查 k0/Ea/n 的初值输入。")
            st.stop()
        initial_parameter_vector_clipped = np.clip(
            initial_parameter_vector, lower_bound, upper_bound
        )
        if np.any(initial_parameter_vector_clipped != initial_parameter_vector):
            st.warning(
                "检测到初值超出边界，已自动裁剪到边界范围内（避免 least_squares 报错 x0 infeasible）。"
            )
            initial_parameter_vector = initial_parameter_vector_clipped

    if st.button("🚀 开始拟合", type="primary", use_container_width=True):
        if initial_parameter_vector.size == 0:
            st.warning("所有参数均被固定，仅进行模拟。")
            fitted_parameter_vector = initial_parameter_vector.copy()
            opt_success = True
            opt_message = "无优化（参数固定）"
        else:
            with st.spinner("正在拟合... 请耐心等待"):
                try:
                    initial_residuals = residual_function(initial_parameter_vector)
                    initial_cost = 0.5 * float(np.dot(initial_residuals, initial_residuals))

                    x_scale_value = "jac" if use_x_scale_jac else 1.0
                    multi_start_report = None

                    if use_multi_start and (n_starts > 1):
                        rng = np.random.default_rng(random_seed)
                        n_fit_k0 = int(np.sum(fit_k0_flags))

                        start_vectors = [initial_parameter_vector]
                        for _ in range(n_starts - 1):
                            random_x0 = lower_bound + rng.random(
                                size=lower_bound.size
                            ) * (upper_bound - lower_bound)

                            # 对 k0 采用对数均匀采样（跨多个数量级时更合理）
                            if n_fit_k0 > 0:
                                k0_lb = np.maximum(lower_bound[:n_fit_k0], 1e-300)
                                k0_ub = np.maximum(upper_bound[:n_fit_k0], 1e-300)
                                ln_lb = np.log(k0_lb)
                                ln_ub = np.log(k0_ub)
                                u = rng.random(size=n_fit_k0)
                                random_x0[:n_fit_k0] = np.exp(ln_lb + u * (ln_ub - ln_lb))

                            random_x0 = np.clip(random_x0, lower_bound, upper_bound)
                            start_vectors.append(random_x0)

                        max_nfev_coarse = min(200, max_nfev)
                        progress_bar = st.progress(0)

                        best_stage1_result = None
                        best_start_index = 0
                        for idx, x0_try in enumerate(start_vectors):
                            result_try = least_squares(
                                fun=residual_function,
                                x0=x0_try,
                                bounds=(lower_bound, upper_bound),
                                method="trf",
                                x_scale=x_scale_value,
                                diff_step=diff_step_rel,
                                max_nfev=max_nfev_coarse,
                            )
                            if (best_stage1_result is None) or (
                                result_try.cost < best_stage1_result.cost
                            ):
                                best_stage1_result = result_try
                                best_start_index = idx
                            progress_bar.progress(
                                int(100 * (idx + 1) / len(start_vectors))
                            )
                        progress_bar.empty()

                        result = least_squares(
                            fun=residual_function,
                            x0=best_stage1_result.x,
                            bounds=(lower_bound, upper_bound),
                            method="trf",
                            x_scale=x_scale_value,
                            diff_step=diff_step_rel,
                            max_nfev=max_nfev,
                        )
                        multi_start_report = (
                            f"multi-start：n_starts={n_starts}, seed={random_seed}, "
                            f"coarse max_nfev={max_nfev_coarse}, best_start={best_start_index + 1}/{n_starts}"
                        )
                    else:
                        result = least_squares(
                            fun=residual_function,
                            x0=initial_parameter_vector,
                            bounds=(lower_bound, upper_bound),
                            method="trf",
                            x_scale=x_scale_value,
                            diff_step=diff_step_rel,
                            max_nfev=max_nfev,
                        )

                    final_cost = float(result.cost)
                    relative_move = float(
                        np.linalg.norm(result.x - initial_parameter_vector)
                        / max(1.0, np.linalg.norm(initial_parameter_vector))
                    )
                    cost_ratio = final_cost / max(initial_cost, 1e-300)
                except ValueError as exc:
                    st.error(f"least_squares 输入参数错误: {exc}")
                    st.stop()
                except Exception as exc:
                    st.error(f"least_squares 运行异常: {exc}")
                    st.stop()
            fitted_parameter_vector = result.x
            opt_success = result.success
            opt_message = result.message

            if multi_start_report:
                st.info(multi_start_report)
            st.caption(
                f"目标函数 cost（=0.5·Σ残差²）：初始 {initial_cost:.3e} → 拟合 {final_cost:.3e}（比例 {cost_ratio:.3e}）；"
                f"参数相对变化 {relative_move:.3e}"
            )
            if (
                np.isfinite(initial_cost)
                and (initial_cost > 0.0)
                and (relative_move < 1e-6)
                and (cost_ratio > 0.99)
            ):
                st.warning(
                    "拟合几乎没有从初值移动/没有明显下降。建议："
                    "1) 增大 diff_step 或 max_nfev；"
                    "2) 开启多起点拟合；"
                    "3) 若方程刚性明显，尝试将 ODE 求解器切换为 BDF 或 Radau；"
                    "4) 检查 k0/Ea/n 的初值与边界是否合理。"
                )

        k0_fit, ea_fit_J_mol, order_fit = _unpack_parameters(
            parameter_vector=fitted_parameter_vector,
            k0_guess=k0_guess,
            ea_guess_J_mol=ea_guess_J_mol,
            order_guess=order_guess,
            fit_k0_flags=fit_k0_flags,
            fit_ea_flags=fit_ea_flags,
            fit_order_flags_matrix=fit_order_flags_matrix,
        )

        # 结果展示区域
        st.divider()
        st.markdown("### 拟合结果")

        col_res1, col_res2 = st.columns(2)
        col_res1.metric(
            "优化状态",
            "成功" if opt_success else "失败",
            delta=None,
            delta_color="normal",
        )
        col_res2.info(f"求解器信息: {opt_message}")

        st.markdown("#### 预测 vs 实验")
        # Ensure the user can select ANY species for plotting, even if not fitted
        plot_species = st.selectbox(
            "选择绘图物种 (可查看未拟合的物种)", options=species_names, index=0
        )

        measured_list = []
        predicted_list = []
        volume_list = []
        status_list = []
        for _, row in data_df.iterrows():
            pred_values, ok, msg = _predict_outputs_for_row(
                row=row,
                species_names=species_names,
                output_mode=output_mode,
                output_species_list=[plot_species],
                stoich_matrix=stoich_matrix,
                k0=k0_fit,
                ea_J_mol=ea_fit_J_mol,
                reaction_order_matrix=order_fit,
                solver_method=solver_method,
                rtol=rtol,
                atol=atol,
            )

            reactor_volume_m3 = row.get("V_m3", np.nan)
            volume_list.append(
                float(reactor_volume_m3) if np.isfinite(reactor_volume_m3) else np.nan
            )
            status_list.append("OK" if ok else f"FAIL: {msg}")

            if output_mode == "Fout (mol/s)":
                col = f"Fout_{plot_species}_mol_s"
            elif output_mode == "Cout (mol/m^3)":
                col = f"Cout_{plot_species}_mol_m3"
            else:
                col = f"X_{plot_species}"

            meas = row.get(col, np.nan)
            measured_list.append(float(meas) if np.isfinite(meas) else np.nan)
            predicted_list.append(float(pred_values[0]) if ok else np.nan)

        plot_df = (
            pd.DataFrame(
                {
                    "V_m3": volume_list,
                    "measured": measured_list,
                    "predicted": predicted_list,
                    "status": status_list,
                }
            )
            .sort_values("V_m3")
            .reset_index(drop=True)
        )

        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.markdown("##### 奇偶校验图 (Parity Plot)")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(
                plot_df["measured"], plot_df["predicted"], "o", label="Data", alpha=0.6
            )

            finite_mask = np.isfinite(plot_df["measured"]) & np.isfinite(
                plot_df["predicted"]
            )
            if finite_mask.any():
                y_min = min(
                    plot_df.loc[finite_mask, "measured"].min(),
                    plot_df.loc[finite_mask, "predicted"].min(),
                )
                y_max = max(
                    plot_df.loc[finite_mask, "measured"].max(),
                    plot_df.loc[finite_mask, "predicted"].max(),
                )
                # 稍微扩大范围（防止 span 为 0 或极小的情况）
                span = y_max - y_min
                if span < 1e-12:
                    span = max(abs(y_max), abs(y_min), 1.0) * 0.1  # 至少有一点范围
                y_min -= span * 0.05
                y_max += span * 0.05
                ax2.plot([y_min, y_max], [y_min, y_max], "k--", label="y=x", alpha=0.5)
                ax2.set_xlim([y_min, y_max])
                ax2.set_ylim([y_min, y_max])

            ax2.set_xlabel("Measured (实验值)", fontsize=10)
            ax2.set_ylabel("Predicted (预测值)", fontsize=10)
            _apply_plot_tick_format(
                ax2,
                number_style=plot_number_style,
                decimal_places=int(plot_decimal_places),
                use_auto=bool(plot_tick_auto),
            )
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

        with col_plot2:
            st.markdown("##### 误差图 (Predicted - Measured)")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            error_values = plot_df["predicted"] - plot_df["measured"]
            ax3.plot(plot_df["V_m3"], error_values, "o-", label="误差", alpha=0.8)
            ax3.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
            ax3.set_xlabel("Volume $V$ [m$^3$]", fontsize=10)
            ax3.set_ylabel(f"Error ({plot_species}, {output_mode})", fontsize=10)
            _apply_plot_tick_format(
                ax3,
                number_style=plot_number_style,
                decimal_places=int(plot_decimal_places),
                use_auto=bool(plot_tick_auto),
            )
            ax3.grid(True, linestyle=":", alpha=0.6)
            ax3.legend()
            st.pyplot(fig3, clear_figure=True)

        st.markdown("##### 优化后动力学参数")
        col_res_p1, col_res_p2 = st.columns(2)
        with col_res_p1:
            st.markdown("**k0 & Ea**")
            result_param_df = pd.DataFrame(
                {"k0": k0_fit, "Ea_J_mol": ea_fit_J_mol},
                index=[f"R{j+1}" for j in range(n_reactions)],
            )
            st.data_editor(
                result_param_df,
                column_config=_build_table_column_config(
                    result_param_df, table_number_format
                ),
                num_rows="fixed",
                key="result_param_table",
                use_container_width=True,
            )

        with col_res_p2:
            st.markdown("**级数 n**")
            result_order_df = pd.DataFrame(
                data=order_fit,
                index=[f"R{j+1}" for j in range(n_reactions)],
                columns=species_names,
            )
            st.data_editor(
                result_order_df,
                column_config=_build_table_column_config(
                    result_order_df, table_number_format
                ),
                num_rows="fixed",
                key="result_order_table",
                use_container_width=True,
            )

        with st.expander("查看详细预测数据"):
            st.data_editor(
                plot_df,
                column_config=_build_table_column_config(plot_df, table_number_format),
                num_rows="fixed",
                key="plot_detail_table",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
