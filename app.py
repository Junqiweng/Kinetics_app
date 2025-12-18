from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import io
from pathlib import Path


R_GAS_J_MOL_K = 8.314462618  # Gas constant [J/(mol*K)]
CONC_EPS_MOL_M3 = 1e-30  # Concentration floor [mol/m^3] (avoid 0^negative -> inf)


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


def _build_example_batch_csv_bytes() -> bytes:
    """
    生成一个 Batch 示例数据（A -> B 一级反应，幂律 n=1）。
    用于帮助页面下载示例 CSV。
    """
    temperature_K = 350.0  # Temperature [K]
    conc_A0_mol_m3 = 2000.0  # Initial concentration [mol/m^3]
    conc_B0_mol_m3 = 0.0  # Initial concentration [mol/m^3]

    k0_1_s = 1.0e6  # Pre-exponential factor [1/s] (for n=1)
    ea_J_mol = 5.0e4  # Activation energy [J/mol]
    rate_constant_1_s = k0_1_s * np.exp(-ea_J_mol / (R_GAS_J_MOL_K * temperature_K))

    time_s = np.array([0, 20, 40, 60, 90, 120, 180, 240, 360, 480], dtype=float)
    conc_A_t = conc_A0_mol_m3 * np.exp(-rate_constant_1_s * time_s)
    conc_B_t = conc_B0_mol_m3 + (conc_A0_mol_m3 - conc_A_t)
    conversion_A = 1.0 - conc_A_t / max(conc_A0_mol_m3, 1e-30)

    data_df = pd.DataFrame(
        {
            "t_s": time_s,
            "T_K": np.full(time_s.size, temperature_K, dtype=float),
            "C0_A_mol_m3": np.full(time_s.size, conc_A0_mol_m3, dtype=float),
            "C0_B_mol_m3": np.full(time_s.size, conc_B0_mol_m3, dtype=float),
            "Cout_A_mol_m3": conc_A_t,
            "Cout_B_mol_m3": conc_B_t,
            "X_A": conversion_A,
        }
    )
    return data_df.to_csv(index=False).encode("utf-8")


def _read_file_bytes_if_exists(file_path: str) -> bytes | None:
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        return path.read_bytes()
    except Exception:
        return None


def _figure_to_image_bytes(fig: plt.Figure, image_format: str) -> bytes:
    """
    将 Matplotlib Figure 导出为字节流，供 Streamlit download_button 使用。

    image_format: "png" / "svg"
    """
    image_format = str(image_format).lower().strip()
    buf = io.BytesIO()

    save_kwargs = {"format": image_format, "bbox_inches": "tight"}
    if image_format in ["png", "jpg", "jpeg", "tif", "tiff"]:
        save_kwargs["dpi"] = 300

    fig.savefig(buf, **save_kwargs)
    return buf.getvalue()


def _render_help_page() -> None:
    st.title("教程 / 帮助")
    st.caption("面向初学者：按步骤完成一次建模与拟合。")

    tab_quick, tab_csv, tab_models, tab_fit, tab_trouble = st.tabs(
        ["快速上手", "CSV 列说明", "动力学模型", "拟合技巧", "常见问题"]
    )

    with tab_quick:
        st.markdown(
            "**推荐流程（一次完整的拟合）**\n"
            "1) 在左侧选择反应器类型与动力学模型；\n"
            "2) 输入物种名与反应数，填写化学计量数矩阵 $\\nu$；\n"
            "3) 设置级数矩阵 $n$、k0/Ea 初值与拟合开关（Fit）；\n"
            "4) 下载并填写 CSV 模板（或用示例数据）；\n"
            "5) 上传 CSV，选择拟合目标变量与目标物种；\n"
            "6) 点击“开始拟合”，查看 Parity Plot 与误差图；\n"
            "7) 导出拟合参数与对比数据。"
        )

        st.markdown("**示例数据下载（可直接用于上手）**")
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            pfr_example_bytes = _read_file_bytes_if_exists(
                "test_data/test_data_matched.csv"
            )
            if pfr_example_bytes is None:
                st.info(
                    "未找到 `test_data/test_data_matched.csv`，请先运行 `test_data/generate_test_data.py` 生成。"
                )
            else:
                st.download_button(
                    "📥 下载 PFR 示例数据 (CSV)",
                    data=pfr_example_bytes,
                    file_name="pfr_example.csv",
                    mime="text/csv",
                    help="示例：A → B 一级反应，列包含 V_m3/T_K/vdot/F0_*/Fout_*。",
                    use_container_width=True,
                )
        with col_ex2:
            batch_example_bytes = _build_example_batch_csv_bytes()
            st.download_button(
                "📥 下载 Batch 示例数据 (CSV)",
                data=batch_example_bytes,
                file_name="batch_example.csv",
                mime="text/csv",
                help="示例：A → B 一级反应，列包含 t_s/T_K/C0_*/Cout_*/X_A。",
                use_container_width=True,
            )

        st.info(
            "提示：你可以先用示例数据跑通流程，再替换为自己的实验数据。"
            "若拟合不动，可尝试：增大 diff_step、增大 max_nfev、开启 multi-start、或切换 ODE 求解器为 BDF/Radau。"
        )

    with tab_csv:
        st.markdown("**PFR 输入列**")
        st.markdown(
            "- `V_m3`：反应器体积 [m³]\n"
            "- `T_K`：温度 [K]\n"
            "- `vdot_m3_s`：体积流量 [m³/s]\n"
            "- `F0_<物种名>_mol_s`：入口摩尔流量 [mol/s]"
        )
        st.markdown("**Batch 输入列**")
        st.markdown(
            "- `t_s`：反应时间 [s]\n"
            "- `T_K`：温度 [K]\n"
            "- `C0_<物种名>_mol_m3`：初始浓度 [mol/m³]"
        )
        st.markdown("**测量值列（任选一种类型）**")
        st.markdown(
            "- PFR：`Fout_<物种名>_mol_s`（出口摩尔流量）\n"
            "- PFR/Batch：`Cout_<物种名>_mol_m3`（出口浓度）\n"
            "- PFR/Batch：`X_<物种名>`（转化率）"
        )
        st.caption(
            "允许缺测：测量值为空/NaN 时，该行会在拟合中被赋予较大惩罚残差（相当于“提醒你这行缺数据”）。"
        )

    with tab_models:
        st.markdown("**(1) 幂律 (Power-law)**")
        st.latex(r"r_j = k_j(T)\prod_i C_i^{n_{ij}}")
        st.latex(r"k_j(T)=k_{0,j}\exp\left(-\frac{E_{a,j}}{RT}\right)")
        st.caption("k0 的单位取决于总反应级数；这是动力学常见现象。")

        st.markdown("**(2) Langmuir-Hinshelwood（吸附抑制）**")
        st.latex(
            r"r_j=\frac{k_j(T)\prod_i C_i^{n_{ij}}}{\left(1+\sum_i K_i(T)C_i\right)^{m_j}}"
        )
        st.latex(r"K_i(T)=K_{0,i}\exp\left(-\frac{E_{a,K,i}}{RT}\right)")
        st.caption(
            "当 $C$ 用 mol/m³，则 $K$ 的单位为 m³/mol（保证 $K_iC_i$ 无量纲）；"
            "$E_{a,K}$ 允许为负值（放热吸附）。"
        )

        st.markdown("**(3) 可逆反应 (Reversible)**")
        st.latex(
            r"r_j=k_j^+(T)\prod_i C_i^{n_{ij}^+}-k_j^-(T)\prod_i C_i^{n_{ij}^-}"
        )
        st.latex(
            r"k_j^{\pm}(T)=k_{0,j}^{\pm}\exp\left(-\frac{E_{a,j}^{\pm}}{RT}\right)"
        )
        st.caption("正/逆反应有各自的 k0/Ea/n，拟合时可分别勾选 Fit。")

    with tab_fit:
        st.markdown("**推荐的拟合设置（更稳健）**")
        st.markdown(
            "- 初值不准/拟合不动：把 `diff_step` 调大到 `1e-2 ~ 1e-3`；并开启 `multi-start`。\n"
            "- 参数量纲差异大（k0、Ea、n 混合）：保持 `x_scale='jac'` 开启。\n"
            "- ODE 刚性明显：将求解器切换为 `BDF` 或 `Radau`。\n"
            "- 极端参数（如 k0 很大）会导致求解困难：先缩紧边界，再逐步放开。"
        )
        st.markdown("**加权策略**")
        st.markdown(
            "- `不加权`：直接最小二乘。\n"
            "- `按测量值相对误差(1/|y|)`：更重视相对误差，小量级数据更容易被照顾。"
        )

    with tab_trouble:
        st.markdown("**常见报错与处理**")
        st.markdown(
            "- `solve_ivp失败`：尝试 `BDF/Radau`，或调松 `rtol/atol`，或缩紧参数边界。\n"
            "- `T_K 无效` / `vdot 无效`：检查 CSV 对应列是否为正数。\n"
            "- 负级数 + 浓度趋近 0：会导致 $C^n$ 发散；程序对负级数使用浓度下限避免 `inf`，但建议检查模型合理性。\n"
            "- `x0 infeasible`：初值超出边界；程序会自动裁剪到边界内，但仍建议你设置更合理的初值与边界。"
        )


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
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, CONC_EPS_MOL_M3)
            rate_value = rate_value * (conc_value**order_value)
        rate_vector[reaction_index] = rate_value
    return rate_vector


def calc_rate_vector_langmuir_hinshelwood(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    K0_ads: np.ndarray,
    Ea_K_J_mol: np.ndarray,
    m_inhibition: np.ndarray,
) -> np.ndarray:
    """
    Langmuir-Hinshelwood 动力学速率计算。

    r_j = k_j(T) * Π_i C_i^(n_ij) / (1 + Σ_i K_i(T) * C_i)^m_j

    其中吸附常数也遵循 Arrhenius 方程：
    K_i(T) = K_{0,i} * exp(-Ea_{K,i} / RT)

    参数:
        conc_mol_m3: 浓度向量 (n_species,) [mol/m³]
        temperature_K: 温度 [K]
        k0: 指前因子 (n_reactions,)
        ea_J_mol: 活化能 (n_reactions,) [J/mol]
        reaction_order_matrix: 反应级数矩阵 (n_reactions, n_species)
        K0_ads: 吸附常数指前因子 (n_species,)
        Ea_K_J_mol: 吸附热 (n_species,) [J/mol]，可为负值（放热吸附）
        m_inhibition: 抑制指数 (n_reactions,)

    返回:
        rate_vector: 反应速率向量 (n_reactions,)
    """
    conc_mol_m3 = _safe_nonnegative(conc_mol_m3)
    k_T = k0 * np.exp(-ea_J_mol / (R_GAS_J_MOL_K * temperature_K))

    # 计算温度依赖的吸附常数 K_i(T)
    K_ads_T = K0_ads * np.exp(-Ea_K_J_mol / (R_GAS_J_MOL_K * temperature_K))

    # 计算分母：(1 + Σ_i K_i(T) * C_i)
    denominator_base = 1.0 + np.sum(K_ads_T * conc_mol_m3)

    n_reactions = k0.size
    rate_vector = np.zeros(n_reactions, dtype=float)
    for reaction_index in range(n_reactions):
        # 分子：k_j(T) * Π_i C_i^(n_ij)
        rate_numerator = k_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = reaction_order_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, CONC_EPS_MOL_M3)
            rate_numerator = rate_numerator * (conc_value**order_value)

        # 分母：(1 + Σ_i K_i(T) * C_i)^m_j
        m_j = m_inhibition[reaction_index]
        denominator = denominator_base**m_j if m_j != 0.0 else 1.0

        rate_vector[reaction_index] = rate_numerator / max(denominator, 1e-30)

    return rate_vector


def calc_rate_vector_reversible(
    conc_mol_m3: np.ndarray,
    temperature_K: float,
    k0_fwd: np.ndarray,
    ea_fwd_J_mol: np.ndarray,
    order_fwd_matrix: np.ndarray,
    k0_rev: np.ndarray,
    ea_rev_J_mol: np.ndarray,
    order_rev_matrix: np.ndarray,
) -> np.ndarray:
    """
    可逆反应动力学速率计算。

    r_j = r_j^+ - r_j^-
        = k_j^+(T) * Π_i C_i^(n_ij^+) - k_j^-(T) * Π_i C_i^(n_ij^-)

    参数:
        conc_mol_m3: 浓度向量 (n_species,) [mol/m³]
        temperature_K: 温度 [K]
        k0_fwd, ea_fwd_J_mol: 正反应指前因子和活化能
        order_fwd_matrix: 正反应级数矩阵 (n_reactions, n_species)
        k0_rev, ea_rev_J_mol: 逆反应指前因子和活化能
        order_rev_matrix: 逆反应级数矩阵 (n_reactions, n_species)

    返回:
        rate_vector: 净反应速率向量 (n_reactions,)
    """
    conc_mol_m3 = _safe_nonnegative(conc_mol_m3)
    k_fwd_T = k0_fwd * np.exp(-ea_fwd_J_mol / (R_GAS_J_MOL_K * temperature_K))
    k_rev_T = k0_rev * np.exp(-ea_rev_J_mol / (R_GAS_J_MOL_K * temperature_K))

    n_reactions = k0_fwd.size
    rate_vector = np.zeros(n_reactions, dtype=float)

    for reaction_index in range(n_reactions):
        # 正反应速率
        rate_fwd = k_fwd_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = order_fwd_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, CONC_EPS_MOL_M3)
            rate_fwd = rate_fwd * (conc_value**order_value)

        # 逆反应速率
        rate_rev = k_rev_T[reaction_index]
        for species_index in range(conc_mol_m3.size):
            order_value = order_rev_matrix[reaction_index, species_index]
            if order_value == 0.0:
                continue
            conc_value = float(conc_mol_m3[species_index])
            if order_value < 0.0:
                conc_value = max(conc_value, CONC_EPS_MOL_M3)
            rate_rev = rate_rev * (conc_value**order_value)

        # 净反应速率
        rate_vector[reaction_index] = rate_fwd - rate_rev

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
    kinetic_model: str = "power_law",
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
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

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_mol_m3.size),
                Ea_K_J_mol=(
                    Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_mol_m3.size)
                ),
                m_inhibition=(
                    m_inhibition if m_inhibition is not None else np.ones(k0.size)
                ),
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0_fwd=k0,
                ea_fwd_J_mol=ea_J_mol,
                order_fwd_matrix=reaction_order_matrix,
                k0_rev=k0_rev if k0_rev is not None else np.zeros(k0.size),
                ea_rev_J_mol=(
                    ea_rev_J_mol if ea_rev_J_mol is not None else np.zeros(k0.size)
                ),
                order_rev_matrix=(
                    order_rev_matrix
                    if order_rev_matrix is not None
                    else np.zeros_like(reaction_order_matrix)
                ),
            )
        else:
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


def integrate_batch_reactor(
    reaction_time_s: float,
    temperature_K: float,
    conc_initial_mol_m3: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    kinetic_model: str = "power_law",
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, bool, str]:
    """
    Batch Reactor 设计方程（恒温，恒容）：
      dC_i/dt = Σ_j nu_{i,j} r_j

    参数:
        reaction_time_s: 反应时间 [s]
        temperature_K: 反应温度 [K]
        conc_initial_mol_m3: 初始浓度向量 [mol/m³]
        stoich_matrix: 化学计量矩阵 (n_species x n_reactions)
        k0, ea_J_mol, reaction_order_matrix: 动力学参数
        solver_method, rtol, atol: ODE 求解器设置
        kinetic_model: 动力学模型类型

    返回:
        conc_final: 最终浓度 [mol/m³]
        success: 求解是否成功
        message: 状态信息
    """
    if not np.isfinite(reaction_time_s):
        return conc_initial_mol_m3.copy(), False, "t_s 无效（NaN/Inf）"
    if reaction_time_s < 0.0:
        return conc_initial_mol_m3.copy(), False, "t_s 不能为负"
    if reaction_time_s == 0.0:
        return conc_initial_mol_m3.copy(), True, "t=0"

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return conc_initial_mol_m3.copy(), False, "温度 T_K 无效"

    if not np.all(np.isfinite(conc_initial_mol_m3)):
        return conc_initial_mol_m3.copy(), False, "初始浓度包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return conc_initial_mol_m3.copy(), False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return conc_initial_mol_m3.copy(), False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return conc_initial_mol_m3.copy(), False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return conc_initial_mol_m3.copy(), False, "反应级数矩阵 n 包含 NaN/Inf"

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        conc_safe = _safe_nonnegative(conc_mol_m3)

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_safe.size),
                Ea_K_J_mol=(
                    Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_safe.size)
                ),
                m_inhibition=(
                    m_inhibition if m_inhibition is not None else np.ones(k0.size)
                ),
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0_fwd=k0,
                ea_fwd_J_mol=ea_J_mol,
                order_fwd_matrix=reaction_order_matrix,
                k0_rev=k0_rev if k0_rev is not None else np.zeros(k0.size),
                ea_rev_J_mol=(
                    ea_rev_J_mol if ea_rev_J_mol is not None else np.zeros(k0.size)
                ),
                order_rev_matrix=(
                    order_rev_matrix
                    if order_rev_matrix is not None
                    else np.zeros_like(reaction_order_matrix)
                ),
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )

        dC_dt = stoich_matrix @ rate_vector
        return dC_dt

    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reaction_time_s)),
            y0=conc_initial_mol_m3.astype(float),
            method=solver_method,
            rtol=rtol,
            atol=atol,
        )
    except Exception as exc:
        return conc_initial_mol_m3.copy(), False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return conc_initial_mol_m3.copy(), False, str(message)

    conc_final = solution.y[:, -1]
    return conc_final, True, "OK"


def integrate_pfr_profile(
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
    n_points: int = 200,
    kinetic_model: str = "power_law",
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    返回 PFR 沿程剖面：
      volume_grid_m3: shape (n_points,)
      molar_flow_profile_mol_s: shape (n_species, n_points)
    """
    n_points = int(n_points)
    if n_points < 2:
        n_points = 2

    if not np.isfinite(reactor_volume_m3):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "V_m3 无效（NaN/Inf）"
    if reactor_volume_m3 < 0.0:
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "V_m3 不能为负"
    if reactor_volume_m3 == 0.0:
        return (
            np.array([0.0], dtype=float),
            molar_flow_inlet_mol_s.astype(float)[:, None],
            True,
            "V=0",
        )

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "温度 T_K 无效"
    if (not np.isfinite(vdot_m3_s)) or (vdot_m3_s <= 0.0):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "体积流量 vdot_m3_s 无效"

    if not np.all(np.isfinite(molar_flow_inlet_mol_s)):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "入口摩尔流量包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return np.array([0.0]), molar_flow_inlet_mol_s[:, None], False, "反应级数矩阵 n 包含 NaN/Inf"

    def ode_fun(volume_m3: float, molar_flow_mol_s: np.ndarray) -> np.ndarray:
        conc_mol_m3 = _safe_nonnegative(molar_flow_mol_s) / max(vdot_m3_s, 1e-30)

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_mol_m3.size),
                Ea_K_J_mol=(
                    Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_mol_m3.size)
                ),
                m_inhibition=(m_inhibition if m_inhibition is not None else np.ones(k0.size)),
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0_fwd=k0,
                ea_fwd_J_mol=ea_J_mol,
                order_fwd_matrix=reaction_order_matrix,
                k0_rev=k0_rev if k0_rev is not None else np.zeros(k0.size),
                ea_rev_J_mol=(ea_rev_J_mol if ea_rev_J_mol is not None else np.zeros(k0.size)),
                order_rev_matrix=(
                    order_rev_matrix
                    if order_rev_matrix is not None
                    else np.zeros_like(reaction_order_matrix)
                ),
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_mol_m3,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )

        dF_dV = stoich_matrix @ rate_vector
        return dF_dV

    volume_grid_m3 = np.linspace(0.0, float(reactor_volume_m3), n_points, dtype=float)

    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reactor_volume_m3)),
            y0=molar_flow_inlet_mol_s.astype(float),
            method=solver_method,
            t_eval=volume_grid_m3,
            rtol=rtol,
            atol=atol,
        )
    except Exception as exc:
        return volume_grid_m3, molar_flow_inlet_mol_s.astype(float)[:, None], False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return volume_grid_m3, molar_flow_inlet_mol_s.astype(float)[:, None], False, str(message)

    return solution.t.astype(float), solution.y.astype(float), True, "OK"


def integrate_batch_profile(
    reaction_time_s: float,
    temperature_K: float,
    conc_initial_mol_m3: np.ndarray,
    stoich_matrix: np.ndarray,
    k0: np.ndarray,
    ea_J_mol: np.ndarray,
    reaction_order_matrix: np.ndarray,
    solver_method: str,
    rtol: float,
    atol: float,
    n_points: int = 200,
    kinetic_model: str = "power_law",
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    返回 Batch 随时间剖面：
      time_grid_s: shape (n_points,)
      conc_profile_mol_m3: shape (n_species, n_points)
    """
    n_points = int(n_points)
    if n_points < 2:
        n_points = 2

    if not np.isfinite(reaction_time_s):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "t_s 无效（NaN/Inf）"
    if reaction_time_s < 0.0:
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "t_s 不能为负"
    if reaction_time_s == 0.0:
        return (
            np.array([0.0], dtype=float),
            conc_initial_mol_m3.astype(float)[:, None],
            True,
            "t=0",
        )

    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "温度 T_K 无效"

    if not np.all(np.isfinite(conc_initial_mol_m3)):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "初始浓度包含 NaN/Inf"
    if not np.all(np.isfinite(stoich_matrix)):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "化学计量数矩阵 ν 包含 NaN/Inf"
    if not np.all(np.isfinite(k0)):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "k0 包含 NaN/Inf"
    if not np.all(np.isfinite(ea_J_mol)):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "Ea 包含 NaN/Inf"
    if not np.all(np.isfinite(reaction_order_matrix)):
        return np.array([0.0]), conc_initial_mol_m3[:, None], False, "反应级数矩阵 n 包含 NaN/Inf"

    def ode_fun(time_s: float, conc_mol_m3: np.ndarray) -> np.ndarray:
        conc_safe = _safe_nonnegative(conc_mol_m3)

        if kinetic_model == "power_law":
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )
        elif kinetic_model == "langmuir_hinshelwood":
            rate_vector = calc_rate_vector_langmuir_hinshelwood(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
                K0_ads=K0_ads if K0_ads is not None else np.zeros(conc_safe.size),
                Ea_K_J_mol=(Ea_K_J_mol if Ea_K_J_mol is not None else np.zeros(conc_safe.size)),
                m_inhibition=(m_inhibition if m_inhibition is not None else np.ones(k0.size)),
            )
        elif kinetic_model == "reversible":
            rate_vector = calc_rate_vector_reversible(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0_fwd=k0,
                ea_fwd_J_mol=ea_J_mol,
                order_fwd_matrix=reaction_order_matrix,
                k0_rev=k0_rev if k0_rev is not None else np.zeros(k0.size),
                ea_rev_J_mol=(ea_rev_J_mol if ea_rev_J_mol is not None else np.zeros(k0.size)),
                order_rev_matrix=(
                    order_rev_matrix
                    if order_rev_matrix is not None
                    else np.zeros_like(reaction_order_matrix)
                ),
            )
        else:
            rate_vector = calc_rate_vector_power_law(
                conc_mol_m3=conc_safe,
                temperature_K=temperature_K,
                k0=k0,
                ea_J_mol=ea_J_mol,
                reaction_order_matrix=reaction_order_matrix,
            )

        dC_dt = stoich_matrix @ rate_vector
        return dC_dt

    time_grid_s = np.linspace(0.0, float(reaction_time_s), n_points, dtype=float)
    try:
        solution = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, float(reaction_time_s)),
            y0=conc_initial_mol_m3.astype(float),
            method=solver_method,
            t_eval=time_grid_s,
            rtol=rtol,
            atol=atol,
        )
    except Exception as exc:
        return time_grid_s, conc_initial_mol_m3.astype(float)[:, None], False, f"solve_ivp异常: {exc}"

    if not solution.success:
        message = solution.message if hasattr(solution, "message") else "solve_ivp失败"
        return time_grid_s, conc_initial_mol_m3.astype(float)[:, None], False, str(message)

    return solution.t.astype(float), solution.y.astype(float), True, "OK"


def _pack_parameters(
    k0_guess: np.ndarray,
    ea_guess_J_mol: np.ndarray,
    order_guess: np.ndarray,
    fit_k0_flags: np.ndarray,
    fit_ea_flags: np.ndarray,
    fit_order_flags_matrix: np.ndarray,
    # L-H 参数
    K0_ads_guess: np.ndarray = None,
    Ea_K_guess: np.ndarray = None,
    m_inhibition_guess: np.ndarray = None,
    fit_K0_ads_flags: np.ndarray = None,
    fit_Ea_K_flags: np.ndarray = None,
    fit_m_flags: np.ndarray = None,
    # 可逆反应参数
    k0_rev_guess: np.ndarray = None,
    ea_rev_guess: np.ndarray = None,
    order_rev_guess: np.ndarray = None,
    fit_k0_rev_flags: np.ndarray = None,
    fit_ea_rev_flags: np.ndarray = None,
    fit_order_rev_flags_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    将所有需要拟合的参数打包成一个向量。
    顺序：k0 -> Ea -> n -> K0_ads -> Ea_K -> m -> k0_rev -> Ea_rev -> n_rev
    """
    parts = []

    # 1. k0 (正反应)
    if np.any(fit_k0_flags):
        parts.append(k0_guess[fit_k0_flags])

    # 2. Ea (正反应)
    if np.any(fit_ea_flags):
        parts.append(ea_guess_J_mol[fit_ea_flags])

    # 3. Orders (正反应)
    order_mask_flat = np.asarray(fit_order_flags_matrix, dtype=bool).ravel()
    if np.any(order_mask_flat):
        parts.append(order_guess.ravel()[order_mask_flat])

    # 4. K0_ads (L-H)
    if fit_K0_ads_flags is not None and np.any(fit_K0_ads_flags):
        parts.append(K0_ads_guess[fit_K0_ads_flags])

    # 5. Ea_K (L-H)
    if fit_Ea_K_flags is not None and np.any(fit_Ea_K_flags):
        parts.append(Ea_K_guess[fit_Ea_K_flags])

    # 6. m_inhibition (L-H)
    if fit_m_flags is not None and np.any(fit_m_flags):
        parts.append(m_inhibition_guess[fit_m_flags])

    # 7. k0_rev (可逆)
    if fit_k0_rev_flags is not None and np.any(fit_k0_rev_flags):
        parts.append(k0_rev_guess[fit_k0_rev_flags])

    # 8. Ea_rev (可逆)
    if fit_ea_rev_flags is not None and np.any(fit_ea_rev_flags):
        parts.append(ea_rev_guess[fit_ea_rev_flags])

    # 9. Orders_rev (可逆)
    if fit_order_rev_flags_matrix is not None:
        order_rev_mask_flat = np.asarray(fit_order_rev_flags_matrix, dtype=bool).ravel()
        if np.any(order_rev_mask_flat):
            parts.append(order_rev_guess.ravel()[order_rev_mask_flat])

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
    # L-H 参数
    K0_ads_guess: np.ndarray = None,
    Ea_K_guess: np.ndarray = None,
    m_inhibition_guess: np.ndarray = None,
    fit_K0_ads_flags: np.ndarray = None,
    fit_Ea_K_flags: np.ndarray = None,
    fit_m_flags: np.ndarray = None,
    # 可逆反应参数
    k0_rev_guess: np.ndarray = None,
    ea_rev_guess: np.ndarray = None,
    order_rev_guess: np.ndarray = None,
    fit_k0_rev_flags: np.ndarray = None,
    fit_ea_rev_flags: np.ndarray = None,
    fit_order_rev_flags_matrix: np.ndarray = None,
) -> dict:
    """
    从参数向量中解包所有参数。
    返回包含所有参数的字典。
    """
    index = 0

    # 初始化为guesses
    k0 = k0_guess.copy().astype(float)
    ea_J_mol = ea_guess_J_mol.copy().astype(float)
    reaction_order_matrix = order_guess.copy().astype(float)
    K0_ads = K0_ads_guess.copy().astype(float) if K0_ads_guess is not None else None
    Ea_K = Ea_K_guess.copy().astype(float) if Ea_K_guess is not None else None
    m_inhibition = (
        m_inhibition_guess.copy().astype(float)
        if m_inhibition_guess is not None
        else None
    )
    k0_rev = k0_rev_guess.copy().astype(float) if k0_rev_guess is not None else None
    ea_rev = ea_rev_guess.copy().astype(float) if ea_rev_guess is not None else None
    order_rev = (
        order_rev_guess.copy().astype(float) if order_rev_guess is not None else None
    )

    # 1. k0
    n_fit_k0 = int(np.sum(fit_k0_flags))
    if n_fit_k0 > 0:
        k0[fit_k0_flags] = parameter_vector[index : index + n_fit_k0]
        index += n_fit_k0

    # 2. Ea
    n_fit_ea = int(np.sum(fit_ea_flags))
    if n_fit_ea > 0:
        ea_J_mol[fit_ea_flags] = parameter_vector[index : index + n_fit_ea]
        index += n_fit_ea

    # 3. Orders
    n_reactions, n_species = reaction_order_matrix.shape
    order_mask_flat = np.asarray(fit_order_flags_matrix, dtype=bool).ravel()
    n_fit_orders = int(np.sum(order_mask_flat))
    if n_fit_orders > 0:
        flat_orders = reaction_order_matrix.ravel()
        flat_orders[order_mask_flat] = parameter_vector[index : index + n_fit_orders]
        reaction_order_matrix = flat_orders.reshape((n_reactions, n_species))
        index += n_fit_orders

    # 4. K0_ads
    if fit_K0_ads_flags is not None:
        n_fit_K0_ads = int(np.sum(fit_K0_ads_flags))
        if n_fit_K0_ads > 0:
            K0_ads[fit_K0_ads_flags] = parameter_vector[index : index + n_fit_K0_ads]
            index += n_fit_K0_ads

    # 5. Ea_K
    if fit_Ea_K_flags is not None:
        n_fit_Ea_K = int(np.sum(fit_Ea_K_flags))
        if n_fit_Ea_K > 0:
            Ea_K[fit_Ea_K_flags] = parameter_vector[index : index + n_fit_Ea_K]
            index += n_fit_Ea_K

    # 6. m_inhibition
    if fit_m_flags is not None:
        n_fit_m = int(np.sum(fit_m_flags))
        if n_fit_m > 0:
            m_inhibition[fit_m_flags] = parameter_vector[index : index + n_fit_m]
            index += n_fit_m

    # 7. k0_rev
    if fit_k0_rev_flags is not None:
        n_fit_k0_rev = int(np.sum(fit_k0_rev_flags))
        if n_fit_k0_rev > 0:
            k0_rev[fit_k0_rev_flags] = parameter_vector[index : index + n_fit_k0_rev]
            index += n_fit_k0_rev

    # 8. Ea_rev
    if fit_ea_rev_flags is not None:
        n_fit_ea_rev = int(np.sum(fit_ea_rev_flags))
        if n_fit_ea_rev > 0:
            ea_rev[fit_ea_rev_flags] = parameter_vector[index : index + n_fit_ea_rev]
            index += n_fit_ea_rev

    # 9. order_rev
    if fit_order_rev_flags_matrix is not None and order_rev is not None:
        order_rev_mask_flat = np.asarray(fit_order_rev_flags_matrix, dtype=bool).ravel()
        n_fit_order_rev = int(np.sum(order_rev_mask_flat))
        if n_fit_order_rev > 0:
            flat_order_rev = order_rev.ravel()
            flat_order_rev[order_rev_mask_flat] = parameter_vector[
                index : index + n_fit_order_rev
            ]
            order_rev = flat_order_rev.reshape(order_rev_guess.shape)
            index += n_fit_order_rev

    return {
        "k0": k0,
        "ea_J_mol": ea_J_mol,
        "reaction_order_matrix": reaction_order_matrix,
        "K0_ads": K0_ads,
        "Ea_K": Ea_K,
        "m_inhibition": m_inhibition,
        "k0_rev": k0_rev,
        "ea_rev": ea_rev,
        "order_rev": order_rev,
    }


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
    # L-H 边界参数
    fit_K0_ads_flags: np.ndarray = None,
    fit_Ea_K_flags: np.ndarray = None,
    fit_m_flags: np.ndarray = None,
    K0_ads_min: float = 1e-10,
    K0_ads_max: float = 1e10,
    Ea_K_min: float = -2e5,  # 允许负值（放热吸附）
    Ea_K_max: float = 2e5,
    m_min: float = 0.0,
    m_max: float = 5.0,
    # 可逆反应边界参数
    fit_k0_rev_flags: np.ndarray = None,
    fit_ea_rev_flags: np.ndarray = None,
    fit_order_rev_flags_matrix: np.ndarray = None,
    k0_rev_min: float = 1e-10,
    k0_rev_max: float = 1e15,
    ea_rev_min: float = 0.0,
    ea_rev_max: float = 5e5,
    order_rev_min: float = -3.0,
    order_rev_max: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建所有拟合参数的边界。
    顺序与_pack_parameters一致。
    """
    lower_parts = []
    upper_parts = []

    # 1. k0
    n_fit_k0 = int(np.sum(fit_k0_flags))
    if n_fit_k0 > 0:
        lower_parts.append(np.full(n_fit_k0, k0_min, dtype=float))
        upper_parts.append(np.full(n_fit_k0, k0_max, dtype=float))

    # 2. Ea
    n_fit_ea = int(np.sum(fit_ea_flags))
    if n_fit_ea > 0:
        lower_parts.append(np.full(n_fit_ea, ea_min_J_mol, dtype=float))
        upper_parts.append(np.full(n_fit_ea, ea_max_J_mol, dtype=float))

    # 3. Orders
    n_fit_orders_total = int(np.sum(np.asarray(fit_order_flags_matrix, dtype=bool)))
    if n_fit_orders_total > 0:
        lower_parts.append(np.full(n_fit_orders_total, order_min, dtype=float))
        upper_parts.append(np.full(n_fit_orders_total, order_max, dtype=float))

    # 4. K0_ads
    if fit_K0_ads_flags is not None:
        n_fit_K0_ads = int(np.sum(fit_K0_ads_flags))
        if n_fit_K0_ads > 0:
            lower_parts.append(np.full(n_fit_K0_ads, K0_ads_min, dtype=float))
            upper_parts.append(np.full(n_fit_K0_ads, K0_ads_max, dtype=float))

    # 5. Ea_K
    if fit_Ea_K_flags is not None:
        n_fit_Ea_K = int(np.sum(fit_Ea_K_flags))
        if n_fit_Ea_K > 0:
            lower_parts.append(np.full(n_fit_Ea_K, Ea_K_min, dtype=float))
            upper_parts.append(np.full(n_fit_Ea_K, Ea_K_max, dtype=float))

    # 6. m_inhibition
    if fit_m_flags is not None:
        n_fit_m = int(np.sum(fit_m_flags))
        if n_fit_m > 0:
            lower_parts.append(np.full(n_fit_m, m_min, dtype=float))
            upper_parts.append(np.full(n_fit_m, m_max, dtype=float))

    # 7. k0_rev
    if fit_k0_rev_flags is not None:
        n_fit_k0_rev = int(np.sum(fit_k0_rev_flags))
        if n_fit_k0_rev > 0:
            lower_parts.append(np.full(n_fit_k0_rev, k0_rev_min, dtype=float))
            upper_parts.append(np.full(n_fit_k0_rev, k0_rev_max, dtype=float))

    # 8. Ea_rev
    if fit_ea_rev_flags is not None:
        n_fit_ea_rev = int(np.sum(fit_ea_rev_flags))
        if n_fit_ea_rev > 0:
            lower_parts.append(np.full(n_fit_ea_rev, ea_rev_min, dtype=float))
            upper_parts.append(np.full(n_fit_ea_rev, ea_rev_max, dtype=float))

    # 9. order_rev
    if fit_order_rev_flags_matrix is not None:
        n_fit_order_rev = int(
            np.sum(np.asarray(fit_order_rev_flags_matrix, dtype=bool))
        )
        if n_fit_order_rev > 0:
            lower_parts.append(np.full(n_fit_order_rev, order_rev_min, dtype=float))
            upper_parts.append(np.full(n_fit_order_rev, order_rev_max, dtype=float))

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
    reactor_type: str = "PFR",
    kinetic_model: str = "power_law",
    K0_ads: np.ndarray = None,
    Ea_K_J_mol: np.ndarray = None,
    m_inhibition: np.ndarray = None,
    k0_rev: np.ndarray = None,
    ea_rev_J_mol: np.ndarray = None,
    order_rev_matrix: np.ndarray = None,
) -> tuple[np.ndarray, bool, str]:
    """
    根据反应器类型和动力学模型预测输出值。
    """
    temperature_K = _to_float_or_nan(row.get("T_K", np.nan))
    if (not np.isfinite(temperature_K)) or (temperature_K <= 0.0):
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            "温度 T_K 无效（请检查 CSV 的 T_K 列）",
        )

    name_to_index = {name: i for i, name in enumerate(species_names)}

    if reactor_type == "PFR":
        # PFR 需要 V_m3, vdot_m3_s, F0_*
        reactor_volume_m3 = _to_float_or_nan(row.get("V_m3", np.nan))
        if not np.isfinite(reactor_volume_m3):
            return np.zeros(len(output_species_list), dtype=float), False, "缺少 V_m3"
        if reactor_volume_m3 < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "V_m3 不能为负",
            )

        vdot_m3_s = _to_float_or_nan(row.get("vdot_m3_s", np.nan))
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
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    f"缺少 {col}",
                )
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
            kinetic_model=kinetic_model,
            K0_ads=K0_ads,
            Ea_K_J_mol=Ea_K_J_mol,
            m_inhibition=m_inhibition,
            k0_rev=k0_rev,
            ea_rev_J_mol=ea_rev_J_mol,
            order_rev_matrix=order_rev_matrix,
        )
        if not ok:
            return np.zeros(len(output_species_list), dtype=float), False, message

        # 计算输出值
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

    elif reactor_type == "Batch":
        # Batch 需要 t_s, C0_*
        reaction_time_s = _to_float_or_nan(row.get("t_s", np.nan))
        if not np.isfinite(reaction_time_s):
            return np.zeros(len(output_species_list), dtype=float), False, "缺少 t_s"
        if reaction_time_s < 0.0:
            return (
                np.zeros(len(output_species_list), dtype=float),
                False,
                "t_s 不能为负",
            )

        conc_initial = np.zeros(len(species_names), dtype=float)
        for i, name in enumerate(species_names):
            col = f"C0_{name}_mol_m3"
            value = _to_float_or_nan(row.get(col, np.nan))
            if not np.isfinite(value):
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    f"缺少 {col}",
                )
            if value < 0.0:
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    f"{col} 不能为负",
                )
            conc_initial[i] = float(value)

        conc_final, ok, message = integrate_batch_reactor(
            reaction_time_s=reaction_time_s,
            temperature_K=temperature_K,
            conc_initial_mol_m3=conc_initial,
            stoich_matrix=stoich_matrix,
            k0=k0,
            ea_J_mol=ea_J_mol,
            reaction_order_matrix=reaction_order_matrix,
            solver_method=solver_method,
            rtol=rtol,
            atol=atol,
            kinetic_model=kinetic_model,
            K0_ads=K0_ads,
            Ea_K_J_mol=Ea_K_J_mol,
            m_inhibition=m_inhibition,
            k0_rev=k0_rev,
            ea_rev_J_mol=ea_rev_J_mol,
            order_rev_matrix=order_rev_matrix,
        )
        if not ok:
            return np.zeros(len(output_species_list), dtype=float), False, message

        # 计算输出值
        output_values = np.zeros(len(output_species_list), dtype=float)
        for out_i, species in enumerate(output_species_list):
            idx = name_to_index[species]
            if output_mode == "Cout (mol/m^3)":
                output_values[out_i] = conc_final[idx]
            elif output_mode == "X (conversion)":
                c0 = conc_initial[idx]
                c_final = conc_final[idx]
                if c0 < 1e-30:
                    output_values[out_i] = np.nan
                else:
                    output_values[out_i] = (c0 - c_final) / c0
            else:
                # Batch 不支持 Fout 模式
                return (
                    np.zeros(len(output_species_list), dtype=float),
                    False,
                    "Batch 反应器不支持 Fout 输出模式，请选择 Cout 或 X",
                )

    else:
        return (
            np.zeros(len(output_species_list), dtype=float),
            False,
            f"未知反应器类型: {reactor_type}",
        )

    return output_values, True, "OK"


def main() -> None:
    st.set_page_config(
        page_title="Kinetics_app | 反应动力学拟合",
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

    with st.sidebar:
        with st.container(border=True):
            st.markdown("#### 导航")
            page_mode = st.radio(
                "页面",
                options=["建模与拟合", "教程/帮助"],
                index=0,
                help="新手建议先看“教程/帮助”，下载示例数据跑通一次流程。",
                key="page_mode",
            )

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
            st.markdown("#### 反应器类型")
            reactor_type = st.selectbox(
                "选择反应器模型",
                options=["PFR", "Batch"],
                index=0,
                format_func=lambda x: {
                    "PFR": "PFR (平推流反应器)",
                    "Batch": "Batch (间歇式反应器)",
                }.get(x, x),
                help="PFR: 积分变量为反应器体积 V；Batch: 积分变量为反应时间 t",
            )
            if reactor_type == "Batch":
                st.info("Batch 反应器不支持 Fout 输出模式，请选择 Cout 或 X。")

        with st.container(border=True):
            st.markdown("#### 动力学模型")
            kinetic_model = st.selectbox(
                "选择动力学模型",
                options=["power_law", "langmuir_hinshelwood", "reversible"],
                index=0,
                format_func=lambda x: {
                    "power_law": "幂律 (Power Law)",
                    "langmuir_hinshelwood": "Langmuir-Hinshelwood",
                    "reversible": "可逆反应 (Reversible)",
                }.get(x, x),
                help="幂律: r = k·∏Cⁿ；LH: 带吸附抑制项；可逆: 正-逆反应",
            )
            if kinetic_model == "langmuir_hinshelwood":
                st.caption("L-H 模型: r = k·∏Cⁿ / (1 + ΣK·C)ᵐ")
            elif kinetic_model == "reversible":
                st.caption("可逆反应: r = k⁺·∏Cⁿ⁺ - k⁻·∏Cⁿ⁻")

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

    if page_mode == "教程/帮助":
        _render_help_page()
        st.stop()

    # ========== 动态主标题 ==========
    if reactor_type == "PFR":
        st.title("⚗️ PFR 反应动力学参数拟合")
        st.caption("PFR 数值积分（solve_ivp）+ 最小二乘拟合（least_squares）")
    else:  # Batch
        st.title("⚗️ Batch 反应动力学参数拟合")
        st.caption("Batch 反应器数值积分（solve_ivp）+ 最小二乘拟合（least_squares）")

    with st.container(border=True):
        st.markdown(
            "**快速流程：**\n"
            "1) 在 **① 反应定义** 中输入物种/反应与初值；\n"
            "2) 在 **② 实验数据** 中上传 CSV，并勾选进入目标函数的物种；\n"
            "3) 在 **③ 参数拟合** 中设置边界并开始拟合。"
        )

    # ========== 动态理论模型说明 ==========
    with st.expander("📖 查看详细理论模型与符号说明", expanded=False):
        st.markdown("#### 1. 反应动力学模型 (Reaction Kinetics)")

        # Arrhenius 方程（所有模型通用）
        st.markdown("所有模型均采用 **Arrhenius 方程** 描述速率常数的温度依赖性：")
        st.latex(r"k_j(T) = k_{0,j} \exp\left(-\frac{E_{a,j}}{R T}\right)")

        # 根据动力学模型显示不同的速率方程
        if kinetic_model == "power_law":
            st.markdown("当前模型：**幂函数定律 (Power Law)**")
            st.latex(r"r_j = k_j(T) \prod_{i} C_i^{n_{i,j}}")

        elif kinetic_model == "langmuir_hinshelwood":
            st.markdown("当前模型：**Langmuir-Hinshelwood**（考虑吸附抑制）")
            st.latex(
                r"r_j = \frac{k_j(T) \prod_{i} C_i^{n_{i,j}}}{\left(1 + \sum_{i} K_i C_i\right)^{m_j}}"
            )
            st.caption(
                "其中 $K_i$ 为物种 $i$ 的吸附常数，$m_j$ 为反应 $j$ 的抑制指数。"
            )

        elif kinetic_model == "reversible":
            st.markdown("当前模型：**可逆反应 (Reversible)**")
            st.latex(
                r"r_j = r_j^{+} - r_j^{-} = k_j^{+}(T) \prod_{i} C_i^{n_{i,j}^{+}} - k_j^{-}(T) \prod_{i} C_i^{n_{i,j}^{-}}"
            )
            st.caption(
                "正反应（$+$）和逆反应（$-$）各有独立的指前因子、活化能和反应级数。"
            )

        st.markdown("#### 2. 反应器设计方程 (Reactor Model)")

        # 根据反应器类型显示不同的设计方程
        if reactor_type == "PFR":
            st.markdown(
                "当前反应器：**平推流反应器 (PFR)**，稳态、恒定体积流量（液相）"
            )
            st.latex(r"\frac{dF_i}{dV} = \sum_{j=1}^{N_{rxn}} \nu_{i,j} r_j")
            st.markdown("其中浓度 $C_i$ 与摩尔流量 $F_i$ 的关系为：")
            st.latex(r"C_i = \frac{F_i}{\dot{v}}")
        else:  # Batch
            st.markdown("当前反应器：**间歇式反应器 (Batch)**，恒温、恒容")
            st.latex(r"\frac{dC_i}{dt} = \sum_{j=1}^{N_{rxn}} \nu_{i,j} r_j")

        st.markdown("#### 3. 参数拟合目标 (Optimization Objective)")
        st.markdown("通过调整参数 $\\theta$ 最小化加权残差平方和：")
        st.latex(
            r"\min_{\theta} \sum_{k} \left[ w_k \cdot \left( y_{\text{pred}, k}(\theta) - y_{\text{meas}, k} \right) \right]^2"
        )

        st.markdown("#### 4. 符号说明 (Nomenclature)")

        # 基础符号表
        nomenclature_base = r"""
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
| $\nu_{i,j}$ | 化学计量系数 (Stoichiometric coeff) | (-), 反应物为负, 生成物为正 |
| $y$ | 拟合目标变量 ($F_{out}, C_{out}, X$) | 取决于选择模式 |
| $w_k$ | 权重系数 | - |
"""
        st.markdown(nomenclature_base)

        # 根据反应器类型添加特定符号
        if reactor_type == "PFR":
            st.markdown("**PFR 专用符号：**")
            st.markdown(
                r"""
| 符号 (Symbol) | 含义 (Description) | 标准单位 (SI Unit) |
| :--- | :--- | :--- |
| $F_i$ | 物种 $i$ 的摩尔流量 | $\text{mol} \cdot \text{s}^{-1}$ |
| $V$ | 反应器体积 (自变量) | $\text{m}^3$ |
| $\dot{v}$ | 体积流量 | $\text{m}^3 \cdot \text{s}^{-1}$ |
"""
            )
        else:  # Batch
            st.markdown("**Batch 专用符号：**")
            st.markdown(
                r"""
| 符号 (Symbol) | 含义 (Description) | 标准单位 (SI Unit) |
| :--- | :--- | :--- |
| $t$ | 反应时间 (自变量) | $\text{s}$ |
"""
            )

        # 根据动力学模型添加特定符号
        if kinetic_model == "langmuir_hinshelwood":
            st.markdown("**Langmuir-Hinshelwood 专用符号：**")
            st.markdown(
                r"""
| 符号 (Symbol) | 含义 (Description) | 标准单位 (SI Unit) |
| :--- | :--- | :--- |
| $K_i$ | 物种 $i$ 的吸附常数 | $\text{m}^3 \cdot \text{mol}^{-1}$ |
| $m_j$ | 反应 $j$ 的抑制指数 | 无量纲 (-) |
"""
            )
        elif kinetic_model == "reversible":
            st.markdown("**可逆反应专用符号：**")
            st.markdown(
                r"""
| 符号 (Symbol) | 含义 (Description) | 标准单位 (SI Unit) |
| :--- | :--- | :--- |
| $k_{0,j}^{+}, k_{0,j}^{-}$ | 正/逆反应指前因子 | 取决于反应级数 |
| $E_{a,j}^{+}, E_{a,j}^{-}$ | 正/逆反应活化能 | $\text{J} \cdot \text{mol}^{-1}$ |
| $n_{i,j}^{+}, n_{i,j}^{-}$ | 正/逆反应中物种 $i$ 的反应级数 | 无量纲 (-) |
"""
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
        st.markdown("#### 反应级数矩阵 n（行=反应）")
        st.caption("每个物种的级数初值后紧跟拟合勾选框")

        # 构建合并的表格：n_物种, Fit_物种, n_物种, Fit_物种 ...
        order_combined_data = {}
        for name in species_names:
            order_combined_data[f"n_{name}"] = np.full(
                n_reactions, 1.0 if name == species_names[0] else 0.0, dtype=float
            )
            order_combined_data[f"Fit_{name}"] = np.full(n_reactions, False, dtype=bool)

        order_combined_default = pd.DataFrame(
            order_combined_data,
            index=[f"R{j+1}" for j in range(n_reactions)],
        )

        order_combined_column_config = {}
        for name in species_names:
            order_combined_column_config[f"n_{name}"] = st.column_config.NumberColumn(
                f"n_{name}", format=table_number_format
            )
            order_combined_column_config[f"Fit_{name}"] = (
                st.column_config.CheckboxColumn(f"拟合 {name}", default=False)
            )

        order_combined_table = st.data_editor(
            order_combined_default,
            use_container_width=True,
            num_rows="fixed",
            key=f"order_combined_table_{n_reactions}_{len(species_names)}",
            column_config=order_combined_column_config,
        )

        # 提取级数初值和拟合标志
        order_guess = np.zeros((n_reactions, len(species_names)), dtype=float)
        fit_order_flags_matrix = np.full(
            (n_reactions, len(species_names)), False, dtype=bool
        )
        for i, name in enumerate(species_names):
            order_col = order_combined_table[f"n_{name}"].to_numpy(dtype=float)
            fit_col = order_combined_table[f"Fit_{name}"].to_numpy(dtype=bool)
            order_guess[:, i] = order_col
            fit_order_flags_matrix[:, i] = fit_col

        if not np.all(np.isfinite(order_guess)):
            st.error("反应级数矩阵 n 中包含空值/非数值，请修正后再继续。")
            st.stop()

    # ========== Langmuir-Hinshelwood 专用参数 ==========
    if kinetic_model == "langmuir_hinshelwood":
        with st.container(border=True):
            st.markdown("#### Langmuir-Hinshelwood 参数")
            st.caption(
                "$r_j = k_j(T) \\cdot \\prod C_i^{n_{ij}} / (1 + \\sum K_i(T) C_i)^{m_j}$，"
                "其中 $K_i(T) = K_{0,i} \\exp(-E_{a,K,i}/RT)$"
            )

            st.markdown("**吸附常数参数 K (对每个物种)**")
            K_ads_default = pd.DataFrame(
                {
                    "K0_ads": np.full(len(species_names), f"{1.0:.2e}", dtype=object),
                    "Fit_K0": np.full(len(species_names), False, dtype=bool),
                    "Ea_K_J_mol": np.full(
                        len(species_names), f"{-2.0e4:.2e}", dtype=object
                    ),
                    "Fit_Ea_K": np.full(len(species_names), False, dtype=bool),
                },
                index=species_names,
            )
            K_ads_column_config = {
                "K0_ads": st.column_config.TextColumn(
                    "K0 初值",
                    help="吸附常数指前因子，支持科学计数法",
                ),
                "Fit_K0": st.column_config.CheckboxColumn("拟合 K0", default=False),
                "Ea_K_J_mol": st.column_config.TextColumn(
                    "Ea_K [J/mol]",
                    help="吸附热（可为负值，放热吸附）",
                ),
                "Fit_Ea_K": st.column_config.CheckboxColumn("拟合 Ea_K", default=False),
            }
            K_ads_table = st.data_editor(
                K_ads_default,
                use_container_width=True,
                num_rows="fixed",
                key="K_ads_table",
                column_config=K_ads_column_config,
            )
            K0_ads = pd.to_numeric(K_ads_table["K0_ads"], errors="coerce").to_numpy(
                dtype=float
            )
            Ea_K_J_mol = pd.to_numeric(
                K_ads_table["Ea_K_J_mol"], errors="coerce"
            ).to_numpy(dtype=float)
            fit_K0_ads_flags = K_ads_table["Fit_K0"].to_numpy(dtype=bool)
            fit_Ea_K_flags = K_ads_table["Fit_Ea_K"].to_numpy(dtype=bool)

            if not np.all(np.isfinite(K0_ads)):
                st.error("K0_ads 列包含空值/非数值，请检查输入。")
                st.stop()
            if not np.all(np.isfinite(Ea_K_J_mol)):
                st.error("Ea_K 列包含空值/非数值，请检查输入。")
                st.stop()
            if np.any(K0_ads < 0):
                st.error("吸附常数 K0 不能为负。")
                st.stop()

            st.markdown("**抑制指数 m (对每个反应)**")
            m_inhibition_default = pd.DataFrame(
                {
                    "m": np.full(n_reactions, 1.0, dtype=float),
                    "Fit_m": np.full(n_reactions, False, dtype=bool),
                },
                index=[f"R{j+1}" for j in range(n_reactions)],
            )
            m_inhibition_column_config = {
                "m": st.column_config.NumberColumn("m 初值", format="%.2f"),
                "Fit_m": st.column_config.CheckboxColumn("拟合 m", default=False),
            }
            m_inhibition_table = st.data_editor(
                m_inhibition_default,
                use_container_width=True,
                num_rows="fixed",
                key="m_inhibition_table",
                column_config=m_inhibition_column_config,
            )
            m_inhibition = m_inhibition_table["m"].to_numpy(dtype=float)
            fit_m_flags = m_inhibition_table["Fit_m"].to_numpy(dtype=bool)
    else:
        # 默认值（不使用）
        K0_ads = np.zeros(len(species_names), dtype=float)
        Ea_K_J_mol = np.zeros(len(species_names), dtype=float)
        m_inhibition = np.ones(n_reactions, dtype=float)
        fit_K0_ads_flags = np.full(len(species_names), False, dtype=bool)
        fit_Ea_K_flags = np.full(len(species_names), False, dtype=bool)
        fit_m_flags = np.full(n_reactions, False, dtype=bool)

    # ========== 可逆反应专用参数 ==========
    if kinetic_model == "reversible":
        with st.container(border=True):
            st.markdown("#### 可逆反应参数（逆反应）")
            st.caption(
                "$r_j = k_j^+(T) \\cdot \\prod C^{n^+} - k_j^-(T) \\cdot \\prod C^{n^-}$"
            )

            st.markdown("**逆反应动力学参数**")
            rev_param_default = pd.DataFrame(
                {
                    "k0_rev": np.full(n_reactions, f"{1.0e2:.2e}", dtype=object),
                    "Fit_k0_rev": np.full(n_reactions, False, dtype=bool),
                    "Ea_rev_J_mol": np.full(n_reactions, f"{9.0e4:.2e}", dtype=object),
                    "Fit_Ea_rev": np.full(n_reactions, False, dtype=bool),
                },
                index=[f"R{j+1}" for j in range(n_reactions)],
            )
            rev_param_column_config = {
                "k0_rev": st.column_config.TextColumn(
                    "k0⁻ 初值",
                    help="逆反应指前因子，支持科学计数法",
                ),
                "Fit_k0_rev": st.column_config.CheckboxColumn(
                    "拟合 k0⁻", default=False
                ),
                "Ea_rev_J_mol": st.column_config.TextColumn(
                    "Ea⁻ [J/mol]",
                    help="逆反应活化能",
                ),
                "Fit_Ea_rev": st.column_config.CheckboxColumn(
                    "拟合 Ea⁻", default=False
                ),
            }
            rev_param_table = st.data_editor(
                rev_param_default,
                use_container_width=True,
                num_rows="fixed",
                key="rev_param_table",
                column_config=rev_param_column_config,
            )
            k0_rev = pd.to_numeric(rev_param_table["k0_rev"], errors="coerce").to_numpy(
                dtype=float
            )
            ea_rev_J_mol = pd.to_numeric(
                rev_param_table["Ea_rev_J_mol"], errors="coerce"
            ).to_numpy(dtype=float)
            fit_k0_rev_flags = rev_param_table["Fit_k0_rev"].to_numpy(dtype=bool)
            fit_ea_rev_flags = rev_param_table["Fit_Ea_rev"].to_numpy(dtype=bool)

            if not np.all(np.isfinite(k0_rev)) or not np.all(np.isfinite(ea_rev_J_mol)):
                st.error("逆反应参数包含空值/非数值。")
                st.stop()
            if np.any(k0_rev < 0):
                st.error("逆反应 k0 不能为负。")
                st.stop()

            st.markdown("**逆反应级数矩阵 n⁻（行=反应）**")
            st.caption("每个物种的级数初值后紧跟拟合勾选框")

            # 构建合并的表格：n_物种, Fit_物种, n_物种, Fit_物种 ...
            order_rev_combined_data = {}
            for name in species_names:
                order_rev_combined_data[f"n⁻_{name}"] = np.zeros(
                    n_reactions, dtype=float
                )
                order_rev_combined_data[f"Fit_{name}"] = np.full(
                    n_reactions, False, dtype=bool
                )

            order_rev_combined_default = pd.DataFrame(
                order_rev_combined_data,
                index=[f"R{j+1}" for j in range(n_reactions)],
            )

            order_rev_combined_column_config = {}
            for name in species_names:
                order_rev_combined_column_config[f"n⁻_{name}"] = (
                    st.column_config.NumberColumn(f"n⁻_{name}", format="%.2f")
                )
                order_rev_combined_column_config[f"Fit_{name}"] = (
                    st.column_config.CheckboxColumn(f"拟合 {name}", default=False)
                )

            order_rev_combined_table = st.data_editor(
                order_rev_combined_default,
                use_container_width=True,
                num_rows="fixed",
                key="order_rev_combined_table",
                column_config=order_rev_combined_column_config,
            )

            # 提取逆反应级数初值和拟合标志
            order_rev = np.zeros((n_reactions, len(species_names)), dtype=float)
            fit_order_rev_flags_matrix = np.full(
                (n_reactions, len(species_names)), False, dtype=bool
            )
            for i, name in enumerate(species_names):
                order_rev[:, i] = order_rev_combined_table[f"n⁻_{name}"].to_numpy(
                    dtype=float
                )
                fit_order_rev_flags_matrix[:, i] = order_rev_combined_table[
                    f"Fit_{name}"
                ].to_numpy(dtype=bool)
    else:
        # 默认值（不使用）
        k0_rev = np.zeros(n_reactions, dtype=float)
        ea_rev_J_mol = np.zeros(n_reactions, dtype=float)
        order_rev = np.zeros((n_reactions, len(species_names)), dtype=float)
        fit_k0_rev_flags = np.full(n_reactions, False, dtype=bool)
        fit_ea_rev_flags = np.full(n_reactions, False, dtype=bool)
        fit_order_rev_flags_matrix = np.full(
            (n_reactions, len(species_names)), False, dtype=bool
        )

    st.divider()
    st.subheader("② 实验数据")

    with st.container(border=True):
        col_up1, col_up2 = st.columns([1.2, 1])
        with col_up1:
            # 根据反应器类型显示不同的数据要求
            if reactor_type == "PFR":
                st.markdown(
                    "**数据要求（PFR）：**\n"
                    "- 每行一个实验点\n"
                    "- **必填列**：`V_m3`, `T_K`, `vdot_m3_s`, 入口摩尔流量 `F0_物种_mol_s`\n"
                    "- **选填列**（取决于拟合目标）：`Fout_物种_mol_s`, `Cout_物种_mol_m3`, `X_物种`"
                )
            else:
                st.markdown(
                    "**数据要求（Batch）：**\n"
                    "- 每行一个实验点\n"
                    "- **必填列**：`t_s`, `T_K`, 初始浓度 `C0_物种_mol_m3`\n"
                    "- **选填列**（取决于拟合目标）：`Cout_物种_mol_m3`, `X_物种`"
                )

            # 生成模板（根据反应器类型）
            if reactor_type == "PFR":
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
            else:
                # Batch 不支持 Fout
                template_measured_mode_options = [
                    "Cout (mol/m^3)",
                    "X (conversion)",
                    "全部",
                ]
                template_measured_mode_display = {
                    "Cout (mol/m^3)": "Cout：出口浓度 [mol/m³]",
                    "X (conversion)": "X：转化率 [-]",
                    "全部": "全部（同时生成 Cout/X）",
                }

            template_measured_mode = st.selectbox(
                "模板中包含的测量列类型",
                options=template_measured_mode_options,
                index=0,
                help="你计划用哪一种测量值做拟合，就在模板里生成相应列；也可以选「全部」。",
                format_func=lambda x: template_measured_mode_display.get(x, x),
            )

            # 根据反应器类型生成模板列
            if reactor_type == "PFR":
                template_columns = ["V_m3", "T_K", "vdot_m3_s"]
                for name in species_names:
                    template_columns.append(f"F0_{name}_mol_s")
            else:
                template_columns = ["t_s", "T_K"]
                for name in species_names:
                    template_columns.append(f"C0_{name}_mol_m3")

            if (
                template_measured_mode in ["Fout (mol/s)", "全部"]
                and reactor_type == "PFR"
            ):
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
            template_filename = (
                "pfr_template.csv" if reactor_type == "PFR" else "batch_template.csv"
            )
            st.download_button(
                "📥 下载 CSV 数据模板",
                data=template_csv,
                file_name=template_filename,
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
            # 根据反应器类型过滤可用的输出模式
            if reactor_type == "PFR":
                output_mode_options = [
                    "Fout (mol/s)",
                    "Cout (mol/m^3)",
                    "X (conversion)",
                ]
            else:  # Batch
                output_mode_options = ["Cout (mol/m^3)", "X (conversion)"]

            output_mode = st.selectbox(
                "拟合目标变量",
                options=output_mode_options,
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
    # 根据反应器类型检查不同的必需列
    if reactor_type == "PFR":
        required_cols_hint = ["V_m3", "T_K", "vdot_m3_s"] + [
            f"F0_{n}_mol_s" for n in species_names
        ]
    else:  # Batch
        required_cols_hint = ["t_s", "T_K"] + [f"C0_{n}_mol_m3" for n in species_names]

    missing = [c for c in required_cols_hint if c not in data_df.columns]
    if missing:
        st.warning(
            f"注意：CSV 中缺少以下标准列（已按 0 自动补列，可能影响计算）：{missing}"
        )
        for col in missing:
            data_df[col] = 0.0

    # 对必需输入列：强制转为数值，无法解析的填 NaN，再统一用 0 填充（便于快速填表）
    input_numeric_cols = list(required_cols_hint)
    for col in input_numeric_cols:
        if col not in data_df.columns:
            data_df[col] = 0.0
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce").fillna(0.0)

    # 对“测量值列”：只转换存在的列，不自动补列；NaN 保留表示“缺测”
    # 这样 residual_function 才能识别缺测并进行惩罚（而不是被 0.0 误当成有效测量值）。
    measured_cols_to_parse = []
    for name in species_names:
        if reactor_type == "PFR":
            measured_cols_to_parse.append(f"Fout_{name}_mol_s")
        measured_cols_to_parse.extend([f"Cout_{name}_mol_m3", f"X_{name}"])

    for col in measured_cols_to_parse:
        if col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

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
        st.caption(
            "边界只对 **勾选参与拟合（Fit）** 的参数生效；不同动力学模型启用的参数不同。"
        )
        if kinetic_model == "langmuir_hinshelwood":
            st.caption(
                "Langmuir-Hinshelwood 分母项："
                "$$r_j = \\frac{k_j(T)\\prod_i C_i^{n_{ij}}}{\\left(1+\\sum_i K_i(T)C_i\\right)^{m_j}}$$  "
                "$$K_i(T)=K_{0,i}\\exp\\left(-\\frac{E_{a,K,i}}{RT}\\right)$$  "
                "若 $C$ 用 mol/m³，则 $K$ 的单位为 m³/mol（保证 $K_iC_i$ 无量纲）。"
            )
        elif kinetic_model == "reversible":
            st.caption(
                "可逆反应模型："
                "$$r_j=k_j^+(T)\\prod_i C_i^{n_{ij}^+}-k_j^-(T)\\prod_i C_i^{n_{ij}^-}$$  "
                "本区的 k0/Ea/n 对应正反应（+）；逆反应（-）有独立边界。"
            )
        col_bounds1, col_bounds2, col_bounds3 = st.columns(3)
        with col_bounds1:
            st.markdown(
                "**k0 范围**" if kinetic_model != "reversible" else "**k0⁺ 范围（正反应）**"
            )
            k0_min = st.number_input(
                "Min",
                value=1e-15,
                min_value=1e-15,
                max_value=1e15,
                format="%.1e",
                key="k0min",
                help="仅对勾选 Fit_k0（或 Fit_k0_rev）生效；k0 的单位取决于总级数（常见现象）。",
            )
            k0_max = st.number_input(
                "Max",
                value=1e15,
                min_value=1e-15,
                max_value=1e15,
                format="%.1e",
                key="k0max",
                help="建议先给出合理范围（不要过宽），再逐步放开。",
            )
        with col_bounds2:
            st.markdown(
                "**Ea 范围 [J/mol]**"
                if kinetic_model != "reversible"
                else "**Ea⁺ 范围 [J/mol]（正反应）**"
            )
            ea_min_J_mol = st.number_input(
                "Min",
                value=1.0e4,
                min_value=1.0e4,
                max_value=3.0e5,
                format="%.1e",
                key="eamin",
                help="活化能下界 [J/mol]。仅对勾选 Fit_Ea（或 Fit_Ea_rev）生效。",
            )
            ea_max_J_mol = st.number_input(
                "Max",
                value=3.0e5,
                min_value=1.0e4,
                max_value=3.0e5,
                format="%.1e",
                key="eamax",
                help="活化能上界 [J/mol]。",
            )
        with col_bounds3:
            st.markdown(
                "**级数 n 范围**"
                if kinetic_model != "reversible"
                else "**级数 n⁺ 范围（正反应）**"
            )
            order_min = st.number_input("Min", value=-2.0, format="%.1f", key="nmin")
            order_max = st.number_input("Max", value=5.0, format="%.1f", key="nmax")

        if kinetic_model == "langmuir_hinshelwood":
            st.markdown("**Langmuir-Hinshelwood 专用边界**")
            col_lhb1, col_lhb2, col_lhb3 = st.columns(3)
            with col_lhb1:
                st.markdown("K0_ads 范围 [m³/mol]")
                K0_ads_min = st.number_input(
                    "Min",
                    value=0.0,
                    min_value=0.0,
                    max_value=1e20,
                    format="%.1e",
                    key="K0_ads_min",
                    help="K0_ads 为吸附常数指前因子。若 C 用 mol/m³，则 K 的单位为 m³/mol。",
                )
                K0_ads_max = st.number_input(
                    "Max",
                    value=1e10,
                    min_value=0.0,
                    max_value=1e20,
                    format="%.1e",
                    key="K0_ads_max",
                    help="建议先给出合理范围（例如 1e-6 ~ 1e6），再根据拟合情况调整。",
                )
            with col_lhb2:
                st.markdown("Ea_K 范围 [J/mol]")
                Ea_K_min = st.number_input(
                    "Min",
                    value=-2.0e5,
                    min_value=-1.0e6,
                    max_value=1.0e6,
                    format="%.1e",
                    key="Ea_K_min",
                    help="吸附热（表观）下界 [J/mol]。允许负值（放热吸附）。",
                )
                Ea_K_max = st.number_input(
                    "Max",
                    value=2.0e5,
                    min_value=-1.0e6,
                    max_value=1.0e6,
                    format="%.1e",
                    key="Ea_K_max",
                    help="吸附热（表观）上界 [J/mol]。",
                )
            with col_lhb3:
                st.markdown("m 范围 [-]")
                m_min = st.number_input(
                    "Min",
                    value=0.0,
                    min_value=0.0,
                    max_value=50.0,
                    format="%.2f",
                    key="m_min",
                    help="抑制指数 m（分母整体幂次）下界，通常取 0~5。",
                )
                m_max = st.number_input(
                    "Max",
                    value=5.0,
                    min_value=0.0,
                    max_value=50.0,
                    format="%.2f",
                    key="m_max",
                    help="抑制指数 m 上界。",
                )
        else:
            K0_ads_min = 0.0
            K0_ads_max = 1.0
            Ea_K_min = -2.0e5
            Ea_K_max = 2.0e5
            m_min = 0.0
            m_max = 5.0

        if kinetic_model == "reversible":
            st.markdown("**可逆反应（逆反应）专用边界**")
            col_rvb1, col_rvb2, col_rvb3 = st.columns(3)
            with col_rvb1:
                st.markdown("k0⁻ 范围")
                k0_rev_min = st.number_input(
                    "Min",
                    value=float(k0_min),
                    min_value=1e-15,
                    max_value=1e15,
                    format="%.1e",
                    key="k0_rev_min",
                    help="逆反应 k0⁻ 的下界（仅对勾选 Fit_k0_rev 的反应生效）。",
                )
                k0_rev_max = st.number_input(
                    "Max",
                    value=float(k0_max),
                    min_value=1e-15,
                    max_value=1e15,
                    format="%.1e",
                    key="k0_rev_max",
                    help="逆反应 k0⁻ 的上界。",
                )
            with col_rvb2:
                st.markdown("Ea⁻ 范围 [J/mol]")
                ea_rev_min_J_mol = st.number_input(
                    "Min",
                    value=float(ea_min_J_mol),
                    min_value=0.0,
                    max_value=1.0e6,
                    format="%.1e",
                    key="ea_rev_min",
                    help="逆反应 Ea⁻ 下界 [J/mol]（仅对勾选 Fit_Ea_rev 的反应生效）。",
                )
                ea_rev_max_J_mol = st.number_input(
                    "Max",
                    value=float(ea_max_J_mol),
                    min_value=0.0,
                    max_value=1.0e6,
                    format="%.1e",
                    key="ea_rev_max",
                    help="逆反应 Ea⁻ 上界 [J/mol]。",
                )
            with col_rvb3:
                st.markdown("n⁻ 范围")
                order_rev_min = st.number_input(
                    "Min",
                    value=float(order_min),
                    format="%.1f",
                    key="n_rev_min",
                    help="逆反应级数 n⁻ 下界（仅对勾选逆反应级数 Fit 的项生效）。",
                )
                order_rev_max = st.number_input(
                    "Max",
                    value=float(order_max),
                    format="%.1f",
                    key="n_rev_max",
                    help="逆反应级数 n⁻ 上界。",
                )
        else:
            k0_rev_min = float(k0_min)
            k0_rev_max = float(k0_max)
            ea_rev_min_J_mol = float(ea_min_J_mol)
            ea_rev_max_J_mol = float(ea_max_J_mol)
            order_rev_min = float(order_min)
            order_rev_max = float(order_max)

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
        params = _unpack_parameters(
            parameter_vector=parameter_vector,
            k0_guess=k0_guess,
            ea_guess_J_mol=ea_guess_J_mol,
            order_guess=order_guess,
            fit_k0_flags=fit_k0_flags,
            fit_ea_flags=fit_ea_flags,
            fit_order_flags_matrix=fit_order_flags_matrix,
            # L-H 参数
            K0_ads_guess=K0_ads,
            Ea_K_guess=Ea_K_J_mol,
            m_inhibition_guess=m_inhibition,
            fit_K0_ads_flags=fit_K0_ads_flags,
            fit_Ea_K_flags=fit_Ea_K_flags,
            fit_m_flags=fit_m_flags,
            # 可逆反应参数
            k0_rev_guess=k0_rev,
            ea_rev_guess=ea_rev_J_mol,
            order_rev_guess=order_rev,
            fit_k0_rev_flags=fit_k0_rev_flags,
            fit_ea_rev_flags=fit_ea_rev_flags,
            fit_order_rev_flags_matrix=fit_order_rev_flags_matrix,
        )

        residuals = []
        for _, row in data_df.iterrows():
            pred_values, ok, _ = _predict_outputs_for_row(
                row=row,
                species_names=species_names,
                output_mode=output_mode,
                output_species_list=output_species_list,
                stoich_matrix=stoich_matrix,
                k0=params["k0"],
                ea_J_mol=params["ea_J_mol"],
                reaction_order_matrix=params["reaction_order_matrix"],
                solver_method=solver_method,
                rtol=rtol,
                atol=atol,
                reactor_type=reactor_type,
                kinetic_model=kinetic_model,
                K0_ads=params["K0_ads"] if params["K0_ads"] is not None else K0_ads,
                Ea_K_J_mol=params["Ea_K"] if params["Ea_K"] is not None else Ea_K_J_mol,
                m_inhibition=(
                    params["m_inhibition"]
                    if params["m_inhibition"] is not None
                    else m_inhibition
                ),
                k0_rev=params["k0_rev"] if params["k0_rev"] is not None else k0_rev,
                ea_rev_J_mol=(
                    params["ea_rev"] if params["ea_rev"] is not None else ea_rev_J_mol
                ),
                order_rev_matrix=(
                    params["order_rev"]
                    if params["order_rev"] is not None
                    else order_rev
                ),
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
        # L-H 参数
        K0_ads_guess=K0_ads,
        Ea_K_guess=Ea_K_J_mol,
        m_inhibition_guess=m_inhibition,
        fit_K0_ads_flags=fit_K0_ads_flags,
        fit_Ea_K_flags=fit_Ea_K_flags,
        fit_m_flags=fit_m_flags,
        # 可逆反应参数
        k0_rev_guess=k0_rev,
        ea_rev_guess=ea_rev_J_mol,
        order_rev_guess=order_rev,
        fit_k0_rev_flags=fit_k0_rev_flags,
        fit_ea_rev_flags=fit_ea_rev_flags,
        fit_order_rev_flags_matrix=fit_order_rev_flags_matrix,
    )

    # 将边界做最小清洗：保证 upper > lower（避免 least_squares 报 x0 infeasible / bounds 错误）
    K0_ads_min_bound = max(float(K0_ads_min), 0.0)
    K0_ads_max_bound = max(float(K0_ads_max), K0_ads_min_bound + 1e-15)
    Ea_K_min_bound = float(min(Ea_K_min, Ea_K_max))
    Ea_K_max_bound = float(max(Ea_K_max, Ea_K_min_bound + 1.0))
    m_min_bound = max(float(m_min), 0.0)
    m_max_bound = max(float(m_max), m_min_bound + 1e-6)

    k0_rev_min_bound = max(float(k0_rev_min), 1e-15)
    k0_rev_max_bound = min(max(float(k0_rev_max), k0_rev_min_bound * 1.0001), 1e15)
    ea_rev_min_bound = max(float(ea_rev_min_J_mol), 0.0)
    ea_rev_max_bound = max(float(ea_rev_max_J_mol), ea_rev_min_bound + 1.0)
    order_rev_min_bound = float(order_rev_min)
    order_rev_max_bound = float(max(float(order_rev_max), order_rev_min_bound + 1e-6))

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
        # L-H 参数边界
        fit_K0_ads_flags=fit_K0_ads_flags,
        fit_Ea_K_flags=fit_Ea_K_flags,
        fit_m_flags=fit_m_flags,
        K0_ads_min=K0_ads_min_bound,
        K0_ads_max=K0_ads_max_bound,
        Ea_K_min=Ea_K_min_bound,
        Ea_K_max=Ea_K_max_bound,
        m_min=m_min_bound,
        m_max=m_max_bound,
        # 可逆反应参数边界
        fit_k0_rev_flags=fit_k0_rev_flags,
        fit_ea_rev_flags=fit_ea_rev_flags,
        fit_order_rev_flags_matrix=fit_order_rev_flags_matrix,
        k0_rev_min=k0_rev_min_bound,
        k0_rev_max=k0_rev_max_bound,
        ea_rev_min=ea_rev_min_bound,
        ea_rev_max=ea_rev_max_bound,
        order_rev_min=order_rev_min_bound,
        order_rev_max=order_rev_max_bound,
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
                    initial_cost = 0.5 * float(
                        np.dot(initial_residuals, initial_residuals)
                    )

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
                                random_x0[:n_fit_k0] = np.exp(
                                    ln_lb + u * (ln_ub - ln_lb)
                                )

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

        fitted_params = _unpack_parameters(
            parameter_vector=fitted_parameter_vector,
            k0_guess=k0_guess,
            ea_guess_J_mol=ea_guess_J_mol,
            order_guess=order_guess,
            fit_k0_flags=fit_k0_flags,
            fit_ea_flags=fit_ea_flags,
            fit_order_flags_matrix=fit_order_flags_matrix,
            # L-H 参数
            K0_ads_guess=K0_ads,
            Ea_K_guess=Ea_K_J_mol,
            m_inhibition_guess=m_inhibition,
            fit_K0_ads_flags=fit_K0_ads_flags,
            fit_Ea_K_flags=fit_Ea_K_flags,
            fit_m_flags=fit_m_flags,
            # 可逆反应参数
            k0_rev_guess=k0_rev,
            ea_rev_guess=ea_rev_J_mol,
            order_rev_guess=order_rev,
            fit_k0_rev_flags=fit_k0_rev_flags,
            fit_ea_rev_flags=fit_ea_rev_flags,
            fit_order_rev_flags_matrix=fit_order_rev_flags_matrix,
        )

        k0_fit = fitted_params["k0"]
        ea_fit_J_mol = fitted_params["ea_J_mol"]
        order_fit = fitted_params["reaction_order_matrix"]
        K0_ads_fit = fitted_params["K0_ads"]
        Ea_K_fit_J_mol = fitted_params["Ea_K"]
        m_inhibition_fit = fitted_params["m_inhibition"]
        k0_rev_fit = fitted_params["k0_rev"]
        ea_rev_fit_J_mol = fitted_params["ea_rev"]
        order_rev_fit = fitted_params["order_rev"]

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
        x_axis_list = []  # V_m3 for PFR, t_s for Batch
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
                reactor_type=reactor_type,
                kinetic_model=kinetic_model,
                K0_ads=K0_ads_fit,
                Ea_K_J_mol=Ea_K_fit_J_mol,
                m_inhibition=m_inhibition_fit,
                k0_rev=k0_rev_fit,
                ea_rev_J_mol=ea_rev_fit_J_mol,
                order_rev_matrix=order_rev_fit,
            )

            # 获取 x 轴数据：PFR 用体积，Batch 用时间
            if reactor_type == "PFR":
                x_val = row.get("V_m3", np.nan)
            else:
                x_val = row.get("t_s", np.nan)
            x_axis_list.append(float(x_val) if np.isfinite(x_val) else np.nan)
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

        # 确定 x 轴列名和标签
        if reactor_type == "PFR":
            x_col_name = "V_m3"
            x_label = "Volume $V$ [m$^3$]"
        else:
            x_col_name = "t_s"
            x_label = "Time $t$ [s]"

        plot_df = (
            pd.DataFrame(
                {
                    x_col_name: x_axis_list,
                    "measured": measured_list,
                    "predicted": predicted_list,
                    "status": status_list,
                }
            )
            .sort_values(x_col_name)
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
            parity_png_bytes = _figure_to_image_bytes(fig2, "png")
            parity_svg_bytes = _figure_to_image_bytes(fig2, "svg")
            st.pyplot(fig2, clear_figure=True)
            plt.close(fig2)

            col_par_exp1, col_par_exp2 = st.columns(2)
            with col_par_exp1:
                st.download_button(
                    label="🖼️ 导出奇偶校验图 (PNG)",
                    data=parity_png_bytes,
                    file_name=f"parity_{reactor_type}_{plot_species}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col_par_exp2:
                st.download_button(
                    label="🖼️ 导出奇偶校验图 (SVG)",
                    data=parity_svg_bytes,
                    file_name=f"parity_{reactor_type}_{plot_species}.svg",
                    mime="image/svg+xml",
                    use_container_width=True,
                )

        with col_plot2:
            st.markdown("##### 误差图 (Predicted - Measured)")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            error_values = plot_df["predicted"] - plot_df["measured"]
            ax3.plot(plot_df[x_col_name], error_values, "o-", label="误差", alpha=0.8)
            ax3.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
            ax3.set_xlabel(x_label, fontsize=10)
            ax3.set_ylabel(f"Error ({plot_species}, {output_mode})", fontsize=10)
            _apply_plot_tick_format(
                ax3,
                number_style=plot_number_style,
                decimal_places=int(plot_decimal_places),
                use_auto=bool(plot_tick_auto),
            )
            ax3.grid(True, linestyle=":", alpha=0.6)
            ax3.legend()
            error_png_bytes = _figure_to_image_bytes(fig3, "png")
            error_svg_bytes = _figure_to_image_bytes(fig3, "svg")
            st.pyplot(fig3, clear_figure=True)
            plt.close(fig3)

            col_err_exp1, col_err_exp2 = st.columns(2)
            with col_err_exp1:
                st.download_button(
                    label="🖼️ 导出误差图 (PNG)",
                    data=error_png_bytes,
                    file_name=f"error_{reactor_type}_{plot_species}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col_err_exp2:
                st.download_button(
                    label="🖼️ 导出误差图 (SVG)",
                    data=error_svg_bytes,
                    file_name=f"error_{reactor_type}_{plot_species}.svg",
                    mime="image/svg+xml",
                    use_container_width=True,
                )

        # ========== 沿程 / 随时间 剖面图 ==========
        st.divider()
        st.markdown("#### 沿程/随时间剖面（Profile）")
        with st.expander("查看剖面图（用于诊断：哪里消耗/生成最快）", expanded=False):
            if data_df.shape[0] == 0:
                st.info("无数据行可用于生成剖面图。")
            else:
                selected_row_index = int(
                    st.number_input(
                        "选择实验点行号（从 0 开始）",
                        min_value=0,
                        max_value=int(data_df.shape[0] - 1),
                        value=0,
                        step=1,
                    )
                )
                profile_points = int(
                    st.slider(
                        "剖面点数（越大越平滑，但越慢）",
                        min_value=30,
                        max_value=400,
                        value=200,
                        step=10,
                    )
                )
                profile_species_list = st.multiselect(
                    "选择要绘制的物种",
                    options=species_names,
                    default=species_names,
                    help="建议先选 1~3 个关键物种，避免曲线太多不易阅读。",
                )

                if len(profile_species_list) == 0:
                    st.warning("请至少选择一个物种。")
                else:
                    row_profile = data_df.iloc[selected_row_index]

                    if reactor_type == "PFR":
                        profile_quantity = st.radio(
                            "剖面输出量",
                            options=["F (mol/s)", "C (mol/m^3)"],
                            index=0,
                            help="PFR 默认以摩尔流量积分；浓度通过 C=F/vdot 计算（恒定体积流量假设）。",
                        )

                        reactor_volume_m3 = float(row_profile.get("V_m3", np.nan))
                        temperature_K_profile = float(row_profile.get("T_K", np.nan))
                        vdot_m3_s_profile = float(row_profile.get("vdot_m3_s", np.nan))
                        molar_flow_inlet_mol_s = np.array(
                            [
                                float(row_profile.get(f"F0_{s}_mol_s", 0.0))
                                for s in species_names
                            ],
                            dtype=float,
                        )

                        volume_grid_m3, flow_profile, ok, msg = integrate_pfr_profile(
                            reactor_volume_m3=reactor_volume_m3,
                            temperature_K=temperature_K_profile,
                            vdot_m3_s=vdot_m3_s_profile,
                            molar_flow_inlet_mol_s=molar_flow_inlet_mol_s,
                            stoich_matrix=stoich_matrix,
                            k0=k0_fit,
                            ea_J_mol=ea_fit_J_mol,
                            reaction_order_matrix=order_fit,
                            solver_method=solver_method,
                            rtol=rtol,
                            atol=atol,
                            n_points=profile_points,
                            kinetic_model=kinetic_model,
                            K0_ads=K0_ads_fit,
                            Ea_K_J_mol=Ea_K_fit_J_mol,
                            m_inhibition=m_inhibition_fit,
                            k0_rev=k0_rev_fit,
                            ea_rev_J_mol=ea_rev_fit_J_mol,
                            order_rev_matrix=order_rev_fit,
                        )

                        if not ok:
                            st.error(f"剖面积分失败：{msg}")
                        else:
                            fig_profile, ax_profile = plt.subplots(figsize=(7, 4.2))
                            for species_name in profile_species_list:
                                species_index = species_names.index(species_name)
                                if profile_quantity == "F (mol/s)":
                                    y_values = flow_profile[species_index, :]
                                    y_label = "Molar flow $F_i$ [mol/s]"
                                else:
                                    y_values = flow_profile[species_index, :] / max(
                                        vdot_m3_s_profile, 1e-30
                                    )
                                    y_label = "Concentration $C_i$ [mol/m$^3$]"
                                ax_profile.plot(
                                    volume_grid_m3,
                                    y_values,
                                    label=str(species_name),
                                    linewidth=2.0,
                                )

                            ax_profile.set_xlabel("Volume $V$ [m$^3$]")
                            ax_profile.set_ylabel(y_label)
                            ax_profile.grid(True, linestyle=":", alpha=0.6)
                            ax_profile.legend()
                            _apply_plot_tick_format(
                                ax_profile,
                                number_style=plot_number_style,
                                decimal_places=int(plot_decimal_places),
                                use_auto=bool(plot_tick_auto),
                            )

                            profile_png_bytes = _figure_to_image_bytes(fig_profile, "png")
                            profile_svg_bytes = _figure_to_image_bytes(fig_profile, "svg")
                            st.pyplot(fig_profile, clear_figure=True)
                            plt.close(fig_profile)

                            col_prof_exp1, col_prof_exp2 = st.columns(2)
                            with col_prof_exp1:
                                st.download_button(
                                    label="🖼️ 导出剖面图 (PNG)",
                                    data=profile_png_bytes,
                                    file_name=f"profile_{reactor_type}_row{selected_row_index}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                )
                            with col_prof_exp2:
                                st.download_button(
                                    label="🖼️ 导出剖面图 (SVG)",
                                    data=profile_svg_bytes,
                                    file_name=f"profile_{reactor_type}_row{selected_row_index}.svg",
                                    mime="image/svg+xml",
                                    use_container_width=True,
                                )

                    else:  # Batch
                        reaction_time_s_profile = float(row_profile.get("t_s", np.nan))
                        temperature_K_profile = float(row_profile.get("T_K", np.nan))
                        conc_initial_mol_m3 = np.array(
                            [
                                float(row_profile.get(f"C0_{s}_mol_m3", 0.0))
                                for s in species_names
                            ],
                            dtype=float,
                        )

                        time_grid_s, conc_profile, ok, msg = integrate_batch_profile(
                            reaction_time_s=reaction_time_s_profile,
                            temperature_K=temperature_K_profile,
                            conc_initial_mol_m3=conc_initial_mol_m3,
                            stoich_matrix=stoich_matrix,
                            k0=k0_fit,
                            ea_J_mol=ea_fit_J_mol,
                            reaction_order_matrix=order_fit,
                            solver_method=solver_method,
                            rtol=rtol,
                            atol=atol,
                            n_points=profile_points,
                            kinetic_model=kinetic_model,
                            K0_ads=K0_ads_fit,
                            Ea_K_J_mol=Ea_K_fit_J_mol,
                            m_inhibition=m_inhibition_fit,
                            k0_rev=k0_rev_fit,
                            ea_rev_J_mol=ea_rev_fit_J_mol,
                            order_rev_matrix=order_rev_fit,
                        )

                        if not ok:
                            st.error(f"剖面积分失败：{msg}")
                        else:
                            fig_profile, ax_profile = plt.subplots(figsize=(7, 4.2))
                            for species_name in profile_species_list:
                                species_index = species_names.index(species_name)
                                ax_profile.plot(
                                    time_grid_s,
                                    conc_profile[species_index, :],
                                    label=str(species_name),
                                    linewidth=2.0,
                                )

                            ax_profile.set_xlabel("Time $t$ [s]")
                            ax_profile.set_ylabel("Concentration $C_i$ [mol/m$^3$]")
                            ax_profile.grid(True, linestyle=":", alpha=0.6)
                            ax_profile.legend()
                            _apply_plot_tick_format(
                                ax_profile,
                                number_style=plot_number_style,
                                decimal_places=int(plot_decimal_places),
                                use_auto=bool(plot_tick_auto),
                            )

                            profile_png_bytes = _figure_to_image_bytes(fig_profile, "png")
                            profile_svg_bytes = _figure_to_image_bytes(fig_profile, "svg")
                            st.pyplot(fig_profile, clear_figure=True)
                            plt.close(fig_profile)

                            col_prof_exp1, col_prof_exp2 = st.columns(2)
                            with col_prof_exp1:
                                st.download_button(
                                    label="🖼️ 导出剖面图 (PNG)",
                                    data=profile_png_bytes,
                                    file_name=f"profile_{reactor_type}_row{selected_row_index}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                )
                            with col_prof_exp2:
                                st.download_button(
                                    label="🖼️ 导出剖面图 (SVG)",
                                    data=profile_svg_bytes,
                                    file_name=f"profile_{reactor_type}_row{selected_row_index}.svg",
                                    mime="image/svg+xml",
                                    use_container_width=True,
                                )

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

        if kinetic_model == "langmuir_hinshelwood":
            st.markdown("##### Langmuir-Hinshelwood 拟合参数")
            col_lh1, col_lh2 = st.columns(2)
            with col_lh1:
                st.markdown("**吸附参数 $K_i(T)$**")
                lh_ads_df = pd.DataFrame(
                    {"K0_ads": K0_ads_fit, "Ea_K_J_mol": Ea_K_fit_J_mol},
                    index=species_names,
                )
                st.data_editor(
                    lh_ads_df,
                    column_config=_build_table_column_config(
                        lh_ads_df, table_number_format
                    ),
                    num_rows="fixed",
                    key="lh_ads_result_table",
                    use_container_width=True,
                )
            with col_lh2:
                st.markdown("**抑制指数 $m_j$**")
                lh_m_df = pd.DataFrame(
                    {"m_inhibition": m_inhibition_fit},
                    index=[f"R{j+1}" for j in range(n_reactions)],
                )
                st.data_editor(
                    lh_m_df,
                    column_config=_build_table_column_config(lh_m_df, table_number_format),
                    num_rows="fixed",
                    key="lh_m_result_table",
                    use_container_width=True,
                )

        if kinetic_model == "reversible":
            st.markdown("##### 可逆反应拟合参数（逆反应）")
            col_rev1, col_rev2 = st.columns(2)
            with col_rev1:
                st.markdown("**$k_0^-$ & $E_a^-$**")
                rev_param_df = pd.DataFrame(
                    {"k0_rev": k0_rev_fit, "Ea_rev_J_mol": ea_rev_fit_J_mol},
                    index=[f"R{j+1}" for j in range(n_reactions)],
                )
                st.data_editor(
                    rev_param_df,
                    column_config=_build_table_column_config(
                        rev_param_df, table_number_format
                    ),
                    num_rows="fixed",
                    key="rev_param_result_table",
                    use_container_width=True,
                )
            with col_rev2:
                st.markdown("**逆反应级数 $n^-$**")
                rev_order_df = pd.DataFrame(
                    data=order_rev_fit,
                    index=[f"R{j+1}" for j in range(n_reactions)],
                    columns=species_names,
                )
                st.data_editor(
                    rev_order_df,
                    column_config=_build_table_column_config(
                        rev_order_df, table_number_format
                    ),
                    num_rows="fixed",
                    key="rev_order_result_table",
                    use_container_width=True,
                )

        # ========== 导出功能 ==========
        st.divider()
        st.markdown("##### 📥 导出拟合结果")
        col_export1, col_export2 = st.columns(2)

        with col_export1:
            # 导出拟合参数 CSV
            export_param_data = {
                "反应": [f"R{j+1}" for j in range(n_reactions)],
                "k0": k0_fit.tolist(),
                "Ea_J_mol": ea_fit_J_mol.tolist(),
            }
            for s_idx, s_name in enumerate(species_names):
                export_param_data[f"n_{s_name}"] = order_fit[:, s_idx].tolist()

            if kinetic_model == "langmuir_hinshelwood":
                export_param_data["m_inhibition"] = m_inhibition_fit.tolist()

            if kinetic_model == "reversible":
                export_param_data["k0_rev"] = k0_rev_fit.tolist()
                export_param_data["Ea_rev_J_mol"] = ea_rev_fit_J_mol.tolist()
                for s_idx, s_name in enumerate(species_names):
                    export_param_data[f"n_rev_{s_name}"] = order_rev_fit[:, s_idx].tolist()

            export_param_df = pd.DataFrame(export_param_data)
            param_csv = export_param_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 导出拟合参数 (CSV)",
                data=param_csv,
                file_name="fitting_params.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if kinetic_model == "langmuir_hinshelwood":
                export_ads_df = pd.DataFrame(
                    {
                        "species": species_names,
                        "K0_ads": K0_ads_fit.tolist(),
                        "Ea_K_J_mol": Ea_K_fit_J_mol.tolist(),
                    }
                )
                ads_csv = export_ads_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 导出吸附参数 (CSV)",
                    data=ads_csv,
                    file_name="lh_adsorption_params.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_export2:
            # 导出对比数据 CSV
            export_compare_df = plot_df.copy()
            export_compare_df["error"] = (
                export_compare_df["predicted"] - export_compare_df["measured"]
            )
            export_compare_df["relative_error_%"] = (
                100.0
                * export_compare_df["error"]
                / export_compare_df["measured"].replace(0, np.nan)
            )
            compare_csv = export_compare_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 导出对比数据 (CSV)",
                data=compare_csv,
                file_name="comparison_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

            build_report_table = st.checkbox(
                "生成报告表（可能较慢）",
                value=False,
                help="报告表会对每一行实验条件做一次预测（调用一次 solve_ivp），数据量大时会耗时。",
            )
            if build_report_table:
                # 导出“报告表”：保留原始 CSV 列 + 预测/误差列（对目标物种列表）
                report_df = data_df.copy()
                report_status_list = []
                predicted_rows = []

                for _, row in data_df.iterrows():
                    pred_values, ok, msg = _predict_outputs_for_row(
                        row=row,
                        species_names=species_names,
                        output_mode=output_mode,
                        output_species_list=output_species_list,
                        stoich_matrix=stoich_matrix,
                        k0=k0_fit,
                        ea_J_mol=ea_fit_J_mol,
                        reaction_order_matrix=order_fit,
                        solver_method=solver_method,
                        rtol=rtol,
                        atol=atol,
                        reactor_type=reactor_type,
                        kinetic_model=kinetic_model,
                        K0_ads=K0_ads_fit,
                        Ea_K_J_mol=Ea_K_fit_J_mol,
                        m_inhibition=m_inhibition_fit,
                        k0_rev=k0_rev_fit,
                        ea_rev_J_mol=ea_rev_fit_J_mol,
                        order_rev_matrix=order_rev_fit,
                    )
                    report_status_list.append("OK" if ok else f"FAIL: {msg}")
                    if ok:
                        predicted_rows.append(pred_values.astype(float).tolist())
                    else:
                        predicted_rows.append([np.nan] * len(output_species_list))

                report_df["solve_status"] = report_status_list
                predicted_matrix = np.array(predicted_rows, dtype=float)

                for output_index, species_name in enumerate(output_species_list):
                    if output_mode == "Fout (mol/s)":
                        meas_col = f"Fout_{species_name}_mol_s"
                    elif output_mode == "Cout (mol/m^3)":
                        meas_col = f"Cout_{species_name}_mol_m3"
                    else:
                        meas_col = f"X_{species_name}"

                    pred_col = f"pred_{meas_col}"
                    err_col = f"error_{meas_col}"
                    rel_col = f"relative_error_%_{meas_col}"

                    if meas_col not in report_df.columns:
                        report_df[meas_col] = np.nan

                    report_df[pred_col] = predicted_matrix[:, output_index]
                    report_df[err_col] = report_df[pred_col] - report_df[meas_col]
                    report_df[rel_col] = (
                        100.0
                        * report_df[err_col]
                        / report_df[meas_col].replace(0, np.nan)
                    )

                report_csv = report_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📑 导出报告表 (CSV)",
                    data=report_csv,
                    file_name="report_table.csv",
                    mime="text/csv",
                    help="包含原始 CSV 列 + solve_status + pred/error/relative_error 列（针对目标物种列表）。",
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
