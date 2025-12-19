from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from .kinetics import R_GAS_J_MOL_K


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


def read_file_bytes_if_exists(file_path: str) -> bytes | None:
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        return path.read_bytes()
    except Exception:
        return None


def render_help_page() -> None:
    st.title("教程 / 帮助")
    st.caption("面向初学者：按步骤完成一次建模、拟合、诊断与导出。")

    tab_quick, tab_csv, tab_models, tab_fit, tab_trouble = st.tabs(
        ["快速上手", "CSV 列说明", "动力学模型", "拟合技巧", "常见问题"]
    )

    with tab_quick:
        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            st.markdown(
                "**推荐流程（一次完整的拟合）**\n"
                "1) 在左侧选择反应器类型与动力学模型；\n"
                "2) 输入物种名与反应数，填写化学计量数矩阵 $\\nu$；\n"
                "3) 设置级数矩阵 $n$、k0/Ea 初值与拟合开关（Fit）；\n"
                "4) 下载并填写 CSV 模板（或用示例数据）；\n"
                "5) 上传 CSV，选择拟合目标变量与进入目标函数的物种；\n"
                "6) 点击“开始拟合”，查看奇偶校验图、残差图与沿程/随时间剖面；\n"
                "7) 导出拟合参数、预测 vs 实验对比表与图像；\n"
                "8) （推荐）用“配置管理”导出 JSON，便于下次直接复现。"
            )

        with col_q2:
            st.info(
                "💡 **提示**：\n"
                "- 你可以先用示例数据跑通流程，再替换为自己的实验数据。\n"
                "- 若拟合不动，可尝试：增大 `diff_step`、增大 `max_nfev`、开启 `multi-start`、或切换 ODE 求解器为 `BDF/Radau`。\n"
                "- 上传的 CSV 会被自动缓存，页面刷新/切换不会丢失；如需删除，请点“删除已上传文件”。"
            )

        st.divider()
        st.markdown("**目标函数（拟合在最小化什么？）**")
        st.latex(
            r"\Phi(\theta)=\frac{1}{2}\sum_{i=1}^{N} r_i(\theta)^2,\quad r_i=y_i^{\mathrm{pred}}-y_i^{\mathrm{meas}}"
        )
        st.caption(
            "其中：$\\theta$ 为待拟合参数向量（如 $k_0,E_a,n$ 等），$N$ 为用于拟合的数据点数（含多个物种/多行数据）。"
        )

        st.divider()
        st.markdown("**示例数据下载（可直接用于上手）**")
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            pfr_example_bytes = read_file_bytes_if_exists(
                "test_data/test_data_matched.csv"
            )
            if pfr_example_bytes is None:
                st.warning(
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

    with tab_csv:
        st.markdown("**核心原则：列名必须与模板一致**（大小写/下划线都要匹配）。")
        st.caption("建议：先下载模板 → 把你的数据粘进去 → 再上传。")
        st.divider()

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
            "重要：当前版本不允许测量列缺失或含 NaN/非数字；否则会停止拟合并提示缺失列/问题行号。"
        )
        st.markdown("**常见建议**")
        st.markdown(
            "- 如果某个物种没有测量值：要么补齐该物种的测量列，要么在“进入目标函数的物种”里取消勾选该物种。\n"
            "- 如果只想拟合转化率：选择 `X (conversion)`，并确保 CSV 里有 `X_<物种名>` 列。"
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
        st.latex(r"r_j=k_j^+(T)\prod_i C_i^{n_{ij}^+}-k_j^-(T)\prod_i C_i^{n_{ij}^-}")
        st.latex(
            r"k_j^{\pm}(T)=k_{0,j}^{\pm}\exp\left(-\frac{E_{a,j}^{\pm}}{RT}\right)"
        )
        st.caption("正/逆反应有各自的 k0/Ea/n，拟合时可分别勾选 Fit。")

    with tab_fit:
        st.markdown("**推荐的拟合设置（更稳健）**")
        st.markdown(
            "- 初值不准/拟合不动：把 `diff_step` 调大到 `1e-2 ~ 1e-3`；并开启 `multi-start`；\n"
            "- 刚性明显（收敛困难/很慢）：ODE 求解器选 `BDF` 或 `Radau`；\n"
            "- 多参数混合拟合：建议开启 `x_scale='jac'`；\n"
            "- 如果看到“达到最大迭代次数上限”：增大 `Max Iterations`，或缩紧参数边界，或改进初值。"
        )
        st.divider()
        st.markdown("**进度条/状态栏的含义**")
        st.markdown(
            "- `Max Iterations` 是外层迭代次数上限；\n"
            "- 状态栏里的“调用≈a/b”是模型函数调用次数的估算进度（数值差分 Jacobian 会导致每次迭代调用多次），属于正常现象。"
        )
        st.divider()
        st.markdown("**导入/导出配置（强烈推荐）**")
        st.markdown(
            "- 侧边栏“配置管理”可以导出当前全部设置为 JSON（包含反应器、动力学、求解器、参数、边界与拟合设置）；\n"
            "- 下次打开 App 会自动恢复上次配置；也可以导入 JSON 一键复现。"
        )

    with tab_trouble:
        st.markdown("**常见报错与处理建议**")
        st.markdown(
            "- `solve_ivp失败`：尝试 `BDF/Radau`，或调松 `rtol/atol`，或缩紧参数边界。\n"
            "- `T_K 无效` / `vdot 无效`：检查 CSV 对应列是否为正数。\n"
            "- 负级数 + 浓度趋近 0：会导致 $C^n$ 发散；程序对负级数使用浓度下限避免 `inf`，但建议检查模型合理性。\n"
            "- `x0 infeasible`：初值超出边界；程序会自动裁剪到边界内，但仍建议你设置更合理的初值与边界。\n"
            "- `数据表缺少所选输出测量列` / `测量列中存在 NaN`：检查 CSV 表头与缺失值；或取消选择对应物种/输出模式。"
        )
