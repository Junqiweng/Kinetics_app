from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from .kinetics import R_GAS_J_MOL_K


def _project_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _docs_dir() -> Path:
    return _project_root_dir() / "docs"


def _build_example_batch_csv_bytes() -> bytes:
    """
    生成一个 BSTR 示例数据（A -> B 一级反应，幂律 n=1）。
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


def read_text_if_exists(file_path: str) -> str | None:
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _render_markdown_file(file_path: Path) -> None:
    text = read_text_if_exists(str(file_path))
    if text is None:
        st.warning(f"未找到文档：`{file_path.as_posix()}`")
        return
    st.markdown(text)


def render_help_page() -> None:
    st.title("教程 / 帮助")
    st.caption("面向初学者：按步骤完成一次建模、拟合、诊断与导出。")

    docs_dir = _docs_dir()
    user_guide_path = docs_dir / "user_guide.md"
    user_guide_bytes = read_file_bytes_if_exists(str(user_guide_path))
    if user_guide_bytes is not None:
        st.download_button(
            "📥 下载《用户指南（详细版）》(Markdown)",
            data=user_guide_bytes,
            file_name="Kinetics_app_user_guide.md",
            mime="text/markdown",
            use_container_width=True,
        )

    tab_quick, tab_csv, tab_models, tab_fit, tab_trouble = st.tabs(
        ["快速上手", "CSV 列说明", "动力学模型", "拟合技巧", "常见问题"]
    )

    with tab_quick:
        _render_markdown_file(docs_dir / "help_quickstart.md")

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
                str(_project_root_dir() / "test_data" / "test_data_matched.csv")
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
                "📥 下载 BSTR 示例数据 (CSV)",
                data=batch_example_bytes,
                file_name="batch_example.csv",
                mime="text/csv",
                help="示例：A → B 一级反应，列包含 t_s/T_K/C0_*/Cout_*/X_A。",
                use_container_width=True,
            )

    with tab_csv:
        _render_markdown_file(docs_dir / "help_csv.md")

    with tab_models:
        _render_markdown_file(docs_dir / "help_models.md")

    with tab_fit:
        _render_markdown_file(docs_dir / "help_fitting.md")

    with tab_trouble:
        _render_markdown_file(docs_dir / "help_troubleshooting.md")
