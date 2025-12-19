import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def figure_to_image_bytes(fig: plt.Figure, image_format: str) -> bytes:
    """
    将 Matplotlib Figure 导出为字节流。
    """
    image_format = str(image_format).lower().strip()
    buf = io.BytesIO()
    save_kwargs = {"format": image_format, "bbox_inches": "tight"}
    if image_format in ["png", "jpg", "jpeg", "tif", "tiff"]:
        save_kwargs["dpi"] = 300
    fig.savefig(buf, **save_kwargs)
    return buf.getvalue()


def apply_plot_tick_format(
    ax: plt.Axes, number_style: str, decimal_places: int, use_auto: bool
) -> None:
    if use_auto:
        return
    decimal_places = int(decimal_places)
    fmt_str = (
        f"{{:.{decimal_places}e}}"
        if number_style == "科学计数"
        else f"{{:.{decimal_places}f}}"
    )
    formatter = FuncFormatter(
        lambda x, pos: "" if not np.isfinite(x) else fmt_str.format(float(x))
    )
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def build_table_column_config(data_df: pd.DataFrame, number_format: str) -> dict:
    column_config: dict = {}
    for col in data_df.columns:
        if pd.api.types.is_numeric_dtype(data_df[col]):
            column_config[col] = st.column_config.NumberColumn(
                col, format=number_format
            )
        else:
            column_config[col] = st.column_config.TextColumn(col)
    return column_config


def render_param_table(
    key_prefix: str,
    index_names: list[str],
    col1_name: str,
    col1_data: np.ndarray,
    col1_help: str,
    col2_name: str,
    col2_data: np.ndarray,
    col2_help: str,
    fit1_default: np.ndarray,
    fit2_default: np.ndarray,
    height: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    通用的双参数表格渲染 (如 k0/Ea, K0/Ea, k0_rev/Ea_rev)。
    返回: val1_arr, val2_arr, fit1_arr, fit2_arr
    """
    df = pd.DataFrame(
        {
            col1_name: [f"{v:.2e}" for v in col1_data],
            f"Fit_{col1_name}": fit1_default,
            col2_name: [f"{v:.2e}" for v in col2_data],
            f"Fit_{col2_name}": fit2_default,
        },
        index=index_names,
    )

    col_cfg = {
        col1_name: st.column_config.TextColumn(col1_name, help=col1_help),
        f"Fit_{col1_name}": st.column_config.CheckboxColumn(
            f"拟合 {col1_name.split(' ')[0]}", default=True
        ),
        col2_name: st.column_config.TextColumn(col2_name, help=col2_help),
        f"Fit_{col2_name}": st.column_config.CheckboxColumn(
            f"拟合 {col2_name.split(' ')[0]}", default=True
        ),
    }

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        height=height,
        column_config=col_cfg,
        key=key_prefix,
    )

    val1 = pd.to_numeric(edited_df[col1_name], errors="coerce").to_numpy(dtype=float)
    val2 = pd.to_numeric(edited_df[col2_name], errors="coerce").to_numpy(dtype=float)
    fit1 = edited_df[f"Fit_{col1_name}"].to_numpy(dtype=bool)
    fit2 = edited_df[f"Fit_{col2_name}"].to_numpy(dtype=bool)

    return val1, val2, fit1, fit2


def render_order_table(
    key_prefix: str,
    row_names: list[str],
    species_names: list[str],
    order_data: np.ndarray | None,
    fit_data: np.ndarray | None,
    height: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """
    渲染反应级数矩阵编辑器。
    返回: order_matrix, fit_matrix
    """
    n_reactions = len(row_names)
    data_dict = {}

    for i, sp_name in enumerate(species_names):
        col_n = f"n_{sp_name}"
        col_fit = f"Fit_{sp_name}"

        if order_data is not None:
            data_dict[col_n] = order_data[:, i]
        else:
            # Default logic handled specifically by caller usually, but simplified here
            data_dict[col_n] = np.zeros(n_reactions)

        if fit_data is not None:
            data_dict[col_fit] = fit_data[:, i]
        else:
            data_dict[col_fit] = np.zeros(n_reactions, dtype=bool)

    df = pd.DataFrame(data_dict, index=row_names)

    col_cfg = {}
    for sp_name in species_names:
        col_cfg[f"n_{sp_name}"] = st.column_config.NumberColumn(
            f"n_{sp_name}", format="%.2f"
        )
        col_cfg[f"Fit_{sp_name}"] = st.column_config.CheckboxColumn(
            f"拟合 {sp_name}", default=False
        )

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        height=height,
        column_config=col_cfg,
        key=key_prefix,
    )

    order_mat = np.zeros((n_reactions, len(species_names)), dtype=float)
    fit_mat = np.zeros((n_reactions, len(species_names)), dtype=bool)

    for i, sp_name in enumerate(species_names):
        order_mat[:, i] = edited_df[f"n_{sp_name}"].to_numpy(dtype=float)
        fit_mat[:, i] = edited_df[f"Fit_{sp_name}"].to_numpy(dtype=bool)

    return order_mat, fit_mat
