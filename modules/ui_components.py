# 文件作用：封装 Streamlit 常用 UI 组件（数值输入、表格编辑、绘图导出等），便于在主应用中复用。

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from .constants import (
    UI_FLOAT_NORMAL_MAX_DECIMALS,
    UI_FLOAT_SCI_DECIMALS,
    UI_FLOAT_SCI_FORMAT_STREAMLIT,
    UI_FLOAT_SCI_HIGH,
    UI_FLOAT_SCI_LOW,
    UI_ORDER_MATRIX_NUMBER_FORMAT,
)


def _trim_trailing_zeros(number_text: str) -> str:
    """
    去掉常规小数显示中无意义的尾随 0（以及可能的末尾小数点）。
    """
    text = str(number_text)
    if ("e" in text) or ("E" in text):
        return text
    if "." not in text:
        return text
    text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "+0", "") else text


def smart_float_to_str(
    value: float | int | None,
    sci_low: float = UI_FLOAT_SCI_LOW,
    sci_high: float = UI_FLOAT_SCI_HIGH,
    sci_decimals: int = UI_FLOAT_SCI_DECIMALS,
    normal_max_decimals: int = UI_FLOAT_NORMAL_MAX_DECIMALS,
) -> str:
    """
    智能数字显示：小于 sci_low 或大于等于 sci_high 时用科学计数，否则用常规数字。

    说明：
    - 常规数字：使用固定小数位再裁剪尾随 0，避免无意义的补 0。
    - 科学计数：使用固定小数位（例如 1.23e+04）。
    """
    if value is None:
        return ""

    try:
        x = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(x):
        return ""

    if x == 0.0:
        return "0"

    abs_x = abs(x)
    if (abs_x < float(sci_low)) or (abs_x >= float(sci_high)):
        sci_decimals = int(max(0, sci_decimals))
        return f"{x:.{sci_decimals}e}"

    normal_max_decimals = int(max(0, normal_max_decimals))
    return _trim_trailing_zeros(f"{x:.{normal_max_decimals}f}")


def smart_number_input(
    label: str,
    value: float,
    key: str,
    min_value: float | None = None,
    max_value: float | None = None,
    step: float | None = None,
    help: str | None = None,
    disabled: bool = False,
    label_visibility: str = "visible",
    container=None,
) -> float:
    """
    智能 number_input：
    - 当数值很小/很大时（阈值在 constants.py 统一配置），用科学计数显示（避免超长数字）。
    - 其它情况不传 format，使用 Streamlit 默认显示（更像“常规数字”，不会强制补 0）。

    说明：判断优先使用当前 session_state 中的值（用户正在编辑时更符合直觉）。
    """
    current_value = st.session_state.get(key, value)
    try:
        x = float(current_value)
    except Exception:
        x = float(value)

    use_sci = (x != 0.0) and (
        (abs(x) < float(UI_FLOAT_SCI_LOW)) or (abs(x) >= float(UI_FLOAT_SCI_HIGH))
    )

    kwargs: dict = {
        "label": label,
        "value": float(value),
        "key": key,
        "help": help,
        "disabled": bool(disabled),
        "label_visibility": str(label_visibility),
    }
    if min_value is not None:
        kwargs["min_value"] = float(min_value)
    if max_value is not None:
        kwargs["max_value"] = float(max_value)
    if step is not None:
        kwargs["step"] = float(step)

    if use_sci:
        kwargs["format"] = UI_FLOAT_SCI_FORMAT_STREAMLIT

    if container is None:
        return float(st.number_input(**kwargs))

    return float(container.number_input(**kwargs))


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
        # 自动模式：根据数量级自动切换科学计数
        try:
            from matplotlib.ticker import ScalarFormatter

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 4))
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
        except Exception:
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


def format_dataframe_for_display(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中的数值列统一转成“智能显示”的字符串，便于 st.dataframe 显示时
    与输入框的数字风格一致（常规/科学计数自动切换）。
    """
    if data_df is None:
        return pd.DataFrame()

    out_df = data_df.copy()
    for col in out_df.columns:
        if pd.api.types.is_numeric_dtype(out_df[col]):
            col_values = out_df[col].to_numpy(dtype=float, copy=False)
            out_df[col] = [smart_float_to_str(v) for v in col_values]
    return out_df


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

    def ensure_1d_length(
        input_array: np.ndarray | list | tuple | float | int | bool | None,
        target_length: int,
        fill_value: float | bool,
    ) -> np.ndarray:
        if target_length <= 0:
            return np.array([], dtype=type(fill_value))

        if input_array is None:
            return np.full(target_length, fill_value)

        arr = np.asarray(input_array)

        if arr.ndim == 0:
            arr = np.array([arr.item()])
        else:
            arr = arr.reshape(-1)

        if arr.size == target_length:
            return arr

        if arr.size == 0:
            return np.full(target_length, fill_value)

        pad_value = arr[-1]
        out = np.full(target_length, pad_value)
        n_copy = min(target_length, arr.size)
        out[:n_copy] = arr[:n_copy]
        return out

    target_length = len(index_names)
    col1_size = 0 if col1_data is None else int(np.asarray(col1_data).size)
    col2_size = 0 if col2_data is None else int(np.asarray(col2_data).size)
    fit1_size = 0 if fit1_default is None else int(np.asarray(fit1_default).size)
    fit2_size = 0 if fit2_default is None else int(np.asarray(fit2_default).size)
    input_sizes = [col1_size, col2_size, fit1_size, fit2_size]
    needs_resize = any((s not in (0, target_length)) for s in input_sizes)

    col1_data = ensure_1d_length(col1_data, target_length, np.nan)
    col2_data = ensure_1d_length(col2_data, target_length, np.nan)
    fit1_default = ensure_1d_length(fit1_default, target_length, False).astype(bool)
    fit2_default = ensure_1d_length(fit2_default, target_length, False).astype(bool)

    # 自动补齐/截断逻辑已在上方 ensure_1d_length 中处理，无需警告

    df = pd.DataFrame(
        {
            col1_name: [smart_float_to_str(v) for v in col1_data],
            f"Fit_{col1_name}": fit1_default,
            col2_name: [smart_float_to_str(v) for v in col2_data],
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

    def ensure_2d_shape(
        input_matrix: np.ndarray | list | tuple | None,
        target_rows: int,
        target_cols: int,
        fill_value: float | bool,
    ) -> np.ndarray:
        if target_rows <= 0 or target_cols <= 0:
            return np.zeros((max(0, target_rows), max(0, target_cols)))

        if input_matrix is None:
            return np.full((target_rows, target_cols), fill_value)

        mat = np.asarray(input_matrix)
        if mat.ndim != 2:
            mat = np.atleast_2d(mat)

        out = np.full((target_rows, target_cols), fill_value)
        n_copy_rows = min(target_rows, mat.shape[0])
        n_copy_cols = min(target_cols, mat.shape[1])
        out[:n_copy_rows, :n_copy_cols] = mat[:n_copy_rows, :n_copy_cols]
        return out

    order_data = ensure_2d_shape(
        order_data, n_reactions, len(species_names), 0.0
    ).astype(float)
    fit_data = ensure_2d_shape(fit_data, n_reactions, len(species_names), False).astype(
        bool
    )

    for i, sp_name in enumerate(species_names):
        col_n = f"n_{sp_name}"
        col_fit = f"Fit_{sp_name}"

        data_dict[col_n] = order_data[:, i]
        data_dict[col_fit] = fit_data[:, i]

    df = pd.DataFrame(data_dict, index=row_names)

    col_cfg = {}
    for sp_name in species_names:
        col_cfg[f"n_{sp_name}"] = st.column_config.NumberColumn(
            f"n_{sp_name}", format=UI_ORDER_MATRIX_NUMBER_FORMAT
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
