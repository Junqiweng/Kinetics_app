from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import streamlit as st

import modules.ui_text as ui_text
from modules.export_config import (
    build_export_config_from_ctx,
    persist_export_config,
    render_export_config_button,
)
from modules.constants import (
    OUTPUT_MODE_COUT,
    OUTPUT_MODES_BATCH,
    OUTPUT_MODES_FLOW,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
    UI_DATA_PREVIEW_HEIGHT_PX,
)
from modules.upload_persistence import (
    _delete_persisted_upload,
    _read_csv_bytes_cached,
    _save_persisted_upload,
)


def _clear_data_editor_widget_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("data_editor_csv_"):
            del st.session_state[key]


def _bump_data_editor_revision() -> int:
    next_revision = int(st.session_state.get("_data_editor_csv_revision", 0)) + 1
    st.session_state["_data_editor_csv_revision"] = next_revision
    return next_revision


def _get_data_editor_key() -> str:
    revision = int(st.session_state.get("_data_editor_csv_revision", 0))
    return f"data_editor_csv_{revision}"


def _should_reset_cached_data(
    *,
    uploaded_file_token: str,
    csv_hash: str,
) -> bool:
    if "data_df_cached" not in st.session_state:
        return True

    last_upload_token = str(st.session_state.get("_data_df_upload_token", "")).strip()
    if uploaded_file_token and uploaded_file_token != last_upload_token:
        return True

    last_source_hash = str(st.session_state.get("_data_df_source_hash", "")).strip()
    return last_source_hash != str(csv_hash).strip()


def _get_data_requirements(
    reactor_type: str,
    pfr_flow_model: str,
    output_mode: str,
    species_names: list[str],
    output_species_list: list[str],
) -> tuple[list[str], list[str]]:
    if reactor_type == REACTOR_TYPE_PFR:
        if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
            required_input_columns = ["V_m3", "T_K", "P_Pa"] + [
                f"F0_{name}_mol_s" for name in species_names
            ]
        else:
            inlet_cols = (
                [f"C0_{name}_mol_m3" for name in species_names]
                if str(output_mode).startswith("C")
                else [f"F0_{name}_mol_s" for name in species_names]
            )
            required_input_columns = ["V_m3", "T_K", "vdot_m3_s"] + inlet_cols
    elif reactor_type == REACTOR_TYPE_CSTR:
        required_input_columns = ["V_m3", "T_K", "vdot_m3_s"] + [
            f"C0_{name}_mol_m3" for name in species_names
        ]
    else:
        required_input_columns = ["t_s", "T_K"] + [
            f"C0_{name}_mol_m3" for name in species_names
        ]

    output_columns: list[str] = []
    for species_name in output_species_list:
        if output_mode.startswith("F"):
            output_columns.append(f"Fout_{species_name}_mol_s")
        elif output_mode.startswith("x"):
            output_columns.append(f"xout_{species_name}")
        else:
            output_columns.append(f"Cout_{species_name}_mol_m3")
    return required_input_columns, output_columns


def _get_invalid_numeric_mask(
    column_name: str,
    numeric_values: np.ndarray,
    *,
    is_output_column: bool = False,
) -> np.ndarray:
    invalid_mask = ~np.isfinite(numeric_values)
    if bool(np.any(invalid_mask)):
        return invalid_mask
    if is_output_column:
        return np.zeros_like(numeric_values, dtype=bool)
    if column_name in ("T_K", "vdot_m3_s", "P_Pa"):
        return numeric_values <= 0.0
    if column_name in ("V_m3", "t_s"):
        return numeric_values < 0.0
    return numeric_values < 0.0


def _render_data_quality_panel(
    data_df: pd.DataFrame,
    reactor_type: str,
    pfr_flow_model: str,
    output_mode: str,
    species_names: list[str],
    output_species_list: list[str],
) -> None:
    required_input_columns, output_columns = _get_data_requirements(
        reactor_type,
        pfr_flow_model,
        output_mode,
        species_names,
        output_species_list,
    )
    missing_input_columns = [
        c for c in required_input_columns if c not in data_df.columns
    ]
    missing_output_columns = [c for c in output_columns if c not in data_df.columns]

    st.markdown("#### 数据质量检查")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("总行数", int(len(data_df)))
    col_q2.metric(
        "必需输入列",
        f"{len(required_input_columns) - len(missing_input_columns)}/{len(required_input_columns)}",
    )
    col_q3.metric(
        "目标测量列",
        f"{len(output_columns) - len(missing_output_columns)}/{len(output_columns)}",
    )

    valid_rows_count = None
    invalid_row_indices: list = []
    invalid_messages: list[str] = []

    if not missing_input_columns and not missing_output_columns:
        valid_mask = np.ones(len(data_df), dtype=bool)
        for column_name in required_input_columns:
            numeric_values = pd.to_numeric(data_df[column_name], errors="coerce").to_numpy(
                dtype=float
            )
            invalid_mask = _get_invalid_numeric_mask(column_name, numeric_values)
            if bool(np.any(invalid_mask)):
                invalid_indices = data_df.index[invalid_mask].tolist()
                preview_text = "、".join([str(i) for i in invalid_indices[:8]])
                invalid_messages.append(
                    f"`{column_name}` 有 {len(invalid_indices)} 行无效"
                    + (f"（示例 index: {preview_text}）" if preview_text else "")
                )
                valid_mask &= ~invalid_mask
        for column_name in output_columns:
            numeric_values = pd.to_numeric(data_df[column_name], errors="coerce").to_numpy(
                dtype=float
            )
            invalid_mask = _get_invalid_numeric_mask(
                column_name,
                numeric_values,
                is_output_column=True,
            )
            if bool(np.any(invalid_mask)):
                invalid_indices = data_df.index[invalid_mask].tolist()
                preview_text = "、".join([str(i) for i in invalid_indices[:8]])
                invalid_messages.append(
                    f"`{column_name}` 有 {len(invalid_indices)} 行无效"
                    + (f"（示例 index: {preview_text}）" if preview_text else "")
                )
                valid_mask &= ~invalid_mask
        valid_rows_count = int(np.count_nonzero(valid_mask))
        invalid_row_indices = data_df.index[~valid_mask].tolist()

    col_q4.metric(
        "当前可拟合行数",
        "--" if valid_rows_count is None else int(valid_rows_count),
    )

    if missing_input_columns or missing_output_columns:
        if missing_input_columns:
            st.error("缺少必需输入列：" + "，".join(missing_input_columns))
        if missing_output_columns:
            st.error("缺少目标测量列：" + "，".join(missing_output_columns))
    elif invalid_messages:
        st.warning("当前数据存在无效值，部分行不能直接参与拟合。")
        for message in invalid_messages:
            st.markdown(f"- {message}")
        if invalid_row_indices:
            preview = "、".join([str(i) for i in invalid_row_indices[:10]])
            st.caption(f"无效行示例 index：{preview}")
    else:
        st.success(
            "当前数据通过基础体检：必需列齐全，且所有行都满足当前拟合模式的基础数值约束。"
        )

    if output_columns:
        rows_species = []
        for species_name, column_name in zip(output_species_list, output_columns):
            if column_name not in data_df.columns:
                valid_count = 0
            else:
                numeric_values = pd.to_numeric(
                    data_df[column_name], errors="coerce"
                ).to_numpy(dtype=float)
                valid_count = int(np.count_nonzero(np.isfinite(numeric_values)))
            coverage_pct = (
                valid_count / max(int(len(data_df)), 1) * 100.0
                if len(data_df) > 0
                else 0.0
            )
            rows_species.append(
                {
                    "物种": str(species_name),
                    "测量列": str(column_name),
                    "有效点数": int(valid_count),
                    "覆盖率(%)": float(coverage_pct),
                }
            )
        if rows_species:
            st.dataframe(
                pd.DataFrame(rows_species),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("请选择至少一个目标物种后，可查看按物种的测量覆盖率。")


def render_data_tab(tab_data, ctx: dict) -> dict:
    get_cfg = ctx["get_cfg"]
    reactor_type = ctx["reactor_type"]
    pfr_flow_model = ctx["pfr_flow_model"]
    species_names = ctx["species_names"]
    session_id = ctx["session_id"]
    export_config_placeholder = ctx["export_config_placeholder"]
    kinetic_model = ctx["kinetic_model"]
    solver_method = ctx["solver_method"]
    rtol = ctx["rtol"]
    atol = ctx["atol"]
    species_text = ctx["species_text"]
    n_reactions = ctx["n_reactions"]
    stoich_matrix = ctx["stoich_matrix"]
    order_guess = ctx["order_guess"]
    fit_order_flags_matrix = ctx["fit_order_flags_matrix"]
    k0_guess = ctx["k0_guess"]
    ea_guess_J_mol = ctx["ea_guess_J_mol"]
    fit_k0_flags = ctx["fit_k0_flags"]
    fit_ea_flags = ctx["fit_ea_flags"]
    K0_ads = ctx["K0_ads"]
    Ea_K_J_mol = ctx["Ea_K_J_mol"]
    fit_K0_ads_flags = ctx["fit_K0_ads_flags"]
    fit_Ea_K_flags = ctx["fit_Ea_K_flags"]
    m_inhibition = ctx["m_inhibition"]
    fit_m_flags = ctx["fit_m_flags"]
    k0_rev = ctx["k0_rev"]
    ea_rev_J_mol = ctx["ea_rev_J_mol"]
    fit_k0_rev_flags = ctx["fit_k0_rev_flags"]
    fit_ea_rev_flags = ctx["fit_ea_rev_flags"]
    order_rev = ctx["order_rev"]
    fit_order_rev_flags_matrix = ctx["fit_order_rev_flags_matrix"]
    # ---------------- 选项卡 2：数据 ----------------
    data_df = st.session_state.get("data_df_cached", None)
    output_mode = OUTPUT_MODE_COUT
    output_species_list = []

    with tab_data:
        # --- 拟合目标变量 + 物种选择（同一行）---
        col_target1, col_target2 = st.columns(2)
        with col_target1:
            opts = (
                OUTPUT_MODES_FLOW
                if reactor_type in (REACTOR_TYPE_PFR, REACTOR_TYPE_CSTR)
                else OUTPUT_MODES_BATCH
            )
            if ("cfg_output_mode" in st.session_state) and (
                str(st.session_state["cfg_output_mode"]) not in opts
            ):
                st.session_state["cfg_output_mode"] = opts[0]
            output_mode = st.selectbox(
                "拟合目标变量",
                opts,
                index=(
                    opts.index(get_cfg("output_mode", opts[0]))
                    if get_cfg("output_mode", opts[0]) in opts
                    else 0
                ),
                format_func=lambda x: ui_text.map_label(
                    ui_text.OUTPUT_MODE_LABELS, str(x)
                ),
                key="cfg_output_mode",
                help="选择用于拟合的测量量类型：Cout（出口浓度）、Fout（出口摩尔流率）、xout（出口摩尔分率）。",
            )

        with col_target2:
            # 读取配置中保存的物种列表，并清理无效物种
            saved_species_list = get_cfg("output_species_list", None)
            if saved_species_list is not None and isinstance(saved_species_list, list):
                # 从配置中读取，过滤掉不在当前物种列表中的物种
                valid_species = [
                    str(x) for x in saved_species_list if str(x) in species_names
                ]
                if valid_species:
                    default_species = valid_species
                else:
                    default_species = list(species_names)
            else:
                default_species = list(species_names)

            # 重要：不要同时给 multiselect 的 default=... 并且又写 session_state[key]，
            # 否则会触发 Streamlit 警告：
            # "The widget with key ... was created with a default value but also had its value set via the Session State API."
            # 这里统一以 session_state 作为单一数据源。
            if "cfg_output_species_list" not in st.session_state:
                st.session_state["cfg_output_species_list"] = default_species
            else:
                current_list = st.session_state.get("cfg_output_species_list", [])
                if not isinstance(current_list, list):
                    current_list = []
                cleaned_list = [str(x) for x in current_list if str(x) in species_names]
                st.session_state["cfg_output_species_list"] = (
                    cleaned_list if cleaned_list else default_species
                )

            fit_mask = st.multiselect(
                "选择进入目标函数的物种",
                species_names,
                key="cfg_output_species_list",
                help="选择哪些物种的测量值用于计算拟合残差",
            )
            output_species_list = fit_mask

        st.divider()
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.markdown("#### 1. 下载模板")
            # 根据 output_mode 决定测量列
            if output_mode.startswith("F"):
                meas_cols = [f"Fout_{s}_mol_s" for s in species_names]
            elif output_mode.startswith("x"):
                meas_cols = [f"xout_{s}" for s in species_names]
            else:
                meas_cols = [f"Cout_{s}_mol_m3" for s in species_names]

            # 根据反应器类型决定输入条件列
            if reactor_type == REACTOR_TYPE_PFR:
                if pfr_flow_model == PFR_FLOW_MODEL_GAS_IDEAL_CONST_P:
                    # 气相 PFR：理想气体、等温、恒压 P（不考虑压降）
                    # 入口强制使用 F0_*，不再允许用 C0_* + vdot 换算。
                    inlet_cols = [f"F0_{s}_mol_s" for s in species_names]
                    cols = ["V_m3", "T_K", "P_Pa"] + inlet_cols + meas_cols
                else:
                    # 液相 PFR：体积流量 vdot 近似恒定
                    # 约定：当拟合目标为 Cout 时，入口也允许使用浓度 C0_*（并由 vdot 自动换算为 F0 参与计算）
                    inlet_cols = (
                        [f"C0_{s}_mol_m3" for s in species_names]
                        if output_mode.startswith("C")
                        else [f"F0_{s}_mol_s" for s in species_names]
                    )
                    cols = ["V_m3", "T_K", "vdot_m3_s"] + inlet_cols + meas_cols
            elif reactor_type == REACTOR_TYPE_CSTR:
                cols = (
                    ["V_m3", "T_K", "vdot_m3_s"]
                    + [f"C0_{s}_mol_m3" for s in species_names]
                    + meas_cols
                )
            else:
                # BSTR
                cols = (
                    ["t_s", "T_K"]
                    + [f"C0_{s}_mol_m3" for s in species_names]
                    + meas_cols
                )

            template_csv = (
                pd.DataFrame(columns=cols).to_csv(index=False).encode("utf-8")
            )
            # 动态生成模板文件名，包含反应器类型和测量类型
            template_filename = f"template_{reactor_type}_{output_mode.split()[0]}.csv"
            st.download_button(
                "📥 下载 CSV 模板", template_csv, template_filename, "text/csv"
            )
            st.caption(
                f"模板包含 {len(cols)} 列：输入条件 + {output_mode.split()[0]} 测量值"
            )

        with col_d2:
            st.markdown("#### 2. 上传数据")
            csv_uploader_key = (
                f"uploaded_csv_{int(st.session_state.get('uploader_ver_csv', 0))}"
            )
            if (
                "uploaded_csv_bytes" in st.session_state
                and st.session_state["uploaded_csv_bytes"]
            ):
                cached_name = str(st.session_state.get("uploaded_csv_name", "")).strip()
                cached_text = (
                    f"已缓存文件：{cached_name}" if cached_name else "已缓存上传文件"
                )
                st.caption(cached_text + "（页面刷新/切换不会丢失，除非手动删除）")
                if st.button("🗑️ 删除已上传文件", key="delete_uploaded_csv"):
                    for k in [
                        "uploaded_csv_bytes",
                        "uploaded_csv_name",
                        "_data_df_source_hash",
                        "_data_df_upload_token",
                        "_data_editor_csv_revision",
                    ]:
                        if k in st.session_state:
                            del st.session_state[k]
                    _clear_data_editor_widget_state()
                    if "data_df_cached" in st.session_state:
                        del st.session_state["data_df_cached"]
                    ok, message = _delete_persisted_upload(session_id)
                    if not ok:
                        st.warning(message)
                    if csv_uploader_key in st.session_state:
                        del st.session_state[csv_uploader_key]
                    st.session_state["uploader_ver_csv"] = (
                        int(st.session_state.get("uploader_ver_csv", 0)) + 1
                    )
                    st.rerun()

            uploaded_file = st.file_uploader(
                "上传 CSV",
                type=["csv"],
                label_visibility="collapsed",
                key=csv_uploader_key,
            )

        if uploaded_file:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_name = str(getattr(uploaded_file, "name", "")).strip()
                st.session_state["uploaded_csv_bytes"] = uploaded_bytes
                st.session_state["uploaded_csv_name"] = uploaded_name
                ok, message = _save_persisted_upload(
                    uploaded_bytes, uploaded_name, session_id
                )
                if not ok:
                    st.warning(message)
            except Exception as exc:
                st.error(f"读取上传文件失败: {exc}")

        if uploaded_file or (
            "uploaded_csv_bytes" in st.session_state
            and st.session_state["uploaded_csv_bytes"]
        ):
            try:
                uploaded_file_token = ""
                if uploaded_file:
                    csv_bytes = uploaded_file.getvalue()
                    uploaded_file_token = str(
                        getattr(uploaded_file, "file_id", "") or ""
                    ).strip()
                else:
                    csv_bytes = st.session_state["uploaded_csv_bytes"]
                data_df = _read_csv_bytes_cached(csv_bytes)
                _csv_hash = hashlib.md5(csv_bytes).hexdigest()
                if _should_reset_cached_data(
                    uploaded_file_token=uploaded_file_token,
                    csv_hash=_csv_hash,
                ):
                    _clear_data_editor_widget_state()
                    _bump_data_editor_revision()
                    st.session_state["data_df_cached"] = data_df.copy()
                st.session_state["_data_df_source_hash"] = _csv_hash
                st.session_state["_data_df_upload_token"] = uploaded_file_token

                st.markdown("#### 数据预览（可直接编辑）")
                edited_df = st.data_editor(
                    st.session_state["data_df_cached"],
                    width="stretch",
                    height=UI_DATA_PREVIEW_HEIGHT_PX,
                    num_rows="dynamic",
                    key=_get_data_editor_key(),
                )
                st.session_state["data_df_cached"] = edited_df
                data_df = edited_df
                _render_data_quality_panel(
                    data_df=edited_df,
                    reactor_type=str(reactor_type),
                    pfr_flow_model=str(pfr_flow_model),
                    output_mode=str(output_mode),
                    species_names=list(species_names),
                    output_species_list=list(output_species_list),
                )
            except Exception as exc:
                st.error(f"CSV 读取失败: {exc}")
                data_df = None

    # --- 构建导出配置（基础版；若拟合页启用高级设置，会在拟合页再次更新）---
    if export_config_placeholder is not None:
        export_cfg = build_export_config_from_ctx(
            ctx,
            output_mode=str(output_mode),
            output_species_list=list(output_species_list),
        )
        ok, message = persist_export_config(export_cfg, session_id)
        if not ok and message:
            st.warning(message)
        render_export_config_button(
            export_config_placeholder,
            export_cfg,
            button_key="export_config_download_basic",
        )
    return {
        "data_df": data_df,
        "output_mode": output_mode,
        "output_species_list": output_species_list,
    }
