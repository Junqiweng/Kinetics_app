"""
会话状态与配置保存回归脚本（不依赖 pytest）。

目标：
1) 验证“导入配置”会清理控件状态与拟合缓存，但保留当前 CSV 数据。
2) 验证“重置默认”会同时清空 CSV 内存缓存。
3) 验证无效配置在自动保存时会返回明确失败信息（不再静默跳过）。

使用方法（项目根目录）：
  python scripts/regression_state_validate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules import config_manager
from modules import export_config
from modules.config_state import (
    _apply_imported_config_to_widget_state,
    _clear_state_for_imported_config,
    _clear_state_for_reset_default,
)
from modules.data_utils import _parse_reaction_equation
from modules.fit_results import (
    _build_initial_guess_updates_from_fit_result,
    _map_parity_temperatures_by_row_index,
)
from modules.fit_setup import derive_effective_fit_flags, resolve_fit_parameter_state
from modules.fit_state import build_fit_state_snapshot, describe_fit_state_differences
from modules.fitting_background import _append_fit_history_entry
from modules.tab_data import _bump_data_editor_revision, _get_data_editor_key, _should_reset_cached_data
from modules.tab_model import _build_stoich_widget_reset_prefixes

# 在“非 streamlit run”模式下，屏蔽无关的 ScriptRunContext 警告，保留断言失败信息。
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)


def _assert_true(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _reset_session_state() -> None:
    # 说明：在 bare mode 下可直接清空 session_state，用于模拟“同一会话内多个场景”。
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def test_import_keeps_csv_data() -> None:
    _reset_session_state()
    st.session_state["uploader_ver_config_json"] = 10
    st.session_state["uploader_ver_csv"] = 20
    st.session_state["cfg_reactor_type"] = "PFR"
    st.session_state["fit_results"] = {"dummy": True}
    st.session_state["imported_config"] = {"dummy": True}
    st.session_state["uploaded_csv_bytes"] = b"a,b\n1,2\n"
    st.session_state["uploaded_csv_name"] = "data.csv"
    st.session_state["data_df_cached"] = "DUMMY_DF"
    st.session_state["base_params_2"] = {"edited_rows": {0: {"k₀": "1e6"}}}

    _clear_state_for_imported_config()

    _assert_true("cfg_reactor_type" not in st.session_state, "导入配置后 cfg_* 应被清理")
    _assert_true("fit_results" not in st.session_state, "导入配置后旧拟合结果应被清理")
    _assert_true(
        st.session_state.get("uploaded_csv_bytes") == b"a,b\n1,2\n",
        "导入配置后应保留 uploaded_csv_bytes",
    )
    _assert_true(
        st.session_state.get("uploaded_csv_name") == "data.csv",
        "导入配置后应保留 uploaded_csv_name",
    )
    _assert_true(
        st.session_state.get("data_df_cached") == "DUMMY_DF",
        "导入配置后应保留 data_df_cached",
    )
    _assert_true(
        "base_params_2" not in st.session_state,
        "导入配置后应清理拟合页参数表缓存 key",
    )
    _assert_true(
        int(st.session_state.get("uploader_ver_csv", -1)) == 20,
        "导入配置后不应重建 CSV uploader",
    )
    _assert_true(
        int(st.session_state.get("uploader_ver_config_json", -1)) == 11,
        "导入配置后应重建配置 uploader",
    )


def test_reset_clears_csv_data() -> None:
    _reset_session_state()
    st.session_state["uploader_ver_config_json"] = 1
    st.session_state["uploader_ver_csv"] = 2
    st.session_state["uploaded_csv_bytes"] = b"a,b\n1,2\n"
    st.session_state["uploaded_csv_name"] = "data.csv"
    st.session_state["data_df_cached"] = "DUMMY_DF"
    st.session_state["fit_results"] = {"dummy": True}
    st.session_state["rev_orders_3"] = {"edited_rows": {0: {"A": 1.0}}}

    _clear_state_for_reset_default()

    _assert_true("uploaded_csv_bytes" not in st.session_state, "重置默认后应清空 uploaded_csv_bytes")
    _assert_true("uploaded_csv_name" not in st.session_state, "重置默认后应清空 uploaded_csv_name")
    _assert_true("data_df_cached" not in st.session_state, "重置默认后应清空 data_df_cached")
    _assert_true("fit_results" not in st.session_state, "重置默认后应清空旧拟合结果")
    _assert_true("rev_orders_3" not in st.session_state, "重置默认后应清理拟合页逆反应表缓存 key")
    _assert_true(
        int(st.session_state.get("uploader_ver_csv", -1)) == 3,
        "重置默认后应重建 CSV uploader",
    )
    _assert_true(
        int(st.session_state.get("uploader_ver_config_json", -1)) == 2,
        "重置默认后应重建配置 uploader",
    )


def test_invalid_export_config_returns_error() -> None:
    cfg = config_manager.get_default_config()
    cfg["max_nfev"] = 0
    ok, message = export_config.persist_export_config(cfg, session_id=None)
    _assert_true(not ok, "无效配置自动保存应返回失败")
    _assert_true(
        "未通过校验" in str(message),
        "无效配置自动保存应返回可读错误信息",
    )


def test_reversible_config_migration() -> None:
    _reset_session_state()
    legacy_cfg = {
        "version": "1.0",
        "reactor_type": "PFR",
        "kinetic_model": "reversible",
        "species_text": "A,B",
        "n_reactions": 1,
    }
    ok, msg = config_manager.validate_config(legacy_cfg)
    _assert_true(ok, f"旧版 reversible 配置应可自动迁移并通过校验：{msg}")
    _assert_true(
        str(legacy_cfg.get("kinetic_model")) == "power_law",
        "旧版 reversible 配置应自动迁移为 power_law",
    )
    _assert_true(
        bool(legacy_cfg.get("reversible_enabled", False)),
        "旧版 reversible 配置应自动补充 reversible_enabled=True",
    )

    _apply_imported_config_to_widget_state(legacy_cfg)
    _assert_true(
        bool(st.session_state.get("cfg_reversible_enabled", False)),
        "导入迁移后的旧配置时，cfg_reversible_enabled 应为 True",
    )


def test_fit_history_entry_is_recorded() -> None:
    _reset_session_state()
    fit_results = {
        "phi_final": 1.23,
        "kinetic_model": "power_law",
        "reactor_type": "PFR",
        "residual_type": "绝对残差",
        "n_fit_params": 3,
        "fit_skipped": False,
        "output_mode": "Cout",
        "output_species": ["A"],
        "species_names": ["A", "B"],
        "stoich_matrix": [[-1.0], [1.0]],
        "params": {
            "k0": [1.0],
            "ea_J_mol": [2.0],
            "reaction_order_matrix": [[1.0, 0.0]],
        },
    }
    _append_fit_history_entry(fit_results)
    history = st.session_state.get("fitting_history", [])
    _assert_true(len(history) == 1, "完成拟合后应写入 fitting_history")
    _assert_true(
        float(history[0].get("phi", 0.0)) == 1.23,
        "拟合历史应记录目标函数值",
    )


def test_reaction_equation_parser_supports_literal_species_names() -> None:
    species_names = ["1-Butene", "i-C4H10", "H2O(l)"]
    nu = _parse_reaction_equation(
        "1-Butene + 0.5 i-C4H10 -> H2O(l)",
        species_names,
    )
    _assert_true(nu is not None, "反应式解析应支持当前物种名中的连字符/括号/数字前缀")
    _assert_true(
        list(nu.astype(float)) == [-1.0, -0.5, 1.0],
        "反应式解析后的计量数应正确",
    )


def test_reaction_equation_parser_supports_ionic_species_names() -> None:
    species_names = ["H+", "OH-", "H2O"]
    nu = _parse_reaction_equation("H+ + OH- -> H2O", species_names)
    _assert_true(nu is not None, "反应式解析应支持物种名中包含加号")
    _assert_true(
        list(nu.astype(float)) == [-1.0, -1.0, 1.0],
        "含离子物种名时，计量数应保持正确",
    )


def test_same_csv_reupload_resets_cached_editor_data() -> None:
    _reset_session_state()
    st.session_state["data_df_cached"] = "EDITED"
    st.session_state["_data_df_source_hash"] = "same-hash"
    st.session_state["_data_df_upload_token"] = "upload-token-1"

    _assert_true(
        not _should_reset_cached_data(
            uploaded_file_token="upload-token-1",
            csv_hash="same-hash",
        ),
        "同一次上传在普通 rerun 中不应覆盖编辑中的表格",
    )
    _assert_true(
        _should_reset_cached_data(
            uploaded_file_token="upload-token-2",
            csv_hash="same-hash",
        ),
        "重新选择同一份 CSV 时，应恢复原始文件内容",
    )


def test_reupload_changes_data_editor_key_revision() -> None:
    _reset_session_state()
    _assert_true(_get_data_editor_key() == "data_editor_csv_0", "初始编辑器 key 应稳定")
    revision = _bump_data_editor_revision()
    _assert_true(revision == 1, "第一次换文件时应提升编辑器 revision")
    _assert_true(
        _get_data_editor_key() == "data_editor_csv_1",
        "编辑器 key 应随 revision 改变，避免复用旧编辑状态",
    )


def test_stoich_widget_reset_prefix_covers_all_reaction_counts() -> None:
    prefixes = _build_stoich_widget_reset_prefixes(3)
    _assert_true(
        prefixes == ["nu_3_"],
        "计量数矩阵重置前缀应覆盖同一物种数下的所有反应数版本",
    )


def test_log_fit_validation_returns_error_for_bad_K0_ads_mask_shape() -> None:
    cfg = config_manager.get_default_config()
    cfg["kinetic_model"] = "langmuir_hinshelwood"
    cfg["species_text"] = "A,B"
    cfg["n_reactions"] = 1
    cfg["K0_ads"] = [1.0, 2.0]
    cfg["fit_K0_ads_flags"] = [True]
    cfg["use_log_K0_ads_fit"] = True

    ok, message = config_manager.validate_config(cfg)
    _assert_true(not ok, "非法的 fit_K0_ads_flags 尺寸应返回校验失败")
    _assert_true(
        "fit_K0_ads_flags" in str(message),
        "K0_ads 的掩码尺寸错误应返回可读错误信息，而不是抛异常",
    )


def test_log_fit_validation_returns_error_for_bad_k0_rev_mask_shape() -> None:
    cfg = config_manager.get_default_config()
    cfg["reversible_enabled"] = True
    cfg["species_text"] = "A,B"
    cfg["n_reactions"] = 2
    cfg["k0_rev"] = [1.0, 2.0]
    cfg["fit_k0_rev_flags"] = [True]
    cfg["use_log_k0_rev_fit"] = True

    ok, message = config_manager.validate_config(cfg)
    _assert_true(not ok, "非法的 fit_k0_rev_flags 尺寸应返回校验失败")
    _assert_true(
        "fit_k0_rev_flags" in str(message),
        "k0_rev 的掩码尺寸错误应返回可读错误信息，而不是抛异常",
    )


def test_fit_result_initial_guess_updates_remap_species_by_name() -> None:
    updates, message = _build_initial_guess_updates_from_fit_result(
        {
            "k0": [1.0],
            "ea_J_mol": [2.0],
            "reaction_order_matrix": [[10.0, 20.0]],
            "K0_ads": [0.1, 0.2],
            "Ea_K": [-1000.0, -2000.0],
        },
        species_names_current=["B", "A"],
        species_names_fit=["A", "B"],
        stoich_matrix_current=[[1.0], [-1.0]],
        stoich_matrix_fit=[[-1.0], [1.0]],
        n_reactions_current=1,
        kinetic_model_current="langmuir_hinshelwood",
        kinetic_model_fit="langmuir_hinshelwood",
        reversible_enabled_current=False,
        reversible_enabled_fit=False,
    )
    _assert_true(updates is not None, f"物种与 ν 同步重排时应允许回填参数：{message}")
    _assert_true(
        updates["order_guess"] == [[20.0, 10.0]],
        "反应级数应按物种名重排后再回填",
    )
    _assert_true(
        updates["K0_ads"] == [0.2, 0.1],
        "物种相关向量参数应按物种名重排后再回填",
    )


def test_fit_result_initial_guess_updates_reject_stoich_change() -> None:
    updates, message = _build_initial_guess_updates_from_fit_result(
        {
            "k0": [1.0],
            "ea_J_mol": [2.0],
            "reaction_order_matrix": [[1.0, 0.0]],
        },
        species_names_current=["A", "B"],
        species_names_fit=["A", "B"],
        stoich_matrix_current=[[-1.0], [1.0]],
        stoich_matrix_fit=[[1.0], [-1.0]],
        n_reactions_current=1,
        kinetic_model_current="power_law",
        kinetic_model_fit="power_law",
        reversible_enabled_current=False,
        reversible_enabled_fit=False,
    )
    _assert_true(updates is None, "化学计量数矩阵变化时不应允许回填旧拟合参数")
    _assert_true(
        "化学计量数矩阵" in str(message),
        "化学计量数变化时应返回明确提示",
    )


def test_fit_result_initial_guess_updates_allow_species_reorder() -> None:
    updates, message = _build_initial_guess_updates_from_fit_result(
        {
            "k0": [1.0],
            "ea_J_mol": [2.0],
            "reaction_order_matrix": [[10.0, 20.0]],
        },
        species_names_current=["B", "A"],
        species_names_fit=["A", "B"],
        stoich_matrix_current=[[1.0], [-1.0]],
        stoich_matrix_fit=[[-1.0], [1.0]],
        n_reactions_current=1,
        kinetic_model_current="power_law",
        kinetic_model_fit="power_law",
        reversible_enabled_current=False,
        reversible_enabled_fit=False,
    )
    _assert_true(updates is not None, f"物种重排且 ν 同步重排时应允许回填：{message}")
    _assert_true(
        updates["order_guess"] == [[20.0, 10.0]],
        "物种重排时应按当前物种顺序回填级数矩阵",
    )


def test_import_clears_fit_setup_widget_keys() -> None:
    _reset_session_state()
    st.session_state["base_params_2"] = {"edited_rows": {0: {"k₀": "1e7"}}}
    st.session_state["base_orders_2"] = {"edited_rows": {0: {"A": 2.0}}}
    st.session_state["lh_ads_3"] = {"edited_rows": {0: {"K₀,ads": "9.9"}}}
    st.session_state["rev_params_2"] = {"edited_rows": {0: {"k₀⁻": "1e2"}}}
    st.session_state["rev_orders_2"] = {"edited_rows": {0: {"A": 1.0}}}

    _clear_state_for_imported_config()

    for key in [
        "base_params_2",
        "base_orders_2",
        "lh_ads_3",
        "rev_params_2",
        "rev_orders_2",
    ]:
        _assert_true(key not in st.session_state, f"导入配置后应清理 {key}")


def test_parity_temperature_mapping_uses_row_labels() -> None:
    data_df = pd.DataFrame({"T_K": [300.0, 325.0]}, index=[1, 3])
    mapped = _map_parity_temperatures_by_row_index(data_df, pd.Series([1, 3, 7]))
    _assert_true(
        list(mapped.astype(float))[:2] == [300.0, 325.0],
        "奇偶校验图按温度着色时应按原始 DataFrame index 映射温度",
    )
    _assert_true(
        pd.isna(mapped.iloc[2]),
        "不存在的 row_index 应返回 NaN，而不是误取其它位置的温度",
    )


def test_hidden_model_sections_preserve_stored_fit_flags() -> None:
    cfg = {
        "fit_K0_ads_flags": [True, False],
        "fit_Ea_K_flags": [False, True],
        "fit_m_flags": [True],
        "fit_k0_rev_flags": [True],
        "fit_ea_rev_flags": [True],
        "fit_order_rev_flags_matrix": [[True, False]],
    }

    def get_cfg(key: str, default):
        return cfg.get(key, default)

    state = resolve_fit_parameter_state(
        get_cfg,
        species_names=["A", "B"],
        n_reactions=1,
        kinetic_model="power_law",
        reversible_enabled=False,
    )

    _assert_true(
        state["fit_K0_ads_flags"].tolist() == [True, False],
        "切换离开 L-H 模型时，不应擦除已保存的 K0_ads 拟合勾选",
    )
    _assert_true(
        state["fit_k0_rev_flags"].tolist() == [True],
        "临时关闭可逆反应时，不应擦除已保存的逆反应拟合勾选",
    )


def test_hidden_fit_flags_keep_new_slots_disabled_after_resize() -> None:
    cfg = {
        "fit_K0_ads_flags": [True],
        "fit_Ea_K_flags": [True],
        "fit_m_flags": [True],
        "fit_k0_rev_flags": [True],
        "fit_ea_rev_flags": [True],
    }

    def get_cfg(key: str, default):
        return cfg.get(key, default)

    state = resolve_fit_parameter_state(
        get_cfg,
        species_names=["A", "B"],
        n_reactions=2,
        kinetic_model="langmuir_hinshelwood",
        reversible_enabled=True,
    )

    _assert_true(
        state["fit_K0_ads_flags"].tolist() == [True, False],
        "新增的吸附参数默认不应继承旧勾选状态",
    )
    _assert_true(
        state["fit_Ea_K_flags"].tolist() == [True, False],
        "新增的吸附热参数默认不应继承旧勾选状态",
    )
    _assert_true(
        state["fit_m_flags"].tolist() == [True, False],
        "新增的抑制指数默认不应继承旧勾选状态",
    )
    _assert_true(
        state["fit_k0_rev_flags"].tolist() == [True, False],
        "新增的逆反应 k0 默认不应继承旧勾选状态",
    )
    _assert_true(
        state["fit_ea_rev_flags"].tolist() == [True, False],
        "新增的逆反应 Ea 默认不应继承旧勾选状态",
    )


def test_execution_layer_disables_inactive_fit_flags() -> None:
    effective_flags = derive_effective_fit_flags(
        {
            "fit_k0_flags": [True],
            "fit_ea_flags": [True],
            "fit_order_flags_matrix": [[True, False]],
            "fit_K0_ads_flags": [True, False],
            "fit_Ea_K_flags": [False, True],
            "fit_m_flags": [True],
            "fit_k0_rev_flags": [True],
            "fit_ea_rev_flags": [True],
            "fit_order_rev_flags_matrix": [[True, False]],
        },
        kinetic_model="power_law",
        reversible_enabled=False,
    )

    _assert_true(
        effective_flags["fit_K0_ads_flags"].tolist() == [False, False],
        "执行层在非 L-H 模型下应禁用吸附参数拟合",
    )
    _assert_true(
        effective_flags["fit_k0_rev_flags"].tolist() == [False],
        "执行层在关闭可逆反应时应禁用逆反应参数拟合",
    )


def test_fit_state_diff_detects_fit_problem_changes() -> None:
    common_kwargs = {
        "data_df": pd.DataFrame({"T_K": [300.0], "Cout_A_mol_m3": [1.0]}),
        "species_names": ["A"],
        "output_mode": "Cout",
        "output_species_list": ["A"],
        "stoich_matrix": [[-1.0]],
        "solver_method": "LSODA",
        "rtol": 1e-6,
        "atol": 1e-8,
        "reactor_type": "PFR",
        "kinetic_model": "power_law",
        "reversible_enabled": False,
        "pfr_flow_model": "liquid_const_vdot",
        "max_step_fraction": 0.0,
        "residual_type": "绝对残差",
        "use_log_k0_fit": True,
        "use_log_k0_rev_fit": False,
        "use_log_K0_ads_fit": False,
        "fit_k0_flags": [True],
        "fit_ea_flags": [True],
        "fit_order_flags_matrix": [[False]],
        "fit_K0_ads_flags": [False],
        "fit_Ea_K_flags": [False],
        "fit_m_flags": [False],
        "fit_k0_rev_flags": [False],
        "fit_ea_rev_flags": [False],
        "fit_order_rev_flags_matrix": [[False]],
        "k0_min": 1.0,
        "k0_max": 1e8,
        "ea_min": 0.0,
        "ea_max": 2e5,
        "ord_min": -2.0,
        "ord_max": 2.0,
        "K0_ads_min": 0.0,
        "K0_ads_max": 1e3,
        "Ea_K_min": -5e4,
        "Ea_K_max": 5e4,
        "k0_rev_min": 1e-6,
        "k0_rev_max": 1e6,
        "ea_rev_min_J_mol": 0.0,
        "ea_rev_max_J_mol": 2e5,
        "order_rev_min": -2.0,
        "order_rev_max": 2.0,
    }
    baseline = build_fit_state_snapshot(**common_kwargs)
    changed = build_fit_state_snapshot(
        **{
            **common_kwargs,
            "fit_k0_flags": [False],
            "k0_max": 1e6,
        }
    )
    reasons = describe_fit_state_differences(changed, baseline)
    _assert_true("待拟合参数选择已变化" in reasons, "拟合掩码变化后应标记结果过期")
    _assert_true("参数边界已变化" in reasons, "参数边界变化后应标记结果过期")


def main() -> None:
    test_import_keeps_csv_data()
    test_reset_clears_csv_data()
    test_invalid_export_config_returns_error()
    test_reversible_config_migration()
    test_fit_history_entry_is_recorded()
    test_reaction_equation_parser_supports_literal_species_names()
    test_reaction_equation_parser_supports_ionic_species_names()
    test_same_csv_reupload_resets_cached_editor_data()
    test_reupload_changes_data_editor_key_revision()
    test_stoich_widget_reset_prefix_covers_all_reaction_counts()
    test_log_fit_validation_returns_error_for_bad_K0_ads_mask_shape()
    test_log_fit_validation_returns_error_for_bad_k0_rev_mask_shape()
    test_fit_result_initial_guess_updates_remap_species_by_name()
    test_fit_result_initial_guess_updates_reject_stoich_change()
    test_fit_result_initial_guess_updates_allow_species_reorder()
    test_import_clears_fit_setup_widget_keys()
    test_parity_temperature_mapping_uses_row_labels()
    test_hidden_model_sections_preserve_stored_fit_flags()
    test_hidden_fit_flags_keep_new_slots_disabled_after_resize()
    test_execution_layer_disables_inactive_fit_flags()
    test_fit_state_diff_detects_fit_problem_changes()
    print("REGRESSION_STATE_VALIDATE: OK")


if __name__ == "__main__":
    main()
