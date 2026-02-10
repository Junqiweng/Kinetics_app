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

    _clear_state_for_reset_default()

    _assert_true("uploaded_csv_bytes" not in st.session_state, "重置默认后应清空 uploaded_csv_bytes")
    _assert_true("uploaded_csv_name" not in st.session_state, "重置默认后应清空 uploaded_csv_name")
    _assert_true("data_df_cached" not in st.session_state, "重置默认后应清空 data_df_cached")
    _assert_true("fit_results" not in st.session_state, "重置默认后应清空旧拟合结果")
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


def main() -> None:
    test_import_keeps_csv_data()
    test_reset_clears_csv_data()
    test_invalid_export_config_returns_error()
    test_reversible_config_migration()
    print("REGRESSION_STATE_VALIDATE: OK")


if __name__ == "__main__":
    main()
