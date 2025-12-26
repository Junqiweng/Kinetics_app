# 文件作用：上传 CSV 的本地缓存与恢复（跨浏览器刷新保留最近一次上传内容）。
# 支持多用户会话隔离：每个会话 ID 使用独立的存储目录。

from __future__ import annotations

import io
import json
import os
import tempfile
import time

import pandas as pd

from .constants import PERSIST_DIR_NAME
from .file_utils import atomic_write_bytes, atomic_write_text

# 持久化基础目录（从 constants 统一管理目录名）
_PERSIST_BASE_DIR = os.path.join(tempfile.gettempdir(), PERSIST_DIR_NAME)


def _read_csv_bytes_cached(uploaded_bytes: bytes) -> pd.DataFrame:
    """
    从 bytes 读取 CSV。
    """
    data_df = pd.read_csv(io.BytesIO(uploaded_bytes))
    try:
        data_df.columns = [str(c).strip() for c in list(data_df.columns)]
    except Exception:
        pass
    return data_df


def _get_persist_dir(session_id: str | None = None) -> str:
    """
    本地持久化目录：用于跨刷新恢复。

    参数:
        session_id: 会话 ID，若提供则使用独立目录

    返回:
        持久化目录路径
    """
    if session_id:
        persist_dir = os.path.join(_PERSIST_BASE_DIR, session_id)
    else:
        persist_dir = _PERSIST_BASE_DIR
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def _get_upload_file_paths(session_id: str | None = None) -> tuple[str, str]:
    """
    返回:
        (csv_bytes_path, meta_json_path)

    参数:
        session_id: 会话 ID，用于多用户隔离
    """
    persist_dir = _get_persist_dir(session_id)
    csv_path = os.path.join(persist_dir, "uploaded.csv")
    meta_path = os.path.join(persist_dir, "uploaded.meta.json")
    return csv_path, meta_path


def _load_persisted_upload(
    session_id: str | None = None,
) -> tuple[bytes | None, str | None, str]:
    """
    加载已缓存的上传文件。

    参数:
        session_id: 会话 ID，用于多用户隔离

    返回:
        (uploaded_csv_bytes, uploaded_csv_name, message)
    """
    csv_path, meta_path = _get_upload_file_paths(session_id)
    if not os.path.exists(csv_path):
        return None, None, "未找到已缓存上传文件"

    try:
        with open(csv_path, "rb") as f:
            uploaded_bytes = f.read()
    except Exception as exc:
        return None, None, f"读取缓存 CSV 失败: {exc}"

    uploaded_name = ""
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            uploaded_name = str(meta.get("name", "")).strip()
        except Exception:
            uploaded_name = ""

    if not uploaded_bytes:
        return None, None, "缓存 CSV 为空"
    return uploaded_bytes, uploaded_name, "OK"


def _save_persisted_upload(
    uploaded_bytes: bytes, uploaded_name: str, session_id: str | None = None
) -> tuple[bool, str]:
    """
    保存上传文件到本地缓存。

    参数:
        uploaded_bytes: CSV 文件内容
        uploaded_name: 文件名
        session_id: 会话 ID，用于多用户隔离

    返回:
        (成功标志, 消息)
    """
    csv_path, meta_path = _get_upload_file_paths(session_id)
    try:
        atomic_write_bytes(csv_path, uploaded_bytes)
        meta = {
            "name": str(uploaded_name).strip(),
            "saved_at_unix_s": float(time.time()),
        }
        atomic_write_text(
            meta_path, json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True, "OK"
    except Exception as exc:
        return False, f"缓存上传文件失败: {exc}"


def _delete_persisted_upload(session_id: str | None = None) -> tuple[bool, str]:
    """
    删除已缓存的上传文件。

    参数:
        session_id: 会话 ID，用于多用户隔离

    返回:
        (成功标志, 消息)
    """
    csv_path, meta_path = _get_upload_file_paths(session_id)
    try:
        for path in [csv_path, meta_path]:
            if os.path.exists(path):
                os.remove(path)
        return True, "OK"
    except Exception as exc:
        return False, f"删除缓存上传文件失败: {exc}"
