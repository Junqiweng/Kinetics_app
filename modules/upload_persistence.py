from __future__ import annotations

import io
import json
import os
import tempfile
import time

import pandas as pd


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


def _get_persist_dir() -> str:
    """
    本地持久化目录：用于跨刷新恢复（只保留一份缓存，新内容覆盖旧内容）。
    """
    persist_dir = os.path.join(tempfile.gettempdir(), "Kinetics_app_persist")
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def _atomic_write_bytes(file_path: str, data: bytes) -> None:
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="tmp_", suffix=".bin", dir=dir_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temp_path, file_path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _atomic_write_text(file_path: str, text: str, encoding: str = "utf-8") -> None:
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="tmp_", suffix=".txt", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(temp_path, file_path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _get_upload_file_paths() -> tuple[str, str]:
    """
    Returns:
        (csv_bytes_path, meta_json_path)

    说明：只保留“一份”上传缓存，新内容覆盖旧内容。
    """
    persist_dir = _get_persist_dir()
    csv_path = os.path.join(persist_dir, "uploaded.csv")
    meta_path = os.path.join(persist_dir, "uploaded.meta.json")
    return csv_path, meta_path


def _load_persisted_upload() -> tuple[bytes | None, str | None, str]:
    """
    Returns:
        (uploaded_csv_bytes, uploaded_csv_name, message)
    """
    csv_path, meta_path = _get_upload_file_paths()
    if not os.path.exists(csv_path):
        return None, None, "未找到已缓存上传文件"

    try:
        uploaded_bytes = open(csv_path, "rb").read()
    except Exception as exc:
        return None, None, f"读取缓存 CSV 失败: {exc}"

    uploaded_name = ""
    if os.path.exists(meta_path):
        try:
            meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
            uploaded_name = str(meta.get("name", "")).strip()
        except Exception:
            uploaded_name = ""

    if not uploaded_bytes:
        return None, None, "缓存 CSV 为空"
    return uploaded_bytes, uploaded_name, "OK"


def _save_persisted_upload(uploaded_bytes: bytes, uploaded_name: str) -> tuple[bool, str]:
    csv_path, meta_path = _get_upload_file_paths()
    try:
        _atomic_write_bytes(csv_path, uploaded_bytes)
        meta = {
            "name": str(uploaded_name).strip(),
            "saved_at_unix_s": float(time.time()),
        }
        _atomic_write_text(
            meta_path, json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True, "OK"
    except Exception as exc:
        return False, f"缓存上传文件失败: {exc}"


def _delete_persisted_upload() -> tuple[bool, str]:
    csv_path, meta_path = _get_upload_file_paths()
    try:
        for path in [csv_path, meta_path]:
            if os.path.exists(path):
                os.remove(path)
        return True, "OK"
    except Exception as exc:
        return False, f"删除缓存上传文件失败: {exc}"
