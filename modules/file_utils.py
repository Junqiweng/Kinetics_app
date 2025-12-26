# 文件作用：提供原子文件写入等通用文件操作工具函数。

"""
文件操作工具模块

提供原子写入等安全的文件操作函数，避免写入过程中断导致数据损坏。
"""

from __future__ import annotations

import os
import tempfile


def atomic_write_text(file_path: str, text: str, encoding: str = "utf-8") -> None:
    """
    原子写入文本文件。

    先写入临时文件，成功后再原子替换目标文件，避免写入中断导致数据损坏。

    参数:
        file_path: 目标文件路径
        text: 要写入的文本内容
        encoding: 文件编码，默认 UTF-8
    """
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


def atomic_write_bytes(file_path: str, data: bytes) -> None:
    """
    原子写入二进制文件。

    先写入临时文件，成功后再原子替换目标文件，避免写入中断导致数据损坏。

    参数:
        file_path: 目标文件路径
        data: 要写入的二进制数据
    """
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
