# 文件作用：清理过期的会话目录，避免磁盘空间占用过多。

"""
会话清理模块

定期清理超过指定时间的旧会话目录。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time


# 持久化基础目录
_PERSIST_BASE_DIR = os.path.join(tempfile.gettempdir(), "Kinetics_app_persist")

# 默认最大会话年龄（小时）
DEFAULT_MAX_AGE_HOURS = 24


def cleanup_old_sessions(max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> int:
    """
    清理超过指定时间的旧会话目录。

    参数:
        max_age_hours: 最大会话年龄（小时），超过此时间的会话目录将被删除

    返回:
        删除的目录数量
    """
    if not os.path.exists(_PERSIST_BASE_DIR):
        return 0

    deleted_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    try:
        for item in os.listdir(_PERSIST_BASE_DIR):
            item_path = os.path.join(_PERSIST_BASE_DIR, item)

            # 只处理目录（会话目录是 UUID 格式）
            if not os.path.isdir(item_path):
                continue

            # 检查目录是否像会话 ID（UUID 格式）
            # UUID 格式: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 字符)
            if len(item) != 36 or item.count("-") != 4:
                continue

            try:
                # 获取目录的最后修改时间
                mtime = os.path.getmtime(item_path)
                age_seconds = current_time - mtime

                if age_seconds > max_age_seconds:
                    # 删除整个会话目录
                    shutil.rmtree(item_path)
                    deleted_count += 1
            except Exception:
                # 忽略单个目录的删除错误
                pass
    except Exception:
        # 忽略整体错误
        pass

    return deleted_count


def get_session_count() -> int:
    """
    获取当前会话目录数量。

    返回:
        会话目录数量
    """
    if not os.path.exists(_PERSIST_BASE_DIR):
        return 0

    count = 0
    try:
        for item in os.listdir(_PERSIST_BASE_DIR):
            item_path = os.path.join(_PERSIST_BASE_DIR, item)
            if os.path.isdir(item_path) and len(item) == 36 and item.count("-") == 4:
                count += 1
    except Exception:
        pass

    return count
