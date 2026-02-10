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

from .constants import (
    DEFAULT_SESSION_MAX_AGE_HOURS,
    PERSIST_DIR_NAME,
    SECONDS_PER_HOUR,
    UUID_HYPHEN_COUNT,
    UUID_STRING_LENGTH,
)

# 持久化基础目录（从 constants 统一管理目录名）
_PERSIST_BASE_DIR = os.path.join(tempfile.gettempdir(), PERSIST_DIR_NAME)

# 默认最大会话年龄（小时）
DEFAULT_MAX_AGE_HOURS = DEFAULT_SESSION_MAX_AGE_HOURS

# 会话活跃标记文件名
_SESSION_ACTIVITY_MARKER = ".last_activity"


def update_session_activity(session_id: str) -> None:
    """
    更新会话的活动时间戳，防止被清理。

    应在每次页面加载或用户交互时调用，以标记会话仍在使用中。

    参数:
        session_id: 会话 ID（UUID 格式）
    """
    session_dir = os.path.join(_PERSIST_BASE_DIR, session_id)
    if not os.path.exists(session_dir):
        # 会话目录不存在，可能还未创建或已被清理
        return

    activity_marker_path = os.path.join(session_dir, _SESSION_ACTIVITY_MARKER)
    try:
        # 创建或更新标记文件的修改时间（touch 操作）
        with open(activity_marker_path, "a", encoding="utf-8"):
            pass
        # 更新文件访问和修改时间为当前时间
        os.utime(activity_marker_path, None)
    except Exception:
        # 忽略错误（例如权限问题），不影响正常功能
        pass


def cleanup_old_sessions(max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> int:
    """
    清理超过指定时间的旧会话目录。

    优先使用活动标记文件的时间来判断会话年龄。如果标记文件不存在，
    则回退到目录修改时间。这样可以避免误删长时间运行的活跃会话。

    参数:
        max_age_hours: 最大会话年龄（小时），超过此时间的会话目录将被删除

    返回:
        删除的目录数量
    """
    if not os.path.exists(_PERSIST_BASE_DIR):
        return 0

    deleted_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * SECONDS_PER_HOUR

    try:
        for item in os.listdir(_PERSIST_BASE_DIR):
            item_path = os.path.join(_PERSIST_BASE_DIR, item)

            # 只处理目录（会话目录是 UUID 格式）
            if not os.path.isdir(item_path):
                continue

            # 检查目录是否像会话 ID（UUID 格式）
            # UUID 格式: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 字符)
            if len(item) != UUID_STRING_LENGTH or item.count("-") != UUID_HYPHEN_COUNT:
                continue

            try:
                # 优先使用活动标记文件的修改时间，如果不存在则使用目录修改时间
                activity_marker_path = os.path.join(item_path, _SESSION_ACTIVITY_MARKER)
                if os.path.exists(activity_marker_path):
                    # 使用活动标记文件的最后修改时间
                    mtime = os.path.getmtime(activity_marker_path)
                else:
                    # 回退到目录的最后修改时间（兼容旧会话）
                    mtime = os.path.getmtime(item_path)

                age_seconds = current_time - mtime

                if age_seconds > max_age_seconds:
                    # 删除整个会话目录
                    shutil.rmtree(item_path)
                    deleted_count += 1
            except Exception:
                # 忽略单个目录的删除错误（例如权限问题或并发删除）
                pass

        # 向后兼容清理：历史版本可能在根目录写入 ls_cfg_chunks_*.json 分片缓存文件。
        for item in os.listdir(_PERSIST_BASE_DIR):
            item_path = os.path.join(_PERSIST_BASE_DIR, item)
            if not os.path.isfile(item_path):
                continue
            if not (item.startswith("ls_cfg_chunks_") and item.endswith(".json")):
                continue
            try:
                mtime = os.path.getmtime(item_path)
                age_seconds = current_time - mtime
                if age_seconds > max_age_seconds:
                    os.remove(item_path)
            except Exception:
                pass
    except Exception:
        # 忽略整体错误
        pass

    return deleted_count
