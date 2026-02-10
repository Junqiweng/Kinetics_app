# 文件作用：通过浏览器 LocalStorage/sessionStorage 持久化配置和会话管理。

"""
浏览器存储持久化模块

用于 Streamlit Cloud 等无法使用本地文件系统的环境。
通过注入 JavaScript 代码与浏览器 LocalStorage/sessionStorage 交互。

实现原理：
1. 会话 ID：使用 sessionStorage 存储唯一会话 ID（刷新保留，关闭清除）
2. 保存配置：通过 components.html 注入 JS，将配置写入 LocalStorage
3. 加载配置：通过 URL 参数传递标志，触发 JS 读取 LocalStorage
"""

from __future__ import annotations

import os
import json
import base64
import hashlib
import tempfile
import uuid
import streamlit as st
import streamlit.components.v1 as components

from .constants import PERSIST_DIR_NAME
from .file_utils import atomic_write_text


# 浏览器存储键名常量
_LS_CONFIG_KEY = "kinetics_app_config_v1"
_SS_SESSION_KEY = "kinetics_session_id_v1"  # sessionStorage 会话 ID 键名
_SS_LS_CFG_LOADED_FLAG = (
    "kinetics_ls_cfg_loaded_v1"  # sessionStorage：已成功加载/不再尝试
)
_SS_LS_CFG_ATTEMPTED_FLAG = (
    "kinetics_ls_cfg_attempted_v1"  # sessionStorage：本标签页已尝试回传一次（避免循环）
)
_SS_LS_CFG_PAYLOAD_B64_KEY = "kinetics_ls_cfg_payload_b64_v1"  # sessionStorage：Base64(JSON)
_SS_LS_CFG_NEXT_PART_KEY = "kinetics_ls_cfg_next_part_v1"  # sessionStorage：下一个分片序号
_SS_LS_CFG_TOTAL_PARTS_KEY = "kinetics_ls_cfg_total_parts_v1"  # sessionStorage：总分片数

# URL 分片传输参数名（避免与业务参数冲突）
_QP_LS_CHUNK = "ls_chunk"
_QP_LS_PART = "ls_part"
_QP_LS_TOTAL = "ls_total"

# 单个 URL 参数在不同浏览器/代理上可能被截断；这里用保守分片长度，代价是需要多次 reload。
_LS_CFG_CHUNK_SIZE = 2500


def _normalize_uuid_string(value: object) -> str | None:
    """
    将输入规范化为 UUID 字符串（xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx）。

    返回:
        - 合法 UUID：返回规范化后的字符串（全小写，含 '-'）
        - 非法输入：返回 None

    目的：
        防止 URL 参数 sid 被用于目录拼接（路径穿越/非法字符等）。
    """
    try:
        text = str(value).strip()
        if not text:
            return None
        return str(uuid.UUID(text))
    except Exception:
        return None


def _get_persist_base_dir() -> str:
    """
    返回本机临时目录下的持久化根目录（用于跨刷新/跨 Streamlit session 的短期缓存）。

    注意：
    - 这里的缓存仅用于“分片回传”的临时聚合，最终配置仍以浏览器 localStorage 为准。
    - 该目录可能会被系统清理；因此只应承载短期数据。
    """
    return os.path.join(tempfile.gettempdir(), PERSIST_DIR_NAME)


def _get_ls_cfg_chunk_cache_path(session_id: str) -> str:
    """
    返回用于聚合 ls_chunk 分片的临时文件路径（按会话目录隔离）。

    设计目标：
    - 允许整页刷新导致的 Streamlit session 变化（st.session_state 清空）仍可继续累积分片
    - 缓存文件置于会话目录内，便于复用现有 TTL 清理策略
    """
    normalized_sid = _normalize_uuid_string(session_id)
    if normalized_sid is None:
        raise ValueError("无效的 session_id，无法构建分片缓存路径")
    session_dir = os.path.join(_get_persist_base_dir(), normalized_sid)
    os.makedirs(session_dir, exist_ok=True)
    return os.path.join(session_dir, "ls_cfg_chunks.json")


def _read_json_file_silent(file_path: str) -> dict | None:
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _remove_file_silent(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass


# ============ 会话 ID 管理 ============


def inject_session_id_script() -> None:
    """
    注入 JavaScript 脚本管理会话 ID。

    该脚本会：
    1. 检查 sessionStorage 中是否存在会话 ID
    2. 如果不存在，生成新的 UUID
    3. 将会话 ID 通过 URL 参数传递给 Streamlit

    特性：
    - sessionStorage 在刷新时保留
    - sessionStorage 在关闭标签页时清除
    """
    # 如果已经有会话 ID，跳过注入
    if st.session_state.get("_session_id_initialized", False):
        return

    # 检查 URL 参数中是否有会话 ID
    query_params = st.query_params
    session_id_from_url = query_params.get("sid", None)

    if session_id_from_url:
        if isinstance(session_id_from_url, (list, tuple)):
            session_id_from_url = session_id_from_url[0] if session_id_from_url else ""

        normalized_sid = _normalize_uuid_string(session_id_from_url)
        if normalized_sid is not None:
            # 从 URL 获取会话 ID（仅接受合法 UUID）
            st.session_state["_current_session_id"] = normalized_sid
            st.session_state["_session_id_initialized"] = True
            # 重要：不要在 Python 端修改 st.query_params（可能触发额外 rerun，导致“打开/刷新会多刷新一次”）。
            # 改为在前端用 history.replaceState 仅移除 sid 参数，不触发 reload。
            js_cleanup = f"""
            <script>
                (function() {{
                    try {{
                        if (window._kinetics_sid_cleanup_done) return;
                        window._kinetics_sid_cleanup_done = true;
                        const url = new URL(window.parent.location.href);
                        if (url.searchParams.has("sid")) {{
                            url.searchParams.delete("sid");
                            window.parent.history.replaceState(null, "", url.toString());
                        }}
                    }} catch (e) {{}}
                }})();
            </script>
            """
            components.html(js_cleanup, height=0, width=0)
            return

        # URL 中存在 sid 但不是合法 UUID：不信任该值，让前端脚本用 sessionStorage 生成/覆盖。

    # 注入 JavaScript 生成/读取会话 ID
    js_code = f"""
    <script>
        (function() {{
            // 避免重复执行
            if (window._kinetics_session_id_done) return;
            window._kinetics_session_id_done = true;
            
            const SESSION_KEY = "{_SS_SESSION_KEY}";
            
            try {{
                // 检查 sessionStorage 中是否有会话 ID
                let sessionId = sessionStorage.getItem(SESSION_KEY);
                
                if (!sessionId) {{
                    // 生成新的 UUID
                    sessionId = crypto.randomUUID ? crypto.randomUUID() : 
                        'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {{
                            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                            return v.toString(16);
                        }});
                    sessionStorage.setItem(SESSION_KEY, sessionId);
                }}
                
                // 将会话 ID 通过 URL 参数传递给 Streamlit
                const currentUrl = new URL(window.parent.location.href);
                const sidInUrl = currentUrl.searchParams.get('sid');
                // 关键：即使 URL 已有 sid（可能来自手动构造/历史链接），也要用 sessionStorage 的值覆盖，避免信任外部输入。
                if (sidInUrl !== sessionId) {{
                    currentUrl.searchParams.set('sid', sessionId);
                    window.parent.history.replaceState(null, '', currentUrl.toString());
                    // 触发 Streamlit 重新加载
                    window.parent.location.reload();
                }}
            }} catch (e) {{
                console.error("Session ID management error:", e);
            }}
        }})();
    </script>
    """
    components.html(js_code, height=0, width=0)


def get_current_session_id() -> str | None:
    """
    获取当前会话 ID。

    返回:
        会话 ID 字符串；若尚未初始化则返回 None
    """
    return st.session_state.get("_current_session_id", None)


def clear_session_id() -> None:
    """
    清除当前会话 ID（用于重置）。
    """
    js_code = f"""
    <script>
        (function() {{
            try {{
                sessionStorage.removeItem("{_SS_SESSION_KEY}");
            }} catch (e) {{
                console.error("Session ID clear error:", e);
            }}
        }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

    # 清除 session_state 中的会话 ID
    if "_current_session_id" in st.session_state:
        del st.session_state["_current_session_id"]
    if "_session_id_initialized" in st.session_state:
        del st.session_state["_session_id_initialized"]


# ============ 配置存储 ============


def _compute_config_hash(config_dict: dict) -> str:
    """计算配置的哈希值，用于检测变化。"""
    try:
        config_json = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def save_config_to_browser(config_dict: dict) -> None:
    """
    将配置保存到浏览器 LocalStorage。

    使用隐藏的 iframe 注入 JavaScript 代码。
    只有当配置发生变化时才真正保存。

    参数:
        config_dict: 配置字典（需要可 JSON 序列化）
    """
    try:
        # 计算配置哈希，避免重复保存相同的配置
        config_hash = _compute_config_hash(config_dict)
        last_hash = st.session_state.get("_browser_config_hash", "")

        if config_hash == last_hash:
            # 配置未变化，跳过保存
            return

        st.session_state["_browser_config_hash"] = config_hash

        config_json = json.dumps(config_dict, ensure_ascii=False)
        # 使用 Base64 编码以避免 JavaScript 字符串转义问题
        encoded_config = base64.b64encode(config_json.encode("utf-8")).decode("ascii")

        js_code = f"""
        <script>
            (function() {{
                try {{
                    const encodedConfig = "{encoded_config}";
                    const configJson = atob(encodedConfig);
                    localStorage.setItem("{_LS_CONFIG_KEY}", configJson);
                    // 同时保存到 sessionStorage，便于页面内快速读取
                    sessionStorage.setItem("{_LS_CONFIG_KEY}", configJson);
                }} catch (e) {{
                    console.error("LocalStorage auto-save error:", e);
                }}
            }})();
        </script>
        """
        components.html(js_code, height=0, width=0)
    except Exception as e:
        # 静默失败，不影响主应用
        pass


def inject_config_loader_script() -> None:
    """
    注入配置加载脚本。

    该脚本会在页面加载时尝试从 LocalStorage 读取配置，
    并通过 Streamlit 的 query_params 机制触发重新加载。
    """
    # 检查是否已经从 LocalStorage 加载过
    if st.session_state.get("_ls_config_loaded", False):
        return

    # 检查 URL 参数
    query_params = st.query_params
    ls_config_b64 = query_params.get("ls_cfg", None)
    ls_chunk = query_params.get(_QP_LS_CHUNK, None)
    ls_part = query_params.get(_QP_LS_PART, None)
    ls_total = query_params.get(_QP_LS_TOTAL, None)
    sid_from_url = query_params.get("sid", None)

    def _as_str_first(value) -> str:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value) if value is not None else ""

    def _b64_decode_utf8(b64_text: str) -> str:
        b64_text = str(b64_text)
        padding = (-len(b64_text)) % 4
        if padding:
            b64_text = b64_text + ("=" * padding)
        return base64.b64decode(b64_text).decode("utf-8")

    def _inject_cleanup_and_mark_loaded(extra_js: str = "") -> None:
        js_cleanup = f"""
        <script>
            (function() {{
                try {{
                    sessionStorage.setItem("{_SS_LS_CFG_LOADED_FLAG}", "1");
                    sessionStorage.removeItem("{_SS_LS_CFG_ATTEMPTED_FLAG}");
                    sessionStorage.removeItem("{_SS_LS_CFG_PAYLOAD_B64_KEY}");
                    sessionStorage.removeItem("{_SS_LS_CFG_NEXT_PART_KEY}");
                    sessionStorage.removeItem("{_SS_LS_CFG_TOTAL_PARTS_KEY}");
                    const url = new URL(window.parent.location.href);
                    url.searchParams.delete("ls_cfg");
                    url.searchParams.delete("{_QP_LS_CHUNK}");
                    url.searchParams.delete("{_QP_LS_PART}");
                    url.searchParams.delete("{_QP_LS_TOTAL}");
                    window.parent.history.replaceState(null, "", url.toString());
                }} catch (e) {{}}
                {extra_js}
            }})();
        </script>
        """
        components.html(js_cleanup, height=0, width=0)

    # --- 路径1：兼容旧版单参数回传（仅适用于小配置） ---
    if ls_config_b64:
        try:
            ls_config_b64 = _as_str_first(ls_config_b64).strip()
            config_json = _b64_decode_utf8(ls_config_b64)
            config_dict = json.loads(config_json)
            st.session_state["_browser_loaded_config"] = config_dict
            st.session_state["_ls_config_loaded"] = True
            _inject_cleanup_and_mark_loaded()
            return
        except Exception as exc:
            st.session_state["_ls_config_loaded"] = True
            st.session_state["_browser_loaded_config"] = None
            st.session_state["_browser_config_load_error"] = f"浏览器配置恢复失败：{exc}"
            _inject_cleanup_and_mark_loaded()
            return

    # --- 路径2：分片回传（支持大配置；会多次 reload，但不会卡死循环） ---
    if ls_chunk and ls_part and ls_total:
        try:
            ls_chunk_text = _as_str_first(ls_chunk)
            part_index = int(float(_as_str_first(ls_part)))
            total_parts = int(float(_as_str_first(ls_total)))
            if total_parts <= 0:
                raise ValueError("ls_total 无效")
            if part_index < 0 or part_index >= total_parts:
                raise ValueError("ls_part 超出范围")

            # 关键修复：
            # - 分片传输依赖多次整页 reload；在不少部署/浏览器场景下 reload 会导致 Streamlit session 变化，
            #   从而 st.session_state 无法跨刷新保存分片。
            # - 因此在 Python 端使用临时文件（系统 temp 目录）聚合分片，确保跨刷新仍可累积。
            #
            # 说明：文件仅用于“分片回传的聚合”，最终配置仍由 localStorage 持久化。
            session_id = get_current_session_id() or _as_str_first(sid_from_url).strip()
            normalized_sid = _normalize_uuid_string(session_id)
            if normalized_sid is None:
                # 如果该次运行还未拿到 sid（通常会在下一次 reload 带上），不处理以避免写入错误键。
                return

            cache_path = _get_ls_cfg_chunk_cache_path(normalized_sid)

            # 优先用“临时文件”聚合分片；若文件系统不可用，则回退到 st.session_state（仅同一 session 有效）。
            use_file_cache = True
            try:
                cache_data = _read_json_file_silent(cache_path) or {}
                cached_total = cache_data.get("total_parts", None)
                if (cached_total is not None) and (int(cached_total) != int(total_parts)):
                    # total_parts 不一致：视为新一轮传输，丢弃旧缓存，避免拼接错配
                    cache_data = {}

                chunks = cache_data.get("chunks", {})
                if not isinstance(chunks, dict):
                    chunks = {}
                chunks[str(int(part_index))] = str(ls_chunk_text)
                cache_data["total_parts"] = int(total_parts)
                cache_data["chunks"] = chunks
                atomic_write_text(
                    cache_path,
                    json.dumps(cache_data, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                use_file_cache = False
                chunks_int: dict[int, str] = st.session_state.get("_ls_cfg_chunks", {})
                if not isinstance(chunks_int, dict):
                    chunks_int = {}
                chunks_int[int(part_index)] = str(ls_chunk_text)
                st.session_state["_ls_cfg_chunks"] = chunks_int
                st.session_state["_ls_cfg_total_parts"] = int(total_parts)
                chunks = {str(k): str(v) for k, v in chunks_int.items()}

            if all(str(i) in chunks for i in range(total_parts)):
                payload_b64 = "".join([chunks[str(i)] for i in range(total_parts)])
                config_json = _b64_decode_utf8(payload_b64)
                config_dict = json.loads(config_json)
                st.session_state["_browser_loaded_config"] = config_dict
                st.session_state["_ls_config_loaded"] = True
                _inject_cleanup_and_mark_loaded()
                if use_file_cache:
                    _remove_file_silent(cache_path)
                return
        except Exception as exc:
            # 失败：阻断继续 reload，并提示
            st.session_state["_ls_config_loaded"] = True
            st.session_state["_browser_loaded_config"] = None
            st.session_state["_browser_config_load_error"] = f"浏览器配置恢复失败：{exc}"
            _inject_cleanup_and_mark_loaded()
            try:
                session_id = get_current_session_id() or _as_str_first(sid_from_url).strip()
                normalized_sid = _normalize_uuid_string(session_id)
                if normalized_sid is not None:
                    _remove_file_silent(_get_ls_cfg_chunk_cache_path(normalized_sid))
            except Exception:
                pass
            return

    # 注入 JavaScript 来读取 LocalStorage，并将其通过 URL 参数回传给 Streamlit。
    #
    # 说明：
    # - Streamlit 的 Python 端无法直接读取浏览器 localStorage/sessionStorage；
    #   因此需要在前端读取后，通过 query_params（URL 参数）触发一次 reload，
    #   让 Python 端在下一次运行时拿到 `ls_cfg` 并解码。
    # - 为避免无限 reload：若 URL 已包含 `ls_cfg`，则不再重复写入。
    js_code = f"""
    <script>
        (function() {{
            // 避免重复执行
            if (window._kinetics_ls_check_done) return;
            window._kinetics_ls_check_done = true;
            
            try {{
                // 若已成功加载/已尝试过，则不再触发（避免无限刷新）
                const LOADED_KEY = "{_SS_LS_CFG_LOADED_FLAG}";
                const ATTEMPT_KEY = "{_SS_LS_CFG_ATTEMPTED_FLAG}";
                const PAYLOAD_KEY = "{_SS_LS_CFG_PAYLOAD_B64_KEY}";
                const NEXT_KEY = "{_SS_LS_CFG_NEXT_PART_KEY}";
                const TOTAL_KEY = "{_SS_LS_CFG_TOTAL_PARTS_KEY}";
                const CHUNK_SIZE = {int(_LS_CFG_CHUNK_SIZE)};
                if (sessionStorage.getItem(LOADED_KEY) === "1") {{
                    return;
                }}
                const currentUrl = new URL(window.parent.location.href);
                // 若 URL 已包含分片参数（本次加载正在传输某一片），不要在同一轮重复写 URL；
                // 但可以在下一轮（reload 后）继续推进。
                const hasChunkParam = currentUrl.searchParams.has("{_QP_LS_CHUNK}") &&
                                      currentUrl.searchParams.has("{_QP_LS_PART}") &&
                                      currentUrl.searchParams.has("{_QP_LS_TOTAL}");
                if (currentUrl.searchParams.has('ls_cfg')) {{
                    return;
                }}

                // 若已有 payload（Base64），复用；否则从 storage 读取并编码一次
                let payloadB64 = sessionStorage.getItem(PAYLOAD_KEY);
                let totalParts = parseInt(sessionStorage.getItem(TOTAL_KEY) || "0");
                if (!payloadB64) {{
                    // 首先检查 sessionStorage（同一标签页内的临时存储）
                    let configJson = sessionStorage.getItem("{_LS_CONFIG_KEY}");
                    // 如果 sessionStorage 没有，尝试 localStorage
                    if (!configJson) {{
                        configJson = localStorage.getItem("{_LS_CONFIG_KEY}");
                        if (configJson) {{
                            sessionStorage.setItem("{_LS_CONFIG_KEY}", configJson);
                        }}
                    }}
                    if (!configJson) {{
                        return;
                    }}

                    window._kinetics_stored_config = configJson;
                    payloadB64 = btoa(unescape(encodeURIComponent(configJson)));
                    if (!payloadB64) {{
                        sessionStorage.setItem(ATTEMPT_KEY, "1");
                        return;
                    }}
                    totalParts = Math.ceil(payloadB64.length / CHUNK_SIZE);
                    sessionStorage.setItem(PAYLOAD_KEY, payloadB64);
                    sessionStorage.setItem(TOTAL_KEY, String(totalParts));
                    sessionStorage.setItem(NEXT_KEY, "0");
                }}

                if (!totalParts || totalParts <= 0) {{
                    sessionStorage.setItem(ATTEMPT_KEY, "1");
                    return;
                }}

                // 如果当前 URL 已带分片参数，视作“本次加载正在传输该分片”，推进 next_part
                if (hasChunkParam) {{
                    const curPart = parseInt(currentUrl.searchParams.get("{_QP_LS_PART}") || "-1");
                    const curTotal = parseInt(currentUrl.searchParams.get("{_QP_LS_TOTAL}") || "0");
                    if (curTotal === totalParts && curPart >= 0) {{
                        sessionStorage.setItem(NEXT_KEY, String(curPart + 1));
                    }}
                }}

                let nextPart = parseInt(sessionStorage.getItem(NEXT_KEY) || "0");
                if (nextPart < 0) nextPart = 0;
                if (nextPart >= totalParts) {{
                    // 已发送完所有分片，等待 Python 端解码成功并写入 LOADED_KEY
                    sessionStorage.setItem(ATTEMPT_KEY, "1");
                    return;
                }}

                // 发送下一分片：写入 URL 参数并 reload
                const start = nextPart * CHUNK_SIZE;
                const end = Math.min(payloadB64.length, (nextPart + 1) * CHUNK_SIZE);
                const chunk = payloadB64.slice(start, end);
                if (!chunk) {{
                    sessionStorage.setItem(ATTEMPT_KEY, "1");
                    return;
                }}

                // 标记：已进入回传流程（避免其它逻辑误触发）
                sessionStorage.setItem(ATTEMPT_KEY, "1");

                currentUrl.searchParams.set("{_QP_LS_PART}", String(nextPart));
                currentUrl.searchParams.set("{_QP_LS_TOTAL}", String(totalParts));
                currentUrl.searchParams.set("{_QP_LS_CHUNK}", chunk);
                window.parent.history.replaceState(null, '', currentUrl.toString());
                window.parent.location.reload();
            }} catch (e) {{
                console.error("LocalStorage check error:", e);
            }}
        }})();
    </script>
    """
    components.html(js_code, height=0, width=0)


def get_browser_loaded_config() -> dict | None:
    """
    获取从浏览器加载的配置。

    返回:
        配置字典；若没有则返回 None
    """
    return st.session_state.get("_browser_loaded_config", None)


def clear_browser_config() -> None:
    """
    清除浏览器中保存的配置。
    """
    js_code = f"""
    <script>
        (function() {{
            try {{
                localStorage.removeItem("{_LS_CONFIG_KEY}");
                sessionStorage.removeItem("{_LS_CONFIG_KEY}");
                sessionStorage.removeItem("{_SS_LS_CFG_LOADED_FLAG}");
                sessionStorage.removeItem("{_SS_LS_CFG_ATTEMPTED_FLAG}");
            }} catch (e) {{
                console.error("LocalStorage clear error:", e);
            }}
        }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

    # 清除 session state 中的相关标记
    if "_browser_config_hash" in st.session_state:
        del st.session_state["_browser_config_hash"]
    if "_browser_loaded_config" in st.session_state:
        del st.session_state["_browser_loaded_config"]
    if "_ls_config_loaded" in st.session_state:
        del st.session_state["_ls_config_loaded"]


def render_config_sync_indicator(is_saved: bool = True) -> None:
    """
    显示配置同步状态指示器。

    参数:
        is_saved: 配置是否已保存
    """
    if is_saved:
        st.caption("✅ 配置已自动保存")
    else:
        st.caption("⏳ 正在保存配置...")


def get_browser_storage_key() -> str:
    """返回配置存储的键名。"""
    return _LS_CONFIG_KEY
