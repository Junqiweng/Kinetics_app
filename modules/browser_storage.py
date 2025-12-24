"""
浏览器 LocalStorage 持久化模块

用于 Streamlit Cloud 等无法使用本地文件系统的环境。
通过注入 JavaScript 代码与浏览器 LocalStorage 交互。

实现原理：
1. 保存：通过 components.html 注入 JS，将配置写入 LocalStorage
2. 加载：通过 URL 参数传递标志，触发 JS 读取 LocalStorage 并存入隐藏的 text_input
"""

from __future__ import annotations

import json
import base64
import hashlib
import streamlit as st
import streamlit.components.v1 as components


# LocalStorage 键名常量
_LS_CONFIG_KEY = "kinetics_app_config_v1"


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

    Args:
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
        # Base64 编码以避免 JavaScript 字符串转义问题
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

    if ls_config_b64:
        # 已经通过 URL 传递了配置，解码并存储
        try:
            config_json = base64.b64decode(ls_config_b64).decode("utf-8")
            config_dict = json.loads(config_json)
            st.session_state["_browser_loaded_config"] = config_dict
            st.session_state["_ls_config_loaded"] = True

            # 清除 URL 参数（避免 URL 过长和重复加载）
            st.query_params.clear()
            return
        except Exception as e:
            # 解码失败，忽略
            pass

    # 注入 JavaScript 来读取 LocalStorage
    # 由于配置可能很大，使用截断的方式或者使用 postMessage
    js_code = f"""
    <script>
        (function() {{
            // 避免重复执行
            if (window._kinetics_ls_check_done) return;
            window._kinetics_ls_check_done = true;
            
            try {{
                // 首先检查 sessionStorage（同一标签页内的临时存储）
                let configJson = sessionStorage.getItem("{_LS_CONFIG_KEY}");
                
                // 如果 sessionStorage 没有，尝试 localStorage
                if (!configJson) {{
                    configJson = localStorage.getItem("{_LS_CONFIG_KEY}");
                    if (configJson) {{
                        // 复制到 sessionStorage
                        sessionStorage.setItem("{_LS_CONFIG_KEY}", configJson);
                    }}
                }}
                
                if (configJson) {{
                    // 将配置存储到 window 对象，供后续读取
                    window._kinetics_stored_config = configJson;
                    
                    // 标记配置已存在
                    const marker = document.createElement("div");
                    marker.id = "kinetics-config-marker";
                    marker.style.display = "none";
                    marker.setAttribute("data-has-config", "1");
                    document.body.appendChild(marker);
                }}
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

    Returns:
        配置字典，如果没有则返回 None
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

    Args:
        is_saved: 配置是否已保存
    """
    if is_saved:
        st.caption("✅ 配置已自动保存")
    else:
        st.caption("⏳ 正在保存配置...")


def get_browser_storage_key() -> str:
    """返回配置存储的键名。"""
    return _LS_CONFIG_KEY
