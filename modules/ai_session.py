from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

import modules.config_manager as config_manager
from modules.export_config import build_export_config_from_ctx


ACTION_BLOCK_PATTERN = re.compile(
    r"<kinetics_action>\s*(\{.*?\})\s*</kinetics_action>",
    flags=re.DOTALL,
)

# 说明：智能会话第二阶段只允许修改“配置项”，不允许修改代码、上传文件或执行任意命令。
# 这里列出的 key 都是当前配置导出/导入链路已经支持校验的字段。
ALLOWED_CONFIG_PATCH_KEYS = {
    "reactor_type",
    "pfr_flow_model",
    "kinetic_model",
    "reversible_enabled",
    "solver_method",
    "rtol",
    "atol",
    "max_step_fraction",
    "species_text",
    "n_reactions",
    "stoich_matrix",
    "order_guess",
    "fit_order_flags_matrix",
    "k0_guess",
    "ea_guess_J_mol",
    "fit_k0_flags",
    "fit_ea_flags",
    "K0_ads",
    "Ea_K_J_mol",
    "fit_K0_ads_flags",
    "fit_Ea_K_flags",
    "m_inhibition",
    "fit_m_flags",
    "k0_rev",
    "ea_rev_J_mol",
    "fit_k0_rev_flags",
    "fit_ea_rev_flags",
    "order_rev",
    "fit_order_rev_flags_matrix",
    "output_mode",
    "output_species_list",
    "k0_min",
    "k0_max",
    "ea_min_J_mol",
    "ea_max_J_mol",
    "order_min",
    "order_max",
    "K0_ads_min",
    "K0_ads_max",
    "Ea_K_min",
    "Ea_K_max",
    "m_min",
    "m_max",
    "k0_rev_min",
    "k0_rev_max",
    "ea_rev_min_J_mol",
    "ea_rev_max_J_mol",
    "order_rev_min",
    "order_rev_max",
    "residual_type",
    "diff_step_rel",
    "max_nfev",
    "use_x_scale_jac",
    "use_log_k0_fit",
    "use_log_k0_rev_fit",
    "use_log_K0_ads_fit",
    "use_multi_start",
    "n_starts",
    "max_nfev_coarse",
    "random_seed",
}


def find_codex_cli() -> str | None:
    """
    查找本机 Codex CLI。

    说明：不在代码里写死 OpenAI API；此功能复用用户本机已经登录的 Codex CLI。
    若 PATH 中找不到 codex，则页面会给出清晰提示。
    """
    return shutil.which("codex")


def _json_safe(value: Any) -> Any:
    """
    将 numpy / pandas 常见对象转换成 JSON 可序列化对象。

    这样做的目的有两个：
    1. 发送给 Codex 的上下文足够清晰；
    2. 后续展示配置差异时不会因为 numpy 类型导致 json.dumps 报错。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if not isinstance(value, (dict, list, tuple, str)):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _json_text(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, default=str)


def _summarize_dataframe(data_df: Any) -> dict:
    if not isinstance(data_df, pd.DataFrame):
        return {"available": False}

    summary: dict[str, Any] = {
        "available": True,
        "rows": int(len(data_df)),
        "columns": [str(c) for c in data_df.columns],
        "preview_rows": data_df.head(8).to_dict(orient="records"),
        "numeric_columns": {},
    }
    for column_name in data_df.columns:
        values = pd.to_numeric(data_df[column_name], errors="coerce")
        finite_values = values[np.isfinite(values)]
        if finite_values.empty:
            continue
        summary["numeric_columns"][str(column_name)] = {
            "valid_count": int(finite_values.size),
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
        }
    return _json_safe(summary)


def _summarize_fit_results(fit_results: Any) -> dict:
    if not isinstance(fit_results, dict):
        return {"available": False}

    params = fit_results.get("params", {})
    param_summary: dict[str, Any] = {}
    if isinstance(params, dict):
        for key, value in params.items():
            arr = np.asarray(value, dtype=object)
            param_summary[str(key)] = {
                "shape": list(arr.shape),
                "values": _json_safe(value),
            }

    return _json_safe(
        {
            "available": True,
            "phi_initial": fit_results.get("phi_initial", fit_results.get("initial_cost")),
            "phi_final": fit_results.get("phi_final", fit_results.get("cost")),
            "residual_type": fit_results.get("residual_type"),
            "n_valid_points": fit_results.get("n_valid_points"),
            "n_fit_params": fit_results.get("n_fit_params"),
            "summary": fit_results.get("summary", ""),
            "params": param_summary,
        }
    )


def build_ai_snapshot(ctx: dict) -> tuple[dict, dict | None, str]:
    """
    构建发送给 Codex 的工程上下文。

    返回:
        snapshot:
            给模型阅读的精简上下文。
        current_config:
            当前完整配置。若当前页面状态无法组成合法导出配置，则返回 None。
        config_error:
            导出配置失败时的错误说明。
    """
    config_error = ""
    current_config: dict | None = None
    try:
        current_config = build_export_config_from_ctx(
            ctx,
            output_mode=str(ctx.get("output_mode", "")),
            output_species_list=list(ctx.get("output_species_list", [])),
        )
        ok, message = config_manager.validate_config(dict(current_config))
        if not ok:
            config_error = message
    except Exception as exc:
        config_error = str(exc)
        current_config = None

    data_df = ctx.get("data_df", None)
    fit_results = ctx.get("fit_results", None)

    snapshot = {
        "app": "Kinetics_app",
        "purpose": "反应动力学建模、参数拟合与拟合问题诊断",
        "current_config": current_config,
        "config_error": config_error,
        "data_summary": _summarize_dataframe(data_df),
        "fit_results_summary": _summarize_fit_results(fit_results),
        "allowed_config_patch_keys": sorted(ALLOWED_CONFIG_PATCH_KEYS),
    }
    return _json_safe(snapshot), current_config, config_error


def build_codex_prompt(
    *,
    user_message: str,
    chat_history: list[dict],
    snapshot: dict,
) -> str:
    history_for_prompt = chat_history[-12:]
    return (
        "你是 Kinetics_app 的本地智能建模助手。请用严谨、专业、简洁的中文回答。\n\n"
        "重要约束：\n"
        "1. 你只能基于下方快照诊断动力学建模、数据质量、参数边界、初值、残差定义和拟合策略。\n"
        "2. 不要修改文件，不要运行命令，不要假设可以直接访问外部 API。\n"
        "3. 若你认为可以自动应用一个安全的配置修改，请在回答末尾追加一个 JSON 操作块。\n"
        "4. JSON 操作块格式必须严格如下，且只允许出现一次：\n"
        "<kinetics_action>\n"
        "{\"summary\":\"一句话说明\", \"config_patch\":{}, \"run_fit_after_apply\": false}\n"
        "</kinetics_action>\n"
        "5. config_patch 只能包含 allowed_config_patch_keys 中的配置项；数组必须给完整数组，不要给局部索引。\n"
        "6. 如果没有足够依据自动修改配置，就不要输出 kinetics_action。\n\n"
        "当前工程快照 JSON：\n"
        f"{json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)}\n\n"
        "最近对话 JSON：\n"
        f"{json.dumps(history_for_prompt, ensure_ascii=False, indent=2, default=str)}\n\n"
        "用户本轮问题：\n"
        f"{user_message}\n"
    )


def call_codex_agent(
    *,
    prompt: str,
    cwd: str,
    timeout_s: int = 240,
) -> tuple[bool, str]:
    """
    通过本机 Codex CLI 调用模型。

    说明：
    - 使用 subprocess 参数列表，避免 shell 注入风险；
    - 使用 read-only sandbox，第一/二阶段只让模型“分析并提出配置补丁”，不让模型改代码；
    - 使用临时文件接收最终回答，避免 stdout 中的进度信息干扰解析。
    """
    codex_path = find_codex_cli()
    if not codex_path:
        return False, "未找到本机 codex 命令。请确认 Codex CLI 已安装且在 PATH 中。"

    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as tmp_file:
        output_path = tmp_file.name

    command = [
        codex_path,
        "exec",
        "--ephemeral",
        "--cd",
        cwd,
        "--sandbox",
        "read-only",
        "--output-last-message",
        output_path,
        prompt,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=int(timeout_s),
            check=False,
        )
        final_message = ""
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                final_message = f.read().strip()
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 1800:
                detail = detail[-1800:]
            return False, detail or "Codex CLI 调用失败，但未返回详细错误。"
        return True, final_message or (result.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return False, f"Codex CLI 在 {int(timeout_s)} 秒内未完成。请缩短问题或稍后重试。"
    except Exception as exc:
        return False, f"Codex CLI 调用异常：{exc}"
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def extract_action_block(message_text: str) -> tuple[str, dict | None, str]:
    """
    从 Codex 回复中提取可应用的配置操作。

    返回:
        clean_text:
            去掉 JSON 操作块后的自然语言回答。
        action:
            解析后的操作字典；没有操作块时为 None。
        error:
            操作块存在但解析失败时的错误说明。
    """
    match = ACTION_BLOCK_PATTERN.search(str(message_text))
    if not match:
        return str(message_text).strip(), None, ""

    clean_text = ACTION_BLOCK_PATTERN.sub("", str(message_text)).strip()
    try:
        action = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return clean_text, None, f"智能操作 JSON 解析失败：{exc}"
    if not isinstance(action, dict):
        return clean_text, None, "智能操作必须是 JSON object。"
    return clean_text, action, ""


def prepare_config_action(
    *,
    current_config: dict | None,
    action: dict,
) -> tuple[dict | None, str]:
    """
    将模型给出的 config_patch 转换成可导入配置。

    这里故意只做“配置合并 + validate_config”，不执行代码修改。
    这样第二阶段可以自动应用参数/模型设置，并复用现有拟合链路重新运行。
    """
    if current_config is None:
        return None, "当前配置不可用，无法应用智能修改。请先完成模型页的基本设置。"

    config_patch = action.get("config_patch", {})
    if not isinstance(config_patch, dict) or not config_patch:
        return None, "智能操作没有提供可应用的 config_patch。"

    unknown_keys = sorted(set(config_patch.keys()) - ALLOWED_CONFIG_PATCH_KEYS)
    if unknown_keys:
        return None, "config_patch 包含不允许自动修改的字段：" + "，".join(unknown_keys)

    merged_config = dict(current_config)
    for key, value in config_patch.items():
        merged_config[str(key)] = _json_safe(value)

    ok, message = config_manager.validate_config(merged_config)
    if not ok:
        return None, "智能修改后的配置未通过校验：" + str(message)

    return _json_safe(merged_config), ""


def build_config_diff_rows(
    *,
    old_config: dict,
    new_config: dict,
    patch_keys: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for key in patch_keys:
        old_value = old_config.get(key)
        new_value = new_config.get(key)
        rows.append(
            {
                "配置项": key,
                "当前值": _json_text(old_value),
                "建议值": _json_text(new_value),
            }
        )
    return rows
