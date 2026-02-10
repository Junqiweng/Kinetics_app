# 文件作用：集中维护 UI 文案与标签映射，保证主界面术语表达一致、专业。

from __future__ import annotations

from .constants import (
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODEL_POWER_LAW,
    KINETIC_MODEL_REVERSIBLE,
    OUTPUT_MODE_COUT,
    OUTPUT_MODE_FOUT,
    OUTPUT_MODE_XOUT,
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P,
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT,
    REACTOR_TYPE_BSTR,
    REACTOR_TYPE_CSTR,
    REACTOR_TYPE_PFR,
)


REACTOR_TYPE_LABELS = {
    REACTOR_TYPE_PFR: "PFR（平推流反应器）",
    REACTOR_TYPE_CSTR: "CSTR（连续搅拌釜）",
    REACTOR_TYPE_BSTR: "BSTR（间歇搅拌釜）",
}

PFR_FLOW_MODEL_LABELS = {
    PFR_FLOW_MODEL_LIQUID_CONST_VDOT: "液相（恒定体积流量）",
    PFR_FLOW_MODEL_GAS_IDEAL_CONST_P: "气相（理想气体、恒压、无压降）",
}

KINETIC_MODEL_LABELS = {
    KINETIC_MODEL_POWER_LAW: "幂律模型（Power-law）",
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD: "Langmuir-Hinshelwood 模型（L-H）",
    KINETIC_MODEL_REVERSIBLE: "可逆反应模型（Reversible）",
}

SOLVER_METHOD_LABELS = {
    "LSODA": "LSODA（自动刚性检测，推荐）",
    "RK45": "RK45（显式 Runge-Kutta）",
    "BDF": "BDF（后向差分）",
    "Radau": "Radau（隐式 Runge-Kutta）",
}

OUTPUT_MODE_LABELS = {
    OUTPUT_MODE_COUT: "出口浓度（Cout）[mol/m^3]",
    OUTPUT_MODE_FOUT: "出口摩尔流率（Fout）[mol/s]",
    OUTPUT_MODE_XOUT: "出口摩尔分率（xout）[-]",
}

PROFILE_KIND_LABELS = {
    "F (mol/s)": "摩尔流率 $F_i$ (mol/s)",
    "C (mol/m^3)": "浓度 $C_i$ (mol/m^3)",
}

# 说明：以下字符串用于 Matplotlib 轴标签，采用 mathtext 以保证上下标显示一致。
AXIS_LABEL_MEASURED = "Measured $y_i$"
AXIS_LABEL_PREDICTED = "Predicted $y_i$"
AXIS_LABEL_RESIDUAL = "Residual $r_i$"
AXIS_LABEL_REACTOR_VOLUME = r"Reactor volume $V$ (m$^3$)"
AXIS_LABEL_TIME = r"Time $t$ (s)"
AXIS_LABEL_CONCENTRATION = r"Concentration $C_i$ (mol/m$^3$)"
AXIS_LABEL_FLOW_RATE = r"Molar flow rate $F_i$ (mol/s)"


def map_label(mapping: dict[str, str], value: str) -> str:
    """按映射表返回显示文案；缺失时回退为原值。"""
    key = str(value)
    return str(mapping.get(key, key))


def kinetic_model_display_text(kinetic_model: str, reversible_enabled: bool) -> str:
    """
    生成动力学显示文案（基础模型 + 可逆选项）。

    兼容历史结果：若 kinetic_model=reversible，则按“幂律 + 可逆”展示。
    """
    model_value = str(kinetic_model).strip()
    if model_value == KINETIC_MODEL_REVERSIBLE:
        model_value = KINETIC_MODEL_POWER_LAW
        reversible_enabled = True

    base_text = map_label(KINETIC_MODEL_LABELS, model_value)
    if bool(reversible_enabled):
        return f"{base_text} + 可逆反应"
    return base_text


def format_plot_unit(unit_text: str) -> str:
    """
    将单位字符串转换为 Matplotlib 友好的 mathtext 形式。

    示例:
      - mol/m^3 -> mol/m$^3$
      - mol/m³  -> mol/m$^3$
    """
    unit = str(unit_text).strip()
    if unit in ("mol/m^3", "mol/m³"):
        return r"mol/m$^3$"
    return unit


def axis_label_with_unit(base_label: str, unit_text: str) -> str:
    """拼接“轴标签 + 单位”。若单位为 '-' 或空则不追加单位。"""
    unit = format_plot_unit(unit_text)
    if (not unit) or (unit == "-"):
        return str(base_label)
    return f"{base_label} [{unit}]"
