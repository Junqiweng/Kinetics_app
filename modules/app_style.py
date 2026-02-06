# 文件作用：定义应用的全局 CSS（Streamlit UI 样式）以及 Matplotlib 绘图风格。

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from .constants import PLOT_FONT_WEIGHT_SEMIBOLD, PLOT_SCI_POWERLIMITS


APP_CSS = """
<style>
:root {
  --app-bg: #F5F5F7;
  --app-bg-grad-top: #FAFAFB;
  --app-bg-grad-bottom: #F3F3F5;
  --card-bg: #FFFFFF;
  --card-bg-elevated: #FCFCFD;
  --text: #1D1D1F;
  --muted: #6E6E73;
  --border: rgba(0, 0, 0, 0.10);
  --border-strong: rgba(0, 0, 0, 0.16);
  --accent: #007AFF;
  --accent-strong: #0066D6;
  --accent-soft: rgba(0, 122, 255, 0.14);
  --radius: 16px;
  --radius-input: 12px;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 10px 30px rgba(0, 0, 0, 0.04);
  --shadow-soft: 0 1px 1px rgba(0, 0, 0, 0.04);
  --focus-ring: 0 0 0 3px var(--accent-soft);
}

/* Apple 式极简：留白、弱边框、轻阴影、系统字体 */
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
               "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  color: var(--text);
  font-size: 15px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

* { box-sizing: border-box; }

.kinetics-card-marker { display: none; }

::selection {
  background: rgba(0, 122, 255, 0.18);
  color: var(--text);
}

/* App 背景 */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 540px at 0% -12%, rgba(255, 255, 255, 0.64), transparent 70%),
    linear-gradient(180deg, var(--app-bg-grad-top) 0%, var(--app-bg-grad-bottom) 100%);
}

/* 主容器：控制最大宽度，让排版更“工整” */
.block-container {
  max-width: 1280px;
  padding-top: 1.25rem;
  padding-bottom: 3.5rem;
  padding-left: clamp(0.95rem, 1.4vw, 1.65rem);
  padding-right: clamp(0.95rem, 1.4vw, 1.65rem);
}

/* 顶部 Header：轻薄半透明 */
[data-testid="stHeader"] {
  background: rgba(245, 245, 247, 0.82);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px);
  box-shadow: inset 0 -1px 0 rgba(255, 255, 255, 0.52);
}

/* 侧边栏：干净白底 + 细分割线 */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #FFFFFF 0%, var(--card-bg-elevated) 100%);
  border-right: 1px solid var(--border);
  box-shadow: inset -1px 0 0 rgba(0, 0, 0, 0.03);
}

[data-testid="stSidebar"] .block-container {
  padding-top: 1rem;
  padding-bottom: 1.5rem;
}

/* 标题：去掉渐变，使用更克制的层级 */
h1 {
  font-size: 2.0rem !important;
  font-weight: 650 !important;
  color: var(--text) !important;
  letter-spacing: -0.02em;
  line-height: 1.18;
  margin-bottom: 0.35rem !important;
}

h2, h3, h4 {
  letter-spacing: -0.01em;
}

h2 {
  font-size: 1.25rem !important;
  font-weight: 650 !important;
  color: var(--text);
  line-height: 1.3;
  margin-top: 1.25rem !important;
  margin-bottom: 0.6rem !important;
}

h3 {
  font-size: 1.08rem !important;
  font-weight: 650 !important;
  color: var(--text);
  line-height: 1.35;
  margin-top: 1.0rem !important;
  margin-bottom: 0.5rem !important;
}

h4 {
  font-size: 1.0rem !important;
  font-weight: 650 !important;
  color: var(--muted);
  line-height: 1.4;
}

/* 说明文字 */
.stMarkdown p, .stMarkdown li {
  line-height: 1.65;
  color: var(--text);
}

.stMarkdown a {
  color: var(--accent);
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}

.stMarkdown a:hover {
  color: var(--accent-strong);
}

.stCaption {
  font-size: 0.88rem !important;
  color: var(--muted) !important;
  line-height: 1.45 !important;
  margin-bottom: 0.35rem !important;
}

/* 仅对“显式 border=True 的容器”做卡片化（避免误伤其它控件布局） */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--radius);
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.kinetics-card-marker) {
  background: var(--card-bg);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-soft);
  padding: 0.9rem 0.95rem;
  position: relative;
  overflow: hidden;
  transition: border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.kinetics-card-marker)::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0.88), rgba(0, 0, 0, 0.03), rgba(255, 255, 255, 0.88));
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.kinetics-card-marker):hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow);
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.kinetics-card-marker) > div {
  padding: 0 !important;
}

/* Tabs：做成接近 iOS 的 segmented control */
.stTabs [data-baseweb="tab-list"] {
  background: linear-gradient(180deg, #FFFFFF 0%, #F9F9FB 100%);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px;
  gap: 4px;
  box-shadow: var(--shadow-soft);
  margin-bottom: 0.8rem;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  padding: 0.55rem 0.95rem;
  font-weight: 650;
  color: var(--muted);
  transition: background 120ms ease, color 120ms ease, box-shadow 120ms ease;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(0, 0, 0, 0.04);
  color: var(--text);
}

.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #ffffff !important;
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.08);
}

/* 按钮：统一圆角与主色，减少夸张位移效果 */
button[data-testid^="baseButton-"] {
  border-radius: var(--radius-input) !important;
  font-weight: 600 !important;
  min-height: 2.4rem;
  transition: background 120ms ease, box-shadow 120ms ease, border-color 120ms ease, transform 120ms ease;
}

button[data-testid^="baseButton-"]:hover {
  transform: translateY(-1px);
}

button[data-testid^="baseButton-"]:active {
  transform: translateY(0);
}

button[data-testid^="baseButton-"]:focus-visible {
  outline: none !important;
  box-shadow: var(--focus-ring) !important;
}

button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-tertiary"] {
  background: linear-gradient(180deg, #FFFFFF 0%, #F9F9FB 100%) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  box-shadow: none !important;
}

button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-tertiary"]:hover {
  border-color: var(--border-strong) !important;
  box-shadow: var(--shadow-soft) !important;
}

button[data-testid="baseButton-primary"] {
  background: var(--accent) !important;
  border: 1px solid transparent !important;
  color: #ffffff !important;
}

button[data-testid="baseButton-primary"]:hover {
  background: var(--accent-strong) !important;
  box-shadow: var(--shadow-soft) !important;
}

/* TextInput：外层容器保持边框，内层 input 去掉边框以避免双边框叠加 */
/* 通用：清除所有输入控件的外层边框 */
[data-baseweb="base-input"] {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

/* 清除 Streamlit 自动生成的 emotion-cache 容器边框（NumberInput 外层白边框来源） */
div[data-testid="stNumberInput"] > div,
div[data-testid="stTextInput"] > div,
div[data-testid="stTextArea"] > div {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

/* 清除所有 emotion-cache 容器可能的边框 */
div[data-testid="stNumberInput"] [class*="emotion-cache"],
div[data-testid="stTextInput"] [class*="emotion-cache"],
div[data-testid="stTextArea"] [class*="emotion-cache"] {
  border: none !important;
  box-shadow: none !important;
}

div[data-testid="stTextInput"] [data-baseweb="input"] {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-input) !important;
  box-shadow: none !important;
  overflow: hidden;
  transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

div[data-testid="stTextInput"] input {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}

div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
  border-color: rgba(0, 122, 255, 0.55) !important;
  box-shadow: var(--focus-ring) !important;
  outline: none !important;
}

/* TextArea：同样处理双边框问题 */
div[data-testid="stTextArea"] [data-baseweb="textarea"] {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-input) !important;
  box-shadow: none !important;
  overflow: hidden;
  transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

div[data-testid="stTextArea"] textarea {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}

div[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within {
  border-color: rgba(0, 122, 255, 0.55) !important;
  box-shadow: var(--focus-ring) !important;
  outline: none !important;
}

/* NumberInput（含 +/- stepper）：避免“内外两层边框”叠加产生重影 */
div[data-testid="stNumberInput"] [data-baseweb="input"] {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-input) !important;
  box-shadow: none !important;
  overflow: hidden;
  transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

div[data-testid="stNumberInput"] input {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}

div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
  border-color: rgba(0, 122, 255, 0.55) !important;
  box-shadow: var(--focus-ring) !important;
  outline: none !important;
}

div[data-testid="stNumberInput"] [data-baseweb="button"] {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

div[data-testid="stNumberInput"] [data-baseweb="button"]:hover {
  background: rgba(0, 0, 0, 0.04) !important;
}

/* NumberInput：隐藏右侧 +/-（stepper）按钮，改为纯手动输入 */
div[data-testid="stNumberInput"] [data-baseweb="end-enhancer"],
div[data-testid="stNumberInput"] [data-baseweb="button"],
div[data-testid="stNumberInput"] button {
  display: none !important;
}

/* 部分浏览器会给 <input type="number"> 自带滚轮/上下箭头，这里一并去掉 */
div[data-testid="stNumberInput"] input[type="number"]::-webkit-outer-spin-button,
div[data-testid="stNumberInput"] input[type="number"]::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

div[data-testid="stNumberInput"] input[type="number"] {
  -moz-appearance: textfield;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-input) !important;
  transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
  border-color: rgba(0, 122, 255, 0.55) !important;
  box-shadow: var(--focus-ring) !important;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
  color: rgba(110, 110, 115, 0.78) !important;
}

/* 表格/数据编辑器：圆角 + 细边框 */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-soft);
  background: var(--card-bg);
  transition: border-color 120ms ease, box-shadow 120ms ease;
}

div[data-testid="stDataFrame"]:hover,
div[data-testid="stDataEditor"]:hover {
  border-color: var(--border-strong);
}

div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stDataEditor"] [role="columnheader"] {
  background: #F7F7F9 !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}

div[data-testid="stDataEditor"] [role="row"]:hover [role="gridcell"] {
  background: rgba(0, 122, 255, 0.03) !important;
}

/* Expander：同卡片风格 */
:where(div, details)[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--card-bg) !important;
  box-shadow: var(--shadow-soft) !important;
  overflow: hidden; /* 避免内部元素的边框/描边“叠加”到外框上，看起来像两层边框 */
  transition: border-color 120ms ease, box-shadow 120ms ease;
}

:where(div, details)[data-testid="stExpander"]:hover {
  border-color: var(--border-strong) !important;
}

/* Streamlit expander 内部常会再包一层 details/summary，自带边框/outline；这里统一去掉，避免双边框交叉 */
:where(div, details)[data-testid="stExpander"] details {
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}

:where(div, details)[data-testid="stExpander"] summary {
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: linear-gradient(180deg, #FFFFFF 0%, #FAFAFB 100%);
  padding: 0.7rem 0.9rem;
}

:where(div, details)[data-testid="stExpander"][open] > summary,
:where(div, details)[data-testid="stExpander"] details[open] > summary {
  border-bottom: 1px solid var(--border) !important;
}

:where(div, details)[data-testid="stExpander"] summary:focus-visible {
  outline: 3px solid rgba(0, 122, 255, 0.18) !important;
  outline-offset: 3px !important;
}

/* 让 expander 标题回归“正文级别”大小与颜色（避免异常放大/变色） */
:where(div, details)[data-testid="stExpander"] summary p,
:where(div, details)[data-testid="stExpander"] summary span {
  color: var(--text) !important;
  font-weight: 650 !important;
  font-size: 1.05rem !important;
  margin: 0 !important;
}

/* 对话框：更像系统弹窗 */
div[role="dialog"][aria-modal="true"] {
  width: 50vw !important;
  max-width: 1280px !important;
  border-radius: 18px !important;
}

/* 分隔线：细且克制 */
hr {
  margin: 1.25rem 0 !important;
  border-color: var(--border) !important;
}

/* 信息框：统一圆角 */
.stInfo, .stSuccess, .stWarning, .stError {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
}

/* 代码块：更紧凑 */
code {
  font-size: 0.9rem !important;
  border-radius: 8px !important;
  background: #F2F2F7 !important;
  padding: 0.11rem 0.36rem;
}

pre code {
  padding: 0 !important;
}

/* 交互细节增强 */
input[type="checkbox"],
input[type="radio"] {
  accent-color: var(--accent);
}

div[data-testid="stProgressBar"] > div > div > div > div {
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
}

/* 复用样式：拟合面板的小标签与概览卡片 */
.kinetics-inline-label {
  font-size: 0.8rem;
  color: var(--muted);
  white-space: nowrap;
  line-height: 1.2;
}

.kinetics-overview-box {
  background: linear-gradient(180deg, #FFFFFF 0%, #FBFBFC 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 14px 16px;
}

.kinetics-overview-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.kinetics-overview-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--accent);
}

.kinetics-overview-title {
  font-weight: 650;
  color: var(--text);
  letter-spacing: -0.01em;
}

.kinetics-overview-list {
  margin: 0 0 0 18px;
  color: var(--text);
  line-height: 1.65;
}

/* 滚动条细节 */
*::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

*::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.04);
}

*::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.18);
  border-radius: 999px;
  border: 2px solid rgba(255, 255, 255, 0.6);
}

*::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.26);
}

/* 移动端与窄屏：保持同风格但更紧凑 */
@media (max-width: 1024px) {
  .block-container {
    padding-top: 1rem;
    padding-bottom: 2.8rem;
  }

  h1 {
    font-size: 1.8rem !important;
  }
}

@media (max-width: 768px) {
  .block-container {
    padding-left: 0.75rem;
    padding-right: 0.75rem;
    padding-bottom: 2.25rem;
  }

  h1 {
    font-size: 1.6rem !important;
  }

  .stTabs [data-baseweb="tab"] {
    padding: 0.5rem 0.8rem;
    font-size: 0.92rem;
  }
}
</style>
"""


def apply_app_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "axes.edgecolor": "#d2d2d7",
            "axes.labelcolor": "#1d1d1f",
            "xtick.color": "#6e6e73",
            "ytick.color": "#6e6e73",
            "text.color": "#1d1d1f",
            "grid.color": "#e5e5ea",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.titleweight": PLOT_FONT_WEIGHT_SEMIBOLD,
            "axes.labelweight": PLOT_FONT_WEIGHT_SEMIBOLD,
            # 智能切换科学计数：当指数 < -3 或 >= 4 时使用科学计数
            "axes.formatter.limits": PLOT_SCI_POWERLIMITS,
            "axes.formatter.use_mathtext": True,
            "axes.formatter.useoffset": False,
        }
    )
