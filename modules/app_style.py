from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


APP_CSS = """
<style>
:root {
  --app-bg: #F5F5F7;
  --card-bg: #FFFFFF;
  --text: #1D1D1F;
  --muted: #6E6E73;
  --border: rgba(0, 0, 0, 0.10);
  --border-strong: rgba(0, 0, 0, 0.14);
  --accent: #007AFF;
  --radius: 16px;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 10px 30px rgba(0, 0, 0, 0.04);
  --shadow-soft: 0 1px 1px rgba(0, 0, 0, 0.04);
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

/* App 背景 */
[data-testid="stAppViewContainer"] {
  background: var(--app-bg);
}

/* 主容器：控制最大宽度，让排版更“工整” */
.block-container {
  max-width: 1280px;
  padding-top: 1.25rem;
  padding-bottom: 3.5rem;
}

/* 顶部 Header：轻薄半透明 */
[data-testid="stHeader"] {
  background: rgba(245, 245, 247, 0.85);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(10px);
}

/* 侧边栏：干净白底 + 细分割线 */
[data-testid="stSidebar"] {
  background: var(--card-bg);
  border-right: 1px solid var(--border);
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
  margin-bottom: 0.35rem !important;
}

h2, h3, h4 {
  letter-spacing: -0.01em;
}

h2 {
  font-size: 1.25rem !important;
  font-weight: 650 !important;
  color: var(--text);
  margin-top: 1.25rem !important;
  margin-bottom: 0.6rem !important;
}

h3 {
  font-size: 1.08rem !important;
  font-weight: 650 !important;
  color: var(--text);
  margin-top: 1.0rem !important;
  margin-bottom: 0.5rem !important;
}

h4 {
  font-size: 1.0rem !important;
  font-weight: 650 !important;
  color: var(--muted);
}

/* 说明文字 */
.stMarkdown p, .stMarkdown li {
  line-height: 1.65;
  color: var(--text);
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
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.kinetics-card-marker) > div {
  padding: 0 !important;
}

/* Tabs：做成接近 iOS 的 segmented control */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card-bg);
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
  transition: background 120ms ease, color 120ms ease;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(0, 0, 0, 0.04);
  color: var(--text);
}

.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #ffffff !important;
}

/* 按钮：统一圆角与主色，减少夸张位移效果 */
button[data-testid^="baseButton-"] {
  border-radius: 12px !important;
  font-weight: 600 !important;
  transition: background 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
}

button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-tertiary"] {
  background: var(--card-bg) !important;
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
  background: #0066D6 !important;
  box-shadow: var(--shadow-soft) !important;
}

/* 输入控件：白底、细边框、聚焦蓝色描边 */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
  border-color: rgba(0, 122, 255, 0.55) !important;
  box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.14) !important;
}

/* NumberInput（含 +/- stepper）：避免“内外两层边框”叠加产生重影 */
div[data-testid="stNumberInput"] [data-baseweb="input"] {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  overflow: hidden;
}

div[data-testid="stNumberInput"] input {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}

div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
  border-color: rgba(0, 122, 255, 0.55) !important;
  box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.14) !important;
}

div[data-testid="stNumberInput"] [data-baseweb="button"] {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

div[data-testid="stNumberInput"] [data-baseweb="button"]:hover {
  background: rgba(0, 0, 0, 0.04) !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* 表格/数据编辑器：圆角 + 细边框 */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-soft);
}

/* Expander：同卡片风格 */
:where(div, details)[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--card-bg) !important;
  box-shadow: var(--shadow-soft) !important;
  overflow: hidden; /* 避免内部元素的边框/描边“叠加”到外框上，看起来像两层边框 */
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
  width: 92vw !important;
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
}

/* 代码块：更紧凑 */
code {
  font-size: 0.9rem !important;
  border-radius: 8px !important;
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
            "axes.titleweight": 600,
            "axes.labelweight": 600,
        }
    )

