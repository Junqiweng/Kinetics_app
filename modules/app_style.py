from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


APP_CSS = """
<style>
/* 基础字体设置 - 增大以提升可读性 */
html, body, [class*="css"] {
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #1e293b;
  font-size: 16px;
}

/* 主容器优化 */
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 4rem;
  max-width: 1440px;
}

/* 侧边栏美化 */
[data-testid="stSidebar"] {
  background: linear-gradient(to bottom, #f8fafc 0%, #ffffff 100%);
  border-right: 2px solid #e2e8f0;
  padding-top: 1rem;
}

[data-testid="stSidebar"] .block-container {
  padding-top: 1rem;
}

/* 标题增强 - 更大更清晰 */
h1 {
  font-size: 2.25rem !important;
  font-weight: 700 !important;
  background: linear-gradient(135deg, #1e40af 0%, #4f46e5 50%, #7c3aed 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem !important;
  letter-spacing: -0.025em;
}

h2 {
  font-size: 1.5rem !important;
  font-weight: 600 !important;
  color: #1e40af;
  margin-top: 1.5rem !important;
  margin-bottom: 0.75rem !important;
}

h3 {
  font-size: 1.25rem !important;
  font-weight: 600 !important;
  color: #334155;
  margin-top: 1rem !important;
  margin-bottom: 0.5rem !important;
}

h4 {
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  color: #475569;
}

/* Tab 样式优化 */
.stTabs [data-baseweb="tab-list"] {
  gap: 1.5rem;
  border-bottom: 2px solid #e2e8f0;
  padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
  font-weight: 600;
  font-size: 1.05rem;
  color: #64748b;
  padding: 0.75rem 1rem;
  transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
  color: #4f46e5;
  background-color: #f1f5f9;
}

.stTabs [aria-selected="true"] {
  color: #4f46e5 !important;
  border-bottom: 3px solid #4f46e5 !important;
  font-weight: 700;
}

/* 文本样式优化 */
.stMarkdown p, .stMarkdown li {
  font-size: 1rem;
  line-height: 1.7;
  color: #334155;
}

.stMarkdown ul, .stMarkdown ol {
  margin-left: 1.25rem;
}

/* Caption优化 - 更清晰 */
.stCaption {
  font-size: 0.9rem !important;
  color: #64748b !important;
  line-height: 1.5 !important;
  margin-bottom: 0.5rem !important;
}

/* Alert样式 */
.stAlert p {
  font-size: 0.95rem;
  line-height: 1.6;
}

/* 按钮美化 */
.stButton > button, .stDownloadButton > button {
  font-size: 1rem;
  font-weight: 500;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  transition: all 0.2s ease;
  border: 1px solid #e2e8f0;
}

.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 输入框优化 */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
  font-size: 0.95rem !important;
  border-radius: 6px !important;
}

/* 数据表格美化 */
.stDataFrame {
  font-size: 0.9rem;
}

[data-testid="stDataFrame"] {
  border-radius: 8px;
  overflow: hidden;
}

/* 容器边框美化 */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
  border-radius: 10px;
}

div[data-testid="stExpander"] {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background-color: #fafbfc;
}

div[data-testid="stExpander"]:hover {
  border-color: #cbd5e1;
}

/* 对话框优化 */
div[role="dialog"][aria-modal="true"] {
  width: 90vw !important;
  max-width: 1300px !important;
  border-radius: 12px !important;
}

/* 进度条美化 */
.stProgress > div > div > div > div {
  background-color: #4f46e5;
}

/* 分隔线 */
hr {
  margin: 1.5rem 0 !important;
  border-color: #e2e8f0 !important;
}

/* 信息框美化 */
.stInfo, .stSuccess, .stWarning, .stError {
  border-radius: 8px;
  font-size: 0.95rem;
}

/* 代码块优化 */
code {
  font-size: 0.9rem !important;
  padding: 0.2rem 0.4rem !important;
  border-radius: 4px !important;
}

pre code {
  font-size: 0.85rem !important;
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
            "axes.edgecolor": "#e2e8f0",
            "axes.labelcolor": "#475569",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "text.color": "#1e293b",
            "grid.color": "#f1f5f9",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )

