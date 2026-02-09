from __future__ import annotations

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator


# 参考图风格：黑色坐标轴 + 空心圆标记 + 无边框图例。
FIT_PLOT_COLOR_CYCLE = [
    "#4d4d4d",  # 深灰
    "#ff2d2d",  # 红
    "#1f77ff",  # 蓝
    "#2ca02c",  # 绿
    "#ff7f0e",  # 橙
    "#17becf",  # 青
]


def _patch_tornado_websocket_noise() -> None:
    """
    兼容补丁：抑制 websocket 关闭时的“Task exception was never retrieved”噪声栈。

    背景：
    - Tornado 的 WebSocketProtocol13.write_message 返回一个可选 await 的 Future。
    - 连接关闭（浏览器刷新/断开）后，该 Future 可能抛 WebSocketClosedError，
      若没有被显式 await，会在控制台打印“Task exception was never retrieved”。
    - 这里仅消费“连接已关闭”这两类异常，不改变其它异常的可见性。
    """
    try:
        import tornado.iostream as tornado_iostream
        import tornado.websocket as tornado_websocket
    except Exception:
        return

    protocol_cls = getattr(tornado_websocket, "WebSocketProtocol13", None)
    if protocol_cls is None:
        return
    if bool(getattr(protocol_cls, "_kinetics_write_message_patched", False)):
        return

    original_write_message = protocol_cls.write_message

    def _patched_write_message(self, message, binary: bool = False):
        fut = original_write_message(self, message, binary=binary)
        try:
            if hasattr(fut, "add_done_callback"):

                def _consume_ws_close_error(task) -> None:
                    try:
                        task.result()
                    except (
                        tornado_websocket.WebSocketClosedError,
                        tornado_iostream.StreamClosedError,
                    ):
                        # 连接关闭属于正常生命周期事件，不需要打印任务未取回异常。
                        pass
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Unexpected websocket write task error."
                        )

                fut.add_done_callback(_consume_ws_close_error)
        except Exception:
            pass
        return fut

    protocol_cls.write_message = _patched_write_message
    protocol_cls._kinetics_write_message_patched = True


def _configure_matplotlib_chinese_font() -> None:
    """
    Matplotlib 中文字体配置（主要解决：图中中文显示为方框/乱码）。

    说明：
    - Streamlit 本身的中文显示通常没问题；常见问题出现在 Matplotlib 渲染的图像里。
    - 这里优先选择系统常见中文字体（Windows/Linux/macOS 兼容候选）。
    - 若系统缺少中文字体，则仍可能无法显示中文；此时需要在运行环境安装中文字体。
    """

    try:
        candidates = [
            # Windows 常见
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            # macOS 常见
            "PingFang SC",
            "Heiti SC",
            # Linux 常见（取决于发行版/镜像）
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Source Han Sans SC",
            "AR PL UMing CN",
        ]

        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = None
        for name in candidates:
            if name in available:
                chosen = name
                break

        # 关键：优先把可用中文字体放到 sans-serif 的最前面
        base_list = list(mpl.rcParams.get("font.sans-serif", []))
        if chosen is not None:
            mpl.rcParams["font.sans-serif"] = [chosen] + [
                x for x in base_list if x != chosen
            ]
        else:
            # 没找到则把候选列表追加到前面，让 Matplotlib 自行尝试匹配
            mpl.rcParams["font.sans-serif"] = candidates + base_list

        # 负号正常显示（否则可能显示为方块）
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        # 字体配置失败不应影响主功能
        return


def apply_runtime_patches() -> None:
    _patch_tornado_websocket_noise()


def _fit_plot_color(index: int) -> str:
    return FIT_PLOT_COLOR_CYCLE[int(index) % len(FIT_PLOT_COLOR_CYCLE)]


def _style_fit_axis(ax: plt.Axes, show_grid: bool = False) -> None:
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.4)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6.0,
        width=1.3,
        colors="#000000",
        labelsize=11,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=3.5,
        width=1.0,
        colors="#000000",
    )
    # 每两个主刻度之间仅保留 1 个副刻度
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if bool(show_grid):
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.8,
            color="#6e6e73",
            alpha=0.25,
        )
    else:
        ax.grid(False)


def _style_fit_legend(ax: plt.Axes, loc: str = "upper left") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    ax.legend(
        handles,
        labels,
        loc=loc,
        frameon=False,
        fontsize=11,
        handlelength=2.2,
        handletextpad=0.4,
    )


def _plot_reference_series(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    label: str,
    color: str,
) -> None:
    n_points = int(np.size(x_values))
    marker_step = max(n_points // 8, 1)
    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=6.0,
        markerfacecolor=color,
        markeredgecolor="#ffffff",
        markeredgewidth=0.9,
        markevery=marker_step,
        label=label,
    )
