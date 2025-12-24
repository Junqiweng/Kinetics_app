# 文件作用：集中定义项目用到的公共常量（物理常数、数值保护阈值、拟合/求解器默认参数等）。

"""
公共常量定义模块

统一定义项目中使用的数值常量，避免魔术数字分散在各处。
"""

from __future__ import annotations

# ========== 物理常量 ==========
R_GAS_J_MOL_K = 8.314462618  # 气体常数 [J/(mol*K)]

# ========== 数值保护常量 ==========
# 用于避免除零、负数开根号等数值问题的极小值
EPSILON_CONCENTRATION = 1e-30  # 浓度下限 [mol/m³]，避免 0^negative -> inf
EPSILON_FLOW_RATE = 1e-30  # 流量下限 [m³/s 或 mol/s]
EPSILON_DENOMINATOR = 1e-30  # 通用分母保护值
EPSILON_RELATIVE = 1e-15  # 相对误差判断阈值

# 用于数值比较的容差
FLOAT_EQUALITY_TOLERANCE = 1e-15  # 浮点数相等判断容差

# ========== 拟合相关常量 ==========
# 残差罚值相关
DEFAULT_RESIDUAL_PENALTY_MULTIPLIER = 1e3  # 罚值乘数
DEFAULT_RESIDUAL_PENALTY_MIN_ABS = 1e3  # 最小罚值绝对值

# 百分比残差的 epsilon 系数（典型值的百分比）
PERCENTAGE_RESIDUAL_EPSILON_FACTOR = 0.01  # 1%

# ========== 求解器相关常量 ==========
DEFAULT_MAX_STEP_FRACTION = 0.1  # 默认 ODE 步长限制比例
DEFAULT_RTOL = 1e-6  # 默认相对容限
DEFAULT_ATOL = 1e-9  # 默认绝对容限
