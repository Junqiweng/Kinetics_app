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

# ========== 拟合/配置默认值（与 UI/导出保持一致） ==========
# 参数边界默认值（与 config_manager.get_default_config / UI 默认保持一致）
DEFAULT_K0_MIN = 1e-15  # 指前因子下界 [SI]
DEFAULT_K0_MAX = 1e15  # 指前因子上界 [SI]
DEFAULT_EA_MIN_J_MOL = 3e4  # 活化能下界 [J/mol]
DEFAULT_EA_MAX_J_MOL = 3e5  # 活化能上界 [J/mol]
DEFAULT_ORDER_MIN = -2.0  # 反应级数下界 [-]
DEFAULT_ORDER_MAX = 5.0  # 反应级数上界 [-]

# Langmuir-Hinshelwood (L-H) 边界默认值
DEFAULT_K0_ADS_MIN = 0.0  # 吸附常数指前因子下界 [1/(mol/m^3)]
DEFAULT_K0_ADS_MAX = 1e10  # 吸附常数指前因子上界 [1/(mol/m^3)]
DEFAULT_EA_K_MIN_J_MOL = -2e5  # 吸附热下界 [J/mol]（允许负值：放热吸附）
DEFAULT_EA_K_MAX_J_MOL = 2e5  # 吸附热上界 [J/mol]
DEFAULT_M_INHIBITION_MIN = 0.0  # 抑制指数下界 [-]
DEFAULT_M_INHIBITION_MAX = 5.0  # 抑制指数上界 [-]

# 可逆反应边界默认值
DEFAULT_K0_REV_MIN = DEFAULT_K0_MIN  # 逆反应指前因子下界 [SI]
DEFAULT_K0_REV_MAX = DEFAULT_K0_MAX  # 逆反应指前因子上界 [SI]
DEFAULT_EA_REV_MIN_J_MOL = DEFAULT_EA_MIN_J_MOL  # 逆反应活化能下界 [J/mol]
DEFAULT_EA_REV_MAX_J_MOL = DEFAULT_EA_MAX_J_MOL  # 逆反应活化能上界 [J/mol]
DEFAULT_ORDER_REV_MIN = DEFAULT_ORDER_MIN  # 逆反应级数下界 [-]
DEFAULT_ORDER_REV_MAX = DEFAULT_ORDER_MAX  # 逆反应级数上界 [-]

# 优化器/数值差分默认值
DEFAULT_DIFF_STEP_REL = 1e-3  # least_squares 数值差分相对步长 [-]
DEFAULT_MAX_NFEV = 3000  # least_squares 最大迭代次数 [-]
DEFAULT_N_STARTS = 10  # multi-start 起点数量 [-]
DEFAULT_MAX_NFEV_COARSE = 300  # multi-start 粗略阶段最大迭代次数 [-]
DEFAULT_RANDOM_SEED = 42  # 随机种子 [-]

# ========== 反应器稳态/剖面计算默认值 ==========
# CSTR 稳态用 least_squares 的默认参数
DEFAULT_CSTR_STEADY_MAX_NFEV = 200  # CSTR 稳态最大迭代次数 [-]
DEFAULT_CSTR_STEADY_XTOL = 1e-10  # 变量收敛阈值 [-]
DEFAULT_CSTR_STEADY_FTOL = 1e-10  # 目标函数收敛阈值 [-]
DEFAULT_CSTR_STEADY_GTOL = 1e-10  # 梯度收敛阈值 [-]

# 剖面点数默认值（PFR/CSTR/BSTR）
DEFAULT_PROFILE_N_POINTS = 200  # 剖面点数 [-]

# ========== 后台拟合/数值保护（UI 友好） ==========
FITTING_UI_UPDATE_INTERVAL_S = 0.6  # 进度条/状态最小刷新间隔 [s]
FITTING_EPSILON_PHI_RATIO = 1e-300  # phi_ratio 计算用的分母保护值 [-]
FITTING_EPSILON_NORM = 1e-12  # 参数相对变化计算的分母保护值 [-]

# 用户点击“停止拟合”后，主线程等待后台退出的轮询参数
FITTING_STOP_WAIT_TRIES = 10  # 轮询次数 [-]
FITTING_STOP_WAIT_SLEEP_S = 0.05  # 每次轮询睡眠时间 [s]

# ========== 会话/持久化相关常量 ==========
PERSIST_DIR_NAME = "Kinetics_app_persist"  # 持久化目录名（在系统临时目录下）
SESSION_CLEANUP_EVERY_N_PAGE_LOADS = 20  # 每 N 次页面加载清理一次 [-]
DEFAULT_SESSION_MAX_AGE_HOURS = 24  # 默认最大会话年龄 [h]
SECONDS_PER_HOUR = 3600  # 1 小时对应秒数 [s]
UUID_STRING_LENGTH = 36  # UUID 字符串长度（含 '-'）[-]
UUID_HYPHEN_COUNT = 4  # UUID 字符串中 '-' 的数量 [-]

# ========== UI 表格/控件默认尺寸 ==========
UI_PARAM_TABLE_HEIGHT_PX = 250  # 参数表格高度 [px]
UI_DATA_PREVIEW_ROWS = 50  # CSV 预览行数 [-]
UI_DATA_PREVIEW_HEIGHT_PX = 200  # CSV 预览表格高度 [px]
UI_COMPARE_TABLE_HEIGHT_PX = 260  # 预测 vs 实验对比表高度 [px]
UI_METRICS_TABLE_HEIGHT_PX = 220  # 误差指标表格高度 [px]

# 剖面点数输入框（UI）
UI_PROFILE_POINTS_DEFAULT = DEFAULT_PROFILE_N_POINTS  # 默认剖面点数 [-]
UI_PROFILE_POINTS_MIN = 20  # 最小剖面点数 [-]
UI_PROFILE_POINTS_MAX = 2000  # 最大剖面点数 [-]
UI_PROFILE_POINTS_STEP = 20  # 剖面点数步长 [-]

# 高级设置中常用 step
UI_MAX_STEP_FRACTION_STEP = 0.05  # max_step_fraction 的步长 [-]
UI_MAX_NFEV_STEP = 500  # max_nfev 的步长 [-]

# ========== 列名提示/报错信息相关 ==========
CSV_COLUMN_PREVIEW_COUNT = 40  # 报错时展示的 CSV 列名前 N 个 [-]
CSV_INVALID_INDEX_PREVIEW_COUNT = 10  # 报错时展示的 index 示例数 [-]
CSV_CLOSE_MATCHES_MAX = 3  # difflib.get_close_matches 的最大建议数 [-]
CSV_CLOSE_MATCHES_CUTOFF = 0.6  # difflib.get_close_matches 相似度阈值 [-]

# ========== 绘图风格相关 ==========
PLOT_SCI_POWERLIMITS = (-3, 3)  # 科学计数法触发指数范围 (min_exp, max_exp)
PLOT_FONT_WEIGHT_SEMIBOLD = 600  # Matplotlib 字重（接近 semibold）[-]
PLOT_EXPORT_DPI = 300  # 图片导出 DPI [-]

# ========== UI 数字显示/输入格式（统一管理） ==========
# 规则：当 |x| < UI_FLOAT_SCI_LOW 或 |x| >= UI_FLOAT_SCI_HIGH 时，使用科学计数法；
# 否则使用常规数字显示（小数位自动裁剪，避免无意义的尾随 0）。
UI_FLOAT_SCI_LOW = 1e-3  # 科学计数法阈值（小数太小）[-]
UI_FLOAT_SCI_HIGH = 1e3  # 科学计数法阈值（数字太大）[-]
UI_FLOAT_NORMAL_MAX_DECIMALS = 4  # 常规数字最多保留小数位（会自动去掉尾随 0）[-]
UI_FLOAT_SCI_DECIMALS = 3  # 科学计数法小数位（例如 1.23e+04）[-]

# Streamlit 的 number_input 仅支持固定显示格式；当触发科学计数时使用该格式。
UI_FLOAT_SCI_FORMAT_STREAMLIT = "%.3e"

# 表格（级数矩阵等）中数字列的固定显示格式
UI_ORDER_MATRIX_NUMBER_FORMAT = "%.3f"

# rtol/atol 等“典型很小”的数值输入格式（固定科学计数）
UI_TOLERANCE_FORMAT_STREAMLIT = "%.1e"

# ========== 枚举/字符串常量（统一口径，避免硬编码分散） ==========
# 说明：
# - 这些字符串会出现在 UI 下拉框、配置导入导出、校验逻辑、结果展示等多个位置。
# - 必须保持字符串值稳定，否则历史配置文件或旧数据模板可能无法兼容。

# 反应器类型
REACTOR_TYPE_PFR = "PFR"
REACTOR_TYPE_CSTR = "CSTR"
REACTOR_TYPE_BSTR = "BSTR"
REACTOR_TYPES = [REACTOR_TYPE_PFR, REACTOR_TYPE_CSTR, REACTOR_TYPE_BSTR]

# PFR 流动模型/相态（仅影响“由摩尔流量 -> 浓度”的换算方式）
# - liquid_const_vdot: 液相近似（或体积流量可视作常数的场景），沿程 vdot 不变
# - gas_ideal_const_p: 气相理想气体、等温、恒压（不考虑压降），沿程用 y_i·P/(R·T) 计算浓度
PFR_FLOW_MODEL_LIQUID_CONST_VDOT = "liquid_const_vdot"
PFR_FLOW_MODEL_GAS_IDEAL_CONST_P = "gas_ideal_const_p"
PFR_FLOW_MODELS = [PFR_FLOW_MODEL_LIQUID_CONST_VDOT, PFR_FLOW_MODEL_GAS_IDEAL_CONST_P]

# 动力学模型类型
KINETIC_MODEL_POWER_LAW = "power_law"
KINETIC_MODEL_LANGMUIR_HINSHELWOOD = "langmuir_hinshelwood"
KINETIC_MODEL_REVERSIBLE = "reversible"
KINETIC_MODELS = [
    KINETIC_MODEL_POWER_LAW,
    KINETIC_MODEL_LANGMUIR_HINSHELWOOD,
    KINETIC_MODEL_REVERSIBLE,
]

# 输出模式（拟合目标变量）
OUTPUT_MODE_FOUT = "Fout (mol/s)"
OUTPUT_MODE_COUT = "Cout (mol/m^3)"
OUTPUT_MODE_XOUT = "xout (mole fraction)"

# 根据反应器类型允许的输出模式（用于 UI 与配置校验）
OUTPUT_MODES_FLOW = [OUTPUT_MODE_COUT, OUTPUT_MODE_FOUT, OUTPUT_MODE_XOUT]  # PFR/CSTR
OUTPUT_MODES_BATCH = [OUTPUT_MODE_COUT]  # BSTR
