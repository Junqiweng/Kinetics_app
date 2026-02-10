# Kinetics_app - 反应动力学参数拟合工具（PFR / CSTR / BSTR）

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-red.svg)](https://streamlit.io/)

一个基于 **Python + Streamlit** 的网页工具，用于 **PFR（平推流）**、**CSTR（连续搅拌釜）** 与 **BSTR（间歇搅拌釜）** 的反应动力学参数拟合与模拟。

## 文档（建议从这里开始）

- **详细用户指南**：`docs/user_guide.md`
- **App 内教程/帮助**：左侧 `📖 教程/帮助`（内容来自 `docs/help_*.md`）

## 最近更新（2026-02-10）

- 清理历史遗留文件与缓存目录（例如 `*.bak`、`__pycache__`）
- 同步 `README / TODO / user_guide / test_data` 的口径，统一“示例 CSV 需脚本生成”的说明
- 增加并固定回归验证命令：`scripts/smoke_validate.py`、`scripts/regression_state_validate.py`

## 功能特性

- 📊 **多物种、多反应系统**：支持自定义物种数量和反应数量
- 🧪 **多反应器类型**：支持 PFR / CSTR / BSTR
- 🔬 **多种动力学模型**：幂律（Power-law）、Langmuir-Hinshelwood（吸附抑制）、可逆反应（Reversible）
- 🎯 **灵活的参数拟合**：可选择性拟合 k₀、Eₐ 和反应级数
- 📈 **多种目标变量**：支持出口摩尔流量、出口浓度或出口摩尔分数作为拟合目标（xout 仅 PFR/CSTR）
- 🚀 **高级拟合选项**：多起点拟合、参数缩放、可调有限差分步长
- 📉 **可视化结果**：奇偶校验图（Parity Plot）、误差图，以及剖面图（PFR: 随 $V$；BSTR: 随 $t$；CSTR: 随 $t$ 逼近稳态）
- 📥 **一键导出**：拟合参数/对比数据/报告表（CSV）与图像（PNG/SVG）
- ⚙️ **配置管理**：配置导入/导出（JSON），并可在启动时自动恢复上次配置

## 理论背景

### 反应动力学模型（可选）

对于第 $j$ 个反应，提供以下动力学模型：

**(1) 幂律 (Power-law)**

$$r_j = k_j(T) \prod_i C_i^{n_{i,j}}$$

**(2) Langmuir-Hinshelwood（吸附抑制）**

$$r_j = \frac{k_j(T) \prod_i C_i^{n_{i,j}}}{\left(1 + \sum_i K_i(T) C_i\right)^{m_j}}$$

$$K_i(T) = K_{0,i} \exp\left(-\frac{E_{a,K,i}}{RT}\right)$$

**(3) 可逆反应 (Reversible)**

$$r_j = k_j^+(T) \prod_i C_i^{n_{i,j}^+} - k_j^-(T) \prod_i C_i^{n_{i,j}^-}$$

其中 Arrhenius 形式为：

$$k_j^{\pm}(T) = k_{0,j}^{\pm} \exp\left(-\frac{E_{a,j}^{\pm}}{RT}\right)$$

### PFR 设计方程

$$\frac{dF_i}{dV} = \sum_{j=1}^{N_{rxn}} \nu_{i,j} r_j$$

其中浓度与摩尔流量关系按流动模型选择：

液相恒定体积流量：

$$C_i = \frac{F_i}{\dot{v}}$$

气相理想气体等温恒压（无压降）：

$$C_i = y_i\frac{P}{RT},\quad y_i=\frac{F_i}{\sum_k F_k}$$

### 符号说明

| 符号 | 含义 | 单位 |
|:---:|:---|:---|
| $r_j$ | 第 $j$ 个反应的反应速率 | mol·m⁻³·s⁻¹ |
| $k_{0,j}$ | 指前因子 | 取决于反应级数 |
| $E_{a,j}$ | 活化能 | J·mol⁻¹ |
| $R$ | 通用气体常数 | 8.314 J·mol⁻¹·K⁻¹ |
| $T$ | 反应温度 | K |
| $C_i$ | 物种 $i$ 的摩尔浓度 | mol·m⁻³ |
| $n_{i,j}$ | 反应 $j$ 中物种 $i$ 的反应级数 | 无量纲 |
| $F_i$ | 物种 $i$ 的摩尔流量 | mol·s⁻¹ |
| $V$ | 反应体积 | m³ |
| $\nu_{i,j}$ | 化学计量系数 (反应物为负，生成物为正) | 无量纲 |
| $\dot{v}$ | 体积流量 | m³·s⁻¹ |

## 安装与运行

### 环境要求

- Python 3.10+（`main.py` 使用了 `X | None` 类型标注语法）
- 依赖库见 `requirements.txt`

### 安装步骤

```bash
# 克隆或下载项目
cd Kinetics_app

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run main.py
```

运行后，浏览器会自动打开应用界面（默认地址：http://localhost:8501）。

## 快速验证（推荐先跑通一次）

1) 生成/更新 PFR 示例数据（仓库默认不提交生成后的 CSV，需本地生成一次）：

```bash
python test_data/generate_orthogonal_design.py
```

2) 打开 App：`streamlit run main.py`

3) 运行轻量脚本验证核心路径：

```bash
python scripts/smoke_validate.py
python scripts/regression_state_validate.py
```

4) 在“教程/帮助”页按推荐流程操作（下载示例数据/模板 → 上传 → 选择目标 → 开始拟合）。

也可以直接阅读 `docs/user_guide.md`（更详细）。

## 使用说明

### 1. 反应定义

- 输入物种名称（逗号分隔，如 `A,B,C`）
- 选择反应器类型（PFR / CSTR / BSTR）与动力学模型（Power-law / L-H / Reversible）
- 设置反应数量
- 填写化学计量数矩阵 ν（反应物为负，生成物为正）
- 设置反应级数矩阵 n
- 输入 k₀ 和 Eₐ 的初始猜测值

### 2. 实验数据

上传 CSV 文件，必须包含以下输入列（按反应器类型）：

**PFR 输入列（按流动模型区分）**

液相恒定体积流量（默认）：

| 列名 | 含义 | 单位 |
|:---|:---|:---|
| `V_m3` | 反应器体积 | m³ |
| `T_K` | 反应温度 | K |
| `vdot_m3_s` | 体积流量 | m³/s |
| `F0_<物种名>_mol_s` | 各物种入口摩尔流量 | mol/s |

气相理想气体、等温恒压（无压降）：

| 列名 | 含义 | 单位 |
|:---|:---|:---|
| `V_m3` | 反应器体积 | m³ |
| `T_K` | 反应温度 | K |
| `P_Pa` | 反应压力 | Pa |
| `F0_<物种名>_mol_s` | 各物种入口摩尔流量 | mol/s |

**CSTR 输入列**

| 列名 | 含义 | 单位 |
|:---|:---|:---|
| `V_m3` | 反应器体积 | m³ |
| `T_K` | 反应温度 | K |
| `vdot_m3_s` | 体积流量 | m³/s |
| `C0_<物种名>_mol_m3` | 各物种入口浓度 | mol/m³ |

**BSTR 输入列**

| 列名 | 含义 | 单位 |
|:---|:---|:---|
| `t_s` | 反应时间 | s |
| `T_K` | 反应温度 | K |
| `C0_<物种名>_mol_m3` | 各物种初始浓度 | mol/m³ |

测量值列（任选一种类型）：
- `Fout_<物种名>_mol_s`：出口摩尔流量 [mol/s]（PFR / CSTR）
- `Cout_<物种名>_mol_m3`：出口浓度 [mol/m³]
- `xout_<物种名>`：出口摩尔分数 [-]（仅 PFR / CSTR）

重要：当前版本对“你选中的测量列”（由 **拟合目标变量 + 目标物种** 决定）要求 **每一行都是数字**，不允许 NaN/空值/非数字；否则会停止拟合并提示具体列与行号。

可在应用内下载 CSV 数据模板（推荐先下载模板再粘贴数据）。

### 3. 参数拟合

- 选择拟合目标变量类型（Fout / Cout / xout）
- 选择参与目标函数的物种
- 设置参数边界和高级拟合选项
- 点击"开始拟合"

## 拟合算法说明（与代码一致）

- 数值积分：`scipy.integrate.solve_ivp`（可选 `RK45 / BDF / Radau`）
- 参数拟合：`scipy.optimize.least_squares`（`method="trf"`）
- 鲁棒性选项：多起点（multi-start）、有限差分步长 `diff_step`、参数缩放 `x_scale="jac"`
- 残差定义：
  - 绝对残差：$r = y_{pred} - y_{meas}$
  - 相对残差：$r = \frac{y_{pred} - y_{meas}}{\mathrm{sign}(y_{meas})\cdot\max(|y_{meas}|,\epsilon)}$
  - 百分比残差：$r = 100\cdot\frac{y_{pred} - y_{meas}}{|y_{meas}|+\epsilon}$（其中 $\epsilon$ 为小正数，避免除零）

## 重要假设与限制（建议使用前确认）

- **PFR**：支持两种模式：液相恒定体积流量（$C_i = F_i/\dot{v}$）与气相理想气体等温恒压（$C_i = y_iP/RT$）；两种模式均未包含压降与非理想气体效应。
- **CSTR**：采用稳态物料衡算 $0=\dot{v}(C_{0,i}-C_i)+V\sum_j\nu_{i,j}r_j$（内部用非线性最小二乘求解稳态 $C$）。
- **BSTR**：采用恒体积、无进出料的 $dC/dt=\nu r$ 模型；BSTR 模式不支持 `Fout` 作为拟合输出。
- **测量缺失**：所选测量列不允许出现 NaN/空值；若某物种/某些行缺测，建议拆分数据文件或取消选择对应目标物种。
- **结果缓存**：拟合完成后会缓存拟合参数与当次配置/数据，结果展示与导出将锁定于该次拟合；如需应用新配置/新数据，请清除缓存并重新拟合。
- **负级数**：当反应级数为负且浓度趋近 0 时，为避免 $0^{n<0}$ 发散，内部会对浓度加很小的下限（`1e-30 mol/m³`）；这可能影响极低浓度区间的拟合稳定性。
- **性能**：每一次目标函数评估都需要对每一行实验条件进行数值计算（PFR/BSTR 用 `solve_ivp`；CSTR 用稳态非线性求解），当数据量很大（例如 >1000 行）会明显变慢。

## 数据与配置的保存位置（本地）

- 上传 CSV 与“上一次配置”会保存到系统临时目录下的 `Kinetics_app_persist`（用于页面刷新后的恢复）。
- 在云端环境（例如 Streamlit Cloud）会额外使用浏览器 LocalStorage 保存配置（见 `modules/browser_storage.py`）。

## 项目结构

```
Kinetics_app/
├── main.py                 # Streamlit 主应用程序
├── modules/                # 核心计算与 UI 辅助模块
│   ├── contexts.py         # App 上下文类型与构建（bootstrap/sidebar/model/data/fit）
│   ├── bootstrap.py        # 启动/会话恢复/全局状态初始化
│   ├── sidebar.py          # 侧边栏渲染（全局模型、求解器、配置管理）
│   ├── tab_model.py        # 模型页签（物种、反应、参数初值）
│   ├── tab_data.py         # 数据页签（模板、上传、预览、配置导出）
│   ├── tab_fit.py          # 拟合页签编排入口（advanced -> execution -> results）
│   ├── fit_advanced.py     # 拟合高级设置与边界
│   ├── fit_execution.py    # 拟合操作与后台任务触发
│   ├── fit_results.py      # 拟合结果展示（参数/奇偶校验/剖面/导出）
│   ├── export_config.py    # 配置导出构建与持久化公共逻辑
│   ├── kinetics.py         # 动力学：幂律 / L-H / 可逆速率
│   ├── reactors.py         # 反应器：PFR/CSTR/BSTR 数值积分/稳态求解 + 剖面
│   ├── fitting.py          # 拟合工具：参数打包/解包、单行预测
│   ├── config_manager.py   # 配置导入/导出/自动保存
│   ├── ui_help.py          # 教程/帮助（渲染 docs/help_*.md）
│   └── ui_components.py    # UI 组件与导出工具
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明文档
├── docs/                   # 更详细的教程与说明
├── scripts/                # 轻量自检/回归脚本
│   ├── smoke_validate.py
│   └── regression_state_validate.py
├── .streamlit/
│   └── config.toml        # Streamlit 主题配置
└── test_data/
    ├── generate_orthogonal_design.py  # PFR 示例数据生成（正交设计）
    ├── orthogonal_design_data.csv     # 生成后得到（默认不提交到仓库）
    ├── generate_complex_data.py       # 复杂验证数据生成
    ├── validation_*.csv               # 生成后得到（默认不提交到仓库）
    └── validation_*.json              # 对应的可导入配置（与 CSV 匹配）
```

## App 联动流程

```text
bootstrap_app_state
  -> render_sidebar
    -> render_model_tab
      -> render_data_tab
        -> render_fit_tab
          -> render_fit_advanced
          -> render_fit_actions
          -> render_fit_results
```

## 测试数据

`test_data/` 提供“数据生成脚本 + 可导入配置”。  
说明：`orthogonal_design_data.csv` 与 `validation_*.csv` 属于脚本产物，默认被 `.gitignore` 忽略，不保证仓库内直接存在。

- **`orthogonal_design_data.csv`**：PFR 示例数据（A 一级反应，27 组工况，带少量噪声；示例测量列为 `Fout_A_mol_s`）
- **`generate_orthogonal_design.py`**：重新生成上述 PFR 示例数据
- **`generate_complex_data.py`**：生成更复杂的验证数据（并附带可导入配置 `validation_*.json`）
- **更多验证数据**：见 `test_data/README.md`

基础测试参数（示例生成用）：
- 反应：A → B（一级反应）
- $k_0 = 1\times 10^6\;\mathrm{s^{-1}}$
- $E_a = 5\times 10^4\;\mathrm{J/mol}$

## 依赖库

- `streamlit>=1.34` - Web 应用框架
- `numpy>=1.24` - 数值计算
- `pandas>=2.0` - 数据处理
- `scipy>=1.10` - 科学计算（ODE 求解、优化）
- `matplotlib>=3.7` - 数据可视化

## 许可证

仓库当前未包含 `LICENSE` 文件；如需明确许可证，请补充后再发布/分发。

## 贡献

欢迎提交 Issue 和 Pull Request！

