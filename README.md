# Kinetics_app - 反应动力学参数拟合工具（PFR / Batch）

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-red.svg)](https://streamlit.io/)

一个基于 **Python + Streamlit** 的网页工具，用于 **平推流反应器（Plug Flow Reactor, PFR）** 与 **间歇釜式反应器（Batch）** 的反应动力学参数拟合与模拟。

## 功能特性

- 📊 **多物种、多反应系统**：支持自定义物种数量和反应数量
- 🧪 **多反应器类型**：支持 PFR 与 Batch
- 🔬 **多种动力学模型**：幂律（Power-law）、Langmuir-Hinshelwood（吸附抑制）、可逆反应（Reversible）
- 🎯 **灵活的参数拟合**：可选择性拟合 k₀、Eₐ 和反应级数
- 📈 **多种目标变量**：支持出口摩尔流量、出口浓度或转化率作为拟合目标
- 🚀 **高级拟合选项**：多起点拟合、参数缩放、可调有限差分步长
- 📉 **可视化结果**：奇偶校验图（Parity Plot）、误差图，以及沿程/随时间剖面图（PFR: $F_i(V)$/$C_i(V)$；Batch: $C_i(t)$）
- 📥 **一键导出**：拟合参数/对比数据/报告表（CSV）与图像（PNG/SVG）
- ⚙️ **配置管理**：配置导入/导出（JSON），并可在启动时自动恢复上次配置

## 理论背景

### 反应动力学模型（可选）

对于第 $j$ 个反应，提供以下动力学模型：

**(1) 幂律 (Power-law)**

$$r_j = k_j(T) \prod_i C_i^{n_{i,j}}$$

**(2) Langmuir-Hinshelwood（吸附抑制）**

$$r_j = \\frac{k_j(T) \\prod_i C_i^{n_{i,j}}}{\\left(1 + \\sum_i K_i(T) C_i\\right)^{m_j}}$$

$$K_i(T) = K_{0,i} \\exp\\left(-\\frac{E_{a,K,i}}{RT}\\right)$$

**(3) 可逆反应 (Reversible)**

$$r_j = k_j^+(T) \\prod_i C_i^{n_{i,j}^+} - k_j^-(T) \\prod_i C_i^{n_{i,j}^-}$$

其中 Arrhenius 形式为：

$$k_j^{\\pm}(T) = k_{0,j}^{\\pm} \\exp\\left(-\\frac{E_{a,j}^{\\pm}}{RT}\\right)$$

### PFR 设计方程

$$\frac{dF_i}{dV} = \sum_{j=1}^{N_{rxn}} \nu_{i,j} r_j$$

其中浓度与摩尔流量的关系为（液相/恒定体积流量假设）：

$$C_i = \frac{F_i}{\dot{v}}$$

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

- Python 3.10+（`app.py` 使用了 `X | None` 类型标注语法）
- 依赖库见 `requirements.txt`

### 安装步骤

```bash
# 克隆或下载项目
cd Kinetics_app

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

运行后，浏览器会自动打开应用界面（默认地址：http://localhost:8501）。

## 快速验证（推荐先跑通一次）

1) 生成 PFR 示例数据（也可直接在 App 的“教程/帮助”里下载）：

```bash
python test_data/generate_test_data.py
```

2) 打开 App：`streamlit run app.py`

3) 在“教程/帮助”页按推荐流程操作（下载示例数据/模板 → 上传 → 选择目标 → 开始拟合）。

## 使用说明

### 1. 反应定义

- 输入物种名称（逗号分隔，如 `A,B,C`）
- 选择反应器类型（PFR / Batch）与动力学模型（Power-law / L-H / Reversible）
- 设置反应数量
- 填写化学计量数矩阵 ν（反应物为负，生成物为正）
- 设置反应级数矩阵 n
- 输入 k₀ 和 Eₐ 的初始猜测值

### 2. 实验数据

上传 CSV 文件，必须包含以下输入列（按反应器类型）：

**PFR 输入列**

| 列名 | 含义 | 单位 |
|:---|:---|:---|
| `V_m3` | 反应器体积 | m³ |
| `T_K` | 反应温度 | K |
| `vdot_m3_s` | 体积流量 | m³/s |
| `F0_<物种名>_mol_s` | 各物种入口摩尔流量 | mol/s |

**Batch 输入列**

| 列名 | 含义 | 单位 |
|:---|:---|:---|
| `t_s` | 反应时间 | s |
| `T_K` | 反应温度 | K |
| `C0_<物种名>_mol_m3` | 各物种初始浓度 | mol/m³ |

测量值列（任选一种类型；允许缺测，但缺测会在拟合中被赋予较大惩罚残差）：
- `Fout_<物种名>_mol_s`：出口摩尔流量 [mol/s]（仅 PFR）
- `Cout_<物种名>_mol_m3`：出口浓度 [mol/m³]
- `X_<物种名>`：转化率 [-]

可在应用内下载 CSV 数据模板。

### 3. 参数拟合

- 选择拟合目标变量类型（Fout / Cout / X）
- 选择参与目标函数的物种
- 设置参数边界和高级拟合选项
- 点击"开始拟合"

## 拟合算法说明（与代码一致）

- 数值积分：`scipy.integrate.solve_ivp`（可选 `RK45 / BDF / Radau`）
- 参数拟合：`scipy.optimize.least_squares`（`method="trf"`）
- 鲁棒性选项：多起点（multi-start）、有限差分步长 `diff_step`、参数缩放 `x_scale="jac"`
- 残差加权：支持“不加权”与“按测量值相对误差 $1/|y|$”

## 重要假设与限制（建议使用前确认）

- **PFR**：采用液相/恒定体积流量假设 $C_i = F_i/\\dot{v}$，未包含压降与体积流量沿程变化。
- **Batch**：采用恒体积、无进出料的 $dC/dt=\\nu r$ 模型；Batch 模式不支持 `Fout` 作为拟合输出。
- **缺测数据**：测量值为 NaN 的行会被赋予较大惩罚残差（相当于提醒“这行缺数据”），但仍会参与优化流程。
- **结果缓存**：拟合完成后会缓存拟合参数与当次配置/数据，结果展示与导出将锁定于该次拟合；如需应用新配置/新数据，请清除缓存并重新拟合。
- **负级数**：当反应级数为负且浓度趋近 0 时，为避免 $0^{n<0}$ 发散，内部会对浓度加很小的下限（`1e-30 mol/m³`）；这可能影响极低浓度区间的拟合稳定性。
- **性能**：每一次目标函数评估都需要对每一行实验条件进行 `solve_ivp`，当数据量很大（例如 >1000 行）会明显变慢。

## 项目结构

```
Kinetics_app/
├── app.py                  # Streamlit 主应用程序
├── kinetics.py             # 动力学：幂律 / L-H / 可逆速率
├── reactors.py             # 反应器：PFR/Batch 数值积分 + 剖面
├── fitting.py              # 拟合工具：参数打包/解包、边界、单行预测
├── ui_help.py              # 教程/帮助页内容与示例数据
├── requirements.txt        # Python 依赖
├── README.md              # 项目说明文档
├── TODO.md                # 待办事项列表
├── .streamlit/
│   └── config.toml        # Streamlit 主题配置
└── test_data/
    ├── generate_test_data.py   # 测试数据生成脚本
    └── test_data_matched.csv   # 示例测试数据
```

## 测试数据

`test_data/` 文件夹包含用于测试的示例数据：

- **`test_data_matched.csv`**：A → B 一级反应的模拟实验数据
- **`generate_test_data.py`**：数据生成脚本，可自定义动力学参数

测试参数：
- 反应：A → B（一级反应）
- k₀ = 1×10⁶ s⁻¹
- Eₐ = 5×10⁴ J/mol
- T = 350 K

## 依赖库

- `streamlit>=1.34` - Web 应用框架
- `numpy>=1.24` - 数值计算
- `pandas>=2.0` - 数据处理
- `scipy>=1.10` - 科学计算（ODE 求解、优化）
- `matplotlib>=3.7` - 数据可视化

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
