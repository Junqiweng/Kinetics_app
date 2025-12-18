# Kinetics_app - PFR 反应动力学参数拟合工具

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-red.svg)](https://streamlit.io/)

一个基于 **Python + Streamlit** 的网页工具，用于 **平推流反应器（Plug Flow Reactor, PFR）** 的反应动力学参数拟合。

## 功能特性

- 📊 **多物种、多反应系统**：支持自定义物种数量和反应数量
- 🔬 **幂律动力学模型**：结合 Arrhenius 温度依赖关系
- 🎯 **灵活的参数拟合**：可选择性拟合 k₀、Eₐ 和反应级数
- 📈 **多种目标变量**：支持出口摩尔流量、出口浓度或转化率作为拟合目标
- 🚀 **高级拟合选项**：多起点拟合、参数缩放、可调有限差分步长
- 📉 **可视化结果**：奇偶校验图（Parity Plot）和误差分布图

## 理论背景

### 反应动力学模型

对于第 $j$ 个反应，反应速率采用幂函数定律结合 Arrhenius 方程：

$$r_j = k_j(T) \prod_i C_i^{n_{i,j}}$$

$$k_j(T) = k_{0,j} \exp\left(-\frac{E_{a,j}}{RT}\right)$$

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

- Python 3.9+
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

## 使用说明

### 1. 反应定义

- 输入物种名称（逗号分隔，如 `A,B,C`）
- 设置反应数量
- 填写化学计量数矩阵 ν（反应物为负，生成物为正）
- 设置反应级数矩阵 n
- 输入 k₀ 和 Eₐ 的初始猜测值

### 2. 实验数据

上传 CSV 文件，必须包含以下列：

| 列名 | 含义 | 单位 |
|:---|:---|:---|
| `V_m3` | 反应器体积 | m³ |
| `T_K` | 反应温度 | K |
| `vdot_m3_s` | 体积流量 | m³/s |
| `F0_<物种名>_mol_s` | 各物种入口摩尔流量 | mol/s |

测量值列（任选一种类型）：
- `Fout_<物种名>_mol_s`：出口摩尔流量 [mol/s]
- `Cout_<物种名>_mol_m3`：出口浓度 [mol/m³]
- `X_<物种名>`：转化率 [-]

可在应用内下载 CSV 数据模板。

### 3. 参数拟合

- 选择拟合目标变量类型（Fout / Cout / X）
- 选择参与目标函数的物种
- 设置参数边界和高级拟合选项
- 点击"开始拟合"

## 项目结构

```
Kinetics_app/
├── app.py                  # Streamlit 主应用程序
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
