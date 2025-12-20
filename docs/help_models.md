# 动力学模型与参数含义

## 1) 幂律（power-law）

$$r_j = k_j(T)\prod_i C_i^{n_{i,j}}$$

$$k_j(T)=k_{0,j}\exp\left(-\frac{E_{a,j}}{RT}\right)$$

- $r_j$：第 $j$ 个反应速率 [mol·m⁻³·s⁻¹]
- $C_i$：物种 $i$ 浓度 [mol·m⁻³]
- $n_{i,j}$：反应级数（可拟合，也可固定）
- $k_{0,j}$：指前因子（单位取决于总级数，这是正常现象）
- $E_{a,j}$：活化能 [J·mol⁻¹]

## 2) Langmuir–Hinshelwood（langmuir_hinshelwood）

$$r_j=\frac{k_j(T)\prod_i C_i^{n_{i,j}}}{\left(1+\sum_i K_i(T)C_i\right)^{m_j}}$$

$$K_i(T)=K_{0,i}\exp\left(-\frac{E_{a,K,i}}{RT}\right)$$

- $K_{0,i}$：吸附常数指前因子（若 $C$ 用 mol/m³，则 $K$ 常见量纲为 m³/mol）
- $E_{a,K,i}$：吸附“活化能/吸附热”参数 [J·mol⁻¹]（允许为负）
- $m_j$：抑制指数（每个反应一个）

## 3) 可逆反应（reversible）

$$r_j=k_j^+(T)\prod_i C_i^{n_{i,j}^+}-k_j^-(T)\prod_i C_i^{n_{i,j}^-}$$

$$k_j^{\pm}(T)=k_{0,j}^{\pm}\exp\left(-\frac{E_{a,j}^{\pm}}{RT}\right)$$

App 中把“正向”和“逆向”的 $k_0,E_a,n$ 分开输入、分开勾选是否拟合。

## 4) 反应器方程（与 App 一致）

### PFR（恒定体积流量假设）

$$\frac{dF_i}{dV} = \sum_j \nu_{i,j} r_j,\quad C_i=\frac{F_i}{\dot{v}}$$

### Batch（恒体积、无进出料）

$$\frac{dC_i}{dt} = \sum_j \nu_{i,j} r_j$$

