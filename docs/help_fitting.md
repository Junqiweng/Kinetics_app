# 拟合与数值设置（让拟合更稳、更快）

## 1) 拟合在做什么？

App 使用非线性最小二乘（`scipy.optimize.least_squares`）最小化：

$$\Phi(\theta)=\frac{1}{2}\sum r_i(\theta)^2,\quad r_i=y_i^{pred}-y_i^{meas}$$

其中 $\theta$ 是你勾选了 `Fit_*` 的参数集合。

## 2) ODE 求解器怎么选？

- 非刚性/简单问题：`RK45`
- 刚性/收敛困难：`BDF` 或 `Radau`

经验：一旦遇到 `solve_ivp失败` 或“拟合很慢/不稳定”，优先把求解器换成 `BDF`。

## 3) `diff_step`（数值差分步长）

Jacobian 通过数值差分得到，`diff_step` 太小会更“敏感”，在噪声/非光滑情况下可能导致：

- 拟合抖动
- 看起来“不动”

建议从 `1e-2 ~ 1e-3` 试起，再逐步调小。

## 4) Multi-start（多起点）

当你拟合参数很多时（尤其包含 $E_a$ 与级数）：

- 勾选 `Multi-start`
- `Start Points： 5~20`

它会更慢，但更不容易被局部最小值困住。

## 5) `x_scale='jac'`

开启后会根据 Jacobian 自动缩放参数尺度，通常能提升稳定性（建议保持开启）。

## 6) `max_step_fraction`（ODE 步长限制）

用于限制 `solve_ivp(max_step)`：

- 设为 `0`：不限制
- 反应很快、曲线很陡时：适当减小可能更稳（但更慢）

