"""
生成与 App 默认设置匹配的测试数据

使用方法:
1. 运行此脚本: python generate_test_data.py
2. 生成的CSV文件将保存在同目录下
3. 将CSV文件上传到 Streamlit App 进行拟合测试
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os

R_GAS = 8.314462618  # J/(mol·K)

# ========== 动力学参数（App 默认设置）==========
# 使用合理的动力学参数
k0_true = 1e6  # 指前因子 [1/s]
Ea_true = 5e4  # 活化能 [J/mol]

# ========== 操作条件 ==========
T_K = 350.0  # 温度 [K]
vdot_m3_s = 1e-4  # 体积流量 [m³/s]
F0_A = 0.01  # A 入口摩尔流量 [mol/s]

# 计算速率常数
k_T = k0_true * np.exp(-Ea_true / (R_GAS * T_K))
print(f"动力学参数：")
print(f"  k0 = {k0_true:.2e} 1/s")
print(f"  Ea = {Ea_true:.2e} J/mol")
print(f"  T  = {T_K} K")
print(f"  k(T) = k0 * exp(-Ea/RT) = {k_T:.6f} 1/s")
print()

# ========== PFR 模拟：A → B 一级反应 ==========
# 化学计量数：A=-1, B=+1
# 反应级数：n_A=1
# 速率方程：r = k * C_A


def ode_pfr(V, F):
    """dF_A/dV = -r = -k * C_A = -k * F_A / vdot"""
    F_A = max(F[0], 0)
    C_A = F_A / vdot_m3_s
    r = k_T * C_A
    dF_dV = -r
    return [dF_dV]


# 模拟不同反应器体积
volumes = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.015, 0.02]

print("模拟结果：")
print(f"{'V [m^3]':^10} | {'F_A,out':^12} | {'X_A':^10}")
print("-" * 40)

rows = []
np.random.seed(42)
for V in volumes:
    sol = solve_ivp(ode_pfr, [0, V], [F0_A], method="RK45", rtol=1e-8, atol=1e-12)
    F_out = sol.y[0, -1]
    X_A = (F0_A - F_out) / F0_A

    # 添加1%噪声模拟实验误差
    F_out_noisy = F_out * (1 + 0.01 * np.random.randn())
    F_out_noisy = max(F_out_noisy, 0)

    print(f"{V:^10.4f} | {F_out_noisy:^12.6f} | {X_A:^10.2%}")

    rows.append(
        {
            "V_m3": V,
            "T_K": T_K,
            "vdot_m3_s": vdot_m3_s,
            "F0_A_mol_s": F0_A,
            "F0_B_mol_s": 0.0,
            "F0_C_mol_s": 0.0,
            "Fout_A_mol_s": F_out_noisy,
        }
    )

# 保存为 CSV（相对于脚本所在目录）
df = pd.DataFrame(rows)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "test_data_matched.csv")
df.to_csv(output_path, index=False)
