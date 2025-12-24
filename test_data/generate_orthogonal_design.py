# 文件作用：生成用于验证拟合功能的正交实验（L27）示例数据，并导出为 CSV 供应用上传测试。

"""
生成正交实验数据 - 温度、入口摩尔流率、反应体积三因素三水平

使用方法:
1. 运行此脚本: python generate_orthogonal_design.py
2. 生成的 CSV 文件将保存为 orthogonal_design_data.csv
3. 将 CSV 文件上传到 Streamlit 应用进行拟合测试
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os

R_GAS = 8.314462618  # 气体常数 [J/(mol·K)]

# ========== 动力学参数（基础设置）==========
k0_true = 1e6  # 指前因子 [1/s]
Ea_true = 5e4  # 活化能 [J/mol]

# ========== 正交实验：三因素三水平 ==========
# 因素1：温度 [K]
temps_K = [330, 350, 370]

# 因素2：入口摩尔流率 [mol/s]
F0_A_values = [0.005, 0.01, 0.02]

# 因素3：反应体积 [m³]
volumes_m3 = [0.001, 0.005, 0.01]

# ========== 标准正交表 L27(3^13) - 27个实验 ==========
# 每行代表一个实验组合
# 列对应：因素1(温度), 因素2(流率), 因素3(体积)
# 数值1,2,3 对应各因素的第1,2,3个水平
# 正交表 L27 是 3^3 全因子设计的完整正交表

orthogonal_table = np.array(
    [
        [1, 1, 1],  # 实验1
        [1, 1, 2],  # 实验2
        [1, 1, 3],  # 实验3
        [1, 2, 1],  # 实验4
        [1, 2, 2],  # 实验5
        [1, 2, 3],  # 实验6
        [1, 3, 1],  # 实验7
        [1, 3, 2],  # 实验8
        [1, 3, 3],  # 实验9
        [2, 1, 1],  # 实验10
        [2, 1, 2],  # 实验11
        [2, 1, 3],  # 实验12
        [2, 2, 1],  # 实验13
        [2, 2, 2],  # 实验14
        [2, 2, 3],  # 实验15
        [2, 3, 1],  # 实验16
        [2, 3, 2],  # 实验17
        [2, 3, 3],  # 实验18
        [3, 1, 1],  # 实验19
        [3, 1, 2],  # 实验20
        [3, 1, 3],  # 实验21
        [3, 2, 1],  # 实验22
        [3, 2, 2],  # 实验23
        [3, 2, 3],  # 实验24
        [3, 3, 1],  # 实验25
        [3, 3, 2],  # 实验26
        [3, 3, 3],  # 实验27
    ]
)

print("=" * 70)
print("正交实验设计：3因素3水平（全因子设计）")
print("=" * 70)
print("\n温度水平 (T_K):")
for i, t in enumerate(temps_K, 1):
    print(f"  水平{i}: {t} K")

print("\n入口摩尔流率水平 (F0_A [mol/s]):")
for i, f in enumerate(F0_A_values, 1):
    print(f"  水平{i}: {f} mol/s")

print("\n反应体积水平 (V [m³]):")
for i, v in enumerate(volumes_m3, 1):
    print(f"  水平{i}: {v} m³")

print("\n" + "=" * 70)
print("实验计划 (L27 正交表 - 全因子设计):")
print("=" * 70)
print(f"{'实验编号':^6} | {'T [K]':^8} | {'F0_A [mol/s]':^12} | {'V [m³]':^10}")
print("-" * 50)

rows = []
np.random.seed(42)

for exp_idx, (idx_t, idx_f, idx_v) in enumerate(orthogonal_table, 1):
    # 从指标获取实际的参数值 (索引从0开始)
    T_K = temps_K[idx_t - 1]
    F0_A = F0_A_values[idx_f - 1]
    V = volumes_m3[idx_v - 1]

    print(f"{exp_idx:^6d} | {T_K:^8.0f} | {F0_A:^12.4f} | {V:^10.5f}")

    # ========== 计算速率常数 ==========
    k_T = k0_true * np.exp(-Ea_true / (R_GAS * T_K))

    # ========== 定义 PFR 微分方程 ==========
    # 流动反应器（PFR）微分方程：dF_A/dV = -r = -k * C_A = -k * F_A / vdot
    # 注：此处假设体积流量恒定（假设密度恒定）
    # 实际上，如果没有指定体积流量，使用恒定入口摩尔流率进行计算

    def ode_pfr_ortho(V_var, F_vars):
        """PFR 动力学方程（一级反应 A → B）"""
        F_A = max(F_vars[0], 0)
        # 反应速率常数
        k = k0_true * np.exp(-Ea_true / (R_GAS * T_K))
        # 浓度（假设体积流量为恒定参考值）
        C_A = F_A / 1e-4  # 参考体积流量 1e-4 m³/s
        # 反应速率
        r = k * C_A
        # 摩尔流量导数
        dF_dV = -r
        return [dF_dV]

    # ========== 求解 PFR ==========
    try:
        sol = solve_ivp(
            ode_pfr_ortho,
            [0, V],
            [F0_A],
            method="RK45",
            rtol=1e-8,
            atol=1e-12,
            dense_output=True,
        )
        F_out = sol.y[0, -1]
    except Exception as e:
        print(f"  警告：实验{exp_idx}求解失败 - {e}")
        F_out = F0_A

    # ========== 添加实验噪声（1%）==========
    noise = 0.01 * np.random.randn()
    F_out_noisy = F_out * (1 + noise)
    F_out_noisy = max(F_out_noisy, 0)

    # 计算转化率
    X_A = (F0_A - F_out_noisy) / F0_A if F0_A > 0 else 0

    # ========== 保存数据行 ==========
    rows.append(
        {
            "V_m3": V,
            "T_K": T_K,
            "vdot_m3_s": 1e-4,  # 假设恒定体积流量
            "F0_A_mol_s": F0_A,
            "F0_B_mol_s": 0.0,
            "F0_C_mol_s": 0.0,
            "Fout_A_mol_s": F_out_noisy,
        }
    )

# ========== 保存为 CSV ==========
df = pd.DataFrame(rows)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "orthogonal_design_data.csv")
df.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print(f"数据已保存到: {output_path}")
print("=" * 70)

# ========== 显示统计信息 ==========
print("\n生成的数据统计：")
print(f"{'参数':^15} | {'最小值':^12} | {'最大值':^12} | {'平均值':^12}")
print("-" * 55)
for col in ["V_m3", "T_K", "F0_A_mol_s", "Fout_A_mol_s"]:
    print(
        f"{col:^15} | {df[col].min():^12.6f} | {df[col].max():^12.6f} | {df[col].mean():^12.6f}"
    )

print(f"\n总实验数: {len(df)} 组")
print(f"CSV 列: {list(df.columns)}")
