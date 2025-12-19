import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os
import sys


def log(msg):
    with open("generation.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


# Ensure stdout is flushed
sys.stdout.reconfigure(line_buffering=True)
log("Script started...")

# ==========================================
# 通用常数与设置
# ==========================================
R_GAS = 8.314  # J/(mol·K)
np.random.seed(42)  # 固定随机种子

# ==========================================
# 场景 1: PFR 反应器 (4个反应, 混合动力学: 幂函数 + 可逆)
# ==========================================
# 反应网络:
# 1. A -> B       (r1 = k1 * CA)
# 2. B -> C       (r2 = k2 * CB)
# 3. 2A -> D      (r3 = k3 * CA^2)
# 4. C <=> E      (r4 = k4f * CC - k4r * CE)


def generate_pfr_data():
    print("正在生成 PFR 验证数据 (PFR_Mixed_Kinetics)...")

    # 动力学参数 (k = k0 * exp(-Ea/RT))
    # 反应1
    k0_1, Ea_1 = 1e6, 50000
    # 反应2
    k0_2, Ea_2 = 5e5, 45000
    # 反应3
    k0_3, Ea_3 = 1e4, 60000  # 二级反应，k0单位不同
    # 反应4 (正向)
    k0_4f, Ea_4f = 2e6, 55000
    # 反应4 (逆向) - 热力学一致性需注意，这里简化给定
    k0_4r, Ea_4r = 1e5, 65000

    def get_rates(C, T):
        CA, CB, CC, CD, CE = C

        k1 = k0_1 * np.exp(-Ea_1 / (R_GAS * T))
        k2 = k0_2 * np.exp(-Ea_2 / (R_GAS * T))
        k3 = k0_3 * np.exp(-Ea_3 / (R_GAS * T))
        k4f = k0_4f * np.exp(-Ea_4f / (R_GAS * T))
        k4r = k0_4r * np.exp(-Ea_4r / (R_GAS * T))

        r1 = k1 * CA
        r2 = k2 * CB
        r3 = k3 * CA**2
        r4 = k4f * CC - k4r * CE

        return r1, r2, r3, r4

    def pfr_ode(V, F, T, vdot):
        # Unpack Molar Flows
        FA, FB, FC, FD, FE = F
        # Concentrations
        CA = FA / vdot
        CB = FB / vdot
        CC = FC / vdot
        CD = FD / vdot
        CE = FE / vdot

        # Rates
        r1, r2, r3, r4 = get_rates([CA, CB, CC, CD, CE], T)

        # Material Balances (dFi/dV = sum(nu_ij * r_j))
        dFA = -r1 - 2 * r3
        dFB = r1 - r2
        dFC = r2 - r4
        dFD = r3
        dFE = r4

        return [dFA, dFB, dFC, dFD, dFE]

    # 操作条件
    vdot = 0.001  # m3/s
    F0_A = 1.0  # mol/s
    F0_others = 0.0

    # 生成数据点 (变化温度和体积)
    data = []

    T_list = [300, 320, 340, 360]
    V_list = [0.01, 0.05, 0.1, 0.2, 0.4]  # m3

    for T in T_list:
        for V_vol in V_list:
            # 积分
            sol = solve_ivp(
                lambda V, F: pfr_ode(V, F, T, vdot),
                [0, V_vol],
                [F0_A, F0_others, F0_others, F0_others, F0_others],
                method="RK45",
            )

            F_out = sol.y[:, -1]

            # 添加 1% 随机噪声
            noise_factor = 1 + 0.01 * np.random.randn(5)
            F_out_noisy = F_out * noise_factor
            F_out_noisy = np.maximum(F_out_noisy, 0)  # 物理约束

            # 构造符合 App 要求的列名
            # PFR/Flow Required: V_m3, T_K, vdot_m3_s, F0_{spec}_mol_s
            # Measured: Fout_{spec}_mol_s
            row = {
                "V_m3": V_vol,
                "T_K": T,
                "vdot_m3_s": vdot,
                "F0_A_mol_s": F0_A,
                "F0_B_mol_s": 0.0,
                "F0_C_mol_s": 0.0,
                "F0_D_mol_s": 0.0,
                "F0_E_mol_s": 0.0,
                "Fout_A_mol_s": F_out_noisy[0],
                "Fout_B_mol_s": F_out_noisy[1],
                "Fout_C_mol_s": F_out_noisy[2],
                "Fout_D_mol_s": F_out_noisy[3],
                "Fout_E_mol_s": F_out_noisy[4],
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("test_data/validation_PFR_Mixed.csv", index=False)
    log("  -> 已保存 validation_PFR_Mixed.csv, 数据点数: " + str(len(df)))


# ==========================================
# 场景 2: CSTR 反应器 (4个反应, L-H 吸附动力学)
# ==========================================
# 反应网络:
# 1. A -> B  (L-H: r1 = k1*CA / (1 + K1*CA))
# 2. B -> C  (L-H: r2 = k2*CB / (1 + K2*CB))
# 3. A -> D  (Power: r3 = k3*CA)
# 4. D -> E  (Power: r4 = k4*CD)


def generate_cstr_data():
    log("正在生成 CSTR 验证数据 (CSTR_LH_Kinetics)...")

    # 参数
    k0_1, Ea_1 = 5e6, 60000
    K_ads0_1, dH_ads_1 = 1e-2, -10000  # 吸附常数 Arrhenius形式 K = K0 * exp(-dH/RT)

    k0_2, Ea_2 = 4e6, 62000
    K_ads0_2, dH_ads_2 = 2e-2, -12000

    k0_3, Ea_3 = 1e5, 70000
    k0_4, Ea_4 = 2e5, 72000

    def solve_cstr(V, T, vdot, F_in):
        CA0 = F_in[0] / vdot  # A进料浓度

        k1 = k0_1 * np.exp(-Ea_1 / (R_GAS * T))
        K1 = K_ads0_1 * np.exp(-dH_ads_1 / (R_GAS * T))

        k2 = k0_2 * np.exp(-Ea_2 / (R_GAS * T))
        K2 = K_ads0_2 * np.exp(-dH_ads_2 / (R_GAS * T))

        k3 = k0_3 * np.exp(-Ea_3 / (R_GAS * T))
        k4 = k0_4 * np.exp(-Ea_4 / (R_GAS * T))

        # 求解方程: F_in - F_out + r * V = 0
        # F_out = C_out * vdot
        # 0 = F_in/vdot - C_out + r * (V/vdot)
        # 0 = C_in - C_out + r * tau
        tau = V / vdot

        def equations(C):
            CA, CB, CC, CD, CE = C

            # 速率
            r1 = k1 * CA / (1 + K1 * CA)
            r2 = k2 * CB / (1 + K2 * CB)
            r3 = k3 * CA
            r4 = k4 * CD

            # 物料衡算
            # A:
            eqA = (F_in[0] / vdot) - CA + (-r1 - r3) * tau
            # B:
            eqB = (F_in[1] / vdot) - CB + (r1 - r2) * tau
            # C:
            eqC = (F_in[2] / vdot) - CC + (r2) * tau
            # D:
            eqD = (F_in[3] / vdot) - CD + (r3 - r4) * tau
            # E:
            eqE = (F_in[4] / vdot) - CE + (r4) * tau

            return [eqA, eqB, eqC, eqD, eqE]

        # 初始猜测
        guess = [CA0 / 2, CA0 / 4, CA0 / 4, CA0 / 10, CA0 / 10]
        C_out = fsolve(equations, guess)
        C_out = np.maximum(C_out, 0)  # 物理约束
        return C_out * vdot  # 返回流量

    # 数据生成
    vdot = 0.005  # m3/s
    F0_A = 2.0  # mol/s

    data = []
    # 更多工况
    T_list = [330, 340, 350, 360]
    V_list = [0.1, 0.2, 0.5, 1.0, 1.5]

    for T in T_list:
        for V_vol in V_list:
            F_out = solve_cstr(V_vol, T, vdot, [F0_A, 0, 0, 0, 0])

            # 噪声
            noise = 1 + 0.01 * np.random.randn(5)
            F_out_noisy = F_out * noise
            F_out_noisy = np.maximum(F_out_noisy, 0)

            # CSTR 也使用 PFR 格式列名以便 App 读取(视为 Fout)
            row = {
                "V_m3": V_vol,
                "T_K": T,
                "vdot_m3_s": vdot,
                "F0_A_mol_s": F0_A,
                "F0_B_mol_s": 0.0,
                "F0_C_mol_s": 0.0,
                "F0_D_mol_s": 0.0,
                "F0_E_mol_s": 0.0,
                "Fout_A_mol_s": F_out_noisy[0],
                "Fout_B_mol_s": F_out_noisy[1],
                "Fout_C_mol_s": F_out_noisy[2],
                "Fout_D_mol_s": F_out_noisy[3],
                "Fout_E_mol_s": F_out_noisy[4],
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("test_data/validation_CSTR_LH.csv", index=False)
    log("  -> 已保存 validation_CSTR_LH.csv, 数据点数: " + str(len(df)))


# ==========================================
# 场景 3: 间歇反应器 (Batch) (4个反应, 连串反应)
# ==========================================
# 反应网络:
# 1. A ->B
# 2. B -> C
# 3. C -> D
# 4. D -> E


def generate_batch_data():
    log("正在生成 Batch 验证数据 (Batch_Series)...")

    # 参数
    k0_list = [1e4, 5e4, 2e4, 1e4]
    Ea_list = [40000, 45000, 42000, 40000]

    def batch_ode(t, C, T):
        CA, CB, CC, CD, CE = C

        Ks = []
        for i in range(4):
            k = k0_list[i] * np.exp(-Ea_list[i] / (R_GAS * T))
            Ks.append(k)

        r1 = Ks[0] * CA
        r2 = Ks[1] * CB
        r3 = Ks[2] * CC
        r4 = Ks[3] * CD

        dCA = -r1
        dCB = r1 - r2
        dCC = r2 - r3
        dCD = r3 - r4
        dCE = r4

        return [dCA, dCB, dCC, dCD, dCE]

    # 初始条件 (浓度 mol/m3)
    C0_A = 1000.0  # mol/m3

    data = []

    T_list = [298, 308, 318, 328]
    # Batch里自变量是 时间 Time
    Time_list = [60, 120, 240, 480, 960, 1500]  # seconds

    for T in T_list:
        sol = solve_ivp(
            lambda t, C: batch_ode(t, C, T),
            [0, max(Time_list)],
            [C0_A, 0, 0, 0, 0],
            method="RK45",
            t_eval=Time_list,
        )

        for i, t_val in enumerate(sol.t):
            C_out = sol.y[:, i]

            # 噪声
            noise = 1 + 0.01 * np.random.randn(5)
            C_out_noisy = C_out * noise
            C_out_noisy = np.maximum(C_out_noisy, 0)

            # Batch Required: t_s, T_K, C0_{spec}_mol_m3
            # Measured: Cout_{spec}_mol_m3
            row = {
                "t_s": t_val,
                "T_K": T,
                "C0_A_mol_m3": C0_A,
                "C0_B_mol_m3": 0.0,
                "C0_C_mol_m3": 0.0,
                "C0_D_mol_m3": 0.0,
                "C0_E_mol_m3": 0.0,
                "Cout_A_mol_m3": C_out_noisy[0],
                "Cout_B_mol_m3": C_out_noisy[1],
                "Cout_C_mol_m3": C_out_noisy[2],
                "Cout_D_mol_m3": C_out_noisy[3],
                "Cout_E_mol_m3": C_out_noisy[4],
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("test_data/validation_Batch_Series.csv", index=False)
    log("  -> 已保存 validation_Batch_Series.csv, 数据点数: " + str(len(df)))


if __name__ == "__main__":
    generate_pfr_data()
    generate_cstr_data()
    generate_batch_data()
