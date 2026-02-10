# test_data：示例与验证数据

本目录用于提供示例/验证数据的**生成脚本**与配置文件，帮助你先跑通流程再替换真实实验数据。

> 提示：`orthogonal_design_data.csv`、`validation_*.csv` 为脚本生成产物，默认被 `.gitignore` 忽略，仓库内可能不存在。

> 注意：当前 App 支持 **PFR / CSTR / BSTR** 三类反应器。

## 1) 快速示例（推荐）

- `orthogonal_design_data.csv`：PFR 示例数据（A 一级反应，27 组工况，带少量噪声；由脚本生成）  
  - 列包含：`V_m3, T_K, vdot_m3_s, F0_A_mol_s, ... , Fout_A_mol_s`
  - 注意：该示例**只有 `Fout_A_mol_s`** 这一列测量输出，因此在 App 里建议：
    - 目标变量选 `Fout (mol/s)`
    - 目标物种只勾选 `A`
- `generate_orthogonal_design.py`：重新生成上述 PFR 示例数据（覆盖写入 `orthogonal_design_data.csv`）

运行（在项目根目录）：

```bash
python test_data/generate_orthogonal_design.py
```

（如需 CSTR/BSTR 的最小示例）可直接在 App 的「教程/帮助」页下载。

## 2) 复杂验证数据（进阶）

`generate_complex_data.py` 会生成更复杂的验证数据（均带 1% 随机噪声）：

```bash
python test_data/generate_complex_data.py
```

生成文件：

1. `validation_PFR_Mixed.csv`
   - 反应器：PFR
   - 反应数：4（混合动力学：幂律 + 可逆）
   - 工况：不同温度 `T_K` 与体积 `V_m3`

   对应的可导入配置：`validation_PFR_Mixed.json`

2. `validation_Batch_Series.csv`
   - 反应器：BSTR
   - 反应数：4（连串反应 A→B→C→D→E）
   - 工况：不同温度 `T_K` 与时间点 `t_s`

   对应的可导入配置：`validation_Batch_Series.json`

3. `validation_PFR_LH.csv`
   - 反应器：PFR
   - 动力学：Langmuir–Hinshelwood（L-H）

   对应的可导入配置：`validation_PFR_LH.json`
