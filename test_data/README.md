# test_data：示例与验证数据

本目录用于提供**可直接上传到 App** 的示例 CSV，以及生成脚本，帮助你先跑通流程再替换真实实验数据。

> 注意：当前 App 仅支持 **PFR** 与 **BSTR** 两类反应器（不支持 CSTR）。

## 1) 快速示例（推荐）

- `test_data_matched.csv`：A → B 一级反应（PFR），已是可用数据文件  
- `generate_test_data.py`：重新生成上述 PFR 示例数据（会覆盖写入 `test_data_matched.csv`）

运行（在项目根目录）：

```bash
python test_data/generate_test_data.py
```

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

2. `validation_Batch_Series.csv`
   - 反应器：BSTR
   - 反应数：4（连串反应 A→B→C→D→E）
   - 工况：不同温度 `T_K` 与时间点 `t_s`

3. `validation_CSTR_LH.csv`（仅供对照/留存）
   - 该文件是 CSTR 稳态数据，列名格式被“伪装成 PFR 的 Fout”以便读取
   - 由于 App 没有 CSTR 模型，**不建议用于拟合**（除非你刻意做“错误模型拟合”的对照实验）
