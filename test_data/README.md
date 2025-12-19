# 验证数据集生成说明

该目录下包含两个生成脚本：

1. `generate_test_data.py`: 生成简单的 PFR 一级反应测试数据。
2. `generate_complex_data.py`: 生成复杂的验证数据集，包含三种反应器类型和复杂的动力学形式。

## 如何生成数据

请在项目根目录下运行以下命令：

```bash
python test_data/generate_complex_data.py
```

## 生成的数据文件

脚本运行后将生成以下三个 CSV 文件：

1. **validation_PFR_Mixed.csv**
   - 反应器: PFR
   - 反应数: 4个 (混合动力学: 幂函数 + 可逆)
   - 包含温度和体积的变化
2. **validation_CSTR_LH.csv**
   - 反应器: CSTR
   - 反应数: 4个 (L-H 吸附动力学 + 幂函数)
   - 包含温度和体积的变化

3. **validation_Batch_Series.csv**
   - 反应器: Batch (间歇反应器)
   - 反应数: 4个 (连串反应 A->B->C->D->E)
   - 包含温度和时间的变化

所有数据均已添加 1% 的随机噪声以模拟实验误差。
