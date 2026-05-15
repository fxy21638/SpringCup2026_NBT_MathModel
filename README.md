# B题：空气质量的评价和分析

基于 UCI AirQuality 数据集的空气质量综合评价、传感器预测建模与改善建议。

## 项目结构

```
B题/
├── README.md
├── requirements.txt
├── .gitignore
├── 题目/
│   ├── B题 空气质量的评价和分析.doc       # 赛题说明
│   └── B题附件：AirQualityUCI.xlsx        # 原始数据
├── program/
│   ├── read_and_clean.py                  # 步骤1: 数据清洗
│   ├── compute_aqi.py                     # 步骤2: AQI 计算
│   └── analysis.py                        # 步骤3: 综合评价与预测建模
└── output/                                # [生成] 统一输出目录（已加入 .gitignore）
    ├── data_clean.csv                     # 清洗后数据
    ├── data_with_aqi.csv                  # 含 AQI 的数据
    ├── missing_summary.csv                # 缺失值统计
    ├── model_evaluation.csv               # 模型评估结果
    ├── xgb_model.pkl                      # 训练好的 XGBoost 模型
    └── picture/
        ├── missing_heatmap.png            # 缺失值模式（柱状图 + 采样矩阵）
        ├── aqi_distribution.png           # AQI 分布直方图
        ├── aqi_grade_pie.png              # 污染等级饼图
        ├── membership_functions.png       # 梯形隶属度函数曲线
        ├── entropy_weights.png            # 熵权法权重柱状图
        ├── timeseries_aqi.png             # AQI 时序（夏/冬代表性时段）
        ├── timeseries_pollutants.png      # 污染物浓度时序（NMHC 真实值期）
        ├── pollutant_correlation.png      # 污染物相关性热图
        ├── monthly_aqi_boxplot.png        # AQI 月际箱线图
        ├── model_comparison.png           # 模型性能对比
        ├── feature_importance.png         # 特征重要性
        ├── timeseries_prediction.png      # 预测 vs 真实值（夏/冬）
        ├── residual_distribution.png      # 预测残差分布 + Q-Q 图
        ├── sensor_correlation.png         # 传感器与 AQI 相关性
        ├── diurnal_aqi.png                # AQI 日变化模式
        ├── fold_evaluation.png            # 各折交叉验证结果
        ├── tree_depth3.png                # XGBoost 回归树结构示意图
        ├── weight_comparison.png          # FCE 权重对比（熵权 vs PCA vs 混合）
        ├── aqi_weight_comparison.png      # AQI 散点对比（熵权 vs 混合权重）
        ├── shap_gain_comparison.png       # Gain vs SHAP 重要性对比
        ├── shap_summary.png               # SHAP summary plot
        └── vif_pruning.png                # VIF 特征剪枝性能对比
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 步骤1: 数据清洗（-200 → NaN 并插值填充，合并 Date+Time）
python program/read_and_clean.py --input "题目/B题附件：AirQualityUCI.xlsx" --out_csv output/data_clean.csv

# 步骤2: 模糊综合评价计算 AQI
python program/compute_aqi.py

# 步骤3: 综合评价、预测建模与可视化
python program/analysis.py

# 步骤4（可选）: 模型改进分析（PCA-熵权混合、SHAP、VIF 剪枝）
python program/model_improvement.py
```

## 方法概述

### 问题1 — 空气质量综合评价

采用**模糊综合评价模型（FCE）**，贴合空气质量等级边界模糊的本质：

- **评价指标集** U = {CO, NMHC, C6H6, NOx, NO2}（5 项真实污染物，合规）
- **评价等级集** V = {优, 良, 轻度污染, 中度污染, 重度污染}
- **隶属度函数**：梯形/半梯形（基于各污染物真实浓度分位数确定断点，保证最小间隔避免退化）
- **权重向量 W**：熵权法客观赋权
  - NMHC 自动降权至 0.03（填充值信息量低，熵权法天然适应高缺失指标）
  - NOx 权重 0.29, NO2 0.28, CO 0.26, C6H6 0.14
- **模糊合成算子**：M(·,+) 加权平均
- **综合评判**：最大隶属度原则判定等级
- **连续 AQI 得分**：去模糊化加权平均（各等级中心值 × 隶属度），取值 0~100

NMHC 隶属度断点仅使用原始真实值（914 个有效样本）计算，避免填充值导致的断点退化。

### 问题2 — 传感器预测 AQI

使用 **XGBoost + TimeSeriesSplit(5-fold)** 从传感器读数预测 AQI：

- **特征**（6个）：5 个传感器(PT08.S1~S5) + 小时(0-23)
- **每折独立 StandardScaler** + **按时间顺序划分**训练/测试集，杜绝数据泄露
- 基线对比：线性回归（同特征、同验证方法）

### 问题3 — 改善建议

基于特征重要性分析，从交通源管制、工业排放控制、气象条件利用、监测网络优化、公众健康提示五个维度提出建议。

## 数据处理要点

- 缺失值标记 `-200` → NaN（仪器故障/未检出标识）
- 短缺口（≤6 小时）：线性插值（传感器短暂掉线）
- 长缺口（>6 小时）：前向填充（传感器故障期，保守假设浓度不变）
- NMHC 特殊处理：90.2% 缺失率（含 8126 小时连续缺口），经间隙感知填充后保留参与 AQI。隶属度断点基于真实值计算，熵权法自动降权（w≈0.03），填充段信息量低不影响评价
- 异常值策略：**不采用 3σ 准则**。污染物高浓度值属于真实污染事件而非统计"异常值"，予以保留。隶属度函数使用分位数断点，天然对极值鲁棒

## 模型性能

| 模型 | R² | RMSE | MAE |
|------|-----|------|-----|
| 线性回归 | 0.71 | 10.18 | 7.76 |
| XGBoost | **0.73 +/- 0.12** | **9.81 +/- 2.26** | **7.20 +/- 1.47** |

XGBoost 5 折交叉验证详细结果：

| Fold | R² | RMSE | 训练/测试样本 |
|------|-----|------|-------------|
| 1 | 0.702 | 10.06 | 1562/1559 |
| 2 | 0.628 | 10.54 | 3121/1559 |
| 3 | 0.602 | 13.00 | 4680/1559 |
| 4 | 0.833 | 8.41 | 6239/1559 |
| 5 | 0.884 | 7.03 | 7798/1559 |

## 模型改进分析（`model_improvement.py`）

针对多重共线性问题的三项改进实验：

### 1. PCA-熵权混合权重（FCE 改进）

| 方法 | CO | NMHC | C6H6 | NOx | NO2 | 交通源合计 |
|------|-----|------|------|-----|-----|-----------|
| 熵权法 | 0.262 | 0.031 | 0.140 | 0.291 | 0.277 | 0.830 |
| PCA 权重 | 0.203 | 0.162 | 0.205 | 0.218 | 0.213 | 0.633 |
| 混合权重 | 0.232 | 0.096 | 0.172 | 0.254 | 0.245 | 0.732 |

- 混合权重 AQI 与熵权法 AQI 相关度 r = 0.9987，等级一致率 95.6%
- 结论：FCE 对权重扰动高度鲁棒，熵权法可直接使用

### 2. SHAP 特征重要性 vs Gain

| 排名 | Gain 重要性 | SHAP 重要性 |
|------|------------|------------|
| 1 | PT08.S5_O3 (0.574) | PT08.S2_NMHC (7.12) |
| 2 | PT08.S2_NMHC (0.185) | PT08.S5_O3 (6.50) |
| 3 | PT08.S3_NOx (0.082) | PT08.S4_NO2 (3.44) |

- SHAP 对特征共线性更鲁棒，推荐用于重要性归因

### 3. VIF 特征剪枝

| 配置 | 特征数 | R² |
|------|--------|-----|
| 基线（全特征） | 6 | 0.730 |
| SHAP Top3 + hour | 4 | **0.744** |
| 保守剪枝（VIF<5） | 3 | 0.546 |

- SHAP 引导剪枝（4 特征）略优于基线，减少 33% 特征

## 环境依赖

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- openpyxl

## 数据说明

- 来源：UCI Machine Learning Repository — Air Quality Data Set
- 时间：2004 年 3 月 10 日 ~ 2005 年 4 月 4 日
- 粒度：小时级别，共 9,357 条记录
- 缺失值：原始数据中以 -200 标记，已通过线性插值处理
