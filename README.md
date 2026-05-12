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
        ├── missing_heatmap.png            # 缺失值热图
        ├── aqi_distribution.png           # AQI 分布直方图
        ├── aqi_grade_pie.png              # 污染等级饼图
        ├── timeseries_aqi.png             # AQI 时间序列
        ├── timeseries_pollutants.png      # 污染物浓度时序
        ├── model_comparison.png           # 模型性能对比
        ├── feature_importance.png         # 特征重要性
        ├── timeseries_prediction.png      # 预测 vs 真实值
        └── fold_evaluation.png            # 各折交叉验证结果
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 步骤1: 数据清洗（-200 → NaN 并插值填充，合并 Date+Time）
python program/read_and_clean.py --input "题目/B题附件：AirQualityUCI.xlsx" --out_csv output/data_clean.csv

# 步骤2: 计算连续百分位 AQI
python program/compute_aqi.py

# 步骤3: 综合评价、预测建模与可视化
python program/analysis.py
```

## 方法概述

### 问题1 — 空气质量指标设计

采用**滑动窗口百分位法**构建综合 AQI（0~100 量纲），杜绝时序泄漏：

- 对 5 种参考污染物（CO, NMHC, C6H6, NOx, NO2），使用 expanding window 计算分位数
- 每个时刻仅用该时刻之前的全部历史数据，严格保证 "过去→未来" 的因果性
- NMHC 原始数据缺失率 90.2%（8126 小时连续缺口），采用间隙感知填充：
  - ≤6 小时缺口 → 线性插值后参与 expanding rank
  - >6 小时缺口 → 子指数置 NaN，不参与加权合成
- 子指数 = 分位数 × 100
- **AQI = 各子指数等权平均**（多指标加权合成，NaN 自动排除后重新归一化）
- 等级划分：优(0-20)、良(20-40)、轻度(40-60)、中度(60-80)、重度(80-100)

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
- NMHC 特殊处理：90.2% 缺失率（含 8126 小时连续缺口），经间隙感知填充后保留参与 AQI 计算；过长缺口段子指数稳定，不影响 AQI 主导污染物判定
- 异常值策略：**不采用 3σ 准则**。污染物高浓度值属于真实污染事件而非统计"异常值"，予以保留。后续 AQI 采用非参数 expanding rank 方法，天然对极值鲁棒

## 模型性能

| 模型 | R² | RMSE | MAE |
|------|-----|------|-----|
| 线性回归 | 0.77 | 12.28 | 9.42 |
| XGBoost | **0.78 +/- 0.11** | **11.93 +/- 3.56** | **8.78 +/- 2.61** |

XGBoost 5 折交叉验证详细结果：

| Fold | R² | RMSE | 训练/测试样本 |
|------|-----|------|-------------|
| 1 | 0.725 | 14.82 | 1562/1559 |
| 2 | 0.642 | 14.77 | 3121/1559 |
| 3 | 0.727 | 13.78 | 4680/1559 |
| 4 | 0.872 | 9.13 | 6239/1559 |
| 5 | 0.920 | 7.15 | 7798/1559 |

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
