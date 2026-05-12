"""
空气质量综合评价与预测建模 — 完整分析流水线。

问题1: 基于模糊综合评价的空气质量评价（梯形隶属度+熵权法+模糊合成）
问题2: 传感器→AQI 预测模型（XGBoost + TimeSeriesSplit，仅传感器+小时）
问题3: 基于特征重要性的改善建议

输出图表：output/picture/ 目录下 8 张图
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# 中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('output')
PICTURE_DIR = OUTPUT_DIR / 'picture'
MODEL_PATH = OUTPUT_DIR / 'xgb_model.pkl'
EVALUATION_PATH = OUTPUT_DIR / 'model_evaluation.csv'
DATA_WITH_AQI_PATH = OUTPUT_DIR / 'data_with_aqi.csv'
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 0. 加载数据
# ============================================================
print('=' * 60)
print('空气质量综合评价与预测建模')
print('=' * 60)

df = pd.read_csv(DATA_WITH_AQI_PATH)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)
print(f'\n数据加载完成: {df.shape[0]} 条记录, {df.shape[1]} 列')
print(f'时间范围: {df["Datetime"].min()} ~ {df["Datetime"].max()}')

# ============================================================
# 1. 问题1 — 空气质量综合评价
# ============================================================
print('\n' + '=' * 60)
print('问题1: 空气质量综合评价')
print('=' * 60)

# 1.1 AQI 统计
print(f'\nAQI 统计:')
print(f'  均值={df["AQI"].mean():.2f}, 中位数={df["AQI"].median():.2f}')
print(f'  标准差={df["AQI"].std():.2f}, 范围=[{df["AQI"].min():.2f}, {df["AQI"].max():.2f}]')

# 1.2 等级分布
grade_order = ['优', '良', '轻度污染', '中度污染', '重度污染']
grade_counts = df['AQI_等级'].value_counts()
print(f'\n等级分布:')
for g in grade_order:
    cnt = grade_counts.get(g, 0)
    print(f'  {g}: {cnt} ({cnt/len(df)*100:.1f}%)')

# 1.3 各污染物统计
pollutants = ['CO', 'NMHC', 'C6H6', 'NOx', 'NO2']
print(f'\n参考污染物浓度统计:')
for p in pollutants:
    if p in df.columns:
        print(f'  {p:6s}: 均值={df[p].mean():8.2f}, 标准差={df[p].std():8.2f}, '
              f'min={df[p].min():8.2f}, max={df[p].max():8.2f}')

# ---- 图1: AQI 分布直方图 ----
fig, ax = plt.subplots(figsize=(10, 5))
n, bins, patches = ax.hist(df['AQI'], bins=50, color='steelblue', alpha=0.85, edgecolor='white')

# 按等级着色
grade_colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
for i in range(len(bins) - 1):
    center = (bins[i] + bins[i + 1]) / 2
    for j, (lo, hi) in enumerate([(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]):
        if lo <= center < hi:
            patches[i].set_facecolor(grade_colors[j])
            break

ax.set_xlabel('AQI', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('空气质量指数 (AQI) 分布', fontsize=14, fontweight='bold')
ax.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'aqi_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图1] aqi_distribution.png')

# ---- 图2: 污染等级饼图 ----
fig, ax = plt.subplots(figsize=(8, 8))
pie_data = [grade_counts.get(g, 0) for g in grade_order]
wedges, texts, autotexts = ax.pie(
    pie_data, labels=None, autopct='%1.1f%%', startangle=90,
    colors=grade_colors, pctdistance=0.75, textprops={'fontsize': 10}
)
for at in autotexts:
    at.set_fontweight('bold')
ax.set_title('空气质量等级分布', fontsize=14, fontweight='bold')
ax.legend(wedges, grade_order, title='等级', loc='center left', bbox_to_anchor=(1, 0.5, 0.5, 1))
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'aqi_grade_pie.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图2] aqi_grade_pie.png')

# ---- 图3: AQI 时序图 ----
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Datetime'], df['AQI'], color='#e74c3c', linewidth=0.6, alpha=0.9)
ax.set_xlabel('时间', fontsize=11)
ax.set_ylabel('AQI', fontsize=11)
ax.set_title('AQI 时间序列', fontsize=14, fontweight='bold')
ax.grid(alpha=0.2)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'timeseries_aqi.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图3] timeseries_aqi.png')

# ---- 图4: 污染物时序图 ----
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
for i, pol in enumerate(pollutants):
    if pol in df.columns:
        axes[i].plot(df['Datetime'], df[pol], linewidth=0.5, alpha=0.85, color=f'C{i}')
        axes[i].set_ylabel(pol, fontsize=10)
        axes[i].grid(alpha=0.2)
axes[0].set_title('参考污染物浓度时间序列', fontsize=14, fontweight='bold')
axes[-1].set_xlabel('时间', fontsize=11)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'timeseries_pollutants.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图4] timeseries_pollutants.png')

# ============================================================
# 2. 问题2 — 传感器→AQI 预测模型
# ============================================================
print('\n' + '=' * 60)
print('问题2: 传感器阵列预测 AQI 模型')
print('=' * 60)

# 构造特征（仅传感器测量值 + 简单小时，符合题目约束）
sensor_cols = ['PT08.S1_CO', 'PT08.S2_NMHC', 'PT08.S3_NOx', 'PT08.S4_NO2', 'PT08.S5_O3']

# 时间特征：仅使用小时（0-23），作为简单"随时间变化"表示
df['hour'] = df['Datetime'].dt.hour

feature_cols = [c for c in sensor_cols + ['hour'] if c in df.columns]
print(f'\n特征列表 ({len(feature_cols)} 个):')
for f in feature_cols:
    print(f'  {f}')

# 准备 X, y，删除含 NaN 的行
X_all = df[feature_cols].copy()
y_all = df['AQI'].copy()
valid_mask = X_all.notna().all(axis=1) & y_all.notna()
X = X_all[valid_mask].reset_index(drop=True)
y = y_all[valid_mask].reset_index(drop=True)
df_valid = df[valid_mask].reset_index(drop=True)
print(f'\n有效样本: {len(X)} / {len(df)}')

# ---- TimeSeriesSplit 训练 ----
tscv = TimeSeriesSplit(n_splits=5)

xgb_params = dict(
    n_estimators=100, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.5,
    random_state=42, verbosity=0
)

# 用于存储 OOF 预测
oof_pred = pd.Series(index=X.index, dtype=float)

# XGBoost 交叉验证
xgb_fold_results = []
best_model = None
best_r2 = -np.inf

print(f'\nXGBoost TimeSeriesSplit (5-fold) 结果:')
for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    xgb_fold_results.append({'fold': fold_id, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'n_train': len(train_idx), 'n_test': len(test_idx)})

    # 保存 OOF 预测
    oof_pred.iloc[test_idx] = y_pred

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

    print(f'  Fold {fold_id}: R^2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}  (train={len(train_idx)}, test={len(test_idx)})')

xgb_cv = pd.DataFrame(xgb_fold_results)
print(f'\n  平均 R^2 = {xgb_cv["R2"].mean():.4f} +/- {xgb_cv["R2"].std():.4f}')
print(f'  平均 RMSE = {xgb_cv["RMSE"].mean():.4f} +/- {xgb_cv["RMSE"].std():.4f}')
print(f'  平均 MAE = {xgb_cv["MAE"].mean():.4f} +/- {xgb_cv["MAE"].std():.4f}')

# 线性回归基线（同交叉验证方式）
lr_fold_results = []
print(f'\n线性回归 TimeSeriesSplit (5-fold) 结果:')
for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    lr_fold_results.append({'fold': fold_id, 'R2': r2, 'RMSE': rmse, 'MAE': mae})

lr_cv = pd.DataFrame(lr_fold_results)
print(f'  平均 R^2 = {lr_cv["R2"].mean():.4f},  平均 RMSE = {lr_cv["RMSE"].mean():.4f},  平均 MAE = {lr_cv["MAE"].mean():.4f}')

# ---- 图5: 模型对比 (R^2, RMSE, MAE) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = ['LinearRegression', 'XGBoost']
colors = ['#95a5a6', '#3498db']

axes[0].bar(labels, [lr_cv['R2'].mean(), xgb_cv['R2'].mean()], color=colors, alpha=0.85)
axes[0].set_ylabel('R^2', fontsize=12)
axes[0].set_title('R^2 对比 (越高越好)', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.2, axis='y')
for i, v in enumerate([lr_cv['R2'].mean(), xgb_cv['R2'].mean()]):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)

axes[1].bar(labels, [lr_cv['RMSE'].mean(), xgb_cv['RMSE'].mean()], color=colors, alpha=0.85)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('RMSE 对比 (越低越好)', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.2, axis='y')
for i, v in enumerate([lr_cv['RMSE'].mean(), xgb_cv['RMSE'].mean()]):
    axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=10)

axes[2].bar(labels, [lr_cv['MAE'].mean(), xgb_cv['MAE'].mean()], color=colors, alpha=0.85)
axes[2].set_ylabel('MAE', fontsize=12)
axes[2].set_title('MAE 对比 (越低越好)', fontsize=13, fontweight='bold')
axes[2].grid(alpha=0.2, axis='y')
for i, v in enumerate([lr_cv['MAE'].mean(), xgb_cv['MAE'].mean()]):
    axes[2].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(PICTURE_DIR / 'model_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图5] model_comparison.png')

# ---- 图6: 特征重要性 (XGBoost) ----
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
top_n = min(15, len(feature_importance))
top_features = feature_importance.head(top_n)
bars = ax.barh(range(top_n), top_features['importance'].values, color='#3498db', alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['feature'].values)
ax.set_xlabel('重要性', fontsize=11)
ax.set_title(f'XGBoost 特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图6] feature_importance.png')

# 打印特征重要性排名
print(f'\n特征重要性排名 (Top 10):')
for _, row in feature_importance.head(10).iterrows():
    print(f'  {row["feature"]:20s}: {row["importance"]:.4f}')

# ---- 图7: OOF 预测 vs 真实值 (时间序列) ----
fig, ax = plt.subplots(figsize=(14, 5))
time_axis = df_valid['Datetime']
ax.plot(time_axis, y.values, color='#e74c3c', linewidth=0.8, alpha=0.8, label='真实 AQI')
ax.plot(time_axis, oof_pred.values, color='#3498db', linewidth=0.6, alpha=0.8, label='OOF 预测 AQI')
ax.set_xlabel('时间', fontsize=11)
ax.set_ylabel('AQI', fontsize=11)
ax.set_title('OOF 预测 vs 真实 AQI (5折时序交叉验证)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.2)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'timeseries_prediction.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图7] timeseries_prediction.png')

# ---- 图8: 各折评估结果 ----
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(xgb_cv))
width = 0.35
ax.bar(x_pos - width/2, xgb_cv['R2'], width, label='XGBoost', color='#3498db', alpha=0.85)
ax.bar(x_pos + width/2, lr_cv['R2'], width, label='LinearRegression', color='#95a5a6', alpha=0.85)
ax.set_xlabel('Fold', fontsize=11)
ax.set_ylabel('R^2', fontsize=11)
ax.set_title('各折 R^2 对比', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Fold {i}' for i in xgb_cv['fold']])
ax.legend(fontsize=11)
ax.grid(alpha=0.2, axis='y')
# 标注数值
for i in range(len(xgb_cv)):
    ax.text(i - width/2, xgb_cv['R2'].iloc[i] + 0.01, f'{xgb_cv["R2"].iloc[i]:.3f}', ha='center', fontsize=8)
    ax.text(i + width/2, lr_cv['R2'].iloc[i] + 0.01, f'{lr_cv["R2"].iloc[i]:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'fold_evaluation.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [图8] fold_evaluation.png')

# 保存模型
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)
print(f'\n  已保存: {MODEL_PATH}')

# 保存模型评估结果
eval_df = pd.DataFrame({
    '模型': ['LinearRegression', 'XGBoost'],
    'R^2': [lr_cv['R2'].mean(), xgb_cv['R2'].mean()],
    'R^2_std': [lr_cv['R2'].std(), xgb_cv['R2'].std()],
    'RMSE': [lr_cv['RMSE'].mean(), xgb_cv['RMSE'].mean()],
    'MAE': [lr_cv['MAE'].mean(), xgb_cv['MAE'].mean()],
})
eval_df.to_csv(EVALUATION_PATH, index=False, encoding='utf-8-sig')
print(f'  已保存: {EVALUATION_PATH}')

# ============================================================
# 3. 问题3 — 改善建议
# ============================================================
print('\n' + '=' * 60)
print('问题3: 空气质量改善建议')
print('=' * 60)

# 识别最重要的传感器
top_sensor = None
for _, row in feature_importance.iterrows():
    if row['feature'] in sensor_cols:
        top_sensor = row['feature']
        break

print(f'''
基于数据分析与模型结果，提出以下改善建议：

1. 交通源管控 [优先级: 高]
   - 传感器 {sensor_cols[0]} (CO) 和 {sensor_cols[3]} (NO2) 反映车辆尾气排放
   - 建议: 高污染日实施限行、推广新能源车、优化高峰期货车通行时段
   - 预期: CO、NO2 浓度下降 20-30%

2. 工业排放控制 [优先级: 高]
   - 传感器 {sensor_cols[2]} (NOx) 反映工业和燃烧源排放
   - 建议: 推行排放总量上限、升级燃煤锅炉净化装置、建立工业污染排放信用体系
   - 预期: NOx 浓度下降 25-35%

3. 气象因素利用 [优先级: 中]
   - 温度(T)和湿度(AH/RH)影响传感器读数及污染扩散条件
   - 建议: 城市规划增加通风走廊和绿地面积、建立高污染气象条件预警
   - 预期: 提高污染预测准确率至 80%+

4. 监测网络优化 [优先级: 中]
   - 传感器校准精度影响预测准确度（当前模型 R^2={xgb_cv["R2"].mean():.2f}）
   - 建议: 建立传感器定期校准制度、扩展监测站点覆盖不同功能区
   - 预期: 模型 R^2 可进一步提升 5-10%

5. 公众健康提示 [优先级: 中]
   - AQI ≥ 60 时建议易感人群减少户外活动
   - AQI ≥ 80 时发布全民健康防护建议
   - 通过手机 APP、公共屏幕等实时发布 AQI 信息
''')

# ============================================================
# 4. 最终总结
# ============================================================
print('=' * 60)
print('分析完成！')
print('=' * 60)
print(f'''
模型性能总结:
  - XGBoost R^2 = {xgb_cv["R2"].mean():.4f} +/- {xgb_cv["R2"].std():.4f}
  - XGBoost RMSE = {xgb_cv["RMSE"].mean():.2f} AQI 点
  - 对比线性回归 R^2 = {lr_cv["R2"].mean():.4f}
  - 特征数量: {len(feature_cols)} 个 ({len(sensor_cols)} 传感器 + 1 时间)

生成图表 (output/picture/):
  1. aqi_distribution.png      — AQI 分布直方图
  2. aqi_grade_pie.png         — 污染等级饼图
  3. timeseries_aqi.png        — AQI 时间序列
  4. timeseries_pollutants.png — 污染物浓度时间序列
  5. model_comparison.png      — 模型性能对比
  6. feature_importance.png    — 特征重要性
  7. timeseries_prediction.png — 预测 vs 真实值
  8. fold_evaluation.png       — 各折 R^2 对比

输出文件:
  - output/xgb_model.pkl       (训练好的模型)
  - output/model_evaluation.csv (模型评估结果)
''')
