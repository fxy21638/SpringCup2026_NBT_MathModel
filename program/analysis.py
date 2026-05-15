"""
空气质量综合评价与预测建模 — 完整分析流水线。

问题1: 基于模糊综合评价的空气质量评价（梯形隶属度+熵权法+模糊合成）
问题2: 传感器→AQI 预测模型（XGBoost + TimeSeriesSplit，仅传感器+小时）
问题3: 基于特征重要性的改善建议

输出图表：output/picture/ 目录下 8 张图
"""

import sys
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

# ---- 图1: AQI 分布直方图（按实际 FCE 等级着色，与饼图同源） ----
fig, ax = plt.subplots(figsize=(10, 5))
n, bins, patches = ax.hist(df['AQI'], bins=50, color='steelblue', alpha=0.85, edgecolor='white')

# 每根柱子按该区间内样本的实际 FCE 等级（多数投票）着色，与饼图判定一致
grade_colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59']
grade_to_color = dict(zip(grade_order, grade_colors))
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    mask = (df['AQI'] >= lo) & (df['AQI'] < hi)
    if i == len(bins) - 2:
        mask = (df['AQI'] >= lo) & (df['AQI'] <= hi)
    grades = df.loc[mask, 'AQI_等级']
    if len(grades) > 0:
        patches[i].set_facecolor(grade_to_color[grades.mode().iloc[0]])

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

# ---- 图2.5a: 梯形隶属度函数可视化 ----
sys.path.insert(0, str(Path(__file__).parent))
from compute_aqi import build_membership_breakpoints, trapezoid_membership, POLLUTANTS as FCE_POLLUTANTS, GRADE_NAMES as FCE_GRADES

bp = build_membership_breakpoints(df)

# 2x2 正方形布局：CO, C6H6, NOx, NO2（NMHC 因 90% 缺失率，隶属度信息量低，省略）
plot_pollutants = ['CO', 'C6H6', 'NOx', 'NO2']
fig = plt.figure(figsize=(14, 11))
grade_colors_fce = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59']
grade_labels = FCE_GRADES

for idx, pol in enumerate(plot_pollutants):
    ax = fig.add_subplot(2, 2, idx + 1)
    if pol not in bp:
        continue

    # 实际数据直方图（底层，展示浓度分布）
    real_data = df[pol].dropna().values
    ax2 = ax.twinx()
    ax2.hist(real_data, bins=60, color='#bdc3c7', alpha=0.5, edgecolor='white', linewidth=0.3)
    ax2.set_ylabel('频数', fontsize=8, color='#7f8c8d')
    ax2.tick_params(axis='y', labelsize=7, colors='#7f8c8d')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.3)  # 给文字留空间

    # 隶属度曲线（上层）
    x_vals = np.linspace(df[pol].min(), df[pol].max(), 500)
    for k, gname in enumerate(grade_labels):
        a, b, c, d = bp[pol][k]
        y = trapezoid_membership(x_vals, a, b, c, d)
        ax.plot(x_vals, y, linewidth=2.0, color=grade_colors_fce[k], label=gname, zorder=5)
        ax.fill_between(x_vals, 0, y, color=grade_colors_fce[k], alpha=0.08, zorder=2)

    # 断点竖线（带标签）
    bp_vals = [bp[pol][0, 2], bp[pol][1, 3], bp[pol][2, 3], bp[pol][3, 3]]
    bp_labels_text = ['p20', 'p40', 'p60', 'p80']
    for j, (pct_val, lbl) in enumerate(zip(bp_vals, bp_labels_text)):
        ax.axvline(x=pct_val, color='#636363', linestyle='--', alpha=0.5, linewidth=1.0, zorder=3)
        ax.text(pct_val, 1.03, lbl, fontsize=7, color='#636363', ha='center', va='bottom',
                transform=ax.get_xaxis_transform())

    # 显示关键统计量
    ax.text(0.97, 0.92, f'范围: [{real_data.min():.1f}, {real_data.max():.1f}]\nn={len(real_data)}',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

    ax.set_xlabel(pol, fontsize=11, fontweight='bold')
    ax.set_ylabel('隶属度', fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, loc='upper left', ncol=5, framealpha=0.7)
    ax.grid(alpha=0.12)
    ax.set_xlim(df[pol].min(), df[pol].max())

plt.suptitle('各污染物梯形隶属度函数（灰色直方图为实际浓度分布，虚线为分位数断点）',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'membership_functions.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图2.5a] membership_functions.png')

# ---- 图2.5b: 熵权法权重柱状图 ----
from compute_aqi import compute_entropy_weights, build_membership_matrix

R = build_membership_matrix(df, bp)
weights = compute_entropy_weights(R)

fig, ax = plt.subplots(figsize=(8, 5))
w_colors = []
for pol in FCE_POLLUTANTS:
    w = weights[FCE_POLLUTANTS.index(pol)]
    if w < 0.05:
        w_colors.append('#fc8d59')
    elif w < 0.15:
        w_colors.append('#fee08b')
    else:
        w_colors.append('#1a9850')
bars = ax.barh(FCE_POLLUTANTS, [weights[i] for i in range(len(FCE_POLLUTANTS))], color=w_colors, alpha=0.9, edgecolor='white')
ax.set_xlabel('权重', fontsize=11)
ax.set_title('熵权法客观权重', fontsize=13, fontweight='bold')
ax.grid(alpha=0.2, axis='x')
for bar, w in zip(bars, weights):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{w:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlim(0, max(weights) * 1.2)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'entropy_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图2.5b] entropy_weights.png')

# ---- 图3: AQI 时序图（选取夏季+冬季各一周代表性时段） ----
ts_start_summer = pd.Timestamp('2004-07-20')
ts_end_summer = pd.Timestamp('2004-08-03')
ts_start_winter = pd.Timestamp('2004-12-10')
ts_end_winter = pd.Timestamp('2004-12-24')

mask_summer = (df['Datetime'] >= ts_start_summer) & (df['Datetime'] <= ts_end_summer)
mask_winter = (df['Datetime'] >= ts_start_winter) & (df['Datetime'] <= ts_end_winter)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for ax, mask, title, color in [
    (axes[0], mask_summer, '夏季 (2004-07-20 ~ 08-03)', '#e74c3c'),
    (axes[1], mask_winter, '冬季 (2004-12-10 ~ 12-24)', '#3498db')]:
    seg = df[mask]
    ax.plot(seg['Datetime'], seg['AQI'], color=color, linewidth=1.0, alpha=0.9)
    ax.fill_between(seg['Datetime'], 0, seg['AQI'], color=color, alpha=0.08)
    ax.set_ylabel('AQI', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(alpha=0.2)
    # 标注等级分界线
    for level, yval, ls in [('优', 20, ':'), ('良', 40, ':'), ('轻度', 60, '--'), ('中度', 80, '--')]:
        ax.axhline(y=yval, color='gray', linestyle=ls, alpha=0.35, linewidth=0.7)
    ax.set_ylim(0, 105)

axes[-1].set_xlabel('时间', fontsize=11)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'timeseries_aqi.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图3] timeseries_aqi.png')

# ---- 图4: 污染物时序图（NMHC 有真实值的时段 2004-03-23 ~ 04-03） ----
ts_pol_start = pd.Timestamp('2004-03-23')
ts_pol_end = pd.Timestamp('2004-04-03')
mask_pol = (df['Datetime'] >= ts_pol_start) & (df['Datetime'] <= ts_pol_end)
seg = df[mask_pol]
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
for i, pol in enumerate(pollutants):
    if pol in df.columns:
        axes[i].plot(seg['Datetime'], seg[pol], linewidth=0.8, alpha=0.9, color=f'C{i}')
        axes[i].fill_between(seg['Datetime'], 0, seg[pol], color=f'C{i}', alpha=0.06)
        axes[i].set_ylabel(pol, fontsize=10)
        axes[i].grid(alpha=0.2)
axes[0].set_title(f'参考污染物浓度时间序列 ({ts_pol_start.date()} ~ {ts_pol_end.date()}, NMHC 真实值时段)', fontsize=14, fontweight='bold')
axes[-1].set_xlabel('时间', fontsize=11)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'timeseries_pollutants.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图4] timeseries_pollutants.png')

# ---- 图4.5a: 污染物相关性热图 ----
fig, ax = plt.subplots(figsize=(8, 7))
poll_data = df[pollutants].copy()
corr = poll_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
im = ax.imshow(corr, cmap='RdBu_r', vmin=-0.2, vmax=1.0, aspect='equal')
for i in range(len(pollutants)):
    for j in range(len(pollutants)):
        if i >= j:
            val = corr.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if abs(val) > 0.68 else '#2c3e50')
ax.set_xticks(range(len(pollutants)))
ax.set_yticks(range(len(pollutants)))
ax.set_xticklabels(pollutants, fontsize=11)
ax.set_yticklabels(pollutants, fontsize=11)
ax.set_title('参考污染物 Pearson 相关系数', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.85, label='r')
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'pollutant_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图4.5a] pollutant_correlation.png')

# ---- 图4.5b: 各月 AQI 箱线图 ----
df['month'] = df['Datetime'].dt.month
month_labels = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
fig, ax = plt.subplots(figsize=(12, 5))
bp_data = [df[df['month'] == m]['AQI'].values for m in range(1, 13) if m in df['month'].values]
bp_months = [m for m in range(1, 13) if m in df['month'].values]
bp_colors = ['#a6d96a' if m in [5,6,7,8,9] else '#fdae61' if m in [3,4,10,11] else '#1a9641' for m in bp_months]
boxes = ax.boxplot(bp_data, patch_artist=True, widths=0.6,
                   medianprops=dict(color='#333', linewidth=1.5),
                   flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, c in zip(boxes['boxes'], bp_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
ax.set_xticklabels([month_labels[m-1] for m in bp_months], fontsize=9)
ax.set_ylabel('AQI', fontsize=11)
ax.set_title('AQI 月际分布（春夏：绿 / 过渡季：橙 / 冬季：深绿）', fontsize=13, fontweight='bold')
ax.grid(alpha=0.2, axis='y')
# 均值连线
means = [df[df['month'] == m]['AQI'].mean() for m in bp_months]
ax.plot(range(1, len(means)+1), means, 'o-', color='#c0392b', linewidth=1.8, markersize=5, label='月均值')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'monthly_aqi_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图4.5b] monthly_aqi_boxplot.png')

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

# ---- 图7: OOF 预测 vs 真实值 (夏季+冬季代表性时段) ----
mask_valid_summer = (df_valid['Datetime'] >= ts_start_summer) & (df_valid['Datetime'] <= ts_end_summer)
mask_valid_winter = (df_valid['Datetime'] >= ts_start_winter) & (df_valid['Datetime'] <= ts_end_winter)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for ax, mask, title in [
    (axes[0], mask_valid_summer, f'夏季 ({ts_start_summer.date()} ~ {ts_end_summer.date()})'),
    (axes[1], mask_valid_winter, f'冬季 ({ts_start_winter.date()} ~ {ts_end_winter.date()})')]:
    seg_valid = df_valid[mask]
    seg_y = y.values[mask.values]
    seg_pred = oof_pred.values[mask.values]
    ax.plot(seg_valid['Datetime'], seg_y, color='#e74c3c', linewidth=1.0, alpha=0.85, label='真实 AQI')
    ax.plot(seg_valid['Datetime'], seg_pred, color='#3498db', linewidth=1.0, alpha=0.85, label='OOF 预测 AQI')
    ax.set_ylabel('AQI', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.2)

axes[-1].set_xlabel('时间', fontsize=11)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'timeseries_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图7] timeseries_prediction.png')

# ---- 图7.5a: 预测残差分布 ----
residuals = y.values - oof_pred.values
valid_res = residuals[~np.isnan(residuals)]
print(f'  [图7.5a] OOF覆盖样本={len(valid_res)}/{len(y)} (首{len(y)-len(valid_res)}行为纯训练集)')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(valid_res, bins=60, color='#3498db', alpha=0.8, edgecolor='white')
axes[0].axvline(x=0, color='#e74c3c', linestyle='--', linewidth=1.5)
axes[0].axvline(x=valid_res.mean(), color='#e74c3c', linestyle='-', linewidth=1.2, label=f'均值={valid_res.mean():.2f}')
axes[0].set_xlabel('残差 (真实 - 预测)', fontsize=11)
axes[0].set_ylabel('频数', fontsize=11)
axes[0].set_title('预测残差分布 (OOF)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.2, axis='y')
# Q-Q 图
from scipy import stats
stats.probplot(valid_res, dist='norm', plot=axes[1])
axes[1].get_lines()[0].set_markerfacecolor('#3498db')
axes[1].get_lines()[0].set_markeredgecolor('#3498db')
axes[1].get_lines()[1].set_color('#e74c3c')
axes[1].set_title('Q-Q 图 (正态性检验)', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.2)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'residual_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  残差均值={valid_res.mean():.3f}, 标准差={valid_res.std():.3f}')

# ---- 图7.5b: 传感器与 AQI 相关性矩阵 ----
corr_cols = sensor_cols + ['AQI']
corr_data = df_valid[corr_cols].copy()
sensor_corr = corr_data.corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sensor_corr, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
labels = ['CO_s', 'NMHC_s', 'NOx_s', 'NO2_s', 'O3_s', 'AQI']
for i in range(len(labels)):
    for j in range(len(labels)):
        val = sensor_corr.iloc[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if val > 0.72 else '#2c3e50')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticklabels(labels, fontsize=10)
ax.set_title('传感器读数与 AQI 相关系数', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.85, label='r')
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'sensor_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图7.5b] sensor_correlation.png')

# ---- 图7.5c: AQI 日变化模式 (diurnal) ----
df_valid_copy = df_valid.copy()
df_valid_copy['hour'] = df_valid_copy['Datetime'].dt.hour
hourly = df_valid_copy.groupby('hour')['AQI'].agg(['mean', 'std', 'count']).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(hourly['hour'],
                hourly['mean'] - hourly['std'],
                hourly['mean'] + hourly['std'],
                color='#3498db', alpha=0.15)
ax.plot(hourly['hour'], hourly['mean'], 'o-', color='#3498db', linewidth=2, markersize=6)
ax.set_xlabel('小时', fontsize=11)
ax.set_ylabel('AQI', fontsize=11)
ax.set_title('AQI 日变化模式（均值 ± 标准差）', fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.2)
# 标注早晚高峰
for h, label in [(8, '早高峰'), (18, '晚高峰')]:
    ax.axvline(x=h, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(h + 0.3, ax.get_ylim()[1] * 0.92, label, color='#e74c3c', fontsize=9)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'diurnal_aqi.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图7.5c] diurnal_aqi.png')

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

# ---- 图9: XGBoost 回归树结构示意图（节点含 AQI 预测值，连线标注是/否） ----
import xgboost as xgb
import re
from matplotlib.patches import FancyBboxPatch

scaler_viz = StandardScaler()
X_scaled_viz = scaler_viz.fit_transform(X)
simple_tree = xgb.XGBRegressor(n_estimators=1, max_depth=3, learning_rate=0.3, random_state=42, verbosity=0)
simple_tree.fit(X_scaled_viz, y)

BASE_SCORE = y.mean()
LR_TREE = 0.3
feat_names = ['PT08.S1_CO', 'PT08.S2_NMHC', 'PT08.S3_NOx', 'PT08.S4_NO2', 'PT08.S5_O3', 'hour']

# 解析树结构
dump = simple_tree.get_booster().get_dump(with_stats=True)[0]
nodes = {}
for line in dump.strip().split('\n'):
    depth = line.count('\t')
    line_s = line.strip()
    m = re.match(r'(\d+):leaf=([^,]+),cover=', line_s)
    if m:
        nid = int(m.group(1))
        nodes[nid] = {'depth': depth, 'is_leaf': True, 'leaf_val': float(m.group(2)),
                      'split_feat': None, 'split_val': None,
                      'left': None, 'right': None, 'gain': None}
        continue
    m = re.match(r'(\d+):\[f(\d+)<([^\]]+)\]', line_s)
    if m:
        nid = int(m.group(1))
        rest = line_s[line_s.index(']')+1:]
        yes_m = re.search(r'yes=(\d+)', rest)
        no_m = re.search(r'no=(\d+)', rest)
        gain_m = re.search(r'gain=([\d.]+)', rest)
        nodes[nid] = {'depth': depth, 'is_leaf': False,
                      'split_feat': int(m.group(2)), 'split_val': float(m.group(3)),
                      'left': int(yes_m.group(1)) if yes_m else None,
                      'right': int(no_m.group(1)) if no_m else None,
                      'gain': float(gain_m.group(1)) if gain_m else 0, 'leaf_val': None}

# 叶子从左到右分配 x 坐标，内部节点取子节点均值
leaf_x = 0
for nid in sorted(nodes.keys()):
    if nodes[nid]['is_leaf']:
        nodes[nid]['x'] = leaf_x
        leaf_x += 1

def assign_x(nid):
    nd = nodes[nid]
    if nd['is_leaf']:
        return
    assign_x(nd['left'])
    assign_x(nd['right'])
    nd['x'] = (nodes[nd['left']]['x'] + nodes[nd['right']]['x']) / 2.0
assign_x(0)

max_depth = max(nd['depth'] for nd in nodes.values())
n_leaves = leaf_x

BOX_W, BOX_H, LEAF_H = 2.6, 0.95, 0.80
X_GAP, Y_GAP = 0.6, 2.2
fig_w = max(14, n_leaves * (BOX_W + X_GAP) * 0.7 + 1.5)
fig_h = (max_depth + 1) * Y_GAP + 3.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
span = (n_leaves - 1) * (BOX_W + X_GAP) if n_leaves > 1 else BOX_W
ax.set_xlim(-BOX_W/2 - X_GAP, span + BOX_W/2 + X_GAP)
ax.set_ylim(-1.0, max_depth * Y_GAP + 2.8)
ax.axis('off')

def draw_node(nid, px=None, is_left=None):
    nd = nodes[nid]
    x = nd['x'] * (BOX_W + X_GAP)
    y = (max_depth - nd['depth']) * Y_GAP
    bw, bh = BOX_W, (LEAF_H if nd['is_leaf'] else BOX_H)

    if nd['is_leaf']:
        aqi_c = LR_TREE * nd['leaf_val']
        aqi_p = BASE_SCORE + aqi_c
        sign = '+' if aqi_c >= 0 else ''
        color = '#d63031' if aqi_c > 0 else '#0984e3'
        box = FancyBboxPatch((x - bw/2, y - bh/2), bw, bh,
                             boxstyle='round,pad=0.12', linewidth=2.2,
                             edgecolor=color, facecolor=color, alpha=0.22, zorder=2)
        ax.add_patch(box)
        ax.text(x, y + 0.08, 'AQI = {:.1f}'.format(aqi_p), ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=color, zorder=3)
        ax.text(x, y - 0.27, '(修正 {}{:.1f})'.format(sign, aqi_c), ha='center', va='center',
                fontsize=8.5, color=color, zorder=3)
    else:
        box = FancyBboxPatch((x - bw/2, y - bh/2), bw, bh,
                             boxstyle='round,pad=0.12', linewidth=2.0,
                             edgecolor='#7f8c8d', facecolor='#f5f6fa', zorder=2)
        ax.add_patch(box)
        ax.text(x, y + 0.1, '{} < {:.2f} ?'.format(feat_names[nd['split_feat']], nd['split_val']),
                ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50', zorder=3)
        ax.text(x, y - 0.28, 'Gain={:.0f}'.format(nd['gain']), ha='center', va='center',
                fontsize=7.5, color='#b2bec3', zorder=3)

    if px is not None:
        py_bot = px[1] - BOX_H/2
        ny_top = y + bh/2
        mid_y = (py_bot + ny_top) / 2
        color = '#00b894' if is_left else '#d63031'
        ax.plot([px[0], px[0], x, x], [py_bot, mid_y, mid_y, ny_top],
                color=color, linewidth=1.8, alpha=0.55, zorder=0, solid_capstyle='round')
        label_x = px[0] + (x - px[0]) * 0.45
        ax.text(label_x, mid_y, '是' if is_left else '否', ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=color, linewidth=0.8, alpha=0.9), zorder=4)

    if not nd['is_leaf']:
        draw_node(nd['left'], (x, y), is_left=True)
        draw_node(nd['right'], (x, y), is_left=False)

draw_node(0)
mid_x = (n_leaves - 1) * (BOX_W + X_GAP) / 2
ax.text(mid_x, max_depth * Y_GAP + 2.3,
        'XGBoost 单棵回归树结构 (max_depth=3, 学习率=0.3)',
        ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
ax.text(mid_x, -0.7,
        '预测: AQI = {:.1f} (基础分) + {} x 叶子修正值     |     '
        '红色节点 = 污染加重     |     蓝色节点 = 污染减轻'.format(BASE_SCORE, LR_TREE),
        ha='center', fontsize=10.5, color='#636e72',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dfe6e9'))
plt.tight_layout(pad=2)
plt.savefig(PICTURE_DIR / 'tree_depth3.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('  [图9] tree_depth3.png')

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
  1.  aqi_distribution.png        — AQI 分布直方图
  2.  aqi_grade_pie.png           — 污染等级饼图
  3.  membership_functions.png    — 梯形隶属度函数曲线
  4.  entropy_weights.png         — 熵权法权重柱状图
  5.  timeseries_aqi.png          — AQI 时间序列 (夏/冬)
  6.  timeseries_pollutants.png   — 污染物浓度时间序列
  7.  pollutant_correlation.png   — 污染物相关性热图
  8.  monthly_aqi_boxplot.png     — AQI 月际箱线图
  9.  model_comparison.png        — 模型性能对比
  10. feature_importance.png      — 特征重要性
  11. timeseries_prediction.png   — 预测 vs 真实值 (夏/冬)
  12. residual_distribution.png   — 预测残差分布 + Q-Q 图
  13. sensor_correlation.png      — 传感器与 AQI 相关性
  14. diurnal_aqi.png             — AQI 日变化模式
  15. fold_evaluation.png         — 各折 R^2 对比
  16. tree_depth3.png             — XGBoost 回归树结构示意图

输出文件:
  - output/xgb_model.pkl       (训练好的模型)
  - output/model_evaluation.csv (模型评估结果)
''')
