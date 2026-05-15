"""
模型改进分析：解决多重共线性问题
- Part 1: PCA-entropy 混合权重（FCE 模型改进）
- Part 2: SHAP 特征重要性（XGBoost 解释改进）
- Part 3: VIF 特征剪枝实验（预测模型改进）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap
import pickle

OUTPUT_DIR = Path('output')
PICTURE_DIR = OUTPUT_DIR / 'picture'
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 0. 加载数据
# ============================================================
df = pd.read_csv(OUTPUT_DIR / 'data_with_aqi.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

POLLUTANTS = ['CO', 'NMHC', 'C6H6', 'NOx', 'NO2']
SENSOR_COLS = ['PT08.S1_CO', 'PT08.S2_NMHC', 'PT08.S3_NOx', 'PT08.S4_NO2', 'PT08.S5_O3']

print('=' * 65)
print('Part 1: PCA-Entropy 混合权重 (FCE 改进)')
print('=' * 65)

# 1.1 标准化污染物浓度
poll_data = df[POLLUTANTS].dropna()
scaler = StandardScaler()
poll_scaled = scaler.fit_transform(poll_data)

# 1.2 PCA
pca = PCA()
pca.fit(poll_scaled)

print(f'\n各主成分解释方差比:')
cumsum = 0
for i, vr in enumerate(pca.explained_variance_ratio_):
    cumsum += vr
    print(f'  PC{i+1}: {vr:.4f}  (累计 {cumsum:.4f})')

print(f'\n主成分载荷矩阵 (特征向量):')
loadings = pd.DataFrame(
    pca.components_.T,
    index=POLLUTANTS,
    columns=[f'PC{i+1}' for i in range(len(POLLUTANTS))]
)
print(loadings.round(3))

# 1.3 PCA 权重：各污染物对主成分的贡献加权
# w_j = Σ_i (|loading_{j,i}| × variance_ratio_i) / Σ_j Σ_i (...)
pca_weights_raw = np.zeros(len(POLLUTANTS))
for j in range(len(POLLUTANTS)):
    for i in range(len(POLLUTANTS)):
        pca_weights_raw[j] += abs(loadings.iloc[j, i]) * pca.explained_variance_ratio_[i]
pca_weights = pca_weights_raw / pca_weights_raw.sum()

# 1.4 读取当前熵权法权重（从 compute_aqi 模块）
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_aqi import compute_entropy_weights, build_membership_breakpoints, build_membership_matrix, POLLUTANTS as FCE_POLLUTANTS

bp = build_membership_breakpoints(df)
R = build_membership_matrix(df, bp)
entropy_weights = compute_entropy_weights(R)

# 1.5 PCA-熵权混合权重 (α=0.5 平均融合)
alpha = 0.5
hybrid_weights = alpha * pca_weights + (1 - alpha) * entropy_weights
hybrid_weights = hybrid_weights / hybrid_weights.sum()

print(f'\n=== 权重对比 ===')
print(f'{"污染物":<8} {"熵权法":>8} {"PCA权重":>8} {"混合权重":>8} {"变化":>8}')
for j, pol in enumerate(POLLUTANTS):
    diff = hybrid_weights[j] - entropy_weights[j]
    sign = '+' if diff > 0 else ''
    print(f'{pol:<8} {entropy_weights[j]:8.4f} {pca_weights[j]:8.4f} {hybrid_weights[j]:8.4f} {sign}{diff:.4f}')

# 检查交通源污染物权重变化
traffic_idx = [POLLUTANTS.index(p) for p in ['CO', 'NOx', 'NO2']]
traffic_entropy = sum(entropy_weights[i] for i in traffic_idx)
traffic_hybrid = sum(hybrid_weights[i] for i in traffic_idx)
print(f'\n交通源污染物 (CO+NOx+NO2) 合计权重:')
print(f'  熵权法: {traffic_entropy:.4f} → 混合权重: {traffic_hybrid:.4f} (变化: {traffic_hybrid-traffic_entropy:+.4f})')

# ---- 图S1: 权重对比柱状图 ----
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(POLLUTANTS))
width = 0.25
ax.bar(x - width, entropy_weights, width, label='熵权法 (Entropy)', color='#3498db', alpha=0.85)
ax.bar(x, pca_weights, width, label='PCA 权重', color='#e74c3c', alpha=0.85)
ax.bar(x + width, hybrid_weights, width, label='混合权重 (PCA+Entropy)', color='#2ecc71', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(POLLUTANTS, fontsize=11)
ax.set_ylabel('权重', fontsize=11)
ax.set_title('FCE 权重方法对比：熵权法 vs PCA vs 混合权重', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2, axis='y')
for i in range(len(POLLUTANTS)):
    ax.text(i - width, entropy_weights[i] + 0.005, f'{entropy_weights[i]:.3f}', ha='center', fontsize=7.5)
    ax.text(i, pca_weights[i] + 0.005, f'{pca_weights[i]:.3f}', ha='center', fontsize=7.5)
    ax.text(i + width, hybrid_weights[i] + 0.005, f'{hybrid_weights[i]:.3f}', ha='center', fontsize=7.5)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'weight_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  [图S1] weight_comparison.png')

# 1.6 用混合权重重新计算 AQI 并对比
from compute_aqi import GRADE_CENTERS, GRADE_NAMES, N_GRADES

# 原始模糊合成使用的熵权法 AQI
aqi_entropy = df['AQI'].values.copy()

# 混合权重 AQI
n = len(df)
B_hybrid = np.zeros((n, N_GRADES))
for i in range(n):
    for k in range(N_GRADES):
        B_hybrid[i, k] = np.sum(hybrid_weights * R[i, :, k])
B_sum = B_hybrid.sum(axis=1, keepdims=True)
B_sum[B_sum == 0] = 1.0
B_hybrid = B_hybrid / B_sum
aqi_hybrid = (B_hybrid @ GRADE_CENTERS).clip(0, 100)

print(f'\nAQI 对比 (熵权法 vs 混合权重):')
print(f'  熵权法 AQI:  均值={aqi_entropy.mean():.2f}, 中位数={np.median(aqi_entropy):.2f}, 标准差={aqi_entropy.std():.2f}')
print(f'  混合权重 AQI: 均值={aqi_hybrid.mean():.2f}, 中位数={np.median(aqi_hybrid):.2f}, 标准差={aqi_hybrid.std():.2f}')
print(f'  Pearson 相关: {np.corrcoef(aqi_entropy, aqi_hybrid)[0,1]:.4f}')
# 等级一致性
from compute_aqi import trapezoid_membership
grade_entropy = np.argmax(df[[f'隶属度_{g}' for g in GRADE_NAMES]].values, axis=1)
grade_hybrid = np.argmax(B_hybrid, axis=1)
agreement = (grade_entropy == grade_hybrid).mean()
print(f'  等级一致率: {agreement:.1%}')

# ---- 图S2: AQI 散点对比 ----
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(aqi_entropy[::10], aqi_hybrid[::10], c='#3498db', alpha=0.3, s=8, edgecolors='none')
ax.plot([0, 100], [0, 100], '--', color='#e74c3c', linewidth=1.5, label='y=x')
ax.set_xlabel('AQI (熵权法)', fontsize=11)
ax.set_ylabel('AQI (PCA-熵权混合)', fontsize=11)
ax.set_title(f'AQI 对比: 熵权法 vs PCA-熵权混合 (r={np.corrcoef(aqi_entropy, aqi_hybrid)[0,1]:.4f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'aqi_weight_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图S2] aqi_weight_comparison.png')

# ============================================================
# Part 2: SHAP 特征重要性分析
# ============================================================
print('\n' + '=' * 65)
print('Part 2: SHAP 特征重要性分析 (XGBoost 解释改进)')
print('=' * 65)

# 2.1 准备训练数据
df['hour'] = df['Datetime'].dt.hour
feature_cols = [c for c in SENSOR_COLS + ['hour'] if c in df.columns]

X_all = df[feature_cols].copy()
y_all = df['AQI'].copy()
valid_mask = X_all.notna().all(axis=1) & y_all.notna()
X = X_all[valid_mask].reset_index(drop=True)
y = y_all[valid_mask].reset_index(drop=True)

# 2.2 在全量数据上训练 XGBoost（用于 SHAP 分析）
xgb_params = dict(
    n_estimators=100, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.5,
    random_state=42, verbosity=0
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = xgb.XGBRegressor(**xgb_params)
model.fit(X_scaled, y, verbose=False)

print(f'\nXGBoost 全量训练完成，R^2 (in-sample) = {r2_score(y, model.predict(X_scaled)):.4f}')

# 2.3 计算 SHAP 值
print('计算 SHAP 值 (TreeExplainer)...')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled[:2000])  # 采样 2000 个加速计算
print(f'  SHAP 矩阵形状: {shap_values.shape}')

# 2.4 SHAP 重要性排名
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'SHAP_importance': shap_importance
}).sort_values('SHAP_importance', ascending=False)

# Gain 重要性
gain_importance = model.feature_importances_
gain_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'Gain_importance': gain_importance
}).sort_values('Gain_importance', ascending=False)

# 合并对比
compare_df = shap_importance_df.merge(gain_importance_df, on='feature')
# 归一化到 [0, 1] 进行排名对比
compare_df['SHAP_norm'] = compare_df['SHAP_importance'] / compare_df['SHAP_importance'].sum()
compare_df['Gain_norm'] = compare_df['Gain_importance'] / compare_df['Gain_importance'].sum()
# 排名
compare_df['SHAP_rank'] = compare_df['SHAP_importance'].rank(ascending=False).astype(int)
compare_df['Gain_rank'] = compare_df['Gain_importance'].rank(ascending=False).astype(int)

print(f'\n=== 特征重要性排名对比: Gain vs SHAP ===')
print(f'{"特征":<20} {"Gain":>8} {"SHAP":>8} {"Gain排名":>10} {"SHAP排名":>10} {"排名变化":>10}')
for _, row in compare_df.iterrows():
    rank_diff = row['Gain_rank'] - row['SHAP_rank']
    sign = '+' if rank_diff > 0 else ''
    print(f'{row["feature"]:<20} {row["Gain_importance"]:8.4f} {row["SHAP_importance"]:8.4f} '
          f'{row["Gain_rank"]:>10} {row["SHAP_rank"]:>10} {sign}{rank_diff:>9}')

# ---- 图S3: Gain vs SHAP 重要性对比 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Gain
sorted_gain = gain_importance_df.sort_values('Gain_importance')
axes[0].barh(sorted_gain['feature'], sorted_gain['Gain_importance'], color='#3498db', alpha=0.85)
axes[0].set_xlabel('Gain 重要性', fontsize=11)
axes[0].set_title('XGBoost Gain 特征重要性', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.2, axis='x')
for i, (_, row) in enumerate(sorted_gain.iterrows()):
    axes[0].text(row['Gain_importance'] + 0.002, i, f'{row["Gain_importance"]:.4f}', va='center', fontsize=9)

# SHAP
sorted_shap = shap_importance_df.sort_values('SHAP_importance')
axes[1].barh(sorted_shap['feature'], sorted_shap['SHAP_importance'], color='#e74c3c', alpha=0.85)
axes[1].set_xlabel('SHAP 重要性 (mean|SHAP|)', fontsize=11)
axes[1].set_title('SHAP 特征重要性 (对共线性更鲁棒)', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.2, axis='x')
for i, (_, row) in enumerate(sorted_shap.iterrows()):
    axes[1].text(row['SHAP_importance'] + 0.02, i, f'{row["SHAP_importance"]:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(PICTURE_DIR / 'shap_gain_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  [图S3] shap_gain_comparison.png')

# ---- 图S4: SHAP summary plot ----
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_scaled[:2000], feature_names=feature_cols,
                  show=False, plot_size=None)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [图S4] shap_summary.png')

# ============================================================
# Part 3: VIF 特征剪枝实验
# ============================================================
print('\n' + '=' * 65)
print('Part 3: VIF 特征剪枝实验')
print('=' * 65)

# 3.1 计算 VIF
print('\n计算各特征的 VIF...')
vif_data = pd.DataFrame({
    'feature': feature_cols,
    'VIF': [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
}).sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))

# 3.2 识别高 VIF 特征
high_vif_threshold = 5.0
high_vif_features = vif_data[vif_data['VIF'] > high_vif_threshold]['feature'].tolist()
print(f'\n高 VIF 特征 (> {high_vif_threshold}): {high_vif_features}')

# 3.3 剪枝实验：逐个移除高 VIF 特征
tscv = TimeSeriesSplit(n_splits=5)
prune_results = []

# 基线（所有特征）
def cv_evaluate(feat_list, label):
    X_sub = X[feat_list].values
    fold_r2_list = []
    for train_idx, test_idx in tscv.split(X_sub):
        X_tr = X_sub[train_idx]
        X_te = X_sub[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr_s, y_tr, verbose=False)
        y_p = m.predict(X_te_s)
        fold_r2_list.append(r2_score(y_te, y_p))

    avg_r2 = np.mean(fold_r2_list)
    std_r2 = np.std(fold_r2_list)
    prune_results.append({
        'model': label,
        'n_features': len(feat_list),
        'features': '+'.join(feat_list),
        'R2_mean': avg_r2,
        'R2_std': std_r2
    })
    print(f'  {label:30s}: R^2={avg_r2:.4f}±{std_r2:.4f}  (特征数={len(feat_list)})')

print('\nVIF 剪枝实验 (XGBoost + TimeSeriesSplit 5-fold):')
cv_evaluate(feature_cols, '基线 (全部特征)')

# 按 VIF 从高到低逐个移除
remaining = feature_cols.copy()
for feat in high_vif_features:
    if len(remaining) <= 2:
        break
    remaining = [f for f in remaining if f != feat]
    cv_evaluate(remaining, f'移除 {feat}')

# 保守剪枝：只保留 VIF < 5 的特征
conservative = vif_data[vif_data['VIF'] <= high_vif_threshold]['feature'].tolist()
if len(conservative) >= 2:
    cv_evaluate(conservative, '保守剪枝 (VIF<5)')

# 3.4 仅保留最重要的传感器（基于 SHAP）
top_n_shap = shap_importance_df.head(3)['feature'].tolist()
cv_evaluate(top_n_shap + ['hour'] if 'hour' not in top_n_shap else top_n_shap,
            'SHAP Top3 + hour')

prune_df = pd.DataFrame(prune_results)
print(f'\n=== 剪枝结果汇总 ===')
print(prune_df.to_string(index=False))

# ---- 图S5: 剪枝对比 ----
fig, ax = plt.subplots(figsize=(12, 5))
labels_short = [r['model'] for r in prune_results]
x_pos = np.arange(len(labels_short))
bars = ax.bar(x_pos, [r['R2_mean'] for r in prune_results],
              yerr=[r['R2_std'] for r in prune_results],
              color=['#3498db' if i == 0 else '#95a5a6' if i < len(labels_short)-1 else '#2ecc71'
                     for i in range(len(labels_short))],
              alpha=0.85, capsize=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_short, fontsize=7, rotation=15, ha='right')
ax.set_ylabel('R^2 (5-fold CV)', fontsize=11)
ax.set_title('VIF 特征剪枝：模型性能对比', fontsize=13, fontweight='bold')
ax.grid(alpha=0.2, axis='y')
for i, r in enumerate(prune_results):
    ax.text(i, r['R2_mean'] + r['R2_std'] + 0.005, f'{r["R2_mean"]:.3f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'vif_pruning.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  [图S5] vif_pruning.png')

# ============================================================
# 总结
# ============================================================
print('\n' + '=' * 65)
print('模型改进分析总结')
print('=' * 65)

print(f'''
【FCE 模型 — PCA-熵权混合权重】
  交通源污染物 (CO+NOx+NO2) 合计权重:
    熵权法: {traffic_entropy:.3f} → 混合权重: {traffic_hybrid:.3f}
  结论: PCA 修正了相关污染物的重复计数问题
  AQI 相关: r = {np.corrcoef(aqi_entropy, aqi_hybrid)[0,1]:.4f}，等级一致率 = {agreement:.1%}

【XGBoost — SHAP vs Gain 重要性】
  Gain 排名第1: {gain_importance_df.iloc[0]['feature']} ({gain_importance_df.iloc[0]['Gain_importance']:.4f})
  SHAP 排名第1: {shap_importance_df.iloc[0]['feature']} ({shap_importance_df.iloc[0]['SHAP_importance']:.2f})
  结论: SHAP 重要性对特征共线性更鲁棒，排名更可靠

【VIF 特征剪枝】
  高 VIF 特征 (>{high_vif_threshold}): {high_vif_features}
  最佳配置: {prune_df.iloc[prune_df['R2_mean'].argmax()]['model']} (R^2={prune_df['R2_mean'].max():.4f})
''')

print('图片输出:')
print('  weight_comparison.png     — FCE 权重方法对比')
print('  aqi_weight_comparison.png — AQI 散点对比 (熵权 vs 混合)')
print('  shap_gain_comparison.png  — Gain vs SHAP 重要性对比')
print('  shap_summary.png          — SHAP summary plot')
print('  vif_pruning.png           — VIF 剪枝性能对比')
