"""
第三题辅助论证图表生成。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
import sys, warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('output')
PICTURE_DIR = OUTPUT_DIR / 'picture'
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 0. 加载数据
# ============================================================
df = pd.read_csv(OUTPUT_DIR / 'data_with_aqi.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour
df['month'] = df['Datetime'].dt.month

# FCE 数据
sys.path.insert(0, str(Path(__file__).parent))
from compute_aqi import (compute_entropy_weights, build_membership_breakpoints,
                         build_membership_matrix, POLLUTANTS as FCE_POLLUTANTS)

bp = build_membership_breakpoints(df)
R = build_membership_matrix(df, bp)
entropy_weights = compute_entropy_weights(R)

# PCA 数据 (5 项)
ALL_POLLUTANTS = ['CO', 'NMHC', 'C6H6', 'NOx', 'NO2']
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
poll_data = df[ALL_POLLUTANTS].dropna()
poll_scaled = StandardScaler().fit_transform(poll_data)
pca = PCA().fit(poll_scaled)
loadings = pd.DataFrame(pca.components_.T, index=ALL_POLLUTANTS,
                        columns=[f'PC{i+1}' for i in range(5)])
pca_weights_raw = np.zeros(5)
for j in range(5):
    for i in range(5):
        pca_weights_raw[j] += abs(loadings.iloc[j, i]) * pca.explained_variance_ratio_[i]
pca_weights_all = pca_weights_raw / pca_weights_raw.sum()

# SHAP & VIF 数据 (从 model_improvement.py 输出提取)
shap_data = {'PT08.S2_NMHC': 6.65, 'PT08.S5_O3': 5.41, 'PT08.S4_NO2': 3.65,
             'PT08.S1_CO': 3.01, 'PT08.S3_NOx': 2.34, 'hour': 2.04}
vif_data  = {'PT08.S2_NMHC': 10.22, 'PT08.S5_O3': 8.06, 'PT08.S1_CO': 7.41,
             'PT08.S3_NOx': 3.16, 'PT08.S4_NO2': 2.92, 'hour': 1.29}
prune_r2  = {'全特征(6)': 0.746, '移除S2_NMHC(5)': 0.700, '移除S5_O3(4)': 0.619,
             '移除S1_CO(3)': 0.571, 'SHAP Top3+hour(4)': 0.762}
sensor_labels = {'PT08.S1_CO': 'S1(CO)', 'PT08.S2_NMHC': 'S2(NMHC)',
                 'PT08.S3_NOx': 'S3(NOx)', 'PT08.S4_NO2': 'S4(NO₂)',
                 'PT08.S5_O3': 'S5(O₃)', 'hour': '小时'}

# ============================================================
# 图 Q3-1: 污染时间窗口热力图 (hour × month)
# ============================================================
pivot = df.pivot_table(values='AQI', index='hour', columns='month', aggfunc='mean')

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r',
               vmin=20, vmax=65, origin='upper')

# 标注数值
for i in range(24):
    for j in range(12):
        val = pivot.iloc[i, j]
        color = 'white' if val > 50 else 'black'
        ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6.5, color=color)

# 高污染窗口矩形框
# 晚高峰 17-21h, 月份 10-2
from matplotlib.patches import Rectangle
# 月份索引: 10月=9, 11月=10, 12月=11, 1月=0, 2月=1
# 画两个矩形: 10-12月, 1-2月
for (x_start, x_width) in [(9, 3), (0, 2)]:  # 10-12月, 1-2月
    rect = Rectangle((x_start - 0.5, 16.5), x_width, 5,  # 17-21h = rows 17-21
                     linewidth=2, edgecolor='#1a1a2e', facecolor='none',
                     linestyle='--', zorder=5)
    ax.add_patch(rect)

# 早高峰框 8-10h
for (x_start, x_width) in [(9, 3), (0, 2)]:
    rect = Rectangle((x_start - 0.5, 7.5), x_width, 3,  # 8-10h = rows 8-10
                     linewidth=2, edgecolor='#16213e', facecolor='none',
                     linestyle=':', zorder=5)
    ax.add_patch(rect)

ax.set_xticks(range(12))
ax.set_xticklabels([f'{m}月' for m in range(1, 13)], fontsize=10)
ax.set_yticks(range(24))
ax.set_yticklabels([f'{h}:00' for h in range(24)], fontsize=8)
ax.set_xlabel('月份', fontsize=12)
ax.set_ylabel('小时', fontsize=12)
ax.set_title('AQI 时空分布热力图（虚线框=高污染窗口）', fontsize=14, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('AQI 均值', fontsize=10)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'q3_pollution_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [Q3-1] q3_pollution_heatmap.png')

# ============================================================
# 图 Q3-2: NMHC 多证据面板 (3 联图)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# Panel A: 权重对比——FCE 排除 vs PCA 权重
ax = axes[0]
fce_4 = list(entropy_weights) + [0.0]  # NMHC排除
pca_4 = [pca_weights_all[ALL_POLLUTANTS.index(p)] if p != 'NMHC' else pca_weights_all[1]
         for p in FCE_POLLUTANTS + ['NMHC']]
labels_a = list(FCE_POLLUTANTS) + ['NMHC']
x = np.arange(5)
w = 0.35
bars1 = ax.bar(x - w/2, fce_4, w, label='FCE 熵权 (4项+N)', color='#3498db', alpha=0.85)
bars2 = ax.bar(x + w/2, pca_4, w, label='PCA 权重 (5项全量)', color='#e74c3c', alpha=0.85)
# 高亮 NMHC
bars1[4].set_facecolor('#bdc3c7')
bars1[4].set_alpha(0.6)
bars2[4].set_facecolor('#e74c3c')
bars2[4].set_edgecolor('#1a1a2e')
bars2[4].set_linewidth(2)
ax.set_xticks(x)
ax.set_xticklabels(labels_a, fontsize=11)
ax.set_ylabel('权重', fontsize=11)
ax.set_title('A. NMHC 被 FCE 排除 (权重=0)\nPCA 给出权重 0.162', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2, axis='y')
for bar, val in zip(bars1, fce_4):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
            ha='center', fontsize=7.5)
for bar, val in zip(bars2, pca_4):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
            ha='center', fontsize=7.5)

# Panel B: PCA 载荷图
ax = axes[1]
load_sorted = loadings.abs().copy()
colors_pca = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
for k in range(2):  # PC1, PC2
    ax.bar(np.arange(5) + k*0.3 - 0.15, loadings.iloc[:, k], 0.28,
           label=f'PC{k+1} ({pca.explained_variance_ratio_[k]*100:.1f}%)',
           color=colors_pca[k], alpha=0.85)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels(ALL_POLLUTANTS, fontsize=11)
ax.set_ylabel('载荷', fontsize=11)
ax.set_title(f'B. PCA 载荷: NMHC 在 PC2 上\n载荷高达 0.970 (解释 18.7%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2, axis='y')

# Panel C: SHAP 重要性
ax = axes[2]
shap_sensors = ['PT08.S2_NMHC', 'PT08.S5_O3', 'PT08.S4_NO2', 'PT08.S1_CO', 'PT08.S3_NOx', 'hour']
shap_vals = [shap_data[s] for s in shap_sensors]
shap_labels = [sensor_labels[s] for s in shap_sensors]
colors_c = ['#e74c3c', '#e74c3c', '#f39c12', '#3498db', '#3498db', '#95a5a6']
bars = ax.barh(range(len(shap_vals))[::-1], shap_vals[::-1],
               color=colors_c[::-1], alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(shap_vals))[::-1])
ax.set_yticklabels(shap_labels[::-1], fontsize=10)
ax.set_xlabel('mean|SHAP|', fontsize=11)
ax.set_title('C. SHAP 重要性: S2(NMHC) 排名第一\n(mean|SHAP| = 6.65)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.2, axis='x')
for bar, val in zip(bars, shap_vals[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
            va='center', fontsize=9, fontweight='bold')

plt.suptitle('NMHC 重要性多证据链：FCE 排除 ≠ 环境不重要', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'q3_nmhc_evidence.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [Q3-2] q3_nmhc_evidence.png')

# ============================================================
# 图 Q3-3: AQI 分布 + 预警阈值
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

n, bins, patches = ax.hist(df['AQI'], bins=55, color='steelblue', alpha=0.75, edgecolor='white', linewidth=0.3)

# 等级背景色带
grade_boundaries = [11, 30, 45, 60, 75, 90]
grade_colors_bg = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59']
grade_names_bg = ['优', '良', '轻度污染', '中度污染', '重度污染']
for i in range(len(grade_boundaries)-1):
    ax.axvspan(grade_boundaries[i], grade_boundaries[i+1], alpha=0.08, color=grade_colors_bg[i])
    mid = (grade_boundaries[i] + grade_boundaries[i+1]) / 2
    ax.text(mid, ax.get_ylim()[1]*0.95 if i == 0 else ax.get_ylim()[1]*0.95,
            grade_names_bg[i], ha='center', fontsize=9, color=grade_colors_bg[i],
            fontweight='bold', alpha=0.7)

# 预警阈值线
thresholds = [(50, '#f39c12', '关注级\nAQI≥50'),
              (65, '#e67e22', '警示级\nAQI≥65'),
              (80, '#c0392b', '警戒级\nAQI≥80')]
for thresh, color, label in thresholds:
    ax.axvline(x=thresh, color=color, linewidth=2.5, linestyle='--', alpha=0.8)
    ax.text(thresh + 0.5, ax.get_ylim()[1]*0.88, label, color=color, fontsize=9.5,
            fontweight='bold', va='top')

# 统计标注
ax.text(0.98, 0.95, f'均值={df["AQI"].mean():.1f}\n标准差={df["AQI"].std():.1f}\nN={len(df)}',
        transform=ax.transAxes, fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

ax.set_xlabel('AQI', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('AQI 分布与分级预警阈值', fontsize=14, fontweight='bold')
ax.set_xlim(10, 90)
ax.grid(alpha=0.15, axis='y')
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'q3_warning_thresholds.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [Q3-3] q3_warning_thresholds.png')

# ============================================================
# 图 Q3-4: 传感器剪枝决策支持
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))

configs = list(prune_r2.keys())[::-1]
r2_vals = [prune_r2[c] for c in configs][::-1]

# 颜色：最优配置用绿色，基线用蓝色，移除后下降用红色
colors_bar = []
for c in configs:
    if 'Top3' in c:
        colors_bar.append('#27ae60')
    elif '全特征' in c:
        colors_bar.append('#3498db')
    else:
        colors_bar.append('#e74c3c')

bars = ax.barh(range(len(configs)), r2_vals, color=colors_bar, alpha=0.85, edgecolor='white')

# 基线参考线
ax.axvline(x=prune_r2['全特征(6)'], color='#3498db', linewidth=1.5, linestyle='--', alpha=0.6,
           label=f'全特征基线 R²={prune_r2["全特征(6)"]:.3f}')

# 标注
for bar, val, cfg in zip(bars, r2_vals, configs):
    delta = val - prune_r2['全特征(6)']
    sign = '+' if delta > 0 else ''
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f} ({sign}{delta:.3f})', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(configs)))
ax.set_yticklabels(configs, fontsize=10)
ax.set_xlabel('R² (5折时间序列交叉验证)', fontsize=12)
ax.set_title('传感器特征剪枝：SHAP Top3 配置最优', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.2, axis='x')
ax.set_xlim(0.50, 0.82)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'q3_sensor_pruning.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [Q3-4] q3_sensor_pruning.png')

print('\nQ3 辅助图表生成完成。')
