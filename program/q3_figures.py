"""
第三题论文图表生成（简化版）。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys, warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('output')
PICTURE_DIR = OUTPUT_DIR / 'picture'
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / 'data_with_aqi.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour
df['month'] = df['Datetime'].dt.month

sys.path.insert(0, str(Path(__file__).parent))
from compute_aqi import (compute_entropy_weights, build_membership_breakpoints,
                         build_membership_matrix, POLLUTANTS as FCE_POLLUTANTS)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ALL_P = ['CO', 'NMHC', 'C6H6', 'NOx', 'NO2']
poll_scaled = StandardScaler().fit_transform(df[ALL_P].dropna())
pca = PCA().fit(poll_scaled)
loadings = pd.DataFrame(pca.components_.T, index=ALL_P, columns=[f'PC{i+1}' for i in range(5)])
pca_w_raw = np.zeros(5)
for j in range(5):
    for i in range(5):
        pca_w_raw[j] += abs(loadings.iloc[j, i]) * pca.explained_variance_ratio_[i]
pca_w_all = pca_w_raw / pca_w_raw.sum()

bp = build_membership_breakpoints(df)
R = build_membership_matrix(df, bp)
entropy_w = compute_entropy_weights(R)

# ============================================================
# 图 5.4-1: AQI 时空热力图（简化：无数字，纯颜色 + 窗口框）
# ============================================================
pivot = df.pivot_table(values='AQI', index='hour', columns='month', aggfunc='mean')

fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', vmin=20, vmax=65, origin='upper')

# 高污染窗口框
for (x0, xw) in [(9, 3), (0, 2)]:
    ax.add_patch(Rectangle((x0-0.5, 16.5), xw, 5, linewidth=2.5, edgecolor='#1a1a2e',
                           facecolor='none', linestyle='--', zorder=5))
    ax.add_patch(Rectangle((x0-0.5, 7.5), xw, 3, linewidth=1.8, edgecolor='#16213e',
                           facecolor='none', linestyle=':', zorder=5))

ax.set_xticks(range(12))
ax.set_xticklabels([f'{m}月' for m in range(1,13)], fontsize=10)
ax.set_yticks(range(0, 24, 2))
ax.set_yticklabels([f'{h}:00' for h in range(0, 24, 2)], fontsize=9)
ax.set_xlabel('月份', fontsize=12)
ax.set_ylabel('小时', fontsize=12)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('AQI', fontsize=10)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'q3_pollution_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [简化] q3_pollution_heatmap.png')

# ============================================================
# 图 5.4-2: NMHC 双证据面板（FCE排除 + SHAP重要性）
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: FCE 权重（NMHC 被排除）
ax = axes[0]
labels = list(FCE_POLLUTANTS) + ['NMHC\n(排除)']
w_vals = list(entropy_w) + [0.0]
colors_a = ['#3498db']*4 + ['#bdc3c7']
bars = ax.bar(range(5), w_vals, color=colors_a, alpha=0.85, edgecolor='white')
bars[4].set_hatch('///')
for bar, val in zip(bars, w_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
            ha='center', fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('权重', fontsize=11)
ax.set_title('A. NMHC 被 FCE 排除 (权重=0)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.2, axis='y')

# Panel B: SHAP 重要性 (水平条)
ax = axes[1]
sensors = ['PT08.S2_NMHC', 'PT08.S5_O3', 'PT08.S4_NO2', 'PT08.S1_CO', 'PT08.S3_NOx', 'hour']
shap_v = [6.65, 5.41, 3.65, 3.01, 2.34, 2.04]
labels_b = ['S2 (NMHC)', 'S5 (O₃)', 'S4 (NO₂)', 'S1 (CO)', 'S3 (NOx)', '小时']
colors_b = ['#e74c3c', '#e74c3c', '#f39c12', '#3498db', '#3498db', '#95a5a6']
bars = ax.barh(range(6)[::-1], shap_v[::-1], color=colors_b[::-1], alpha=0.85, edgecolor='white')
ax.set_yticks(range(6)[::-1]); ax.set_yticklabels(labels_b[::-1], fontsize=10)
ax.set_xlabel('mean|SHAP|', fontsize=11)
ax.set_title('B. SHAP 重要性: S2(NMHC) 排名第一', fontsize=11, fontweight='bold')
ax.grid(alpha=0.2, axis='x')
for bar, val in zip(bars, shap_v[::-1]):
    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f'{val:.1f}',
            va='center', fontsize=9, fontweight='bold')

plt.suptitle('NMHC 的FCE排除与传感器证据的对比', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'q3_nmhc_evidence.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [简化] q3_nmhc_evidence.png')

# ============================================================
# 图 5.4-3: AQI 分布 + 预警阈值（保留，轻微去杂）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
n, bins, patches = ax.hist(df['AQI'], bins=50, color='#5D8AA8', alpha=0.8, edgecolor='white', linewidth=0.3)

# 等级色带
for lo, hi, c, name in [(11,30,'#1a9850','优'),(30,45,'#91cf60','良'),(45,60,'#d9ef8b','轻度'),
                          (60,75,'#fee08b','中度'),(75,90,'#fc8d59','重度')]:
    ax.axvspan(lo, hi, alpha=0.07, color=c)

# 三级阈值线
for thresh, color, label in [(50, '#f39c12', '关注级\nAQI≥50'),
                              (65, '#e67e22', '警示级\nAQI≥65'),
                              (80, '#c0392b', '警戒级\nAQI≥80')]:
    ax.axvline(x=thresh, color=color, linewidth=2.5, linestyle='--', alpha=0.8)
    ax.text(thresh+0.5, ax.get_ylim()[1]*0.86, label, color=color, fontsize=9, fontweight='bold')

ax.text(0.98, 0.95, f'μ={df.AQI.mean():.1f}  σ={df.AQI.std():.1f}',
        transform=ax.transAxes, fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
ax.set_xlabel('AQI', fontsize=12); ax.set_ylabel('频数', fontsize=12)
ax.set_xlim(10, 90); ax.grid(alpha=0.15, axis='y')
plt.tight_layout()
plt.savefig(PICTURE_DIR / 'q3_warning_thresholds.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [简化] q3_warning_thresholds.png')

print('\nQ3 论文图表生成完成。')
