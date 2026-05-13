"""
读取并清洗 AirQualityUCI 数据的脚本。
- 读取 Excel（默认路径：题目/B题附件：AirQualityUCI.xlsx）
- 将 -200 替换为 NaN（缺失值），并进行间隙感知填充
- 合并 Date + Time 为 Datetime
- 保存清洗后的 CSV 到 output/data_clean.csv
- 输出缺失率表并保存为 output/missing_summary.csv
- 生成缺失热图到 output/picture/

异常值处理策略：
  污染物浓度的高值属于真实污染事件，不视为统计"异常值"予以删除。
  不采用 3σ 准则（正态假设不适用于偏态污染数据，会误删高污染时段）。
  仅将 -200 标记为缺失值（仪器故障/未检出标识），替换为 NaN 后填充。
  后续 AQI 计算采用非参数 expanding rank 方法，天然对极值鲁棒。

用法：
python read_and_clean.py --input "题目/B题附件：AirQualityUCI.xlsx" --out_csv output/data_clean.csv
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

DEFAULT_INPUT = Path('题目/B题附件：AirQualityUCI.xlsx')
OUTPUT_DIR = Path('output')
DEFAULT_OUT = OUTPUT_DIR / 'data_clean.csv'
PICTURE_DIR = OUTPUT_DIR / 'picture'
MISSING_SUMMARY_OUT = OUTPUT_DIR / 'missing_summary.csv'

# 目标列名（标准化后）
TARGET_COLS = [
    'Date', 'Time', 'CO', 'PT08.S1_CO', 'NMHC', 'C6H6',
    'PT08.S2_NMHC', 'NOx', 'PT08.S3_NOx', 'NO2',
    'PT08.S4_NO2', 'PT08.S5_O3', 'T', 'RH', 'AH'
]

# 原始 Excel 列到标准列名的映射
COL_MAP = {
    'CO(GT)': 'CO',
    'NMHC(GT)': 'NMHC',
    'C6H6(GT)': 'C6H6',
    'NOx(GT)': 'NOx',
    'NO2(GT)': 'NO2',
    'PT08.S1(CO)': 'PT08.S1_CO',
    'PT08.S2(NMHC)': 'PT08.S2_NMHC',
    'PT08.S3(NOx)': 'PT08.S3_NOx',
    'PT08.S4(NO2)': 'PT08.S4_NO2',
    'PT08.S5(O3)': 'PT08.S5_O3',
}


def read_excel(path: Path) -> pd.DataFrame:
    """读取 Excel 并标准化列名，仅保留 15 个目标列"""
    df = pd.read_excel(path, header=0)

    # 重命名已知列
    df = df.rename(columns=COL_MAP)

    # 仅保留目标列
    available = [c for c in TARGET_COLS if c in df.columns]
    df = df[available].copy()

    # 确保 Date 列为 datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 确保 Time 为时间类型（可能是 str 或 time 对象）
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.time

    return df


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """使用 Timestamp.combine 安全拼接 Date + Time → Datetime"""
    if 'Date' not in df.columns or 'Time' not in df.columns:
        return df

    datetimes = []
    for _, row in df.iterrows():
        d = row['Date']
        t = row['Time']
        if pd.notna(d) and pd.notna(t):
            try:
                datetimes.append(pd.Timestamp.combine(d.date() if hasattr(d, 'date') else d, t))
            except Exception:
                datetimes.append(pd.NaT)
        else:
            datetimes.append(pd.NaT)

    df['Datetime'] = datetimes
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    return df


def fill_gap_aware(series: pd.Series, max_interp_gap: int = 6) -> pd.Series:
    """
    分段填充缺失值:
    - gap <= max_interp_gap: 线性插值（传感器短暂掉线）
    - gap >  max_interp_gap: 前向填充（传感器故障期，假设浓度不变）
    """
    s = series.copy()
    is_na = s.isna()

    # 标记连续 NaN 的运行编号
    run_id = (is_na != is_na.shift()).cumsum()
    na_runs = run_id[is_na]
    run_sizes = na_runs.value_counts().to_dict()

    # 短缺口：线性插值
    short_run_ids = {k for k, v in run_sizes.items() if v <= max_interp_gap}
    short_mask = is_na & run_id.isin(short_run_ids)
    s[short_mask] = np.nan  # 保持 NaN 供插值使用
    s = s.interpolate(method='linear', limit_direction='both', limit_area='inside')

    # 长缺口：前向填充（然后后向填首部残留）
    s = s.ffill().bfill()

    return s


MEASURE_COLS = [
    'CO', 'PT08.S1_CO', 'NMHC', 'C6H6', 'PT08.S2_NMHC',
    'NOx', 'PT08.S3_NOx', 'NO2', 'PT08.S4_NO2', 'PT08.S5_O3',
    'T', 'RH', 'AH'
]


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    """将 -200 替换为 NaN，强制数值类型。返回含 NaN 的 DataFrame（不填充）。"""
    for col in MEASURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.replace(-200, np.nan)


def clean_values(df: pd.DataFrame, fill_method: str = 'interpolate') -> pd.DataFrame:
    """按间隙长度分策略填充缺失值（-200 已替换为 NaN）"""
    numeric_cols = [c for c in MEASURE_COLS if c in df.columns]

    if fill_method == 'interpolate':
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = fill_gap_aware(df[col], max_interp_gap=6)
    elif fill_method == 'ffill':
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df


def save_missing_summary(df: pd.DataFrame, out_path: Path):
    """保存缺失率统计"""
    miss = df.isnull().sum().rename('missing_count')
    pct = (df.isnull().mean() * 100).rename('missing_pct')
    summary = pd.concat([miss, pct], axis=1)
    summary.to_csv(out_path, encoding='utf-8-sig')
    return summary


def plot_missing_heatmap(df: pd.DataFrame, out_png: Path):
    """缺失值可视化：上侧柱状图（缺失率%），下侧采样矩阵热图（每50行采样，展示缺失模式）"""
    numeric_cols = [c for c in MEASURE_COLS if c in df.columns]
    if not numeric_cols:
        print('No numeric columns for missing heatmap; skipping.')
        return

    # 采样：每 stride 行取一行，避免图像过于拥挤
    stride = 50
    sample_idx = list(range(0, len(df), stride))
    sample = df[numeric_cols].iloc[sample_idx]

    miss_pct = (sample.isnull().mean() * 100).sort_values(ascending=False)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 3], hspace=0.08)

    # ---- 上侧：柱状图 ----
    ax_bar = fig.add_subplot(gs[0])
    colors = ['#d7191c' if v > 50 else '#fdae61' if v > 10 else '#a6d96a' for v in miss_pct.values]
    ax_bar.bar(range(len(miss_pct)), miss_pct.values, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax_bar.set_xticks(range(len(miss_pct)))
    ax_bar.set_xticklabels([])
    ax_bar.set_ylabel('缺失率 (%)', fontsize=10)
    ax_bar.set_ylim(0, max(miss_pct.max() * 1.15, 10))
    ax_bar.grid(axis='y', alpha=0.25)
    for i, (_, v) in enumerate(miss_pct.items()):
        ax_bar.text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=7.5, fontweight='bold',
                    color='#c0392b' if v > 50 else '#555')
    # 高缺失率标注
    high_miss = miss_pct[miss_pct > 50]
    if len(high_miss) > 0:
        ax_bar.set_title(f'各变量缺失率（{high_miss.index[0]} 缺失 {high_miss.iloc[0]:.1f}%，需特殊处理）',
                         fontsize=12, fontweight='bold')

    # ---- 下侧：采样矩阵热图 ----
    ax_mat = fig.add_subplot(gs[1])
    miss_mask = sample.isnull().reindex(columns=miss_pct.index)
    ax_mat.imshow(miss_mask.T, aspect='auto', cmap=plt.cm.Reds, interpolation='nearest', vmin=0, vmax=1)
    ax_mat.set_yticks(range(len(miss_pct)))
    ax_mat.set_yticklabels(miss_pct.index, fontsize=8)
    ax_mat.set_xlabel(f'样本（每 {stride} 行采样，共 {len(sample_idx)} 个时间点）', fontsize=10)
    # X 轴刻度：显示对应的行索引
    xtick_step = max(1, len(sample_idx) // 10)
    ax_mat.set_xticks(range(0, len(sample_idx), xtick_step))
    ax_mat.set_xticklabels([str(sample_idx[i]) for i in range(0, len(sample_idx), xtick_step)], fontsize=7)

    plt.suptitle('数据缺失模式（-200 标记值替换为 NaN）', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=str(DEFAULT_INPUT))
    parser.add_argument('--out_csv', '-o', default=str(DEFAULT_OUT))
    parser.add_argument('--fill', choices=['interpolate', 'ffill', 'none'], default='interpolate')
    args = parser.parse_args()

    inp = Path(args.input)
    out_csv = Path(args.out_csv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    picture_dir = PICTURE_DIR
    picture_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading {inp}...')
    df = read_excel(inp)
    print(f'Raw data shape: {df.shape}')

    # 过滤可能的标题行混入
    df = df[df['Date'].notna()]

    print('Building Datetime from Date + Time...')
    df = build_datetime(df)

    print('Replacing -200 with NaN...')
    df = mark_missing(df)

    # 缺失热图（在填充之前生成，展示真实缺失模式）
    missing_summary = save_missing_summary(df, MISSING_SUMMARY_OUT)
    print('Missing summary:')
    print(missing_summary)
    plot_missing_heatmap(df, picture_dir / 'missing_heatmap.png')
    print(f'Missing heatmap saved to {picture_dir / "missing_heatmap.png"}')

    print('Filling missing values...')
    df = clean_values(df, fill_method=args.fill)

    # 保存清洗后数据
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig', float_format='%.6g')
    print(f'Saved cleaned data to {out_csv}')
    print(f'Plots saved to {picture_dir}')

    # 验证 Datetime
    if 'Datetime' in df.columns:
        print(f'Datetime range: {df["Datetime"].min()} → {df["Datetime"].max()}')
        print(f'Sample Datetimes: {df["Datetime"].head(5).tolist()}')


if __name__ == '__main__':
    main()
