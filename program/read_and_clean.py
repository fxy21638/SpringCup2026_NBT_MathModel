"""
读取并清洗 AirQualityUCI 数据的脚本。
- 读取 Excel（默认路径：题目/B题附件：AirQualityUCI.xlsx）
- 将 -200 替换为 NaN（缺失值），并进行插值填充
- 合并 Date + Time 为 Datetime
- 保存清洗后的 CSV 到 output/data_clean.csv
- 输出缺失率表并保存为 output/missing_summary.csv
- 生成缺失热图到 output/picture/

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
import seaborn as sns

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


def clean_values(df: pd.DataFrame, fill_method: str = 'interpolate') -> pd.DataFrame:
    """将 -200 替换为 NaN 并按间隙长度分策略填充"""
    measure_cols = [
        'CO', 'PT08.S1_CO', 'NMHC', 'C6H6', 'PT08.S2_NMHC',
        'NOx', 'PT08.S3_NOx', 'NO2', 'PT08.S4_NO2', 'PT08.S5_O3',
        'T', 'RH', 'AH'
    ]

    # 强制数值类型
    for col in measure_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # -200 → NaN
    df = df.replace(-200, np.nan)

    numeric_cols = [c for c in measure_cols if c in df.columns]

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
    """生成缺失值热图（前1000行数值列）"""
    plot_df = df.select_dtypes(include=[np.number])
    if plot_df.empty:
        print('No numeric columns for missing heatmap; skipping.')
        return
    sample = plot_df.iloc[:min(1000, len(plot_df))]
    if sample.shape[0] == 0 or sample.shape[1] == 0:
        print('No data for missing heatmap; skipping.')
        return
    try:
        plt.figure(figsize=(12, max(4, sample.shape[1] * 0.5)))
        sns.heatmap(sample.isnull(), cbar=False, cmap='Reds')
        plt.title('Missingness heatmap (numeric cols, first 1000 rows)')
        plt.tight_layout()
        plt.savefig(out_png, dpi=100)
        plt.close()
    except Exception as e:
        print(f'Failed to plot missing heatmap: {e}')


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

    print('Replacing -200 with NaN and filling...')
    df = clean_values(df, fill_method=args.fill)

    # 保存清洗后数据
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig', float_format='%.6g')
    print(f'Saved cleaned data to {out_csv}')

    # 缺失统计
    missing_summary = save_missing_summary(df, MISSING_SUMMARY_OUT)
    print('Missing summary:')
    print(missing_summary)

    # 缺失热图
    plot_missing_heatmap(df, picture_dir / 'missing_heatmap.png')
    print(f'Plots saved to {picture_dir}')

    # 验证 Datetime
    if 'Datetime' in df.columns:
        print(f'Datetime range: {df["Datetime"].min()} → {df["Datetime"].max()}')
        print(f'Sample Datetimes: {df["Datetime"].head(5).tolist()}')


if __name__ == '__main__':
    main()
