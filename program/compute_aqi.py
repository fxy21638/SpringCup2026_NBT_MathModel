"""
基于参考污染物浓度计算空气质量指数 (AQI)。

方法：滑动窗口百分位法
- 对5种参考污染物(CO, NMHC, C6H6, NOx, NO2)，使用 expanding window 计算分位数
- 每个时刻仅使用该时刻之前的历史数据，杜绝时序泄漏
- NMHC 原始数据缺失率 90.2%（8126小时连续缺口），处理方案：
  * 短缺口（≤6小时）：线性插值后参与 expanding rank
  * 长缺口（>6小时）：不参与 rank，子指数置 NaN 不纳入 max 计算
  * 此方案在有 NMHC 数据时段纳入评价，无数据时段基于其余4种污染物计算
- 子指数 = 分位数 × 100
- 综合 AQI = max(各子指数)，取值 0~100

输出：output/data_with_aqi.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('output')
DATA_CLEAN_PATH = OUTPUT_DIR / 'data_clean.csv'
DATA_WITH_AQI_PATH = OUTPUT_DIR / 'data_with_aqi.csv'
RAW_DATA_PATH = Path('题目/B题附件：AirQualityUCI.xlsx')

# 5种参考污染物（题目要求包含各项污染物指标）
POLLUTANTS = ['CO', 'NMHC', 'C6H6', 'NOx', 'NO2']

# 原始 Excel 列名映射（仅 NMHC 需要读取缺失标记）
COL_MAP = {
    'CO(GT)': 'CO', 'NMHC(GT)': 'NMHC', 'C6H6(GT)': 'C6H6',
    'NOx(GT)': 'NOx', 'NO2(GT)': 'NO2',
}

# AQI 等级划分
GRADE_BINS = [0, 20, 40, 60, 80, 100]
GRADE_LABELS = ['优(0-20)', '良(20-40)', '轻度污染(40-60)', '中度污染(60-80)', '重度污染(80-100)']


def get_nmhc_long_gap_mask() -> np.ndarray:
    """
    从原始 Excel 读取 NMHC 的 -200 标记，识别长缺口（>6小时）位置。
    返回 bool 数组: True = 长缺口填充值（不参与 rank），False = 真实值或短缺口。
    """
    raw = pd.read_excel(RAW_DATA_PATH, header=0)
    raw = raw.rename(columns=COL_MAP)
    raw = raw[raw['Date'].notna()].reset_index(drop=True)

    if 'NMHC' not in raw.columns:
        return np.array([])

    raw['NMHC'] = pd.to_numeric(raw['NMHC'], errors='coerce')
    is_missing = (raw['NMHC'] == -200).values

    # 标记连续缺失段，找出 >6 小时的长缺口
    run_id = (is_missing != np.roll(is_missing, 1)).cumsum()
    run_sizes = {}
    for i in range(len(is_missing)):
        if is_missing[i]:
            rid = run_id[i]
            run_sizes[rid] = run_sizes.get(rid, 0) + 1

    long_gap_run_ids = {k for k, v in run_sizes.items() if v > 6}
    long_gap_mask = np.array([
        is_missing[i] and run_id[i] in long_gap_run_ids
        for i in range(len(is_missing))
    ])

    return long_gap_mask


def compute_expanding_percentile_aqi(df: pd.DataFrame, nmhc_long_gap_mask: np.ndarray) -> pd.DataFrame:
    """
    使用 expanding window 计算每个时刻的分位数，避免时序泄漏。
    NMHC 长缺口位置不参与 expanding rank，子指数取前向填充值。
    """
    df = df.copy()
    df = df.sort_values('Datetime').reset_index(drop=True)

    sub_indices = {}
    for pol in POLLUTANTS:
        if pol not in df.columns:
            continue

        if pol == 'NMHC' and len(nmhc_long_gap_mask) > 0:
            # NMHC: 仅在非长缺口位置计算 expanding rank，长缺口处置 NaN（不参与 max）
            nmhc_values = df[pol].values
            n = len(nmhc_values)
            mask = nmhc_long_gap_mask[:n] if len(nmhc_long_gap_mask) >= n else \
                   np.pad(nmhc_long_gap_mask, (0, n - len(nmhc_long_gap_mask)), constant_values=False)

            sub_idx = np.full(n, np.nan)
            real_values = []
            for i in range(n):
                if not mask[i]:
                    real_values.append(nmhc_values[i])
                    if len(real_values) > 0:
                        arr = np.array(real_values)
                        rank_i = (arr <= nmhc_values[i]).sum() / len(arr)
                        sub_idx[i] = rank_i * 100
                # 长缺口位置保持 NaN，不参与后续 max 计算

            sub_indices[pol] = sub_idx
        else:
            expanding_rank = df[pol].expanding(min_periods=1).rank(pct=True)
            sub_indices[pol] = expanding_rank * 100

    # 综合 AQI = max(子指数)
    sub_df = pd.DataFrame(sub_indices)
    df['AQI'] = sub_df.max(axis=1)

    # 保留各污染物子指数
    for pol in sub_indices:
        df[f'AQI_{pol}'] = sub_indices[pol]

    # 等级划分
    df['AQI_等级'] = pd.cut(
        df['AQI'], bins=GRADE_BINS, labels=GRADE_LABELS,
        right=True, include_lowest=True
    )

    return df


def main():
    print('加载清洗后数据...')
    df = pd.read_csv(DATA_CLEAN_PATH)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f'数据形状: {df.shape}')

    print('识别 NMHC 长缺口（>6小时）填充位置...')
    nmhc_long_gap_mask = get_nmhc_long_gap_mask()
    n_long_gap = nmhc_long_gap_mask.sum()
    print(f'  NMHC 总缺失: {n_long_gap} / {len(nmhc_long_gap_mask)} ({n_long_gap/len(nmhc_long_gap_mask)*100:.1f}%)')
    print(f'  有 NMHC 真实数据时段纳入 AQI，长缺口段仅基于其余4种污染物')

    print('计算滑动窗口 AQI（expanding rank，杜绝时序泄漏）...')
    print('  使用污染物: CO, NMHC, C6H6, NOx, NO2')
    df = compute_expanding_percentile_aqi(df, nmhc_long_gap_mask)

    # 统计
    print(f'\nAQI 统计:')
    print(f'  均值:   {df["AQI"].mean():.2f}')
    print(f'  中位数: {df["AQI"].median():.2f}')
    print(f'  标准差: {df["AQI"].std():.2f}')
    print(f'  最小值: {df["AQI"].min():.2f}')
    print(f'  最大值: {df["AQI"].max():.2f}')

    print(f'\nAQI 等级分布:')
    grade_counts = df['AQI_等级'].value_counts().sort_index()
    for grade, count in grade_counts.items():
        print(f'  {grade}: {count} ({count/len(df)*100:.1f}%)')

    # 天花板检查
    top_val = df['AQI'].max()
    near_top = (df['AQI'] >= top_val * 0.99).sum()
    print(f'\n天花板检查: {near_top}/{len(df)} ({near_top/len(df)*100:.1f}%) 样本在 max 的 99% 以上')

    # 主导污染物分析
    sub_cols = [c for c in df.columns if c.startswith('AQI_') and c != 'AQI_等级']
    sub_df = df[sub_cols]
    dominant = sub_df.idxmax(axis=1).value_counts()
    print(f'\n主导污染物频次:')
    for pol, cnt in dominant.items():
        print(f'  {pol}: {cnt} ({cnt/len(df)*100:.1f}%)')

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_WITH_AQI_PATH, index=False, encoding='utf-8-sig', float_format='%.6g')
    print(f'\n已保存: {DATA_WITH_AQI_PATH}')


if __name__ == '__main__':
    main()
