"""
基于参考污染物浓度计算空气质量指数 (AQI)。

方法：滑动窗口百分位法
- 对4种参考污染物(CO, C6H6, NOx, NO2)，使用 expanding window 计算分位数
- 每个时刻仅使用该时刻之前的历史数据，杜绝时序泄漏
- NMHC 因 90.2% 数据缺失（8126小时连续缺口）已从计算中移除
- 子指数 = 分位数 × 100
- 综合 AQI = max(子指数)，取值 0~100

输出：output/data_with_aqi.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('output')
DATA_CLEAN_PATH = OUTPUT_DIR / 'data_clean.csv'
DATA_WITH_AQI_PATH = OUTPUT_DIR / 'data_with_aqi.csv'

# 参考污染物（移除 NMHC：90.2% 数据缺失，8126 小时连续缺口）
POLLUTANTS = ['CO', 'C6H6', 'NOx', 'NO2']

# AQI 等级划分
GRADE_BINS = [0, 20, 40, 60, 80, 100]
GRADE_LABELS = ['优(0-20)', '良(20-40)', '轻度污染(40-60)', '中度污染(60-80)', '重度污染(80-100)']


def compute_expanding_percentile_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 expanding window 计算每个时刻的分位数，避免时序泄漏。
    - 每个 t 时刻仅使用 [0:t] 范围内的数据计算 rank(pct=True)
    - 无未来数据泄露，严格满足时间序列建模要求
    """
    df = df.copy()
    df = df.sort_values('Datetime').reset_index(drop=True)

    sub_indices = {}
    for pol in POLLUTANTS:
        if pol not in df.columns:
            continue
        # expanding().rank(pct=True): 在已观察到的数据中计算分位数
        # min_periods=1: 第1个样本 rank=1.0，后续逐步稳定
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

    print('计算滑动窗口 AQI（expanding rank，杜绝时序泄漏）...')
    print('  使用污染物: CO, C6H6, NOx, NO2 (NMHC 已移除)')
    df = compute_expanding_percentile_aqi(df)

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
