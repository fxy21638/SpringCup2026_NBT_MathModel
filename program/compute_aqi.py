"""
基于参考污染物浓度计算连续型空气质量指数 (AQI)。

方法：连续百分位法
- 对每种参考污染物(CO, NMHC, C6H6, NOx, NO2)计算经验CDF分位数(0~1连续值)
- 子指数 = 分位数 × 100
- 综合 AQI = max(子指数)，取值 0~100
- 等级划分：优0-20, 良20-40, 轻度40-60, 中度60-80, 重度80-100

输出：program/data_with_aqi.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 参考污染物（真实分析仪测量值）
POLLUTANTS = ['CO', 'NMHC', 'C6H6', 'NOx', 'NO2']

# AQI 等级划分
GRADE_BINS = [0, 20, 40, 60, 80, 100]
GRADE_LABELS = ['优(0-20)', '良(20-40)', '轻度污染(40-60)', '中度污染(60-80)', '重度污染(80-100)']


def compute_percentile_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算连续百分位 AQI。
    对每种污染物，计算其在整个数据集中所处的分位数，
    然后取最大分位数 × 100 作为综合 AQI。
    """
    df = df.copy()

    # 对每种污染物计算分位数
    sub_indices = {}
    for pol in POLLUTANTS:
        if pol not in df.columns:
            continue
        # 使用 rank(pct=True) 计算经验CDF分位数 (0~1)
        percentile = df[pol].rank(pct=True)
        sub_indices[pol] = percentile * 100  # 映射到 0~100

    # 综合 AQI = max(子指数)
    sub_df = pd.DataFrame(sub_indices)
    df['AQI'] = sub_df.max(axis=1)

    # 各污染物的子指数也保留
    for pol in sub_indices:
        df[f'AQI_{pol}'] = sub_indices[pol]

    # 等级划分
    df['AQI_等级'] = pd.cut(df['AQI'], bins=GRADE_BINS, labels=GRADE_LABELS, right=True, include_lowest=True)

    return df


def main():
    print('加载清洗后数据...')
    df = pd.read_csv('program/data_clean.csv')
    print(f'数据形状: {df.shape}')

    print('计算连续百分位 AQI...')
    df = compute_percentile_aqi(df)

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

    # 检查天花板效应
    top_val = df['AQI'].max()
    near_top = (df['AQI'] >= top_val * 0.99).sum()
    print(f'\n天花板检查: 仅 {near_top}/{len(df)} ({near_top/len(df)*100:.1f}%) 样本在 max 的 99% 以上')

    # 保存
    df.to_csv('program/data_with_aqi.csv', index=False, encoding='utf-8-sig', float_format='%.6g')
    print(f'\n已保存: program/data_with_aqi.csv')


if __name__ == '__main__':
    main()
