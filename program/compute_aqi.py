"""
基于模糊综合评价的空气质量指数 (AQI) 计算。

方法：模糊综合评价模型 (Fuzzy Comprehensive Evaluation)
- 评价指标集 U = {CO, C6H6, NOx, NO2}（4 项真实污染物，NMHC 因 90% 缺失不参与评价）
- 评价等级集 V = {优, 良, 轻度污染, 中度污染, 重度污染}
- 隶属度函数：梯形/半梯形（基于各污染物浓度分位数确定断点）
- 权重向量 W：熵权法客观赋权
- 模糊合成算子：M(·,+) 加权平均
- 综合评判：最大隶属度原则判定等级
- 连续 AQI 得分：去模糊化加权平均（等级中心值加权）

特点：
- 贴合空气质量等级边界的模糊本质
- 只使用 5 项真实污染物，不涉及传感器或气象
- 仅使用有真实测量的污染物，NMHC 因 90% 填充率不参与评价
- 每时刻独立评价，无时序泄漏

输出：output/data_with_aqi.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('output')
DATA_CLEAN_PATH = OUTPUT_DIR / 'data_clean.csv'
DATA_WITH_AQI_PATH = OUTPUT_DIR / 'data_with_aqi.csv'
RAW_DATA_PATH = Path('题目/B题附件：AirQualityUCI.xlsx')

# 原始 Excel 列名映射
COL_MAP = {
    'CO(GT)': 'CO', 'NMHC(GT)': 'NMHC', 'C6H6(GT)': 'C6H6',
    'NOx(GT)': 'NOx', 'NO2(GT)': 'NO2',
}

# 5 种参考污染物（题目要求的各项污染物指标）
POLLUTANTS = ['CO', 'C6H6', 'NOx', 'NO2']

# 评价等级
GRADE_NAMES = ['优', '良', '轻度污染', '中度污染', '重度污染']
GRADE_CENTERS = np.array([10.0, 30.0, 50.0, 70.0, 90.0])  # 各等级中心值（0~100）
N_GRADES = len(GRADE_NAMES)


def get_real_values(pol: str) -> np.ndarray:
    """从原始数据读取某污染物的真实（非 -200）浓度值，用于计算隶属度断点"""
    raw = pd.read_excel(RAW_DATA_PATH, header=0)
    raw = raw.rename(columns=COL_MAP)
    raw = raw[raw['Date'].notna()].reset_index(drop=True)
    if pol not in raw.columns:
        return np.array([])
    raw[pol] = pd.to_numeric(raw[pol], errors='coerce')
    real = raw.loc[raw[pol] != -200, pol].dropna().values
    return real


def build_membership_breakpoints(df: pd.DataFrame) -> dict:
    """
    为每种污染物构造梯形隶属度函数的断点。
    使用真实（非填充）浓度值的分位数确定，保证绝对物理含义。
    最小间隔保证：相邻断点间距 ≥ (max-min)/20，防止梯形退化。

    返回: {pollutant: np.array of shape (5, 4)}
    """
    breakpoints = {}
    for pol in POLLUTANTS:
        if pol not in df.columns:
            continue
        real = get_real_values(pol)
        if len(real) == 0:
            vals = df[pol].dropna().values
        else:
            vals = real

        if len(vals) == 0:
            continue

        vmin, vmax = vals.min(), vals.max()
        min_gap = (vmax - vmin) / 20  # 最小间隔

        # 分位数断点，确保单调递增且有最小间隔
        p_raw = np.percentile(vals, [20, 40, 60, 80])
        p20 = p_raw[0]
        p40 = max(p_raw[1], p20 + min_gap)
        p60 = max(p_raw[2], p40 + min_gap)
        p80 = max(p_raw[3], p60 + min_gap)

        # 三角形隶属度（无平台段，避免 AQI 退化聚集）
        # 峰点序列: vmin → p40 → p60 → p80 → vmax
        params = np.array([
            [vmin, vmin, vmin, p40],   # 优: 降半三角（峰在 vmin）
            [vmin, p40,  p40,  p60],   # 良: 三角形（峰在 p40）
            [p40,  p60,  p60,  p80],   # 轻度: 三角形（峰在 p60）
            [p60,  p80,  p80,  vmax],  # 中度: 三角形（峰在 p80）
            [p80,  vmax, vmax, vmax],  # 重度: 升半三角（峰在 vmax）
        ])
        breakpoints[pol] = params
    return breakpoints


def trapezoid_membership(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    梯形隶属度函数:
      μ(x) = 0,                x ≤ a 或 x ≥ d
      μ(x) = (x-a)/(b-a),      a < x < b
      μ(x) = 1,                b ≤ x ≤ c
      μ(x) = (d-x)/(d-c),      c < x < d
    """
    result = np.zeros_like(x, dtype=float)
    # 上升段
    mask_up = (x > a) & (x < b)
    if b != a:
        result[mask_up] = (x[mask_up] - a) / (b - a)
    # 平台段
    mask_plateau = (x >= b) & (x <= c)
    result[mask_plateau] = 1.0
    # 下降段
    mask_down = (x > c) & (x < d)
    if d != c:
        result[mask_down] = (d - x[mask_down]) / (d - c)
    return result


def build_membership_matrix(df: pd.DataFrame, breakpoints: dict) -> np.ndarray:
    """
    构建隶属度矩阵 R，形状 (n_samples, n_pollutants, n_grades)。
    R[i, j, k] = 第 i 个样本第 j 种污染物对第 k 个等级的隶属度。
    """
    n = len(df)
    p = len(POLLUTANTS)
    R = np.zeros((n, p, N_GRADES))

    for j, pol in enumerate(POLLUTANTS):
        if pol not in breakpoints:
            continue
        vals = df[pol].values.astype(float)
        bp = breakpoints[pol]  # shape (5, 4)
        for k in range(N_GRADES):
            a, b, c, d = bp[k]
            R[:, j, k] = trapezoid_membership(vals, a, b, c, d)

    return R


def compute_entropy_weights(R: np.ndarray) -> np.ndarray:
    """
    熵权法计算各污染物权重。

    步骤:
    1. 聚合：计算各污染物对各等级的平均隶属度 → (n_samples, p) 的综合得分
    2. 归一化：min-max 到 [0.01, 1] 避免 log(0)
    3. 计算熵值 E_j
    4. 权重 w_j = (1 - E_j) / Σ(1 - E_j)

    返回: weights, shape (p,)
    """
    n, p, k = R.shape

    # 将隶属度矩阵转换为综合得分: 对每个样本每个污染物，加权等级中心
    scores = R @ GRADE_CENTERS  # (n, p), 每个样本每污染物得分 0~100

    # Min-max 归一化到 [eps, 1]
    eps = 0.01
    s_min = scores.min(axis=0)
    s_max = scores.max(axis=0)
    denom = s_max - s_min
    denom[denom == 0] = 1.0
    s_norm = (scores - s_min) / denom
    s_norm = np.clip(s_norm, eps, 1.0)

    # 计算比重 p_ij
    col_sums = s_norm.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    p_ij = s_norm / col_sums

    # 熵值
    k_entropy = 1.0 / np.log(n) if n > 1 else 1.0
    E = -k_entropy * np.sum(p_ij * np.log(p_ij + 1e-12), axis=0)

    # 权重
    d = 1 - E  # 信息效用值
    weights = d / d.sum()
    return weights


def fuzzy_evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """
    执行模糊综合评价全流程。
    """
    df = df.copy()
    df = df.sort_values('Datetime').reset_index(drop=True)
    n = len(df)

    # 1. 构建隶属度断点
    print('  [1/4] 构建梯形隶属度函数（基于全局分位数断点）...')
    breakpoints = build_membership_breakpoints(df)
    for pol in POLLUTANTS:
        if pol in breakpoints:
            bp = breakpoints[pol]
            print(f'    {pol:6s}: 优≤{bp[0,3]:.1f}(峰{bp[0,1]:.1f}), 良峰{bp[1,1]:.1f}≤{bp[1,3]:.1f}, 轻度峰{bp[2,1]:.1f}≤{bp[2,3]:.1f}, 中度峰{bp[3,1]:.1f}≤{bp[3,3]:.1f}, 重度≥{bp[4,0]:.1f}(峰{bp[4,1]:.1f})')

    # 2. 构建隶属度矩阵
    print('  [2/4] 构建隶属度矩阵 R (9357×5×5)...')
    R = build_membership_matrix(df, breakpoints)

    # 3. 熵权法计算权重
    print('  [3/4] 熵权法计算各污染物权重...')
    weights = compute_entropy_weights(R)
    for j, pol in enumerate(POLLUTANTS):
        print(f'    {pol:6s}: w={weights[j]:.4f}')

    # 4. 模糊合成 B = W · R（对每个样本，加权平均各污染物的隶属度）
    print('  [4/4] 模糊合成 M(·,+) ...')
    B = np.zeros((n, N_GRADES))  # 综合隶属度矩阵
    for i in range(n):
        for k in range(N_GRADES):
            B[i, k] = np.sum(weights * R[i, :, k])

    # 确保 B 每行归一化
    B_sum = B.sum(axis=1, keepdims=True)
    B_sum[B_sum == 0] = 1.0
    B = B / B_sum

    # 最大隶属度原则判定等级
    grade_indices = np.argmax(B, axis=1)  # 0-based
    df['AQI_等级'] = [GRADE_NAMES[idx] for idx in grade_indices]

    # 连续 AQI 得分：去模糊化加权平均
    df['AQI'] = B @ GRADE_CENTERS  # (n,)
    df['AQI'] = df['AQI'].clip(0, 100)

    # 保留模糊隶属度（各等级的隶属度）
    for k in range(N_GRADES):
        df[f'隶属度_{GRADE_NAMES[k]}'] = B[:, k]

    # 保留各污染物子得分（用于分析）
    for j, pol in enumerate(POLLUTANTS):
        df[f'AQI_{pol}'] = R[:, j, :] @ GRADE_CENTERS

    return df


def main():
    print('加载清洗后数据...')
    df = pd.read_csv(DATA_CLEAN_PATH)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f'数据形状: {df.shape}')

    print('\n模糊综合评价模型:')
    print('  评价指标: CO, C6H6, NOx, NO2（NMHC 因 90% 填充不参与）')
    print('  评价等级: 优 / 良 / 轻度污染 / 中度污染 / 重度污染')
    print('  隶属度函数: 梯形/半梯形（基于全局分位数断点）')
    print('  权重方法: 熵权法（客观赋权）')
    print('  模糊合成: M(·,+) 加权平均')

    df = fuzzy_evaluate(df)

    # 统计
    print(f'\nAQI (连续得分) 统计:')
    print(f'  均值:   {df["AQI"].mean():.2f}')
    print(f'  中位数: {df["AQI"].median():.2f}')
    print(f'  标准差: {df["AQI"].std():.2f}')
    print(f'  最小值: {df["AQI"].min():.2f}')
    print(f'  最大值: {df["AQI"].max():.2f}')

    print(f'\n评价等级分布（最大隶属度原则）:')
    grade_counts = df['AQI_等级'].value_counts()
    for grade in GRADE_NAMES:
        cnt = grade_counts.get(grade, 0)
        bar = '█' * int(cnt / len(df) * 50)
        print(f'  {grade:6s}: {cnt:5d} ({cnt/len(df)*100:5.1f}%) {bar}')

    # 天花板检查
    top_val = df['AQI'].max()
    near_top = (df['AQI'] >= top_val * 0.99).sum()
    print(f'\n天花板检查: {near_top}/{len(df)} ({near_top/len(df)*100:.1f}%) 样本在 max 的 99% 以上')

    # 各污染物子得分统计
    sub_cols = [c for c in df.columns if c.startswith('AQI_') and c != 'AQI_等级']
    print(f'\n各污染物子得分均值:')
    for col in sub_cols:
        print(f'  {col}: {df[col].mean():.2f}')

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_WITH_AQI_PATH, index=False, encoding='utf-8-sig', float_format='%.6g')
    print(f'\n已保存: {DATA_WITH_AQI_PATH}')


if __name__ == '__main__':
    main()
