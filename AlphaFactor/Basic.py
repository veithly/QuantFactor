import numpy as np
import pandas as pd
import talib as ta


def rank(A):
    """
    RANK(A) 向量 A 升序排序
    :param A:
    :return:
    """
    return np.argsort(np.argsort(A, axis=0), axis=0)


def corr(df1: pd.DataFrame, df2: pd.DataFrame, n):
    """
    CORR(A, B, n) 向量 A 和 B 过去 n 天的相关系数
    :param df1:
    :param df2:
    :param n:
    :return:
    """
    return df1.rolling(n).corr(df2)


def stddev(df, n):
    """
    STDDEV(A, n) 向量 A 过去 n 天的标准差
    :param df:
    :param n:
    :return:
    """
    return df.rolling(n).std()


def ret(close_df, n=1):
    """
    RET(A, n) 向量 A 过去 n 天的收益率
    :param close_df:
    :param n:
    :return:
    """
    return close_df.pct_change(n)


def vwap(close_df, volume_df):
    """
    VWAP(CLOSE, VOLUME) 成交量加权平均价
    :param close_df:
    :param volume_df:
    :return:
    """
    return (close_df * volume_df).cumsum() / volume_df.cumsum()


def dtm(open_df, high_df, n=1):
    """
    DTM: (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
    :param open_df:
    :param high_df:
    :param n:
    :return:
    """
    cond = open_df <= open_df.shift(1)  # a boolean Series
    term1 = high_df - open_df  # a numeric Series
    term2 = open_df - open_df.shift(1)  # a numeric Series
    result = cond * 0 + ~cond * term1.combine(term2, max)  # a numeric Series
    return result


def dbm(open_df, low_df, n=1):
    """
    DBM: (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    :param open_df:
    :param low_df:
    :param n:
    :return:
    """
    cond = open_df >= open_df.shift(1)  # a boolean Series
    term1 = open_df - low_df  # a numeric Series
    term2 = open_df - open_df.shift(1)  # a numeric Series
    result = cond * 0 + ~cond * term1.combine(term2, max)  # a numeric Series
    return result

def mean(df, n):
    """
    MEAN(A, n) 向量 A 过去 n 天的均值
    :param df:
    :param n:
    :return:
    """
    return df.rolling(n).mean()


def tr(high_df, low_df, close_df, n=1):
    """
    TR: MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
    :param high_df:
    :param low_df:
    :param close_df:
    :param n:
    :return:
    """
    return max(max(high_df - low_df, abs(high_df - close_df.shift(n))), abs(low_df - close_df.shift(n)))


def get_hd(high_df, n=1):
    """
    HD: HIGH-DELAY(HIGH,1)
    :param high_df:
    :param n:
    :return:
    """
    return high_df - high_df.shift(n)


def get_ld(low_df, n=1):
    """
    LD: DELAY(LOW,1)-LOW
    :param low_df:
    :param n:
    :return:
    """
    return low_df.shift(n) - low_df


def delta(df, n):
    """
    DELTA(A, n) 向量 A 过去 n 天的变化
    :param df:
    :param n:
    :return:
    """
    return df.diff(n)


def tsrank(A, n):
    """
    TSRANK(A, n) 序列 A 的末位值在过去 n 天的顺序排位
    :param A:
    :param n:
    :return:
    """
    # 将A转换为numpy数组
    A = np.array(A)
    # 获取A的长度
    m = len(A)
    # 创建一个空数组，用于存储排位结果
    rank_res = np.zeros(m)
    # 对于每个位置i，从i-n+1到i的子序列中，计算A[i]的排位
    for i in range(m):
        # 获取子序列的起始位置
        start = max(0, i - n + 1)
        # 获取子序列
        sub = A[start:i + 1]
        # 计算A[i]在子序列中的排位，即有多少元素小于等于它
        rank_res[i] = np.sum(sub <= A[i])
    # 返回排位结果
    return rank_res


def get_beta(x, y):
    """
    GET_BETA(x, y) 计算x和y的线性回归斜率
    """
    # 计算x和y的协方差矩阵
    cov = np.cov(x, y)
    # 计算x和y的相关系数
    # corr = np.corrcoef(x, y)[0, 1]
    # 计算x和y的标准差
    # std_x = np.std(x)
    # std_y = np.std(y)
    # 计算线性回归的斜率
    return cov[0, 1] / cov[0, 0]


def regbeta(A, B, n):
    """
    REGBETA(A, B, n) 每 n 期样本 A 对 B 做回归所得回归系数
    :param A:
    :param B:
    :param n:
    :return:
    """
    # A和B是输入的序列，长度相同，n是回归的周期
    if not isinstance(A, pd.Series):
        A = pd.Series(A)
    if not isinstance(B, pd.Series):
        B = pd.Series(B)
    # 初始化输出序列
    output = pd.Series(index=A.index)
    # 循环计算每n期的REGBETA值
    for i in range(n - 1, len(A)):
        # 取出A和B的第i-n+1到第i个值
        x = A.iloc[i - n + 1:i + 1]
        y = B.iloc[i - n + 1:i + 1]
        # 计算线性回归的斜率
        beta = get_beta(x, y)
        # 将REGBETA值赋给输出序列的第i个值
        output.iloc[i] = beta
    # 返回输出序列
    return output


def regresi(A, B, n):
    """
    REGRESI(A, B, n) 每 n 期样本 A 对 B 做回归所得的残差
    :param A:
    :param B:
    :param n:
    :return:
    """
    # A和B是输入的序列，长度相同，n是回归的周期
    if not isinstance(A, pd.Series):
        A = pd.Series(A)
    if not isinstance(B, pd.Series):
        B = pd.Series(B)
    # 初始化输出序列
    output = pd.Series(index=A.index)
    # 循环计算每n期的REGRSI值
    for i in range(n - 1, len(A)):
        # 取出A和B的第i-n+1到第i个值
        x = A.iloc[i - n + 1:i + 1]
        y = B.iloc[i - n + 1:i + 1]
        # 计算x和y的协方差矩阵
        beta = get_beta(x, y)
        alpha = np.mean(y) - beta * np.mean(x)
        # 计算线性回归的残差
        e = y - (alpha + beta * x)
        # 计算残差的标准差
        regrsi = np.std(e)
        # 将REGRSI值赋给输出序列的第i个值
        output.iloc[i] = regrsi
    # 返回输出序列
    return output


def filter_cond(A, condition):
    """
    FILTER(A, condition) 从 A 中筛选出满足条件 condition 的值
    :param A:
    :param condition:
    :return:
    """
    return A[condition]


def decaylinear(df, n):
    """
    DECAYLINEAR(A, n) 对 A 序列计算移动平均加权，其中权重对应 d,d-1,…,1（权重和为 1）
    :param df:
    :param n:
    :return:
    """
    if isinstance(df, np.ndarray): # 如果df是numpy.ndarray对象
        df = pd.DataFrame(df) # 把df转换成pandas.DataFrame对象
    return df.rolling(n).apply(lambda x: np.sum(x * np.arange(1, n + 1) / np.sum(np.arange(1, n + 1))))


def delay(A, n):
    """
    DELAY(A, n) 向量 A 过去 n 天的值
    :param A:
    :param n:
    :return:
    """
    if type(A) == pd.Series:
        return A.shift(n)
    else:
        A = np.roll(A, n)
        A[:n] = np.nan
        return A


def count(condition, n):
    """
    COUNT(condition, n) 计算前 n 期满足条件 condition 的样本个数
    :param condition:
    :param n:
    :return:
    """
    return ta.SUM(np.where(condition.astype(bool), 1, 0).astype(float), n)


def sum_if(x, n, condition):
    """
    SUMIF(x, n, condition) 计算前 n 期满足条件 condition 的样本值之和
    :param x:
    :param n:
    :param condition:
    :return:
    """
    return ta.SUM(np.where(condition, x, 0), n)


def coviance(A, B, n):
    """
    COVIANCE(A, B, n) 前 n 期样本 A 对 B 做回归所得回归系数
    :param A:
    :param B:
    :param n:
    :return:
    """
    return A.rolling(n).cov(B)


def zero_to_one(x):
    """
    ZERO2ONE(x) 将 x 中的 0 替换为 1
    :param x:
    :return:
    """
    return np.where(x == 0, 1, x)


def sma(arr, n, m=1):
    sma_values = np.zeros(len(arr))
    start_flag = False
    for i in range(len(arr)):
        if start_flag is False:
            if np.isnan(arr[i]):
                sma_values[i] = np.nan
            else:
                sma_values[i] = arr[i]
                start_flag = True
        else:
            sma_values[i] = (arr[i] * m + sma_values[i - 1] * (n - m)) / n
            if sma_values[i] == np.nan:
                sma_values[i] = 0

    return sma_values


def filter_cond(A, condition):
    """
    FILTER(A, condition) 从 A 中筛选出满足条件 condition 的值
    :param A:
    :param condition:
    :return:
    """
    # 如果A是numpy数组，使用numpy的where函数来筛选
    if isinstance(A, np.ndarray):
        return np.where(condition, A, 0)
    # 如果A是pandas的Series或DataFrame，使用pandas的mask函数来筛选
    elif isinstance(A, (pd.Series, pd.DataFrame)):
        return A.mask(~condition, 0)


def get_fama_french_factors(market_cap, book_to_market, returns, date):
    """
  Calculates the HML, SMB, and MKE factors proposed by Fama and French.

  Args:
    market_cap: A pandas DataFrame containing the market capitalizations for
        each stock and date.
    book_to_market: A pandas DataFrame containing the book to market ratios
        for each stock and date.
    returns: A pandas DataFrame containing the returns for each stock and date.
    date: A pandas DatetimeIndex containing the dates.

  Returns:
    A pandas DataFrame containing the HML, SMB, and MKE factors.
  """

    # Calculate the size factor.
    size = market_cap.groupby(date).median()

    # Calculate the value factor.
    value = book_to_market.groupby(date).median()

    # Calculate the momentum factor.
    momentum = returns.groupby(date).mean()

    # Calculate the HML factor.
    hml = value - size

    # Calculate the SMB factor.
    smb = (returns - momentum) - size

    # Calculate the MKE factor.
    mke = (returns - momentum) - value

    return hml, smb, mke
