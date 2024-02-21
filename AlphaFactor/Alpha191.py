from .Basic import *


def alpha191_1(volume_df, open_df, close_df, n=6):
    """
    Alpha#1	 (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    这个因子是对6期内的成交量变化率的排名和开盘收盘价涨跌幅的排名的相关系数取负值。这个因子反映了成交量和价格变动之间的反向关系，
    也就是说，当成交量增加时，价格下跌的可能性增大，反之亦然。这个因子可能捕捉到了市场的恐慌情绪或者逆势操作的信号。
    :param volume_df:
    :param open_df:
    :param close_df:
    :param n: 6
    :return:
    """
    return -1 * ta.CORREL(rank(delta(np.log(volume_df), 1)), rank(((close_df - open_df) / open_df)), n)


def alpha191_2(high_df, low_df, close_df):
    """
    Alpha#2 (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    计算当日收盘价与最低价的差值减去最高价与收盘价的差值，再除以当日最高价与最低价的差值，得到一个反映当日价格走势的指标，
    然后再计算这个指标与前一日的变化率，并乘以-1。这个因子反映了股票价格在短期内的波动性和趋势性，如果这个因子值较大，说
    明股票价格在当日有较大幅度的上涨，并且上涨幅度超过了前一日，反之则说明股票价格在当日有较大幅度的下跌，并且下跌幅度超
    过了前一日。
    :param high_df:
    :param low_df:
    :param close_df:
    :return:
    """
    return -1 * delta((((close_df - low_df) - (high_df - close_df)) / (high_df - low_df)), 1)


def alpha191_3(close_df, high_df, low_df, n=6):
    """
    Alpha#3 SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    计算当日收盘价与前一日收盘价的差值，如果两者相等，则为0，如果当日收盘价高于前一日收盘价，则为当日收盘价减去当日最低价和前一日收盘价中
    的较小值，如果当日收盘价低于前一日收盘价，则为当日收盘价减去当日最高价和前一日收盘价中的较大值，然后再对这个差值进行6日累加。这个因子
    反映了股票价格在短期内的变化幅度和方向，如果这个因子值较大，说明股票价格在近期有较强的上涨动能，反之则说明股票价格在近期有较强的下跌
    动能。
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: 6
    :return:
    """
    # 定义一个变量delay_close，表示上一期的收盘价
    delay_close = delay(close_df, 1)
    # 定义一个变量cond1，表示收盘价是否等于上一期的收盘价
    cond1 = close_df == delay_close
    # 定义一个变量cond2，表示收盘价是否大于上一期的收盘价
    cond2 = close_df > delay_close
    # 定义一个变量min_low，表示最低价和上一期收盘价的较小值
    min_low = np.minimum(low_df, delay_close)
    # 定义一个变量max_high，表示最高价和上一期收盘价的较大值
    max_high = np.maximum(high_df, delay_close)
    # 定义一个变量expr，表示表达式的结果
    expr = cond1 * 0 + (close_df - cond2 * min_low) + (close_df <= delay_close) * max_high
    # 返回表达式的6期求和结果
    return ta.SUM(expr, n)


def alpha191_4(volume_df, close_df, n=8, m=20):
    """
    Alpha4
    ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <
    ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) ||
    ((VOLUME / MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
    这个函数的意思是根据收盘价和成交量的短期变化，判断股票的涨跌趋势，并返回-1或1作为信号。具体来说，如果收盘价的8日均值加上标准差小于2日均值
    ，说明股票价格下跌压力大，返回-1；如果收盘价的2日均值小于8日均值减去标准差，说明股票价格上涨动能强，返回1
    :param volume_df:
    :param close_df:
    :param n: 8
    :param m: 20
    :return:
    """
    sum_close_n = ta.SUM(close_df, n) / n
    std_close_n = ta.STDDEV(close_df, n)

    cond1 = (sum_close_n + std_close_n) < (ta.SUM(close_df, 2) / 2)
    cond2 = (ta.SUM(close_df, 2) / 2) < (sum_close_n - std_close_n)
    cond3 = (1 < (volume_df / ta.MA(volume_df, m))) | ((volume_df / ta.MA(volume_df, m)) == 1)

    return np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))


def alpha191_5(volume_df, high_df, n=5, m=3):
    """
    Alpha5 (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    计算成交量的5日顺序排位与最高价的5日顺序排位的5日相关系数，然后再对这个相关系数取3日内的最大值，并乘以-1。这个因子反映了股票价格与成交量
    在短期内的协同性，如果这个因子值较小，说明股票价格与成交量在近期有较强的正相关性，即价格上涨时成交量增加，价格下跌时成交量减少，反之则说
    明股票价格与成交量在近期有较强的负相关性，即价格上涨时成交量减少，价格下跌时成交量增加。
    :param volume_df:
    :param high_df:
    :param n: 5
    :param m: 3
    :return:
    """
    return -1 * ta.MAX(ta.CORREL(tsrank(volume_df, n), tsrank(high_df, n), n), m)


def alpha191_6(open_df, high_df, n=4):
    """
    Alpha6 (-1 * RANK(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))
    计算当日开盘价的0.85倍加上当日最高价的0.15倍，得到一个反映当日价格走势的指标，然后再计算这个指标与4日前的差值，并对这个差值进行降序
    排位，并乘以-1。这个因子反映了股票价格在短期内的变化速度和排名，如果这个因子值较大，说明股票价格在近期有较快的上涨，并且在所有股票中
    排名靠前，反之则说明股票价格在近期有较快的下跌，并且在所有股票中排名靠后。
    :param open_df:
    :param high_df:
    :param n:
    :return:
    """
    return -1 * rank(delta((open_df * 0.85 + high_df * 0.15), n))


def alpha191_7(volume_df, close_df, vwap_df, n=3):
    """
    Alpha7 ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    计算当日均价与收盘价的差值，然后再分别取3日内的最大值和最小值，并对这两个值进行升序排位，然后再将这两个排位相加，得到一个反映当日价格偏
    离程度的指标，然后再乘以成交量的3日变化率的升序排位，得到最终的因子值。这个因子反映了股票价格与成交量在短期内的波动性和相关性，如果这个
    因子值较大，说明股票价格在近期有较大的偏离均价，并且成交量有较大的增加，并且在所有股票中排名靠前，反之则说明股票价格在近期有较小的偏离
    均价，并且成交量有较大的减少，并且在所有股票中排名靠后。
    :param volume_df:
    :param close_df:
    :param vwap_df:
    :param n:
    :return:
    """
    return (rank(ta.MAX((vwap_df - close_df), n)) + rank(ta.MIN((vwap_df - close_df), n))) * rank(
        delta(volume_df, n))


def alpha191_8(high_df, low_df, vwap_df, n=4):
    """
    Alpha8 RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    通过将价格和成交量信息结合起来，计算出一种衡量股票相对强弱的值，并对其进行排名。
    :param high_df:
    :param low_df:
    :param vwap_df:
    :param n:
    :return:
    """
    return rank(delta((((high_df + low_df) / 2) * 0.2) + (vwap_df * 0.8), n)) * -1


def alpha191_9(high_df, low_df, volume_df):
    """
    Alpha9 SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    计算基于价格和成交量的移动平均，并提供一种衡量股票走势的指标。
    :param high_df:
    :param low_df:
    :param volume_df:
    :return:
    """
    return sma((((high_df + low_df) / 2 - (high_df.shift(1) + low_df.shift(1)) / 2) * (
            high_df - low_df) / zero_to_one(volume_df)), 7, 2)


def alpha191_10(close_df, ret_df=None):
    """
    Alpha10 (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
    :param close_df:
    :param ret_df:
    :return:
    """
    if ret_df is None:
        ret_df = close_df.pct_change()
    return rank(ta.MAX(((ret_df < 0) * ta.STDDEV(ret_df, 20) + (ret_df >= 0) * close_df) ** 2, 5))


def alpha191_11(close_df, high_df, low_df, volume_df, n=6):
    """
    Alpha11 SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: 6
    :return:
    """
    return ta.SUM(((close_df - low_df) - (high_df - close_df)) / zero_to_one(high_df - low_df) * volume_df, n)


def alpha191_12(open_df, close_df, vwap_df):
    """
    Alpha12 (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
    :param open_df:
    :param close_df:
    :param vwap_df:
    :return:
    """
    return (rank((open_df - (ta.SUM(vwap_df, 10) / 10)))) * (-1 * (rank(np.abs((close_df - vwap_df)))))


def alpha191_13(high_df, low_df, vwap_df):
    """
    Alpha13 (((HIGH * LOW)^0.5) - VWAP)
    :param high_df:
    :param low_df:
    :param vwap_df:
    :return:
    """
    return np.sqrt(high_df * low_df) - vwap_df


def alpha191_14(close_df, n=5):
    """
    Alpha14 CLOSE-DELAY(CLOSE,5)
    :param close_df:
    :param n: default 5
    :return:
    """
    return close_df - close_df.shift(n)


def alpha191_15(open_df, close_df):
    """
    Alpha15 OPEN/DELAY(CLOSE,1)-1
    :param open_df:
    :param close_df:
    :return:
    """
    return open_df / close_df.shift(1) - 1


def alpha191_16(volume_df, vwap_df, n=5):
    """
    Alpha16 (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    :param volume_df:
    :param vwap_df:
    :param n: default 5
    :return:
    """
    return -1 * ta.MAX(rank(ta.CORREL(rank(volume_df), rank(vwap_df), n)), n)


def alpha191_17(vwap_df, close_df, n=5):
    """
    Alpha17 RANK((VWAP - MAX(VWAP, 5)))^DELTA(CLOSE, 5)
    :param vwap_df:
    :param close_df:
    :param n: default 15
    :return:
    """
    return np.power(rank(vwap_df - ta.MAX(vwap_df, n)), delta(close_df, n))


def alpha191_18(close_df, n=5):
    """
    Alpha18 CLOSE/DELAY(CLOSE,5)
    :param close_df:
    :param n: default 5
    :return:
    """
    return close_df / close_df.shift(n)


def alpha191_19(close_df, n=5):
    """
    Alpha19
    (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
    计算基于收盘价的变化率，并根据条件进行选择。
    :param close_df:
    :param n: default 5
    :return:
    """
    return np.where(close_df < close_df.shift(n), (close_df - close_df.shift(n)) / close_df.shift(n),
                    np.where(close_df == close_df.shift(n), 0, (close_df - close_df.shift(n)) / close_df))


def alpha191_20(close_df, n=6):
    """
    Alpha20 (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    :param close_df:
    :param n: default 6
    :return:
    """
    return (close_df - close_df.shift(n)) / close_df.shift(n) * 100


def alpha191_21(close_df, n=6):
    """
    Alpha21 REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
    :param close_df:
    :param n: default 6
    :return:
    """
    return regbeta(ta.MA(close_df, n), np.arange(len(close_df)), n)


def alpha191_22(close_df, n=6):
    """
    Alpha22 SMA(((CLOSE-MEAN(CLOSE,6)) /MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    :param close_df:
    :param n: default 6
    :return:
    """
    mean_n = ta.MA(close_df, n)
    diff_mean_n = close_df - mean_n
    return sma((diff_mean_n / mean_n - delta(diff_mean_n / mean_n, 3)), 12, 1)


def alpha191_23(close_df, n=20):
    """
    Alpha23
    SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1
    )+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
    :param close_df:
    :param n: default 20
    :return:
    """
    # 计算CLOSE序列的1天滞后值
    # DELAY_CLOSE_1 = DELAY(CLOSE, 1)
    delay_close_1 = delay(close_df, 1)
    # 计算CLOSE序列的20天标准差值
    std_close_20 = ta.STDDEV(close_df, n)
    # 创建一个空数组，用于存储条件判断结果
    COND = np.zeros(len(close_df))
    # 对于每个位置i，根据条件判断赋值给COND数组
    for i in range(len(close_df)):
        # 如果CLOSE大于滞后值，则取标准差值
        if close_df[i] > delay_close_1[i] and std_close_20[i] > 0:
            COND[i] = std_close_20[i]
        # 否则，取0
        else:
            COND[i] = 0
    # 计算COND序列的20天中国SMA值，并乘以100
    SMA_COND_1 = sma(COND, n, 1) * 100
    # 计算COND序列的反向条件判断结果，并求其20天中国SMA值，并乘以100
    SMA_NOT_COND_1 = sma(1 - COND, n, 1) * 100
    # 计算最终结果，并返回之
    result = SMA_COND_1 / (SMA_COND_1 + SMA_NOT_COND_1)
    return result


def alpha191_24(close_df, n=5):
    """
    Alpha24 SMA(CLOSE-DELAY(CLOSE,5),5,1)
    :param close_df:
    :param n: default 5
    :return:
    """
    return sma(close_df - close_df.shift(n), n, 1)


def alpha191_25(close_df, ret_df, volume_df, n=250):
    """
    Alpha25
    ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 +
    RANK(SUM(RET, 250))))
    :param close_df:
    :param ret_df:
    :param volume_df:
    :param n: default 250
    :return:
    """
    return ((-1 * rank(delta(close_df, 7) * (1 - rank(decaylinear((volume_df / ta.MA(volume_df, 20)), 9))))) * (
            1 + rank(ta.SUM(ret_df, n))))


def alpha191_26(close_df, vwap_df, n=230):
    """
    Alpha26 ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
    :param close_df:
    :param vwap_df:
    :param n: default 230
    :return:
    """
    return (ta.SUM(close_df, 7) / 7 - close_df) + ta.CORREL(vwap_df, close_df.shift(5), n)


def alpha191_27(close_df, n=12):
    """
    Alpha27 WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    :param close_df:
    :param n: default 12
    :return:
    """
    return ta.WMA(
        (close_df - close_df.shift(3)) / close_df.shift(3) * 100 + (close_df - close_df.shift(6)) / close_df.shift(
            6) * 100, n)


def alpha191_28(close_df, high_df, low_df, n=9):
    """
    Alpha28
    3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(
    MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 9
    :return:
    """
    return 3 * sma((close_df - ta.MIN(low_df, n)) / (ta.MAX(high_df, n) - ta.MIN(low_df, n)) * 100, 3,
                   1) - 2 * sma(
        sma((close_df - ta.MIN(low_df, n)) / (ta.MAX(high_df, n) - ta.MIN(low_df, n)) * 100, 3, 1), 3, 1)


def alpha191_29(close_df, volume_df, n=6):
    """
    Alpha29 (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    :param close_df:
    :param volume_df:
    :param n: default 6
    :return:
    """
    return (close_df - close_df.shift(n)) / close_df.shift(n) * volume_df


def alpha191_30(close_df, mkt_df, smb_df, hml_df, n=60, m=20):
    """
    Alpha30 WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML，60))^2,20)
    :param close_df:
    :param mkt_df:
    :param smb_df:
    :param hml_df:
    :param n: default 60
    :param m: default 20
    :return:
    """
    return ta.WMA((regresi(close_df / close_df.shift(1) - 1, mkt_df, smb_df, hml_df, n)) ** 2, m)


def alpha191_31(close_df, n=12):
    """
    Alpha31 (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    :param close_df:
    :param n: default 12
    :return:
    """
    return (close_df - ta.MA(close_df, n)) / ta.MA(close_df, n) * 100


def alpha191_32(high_df, volume_df, n=3):
    """
    Alpha32 (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    :param high_df:
    :param volume_df:
    :param n: default 3
    :return:
    """
    return -1 * ta.SUM(rank(ta.CORREL(rank(high_df), rank(volume_df), 3)), 3)


def alpha191_33(close_df, low_df, volume_df, ret_df=None, n=5, m=240):
    """
    Alpha33
    ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *
    TSRANK(VOLUME, 5))
    :param close_df:
    :param low_df:
    :param ret_df:
    :param volume_df:
    :param n: default 5
    :param m: default 240
    :return:
    """
    if ret_df is None:
        ret_df = close_df.pct_change(1)
    return (((-1 * ta.MIN(low_df, n)) + delay(ta.MIN(low_df, n), n)) *
            rank(((ta.SUM(ret_df, m) - ta.SUM(ret_df, 20)) / 220))) * tsrank(volume_df, n)


def alpha191_34(close_df, n=12):
    """
    Alpha34 MEAN(CLOSE,12)/CLOSE
    :param close_df:
    :param n: default 12
    :return:
    """
    return ta.MA(close_df, n) / close_df


def alpha191_35(open_df, volume_df, n=15, m=17):
    """
    Alpha35
    (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +
    (OPEN *0.35)), 17),7))) * -1)
    :param open_df:
    :param volume_df:
    :param n: default 15
    :param m: default 17
    :return:
    """
    return np.minimum(rank(decaylinear(delta(open_df, 1), n)),
                      rank(decaylinear(ta.CORREL(volume_df, ((open_df * 0.65) + (open_df * 0.35)), m), 7))) * -1


def alpha191_36(volume_df, vwap_df, n=6):
    """
    Alpha36 RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)
    :param volume_df:
    :param vwap_df:
    :param n: default 6
    :return:
    """
    return tsrank(ta.SUM(ta.CORREL(rank(volume_df), rank(vwap_df)), n), 2)


def alpha191_37(open_df, ret_df=None, n=5, m=10):
    """
    Alpha37 (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    :param open_df:
    :param ret_df:
    :param n: default 5
    :param m: default 10
    :return:
    """
    if ret_df is None:
        ret_df = open_df.pct_change(1)
    return -1 * rank(((ta.SUM(open_df, n) * ta.SUM(ret_df, n)) - delay((ta.SUM(open_df, n) * ta.SUM(ret_df, n)), m)))


def alpha191_38(high_df, n=20):
    """
    Alpha38 (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
    :param high_df:
    :param n: default 20
    :return:
    """
    return np.where(((ta.SUM(high_df, n) / n) < high_df), (-1 * delta(high_df, 2)), 0)


def alpha191_39(close_df, open_df, volume_df, vwap_df, n=8, m=14, l=12):
    """
    Alpha39
    ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),
    SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
    :param close_df:
    :param open_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 8
    :param m: default 14
    :param l: default 12
    :return:
    """
    return ((rank(decaylinear(delta(close_df, 2), n)) -
             rank(decaylinear(ta.CORREL(((vwap_df * 0.3) + (open_df * 0.7)),
                                        ta.SUM(ta.MA(volume_df, 180), 37), m), l))) * -1)


def alpha191_40(close_df, volume_df, n=26):
    """
    Alpha40 SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
    :param close_df:
    :param volume_df:
    :param n: default 26
    :return:
    """
    return ta.SUM(np.where(close_df > delay(close_df, 1), volume_df, 0), n) / \
        ta.SUM(np.where(close_df <= delay(close_df, 1), volume_df, 0), n) * 100


def alpha191_41(vwap_df, n=5):
    """
    Alpha41 (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
    :param vwap_df:
    :param n: default 5
    :return:
    """
    return rank(ta.MAX(delta(vwap_df, 3), n)) * -1


def alpha191_42(high_df, volume_df, n=10):
    """
    Alpha42 ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
    :param high_df:
    :param volume_df:
    :param n: default 10
    :return:
    """
    return (-1 * rank(ta.STDDEV(high_df, n))) * ta.CORREL(high_df, volume_df, n)


def alpha191_43(close_df, volume_df, n=6):
    """
    Alpha43 SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
    :param close_df:
    :param volume_df:
    :param n: default 6
    :return:
    """
    return ta.SUM(np.where(close_df > delay(close_df, 1), volume_df,
                           np.where(close_df < delay(close_df, 1), -volume_df, 0)), n)


def alpha191_44(low_df, volume_df, vwap_df, n=10, m=7):
    """
    Alpha44
    (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) +
    TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
    :param low_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 10
    :param m: default 7
    :return:
    """
    return tsrank(decaylinear(ta.CORREL(low_df, ta.MA(volume_df, 10), m), 6), 4) + \
        tsrank(decaylinear(delta(vwap_df, 3), n), 15)


def alpha191_45(close_df, open_df, volume_df, vwap_df, n=15):
    """
    Alpha45 (RANK(DELTA(((((CLOSE * 0.6) + (OPEN * 0.4)))), 1)) *
    RANK(CORR(VWAP, MEAN(VOLUME, 150), 15)))
    :param close_df:
    :param open_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 15
    :return:
    """
    return rank(delta((close_df * 0.6 + open_df * 0.4), 1)) * \
        rank(ta.CORREL(vwap_df, ta.MA(volume_df, 150), n))


def alpha191_46(close_df, volume_df, n=3):
    """
    Alpha46 (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    :param close_df:
    :param volume_df:
    :param n: default 12
    :return:
    """
    return -1 * ta.SUM(rank(ta.CORREL(rank(close_df), rank(volume_df), 3)), n)


def alpha191_47(close_df, high_df, low_df, n=6, m=9):
    """
    Alpha47 SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 6
    :param m: default 9
    :return:
    """
    return sma((ta.MAX(high_df, n) - close_df) / (ta.MAX(high_df, n) - ta.MIN(low_df, n)) * 100, m, 1)


def alpha191_48(close_df, volume_df, n=5, m=20):
    """
    Alpha48
    (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) +
    SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
    :param close_df:
    :param volume_df:
    :param n: default 5
    :param m: default 20
    :return:
    """
    return -1 * ((rank(((np.sign(close_df - close_df.shift(1)) + np.sign(close_df.shift(1) - close_df.shift(2))) +
                        np.sign(close_df.shift(2) - close_df.shift(3))))) * ta.SUM(volume_df, n)) / ta.SUM(volume_df, m)


def alpha191_49(high_df, low_df, n=12):
    """
    Alpha49
    SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L
    OW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L
    OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI
    GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    :param high_df:
    :param low_df:
    :param n: default 12
    :return:
    """
    delay_high = np.roll(high_df, 1)
    delay_low = np.roll(low_df, 1)
    cond1 = (high_df + low_df) >= (delay_high + delay_low)
    cond2 = (high_df + low_df) <= (delay_high + delay_low)

    max1 = np.maximum(np.abs(high_df - delay_high), np.abs(low_df - delay_low))

    numerator = ta.SUM(np.where(cond1, 0, max1), n)
    denominator = numerator + ta.SUM(np.where(cond2, 0, max1), n)
    result = numerator / denominator
    return result


def alpha191_50(high_df, low_df, n=12):
    """
    Alpha50
    SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L
    OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L
    OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI
    GH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HI
    GH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:
    MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA
    Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    :param high_df:
    :param low_df:
    :param n: default 12
    :return:
    """
    delay_high = np.roll(high_df, 1)
    delay_low = np.roll(low_df, 1)
    cond1 = (high_df + low_df) <= (delay_high + delay_low)
    cond2 = (high_df + low_df) >= (delay_high + delay_low)

    max1 = np.maximum(np.abs(high_df - delay_high), np.abs(low_df - delay_low))

    numerator = ta.SUM(np.where(cond1, 0, max1), n)
    denominator = numerator + ta.SUM(np.where(cond2, 0, max1), n)
    result = (numerator - ta.SUM(np.where(cond2, 0, max1), n)) / denominator
    return result


def alpha191_51(high_df, low_df, n=12):
    """
    Alpha51
    SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L
    OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L
    OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI
    GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    :param high_df:
    :param low_df:
    :param n: default 12
    :return:
    """
    delay_high = np.roll(high_df, 1)
    delay_low = np.roll(low_df, 1)
    cond1 = (high_df + low_df) >= (delay_high + delay_low)
    cond2 = (high_df + low_df) <= (delay_high + delay_low)

    max1 = np.maximum(np.abs(high_df - delay_high), np.abs(low_df - delay_low))

    numerator = ta.SUM(np.where(cond1, 0, max1), n)
    denominator = numerator + ta.SUM(np.where(cond2, 0, max1), n)
    result = numerator / denominator
    return result


def alpha191_52(close_df, high_df, low_df, n=26):
    """
    Alpha52
    SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*
    100
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 26
    :return:
    """
    return ta.SUM(np.maximum(0, high_df - delay((high_df + low_df + close_df) / 3, 1)), n) / \
        ta.SUM(np.maximum(0, delay((high_df + low_df + close_df) / 3, 1) - low_df), n) * 100


def alpha191_53(df, change_flag=True, n=12):
    """
    Alpha53 COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    :param df: close_df or change
    :param change_flag: is change?
    :param n: default 12
    :return:
    """
    change = df.copy() if change_flag else df.pct_change(1)
    return count(change > 0, n) / n * 100


def alpha191_54(close_df, open_df, n=10):
    """
    Alpha54 (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
    :param close_df:
    :param open_df:
    :param n: default 10
    :return:
    """
    return -1 * rank(ta.STDDEV(np.abs(close_df - open_df)) + (close_df - open_df) + ta.CORREL(close_df, open_df, n))


def alpha191_55(close_df, open_df, high_df, low_df, n=20):
    """
    Alpha55
    SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CL
    OSE,1))>ABS(LOW-DELAY(CLOSE,1)) &
    ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS
    E,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) &
    ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLO
    SE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OP
    EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
    :param close_df:
    :param open_df:
    :param high_df:
    :param low_df:
    :param n: default 20
    :return:
    """
    # 计算各种价格变化和绝对值
    close_change = close_df - close_df.shift(1)
    high_change = high_df - high_df.shift(1)
    low_change = low_df - low_df.shift(1)
    abs_close_change = np.abs(close_change)
    abs_high_change = np.abs(high_change)
    abs_low_change = np.abs(low_change)

    # 计算分母中的条件表达式
    cond1 = (abs_high_change > abs_low_change) & (abs_high_change > abs_high_change.shift(1))
    cond2 = (abs_low_change > abs_high_change.shift(1)) & (abs_low_change > abs_high_change)

    # 计算分母
    denom = np.where(cond1, abs_high_change + abs_low_change / 2 + abs_close_change / 4,
                     np.where(cond2, abs_low_change + abs_high_change / 2 + abs_close_change / 4,
                              abs_high_change.shift(1) + abs_close_change / 4))

    # 计算分子
    numer = 16 * (close_change + (close_df - open_df) / 2 + close_df.shift(1) - open_df.shift(1))

    # 计算因子值
    factor = ta.SUM(numer / denom * np.maximum(abs_high_change, abs_low_change), n)

    return factor


def alpha191_56(open_df, high_df, low_df, volume_df, n1=12, n2=19, n3=40, n4=13):
    """
    Alpha56
    (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),
    SUM(MEAN(VOLUME,40), 19), 13))^5)))
    :param open_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n1: default 12
    :param n2: default 19
    :param n3: default 40
    :param n4: default 13
    :return:
    """
    rank1 = rank(open_df - ta.MIN(open_df, n1))
    rank2 = rank(ta.CORREL(ta.SUM((high_df + low_df) / 2, n2), ta.SUM(ta.MA(volume_df, n3), n2), n4) ** 5)
    return np.where(rank1 < rank2, 1, 0)


def alpha191_57(close_df, high_df, low_df, n=9):
    """
    Alpha57 SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 9
    :return:
    """
    return sma((close_df - ta.MIN(low_df, n)) / (ta.MAX(high_df, n) - ta.MIN(low_df, n)) * 100, 3, 1)


def alpha191_58(df, change_flag=True, n=20):
    """
    Alpha58 COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    :param df: close_df or change
    :param change_flag: is change?
    :param n: default 20
    :return:
    """
    return alpha191_53(df, change_flag, n)


def alpha191_59(close_df, high_df, low_df, n=20):
    """
    Alpha59
    SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,D
    ELAY(CLOSE,1)))),20)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 20
    :return:
    """
    delay_close = np.roll(close_df, 1)
    cond1 = close_df == delay_close
    cond2 = close_df > delay_close
    result = np.where(cond1, 0, np.where(cond2, close_df - np.minimum(low_df, delay_close),
                                         close_df - np.maximum(high_df, delay_close)))
    return ta.SUM(result, n)


def alpha191_60(close_df, high_df, low_df, volume_df, n=20):
    """
    Alpha60 SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 20
    :return:
    """
    return ta.SUM(((close_df - low_df) - (high_df - close_df)) / zero_to_one(high_df - low_df) * volume_df, n)


def alpha191_61(vwap_df, low_df, volume_df, n1=80, n2=8, n3=12, n4=17):
    """
    Alpha61
    (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
    RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)
    :param vwap_df:
    :param low_df:
    :param volume_df:
    :param n1: default 80
    :param n2: default 8
    :param n3: default 12
    :param n4: default 17
    :return:
    """
    rank1 = rank(decaylinear(delta(vwap_df, 1), n3))
    rank2 = rank(decaylinear(rank(ta.CORREL(low_df, ta.MA(volume_df, n1), n2)), n4))
    return np.maximum(rank1, rank2) * -1


def alpha191_62(high_df, volume_df, n=5):
    """
    Alpha62 (-1 * CORR(HIGH, RANK(VOLUME), 5))
    :param high_df:
    :param volume_df:
    :param n: default 5
    :return:
    """
    return -1 * ta.CORREL(high_df, rank(volume_df), n)


def alpha191_63(close_df, n=6):
    """
    Alpha63 SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    :param close_df:
    :param n: default 6
    :return:
    """
    return sma(np.maximum(close_df - np.roll(close_df, 1), 0), n, 1) / sma(
        np.abs(close_df - np.roll(close_df, 1)), n, 1) * 100


def alpha191_64(close_df, volume_df, n1=60, n2=4, n3=13, n4=14):
    """
    Alpha64
    (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
    RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)
    :param close_df:
    :param volume_df:
    :param n1: default 60
    :param n2: default 4
    :param n3: default 13
    :param n4: default 14
    :return:
    """
    rank1 = rank(decaylinear(ta.CORREL(rank(close_df), rank(ta.MA(volume_df, n1)), n2), n3))
    rank2 = rank(decaylinear(np.maximum(ta.CORREL(rank(close_df), rank(volume_df), n2), n3), n4))
    return np.maximum(rank1, rank2) * -1


def alpha191_65(close_df, n=6):
    """
    Alpha65 MEAN(CLOSE,6)/CLOSE
    :param close_df:
    :param n: default 6
    :return:
    """
    return ta.MA(close_df, n) / close_df


def alpha191_66(close_df, n=6):
    """
    Alpha66 (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    :param close_df:
    :param n: default 6
    :return:
    """
    return (close_df - ta.MA(close_df, n)) / ta.MA(close_df, n) * 100


def alpha191_67(close_df, n=24):
    """
    Alpha67 SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    :param close_df:
    :param n: default 24
    :return:
    """
    return sma(np.maximum(close_df - np.roll(close_df, 1), 0), n, 1) / sma(
        np.abs(close_df - np.roll(close_df, 1)), n, 1) * 100


def alpha191_68(high_df, low_df, volume_df, n=15):
    """
    Alpha68 SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 15
    :return:
    """
    return sma(((high_df + low_df) / 2 - (np.roll(high_df, 1) + np.roll(low_df, 1)) / 2) *
            high_df - low_df / zero_to_one(volume_df), n, 2)


def alpha191_69(close_df, open_df, n=20):
    """
    Alpha69
    (SUM(DTM,20)>SUM(DBM,20)？(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)：(SUM(DTM,20)=SUM(DBM,20)？
    0：(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    :param close_df:
    :param open_df:
    :param n: default 20
    :return:
    """
    dtm_df = dtm(close_df, open_df)
    dbm_df = dbm(close_df, open_df)
    dtm_sum = ta.SUM(dtm_df, n)
    dbm_sum = ta.SUM(dbm_df, n)
    return np.where(dtm_sum > dbm_sum, (dtm_sum - dbm_sum) / dtm_sum,
                    np.where(dtm_sum == dbm_sum, 0, (dtm_sum - dbm_sum) / dbm_sum))


def alpha191_70(amount_df, n=6):
    """
    Alpha70 STD(AMOUNT,6)
    :param amount_df:
    :param n: default 6
    :return:
    """
    return ta.STDDEV(amount_df, n)


def alpha191_71(close_df, n=24):
    """
    Alpha71 (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    :param close_df:
    :param n: default 24
    :return:
    """
    return (close_df - ta.MA(close_df, n)) / ta.MA(close_df, n) * 100


def alpha191_72(close_df, high_df, low_df, n=15):
    """
    Alpha72 SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 15
    :return:
    """
    return sma((ta.MAX(high_df, 6) - close_df) / (ta.MAX(high_df, 6) - ta.MIN(low_df, 6)) * 100, n, 1)


def alpha191_73(close_df, volume_df, vwap_df, n1=10, n2=16, n3=4, n4=5, n5=3):
    """
    Alpha73
    ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -
    RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
    :param close_df:
    :param volume_df:
    :param vwap_df:
    :param n1: default 10
    :param n2: default 16
    :param n3: default 4
    :param n4: default 5
    :param n5: default 3
    :return:
    """
    return ((tsrank(decaylinear(decaylinear(ta.CORREL(close_df, volume_df, n1), n2), n3), n4) -
             rank(decaylinear(ta.CORREL(vwap_df, ta.MA(volume_df, 30), n3), n5))) * -1)


def alpha191_74(low_df, volume_df, vwap_df, n1=20, n2=40, n3=7, n4=6):
    """
    Alpha74
    (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) +
    RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
    :param low_df:
    :param volume_df:
    :param vwap_df:
    :param n1: default 20
    :param n2: default 40
    :param n3: default 7
    :param n4: default 6
    :return:
    """
    return (rank(ta.CORREL(ta.SUM(((low_df * 0.35) + (vwap_df * 0.65)), n1), ta.SUM(ta.MA(volume_df, n2), n1), n3)) +
            rank(ta.CORREL(rank(vwap_df), rank(volume_df), n4)))


def alpha191_75(close_df, open_df, benchmark_close_df, benchmark_open_df, n=50):
    """
    Alpha75
    COUNT(CLOSE>OPEN &
    BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKIN
    DEXOPEN,50)
    :param close_df:
    :param open_df:
    :param benchmark_close_df:
    :param benchmark_open_df:
    :param n: default 50
    :return:
    """
    cond1 = np.logical_and(close_df > open_df, benchmark_close_df < benchmark_open_df)
    cond2 = np.where(benchmark_close_df < benchmark_open_df, True, False)
    return count(cond1, n) / count(cond2, n)


def alpha191_76(close_df, volume_df, n1=20, n2=20):
    """
    Alpha76 STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    :param close_df:
    :param volume_df:
    :param n1: default 20
    :param n2: default 20
    :return:
    """
    return (ta.STDDEV(np.abs((close_df / delay(close_df, 1) - 1)) / zero_to_one(volume_df), n1) /
            ta.MA(np.abs((close_df / delay(close_df, 1) - 1)) / zero_to_one(volume_df), n2))


def alpha191_77(high_df, low_df, volume_df, vwap_df, n1=20, n2=3, n3=6):
    """
    Alpha77
    MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),
    RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
    :param high_df:
    :param low_df:
    :param volume_df:
    :param vwap_df:
    :param n1: default 20
    :param n2: default 3
    :param n3: default 6
    :return:
    """
    return np.minimum(rank(decaylinear((((high_df + low_df) / 2) + high_df) - (vwap_df + high_df), n1)),
                      rank(decaylinear(ta.CORREL(((high_df + low_df) / 2), ta.MA(volume_df, 40), n2), n3)))


def alpha191_78(close_df, high_df, low_df, n=12):
    """
    Alpha78
    ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOS
    E)/3,12)),12))
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 12
    :return:
    """
    return ((high_df + low_df + close_df) / 3 - ta.MA((high_df + low_df + close_df) / 3, n)) / (
            0.015 * ta.MA(np.abs(close_df - ta.MA((high_df + low_df + close_df) / 3, n)), n))


def alpha191_79(close_df, n=12):
    """
    Alpha79 SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    :param close_df:
    :param n: default 12
    :return:
    """
    return sma(np.maximum(close_df - delay(close_df, 1), 0), n, 1) / sma(np.abs(close_df - delay(close_df, 1)),
                                                                         n, 1) * 100


def alpha191_80(volume_df, n=5):
    """
    Alpha80 (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    :param volume_df:
    :param n: default 5
    :return:
    """
    return (volume_df - delay(volume_df, n)) / delay(volume_df, n) * 100


def alpha191_81(volume_df, n=21):
    """
    Alpha81 SMA(VOLUME,21,2)
    :param volume_df:
    :param n: default 21
    :return:
    """
    return sma(volume_df, n, 2)


def alpha191_82(close_df, high_df, low_df, n1=6, n2=20):
    """
    Alpha82 SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n1: default 6
    :param n2: default 20
    :return:
    """
    return sma((ta.MAX(high_df, n1) - close_df) / (ta.MAX(high_df, n1) - ta.MIN(low_df, n1)) * 100, n2, 1)


def alpha191_83(high_df, volume_df, n=5):
    """
    Alpha83 (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
    :param high_df:
    :param volume_df:
    :param n: default 5
    :return:
    """
    return -1 * rank(coviance(rank(high_df), rank(volume_df), n))


def alpha191_84(close_df, volume_df, n=20):
    """
    Alpha84 SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    :param close_df:
    :param volume_df:
    :param n: default 20
    :return:
    """
    return ta.SUM(np.where(close_df > delay(close_df, 1), volume_df, np.where(close_df < delay(close_df, 1),
                                                                              -volume_df, 0)), n)


def alpha191_85(close_df, volume_df, n1=20, n2=8):
    """
    Alpha85 (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
    :param close_df:
    :param volume_df:
    :param n1: default 20
    :param n2: default 8
    :return:
    """
    return tsrank(volume_df / ta.MA(volume_df, n1), n1) * tsrank(-1 * delta(close_df, 7), n2)


def alpha191_86(close_df, n1=10, n2=20):
    """
    Alpha86 ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :
    (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) *
    (CLOSE - DELAY(CLOSE, 1)))))
    :param close_df:
    :param n1: default 10
    :param n2: default 20
    :return:
    """
    return np.where((0.25 < ((delay(close_df, n2) - delay(close_df, n1)) / n1 - (
            delay(close_df, n1) - close_df) / n1)), -1, np.where(
        (((delay(close_df, n2) - delay(close_df, n1)) / n1 - (delay(close_df, n1) - close_df) / n1) < 0), 1,
        -1 * (close_df - delay(close_df, 1))))


def alpha191_87(open_df, high_df, low_df, vwap_df, n1=4, n2=7, n3=11):
    """
    Alpha87 ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /
    (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
    :param open_df:
    :param high_df:
    :param low_df:
    :param vwap_df:
    :param n1: default 4
    :param n2: default 7
    :param n3: default 11
    :return:
    """
    return (rank(decaylinear(delta(vwap_df, n1), n2)) + tsrank(
        decaylinear((low_df * 0.9 + low_df * 0.1 - vwap_df) / (
                open_df - (high_df + low_df) / 2), n3), n2)) * -1


def alpha191_88(close_df, n=20):
    """
    Alpha88 (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    :param close_df:
    :param n: default 20
    :return:
    """
    return (close_df - delay(close_df, n)) / delay(close_df, n) * 100


def alpha191_89(close_df, n1=13, n2=27, n3=10):
    """
    Alpha89 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    :param close_df:
    :param n1: default 13
    :param n2: default 27
    :param n3: default 10
    :return:
    """
    return 2 * (sma(close_df, n1, 2) - sma(close_df, n2, 2) - sma(
        sma(close_df, n1, 2) - sma(close_df, n2, 2), n3, 2))


def alpha191_90(vwap_df, volume_df, n=5):
    """
    Alpha90 ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
    :param vwap_df:
    :param volume_df:
    :param n: default 5
    :return:
    """
    return rank(ta.CORREL(rank(vwap_df), rank(volume_df), n)) * -1


def alpha191_91(close_df, volume_df, low_df, n1=5, n2=40):
    """
    Alpha91 ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
    :param close_df:
    :param volume_df:
    :param low_df:
    :param n1: default 5
    :param n2: default 40
    :return:
    """
    return (rank((close_df - ta.MAX(close_df, n1))) * rank(ta.CORREL(ta.MA(volume_df, n2), low_df, n1))) * -1


def alpha191_92(close_df, vwap_df, volume_df, n1=2, n2=3, n3=13, n4=5, n5=15):
    """
    Alpha92 (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),
    TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)
    :param close_df:
    :param vwap_df:
    :param volume_df:
    :param n1: default 2
    :param n2: default 3
    :param n3: default 3
    :param n4: default 5
    :param n5: default 15
    :return:
    """
    return (np.maximum(rank(decaylinear(delta(close_df * 0.35 + vwap_df * 0.65, n1), n2)),
                       tsrank(decaylinear(np.abs(ta.CORREL(ta.MA(volume_df, 180), close_df, n3)), n4), n5))) * -1


def alpha191_93(open_df, low_df, n=20):
    """
    Alpha93 SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    :param open_df:
    :param low_df:
    :param n: default 20
    :return:
    """
    return ta.SUM(np.where(open_df >= delay(open_df, 1), 0, np.maximum(open_df - low_df, open_df - delay(open_df, 1))),
                  n)


def alpha191_94(close_df, volume_df, n=30):
    """
    Alpha94 SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    :param close_df:
    :param volume_df:
    :param n: default 30
    :return:
    """
    return ta.SUM(np.where(close_df > delay(close_df, 1), volume_df,
                           np.where(close_df < delay(close_df, 1), -volume_df, 0)), n)


def alpha191_95(amount_df, n=20):
    """
    Alpha95 STD(AMOUNT,20)
    :param amount_df:
    :param n: default 20
    :return:
    """
    return ta.STDDEV(amount_df, n)


def alpha191_96(close_df, high_df, low_df, n1=9, n2=3):
    """
    Alpha96 SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n1: default 9
    :param n2: default 3
    :return:
    """
    return sma(sma((close_df - ta.MIN(low_df, n1)) / (ta.MAX(high_df, n1) - ta.MIN(low_df, n1)) * 100, n2),
               n2)


def alpha191_97(volume_df, n=10):
    """
    Alpha97 STD(VOLUME,10)
    :param volume_df:
    :param n: default 10
    :return:
    """
    return ta.STDDEV(volume_df, n)


def alpha191_98(close_df, n1=100, n2=100, n3=3):
    """
    Alpha98
    ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /
    DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
    :param close_df:
    :param n1: default 100
    :param n2: default 100
    :param n3: default 3
    :return:
    """
    return np.where(((delta(ta.SUM(close_df, n1) / n1, n1) / delay(close_df, n1)) < 0.05) |
                    ((delta(ta.SUM(close_df, n1) / n1, n1) / delay(close_df, n1)) == 0.05),
                    (-1 * (close_df - ta.MIN(close_df, n2))), (-1 * delta(close_df, n3)))


def alpha191_99(close_df, volume_df, n=5):
    """
    Alpha99 (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    :param close_df:
    :param volume_df:
    :param n: default 5
    :return:
    """
    return -1 * rank(coviance(rank(close_df), rank(volume_df), n))


def alpha191_100(volume_df, n=20):
    """
    Alpha100 STD(VOLUME,20)
    :param volume_df:
    :param n: default 20
    :return:
    """
    return ta.STDDEV(volume_df, n)


def alpha191_101(close_df, volume_df, high_df, vwap_df, n1=30, n2=37, n3=15, n4=11):
    """
    Alpha101
    ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
    RANK(VOLUME), 11))) * -1)
    :param close_df:
    :param volume_df:
    :param high_df:
    :param vwap_df:
    :param n1: default 30
    :param n2: default 37
    :param n3: default 15
    :param n4: default 11
    :return:
    """
    return np.where(rank(ta.CORREL(close_df, ta.SUM(ta.MA(volume_df, n1), n2), n3)) <
                    rank(ta.CORREL(rank(high_df * 0.1 + vwap_df * 0.9), rank(volume_df), n4)), -1, 1)


def alpha191_102(volume_df, n=6):
    """
    Alpha102 SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    :param volume_df:
    :param n: default 6
    :return:
    """
    return sma(np.where(volume_df - delay(volume_df, 1) > 0, volume_df - delay(volume_df, 1), 0), n, 1) / \
        sma(np.abs(volume_df - delay(volume_df, 1)), n) * 100


def alpha191_103(low_df, n=20):
    """
    Alpha103 ((20-LOWDAY(LOW,20))/20)*100
    :param low_df:
    :param n: default 20
    :return:
    """
    return (n - ta.MININDEX(low_df, n)) / n * 100


def alpha191_104(close_df, high_df, volume_df, n=5, m=20):
    """
    Alpha104 (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
    :param close_df:
    :param high_df:
    :param volume_df:
    :param n: default 5
    :param m: default 20
    :return:
    """
    return -1 * (delta(ta.CORREL(high_df, volume_df, n), n) * rank(ta.STDDEV(close_df, m)))


def alpha191_105(open_df, volume_df, n=10):
    """
    Alpha105 (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
    :param open_df:
    :param volume_df:
    :param n: default 10
    :return:
    """
    return -1 * ta.CORREL(rank(open_df), rank(volume_df), n)


def alpha191_106(close_df, n=20):
    """
    Alpha106 CLOSE-DELAY(CLOSE,20)
    :param close_df:
    :param n: default 20
    :return:
    """
    return close_df - delay(close_df, n)


def alpha191_107(open_df, high_df, close_df, low_df):
    """
    Alpha107 (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
    :param open_df:
    :param high_df:
    :param close_df:
    :param low_df:
    :return:
    """
    return -1 * rank(open_df - delay(high_df, 1)) * rank(open_df - delay(close_df, 1)) * rank(
        open_df - delay(low_df, 1))


def alpha191_108(high_df, vwap_df, volume_df, n=6, m=120):
    """
    Alpha108 ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)
    :param high_df:
    :param vwap_df:
    :param volume_df:
    :param n: default 6
    :param m: default 120
    :return:
    """
    return np.power(rank(high_df - ta.MIN(high_df, 2)), rank(ta.CORREL(vwap_df, ta.MA(volume_df, m), n))) * -1


def alpha191_109(high_df, low_df, n=10, m=2):
    """
    Alpha109 SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    :param high_df:
    :param low_df:
    :param n: default 10
    :param m: default 10
    :return:
    """
    return sma(high_df - low_df, n, m) / sma(sma(high_df - low_df, n, m), n, m)


def alpha191_110(close_df, high_df, low_df, n=20):
    """
    Alpha110 SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 20
    :return:
    """
    return ta.SUM(np.maximum(0, high_df - delay(close_df, 1)), n) / ta.SUM(np.maximum(0, delay(close_df, 1) - low_df),
                                                                           n) * 100


def alpha191_111(close_df, low_df, high_df, volume_df, n=11, m=2):
    """
    Alpha111 SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-L
    OW),4,2)
    :param close_df:
    :param low_df:
    :param high_df:
    :param volume_df:
    :param n: default 11
    :param m: default 2
    :return:
    """
    return sma(volume_df * ((close_df - low_df) - (high_df - close_df)) / zero_to_one(high_df - low_df), n, m) - sma(
        volume_df * ((close_df - low_df) - (high_df - close_df)) / zero_to_one(high_df - low_df), 4, m)


def alpha191_112(close_df, n=12):
    """
    Alpha112
    (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOS
    E-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DE
    LAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    :param close_df:
    :param n: default 12
    :return:
    """
    return (ta.SUM(np.where(close_df - delay(close_df, 1) > 0, close_df - delay(close_df, 1), 0), n) - ta.SUM(
        np.where(close_df - delay(close_df, 1) < 0, np.abs(close_df - delay(close_df, 1)), 0), n)) / (
            ta.SUM(np.where(close_df - delay(close_df, 1) > 0, close_df - delay(close_df, 1), 0),
                   n) + ta.SUM(np.where(close_df - delay(close_df, 1) < 0,
                                        np.abs(close_df - delay(close_df, 1)), 0), n)) * 100


def alpha191_113(close_df, volume_df, n=5, m=20):
    """
    Alpha113 (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),
    SUM(CLOSE, 20), 2))))
    :param close_df:
    :param volume_df:
    :param n: default 5
    :param m: default 20
    :return:
    """
    return -1 * ((rank(ta.SUM(delay(close_df, n), m) / m) * ta.CORREL(close_df, volume_df, 2)) * rank(
        ta.CORREL(ta.SUM(close_df, n), ta.SUM(close_df, m), 2)))


def alpha191_114(close_df, high_df, low_df, volume_df, vwap_df, n=5):
    """
    Alpha114
    ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /
    (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 5
    :return:
    """
    return (rank(delay((high_df - low_df) / (ta.SUM(close_df, n) / n), 2)) * rank(
        rank(volume_df))) / (((high_df - low_df) / (ta.SUM(close_df, n) / n)) / (vwap_df - close_df))


def alpha191_115(close_df, high_df, low_df, volume_df, n=30, m=10):
    """
    Alpha115
    (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /
    2), 4), TSRANK(VOLUME, 10), 7)))
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 30
    :param m: default 10
    :return:
    """
    return (rank(ta.CORREL(((high_df * 0.9) + (close_df * 0.1)), ta.MA(volume_df, n), m)) ** rank(
        ta.CORREL(tsrank(((high_df + low_df) / 2), 4), tsrank(volume_df, 10), 7)))


def alpha191_116(close_df, n=20):
    """
    Alpha116 REGBETA(CLOSE,SEQUENCE,20)
    :param close_df:
    :param n: default 20
    :return:
    """
    return alpha191_21(close_df, n)


def alpha191_117(close_df, high_df, low_df, volume_df, ret_df, n=32, m=16):
    """
    Alpha117 ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param ret_df:
    :param n: default 32
    :param m: default 16
    :return:
    """
    return (tsrank(volume_df, n) * (1 - tsrank(((close_df + high_df) - low_df), m))) * (
            1 - tsrank(ret_df, n))


def alpha191_118(open_df, high_df, low_df, n=20):
    """
    Alpha118 SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    :param open_df:
    :param high_df:
    :param low_df:
    :param n: default 20
    :return:
    """
    return ta.SUM(high_df - open_df, n) / ta.SUM(open_df - low_df, n) * 100


def alpha191_119(open_df, vwap_df, volume_df, n=5, m=26):
    """
    Alpha119
    (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) -
    RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
    :param open_df:
    :param vwap_df:
    :param volume_df:
    :param n: default 5
    :param m: default 26
    :return:
    """
    return (rank(decaylinear(ta.CORREL(vwap_df, ta.SUM(ta.MA(volume_df, n), m), 5), 7)) -
            rank(decaylinear(tsrank(np.minimum(ta.CORREL(rank(open_df), rank(ta.MA(volume_df, 15)), 21), 9), 7), 8)))


def alpha191_120(close_df, vwap_df):
    """
    Alpha120 (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
    :param close_df:
    :param vwap_df:
    :return:
    """
    return rank((vwap_df - close_df)) / rank((vwap_df + close_df))


def alpha191_121(volume_df, vwap_df, n=20, m=60):
    """
    Alpha121
    ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)
    :param volume_df:
    :param vwap_df:
    :param n: default 20
    :param m: default 60
    :return:
    """
    return ((rank((vwap_df - np.minimum(vwap_df, 12))) ** tsrank(
        ta.CORREL(tsrank(vwap_df, n), tsrank(ta.MA(volume_df, m), 2), 18), 3)) * -1)


def alpha191_122(close_df, n=13):
    """
    Alpha122
    (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    :param close_df:
    :param n: default 13
    :return:
    """
    return (sma(sma(sma(np.log(close_df), n), n), n) - delay(
        sma(sma(sma(np.log(close_df), n), n), n), 1)) / delay(
        sma(sma(sma(np.log(close_df), n), n), n), 1)


def alpha191_123(high_df, low_df, volume_df, n=20, m=60):
    """
    Alpha123
    ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 20
    :param m: default 60
    :return:
    """
    return ((rank(ta.CORREL(ta.SUM(((high_df + low_df) / 2), n), ta.SUM(ta.MA(volume_df, m), n), 9)) <
             rank(ta.CORREL(low_df, volume_df, 6))) * -1)


def alpha191_124(close_df, vwamp_df, n=30):
    """
    Alpha124 (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
    :param close_df:
    :param vwamp_df:
    :param n: default 30
    :return:
    """
    return (close_df - vwamp_df) / decaylinear(rank(ta.MAX(close_df, n)), 2)


def alpha191_125(close_df, volume_df, vwap_df, n=17, m=20):
    """
    Alpha125
    (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5)
    + (VWAP * 0.5)), 3), 16)))
    :param close_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 17
    :param m: default 20
    :return:
    """
    return (rank(decaylinear(ta.CORREL(vwap_df, ta.MA(volume_df, 80), n), m)) /
            rank(decaylinear(delta(((close_df * 0.5) + (vwap_df * 0.5)), 3), 16)))


def alpha191_126(close_df, high_df, low_df):
    """
    Alpha126 (CLOSE+HIGH+LOW)/3
    :param close_df:
    :param high_df:
    :param low_df:
    :return:
    """
    return (close_df + high_df + low_df) / 3


def alpha191_127(close_df, n=12):
    """
    Alpha127 (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
    :param close_df:
    :param n: default 12
    :return:
    """
    return np.sqrt(ta.MA((100 * (close_df - ta.MAX(close_df, n)) / (ta.MAX(close_df, n))) ** 2, n))


def alpha191_128(close_df, high_df, low_df, volume_df, n=14):
    """
    Alpha128
    100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 14
    :return:
    """
    return 100 - (100 / (1 + ta.SUM(((high_df + low_df + close_df) / 3 > delay((high_df + low_df + close_df) / 3, 1))
                                    * volume_df, n) / ta.SUM(((high_df + low_df + close_df) / 3
                                                              < delay((high_df + low_df + close_df) / 3,
                                                                      1)) * volume_df, n)))


def alpha191_129(close_df, n=12):
    """
    Alpha129 SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    :param close_df:
    :param n: default 12
    :return:
    """
    return ta.SUM(np.where(close_df - delay(close_df, 1) < 0, np.abs(close_df - delay(close_df, 1)), 0), n)


def alpha191_130(high_df, low_df, volume_df, vwap_df, n=9, m=10, k=7, l=3):
    """
    Alpha130
    (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) /
    RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))
    :param high_df:
    :param low_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 9
    :param m: default 10
    :param k: default 7
    :param l: default 3
    :return:
    """
    return (rank(decaylinear(ta.CORREL(((high_df + low_df) / 2), ta.MA(volume_df, 40), n), m)) /
            rank(decaylinear(ta.CORREL(rank(vwap_df), rank(volume_df), k), l)))


def alpha191_131(close_df, volume_df, vwap_df, n=18, m=18):
    """
    Alpha131 (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
    :param close_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 18
    :param m: default 18
    :return:
    """
    return rank(delta(vwap_df, 1)) ** tsrank(ta.CORREL(close_df, ta.MA(volume_df, 50), n), m)


def alpha191_132(amount_df, n=20):
    """
    Alpha132 MEAN(AMOUNT,20)
    :param amount_df:
    :param n: default 20
    :return:
    """
    return ta.MA(amount_df, n)


def alpha191_133(high_df, low_df, n=20):
    """
    Alpha133 ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    :param high_df:
    :param low_df:
    :param n: default 20
    :return:
    """
    return ((n - ta.MAXINDEX(high_df, n)) / n) * 100 - ((n - ta.MININDEX(low_df, n)) / n) * 100


def alpha191_134(close_df, volume_df, n=12):
    """
    Alpha134 (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    :param close_df:
    :param volume_df:
    :param n: default 12
    :return:
    """
    return (close_df - delay(close_df, n)) / delay(close_df, n) * volume_df


def alpha191_135(close_df, n=20):
    """
    Alpha135 SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    :param close_df:
    :param n: default 20
    :return:
    """
    return sma(delay(close_df / delay(close_df, n), 1), n, 1)


def alpha191_136(open_df, volume_df, n=10):
    """
    Alpha136 ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    :param open_df:
    :param volume_df:
    :param n: default 10
    :return:
    """
    return -1 * rank(delta(open_df, 3)) * ta.CORREL(open_df, volume_df, n)


def alpha191_137(open_df, close_df, high_df, low_df):
    """
    Alpha137
    16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,
    1))>ABS(LOW-DELAY(CLOSE,1)) &
    ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS
    E,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) &
    ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLO
    SE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OP
    EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    :param open_df:
    :param close_df:
    :param high_df:
    :param low_df:
    :return:
    """
    delay_close = delay(close_df, 1)
    delay_open = delay(open_df, 1)
    delay_low = delay(low_df, 1)

    abs_high_close = np.abs(high_df - delay_close)
    abs_low_close = np.abs(low_df - delay_close)
    abs_high_low = np.abs(high_df - delay_low)

    cond1 = (abs_high_close > abs_low_close) & (abs_high_close > abs_high_low)
    cond2 = (abs_low_close > abs_high_low) & (abs_low_close > abs_high_close)

    numerator = 16 * (close_df - delay_close + (close_df - open_df) / 2 + delay_close - delay_open)
    denominator = np.where(cond1, abs_high_close + abs_low_close / 2 + np.abs(delay_close - delay_open) / 4,
                           np.where(cond2, abs_low_close + abs_high_close / 2 + np.abs(delay_close - delay_open) / 4,
                                    abs_high_low + np.abs(delay_close - delay_open) / 4))
    result = numerator / denominator * np.maximum(abs_high_close, abs_low_close)
    return result


def alpha191_138(close_df, volume_df, vwap_df, n=20):
    """
    Alpha138
    ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) -
    TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)
    :param close_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 20
    :return:
    """
    low_df = close_df * 0.7 + vwap_df * 0.3
    delta_df = delta(low_df, 3)
    decay_df = decaylinear(delta_df, n)
    rank_df = rank(decay_df)

    tsrank_df = tsrank(low_df, 8)
    mean_df = ta.MA(volume_df, 60)
    tsrank_df1 = tsrank(mean_df, 17)
    corr_df = ta.CORREL(tsrank_df, tsrank_df1, 5)
    tsrank_df2 = tsrank(corr_df, 19)
    decay_df1 = decaylinear(tsrank_df2, 16)
    tsrank_df3 = tsrank(decay_df1, 7)

    return (rank_df - tsrank_df3) * -1


def alpha191_139(open_df, volume_df, n=10):
    """
    Alpha139 (-1 * CORR(OPEN, VOLUME, 10))
    :param open_df:
    :param volume_df:
    :param n: default 10
    :return:
    """
    return -1 * ta.CORREL(open_df, volume_df, n)


def alpha191_140(open_df, close_df, high_df, low_df, volume_df, n=8):
    """
    Alpha140
    MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)),
    TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))
    :param open_df:
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 8
    :return:
    """
    rank_open = rank(open_df)
    rank_low = rank(low_df)
    rank_high = rank(high_df)
    rank_close = rank(close_df)

    rank_open_low = rank_open + rank_low
    rank_high_close = rank_high + rank_close

    decay_df = decaylinear(rank_open_low - rank_high_close, n)
    rank_decay_df = rank(decay_df)

    tsrank_close_df = tsrank(close_df, 8)
    mean_volume_df = ta.MA(volume_df, 60)
    tsrank_mean_volume_df = tsrank(mean_volume_df, 20)
    corr_df = ta.CORREL(tsrank_close_df, tsrank_mean_volume_df, 8)
    decay_df1 = decaylinear(corr_df, 7)
    tsrank_decay_df1 = tsrank(decay_df1, 3)

    return np.minimum(rank_decay_df, tsrank_decay_df1)


def alpha191_141(high_df, volume_df, n=9):
    """
    Alpha141 (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
    :param high_df:
    :param volume_df:
    :param n: default 9
    :return:
    """
    rank_high_df = rank(high_df)
    mean_volume_df = ta.MA(volume_df, 15)
    rank_mean_volume_df = rank(mean_volume_df)
    corr_df = ta.CORREL(rank_high_df, rank_mean_volume_df, n)
    rank_corr_df = rank(corr_df)

    return rank_corr_df * -1


def alpha191_142(close_df, volume_df, n1=10, n2=1, n3=5):
    """
    Alpha142
    (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME
    /MEAN(VOLUME,20)), 5)))
    :param close_df:
    :param volume_df:
    :param n1: default 10
    :param n2: default 1
    :param n3: default 5
    :return:
    """
    tsrank_close_df = tsrank(close_df, n1)
    rank_tsrank_close_df = rank(tsrank_close_df) * -1

    delta_close_df = delta(close_df, n2)
    delta_delta_close_df = delta(delta_close_df, n2)
    rank_delta_delta_close_df = rank(delta_delta_close_df)

    mean_volume_df = ta.MA(volume_df, 20)
    volume_mean_volume_df = volume_df / mean_volume_df
    tsrank_volume_mean_volume_df = tsrank(volume_mean_volume_df, n3)
    rank_tsrank_volume_mean_volume_df = rank(tsrank_volume_mean_volume_df)

    return rank_tsrank_close_df * rank_delta_delta_close_df * rank_tsrank_volume_mean_volume_df


def alpha191_143(close_df):
    """
    Alpha143 CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    :param close_df:
    :return:
    """
    close_df1 = np.zeros_like(close_df)
    start_index = close_df.index[0] + 1
    for i in range(start_index, len(close_df)):
        if close_df[i] > close_df[i - 1]:
            close_df1[i] = (close_df[i] - close_df[i - 1]) / close_df[i - 1] * close_df1[i - 1]
        else:
            close_df1[i] = close_df1[i - 1]

    return close_df1


def alpha191_144(close_df, amount_df, n=20):
    """
    Alpha144
    SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT, 20, CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    :param close_df:
    :param amount_df:
    :param n: default 12
    :return:
    """
    close_shifted = delay(close_df, 1)  # DELAY(CLOSE, 1)
    close_condition = close_df < close_shifted  # CLOSE < DELAY(CLOSE, 1)

    close_ratio = np.abs((close_df / close_shifted) - 1) / amount_df

    sumif_result = sum_if(close_ratio, n, close_condition)
    count_result = count(close_condition, n)

    return sumif_result / count_result


def alpha191_145(volume_df, n1=9, n2=26, n3=12):
    """
    Alpha145 (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    :param volume_df:
    :param n1: default 9
    :param n2: default 26
    :param n3: default 12
    :return:
    """
    mean_volume_df1 = ta.MA(volume_df, n1)
    mean_volume_df2 = ta.MA(volume_df, n2)
    mean_volume_df3 = ta.MA(volume_df, n3)

    return (mean_volume_df1 - mean_volume_df2) / mean_volume_df3 * 100


def alpha191_146(close_df, n1=20, n2=61, n3=60, m=2):
    """
    Alpha146
    MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((
    CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOS
    E-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,
    1))/DELAY(CLOSE,1),61,2)))^2,60);
    :param close_df:
    :param n1: default 20
    :param n2: default 61
    :param n3: default 60
    :param m: default 2
    :return:
    """
    close_shifted = delay(close_df, 1)  # DELAY(CLOSE, 1)

    close_diff_ratio = (close_df - close_shifted) / close_shifted
    sma_diff_ratio = sma(close_diff_ratio, n2, m)

    diff = close_diff_ratio - sma_diff_ratio
    sma_diff_squared = sma(diff ** 2, n3)

    result = ta.MA(diff * diff / sma_diff_squared, n1)

    return result


def alpha191_147(close_df, n=12):
    """
    Alpha147 REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    :param close_df:
    :param n: default 12
    :return:
    """
    return alpha191_21(close_df, n)


def alpha191_148(open_df, volume_df, n1=6, n2=14):
    """
    Alpha148 ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
    :param open_df:
    :param volume_df:
    :param n1: default 6
    :param n2: default 14
    :return:
    """
    mean_volume_df = ta.MA(volume_df, 60)
    sum_mean_volume_df = sum(mean_volume_df, 9)

    # corr_result = ta.CORREL(open_df, sum_mean_volume_df, n1)
    # corr_rank = rank(corr_result)

    open_shifted = delay(open_df, 1)
    open_min = ta.MIN(open_df, n2)

    open_condition = open_df < open_shifted
    open_min_condition = open_df < open_min

    open_rank = rank(open_condition)
    open_min_rank = rank(open_min_condition)

    return (open_rank < open_min_rank) * -1


def alpha191_149(close_df, benchmark_close_df, close_ret_df=None, benchmark_close_ret_df=None, n=252):
    """
    Alpha149
    REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)
    ),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELA
    Y(BANCHMARKINDEXCLOSE,1)),252)
    :param close_df:
    :param benchmark_close_df:
    :param close_ret_df:
    :param benchmark_close_ret_df:
    :param n: default 252
    :return:
    """
    # 计算收盘价的日收益率
    ret_close = close_ret_df if close_ret_df is not None else close_df.pct_change(1)
    # 计算基准指数收盘价的日收益率
    ret_index = benchmark_close_ret_df if benchmark_close_ret_df is not None else benchmark_close_df.pct_change(1)
    # 过滤出基准指数收盘价下跌的日收益率
    ret_close_filtered = filter_cond(ret_close, ret_index < 0)
    ret_index_filtered = filter_cond(ret_index, ret_index < 0)
    # 计算过滤后日收益率的252日线性回归斜率
    res = regbeta(ret_close_filtered, ret_index_filtered, n)
    # 返回alpha159序列
    return res


def alpha191_150(close_df, high_df, low_df, volume_df):
    """
    Alpha150 (CLOSE+HIGH+LOW)/3*VOLUME
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :return:
    """
    return (close_df + high_df + low_df) / zero_to_one(3 * volume_df)


def alpha191_151(close_df, n=20):
    """
    Alpha151 SMA(CLOSE-DELAY(CLOSE,20),20,1)
    :param close_df:
    :param n: default 20
    :return:
    """
    close_shifted = delay(close_df, 20)
    close_diff = close_df - close_shifted

    return sma(close_diff, n, 1)


def alpha191_152(close_df, n=9, m=12, l=26):
    """
    Alpha152 SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY
    (CLOSE,9),1),9,1),1),26),9,1)
    :param close_df:
    :param n: default 9
    :param m: default 12
    :param l: default 26
    :return:
    """
    close_shifted = delay(close_df, n)
    close_ratio = close_df / close_shifted

    close_ratio_shifted = delay(close_ratio, 1)
    sma_close_ratio = sma(close_ratio_shifted, n, 1)

    sma_close_ratio_shifted = delay(sma_close_ratio, 1)
    sma_close_ratio_mean = ta.MA(sma_close_ratio_shifted, m)

    sma_close_ratio_mean_shifted = delay(sma_close_ratio_mean, 1)
    sma_close_ratio_mean_mean = ta.MA(sma_close_ratio_mean_shifted, l)

    return sma_close_ratio_mean - sma_close_ratio_mean_mean


def alpha191_153(close_df):
    """
    Alpha153 (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    :param close_df:
    :return:
    """
    mean3 = ta.MA(close_df, 3)
    mean6 = ta.MA(close_df, 6)
    mean12 = ta.MA(close_df, 12)
    mean24 = ta.MA(close_df, 24)

    return (mean3 + mean6 + mean12 + mean24) / 4


def alpha191_154(vwap_df, volume_df, n=180, m=16, l=18):
    """
    Alpha154 (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))
    :param vwap_df:
    :param volume_df:
    :param n: default 180
    :param m: default 16
    :param l: default 18
    :return:
    """
    vwap_min = ta.MIN(vwap_df, m)
    vwap_min_shifted = delay(vwap_min, 1)
    vwap_diff = vwap_df - vwap_min_shifted

    volume_mean = ta.MA(volume_df, n)
    volume_mean_shifted = delay(volume_mean, 1)

    return ta.CORREL(vwap_diff, volume_mean_shifted, l)


def alpha191_155(volume_df, n=13, m=27, l=10):
    """
    Alpha155 SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    :param volume_df:
    :param n: default 13
    :param m: default 27
    :param l: default 10
    :return:
    """
    sma_volume1 = sma(volume_df, n, 2)
    sma_volume2 = sma(volume_df, m, 2)
    sma_volume_diff = sma_volume1 - sma_volume2
    sma_volume_diff_sma = sma(sma_volume_diff, l, 2)

    return sma_volume1 - sma_volume2 - sma_volume_diff_sma


def alpha191_156(close_df, open_df, low_df, volume_df, vwap_df, n=5, m=2, l=3):
    """
    Alpha156
    (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),
    2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)
    :param close_df:
    :param open_df:
    :param low_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 5
    :param m: default 2
    :param l: default 3
    :return:
    """
    vwap_delta = delta(vwap_df, n)
    vwap_decay = decaylinear(vwap_delta, l)
    vwap_decay_rank = rank(vwap_decay)

    open_low = open_df * 0.15 + low_df * 0.85
    open_low_delta = delta(open_low, m)
    open_low_ratio = open_low_delta / open_low
    open_low_ratio_delta = delta(open_low_ratio, m)
    open_low_ratio_delta_rank = rank(open_low_ratio_delta)

    return np.maximum(vwap_decay_rank, open_low_ratio_delta_rank) * -1


def alpha191_157(close_df, high_df, volume_df, vwap_df, n=30, m=37):
    """
    Alpha157 (RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30),37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
    RANK(VOLUME), 11)))
    :param close_df:
    :param high_df:
    :param volume_df:
    :param vwap_df:
    :param n: default 20
    :param m: default 20
    :return:
    """
    volume_mean = ta.MA(volume_df, n)
    volume_mean_sum = ta.SUM(volume_mean, m)
    volume_mean_sum_shifted = delay(volume_mean_sum, 1)

    corr1 = ta.CORREL(close_df, volume_mean_sum_shifted, 15)
    corr1_rank = rank(corr1)

    high_vwap = high_df * 0.1 + vwap_df * 0.9
    high_vwap_rank = rank(high_vwap)

    volume_rank = rank(volume_df)

    corr2 = ta.CORREL(high_vwap_rank, volume_rank, 11)
    corr2_rank = rank(corr2)

    return corr1_rank < corr2_rank


def alpha191_158(close_df, high_df, low_df, n=15, m=2):
    """
    Alpha158 ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 15
    :param m: default 2
    :return:
    """
    close_sma = sma(close_df, n, m)
    high_sma = sma(high_df, n, m)
    low_sma = sma(low_df, n, m)

    return (high_df - high_sma - low_df + low_sma) / close_df


def alpha191_159(close_df, high_df, low_df, n=6, m=12, l=24):
    """
    Alpha159
    ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)
    *12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CL
    OSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,D
    ELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 6
    :param m: default 12
    :param l: default 24
    :return:
    """
    # 计算公式中的各个部分
    part1 = (close_df - ta.SUM(np.minimum(low_df, delay(close_df, 1)), n)) / \
            ta.SUM(np.maximum(high_df, delay(close_df, 1)) - np.minimum(low_df, delay(close_df, 1)), n)
    part2 = (close_df - ta.SUM(np.minimum(low_df, delay(close_df, 1)), m)) / \
            ta.SUM(np.maximum(high_df, delay(close_df, 1)) - np.minimum(low_df, delay(close_df, 1)), m)
    part3 = (close_df - ta.SUM(np.minimum(low_df, delay(close_df, 1)), l)) / \
            ta.SUM(np.maximum(high_df, delay(close_df, 1)) - np.minimum(low_df, delay(close_df, 1)), l)

    # 计算最终结果并乘以100
    result = (part1 * m * l + part2 * n * l + part3 * n * l) * 100 / (n * m + n * l + m * l)

    # 返回结果
    return result


def alpha191_160(close_df, n=20, m=1):
    """
    Alpha160 SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    :param close_df:
    :param n: default 20
    :param m: default 1
    :return:
    """
    close_std = ta.STDDEV(close_df, n)
    close_std_shifted = delay(close_std, 1)

    close_le_close_shifted = close_df <= close_std_shifted

    return sma(close_le_close_shifted, n, m)


def alpha191_161(close_df, high_df, low_df, n=12):
    """
    Alpha161 MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 12
    :return:
    """
    # 计算公式中的各个部分
    part1 = high_df - low_df
    part2 = np.abs(delay(close_df, 1) - high_df)
    part3 = np.abs(delay(close_df, 1) - low_df)

    # 计算最终结果
    result = ta.MA(np.maximum(np.maximum(part1, part2), part3), n)

    # 返回结果
    return result


def alpha191_162(close_df, n=12, m=1):
    """
    Alpha162
    (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOS
    E-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(C
    LOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,
    1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    :param close_df:
    :param n: default 12
    :param m: default 1
    :return:
    """
    # 计算公式中的各个部分
    part1 = sma(np.maximum(close_df - delay(close_df, 1), 0), n, m)
    part2 = sma(np.abs(close_df - delay(close_df, 1)), n, m)

    # 计算最终结果
    result = (part1 / part2 * 100 - ta.MIN(part1 / part2 * 100, n)) / \
             (ta.MAX(part1 / part2 * 100, n) - ta.MIN(part1 / part2 * 100, n))

    # 返回结果
    return result


def alpha191_163(close_df, volume_df, vwap_df, high_df, n=20):
    """
    Alpha163 RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    :param close_df:
    :param volume_df:
    :param vwap_df:
    :param high_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = -1 * ret(close_df)
    part2 = ta.MA(volume_df, n)
    part3 = vwap_df
    part4 = high_df - close_df

    # 计算最终结果
    result = rank((((part1 * part2) * part3) * part4))

    # 返回结果
    return result


def alpha191_164(close_df, high_df, low_df, l=12, n=13, m=2):
    """
    Alpha164
    SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
    :param close_df:
    :param high_df:
    :param low_df:
    :param l: default 12
    :param n: default 13
    :param m: default 2
    :return:
    """
    # 计算公式中的各个部分
    part1 = (close_df > delay(close_df, 1)) / (close_df - delay(close_df, 1))
    part2 = ta.MIN((close_df > delay(close_df, 1)) / (close_df - delay(close_df, 1)), l)
    part3 = high_df - low_df

    # 计算最终结果
    result = sma((part1 - part2) / zero_to_one(part3 * 100), n, m)

    # 返回结果
    return result


def alpha191_165(close_df, n=48):
    """
    Alpha165 MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    计算差异值累计的范围相对于价格波动的变化程度。
    :param close_df:
    :param n: default 48
    :return:
    """
    # 计算公式中的各个部分
    part1 = ta.SUM(close_df - ta.MA(close_df, n))
    part2 = ta.STDDEV(close_df, n)

    # 计算最终结果
    result = ta.MAX(part1, n) - ta.MIN(part1, n) / part2

    # 返回结果
    return result


def alpha191_166(close_df, n=20):
    """
    Alpha166
    -20* （ 20-1 ）^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    :param close_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = close_df / delay(close_df, 1) - 1
    part2 = ta.MA(close_df / delay(close_df, 1) - 1, n)
    part3 = ta.SUM(close_df / delay(close_df, 1), 20) ** 2

    # 计算最终结果
    result = -20 * (n - 1) ** 1.5 * ta.SUM(part1 - part2, n) / ((n - 1) * (n - 2) * part3 ** 1.5)

    # 返回结果
    return result


def alpha191_167(close_df, n=12):
    """
    Alpha167 SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    :param close_df:
    :param n: default 12
    :return:
    """
    # 计算公式中的各个部分
    part1 = close_df - delay(close_df, 1)

    # 计算最终结果
    result = ta.SUM(np.where(part1 > 0, part1, 0), n)

    # 返回结果
    return result


def alpha191_168(volume_df, n=20):
    """
    Alpha168 (-1*VOLUME/MEAN(VOLUME,20))
    :param volume_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = volume_df

    # 计算最终结果
    result = -1 * part1 / ta.MA(part1, n)

    # 返回结果
    return result


def alpha191_169(close_df, n=9, m=12):
    """
    Alpha169
    SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),
    26),10,1)
    :param close_df:
    :param n: default 9
    :param m: default 12
    :return:
    """
    # 计算公式中的各个部分
    part1 = sma(close_df - delay(close_df, 1), n, 1)
    part2 = sma(part1, 1, 1)
    part3 = sma(part1, 1, 1)

    # 计算最终结果
    result = sma(part2 - part3, m, 1)

    # 返回结果
    return result


def alpha191_170(close_df, volume_df, high_df, vwap_df, n=5):
    """
    Alpha170
    ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /
    5))) - RANK((VWAP - DELAY(VWAP, 5))))
    :param close_df:
    :param volume_df:
    :param high_df:
    :param vwap_df:
    :param n: default 5
    :return:
    """
    # 计算公式中的各个部分
    part1 = rank(1 / close_df) * volume_df / ta.MA(volume_df, 20)
    part2 = high_df * rank(high_df - close_df) / (ta.SUM(high_df, n) / n)
    part3 = rank(vwap_df - delay(vwap_df, n))

    # 计算最终结果
    result = part1 * part2 - part3

    # 返回结果
    return result


def alpha191_171(open_df, close_df, high_df, low_df, n=5):
    """
    Alpha171 ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
    :param open_df:
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 5
    :return:
    """
    # 计算公式中的各个部分
    part1 = -1 * (low_df - close_df) * (open_df ** n)
    part2 = (close_df - high_df) * (close_df ** n)

    # 计算最终结果
    result = part1 / part2

    # 返回结果
    return result


def alpha191_172(close_df, high_df, low_df, n=14):
    """
    Alpha172
    MEAN(ABS(SUM((LD>0 & LD>HD) ?LD : 0 ,14) *100/ SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/
    (SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 14
    :return:
    """
    trange = ta.TRANGE(high_df, low_df, close_df)
    low_change = get_ld(low_df)
    high_change = get_hd(high_df)
    # 计算公式中的各个部分
    part1 = np.where((low_change > 0) & (low_change > high_change), low_change, 0)  # 选择满足条件的low_change，否则为0
    part2 = np.where((high_change > 0) & (high_change > low_change), high_change, 0)  # 选择满足条件的high_change，否则为0
    part3 = ta.SUM(part1, n) * 100 / ta.SUM(trange, n)  # 计算part1在n天内的累加值占tr在n天内的累加值的百分比
    part4 = ta.SUM(part2, n) * 100 / ta.SUM(trange, n)  # 计算part2在n天内的累加值占tr在n天内的累加值的百分比

    # 计算最终结果
    result = ta.MA(np.abs(part3 - part4) / (part3 + part4) * 100, 6)

    # 返回结果
    return result


def alpha191_173(close_df, n=13):
    """
    Alpha173 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);
    :param close_df:
    :param n: default 13
    :return:
    """
    # 计算公式中的各个部分
    part1 = 3 * sma(close_df, n, 2)
    part2 = 2 * sma(sma(close_df, n, 2), n, 2)
    part3 = sma(sma(sma(np.log(close_df), n, 2), n, 2), n, 2)

    # 计算最终结果
    result = part1 - part2 + part3

    # 返回结果
    return result


def alpha191_174(close_df, n=20):
    """
    Alpha174 SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    :param close_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = np.where(close_df > delay(close_df, 1), ta.STDDEV(close_df, n), 0)

    # 计算最终结果
    result = sma(part1, n, 1)

    # 返回结果
    return result


def alpha191_175(close_df, high_df, low_df, n=6):
    """
    Alpha175 MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 6
    :return:
    """
    # 计算公式中的各个部分
    part1 = np.maximum(high_df - low_df, np.abs(delay(close_df, 1) - high_df))
    part2 = np.abs(delay(close_df, 1) - low_df)
    part3 = np.maximum(part1, part2)

    # 计算最终结果
    result = ta.MA(part3, n)

    # 返回结果
    return result


def alpha191_176(close_df, high_df, low_df, volume_df, n=12, m=6):
    """
    Alpha176 CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 12
    :param m: default 6
    :return:
    """
    # 计算公式中的各个部分
    part1 = (close_df - ta.MIN(low_df, n)) / (ta.MAX(high_df, n) - ta.MIN(low_df, n))
    part2 = rank(part1)
    part3 = rank(volume_df)

    # 计算最终结果
    result = ta.CORREL(part2, part3, m)

    # 返回结果
    return result


def alpha191_177(high_df, n=20):
    """
    Alpha177 ((20-HIGHDAY(HIGH,20))/20)*100
    :param high_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = n - ta.MAXINDEX(high_df, n)

    # 计算最终结果
    result = part1 / n * 100

    # 返回结果
    return result


def alpha191_178(close_df, n=20):
    """
    Alpha178 ((20-LOWDAY(LOW,20))/20)*100
    :param close_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = n - ta.MININDEX(close_df, n)

    # 计算最终结果
    result = part1 / n * 100

    # 返回结果
    return result


def alpha191_179(volume_df, vwap_df, low_df, n=50, m=4, l=12):
    """
    Alpha179 (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
    :param volume_df:
    :param vwap_df:
    :param low_df:
    :param n: default 50
    :param m: default 4
    :param l: default 12
    :return:
    """
    # 计算公式中的各个部分
    part1 = rank(ta.CORREL(vwap_df, volume_df, m))
    part2 = rank(ta.CORREL(rank(low_df), rank(ta.MA(volume_df, n)), l))

    # 计算最终结果
    result = part1 * part2

    # 返回结果
    return result


def alpha191_180(close_df, volume_df, n=20, m=7):
    """
    Alpha180
    ((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME)))
    :param close_df:
    :param volume_df:
    :param n: default 20
    :param m: default 7
    :return:
    """
    # 计算公式中的各个部分
    part1 = ta.MA(volume_df, n)
    part2 = np.where(part1 < volume_df, -1 * tsrank(np.abs(delta(close_df, m)), 60) * np.sign(delta(close_df, m)),
                     -1 * volume_df)

    # 返回结果
    return part2


def alpha191_181(close_df, benchmark_close_df, n=20):
    """
    Alpha181
    SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(B
    ANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
    :param close_df:
    :param benchmark_close_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = ((close_df / delay(close_df, 1) - 1) - ta.MA((close_df / delay(close_df, 1) - 1), n)) - (
            benchmark_close_df - ta.MA(benchmark_close_df, n)) ** 2
    part2 = (benchmark_close_df - ta.MA(benchmark_close_df, n)) ** 3

    # 计算最终结果
    result = sum(part1, n) / sum(part2, n)

    # 返回结果
    return result


def alpha191_182(close_df, open_df, banchmark_close_df, banchmark_open_df, n=20):
    """
    Alpha182
    COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN &
    BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
    :param close_df:
    :param open_df:
    :param banchmark_close_df:
    :param banchmark_open_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    part1 = np.where((close_df > open_df) & (banchmark_close_df > banchmark_open_df), 1, 0)
    part2 = np.where((close_df < open_df) & (banchmark_close_df < banchmark_open_df), 1, 0)

    # 计算最终结果
    result = count(part1 | part2, n) / n

    # 返回结果
    return result


def alpha191_183(close_df, n=24):
    """
    Alpha183 MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    :param close_df:
    :param n: default 24
    :return:
    """
    # 计算CLOSE和它的24天平均值的差值
    diff = close_df - ta.MA(close_df, n)
    # 计算diff的累加值
    sumac = ta.SUM(np.abs(diff), n)
    # 计算sumac的最大值和最小值
    max_sumac = ta.MAX(sumac, n)
    min_sumac = ta.MIN(sumac, n)

    # 计算CLOSE的24天标准差
    std_close = ta.STDDEV(close_df, n)
    # 计算最终结果
    result = (max_sumac - min_sumac) / std_close
    # 返回结果
    return result


def alpha191_184(close_df, open_df, n=200):
    """
    Alpha184 (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
    :param close_df:
    :param open_df:
    :param n: default 200
    :return:
    """
    # 计算公式中的各个部分
    part1 = rank(ta.CORREL(delay(open_df - close_df, 1), close_df, n))
    part2 = rank(open_df - close_df)

    # 计算最终结果
    result = part1 + part2

    # 返回结果
    return result


def alpha191_185(close_df, open_df):
    """
    Alpha185 RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
    :param close_df:
    :param open_df:
    :return:
    """
    # 计算公式中的各个部分
    part1 = 1 - (open_df / close_df)
    part2 = part1 ** 2
    part3 = -1 * part2

    # 计算最终结果
    result = rank(part3)

    # 返回结果
    return result


def alpha191_186(close_df, high_df, low_df, n=14, m=6):
    """
    Alpha186
    (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &
    HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &
    HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 &
    LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 &
    LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
    :param close_df:
    :param high_df:
    :param low_df:
    :param n: default 14
    :param m: default 6
    :return:
    """
    # 计算公式中的各个部分
    # 计算LD和HD
    ld = get_ld(low_df)
    hd = get_hd(high_df)
    # 计算TR
    trange = ta.TRANGE(high_df, low_df, close_df)
    # 计算LD>0 & LD>HD
    ld_gt0 = ld > 0
    ld_gt0_hd = ld_gt0 & (ld > hd)
    # 计算HD>0 & HD>LD
    hd_gt0 = hd > 0
    hd_gt0_ld = hd_gt0 & (hd > ld)
    # 计算SUM((LD>0 & LD>HD)?LD:0,14)
    sum_ld = ta.SUM(np.where(ld_gt0_hd, ld, 0), n)
    # 计算SUM((HD>0 & HD>LD)?HD:0,14)
    sum_hd = ta.SUM(np.where(hd_gt0_ld, hd, 0), n)
    # 计算SUM(TR,14)
    sum_tr = ta.SUM(trange, n)
    # 计算sum_ld*100/sum_tr
    part1 = sum_ld * 100 / sum_tr
    # 计算sum_hd*100/sum_tr
    part2 = sum_hd * 100 / sum_tr
    # 计算part1 - part2
    part3 = part1 - part2
    # 计算ABS(part3)
    part4 = np.abs(part3)
    # 计算part4 / (sum_ld*100/sum_tr + sum_hd*100/sum_tr)
    part5 = part4 / (sum_ld * 100 / sum_tr + sum_hd * 100 / sum_tr)
    # 计算part5 * 100
    part6 = part5 * 100
    # 计算MEAN(part6, 6)
    part7 = ta.MA(part6, m)
    # 计算DELAY(part7, 6)
    part8 = delay(part7, m)
    # 计算part7 + part8
    part9 = part7 + part8
    # 计算part9 / 2
    result = part9 / 2

    # 返回结果
    return result


def alpha191_187(open_df, high_df, n=20):
    """
    Alpha187 SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
    :param open_df:
    :param high_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    # 计算OPEN<=DELAY(OPEN,1)
    open_le_delay_open = open_df <= delay(open_df, 1)
    # 计算MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))
    part1 = np.maximum(high_df - open_df, open_df - delay(open_df, 1))
    # 计算OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))
    part2 = np.where(open_le_delay_open, 0, part1)
    # 计算SUM(part2, 20)
    result = ta.SUM(part2, n)

    # 返回结果
    return result


def alpha191_188(high_df, low_df, n=11, m=2):
    """
    Alpha188 ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    :param high_df:
    :param low_df:
    :param n: default 11
    :param m: default 2
    :return:
    """
    # 计算公式中的各个部分
    # 计算HIGH-LOW
    high_sub_low = high_df - low_df
    # 计算SMA(HIGH-LOW,11,2)
    part1 = sma(high_sub_low, n, m)
    # 计算HIGH-LOW–SMA(HIGH-LOW,11,2)
    part2 = high_sub_low - part1
    # 计算part2 / SMA(HIGH-LOW,11,2)
    part3 = part2 / part1
    # 计算part3 * 100
    result = part3 * 100

    # 返回结果
    return result


def alpha191_189(close_df, n=6):
    """
    Alpha189 MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    :param close_df:
    :param n: default 6
    :return:
    """
    # 计算公式中的各个部分
    # 计算CLOSE-MEAN(CLOSE,6)
    part1 = close_df - ta.MA(close_df, n)
    # 计算ABS(CLOSE-MEAN(CLOSE,6))
    part2 = np.abs(part1)
    # 计算MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    result = ta.MA(part2, n)

    # 返回结果
    return result


def alpha191_190(close_df, n=20):
    """
    Alpha190
    LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(C
    LOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-
    1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOS
    E)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))
    )
    :param close_df:
    :param n: default 20
    :return:
    """
    # 计算公式中的各个部分
    # 计算CLOSE/DELAY(CLOSE)
    close_div_delay_close = close_df / delay(close_df, 1)
    # 计算CLOSE/DELAY(CLOSE,19)
    close_div_delay_close_19 = close_df / delay(close_df, 19)
    # 计算(CLOSE/DELAY(CLOSE,19))^(1/20)
    part1 = np.power(close_div_delay_close_19, 1 / 20)
    # 计算CLOSE/DELAY(CLOSE)-1
    part2 = close_div_delay_close - 1
    # 计算part2-part1-1
    part3 = part2 - part1 - 1
    # 计算COUNT(part3>0, 20)
    part4 = count(part3 > 0, n)
    # 计算COUNT(part3<0, 20)
    part5 = count(part3 < 0, n)
    # 计算SUMIF(part3^2, 20, part3<0)
    part6 = sum_if(np.power(part3, 2), n, part3 < 0)
    # 计算SUMIF(part3^2, 20, part3>0)
    part7 = sum_if(np.power(part3, 2), n, part3 > 0)
    # 计算part4-1
    part8 = part4 - 1
    # 计算part5*part6
    part9 = part5 * part6
    # 计算part8*part9
    part10 = part8 * part9
    # 计算part5*part7
    part11 = part5 * part7
    # 计算log(part10/part11)
    result = np.log(part10 / part11)

    # 返回结果
    return result


def alpha191_191(close_df, high_df, low_df, volume_df, n=20, m=5):
    """
    Alpha191 ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
    :param close_df:
    :param high_df:
    :param low_df:
    :param volume_df:
    :param n: default 20
    :param m: default 5
    :return:
    """
    # 计算公式中的各个部分
    # 计算MEAN(VOLUME,20)
    part1 = ta.MA(volume_df, n)
    # 计算CORR(MEAN(VOLUME,20), LOW, 5)
    part2 = ta.CORREL(part1, low_df, m)
    # 计算(HIGH + LOW) / 2
    part3 = (high_df + low_df) / 2
    # 计算CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)
    part4 = part2 + part3
    # 计算((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
    result = part4 - close_df

    # 返回结果
    return result

#%%
