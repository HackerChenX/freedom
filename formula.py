# This is a sample Python script.
from typing import Callable, Any

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import efinance as ef
from MyTT import *
import akshare as ak
import numpy as np


class StockData:
    close = []
    open = []
    high = []
    low = []
    volume = []
    turnover_rate = 0.0
    history = []

    def __init__(self, level="日"):
        klt = 101
        if level == "周":
            klt = 102
        elif level == "月":
            klt = 103
        elif level == "15":
            klt = 15
        elif level == "30":
            klt = 30
        elif level == "60":
            klt = 60
        history = ef.stock.get_quote_history(stock_code, start_time, end_time, klt)
        self.close = history['收盘'].values
        self.open = history['开盘'].values
        self.high = history['最高'].values
        self.low = history['最低'].values
        self.volume = history['成交量'].values
        if level == "日":
            self.turnover_rate = history['换手率'].values[-1]


stock_code = "605178"
start_time = "20020101"
end_time = "20230421"
dataDay = StockData("日")
dataWeek = StockData("周")
dataMonth = StockData("月")
data15 = StockData("15")
data30 = StockData("30")
data60 = StockData("60")


def init(code, start, end):
    global stock_code, start_time, end_time, dataDay, dataWeek, dataMonth, data15, data30, data60
    stock_code = code
    start_time = start
    end_time = end
    dataDay = StockData("日")
    dataWeek = StockData("周")
    dataMonth = StockData("月")
    data15 = StockData("15")
    data30 = StockData("30")
    data60 = StockData("60")


def 弹性() -> bool:
    low = dataDay.low
    high = dataDay.high
    close = dataDay.close
    振幅 = 100 * (high - low) / low > 8.1
    涨幅 = close / REF(close, 1) > 1.07
    if len(high) > 110:
        振幅周期 = 110
        涨幅周期 = 60
    else:
        振幅周期 = len(high)
        涨幅周期 = len(high)
    return EXIST(振幅, 振幅周期)[-1] and EXIST(涨幅, 涨幅周期)[-1]


def 日均线上移() -> bool:
    close = dataDay.close
    日线10 = MA(close, 10) >= REF(MA(close, 10), 1)
    日线20 = MA(close, 20) >= REF(MA(close, 20), 1)
    日线30 = MA(close, 30) >= REF(MA(close, 30), 1)
    日线60 = MA(close, 60) >= REF(MA(close, 60), 1)
    return 日线10[-1] and 日线20[-1] and 日线30[-1] and 日线60[-1]


def 周均线上移() -> bool:
    week_close = dataWeek.close
    周线10 = MA(week_close, 10) >= REF(MA(week_close, 10), 1)
    周线20 = MA(week_close, 20) >= REF(MA(week_close, 20), 1)
    周线30 = MA(week_close, 30) >= REF(MA(week_close, 30), 1)
    return 周线10[-1] and 周线20[-1] and 周线30[-1]


def KDJ(level) -> (list, list, list):
    data = getLevelData(level)
    close = data.close
    low = data.low
    high = data.high
    RSV = (close - LLV(low, 9)) / (HHV(high, 9) - LLV(low, 9)) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    J = 3 * K - 2 * D
    return K, D, J


def MACD(level) -> (list, list, list):
    data = getLevelData(level)
    close = data.close
    diff = EMA(close, 12) - EMA(close, 26)
    dea = EMA(diff, 9)
    macd = 2 * (diff - dea)
    return diff, dea, macd


def K和D上移(level) -> bool:
    return K上移(level) and D上移(level)


def D或K或J上移(level) -> bool:
    return D上移(level) or K上移(level) or J上移(level)


def J或K或D或DEA上移(level) -> bool:
    return J上移(level) or K上移(level) or D上移(level) or DEA上移(level)


def D或DEA上移(level) -> bool:
    return D上移(level) or DEA上移(level)


def K和D和DEA上移(level) -> bool:
    return D上移(level) and K上移(level) and DEA上移(level)


def D上移(level) -> bool:
    K, D, J = KDJ(level)
    return mergeAndGetLast(D, REF(D, 1), ge())


def K上移(level) -> bool:
    K, D, J = KDJ(level)
    return mergeAndGetLast(K, REF(K, 1), ge())


def J上移(level) -> bool:
    K, D, J = KDJ(level)
    return mergeAndGetLast(J, REF(J, 1), ge())


def DEA上移(level) -> bool:
    diff, dea, macd = MACD(level)
    return mergeAndGetLast(dea, REF(dea, 1), ge())


def DIFF上移(level) -> bool:
    diff, dea, macd = MACD(level)
    return mergeAndGetLast(diff, REF(diff, 1), ge())


def MACD上移(level) -> bool:
    diff, dea, macd = MACD(level)
    return mergeAndGetLast(macd, REF(macd, 1), ge())


def MACD小于(level, num) -> bool:
    diff, dea, macd = MACD(level)
    return macd[-1] < num


def 换手率大于(num) -> bool:
    return dataDay.turnover_rate > num


def 缩量() -> bool:
    volume = dataDay.volume
    volumeDiff = mergeList(volume, MA(volume, 2), 5, lambda x, y: (x / y) < 0.95)
    return countTrueList(volumeDiff, 3)


def 回踩均线(level, n=4) -> bool:
    close = getLevelData(level).close
    arrays = [__均线偏移量(close, 5), __均线偏移量(close, 10), __均线偏移量(close, 20), __均线偏移量(close, 30),
              __均线偏移量(close, 60), __均线偏移量(close, 120)]
    for i in range(0, arrays.__len__()):
        if countList(arrays[i], 3, lt(n)):
            return True
    return False


def __均线偏移量(close, n):
    return ABS((close / MA(close, n) - 1) * 100)


def 吸筹(level, n=10) -> bool:
    data = getLevelData(level)
    # n = 吸筹周期(level, day)
    if data.close.__len__() == 0:
        return False
    C = data.close
    L = data.low
    H = data.high
    周期 = 55 if len(L) > 55 else len(L)
    llv = LLV(L, 周期)
    hhv = HHV(H, 周期)
    v11 = 3 * SMA((C - llv) / (hhv - llv) * 100, 5, 1) - 2 * SMA(SMA((C - llv) / (hhv - llv) * 100, 5, 1), 3, 1)
    v12 = (EMA(v11, 3) - REF(EMA(v11, 3), 1)) / REF(EMA(v11, 3), 1) * 100
    ema_v11 = EMA(v11, 3)
    return countList(ema_v11, n, lt(13)) or (countList(ema_v11, n, lt(13)) and countList(v12, n, gt(13)))


def getLevelData(leve) -> StockData:
    if leve == "日":
        return dataDay
    elif leve == "周":
        return dataWeek
    elif leve == "月":
        return dataMonth
    elif leve == "15":
        return data15
    elif leve == "30":
        return data30
    elif leve == "60":
        return data60


def mergeList(list1, list2, n, func):
    if len(list1) != len(list2):
        return None
    result = []
    for i in range(1, n + 1):
        result.append(func(list1[-i], list2[-i]))
    return result


def mergeAndGetLast(list1, list2, func):
    if len(list1) != len(list2):
        return None
    result = []
    for i in range(1, len(list1)):
        result.append(func(list1[i], list2[i]))
    return result[-1]


# 判断列表中是否包含指定元素，且连续出现n次
def countTrueList(lst, n):
    n = n if n < len(lst) else len(lst)
    return any([lst[i:i + n] == [True] * n for i in range(len(lst) - n + 1)])


# 判断列表中倒数N个元素是否有满足lambda表达式的
def countList(lst, n, func):
    n = n if n < len(lst) else len(lst)
    return any([func(lst[i]) for i in range(len(lst) - n, len(lst))])


def lt(n):
    return lambda x: x < n


def ge():
    return lambda x, y: x >= y


def gt(n):
    return lambda x: x > n
