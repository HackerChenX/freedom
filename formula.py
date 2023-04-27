# This is a sample Python script.
from typing import Callable, Any

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import efinance as ef
import numpy
from MyTT import *
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
        history = ef.stock.get_quote_history(stock_code, start_time, end_time, klt)
        self.close = history['收盘'].values
        self.open = history['开盘'].values
        self.high = history['最高'].values
        self.low = history['最低'].values
        self.volume = history['成交量'].values
        if level == "日":
            self.turnover_rate = history['换手率'].values[-1]


stock_code = "300964"
start_time = "20221101"
end_time = "20230301"
dayData = StockData("日")
weekData = StockData("周")
monthData = StockData("月")


def init(code, start, end):
    global stock_code, start_time, end_time
    stock_code = code
    start_time = start
    end_time = end


def 弹性() -> bool:
    low = dayData.low
    high = dayData.high
    close = dayData.close
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
    close = dayData.close
    日线10 = MA(close, 10) >= REF(MA(close, 10), 1)
    日线20 = MA(close, 20) >= REF(MA(close, 20), 1)
    日线30 = MA(close, 30) >= REF(MA(close, 30), 1)
    日线60 = MA(close, 60) >= REF(MA(close, 60), 1)
    return 日线10[-1] and 日线20[-1] and 日线30[-1] and 日线60[-1]


def 周均线上移() -> bool:
    week_close = weekData.close
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


def ge() -> Callable[[Any, Any], bool]:
    return lambda x, y: x >= y


def MACD小于(level, num) -> bool:
    diff, dea, macd = MACD(level)
    return macd[-1] < num


def 换手率大于(num) -> bool:
    return dayData.turnover_rate > num


def 缩量() -> bool:
    volume = dayData.volume
    volumeDiff = mergeList(volume, MA(volume, 2), 5, lambda x, y: (x / y) < 0.95)
    return countTrueList(volumeDiff, 3)


def 回踩均线(level, n=4) -> bool:
    close = getLevelData(level).close
    A20 = ABS((close / MA(close, 20) - 1) * 100)[-1] <= n
    A30 = ABS((close / MA(close, 30) - 1) * 100)[-1] <= n
    A60 = ABS((close / MA(close, 60) - 1) * 100)[-1] <= n
    A120 = ABS((close / MA(close, 120) - 1) * 100)[-1] <= n
    return A20 or A30 or A60 or A120


def 吸筹(level, n=6) -> bool:
    data = getLevelData(level)
    C = data.close
    L = data.low
    H = data.high
    V11 = 3 * SMA((C - LLV(L, 55)) / (HHV(H, 55) - LLV(L, 55)) * 100, 5, 1) - 2 * SMA(
        SMA((C - LLV(L, 55)) / (HHV(H, 55) - LLV(L, 55)) * 100, 5, 1), 3, 1)
    V12 = (EMA(V11, 3) - REF(EMA(V11, 3), 1)) / REF(EMA(V11, 3), 1) * 100
    AA = (EMA(V11, 3) <= 13) and FILTER((EMA(V11, 3) <= 13), 15)
    BB = (EMA(V11, 3) <= 13 < V12) and FILTER((EMA(V11, 3) <= 13 < V12), 10)
    for i in range(1, n + 1):
        if COUNT(AA or BB, i)[-1]:
            return True
    return False


def getLevelData(leve) -> StockData:
    if leve == "日":
        return dayData
    elif leve == "周":
        return weekData
    elif leve == "月":
        return monthData


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
    return any([lst[i:i + n] == [True] * n for i in range(len(lst) - n + 1)])
