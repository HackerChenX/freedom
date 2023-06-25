import efinance as ef
from MyTT import *
import akshare as ak
import numpy as np
import db


class StockData:
    close = []
    open = []
    high = []
    low = []
    volume = []
    turnover_rate = []
    history = []

    def __init__(self, stock_code="", level="日", start="20170101", end="20240101", sync=False):
        if sync:
            self.init_ef(stock_code, level, sync, start, end)
        else:
            self.init_db(stock_code, level, start, end)

    def init_ef(self, stock_code, level="日", sync=False, start="20170101", end="20240101"):
        klt = 101
        if level == "日":
            klt = 101
        elif level == "周":
            klt = 102
        elif level == "月":
            klt = 103
        elif level == "30":
            klt = 30
        elif level == "60":
            klt = 60
        elif level == "15":
            klt = 15
        history = ef.stock.get_quote_history(stock_code, start, end, klt)
        self.close = history['收盘'].values
        self.open = history['开盘'].values
        self.high = history['最高'].values
        self.low = history['最低'].values
        self.volume = history['成交量'].values
        if level == "日":
            self.turnover_rate = history['换手率'].values
        if sync:
            db.save_stock_info(history, level)

    def init_db(self, stock_code, level="日", start="20170101", end="20240101"):
        db_data = db.get_stock_info(stock_code, level, start, end)
        history = pd.DataFrame(db_data,
                               columns=['id', 'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                                        'volume', 'price_change', 'price_range', 'turnover_rate', "industry"])
        self.close = history['close'].values
        self.open = history['open'].values
        self.high = history['high'].values
        self.low = history['low'].values
        self.volume = history['volume'].values
        self.turnover_rate = history['turnover_rate'].values
        self.history = history


class IndustryData:
    close = []
    open = []
    high = []
    low = []
    volume = []
    turnover_rate = 0.0
    history = []

    def __init__(self, symbol, level="日"):
        history = ak.stock_board_industry_index_ths(symbol)
        self.close = history['收盘'].values
        self.open = history['开盘'].values
        self.high = history['最高'].values
        self.low = history['最低'].values
        self.volume = history['成交量'].values
        if level == "日":
            self.turnover_rate = history['换手率'].values[-1]


class Formula:
    stock_code = None
    name = None
    industry = None
    start_time = "20020101"
    end_time = "20230421"
    dataDay = None
    dataWeek = None
    dataMonth = None
    data15 = None
    data30 = None
    data60 = None

    def __init__(self, code, start="20210101", end="20241231", sync=False):
        self.stock_code = code
        self.start_time = start
        self.end_time = end
        self.dataDay = StockData(code, "日", start, end, sync)
        self.dataWeek = StockData(code, "周", start, end, sync)
        self.dataMonth = StockData(code, "月", start, end, sync)
        self.data15 = StockData(code, "15", start, end, sync)
        self.data30 = 合并K线(self.data15)
        self.data60 = 合并K线(self.data30)
        self.name = self.dataDay.history['name'].values[0]
        self.industry = self.dataDay.history['industry'].values[0]

    def get_code(self):
        return self.stock_code

    def get_desc(self):
        return self.stock_code + " " + self.name + " " + self.industry

    def 弹性(self) -> bool:
        low = self.dataDay.low
        high = self.dataDay.high
        close = self.dataDay.close
        np.seterr(divide='ignore', invalid='ignore')
        振幅 = 100 * (high - low) / low > 8.1
        涨幅 = close / REF(close, 1) > 1.07
        if len(high) > 110:
            振幅周期 = 110
            涨幅周期 = 60
        else:
            振幅周期 = len(high)
            涨幅周期 = len(high)
        return EXIST(振幅, 振幅周期)[-1] and EXIST(涨幅, 涨幅周期)[-1]

    def 日均线上移(self) -> bool:
        close = self.dataDay.close
        lst = [30, 60, 120]
        result = []
        for i in lst:
            result.append(mergeAndGetLast(MA(close, i), REF(MA(close, i), 1), ge()))
        return countTrue(result, 2)

    def 周均线上移(self) -> bool:
        week_close = self.dataWeek.close
        lst = [10, 20, 30]
        result = []
        for i in lst:
            result.append(mergeAndGetLast(MA(week_close, i), REF(MA(week_close, i), 1), ge()))
        return countTrue(result, 2)

    def KDJ(self, level) -> (list, list, list):
        data = self.getLevelData(level)
        close = data.close
        low = data.low
        high = data.high
        llv = LLV(low, 9)
        hhv = HHV(high, 9)
        RSV = (close - llv) / (hhv - llv) * 100
        K = EMA(RSV, (3 * 2 - 1))
        D = EMA(K, (3 * 2 - 1))
        J = K * 3 - D * 2
        return K, D, J

    def MACD(self, level) -> (list, list, list):
        data = self.getLevelData(level)
        close = data.close
        DIF = EMA(close, 12) - EMA(close, 26)
        DEA = EMA(DIF, 9)
        macd = (DIF - DEA) * 2
        return RD(DIF), RD(DEA), RD(macd)

    def K和D上移(self, level) -> bool:
        return self.K上移(level) and self.D上移(level)

    def K和D和J上移(self, level) -> bool:
        return self.K上移(level) and self.D上移(level) and self.J上移(level)

    def D或K或J上移(self, level) -> bool:
        return self.D上移(level) or self.K上移(level) or self.J上移(level)

    def J或K或D或DEA上移(self, level) -> bool:
        return self.J上移(level) or self.K上移(level) or self.D上移(level) or self.DEA上移(level)

    def macd_kdj任一指标上移(self, level) -> bool:
        return self.D上移(level) or self.K上移(level) or self.J上移(level) or self.DEA上移(level) or self.MACD上移(
            level) or self.DIFF上移(level)

    def D或DEA上移(self, level) -> bool:
        return self.D上移(level) or self.DEA上移(level)

    def K和D和DEA上移(self, level) -> bool:
        return self.D上移(level) and self.K上移(level) and self.DEA上移(level)

    def D上移(self, level) -> bool:
        K, D, J = self.KDJ(level)
        return mergeAndGetLast(D, REF(D, 1), ge())

    def K上移(self, level) -> bool:
        K, D, J = self.KDJ(level)
        return mergeAndGetLast(K, REF(K, 1), ge())

    def J上移(self, level) -> bool:
        K, D, J = self.KDJ(level)
        return mergeAndGetLast(J, REF(J, 1), ge())

    def DEA上移(self, level) -> bool:
        diff, dea, macd = self.MACD(level)
        return mergeAndGetLast(dea, REF(dea, 1), ge())

    def DIFF上移(self, level) -> bool:
        diff, dea, macd = self.MACD(level)
        return mergeAndGetLast(diff, REF(diff, 1), ge())

    def MACD上移(self, level) -> bool:
        diff, dea, macd = self.MACD(level)
        return mergeAndGetLast(macd, REF(macd, 1), ge())

    def MACD小于(self, level, num) -> bool:
        diff, dea, macd = self.MACD(level)
        return macd[-1] < num

    def 换手率大于(self, num) -> bool:
        return self.dataDay.turnover_rate[-1] > num

    def 缩量(self) -> bool:
        volume = self.dataDay.volume
        volumeDiff = mergeList(volume, MA(volume, 2), 5, lambda x, y: (x / y))
        return countListAnyMatch(volumeDiff, 3, ltn(0.95))

    def 回踩均线(self, level, n=4) -> bool:
        close = self.getLevelData(level).close
        arrays = [self.__均线偏移量(close, 10), self.__均线偏移量(close, 20), self.__均线偏移量(close, 30),
                  self.__均线偏移量(close, 60), self.__均线偏移量(close, 120)]
        for i in range(0, arrays.__len__()):
            if countListAnyMatch(arrays[i], 3, ltn(n)):
                return True
        return False

    def __均线偏移量(self, close, n):
        return ABS((close / MA(close, n) - 1) * 100)

    def macd任意2个上移(self, level) -> bool:
        return countTrue([self.DEA上移(level), self.DIFF上移(level), self.MACD上移(level)], 2)

    def 吸筹(self, level, n=10) -> bool:
        data = self.getLevelData(level)
        # n = 吸筹周期(level, day)
        if data.close.__len__() == 0:
            return False
        C = data.close
        L = data.low
        H = data.high
        llv = LLV(L, 55)
        hhv = HHV(H, 55)
        v11 = 3 * SMA((C - llv) / (hhv - llv) * 100, 5, 1) - 2 * SMA(SMA((C - llv) / (hhv - llv) * 100, 5, 1), 3, 1)
        v12 = (EMA(v11, 3) - REF(EMA(v11, 3), 1)) / REF(EMA(v11, 3), 1) * 100
        ema_v11 = EMA(v11, 3)
        return countListAnyMatch(ema_v11, n, ltn(13)) or (
                countListAnyMatch(ema_v11, n, ltn(13)) and countListAnyMatch(v12, n, gt(13)))

    def boll(self):
        dataDay = self.dataDay
        return BOLL(dataDay.close, 20, 2)

    def 大于boll中轨(self):
        return self.dataDay.close[-1] > self.boll()[1][-1]

    def getLevelData(self, leve) -> StockData:
        if leve == "日":
            return self.dataDay
        elif leve == "周":
            return self.dataWeek
        elif leve == "月":
            return self.dataMonth
        elif leve == "15":
            return self.data15
        elif leve == "30":
            return self.data30
        elif leve == "60":
            return self.data60


def mergeList(list1, list2, n, func):
    if len(list1) != len(list2):
        return None
    result = []
    for i in range(1, n + 1):
        result.insert(0, func(list1[-i], list2[-i]))
    return result


def mergeAndGetLast(list1, list2, func):
    if len(list1) != len(list2):
        return None
    result = []
    for i in range(1, len(list1)):
        result.append(func(list1[i], list2[i]))
    return result[-1]


# 判断列表中是否包含指定元素，且连续出现n次
def countListAllMatch(lst, n, m, func):
    count = 0
    for i in range(len(lst) - n, len(lst)):
        if func(lst[i]):
            count += 1
    return count >= m


# 判断列表最后n个元素是否包含指定元素
def countListAnyMatch(lst, n, func):
    n = n if n < len(lst) else len(lst)
    return any([func(lst[i]) for i in range(len(lst) - n, len(lst))])


def ltn(n):
    return lambda x: x < n


def lt():
    return lambda x, y: x < y


def ge():
    return lambda x, y: x >= y


def gt(n):
    return lambda x: x > n


def LLV(low, n):
    return pd.Series(low).rolling(window=n, min_periods=1).min().values


def HHV(high, n):
    return pd.Series(high).rolling(window=n, min_periods=1).max().values


# list中包含True的个数大于N次
def countTrue(lst, n):
    return lst.count(True) >= n


def 主线():
    df = ak.stock_board_industry_summary_ths()
    # 计算涨跌家数比，=上涨家数/(上涨家数+下跌家数)
    df['涨跌家数比'] = df['上涨家数'] / (df['上涨家数'] + df['下跌家数'])
    # 按照上涨家数/(上涨家数+下跌家数)排序，取前十个
    df_涨跌家数比 = df.sort_values(by='涨跌家数比', ascending=False).head(10)
    # 取序号排名前十的行业
    df_排名 = df.sort_values(by='序号').head(10)
    # 合并df_涨跌家数比和df_排名，去除重复，输出板块列表
    df = pd.concat([df_涨跌家数比, df_排名]).drop_duplicates()
    return df['板块'].tolist()


def 合并K线(stock_data: StockData):
    low_level_close = stock_data.close
    low_level_open = stock_data.open
    low_level_high = stock_data.high
    low_level_low = stock_data.low
    low_level_volume = stock_data.volume
    low_level_turnover_rate = stock_data.turnover_rate
    close = low_level_close[1::2]
    open = low_level_open[::2]
    # low_level_high每两个一组，取最大的组成新的数组
    high = np.maximum(low_level_high[::2], low_level_high[1::2])
    # low_level_low每两个一组，取最小的组成新的数组
    low = np.minimum(low_level_low[::2], low_level_low[1::2])
    # low_level_volume每两个一组，取和组成新的数组
    volume = low_level_volume[::2] + low_level_volume[1::2]
    # low_level_turnover_rate每两个一组，取和组成新的数组，新数组的元素保留两位小数
    turnover_rate = np.round(low_level_turnover_rate[::2] + low_level_turnover_rate[1::2], 2)
    high_level_data = StockData()
    high_level_data.close = close
    high_level_data.open = open
    high_level_data.high = high
    high_level_data.low = low
    high_level_data.volume = volume
    high_level_data.turnover_rate = turnover_rate
    return high_level_data
