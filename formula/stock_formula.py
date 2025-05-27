import numpy as np
from enums.indicators import *
from enums.kline_period import KlinePeriod
from formula.utils import mergeList, mergeAndGetLast, countListAnyMatch, ltn, gt, countTrue, ge
from db.db_manager import DBManager
from utils.logger import stock_logger


# 添加缺失的技术指标函数
def SMA(series, n, m=1):
    """
    计算简单移动平均线
    
    Args:
        series: 数据序列
        n: 周期
        m: 权重
        
    Returns:
        np.array: SMA值
    """
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    
    for i in range(1, len(series)):
        result[i] = (m * series[i] + (n - m) * result[i-1]) / n
    
    return result


def EMA(series, n):
    """
    计算指数移动平均线
    
    Args:
        series: 数据序列
        n: 周期
        
    Returns:
        np.array: EMA值
    """
    alpha = 2 / (n + 1)
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    
    return result


def REF(series, n):
    """
    计算向前引用值
    
    Args:
        series: 数据序列
        n: 引用的偏移量
        
    Returns:
        np.array: 引用结果
    """
    if n <= 0:
        return series
    
    result = np.zeros_like(series, dtype=float)
    result[:n] = np.nan
    result[n:] = series[:-n]
    
    return result


def LLV(series, n):
    """
    计算周期内最低值
    
    Args:
        series: 数据序列
        n: 周期
        
    Returns:
        np.array: 最低值序列
    """
    result = np.zeros_like(series, dtype=float)
    
    for i in range(len(series)):
        if i < n - 1:
            result[i] = np.min(series[:i+1])
        else:
            result[i] = np.min(series[i-n+1:i+1])
    
    return result


def HHV(series, n):
    """
    计算周期内最高值
    
    Args:
        series: 数据序列
        n: 周期
        
    Returns:
        np.array: 最高值序列
    """
    result = np.zeros_like(series, dtype=float)
    
    for i in range(len(series)):
        if i < n - 1:
            result[i] = np.max(series[:i+1])
        else:
            result[i] = np.max(series[i-n+1:i+1])
    
    return result


def KDJ(series, high, low, n=9, m1=3, m2=3):
    """
    计算KDJ指标
    
    Args:
        series: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        n: RSV周期
        m1: K值平滑周期
        m2: D值平滑周期
        
    Returns:
        tuple: (K值序列, D值序列, J值序列)
    """
    # 计算RSV
    rsv = np.zeros_like(series, dtype=float)
    
    for i in range(len(series)):
        if i < n - 1:
            rsv[i] = 50  # 初始值设为50
        else:
            llv = np.min(low[i-n+1:i+1])
            hhv = np.max(high[i-n+1:i+1])
            if hhv == llv:
                rsv[i] = 50
            else:
                rsv[i] = (series[i] - llv) / (hhv - llv) * 100
    
    # 计算K、D、J值
    K = SMA(rsv, m1, 1)
    D = SMA(K, m2, 1)
    J = 3 * K - 2 * D
    
    return K, D, J


def MACD(series, short=12, long=26, mid=9):
    """
    计算MACD指标
    
    Args:
        series: 数据序列
        short: 短期EMA周期
        long: 长期EMA周期
        mid: DEA周期
        
    Returns:
        tuple: (DIFF, DEA, MACD)
    """
    ema_short = EMA(series, short)
    ema_long = EMA(series, long)
    diff = ema_short - ema_long
    dea = EMA(diff, mid)
    macd = 2 * (diff - dea)
    
    return diff, dea, macd


def BOLL(series, n=20, p=2):
    """
    计算布林带指标
    
    Args:
        series: 数据序列
        n: 周期
        p: 标准差倍数
        
    Returns:
        tuple: (上轨, 中轨, 下轨)
    """
    mid = np.zeros_like(series, dtype=float)
    upper = np.zeros_like(series, dtype=float)
    lower = np.zeros_like(series, dtype=float)
    
    for i in range(len(series)):
        if i < n - 1:
            mid[i] = np.mean(series[:i+1])
            std = np.std(series[:i+1])
        else:
            mid[i] = np.mean(series[i-n+1:i+1])
            std = np.std(series[i-n+1:i+1])
        
        upper[i] = mid[i] + p * std
        lower[i] = mid[i] - p * std
    
    return upper, mid, lower


def ABS(series):
    """
    计算绝对值
    
    Args:
        series: 数据序列
        
    Returns:
        np.array: 绝对值序列
    """
    return np.abs(series)


def MA(series, n):
    """
    计算简单移动平均线
    
    Args:
        series: 数据序列
        n: 周期
        
    Returns:
        np.array: MA值
    """
    result = np.zeros_like(series, dtype=float)
    
    for i in range(len(series)):
        if i < n - 1:
            result[i] = np.mean(series[:i+1])
        else:
            result[i] = np.mean(series[i-n+1:i+1])
    
    return result


class StockData:
    """
    股票数据类，用于封装不同周期的股票数据
    """
    
    def __init__(self, stock_code="", level=KlinePeriod.DAILY, start="20170101", end="20240101", sync=False):
        # 初始化实例变量，避免类变量导致的数据共享问题
        self.close = []
        self.open = []
        self.high = []
        self.low = []
        self.volume = []
        self.turnover_rate = []
        self.history = None
        
        if sync:
            self.init_ef(stock_code, level, sync, start, end)
        else:
            self.init_db(stock_code, level, start, end)

    def init_ef(self, stock_code, level=KlinePeriod.DAILY, sync=False, start="20170101", end="20240101"):
        import efinance as ef
        
        # 使用数据库管理器单例
        db_manager = DBManager.get_instance()
        
        try:
            klt = KlinePeriod.get_klt_code(level)
            history = ef.stock.get_quote_history(stock_code, start, end, klt)
            
            if history is None or history.empty:
                stock_logger.warning(f"获取股票 {stock_code} {level.value} 数据为空")
                return
                
            self.close = history['收盘'].values
            self.open = history['开盘'].values
            self.high = history['最高'].values
            self.low = history['最低'].values
            self.volume = history['成交量'].values
            if level == KlinePeriod.DAILY:
                self.turnover_rate = history['换手率'].values
            
            if sync:
                db_manager.save_stock_info(history, str(level))
                
            stock_logger.info(f"成功初始化股票 {stock_code} {level.value} 数据，共 {len(history)} 条记录")
        except Exception as e:
            stock_logger.error(f"初始化股票 {stock_code} {level.value} 数据失败: {e}")

    def init_db(self, stock_code, level=KlinePeriod.DAILY, start="20170101", end="20240101"):
        import pandas as pd
        
        # 使用数据库管理器单例
        db_manager = DBManager.get_instance()
        
        try:
            db_data = db_manager.get_stock_info(stock_code, str(level), start, end)
            
            # 使用DataFrame.empty属性判断结果是否为空
            if db_data.empty:
                stock_logger.warning(f"数据库中没有股票 {stock_code} {level.value} 数据")
                return
                
            # 检查是否是DataFrame格式
            if isinstance(db_data, pd.DataFrame):
                history = db_data
            else:
                # 如果返回的不是DataFrame，尝试转换
                try:
                    # 检查数据列数以适应可能的额外字段(datetime, seq)
                    if len(db_data[0]) > 13:
                        # 如果返回了额外的列，只使用前13列
                        history = pd.DataFrame([row[:13] for row in db_data],
                                        columns=['code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                                                'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'])
                    else:
                        # 按原来的方式处理
                        history = pd.DataFrame(db_data,
                                        columns=['code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                                                'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'])
                except Exception as e:
                    stock_logger.error(f"转换股票数据格式失败: {e}")
                    return
            
            self.close = history['close'].values
            self.open = history['open'].values
            self.high = history['high'].values
            self.low = history['low'].values
            self.volume = history['volume'].values
            self.turnover_rate = history['turnover_rate'].values
            self.history = history
            
            stock_logger.info(f"成功从数据库加载股票 {stock_code} {level.value} 数据，共 {len(history)} 条记录")
        except Exception as e:
            stock_logger.error(f"从数据库加载股票 {stock_code} {level.value} 数据失败: {e}")


class IndustryData:
    """
    行业数据类，用于封装行业指数数据
    """
    
    def __init__(self, symbol, start="20200101", end="20250101", sync=False):
        # 初始化实例变量
        self.close = []
        self.open = []
        self.high = []
        self.low = []
        self.volume = []
        self.history = None
        self.date = []
        self.symbol = symbol
        
        if sync:
            self.init_ef(symbol, start, end, sync)
        else:
            self.init_db(symbol, start, end)

    def init_ef(self, symbol, start="20200101", end="20240101", sync=False):
        import akshare as ak
        
        # 使用数据库管理器单例
        db_manager = DBManager.get_instance()
        
        try:
            history = ak.stock_board_industry_index_ths(symbol, start, end)
            
            if history is None or history.empty:
                stock_logger.warning(f"获取行业 {symbol} 数据为空")
                return
                
            self.close = history['收盘价'].values
            self.open = history['开盘价'].values
            self.high = history['最高价'].values
            self.low = history['最低价'].values
            self.volume = history['成交量'].values
            
            # 填充history[板块]=symbol
            if sync:
                for i in range(len(history)):
                    history.loc[i, '板块'] = symbol
                db_manager.save_industry_info(history)
                
            stock_logger.info(f"成功初始化行业 {symbol} 数据，共 {len(history)} 条记录")
        except Exception as e:
            stock_logger.error(f"初始化行业 {symbol} 数据失败: {e}")

    def init_db(self, symbol, start="20200101", end="20240101"):
        import pandas as pd
        
        # 使用数据库管理器单例
        db_manager = DBManager.get_instance()
        
        try:
            db_data = db_manager.get_industry_info(symbol, start, end)
            
            # 使用DataFrame.empty属性判断结果是否为空
            if db_data.empty:
                stock_logger.warning(f"数据库中没有行业 {symbol} 数据")
                return
                
            # 检查是否是DataFrame格式
            if isinstance(db_data, pd.DataFrame):
                history = db_data
            else:
                # 如果返回的不是DataFrame，尝试转换
                try:
                    history = pd.DataFrame(db_data,
                                      columns=['symbol', 'date', 'open', 'close', 'high', 'low',
                                              'volume'])
                except Exception as e:
                    stock_logger.error(f"转换行业数据格式失败: {e}")
                    return
            
            self.close = history['close'].values
            self.open = history['open'].values
            self.high = history['high'].values
            self.low = history['low'].values
            self.volume = history['volume'].values
            self.date = history['date'].values
            self.history = history
            
            stock_logger.info(f"成功从数据库加载行业 {symbol} 数据，共 {len(history)} 条记录")
        except Exception as e:
            stock_logger.error(f"从数据库加载行业 {symbol} 数据失败: {e}")

    def 吸筹(self, n=10) -> bool:
        if not self.close or len(self.close) == 0:
            return False
            
        C = self.close
        L = self.low
        H = self.high
        
        # 确保数据量充足
        if len(C) < 55 or len(L) < 55 or len(H) < 55:
            stock_logger.debug(f"行业 {self.symbol} 数据量不足，无法计算吸筹")
            return False
            
        try:
            llv = LLV(L, 55)
            hhv = HHV(H, 55)
            
            # 防止除以零
            denominator = hhv - llv
            if np.any(denominator == 0):
                stock_logger.debug(f"行业 {self.symbol} 计算吸筹时出现除零错误")
                return False
            
            v11 = 3 * SMA((C - llv) / (hhv - llv) * 100, 5, 1) - 2 * SMA(SMA((C - llv) / (hhv - llv) * 100, 5, 1), 3, 1)
            v12 = (EMA(v11, 3) - REF(EMA(v11, 3), 1)) / REF(EMA(v11, 3), 1) * 100
            ema_v11 = EMA(v11, 3)
            
            return countListAnyMatch(ema_v11, n, ltn(13)) or (
                    countListAnyMatch(ema_v11, n, ltn(13)) and countListAnyMatch(v12, n, gt(13)))
        except Exception as e:
            stock_logger.error(f"计算行业 {self.symbol} 吸筹指标失败: {e}")
            return False


class StockFormula:
    """
    股票公式类，封装各种技术指标的计算和选股条件
    """
    
    def __init__(self, code, start="20000101", end="20241231", sync=False):
        self.stock_code = code
        self.start_time = start
        self.end_time = end
        
        # 初始化各周期数据
        self.dataDay = StockData(code, KlinePeriod.DAILY, start, end, sync)
        self.dataWeek = StockData(code, KlinePeriod.WEEKLY, start, end, sync)
        self.dataMonth = StockData(code, KlinePeriod.MONTHLY, start, end, sync)
        self.data15 = StockData(code, KlinePeriod.MIN_15, start, end, sync)
        self.data30 = StockData(code, KlinePeriod.MIN_30, start, end, sync)
        self.data60 = StockData(code, KlinePeriod.MIN_60, start, end, sync)
        
        # 确保日线数据已加载
        if self.dataDay.history is not None and len(self.dataDay.history) > 0:
            self.name = self.dataDay.history['name'].values[0]
            self.industry = self.dataDay.history['industry'].values[0]
        else:
            self.name = ""
            self.industry = ""
            stock_logger.warning(f"股票 {code} 日线数据为空，无法获取名称和行业信息")

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
        
        # 检查数据有效性
        if not isinstance(close, (list, np.ndarray)) or len(close) == 0:
            # 返回三个空数组，而不是None
            return np.array([]), np.array([]), np.array([])
            
        return KDJ(close, high, low, 9, 3, 3)

    def MACD(self, level) -> (list, list, list):
        data = self.getLevelData(level)
        close = data.close
        return MACD(close, 12, 26, 9)

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
                  self.__均线偏移量(close, 60), self.__均线偏移量(close, 120), self.__均线偏移量(close, 244)]
        for i in range(0, arrays.__len__()):
            if countListAnyMatch(arrays[i], 8, ltn(n)):
                return True
        return False

    def __均线偏移量(self, close, n):
        return ABS((close / MA(close, n) - 1) * 100)

    def macd任意2个上移(self, level) -> bool:
        return countTrue([self.DEA上移(level), self.DIFF上移(level), self.MACD上移(level)], 2)

    def 吸筹(self, level, n=10) -> bool:
        data = self.getLevelData(level)
        # n = 吸筹周期(level, day)
        
        # 检查数据是否为空
        if not hasattr(data, 'close') or not isinstance(data.close, (list, np.ndarray)) or len(data.close) == 0:
            return False
            
        C = data.close
        L = data.low
        H = data.high
        
        # 确保数据量充足
        if len(C) < 55 or len(L) < 55 or len(H) < 55:
            return False
            
        try:
            llv = LLV(L, 55)
            hhv = HHV(H, 55)
            
            # 防止除以零
            denominator = hhv - llv
            if np.any(denominator == 0):
                return False
            
            v11 = 3 * SMA((C - llv) / (hhv - llv) * 100, 5, 1) - 2 * SMA(SMA((C - llv) / (hhv - llv) * 100, 5, 1), 3, 1)
            ema_v11 = EMA(v11, 3)
            
            # 检查ema_v11是否包含NaN
            if np.isnan(ema_v11).any():
                return False
                
            # 计算v12前检查分母是否为0或NaN
            ref_ema = REF(ema_v11, 1)
            if np.any(np.isnan(ref_ema)) or np.any(ref_ema == 0):
                # 如果只验证ema_v11小于13的条件
                return countListAnyMatch(ema_v11, n, ltn(13))
                
            v12 = (ema_v11 - ref_ema) / ref_ema * 100
            
            return countListAnyMatch(ema_v11, n, ltn(13)) or (
                    countListAnyMatch(ema_v11, n, ltn(13)) and countListAnyMatch(v12, n, gt(13)))
        except Exception as e:
            stock_logger.error(f"计算股票 {self.stock_code} 级别 {level} 吸筹指标失败: {e}")
            return False

    def boll(self):
        dataDay = self.dataDay
        return BOLL(dataDay.close, 20, 2)

    def 大于boll中轨(self):
        return self.dataDay.close[-1] > self.boll()[1][-1]

    def getLevelData(self, level) -> StockData:
        # 如果已经是KlinePeriod类型，直接使用
        if isinstance(level, KlinePeriod):
            period = level
        else:
            # 否则尝试将字符串转换为KlinePeriod枚举
            period_map = {
                "日": KlinePeriod.DAILY,
                "周": KlinePeriod.WEEKLY,
                "月": KlinePeriod.MONTHLY,
                "15": KlinePeriod.MIN_15,
                "15分钟": KlinePeriod.MIN_15,
                "30": KlinePeriod.MIN_30,
                "30分钟": KlinePeriod.MIN_30,
                "60": KlinePeriod.MIN_60,
                "60分钟": KlinePeriod.MIN_60
            }
            period = period_map.get(level, KlinePeriod.DAILY)
        
        if period == KlinePeriod.DAILY:
            return self.dataDay
        elif period == KlinePeriod.WEEKLY:
            return self.dataWeek
        elif period == KlinePeriod.MONTHLY:
            return self.dataMonth
        elif period == KlinePeriod.MIN_15:
            return self.data15
        elif period == KlinePeriod.MIN_30:
            return self.data30
        elif period == KlinePeriod.MIN_60:
            return self.data60


def 主线():
    """
    主线函数，用于分析所有行业
    """
    industries = ['电子元件', '半导体', '互联网', '电子信息', '汽车零部件', '通讯设备', '仪器仪表', '化工行业', '工业机械', '电力行业', '煤炭行业',
                  '化学制药', '贵金属', '家用电器', '游戏', '电池', '塑胶制品', '造纸印刷', '新材料', '有色金属', '保险', '玻璃陶瓷', '水泥建材',
                  '农药兽药', '旅游酒店', '软件服务', '机场航运', '石油行业', '食品饮料', '装修装饰', '园林工程', '安防设备', '公用事业', '电子商务',
                  '船舶制造', '环保工程']
    for industry in industries:
        industryData = IndustryData(industry)
        if industryData.吸筹():
            print(industry)


def 吸筹板块():
    """
    找出正在吸筹的行业板块
    """
    industries = ['电子元件', '半导体', '互联网', '电子信息', '汽车零部件', '通讯设备', '仪器仪表', '化工行业', '工业机械', '电力行业', '煤炭行业',
                  '化学制药', '贵金属', '家用电器', '游戏', '电池', '塑胶制品', '造纸印刷', '新材料', '有色金属', '保险', '玻璃陶瓷', '水泥建材',
                  '农药兽药', '旅游酒店', '软件服务', '机场航运', '石油行业', '食品饮料', '装修装饰', '园林工程', '安防设备', '公用事业', '电子商务',
                  '船舶制造', '环保工程']
    result = []
    for industry in industries:
        industryData = IndustryData(industry)
        if industryData.吸筹():
            result.append(industry)
    return result 