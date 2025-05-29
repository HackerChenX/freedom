"""
ZXM体系趋势识别指标模块

实现ZXM体系的7个趋势识别指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from scipy.stats import linregress

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMDailyTrendUp(BaseIndicator):
    """
    ZXM趋势-日线上移指标
    
    判断60日或120日均线是否向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-日线上移指标"""
        super().__init__(name="ZXMDailyTrendUp", description="ZXM趋势-日线上移指标，判断日线均线是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-日线上移指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        j1:MA(C,60)>=REF(MA(C,60),1);
        j2:MA(C,120)>=REF(MA(C,120),1);
        xg:j1 OR j2;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算60日均线和120日均线
        ma60 = data["close"].rolling(window=60).mean()
        ma120 = data["close"].rolling(window=120).mean()
        
        # 计算均线是否上移
        j1 = ma60 >= ma60.shift(1)
        j2 = ma120 >= ma120.shift(1)
        
        # 计算趋势信号
        xg = j1 | j2
        
        # 添加计算结果到数据框
        result["MA60"] = ma60
        result["MA120"] = ma120
        result["J1"] = j1
        result["J2"] = j2
        result["XG"] = xg
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算日线上移指标的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. 均线上移信号(XG)：趋势向好，+25分
        score[result["XG"]] += 25
        
        # 2. 均线趋势强度评分
        # 如果价格位于均线上方，额外加分
        price_above_ma60 = data["close"] > result["MA60"]
        price_above_ma120 = data["close"] > result["MA120"]
        
        # 价格位于60日均线上方加分
        score[price_above_ma60] += 5
        
        # 价格位于120日均线上方加分
        score[price_above_ma120] += 10
        
        # 3. 连续性评分
        # 连续多日均线上移，表示趋势较强
        continuous_up_days = pd.Series(0, index=data.index)
        for i in range(5, len(data)):
            if all(result["XG"].iloc[i-5:i+1]):
                continuous_up_days.iloc[i] = 5
            elif all(result["XG"].iloc[i-3:i+1]):
                continuous_up_days.iloc[i] = 3
            elif all(result["XG"].iloc[i-1:i+1]):
                continuous_up_days.iloc[i] = 1
        
        # 根据连续上移天数加分
        score[continuous_up_days == 1] += 5
        score[continuous_up_days == 3] += 10
        score[continuous_up_days == 5] += 15
        
        # 4. 双均线共振评分
        # 60日均线和120日均线同时上移，信号更强
        both_up = result["J1"] & result["J2"]
        score[both_up] += 10
        
        # 5. 均线角度评分
        # 计算均线斜率
        ma60_slope = result["MA60"].diff(5) / result["MA60"].shift(5)
        ma120_slope = result["MA120"].diff(5) / result["MA120"].shift(5)
        
        # 根据斜率大小加分，斜率越大上升越快
        score[ma60_slope > 0.02] += 5  # 周涨幅>2%
        score[ma60_slope > 0.05] += 5  # 周涨幅>5%
        
        score[ma120_slope > 0.01] += 5  # 周涨幅>1%
        score[ma120_slope > 0.03] += 5  # 周涨幅>3%
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别日线上移指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础形态判断
            if last_row["XG"]:
                if last_row["J1"] and last_row["J2"]:
                    patterns.append("双均线上移")
                elif last_row["J1"]:
                    patterns.append("60日均线上移")
                elif last_row["J2"]:
                    patterns.append("120日均线上移")
            else:
                patterns.append("均线趋势走平或下移")
            
            # 均线位置关系
            if data["close"].iloc[-1] > last_row["MA60"] > last_row["MA120"]:
                patterns.append("价格站上双均线，多头排列")
            elif last_row["MA60"] > last_row["MA120"] and data["close"].iloc[-1] < last_row["MA60"]:
                patterns.append("均线多头排列，价格回踩60日线")
            elif last_row["MA60"] < last_row["MA120"] and data["close"].iloc[-1] < last_row["MA60"]:
                patterns.append("均线空头排列，价格在均线下方")
            
            # 近期趋势变化
            if len(result) >= 10:
                if not result["XG"].iloc[-10:-5].any() and result["XG"].iloc[-5:].all():
                    patterns.append("趋势由弱转强")
                elif result["XG"].iloc[-10:-5].all() and not result["XG"].iloc[-5:].any():
                    patterns.append("趋势由强转弱")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals['buy_signal'] = result["XG"]
        signals['sell_signal'] = ~result["XG"]
        signals['neutral_signal'] = False
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[result["XG"], 'trend'] = 1  # 均线上移看涨
        signals.loc[~result["XG"], 'trend'] = -1  # 均线不上移看跌
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["XG"], 'signal_type'] = 'daily_ma_up'
        signals.loc[~result["XG"], 'signal_type'] = 'daily_ma_down'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        signals.loc[result["XG"] & result["J1"] & result["J2"], 'signal_desc'] = '双均线上移，趋势强劲'
        signals.loc[result["XG"] & result["J1"] & ~result["J2"], 'signal_desc'] = '60日均线上移，中期趋势向好'
        signals.loc[result["XG"] & ~result["J1"] & result["J2"], 'signal_desc'] = '120日均线上移，长期趋势向好'
        signals.loc[~result["XG"], 'signal_desc'] = '均线不上移，趋势走弱'
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        # 双均线上移，置信度更高
        signals.loc[result["XG"] & result["J1"] & result["J2"], 'confidence'] = 80
        # 评分高的信号，置信度更高
        signals.loc[score > 70, 'confidence'] = 75
        signals.loc[score > 85, 'confidence'] = 90
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[result["XG"], 'position_size'] = 0.3  # 基础仓位
        signals.loc[(result["XG"]) & (score > 70), 'position_size'] = 0.5  # 高分仓位
        signals.loc[(result["XG"]) & (score > 85), 'position_size'] = 0.7  # 极高分仓位
        
        # 止损位 - 使用均线作为参考
        signals['stop_loss'] = 0.0
        mask = result["XG"]
        for i in data.index[mask]:
            ma60_val = result.loc[i, "MA60"]
            close_val = data.loc[i, "close"]
            # 如果价格在均线上方，则以均线为止损位
            if close_val > ma60_val:
                signals.loc[i, 'stop_loss'] = ma60_val * 0.98  # 均线下方2%
            else:
                # 否则使用近期低点
                try:
                    idx = data.index.get_loc(i)
                    if idx >= 10:
                        low_price = data.iloc[idx-10:idx+1]['low'].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
                except:
                    continue
        
        # 市场环境和成交量确认
        signals['market_env'] = 'normal'
        signals['volume_confirmation'] = False
        
        return signals


class ZXMWeeklyTrendUp(BaseIndicator):
    """
    ZXM趋势-周线上移指标
    
    判断周线10周、20周或30周均线是否向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-周线上移指标"""
        super().__init__(name="ZXMWeeklyTrendUp", description="ZXM趋势-周线上移指标，判断周线均线是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周线上移指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        a1:MA(C,10)>=REF(MA(C,10),1);
        b1:MA(C,20)>=REF(MA(C,20),1);
        c1:MA(C,30)>=REF(MA(C,30),1);
        xg:a1 OR b1 OR c1;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算10周、20周和30周均线
        ma10 = data["close"].rolling(window=10).mean()
        ma20 = data["close"].rolling(window=20).mean()
        ma30 = data["close"].rolling(window=30).mean()
        
        # 计算均线是否上移
        a1 = ma10 >= ma10.shift(1)
        b1 = ma20 >= ma20.shift(1)
        c1 = ma30 >= ma30.shift(1)
        
        # 计算趋势信号
        xg = a1 | b1 | c1
        
        # 添加计算结果到数据框
        result["MA10"] = ma10
        result["MA20"] = ma20
        result["MA30"] = ma30
        result["A1"] = a1
        result["B1"] = b1
        result["C1"] = c1
        result["XG"] = xg
        
        return result


class ZXMMonthlyKDJTrendUp(BaseIndicator):
    """
    ZXM趋势-月KDJ·D及K上移指标
    
    判断月线KDJ指标的D值和K值是否同时向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-月KDJ·D及K上移指标"""
        super().__init__(name="ZXMMonthlyKDJTrendUp", description="ZXM趋势-月KDJ·D及K上移指标，判断月线KDJ·D和K值是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-月KDJ·D及K上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为月线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        xg:D>=REF(D,1) AND K>=REF(K,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算趋势信号
        xg = (d >= d.shift(1)) & (k >= k.shift(1))
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result


class ZXMWeeklyKDJDOrDEATrendUp(BaseIndicator):
    """
    ZXM趋势-周KDJ·D/DEA上移指标
    
    判断周线KDJ指标的D值或MACD的DEA值是否有一个向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-周KDJ·D/DEA上移指标"""
        super().__init__(name="ZXMWeeklyKDJDOrDEATrendUp", description="ZXM趋势-周KDJ·D/DEA上移指标，判断周线KDJ·D或DEA值是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周KDJ·D/DEA上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        xg:D>=REF(D,1) OR DEA>=REF(DEA,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        
        # 计算趋势信号
        xg = (d >= d.shift(1)) | (dea >= dea.shift(1))
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["DIFF"] = diff
        result["DEA"] = dea
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result


class ZXMWeeklyKDJDTrendUp(BaseIndicator):
    """
    ZXM趋势-周KDJ·D上移指标
    
    判断周线KDJ指标的D值是否向上移动
    """
    
    def __init__(self):
        """初始化ZXM趋势-周KDJ·D上移指标"""
        super().__init__(name="ZXMWeeklyKDJDTrendUp", description="ZXM趋势-周KDJ·D上移指标，判断周线KDJ·D值是否向上")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周KDJ·D上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        xg:D>=REF(D,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算趋势信号
        xg = d >= d.shift(1)
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result


class ZXMMonthlyMACD(BaseIndicator):
    """
    ZXM趋势-月MACD<1.5指标
    
    判断月线MACD指标是否小于1.5
    """
    
    def __init__(self):
        """初始化ZXM趋势-月MACD<1.5指标"""
        super().__init__(name="ZXMMonthlyMACD", description="ZXM趋势-月MACD<1.5指标，判断月线MACD值是否小于1.5")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-月MACD<1.5指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为月线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        MACD:=2*(DIFF-DEA);
        xg:MACD<1.5
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd = 2 * (diff - dea)
        
        # 计算趋势信号
        xg = macd < 1.5
        
        # 添加计算结果到数据框
        result["EMA12"] = ema12
        result["EMA26"] = ema26
        result["DIFF"] = diff
        result["DEA"] = dea
        result["MACD"] = macd
        result["XG"] = xg
        
        return result


class ZXMWeeklyMACD(BaseIndicator):
    """
    ZXM趋势-周MACD<2指标
    
    判断周线MACD指标是否小于2
    """
    
    def __init__(self):
        """初始化ZXM趋势-周MACD<2指标"""
        super().__init__(name="ZXMWeeklyMACD", description="ZXM趋势-周MACD<2指标，判断周线MACD值是否小于2")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周MACD<2指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        MACD:=2*(DIFF-DEA);
        xg:MACD<2
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd = 2 * (diff - dea)
        
        # 计算趋势信号
        xg = macd < 2
        
        # 添加计算结果到数据框
        result["EMA12"] = ema12
        result["EMA26"] = ema26
        result["DIFF"] = diff
        result["DEA"] = dea
        result["MACD"] = macd
        result["XG"] = xg
        
        return result


class TrendDetector(BaseIndicator):
    """
    ZXM趋势检测器
    
    识别价格趋势的方向和强度
    """
    
    def __init__(self, short_period: int = 20, long_period: int = 60):
        """
        初始化ZXM趋势检测器
        
        Args:
            short_period: 短期周期，默认20日
            long_period: 长期周期，默认60日
        """
        super().__init__(name="TrendDetector", description="ZXM趋势检测器，识别价格趋势的方向和强度")
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势方向和强度
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算均线
        ma_short = data["close"].rolling(window=self.short_period).mean()
        ma_long = data["close"].rolling(window=self.long_period).mean()
        
        # 计算均线斜率（短期和长期）
        ma_short_slope = ma_short.diff(5) / ma_short.shift(5) * 100  # 短期均线5日变化率（%）
        ma_long_slope = ma_long.diff(10) / ma_long.shift(10) * 100   # 长期均线10日变化率（%）
        
        # 判断趋势方向
        # 1. 多头趋势：短期均线>长期均线 且 短期均线斜率>0
        result["UpTrend"] = (ma_short > ma_long) & (ma_short_slope > 0)
        
        # 2. 空头趋势：短期均线<长期均线 且 短期均线斜率<0
        result["DownTrend"] = (ma_short < ma_long) & (ma_short_slope < 0)
        
        # 3. 震荡趋势：不满足上述两种情况
        result["SidewaysTrend"] = ~(result["UpTrend"] | result["DownTrend"])
        
        # 4. 计算趋势强度指标（0-100）
        # 4.1 短期和长期均线距离（归一化）
        ma_distance = (ma_short - ma_long).abs() / ma_long * 100
        
        # 4.2 斜率强度
        slope_strength = ma_short_slope.abs()
        
        # 4.3 趋势持续性（近期连续保持同一趋势的天数）
        up_days = pd.Series(0, index=data.index)
        down_days = pd.Series(0, index=data.index)
        
        for i in range(1, len(data)):
            if ma_short.iloc[i] > ma_long.iloc[i] and ma_short_slope.iloc[i] > 0:
                up_days.iloc[i] = up_days.iloc[i-1] + 1
            elif ma_short.iloc[i] < ma_long.iloc[i] and ma_short_slope.iloc[i] < 0:
                down_days.iloc[i] = down_days.iloc[i-1] + 1
        
        # 4.4 计算趋势强度（0-100分）
        # 距离因子：最大20分
        distance_factor = ma_distance.clip(0, 10) * 2
        
        # 斜率因子：最大40分
        slope_factor = slope_strength.clip(0, 4) * 10
        
        # 持续性因子：最大40分
        continuity_factor = pd.Series(0, index=data.index)
        continuity_factor[result["UpTrend"]] = (up_days[result["UpTrend"]].clip(0, 20)) * 2
        continuity_factor[result["DownTrend"]] = (down_days[result["DownTrend"]].clip(0, 20)) * 2
        
        # 综合趋势强度（0-100）
        result["TrendStrength"] = (distance_factor + slope_factor + continuity_factor).clip(0, 100)
        
        # 5. 趋势方向指标（1=上升，-1=下降，0=震荡）
        result["TrendDirection"] = 0
        result.loc[result["UpTrend"], "TrendDirection"] = 1
        result.loc[result["DownTrend"], "TrendDirection"] = -1
        
        # 6. 趋势转折信号
        # 6.1 多头转空头
        result["BullToBearSignal"] = False
        
        # 从多头转为空头的条件：前一日是多头趋势，当前是空头趋势
        for i in range(1, len(data)):
            if result["UpTrend"].iloc[i-1] and result["DownTrend"].iloc[i]:
                result["BullToBearSignal"].iloc[i] = True
        
        # 6.2 空头转多头
        result["BearToBullSignal"] = False
        
        # 从空头转为多头的条件：前一日是空头趋势，当前是多头趋势
        for i in range(1, len(data)):
            if result["DownTrend"].iloc[i-1] and result["UpTrend"].iloc[i]:
                result["BearToBullSignal"].iloc[i] = True
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM趋势检测器的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 将趋势强度转换为评分
        # 上升趋势：50-100分（取决于趋势强度）
        # 下降趋势：0-50分（取决于趋势强度）
        # 震荡趋势：40-60分（中性评分）
        
        # 初始化评分为中性（50分）
        score = pd.Series(50, index=data.index)
        
        # 上升趋势
        up_trend_mask = result["UpTrend"]
        if up_trend_mask.any():
            # 上升趋势评分 = 50 + 趋势强度/2
            score[up_trend_mask] = 50 + result.loc[up_trend_mask, "TrendStrength"] / 2
        
        # 下降趋势
        down_trend_mask = result["DownTrend"]
        if down_trend_mask.any():
            # 下降趋势评分 = 50 - 趋势强度/2
            score[down_trend_mask] = 50 - result.loc[down_trend_mask, "TrendStrength"] / 2
        
        # 震荡趋势保持在40-60分范围内（根据偏向性微调）
        sideways_mask = result["SidewaysTrend"]
        if sideways_mask.any():
            # 计算均线
            ma_short = data["close"].rolling(window=self.short_period).mean()
            ma_long = data["close"].rolling(window=self.long_period).mean()
            
            # 判断震荡趋势的偏向性
            sideways_up_bias = (ma_short > ma_long) & sideways_mask
            sideways_down_bias = (ma_short < ma_long) & sideways_mask
            sideways_neutral = sideways_mask & ~(sideways_up_bias | sideways_down_bias)
            
            # 根据偏向性微调评分
            score[sideways_up_bias] = 55
            score[sideways_down_bias] = 45
            score[sideways_neutral] = 50
        
        # 趋势转折加分/减分
        bull_to_bear = result["BullToBearSignal"]
        bear_to_bull = result["BearToBullSignal"]
        
        # 多头转空头是强烈卖出信号
        score[bull_to_bear] = 20
        
        # 空头转多头是强烈买入信号
        score[bear_to_bull] = 80
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM趋势检测器相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 判断趋势方向
            if last_row["UpTrend"]:
                patterns.append("上升趋势")
                
                # 判断趋势强度
                strength = last_row["TrendStrength"]
                if strength >= 80:
                    patterns.append("强势上涨趋势")
                elif strength >= 60:
                    patterns.append("中强度上涨趋势")
                else:
                    patterns.append("弱上涨趋势")
                
                # 判断是否刚转为上升趋势
                if last_row["BearToBullSignal"]:
                    patterns.append("趋势反转：空头转多头")
            
            elif last_row["DownTrend"]:
                patterns.append("下降趋势")
                
                # 判断趋势强度
                strength = last_row["TrendStrength"]
                if strength >= 80:
                    patterns.append("强势下跌趋势")
                elif strength >= 60:
                    patterns.append("中强度下跌趋势")
                else:
                    patterns.append("弱下跌趋势")
                
                # 判断是否刚转为下降趋势
                if last_row["BullToBearSignal"]:
                    patterns.append("趋势反转：多头转空头")
            
            else:  # 震荡趋势
                patterns.append("震荡趋势")
                
                # 计算均线判断震荡趋势的偏向性
                ma_short = data["close"].rolling(window=self.short_period).mean().iloc[-1]
                ma_long = data["close"].rolling(window=self.long_period).mean().iloc[-1]
                
                if ma_short > ma_long:
                    patterns.append("震荡偏多头")
                elif ma_short < ma_long:
                    patterns.append("震荡偏空头")
                else:
                    patterns.append("纯震荡无偏向")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals['buy_signal'] = result["BearToBullSignal"] | ((result["UpTrend"]) & (result["TrendStrength"] >= 70))
        signals['sell_signal'] = result["BullToBearSignal"] | ((result["DownTrend"]) & (result["TrendStrength"] >= 70))
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result["TrendDirection"]
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["BearToBullSignal"], 'signal_type'] = 'trend_reversal_bullish'
        signals.loc[result["BullToBearSignal"], 'signal_type'] = 'trend_reversal_bearish'
        signals.loc[(result["UpTrend"]) & (result["TrendStrength"] >= 70) & (~result["BearToBullSignal"]), 'signal_type'] = 'strong_uptrend'
        signals.loc[(result["DownTrend"]) & (result["TrendStrength"] >= 70) & (~result["BullToBearSignal"]), 'signal_type'] = 'strong_downtrend'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "BearToBullSignal"]:
                signals.loc[i, 'signal_desc'] = f"趋势反转：空头转多头，趋势强度{result.loc[i, 'TrendStrength']:.1f}"
            elif result.loc[i, "BullToBearSignal"]:
                signals.loc[i, 'signal_desc'] = f"趋势反转：多头转空头，趋势强度{result.loc[i, 'TrendStrength']:.1f}"
            elif result.loc[i, "UpTrend"]:
                signals.loc[i, 'signal_desc'] = f"上升趋势，强度{result.loc[i, 'TrendStrength']:.1f}"
            elif result.loc[i, "DownTrend"]:
                signals.loc[i, 'signal_desc'] = f"下降趋势，强度{result.loc[i, 'TrendStrength']:.1f}"
            else:
                signals.loc[i, 'signal_desc'] = "震荡趋势，无明确方向"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势强度调整置信度
        signals['confidence'] = signals['confidence'] + (result["TrendStrength"] / 5).clip(0, 20)
        
        # 趋势反转信号有更高的置信度
        signals.loc[result["BearToBullSignal"] | result["BullToBearSignal"], 'confidence'] = 80
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        signals.loc[result["TrendStrength"] >= 80, 'risk_level'] = '低'  # 强趋势风险较低
        signals.loc[result["SidewaysTrend"], 'risk_level'] = '高'  # 震荡市风险较高
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据趋势强度调整仓位
        strong_trend = (result["UpTrend"]) & (result["TrendStrength"] >= 80)
        signals.loc[strong_trend, 'position_size'] = 0.5  # 强趋势，加大仓位
        
        reversal_signal = result["BearToBullSignal"]
        signals.loc[reversal_signal, 'position_size'] = 0.4  # 反转信号，中等偏大仓位
        
        # 止损位 - 使用支撑位或移动平均线
        signals['stop_loss'] = 0.0
        ma_long = data["close"].rolling(window=self.long_period).mean()
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= self.long_period:
                    # 使用长期均线作为止损位
                    signals.loc[i, 'stop_loss'] = ma_long.iloc[idx] * 0.95  # 长期均线下方5%
            except:
                continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        signals.loc[result["UpTrend"] & (result["TrendStrength"] >= 70), 'market_env'] = 'bull_market'
        signals.loc[result["DownTrend"] & (result["TrendStrength"] >= 70), 'market_env'] = 'bear_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals['volume_confirmation'] = True
        
        return signals


class TrendStrength(BaseIndicator):
    """
    ZXM趋势强度指标
    
    量化趋势的强度和持续性
    """
    
    def __init__(self, short_period: int = 20, long_period: int = 60):
        """
        初始化ZXM趋势强度指标
        
        Args:
            short_period: 短期周期，默认20日
            long_period: 长期周期，默认60日
        """
        super().__init__(name="TrendStrength", description="ZXM趋势强度指标，量化趋势的强度和持续性")
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势强度指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势强度指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算均线
        ma_short = data["close"].rolling(window=self.short_period).mean()
        ma_long = data["close"].rolling(window=self.long_period).mean()
        
        # 计算价格与均线距离的比值
        price_ma_short_ratio = (data["close"] - ma_short) / ma_short * 100
        price_ma_long_ratio = (data["close"] - ma_long) / ma_long * 100
        
        # 计算价格趋势方向 (1=上升, -1=下降, 0=中性)
        price_trend = np.zeros(len(data))
        for i in range(10, len(data)):
            # 计算近10日收盘价线性回归斜率
            x = np.arange(10)
            y = data["close"].iloc[i-10:i].values
            slope, _, _, _, _ = linregress(x, y)
            
            if slope > 0:
                price_trend[i] = 1
            elif slope < 0:
                price_trend[i] = -1
        
        result["PriceTrend"] = price_trend
        
        # 计算均线趋势方向
        ma_short_trend = np.zeros(len(data))
        ma_long_trend = np.zeros(len(data))
        
        for i in range(5, len(data)):
            # 短期均线方向
            if ma_short.iloc[i] > ma_short.iloc[i-5]:
                ma_short_trend[i] = 1
            elif ma_short.iloc[i] < ma_short.iloc[i-5]:
                ma_short_trend[i] = -1
            
            # 长期均线方向
            if ma_long.iloc[i] > ma_long.iloc[i-5]:
                ma_long_trend[i] = 1
            elif ma_long.iloc[i] < ma_long.iloc[i-5]:
                ma_long_trend[i] = -1
        
        result["MAShortTrend"] = ma_short_trend
        result["MALongTrend"] = ma_long_trend
        
        # 计算均线相对位置
        result["MARelation"] = np.where(ma_short > ma_long, 1, np.where(ma_short < ma_long, -1, 0))
        
        # 计算趋势持续性 - 连续保持同一趋势的天数
        trend_persistence = np.zeros(len(data))
        
        for i in range(1, len(data)):
            if result["PriceTrend"].iloc[i] == result["PriceTrend"].iloc[i-1] and result["PriceTrend"].iloc[i] != 0:
                trend_persistence[i] = trend_persistence[i-1] + 1
            else:
                trend_persistence[i] = 1
        
        result["TrendPersistence"] = trend_persistence
        
        # 计算波动范围
        atr = self._calculate_atr(data, period=14)
        result["ATR"] = atr
        
        # 波动率占价格比例
        result["ATRRatio"] = atr / data["close"] * 100
        
        # 计算趋势强度指标（0-100）
        # 1. 价格趋势强度
        price_strength = np.zeros(len(data))
        for i in range(20, len(data)):
            # 趋势方向一致性
            direction_consistency = np.abs(np.sum(result["PriceTrend"].iloc[i-20:i]) / 20)
            
            # 价格与均线距离（归一化）
            price_distance = np.abs(price_ma_short_ratio.iloc[i]) / 5  # 除以5进行归一化，5%距离得分1
            
            # 组合得分，最大25分
            price_strength[i] = min(25, direction_consistency * 15 + price_distance * 10)
        
        result["PriceStrength"] = price_strength
        
        # 2. 均线趋势强度
        ma_strength = np.zeros(len(data))
        for i in range(20, len(data)):
            # 短期均线方向一致性
            short_consistency = np.abs(np.sum(result["MAShortTrend"].iloc[i-20:i]) / 20)
            
            # 长期均线方向一致性
            long_consistency = np.abs(np.sum(result["MALongTrend"].iloc[i-20:i]) / 20)
            
            # 均线排列得分
            ma_relation_score = 1 if result["MARelation"].iloc[i] != 0 else 0
            
            # 组合得分，最大25分
            ma_strength[i] = min(25, short_consistency * 10 + long_consistency * 10 + ma_relation_score * 5)
        
        result["MAStrength"] = ma_strength
        
        # 3. 趋势持续性强度
        persistence_strength = np.zeros(len(data))
        for i in range(len(data)):
            # 持续天数得分，最多20天，每天1分，最多20分
            persistence_strength[i] = min(20, result["TrendPersistence"].iloc[i])
        
        result["PersistenceStrength"] = persistence_strength
        
        # 4. 波动强度
        volatility_strength = np.zeros(len(data))
        for i in range(20, len(data)):
            # 计算波动率变化
            atr_change = atr.iloc[i] / atr.iloc[i-20] - 1
            
            # 波动率扩大表示趋势增强，但幅度不能过大
            if atr_change > 0:
                volatility_score = min(15, atr_change * 100)
            else:
                volatility_score = 0
            
            # 波动率占比得分
            ratio_score = min(15, result["ATRRatio"].iloc[i])
            
            # 组合得分，最大30分
            volatility_strength[i] = min(30, volatility_score + ratio_score)
        
        result["VolatilityStrength"] = volatility_strength
        
        # 综合趋势强度（0-100）
        result["TrendStrength"] = (result["PriceStrength"] + result["MAStrength"] + 
                                   result["PersistenceStrength"] + result["VolatilityStrength"]).clip(0, 100)
        
        # 趋势类型 (1=上升趋势, -1=下降趋势, 0=震荡/无趋势)
        result["TrendType"] = np.where(
            (result["PriceTrend"] == 1) & (result["MAShortTrend"] == 1) & (result["MARelation"] == 1), 
            1, 
            np.where(
                (result["PriceTrend"] == -1) & (result["MAShortTrend"] == -1) & (result["MARelation"] == -1), 
                -1, 
                0
            )
        )
        
        return result
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算平均真实波幅(ATR)
        
        Args:
            data: 输入数据，包含OHLC数据
            period: 计算周期，默认14天
            
        Returns:
            pd.Series: ATR值
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算真实波幅(TR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        
        # 计算ATR - 简单移动平均
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM趋势强度指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 根据趋势类型和强度调整评分
        for i in range(len(data)):
            trend_type = result["TrendType"].iloc[i]
            trend_strength = result["TrendStrength"].iloc[i]
            
            # 上升趋势 - 基础分50 + 趋势强度/2
            if trend_type == 1:
                score.iloc[i] = 50 + trend_strength / 2
            
            # 下降趋势 - 基础分50 - 趋势强度/2
            elif trend_type == -1:
                score.iloc[i] = 50 - trend_strength / 2
            
            # 震荡/无趋势 - 基础分50左右小幅波动
            else:
                # 判断微弱偏向
                if result["PriceTrend"].iloc[i] == 1:
                    score.iloc[i] = 55
                elif result["PriceTrend"].iloc[i] == -1:
                    score.iloc[i] = 45
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM趋势强度指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 趋势类型
            if last_row["TrendType"] == 1:
                patterns.append("上升趋势")
                
                # 趋势强度
                strength = last_row["TrendStrength"]
                if strength >= 80:
                    patterns.append("超强上升趋势")
                elif strength >= 60:
                    patterns.append("强上升趋势")
                elif strength >= 40:
                    patterns.append("中等上升趋势")
                else:
                    patterns.append("弱上升趋势")
                    
                # 趋势持续性
                persistence = last_row["TrendPersistence"]
                if persistence >= 20:
                    patterns.append("长期上升趋势")
                elif persistence >= 10:
                    patterns.append("中期上升趋势")
                else:
                    patterns.append("短期上升趋势")
                
            elif last_row["TrendType"] == -1:
                patterns.append("下降趋势")
                
                # 趋势强度
                strength = last_row["TrendStrength"]
                if strength >= 80:
                    patterns.append("超强下降趋势")
                elif strength >= 60:
                    patterns.append("强下降趋势")
                elif strength >= 40:
                    patterns.append("中等下降趋势")
                else:
                    patterns.append("弱下降趋势")
                    
                # 趋势持续性
                persistence = last_row["TrendPersistence"]
                if persistence >= 20:
                    patterns.append("长期下降趋势")
                elif persistence >= 10:
                    patterns.append("中期下降趋势")
                else:
                    patterns.append("短期下降趋势")
                
            else:
                patterns.append("震荡/无趋势")
                
                # 震荡强度
                if last_row["VolatilityStrength"] >= 20:
                    patterns.append("强震荡")
                else:
                    patterns.append("弱震荡")
            
            # 波动特征
            if last_row["ATRRatio"] >= 5:
                patterns.append("大幅波动")
            elif last_row["ATRRatio"] <= 1:
                patterns.append("微弱波动")
            
            # 趋势变化检测
            if len(result) >= 3:
                if result["TrendType"].iloc[-3] == -1 and result["TrendType"].iloc[-1] == 1:
                    patterns.append("趋势反转：由空转多")
                elif result["TrendType"].iloc[-3] == 1 and result["TrendType"].iloc[-1] == -1:
                    patterns.append("趋势反转：由多转空")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals['buy_signal'] = (result["TrendType"] == 1) & (result["TrendStrength"] >= 60)
        signals['sell_signal'] = (result["TrendType"] == -1) & (result["TrendStrength"] >= 60)
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result["TrendType"]
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[signals['buy_signal'], 'signal_type'] = 'strong_uptrend'
        signals.loc[signals['sell_signal'], 'signal_type'] = 'strong_downtrend'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            trend_type = result.loc[i, "TrendType"]
            trend_strength = result.loc[i, "TrendStrength"]
            persistence = result.loc[i, "TrendPersistence"]
            
            if trend_type == 1:
                signals.loc[i, 'signal_desc'] = f"上升趋势，强度{trend_strength:.1f}，持续{persistence:.0f}天"
            elif trend_type == -1:
                signals.loc[i, 'signal_desc'] = f"下降趋势，强度{trend_strength:.1f}，持续{persistence:.0f}天"
            else:
                signals.loc[i, 'signal_desc'] = "震荡/无趋势"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势强度调整置信度
        signals['confidence'] = signals['confidence'] + (result["TrendStrength"] / 5).clip(0, 20)
        
        # 趋势持续时间长的信号有更高的置信度
        signals.loc[result["TrendPersistence"] >= 15, 'confidence'] += 10
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 趋势初期或变化点风险较高
        early_trend = (result["TrendMaturity"] < 30) & ((result["TrendState"] != 0))
        signals.loc[early_trend | result["TrendChange"], 'risk_level'] = '高'
        
        # 健康稳定的中期趋势风险较低
        stable_trend = (result["TrendMaturity"] >= 30) & (result["TrendMaturity"] <= 70) & (result["TrendHealth"] >= 70)
        signals.loc[stable_trend, 'risk_level'] = '低'
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据趋势健康度调整仓位
        signals.loc[(signals['buy_signal']) & (result["TrendHealth"] >= 80), 'position_size'] = 0.5  # 健康趋势，加大仓位
        
        # 止损位
        signals['stop_loss'] = 0.0
        ma20 = data["close"].rolling(window=20).mean()
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= 20:
                    # 使用20日均线作为止损参考
                    signals.loc[i, 'stop_loss'] = ma20.iloc[idx] * 0.95  # 均线下方5%
            except:
                continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        signals.loc[result["TrendState"] == 1, 'market_env'] = 'bull_market'
        signals.loc[result["TrendState"] == -1, 'market_env'] = 'bear_market'
        signals.loc[result["TrendState"] == 0, 'market_env'] = 'sideways_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals['volume_confirmation'] = True
        
        return signals


class TrendDuration(BaseIndicator):
    """
    ZXM趋势持续性分析
    
    分析趋势的持续时间和周期变化
    """
    
    def __init__(self, lookback_period: int = 120):
        """
        初始化ZXM趋势持续性分析
        
        Args:
            lookback_period: 回溯周期，默认120天
        """
        super().__init__(name="TrendDuration", description="ZXM趋势持续性分析，分析趋势的持续时间和周期变化")
        self.lookback_period = lookback_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势持续性指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势持续性指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算短期和长期均线
        ma20 = data["close"].rolling(window=20).mean()
        ma60 = data["close"].rolling(window=60).mean()
        
        # 判断趋势方向
        # 1. 上升趋势：短期均线 > 长期均线
        uptrend = ma20 > ma60
        
        # 2. 下降趋势：短期均线 < 长期均线
        downtrend = ma20 < ma60
        
        # 3. 震荡趋势：短期均线和长期均线接近
        # 定义接近程度为1%以内
        close_to = (abs(ma20 - ma60) / ma60) < 0.01
        
        # 计算趋势状态 (1=上升, -1=下降, 0=震荡)
        trend_state = pd.Series(0, index=data.index)
        trend_state[uptrend] = 1
        trend_state[downtrend] = -1
        trend_state[close_to] = 0
        
        result["TrendState"] = trend_state
        
        # 计算趋势持续天数
        trend_duration = np.zeros(len(data))
        current_trend = 0
        current_duration = 0
        
        for i in range(len(data)):
            if i == 0:
                current_trend = trend_state.iloc[i]
                current_duration = 1
            elif trend_state.iloc[i] == current_trend:
                current_duration += 1
            else:
                current_trend = trend_state.iloc[i]
                current_duration = 1
            
            trend_duration[i] = current_duration
        
        result["TrendDuration"] = trend_duration
        
        # 计算趋势变化点
        trend_change = pd.Series(False, index=data.index)
        for i in range(1, len(data)):
            if trend_state.iloc[i] != trend_state.iloc[i-1] and trend_state.iloc[i] != 0:
                trend_change.iloc[i] = True
        
        result["TrendChange"] = trend_change
        
        # 计算周期性指标
        # 1. 记录趋势变化点的时间间隔
        change_intervals = []
        last_change_idx = 0
        
        for i in range(1, len(data)):
            if trend_change.iloc[i]:
                if last_change_idx > 0:
                    interval = i - last_change_idx
                    change_intervals.append(interval)
                last_change_idx = i
        
        # 2. 计算平均周期长度
        avg_cycle_length = np.mean(change_intervals) if change_intervals else 0
        
        # 3. 计算周期规律性 (变异系数，越小越规律)
        cycle_regularity = np.std(change_intervals) / np.mean(change_intervals) if change_intervals and np.mean(change_intervals) > 0 else 1
        
        # 4. 计算趋势偏向性 (上升趋势占比)
        if len(data) > self.lookback_period:
            uptrend_ratio = np.sum(trend_state.iloc[-self.lookback_period:] == 1) / self.lookback_period
            downtrend_ratio = np.sum(trend_state.iloc[-self.lookback_period:] == -1) / self.lookback_period
        else:
            uptrend_ratio = np.sum(trend_state == 1) / len(data)
            downtrend_ratio = np.sum(trend_state == -1) / len(data)
        
        # 将这些汇总统计量添加为常量列
        result["AvgCycleLength"] = avg_cycle_length
        result["CycleRegularity"] = cycle_regularity
        result["UptrendRatio"] = uptrend_ratio
        result["DowntrendRatio"] = downtrend_ratio
        
        # 计算趋势成熟度 (0-100)
        # 基于当前趋势持续时间与平均周期的比值
        maturity = np.zeros(len(data))
        
        if avg_cycle_length > 0:
            for i in range(len(data)):
                # 成熟度 = 当前持续时间 / 平均周期长度 * 100，最高100
                maturity[i] = min(100, (trend_duration[i] / avg_cycle_length) * 100)
        
        result["TrendMaturity"] = maturity
        
        # 计算趋势健康度 (0-100)
        # 基于价格与均线的关系、均线斜率和成交量配合
        health = np.zeros(len(data))
        
        for i in range(20, len(data)):
            if trend_state.iloc[i] == 1:  # 上升趋势
                # 1. 价格与均线关系
                price_ma_relation = (data["close"].iloc[i] - ma20.iloc[i]) / ma20.iloc[i] * 100
                relation_score = min(40, max(0, 20 + price_ma_relation * 2))
                
                # 2. 均线斜率
                ma_slope = (ma20.iloc[i] - ma20.iloc[i-20]) / ma20.iloc[i-20] * 100
                slope_score = min(40, max(0, ma_slope * 4))
                
                # 3. 简单波动性检查
                volatility = data["close"].iloc[i-20:i].std() / data["close"].iloc[i-20:i].mean() * 100
                volatility_score = min(20, max(0, 20 - abs(volatility - 3) * 2))
                
                health[i] = relation_score + slope_score + volatility_score
                
            elif trend_state.iloc[i] == -1:  # 下降趋势
                # 1. 价格与均线关系
                price_ma_relation = (data["close"].iloc[i] - ma20.iloc[i]) / ma20.iloc[i] * 100
                relation_score = min(40, max(0, 20 - price_ma_relation * 2))
                
                # 2. 均线斜率
                ma_slope = (ma20.iloc[i] - ma20.iloc[i-20]) / ma20.iloc[i-20] * 100
                slope_score = min(40, max(0, -ma_slope * 4))
                
                # 3. 简单波动性检查
                volatility = data["close"].iloc[i-20:i].std() / data["close"].iloc[i-20:i].mean() * 100
                volatility_score = min(20, max(0, 20 - abs(volatility - 3) * 2))
                
                health[i] = relation_score + slope_score + volatility_score
                
            else:  # 震荡趋势
                health[i] = 50  # 默认中等健康度
        
        result["TrendHealth"] = health
        
        # 计算趋势可能的剩余时间
        remaining_time = np.zeros(len(data))
        
        if avg_cycle_length > 0:
            for i in range(len(data)):
                if trend_state.iloc[i] != 0:  # 非震荡趋势
                    # 平均周期长度减去已经持续的时间
                    remaining = max(0, avg_cycle_length - trend_duration[i])
                    remaining_time[i] = remaining
        
        result["RemainingTime"] = remaining_time
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM趋势持续性指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 根据趋势状态和成熟度调整评分
        for i in range(len(data)):
            trend_state = result["TrendState"].iloc[i]
            maturity = result["TrendMaturity"].iloc[i]
            health = result["TrendHealth"].iloc[i]
            
            # 基础调整 - 趋势状态
            if trend_state == 1:  # 上升趋势
                state_score = 20
            elif trend_state == -1:  # 下降趋势
                state_score = -20
            else:  # 震荡趋势
                state_score = 0
            
            # 成熟度调整
            # 成熟度较低(0-30)：趋势初期，加分
            # 成熟度中等(30-70)：趋势中期，保持
            # 成熟度较高(70-100)：趋势后期，减分
            if maturity < 30:
                maturity_score = 10
            elif maturity > 70:
                maturity_score = -10
            else:
                maturity_score = 0
            
            # 健康度调整
            health_score = (health - 50) / 5  # -10到+10的范围
            
            # 综合评分调整
            score.iloc[i] = 50 + state_score + maturity_score + health_score
        
        # 趋势变化点额外调整
        trend_change = result["TrendChange"]
        trend_state = result["TrendState"]
        
        # 由下降转为上升：强烈买入信号
        up_change = trend_change & (trend_state == 1)
        score[up_change] = 80
        
        # 由上升转为下降：强烈卖出信号
        down_change = trend_change & (trend_state == -1)
        score[down_change] = 20
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM趋势持续性指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 趋势状态
            if last_row["TrendState"] == 1:
                patterns.append("上升趋势")
                
                # 趋势成熟度
                maturity = last_row["TrendMaturity"]
                if maturity < 30:
                    patterns.append("上升趋势初期")
                elif maturity < 70:
                    patterns.append("上升趋势中期")
                else:
                    patterns.append("上升趋势后期")
                
                # 趋势健康度
                health = last_row["TrendHealth"]
                if health >= 80:
                    patterns.append("健康上升趋势")
                elif health >= 60:
                    patterns.append("稳定上升趋势")
                else:
                    patterns.append("虚弱上升趋势")
                
            elif last_row["TrendState"] == -1:
                patterns.append("下降趋势")
                
                # 趋势成熟度
                maturity = last_row["TrendMaturity"]
                if maturity < 30:
                    patterns.append("下降趋势初期")
                elif maturity < 70:
                    patterns.append("下降趋势中期")
                else:
                    patterns.append("下降趋势后期")
                
                # 趋势健康度
                health = last_row["TrendHealth"]
                if health >= 80:
                    patterns.append("健康下降趋势")
                elif health >= 60:
                    patterns.append("稳定下降趋势")
                else:
                    patterns.append("虚弱下降趋势")
                
            else:
                patterns.append("震荡/无趋势")
            
            # 趋势变化点
            if last_row["TrendChange"]:
                if last_row["TrendState"] == 1:
                    patterns.append("趋势转折：由空转多")
                elif last_row["TrendState"] == -1:
                    patterns.append("趋势转折：由多转空")
            
            # 趋势持续性
            duration = last_row["TrendDuration"]
            if duration >= 60:
                patterns.append("超长期趋势")
            elif duration >= 30:
                patterns.append("长期趋势")
            elif duration >= 10:
                patterns.append("中期趋势")
            else:
                patterns.append("短期趋势")
            
            # 周期特性
            if last_row["CycleRegularity"] < 0.3:
                patterns.append("规律性强的周期")
            elif last_row["CycleRegularity"] > 0.7:
                patterns.append("不规律的周期")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        # 买入信号：
        # 1. 趋势由下降转为上升
        # 2. 上升趋势初期（成熟度<30）且健康度高（>=70）
        buy_condition1 = result["TrendChange"] & (result["TrendState"] == 1)
        buy_condition2 = (result["TrendState"] == 1) & (result["TrendMaturity"] < 30) & (result["TrendHealth"] >= 70)
        signals['buy_signal'] = buy_condition1 | buy_condition2
        
        # 卖出信号：
        # 1. 趋势由上升转为下降
        # 2. 上升趋势后期（成熟度>80）且健康度低（<60）
        sell_condition1 = result["TrendChange"] & (result["TrendState"] == -1)
        sell_condition2 = (result["TrendState"] == 1) & (result["TrendMaturity"] > 80) & (result["TrendHealth"] < 60)
        signals['sell_signal'] = sell_condition1 | sell_condition2
        
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result["TrendState"]
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[signals['buy_signal'], 'signal_type'] = 'trend_reversal_bullish'
        signals.loc[signals['sell_signal'], 'signal_type'] = 'trend_reversal_bearish'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            trend_state = result.loc[i, "TrendState"]
            duration = result.loc[i, "TrendDuration"]
            maturity = result.loc[i, "TrendMaturity"]
            health = result.loc[i, "TrendHealth"]
            
            if result.loc[i, "TrendChange"]:
                if trend_state == 1:
                    signals.loc[i, 'signal_desc'] = f"趋势反转：由空转多，健康度{health:.1f}"
                elif trend_state == -1:
                    signals.loc[i, 'signal_desc'] = f"趋势反转：由多转空，健康度{health:.1f}"
            elif trend_state == 1:
                maturity_desc = "初期" if maturity < 30 else "中期" if maturity < 70 else "后期"
                signals.loc[i, 'signal_desc'] = f"上升趋势{maturity_desc}，持续{duration:.0f}天，健康度{health:.1f}"
            elif trend_state == -1:
                maturity_desc = "初期" if maturity < 30 else "中期" if maturity < 70 else "后期"
                signals.loc[i, 'signal_desc'] = f"下降趋势{maturity_desc}，持续{duration:.0f}天，健康度{health:.1f}"
            else:
                signals.loc[i, 'signal_desc'] = "震荡/无趋势"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势健康度调整置信度
        health_confidence_adj = ((result["TrendHealth"] - 50) / 50 * 20).clip(-20, 20)
        signals['confidence'] = signals['confidence'] + health_confidence_adj
        
        # 趋势变化点有更高的置信度
        signals.loc[result["TrendChange"], 'confidence'] += 10
        
        # 周期规律性高的趋势有更高的置信度
        signals.loc[result["CycleRegularity"] < 0.3, 'confidence'] += 10
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 趋势初期或变化点风险较高
        early_trend = (result["TrendMaturity"] < 30) & ((result["TrendState"] != 0))
        signals.loc[early_trend | result["TrendChange"], 'risk_level'] = '高'
        
        # 健康稳定的中期趋势风险较低
        stable_trend = (result["TrendMaturity"] >= 30) & (result["TrendMaturity"] <= 70) & (result["TrendHealth"] >= 70)
        signals.loc[stable_trend, 'risk_level'] = '低'
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据趋势健康度调整仓位
        signals.loc[(signals['buy_signal']) & (result["TrendHealth"] >= 80), 'position_size'] = 0.5  # 健康趋势，加大仓位
        
        # 止损位
        signals['stop_loss'] = 0.0
        ma20 = data["close"].rolling(window=20).mean()
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= 20:
                    # 使用最近20天的最低价作为止损参考
                    low_price = data.iloc[idx-20:idx+1]['low'].min()
                    signals.loc[i, 'stop_loss'] = low_price * 0.95  # 最低价下方5%
            except:
                continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        signals.loc[result["TrendState"] == 1, 'market_env'] = 'bull_market'
        signals.loc[result["TrendState"] == -1, 'market_env'] = 'bear_market'
        signals.loc[result["TrendState"] == 0, 'market_env'] = 'sideways_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals['volume_confirmation'] = True
        
        return signals 