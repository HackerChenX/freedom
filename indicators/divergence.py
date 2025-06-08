"""
量价背离指标模块

实现量价背离识别和分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class DivergenceType(Enum):
    """背离类型枚举"""
    NONE = 0             # 无背离
    POSITIVE = 1         # 正背离（底背离）：价格创新低，指标未创新低，看涨信号
    NEGATIVE = 2         # 负背离（顶背离）：价格创新高，指标未创新高，看跌信号
    HIDDEN_POSITIVE = 3  # 隐藏正背离：价格未创新低，指标创新低，看涨信号
    HIDDEN_NEGATIVE = 4  # 隐藏负背离：价格未创新高，指标创新高，看跌信号


class DIVERGENCE(BaseIndicator):
    """
    量价背离指标
    
    用于识别价格与技术指标之间的背离，预示可能的趋势反转
    """
    
    def __init__(self, lookback_period: int = 20, confirm_period: int = 5):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化量价背离指标
        
        Args:
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
        """
        super().__init__(name="DIVERGENCE", description="量价背离指标")
        self.lookback_period = lookback_period
        self.confirm_period = confirm_period
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算量价背离指标
        
        Args:
            df: 输入数据，包含价格和成交量数据
                
        Returns:
            包含量价背离指标的DataFrame
        """
        # 计算价格与成交量背离
        return self.price_volume_divergence(df, self.lookback_period, self.confirm_period)
    
    def price_volume_divergence(self, data: pd.DataFrame, lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算价格与成交量的背离
        
        Args:
            data: 输入数据，包含价格和成交量数据
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "volume"])
        
        # 提取数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数据框
        result = data.copy()
        
        # 初始化背离标志
        price_volume_divergence = np.zeros(len(close), dtype=bool)
        divergence_type = np.full(len(close), "none", dtype=object)
        
        # 计算价格与成交量背离
        for i in range(lookback_period, len(close)):
            # 价格上涨但成交量下降，负背离
            if i >= 5 and close[i] > close[i-5] and volume[i] < volume[i-5]:
                price_volume_divergence[i] = True
                divergence_type[i] = "negative"
            
            # 价格下跌但成交量上升，正背离
            elif i >= 5 and close[i] < close[i-5] and volume[i] > volume[i-5]:
                price_volume_divergence[i] = True
                divergence_type[i] = "positive"
        
        # 添加计算结果到数据框
        result["price_volume_divergence"] = price_volume_divergence
        result["divergence_type"] = divergence_type
        
        return result
    
    def calculate(self, data: pd.DataFrame, indicator_name: str = None, 
                  lookback_period: int = 20, confirm_period: int = 5, 
                  *args, **kwargs) -> pd.DataFrame:
        """
        计算量价背离指标
        
        Args:
            data: 输入数据，包含价格和技术指标数据
            indicator_name: 用于对比的技术指标列名。如果为None，则默认计算价格与成交量的背离。
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果，包含各类背离信号
            
        公式说明：
        PRICE_NEWLOW:=LOW=LLV(LOW,N);
        MACD_NO_NEWLOW:=MACD>LLV(MACD,N);
        POSITIVE_DIVERGENCE:=PRICE_NEWLOW AND MACD_NO_NEWLOW;

        PRICE_NEWHIGH:=HIGH=HHV(HIGH,N);
        MACD_NO_NEWHIGH:=MACD<HHV(MACD,N);
        NEGATIVE_DIVERGENCE:=PRICE_NEWHIGH AND MACD_NO_NEWHIGH;
        """
        # 如果没有提供指标名称，则默认计算价格与成交量的背离
        if indicator_name is None:
            return self.price_volume_divergence(data, lookback_period, confirm_period)
        
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low", indicator_name])
        
        # 提取数据
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        indicator = data[indicator_name].values
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算价格新高新低
        price_newlow = np.zeros(len(close), dtype=bool)
        price_newhigh = np.zeros(len(close), dtype=bool)
        
        # 计算指标新高新低
        indicator_newlow = np.zeros(len(close), dtype=bool)
        indicator_newhigh = np.zeros(len(close), dtype=bool)
        
        # 初始化各类背离
        positive_divergence = np.zeros(len(close), dtype=bool)  # 正背离（底背离）
        negative_divergence = np.zeros(len(close), dtype=bool)  # 负背离（顶背离）
        hidden_positive_divergence = np.zeros(len(close), dtype=bool)  # 隐藏正背离
        hidden_negative_divergence = np.zeros(len(close), dtype=bool)  # 隐藏负背离
        
        # 计算各类背离
        for i in range(lookback_period, len(close)):
            # 计算价格新低：当前低点是否是lookback_period周期内的最低点
            price_newlow[i] = low[i] == np.min(low[i-lookback_period+1:i+1])
            
            # 计算价格新高：当前高点是否是lookback_period周期内的最高点
            price_newhigh[i] = high[i] == np.max(high[i-lookback_period+1:i+1])
            
            # 计算指标新低：当前指标是否是lookback_period周期内的最低点
            indicator_newlow[i] = indicator[i] == np.min(indicator[i-lookback_period+1:i+1])
            
            # 计算指标新高：当前指标是否是lookback_period周期内的最高点
            indicator_newhigh[i] = indicator[i] == np.max(indicator[i-lookback_period+1:i+1])
            
            # 计算正背离（底背离）：价格创新低，但指标未创新低
            if price_newlow[i]:
                # 检查指标是否背离（未创新低）
                indicator_value = indicator[i]
                min_indicator = np.min(indicator[i-confirm_period:i])
                if indicator_value > min_indicator:
                    positive_divergence[i] = True
            
            # 计算负背离（顶背离）：价格创新高，但指标未创新高
            if price_newhigh[i]:
                # 检查指标是否背离（未创新高）
                indicator_value = indicator[i]
                max_indicator = np.max(indicator[i-confirm_period:i])
                if indicator_value < max_indicator:
                    negative_divergence[i] = True
            
            # 计算隐藏正背离：价格未创新低，但指标创新低
            if not price_newlow[i] and indicator_newlow[i]:
                # 检查价格是否在上升趋势中（近期低点高于前期低点）
                current_low = low[i]
                previous_low = np.min(low[i-confirm_period:i])
                if current_low > previous_low:
                    hidden_positive_divergence[i] = True
            
            # 计算隐藏负背离：价格未创新高，但指标创新高
            if not price_newhigh[i] and indicator_newhigh[i]:
                # 检查价格是否在下降趋势中（近期高点低于前期高点）
                current_high = high[i]
                previous_high = np.max(high[i-confirm_period:i])
                if current_high < previous_high:
                    hidden_negative_divergence[i] = True
        
        # 添加计算结果到数据框
        result["price_newlow"] = price_newlow
        result["price_newhigh"] = price_newhigh
        result["indicator_newlow"] = indicator_newlow
        result["indicator_newhigh"] = indicator_newhigh
        result["positive_divergence"] = positive_divergence
        result["negative_divergence"] = negative_divergence
        result["hidden_positive_divergence"] = hidden_positive_divergence
        result["hidden_negative_divergence"] = hidden_negative_divergence
        
        # 计算综合背离信号
        result["any_positive_divergence"] = result["positive_divergence"] | result["hidden_positive_divergence"]
        result["any_negative_divergence"] = result["negative_divergence"] | result["hidden_negative_divergence"]
        result["any_divergence"] = result["any_positive_divergence"] | result["any_negative_divergence"]
        
        # 存储计算结果
        self._result = result
        
        return result

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算量价背离指标原始评分 (0-100分)
        
        Args:
            data: 输入数据，包含价格和技术指标数据
            **kwargs: 额外参数
                indicator_name: 用于对比的技术指标列名，默认为'macd'
                lookback_period: 回溯周期，默认为20
                confirm_period: 确认周期，默认为5
                
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 获取参数
        indicator_name = kwargs.get('indicator_name', 'macd')
        lookback_period = kwargs.get('lookback_period', self.lookback_period)
        confirm_period = kwargs.get('confirm_period', self.confirm_period)
        
        # 确保已计算指标
        if not self.has_result():
            try:
                result = self.calculate(data, indicator_name, lookback_period, confirm_period)
            except Exception as e:
                logger.error(f"计算背离指标时出错: {e}")
                return pd.Series(50.0, index=data.index)  # 返回中性评分
        else:
            result = self._result
        
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 检查结果是否有效
        if result.empty:
            return score
        
        # 1. 基于正背离的评分 (看涨信号，加分)
        if "positive_divergence" in result.columns:
            # 正背离（底背离）加分较多
            score[result["positive_divergence"]] += 25
        
        if "hidden_positive_divergence" in result.columns:
            # 隐藏正背离加分较少
            score[result["hidden_positive_divergence"]] += 15
        
        # 2. 基于负背离的评分 (看跌信号，减分)
        if "negative_divergence" in result.columns:
            # 负背离（顶背离）减分较多
            score[result["negative_divergence"]] -= 25
        
        if "hidden_negative_divergence" in result.columns:
            # 隐藏负背离减分较少
            score[result["hidden_negative_divergence"]] -= 15
        
        # 3. 背离持续性评分
        # 检查最近N天内是否有连续背离信号
        window_size = min(5, len(score))
        if window_size > 0:
            for i in range(window_size, len(score)):
                # 获取最近窗口的背离数据
                recent_window = result.iloc[i-window_size:i]
                
                # 计算正背离和负背离的数量
                positive_count = 0
                negative_count = 0
                
                if "positive_divergence" in recent_window.columns:
                    positive_count += recent_window["positive_divergence"].sum()
                
                if "hidden_positive_divergence" in recent_window.columns:
                    positive_count += recent_window["hidden_positive_divergence"].sum()
                
                if "negative_divergence" in recent_window.columns:
                    negative_count += recent_window["negative_divergence"].sum()
                
                if "hidden_negative_divergence" in recent_window.columns:
                    negative_count += recent_window["hidden_negative_divergence"].sum()
                
                # 根据背离持续性进行额外评分调整
                if positive_count > 1:  # 多次正背离，强化看涨信号
                    score.iloc[i] += positive_count * 2
                
                if negative_count > 1:  # 多次负背离，强化看跌信号
                    score.iloc[i] -= negative_count * 2
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def macd_divergence(self, data: pd.DataFrame, lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算MACD与价格的背离
        
        Args:
            data: 输入数据，包含价格和MACD数据
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low", "macd"])
        
        # 调用通用背离计算方法
        return self.calculate(data, "macd", lookback_period, confirm_period)
    
    def rsi_divergence(self, data: pd.DataFrame, period: int = 14, 
                      lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算RSI与价格的背离
        
        Args:
            data: 输入数据，包含价格数据
            period: RSI计算周期，默认为14
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 如果数据中没有RSI，计算RSI
        if f"RSI_{period}" not in data.columns:
            # 计算RSI
            close = data["close"].values
            delta = np.zeros(len(close))
            delta[1:] = close[1:] - close[:-1]
            
            up = np.zeros_like(delta)
            down = np.zeros_like(delta)
            up[delta > 0] = delta[delta > 0]
            down[delta < 0] = -delta[delta < 0]
            
            roll_up = np.zeros_like(up)
            roll_down = np.zeros_like(down)
            roll_up[0] = up[0]
            roll_down[0] = down[0]
            
            alpha = 1 / period
            for i in range(1, len(close)):
                roll_up[i] = roll_up[i-1] * (1 - alpha) + up[i] * alpha
                roll_down[i] = roll_down[i-1] * (1 - alpha) + down[i] * alpha
            
            rs = np.zeros_like(close)
            for i in range(len(close)):
                if roll_down[i] != 0:
                    rs[i] = roll_up[i] / roll_down[i]
                else:
                    rs[i] = 100.0
            
            rsi = 100 - (100 / (1 + rs))
            
            # 添加RSI到数据
            data_with_rsi = data.copy()
            data_with_rsi[f"RSI_{period}"] = rsi
        else:
            data_with_rsi = data
        
        return self.calculate(data_with_rsi, f"RSI_{period}", lookback_period, confirm_period)
    
    def obv_divergence(self, data: pd.DataFrame, 
                      lookback_period: int = 20, confirm_period: int = 5) -> pd.DataFrame:
        """
        计算OBV与价格的背离
        
        Args:
            data: 输入数据，包含价格和成交量数据
            lookback_period: 回溯周期，默认为20
            confirm_period: 确认周期，默认为5
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low", "volume"])
        
        # 如果数据中没有OBV，计算OBV
        if "OBV" not in data.columns:
            # 计算OBV
            close = data["close"].values
            volume = data["volume"].values
            
            obv = np.zeros_like(close)
            
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            # 添加OBV到数据
            data_with_obv = data.copy()
            data_with_obv["OBV"] = obv
        else:
            data_with_obv = data
        
        return self.calculate(data_with_obv, "OBV", lookback_period, confirm_period)
    
    def _ema(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算指数移动平均
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: EMA结果
        """
        alpha = 2 / (n + 1)
        result = np.zeros_like(series)
        result[0] = series[0]
        
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
        
        return result

    def generate_signals(self, data: pd.DataFrame, indicator_name: str = 'macd', *args, **kwargs) -> pd.DataFrame:
        """
        生成背离指标交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            indicator_name: 用于对比的技术指标列名，默认为'macd'
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 设置参数
        lookback_period = kwargs.get('lookback_period', self.lookback_period)
        confirm_period = kwargs.get('confirm_period', self.confirm_period)
        
        # 计算背离指标
        if indicator_name == 'macd':
            # 使用MACD背离
            if 'macd' not in data.columns:
                # 如果没有MACD列，则计算MACD背离
                divergence_result = self.macd_divergence(data, lookback_period, confirm_period)
            else:
                # 如果有MACD列，则直接使用MACD背离
                divergence_result = self.calculate(data, 'macd', lookback_period, confirm_period)
        elif indicator_name == 'rsi':
            # 使用RSI背离
            if 'rsi' not in data.columns:
                # 如果没有RSI列，则计算RSI背离
                divergence_result = self.rsi_divergence(data, 14, lookback_period, confirm_period)
            else:
                # 如果有RSI列，则直接使用RSI背离
                divergence_result = self.calculate(data, 'rsi', lookback_period, confirm_period)
        elif indicator_name == 'obv':
            # 使用OBV背离
            divergence_result = self.obv_divergence(data, lookback_period, confirm_period)
        elif indicator_name == 'volume':
            # 使用价格与成交量背离
            divergence_result = self.price_volume_divergence(data, lookback_period, confirm_period)
        else:
            # 使用自定义指标背离
            divergence_result = self.calculate(data, indicator_name, lookback_period, confirm_period)
        
        # 提取背离信号
        if 'positive_divergence' in divergence_result.columns:
            positive_divergence = divergence_result['positive_divergence']
        else:
            positive_divergence = pd.Series(False, index=data.index)
        
        if 'negative_divergence' in divergence_result.columns:
            negative_divergence = divergence_result['negative_divergence']
        else:
            negative_divergence = pd.Series(False, index=data.index)
        
        if 'hidden_positive_divergence' in divergence_result.columns:
            hidden_positive = divergence_result['hidden_positive_divergence']
        else:
            hidden_positive = pd.Series(False, index=data.index)
        
        if 'hidden_negative_divergence' in divergence_result.columns:
            hidden_negative = divergence_result['hidden_negative_divergence']
        else:
            hidden_negative = pd.Series(False, index=data.index)
        
        # 生成买入信号（正背离和隐藏正背离）
        buy_indices = positive_divergence | hidden_positive
        signals.loc[buy_indices, 'buy_signal'] = True
        signals.loc[buy_indices, 'neutral_signal'] = False
        signals.loc[buy_indices, 'trend'] = 1
        
        # 生成卖出信号（负背离和隐藏负背离）
        sell_indices = negative_divergence | hidden_negative
        signals.loc[sell_indices, 'sell_signal'] = True
        signals.loc[sell_indices, 'neutral_signal'] = False
        signals.loc[sell_indices, 'trend'] = -1
        
        # 添加信号类型和描述
        for i in range(len(signals)):
            if positive_divergence.iloc[i]:
                signals.loc[signals.index[i], 'signal_type'] = '正背离(底背离)'
                signals.loc[signals.index[i], 'signal_desc'] = f'{indicator_name}指标正背离：价格创新低，指标未创新低，看涨信号'
                signals.loc[signals.index[i], 'confidence'] = 75
                signals.loc[signals.index[i], 'market_env'] = 'reversal_market'
                signals.loc[signals.index[i], 'score'] = 75
            
            elif negative_divergence.iloc[i]:
                signals.loc[signals.index[i], 'signal_type'] = '负背离(顶背离)'
                signals.loc[signals.index[i], 'signal_desc'] = f'{indicator_name}指标负背离：价格创新高，指标未创新高，看跌信号'
                signals.loc[signals.index[i], 'confidence'] = 75
                signals.loc[signals.index[i], 'market_env'] = 'reversal_market'
                signals.loc[signals.index[i], 'score'] = 25
            
            elif hidden_positive.iloc[i]:
                signals.loc[signals.index[i], 'signal_type'] = '隐藏正背离'
                signals.loc[signals.index[i], 'signal_desc'] = f'{indicator_name}指标隐藏正背离：价格未创新低，指标创新低，看涨信号'
                signals.loc[signals.index[i], 'confidence'] = 65
                signals.loc[signals.index[i], 'market_env'] = 'trend_continuation'
                signals.loc[signals.index[i], 'score'] = 70
            
            elif hidden_negative.iloc[i]:
                signals.loc[signals.index[i], 'signal_type'] = '隐藏负背离'
                signals.loc[signals.index[i], 'signal_desc'] = f'{indicator_name}指标隐藏负背离：价格未创新高，指标创新高，看跌信号'
                signals.loc[signals.index[i], 'confidence'] = 65
                signals.loc[signals.index[i], 'market_env'] = 'trend_continuation'
                signals.loc[signals.index[i], 'score'] = 30
        
        # 成交量确认
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 成交量放大确认
            high_volume = vol_ratio > 1.5
            signals.loc[high_volume, 'volume_confirmation'] = True
            
            # 成交量确认增强信号可靠性
            for i in range(len(signals)):
                if (signals['buy_signal'].iloc[i] or signals['sell_signal'].iloc[i]) and high_volume.iloc[i]:
                    current_confidence = signals['confidence'].iloc[i]
                    signals.loc[signals.index[i], 'confidence'] = min(90, current_confidence + 10)
        
        # 更新风险等级和仓位建议
        for i in range(len(signals)):
            confidence = signals['confidence'].iloc[i]
            
            # 根据信号强度和置信度设置风险等级
            if confidence >= 80:
                signals.loc[signals.index[i], 'risk_level'] = '低'
            elif confidence >= 65:
                signals.loc[signals.index[i], 'risk_level'] = '中'
            else:
                signals.loc[signals.index[i], 'risk_level'] = '高'
            
            # 设置建议仓位
            if signals['buy_signal'].iloc[i] or signals['sell_signal'].iloc[i]:
                if confidence >= 80:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif confidence >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif confidence >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
        
        # 计算动态止损
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i]:
                # 买入信号的止损
                if i >= 5 and i < len(data):
                    # 使用前5个交易日的最低价作为止损参考
                    stop_level = data['low'].iloc[i-5:i].min() * 0.98
                    signals.loc[signals.index[i], 'stop_loss'] = stop_level
            
            elif signals['sell_signal'].iloc[i]:
                # 卖出信号的止损
                if i >= 5 and i < len(data):
                    # 使用前5个交易日的最高价作为止损参考
                    stop_level = data['high'].iloc[i-5:i].max() * 1.02
                    signals.loc[signals.index[i], 'stop_loss'] = stop_level
        
        return signals 