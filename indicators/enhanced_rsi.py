"""
增强型相对强弱指标(RSI)模块

实现改进版的RSI指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.rsi import RSI
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRSI(RSI):
    """
    增强型相对强弱指标(RSI)
    
    在标准RSI基础上增加了动态超买超卖阈值、多周期分析、形态识别和背离检测等功能
    """
    
    def __init__(self, 
                period: int = 14, 
                sensitivity: float = 1.0,
                multi_periods: List[int] = None):
        """
        初始化增强型RSI指标
        
        Args:
            period: RSI计算周期，默认为14
            sensitivity: 灵敏度参数，控制对价格变化的响应程度，默认为1.0
            multi_periods: 多周期分析参数，默认为[9, 14, 21]
        """
        self.indicator_type = "oscillator"  # 指标类型：震荡类
        super().__init__(period=period)
        self.name = "EnhancedRSI"
        self.description = "增强型相对强弱指标，优化计算方法和信号质量，增加多周期适应和市场环境感知"
        self.sensitivity = sensitivity
        self.multi_periods = multi_periods or [9, 14, 21]
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return self.indicator_type

    def ensure_columns(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        确保数据包含必需的列

        Args:
            data: 输入数据
            required_columns: 必需的列名列表

        Raises:
            ValueError: 如果缺少必需的列
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需的列: {missing_columns}")
        
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取RSI指标的形态信号

        Args:
            data: 输入数据，通常是已经调用 _calculate 方法计算过的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含各种形态信号的DataFrame
        """
        if not hasattr(self, '_result') or self._result is None:
            self.calculate(data)

        # 定义所有可能的形态信号列
        pattern_cols = [col for col in self._result.columns if
                        ('overbought' in col or
                         'oversold' in col or
                         'cross_up' in col or
                         'cross_down' in col or
                         'divergence' in col) and isinstance(self._result[col].iloc[0], (bool, np.bool_))]

        # 筛选出data中存在的形态列
        existing_patterns = [col for col in pattern_cols if col in self._result.columns]
        
        if not existing_patterns:
            logger.warning("未在输入数据中找到任何形态信号，请先运行 a.calculate(df)")
            return pd.DataFrame(index=data.index)
            
        return self._result[existing_patterns].copy()
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算增强型RSI指标
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            pd.DataFrame: 计算结果，包含RSI及其多周期指标
        """
        # 调用父类的_calculate方法，只获取指标列
        indicator_data = super()._calculate(data, *args, **kwargs)

        # 将指标列与原始数据合并
        result = data.join(indicator_data)

        # 确保数据包含必需的列
        self.ensure_columns(result, ["close", f"rsi_{self.period}"])
        
        # 获取标准RSI结果
        rsi = result[f"rsi_{self.period}"]
        
        # 计算多周期RSI
        for period in self.multi_periods:
            if period != self.period:  # 避免重复计算
                # 注意：这里我们假设父类的calculate已经处理了RSI的计算
                # 如果需要不同的RSI计算方式，需要单独调用
                rsi_period = self._calculate_rsi(result["close"], period)
                result[f"rsi_{period}"] = rsi_period

        # 计算RSI动量 - 衡量RSI的变化速率
        result["rsi_momentum"] = self._calculate_rsi_momentum(rsi)
        
        # 计算RSI变化率 - 相对变化百分比
        result["rsi_rate"] = rsi.diff(periods=1).fillna(0)
        
        # 计算动态超买超卖阈值
        thresholds = self._calculate_dynamic_thresholds(data)
        result["oversold_threshold"] = thresholds["oversold"]
        result["overbought_threshold"] = thresholds["overbought"]
        
        # 保存结果
        self._result = result
        
        return result
    
    def _calculate_rsi(self, price_series: pd.Series, period: int) -> pd.Series:
        """
        计算RSI
        
        Args:
            price_series: 价格序列
            period: 计算周期
            
        Returns:
            pd.Series: RSI序列
        """
        # 计算价格变化
        delta = price_series.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            gain = gain * self.sensitivity
            loss = loss * self.sensitivity
        
        # 使用 ewm 来保持与父类一致的计算方式，避免前导 NaN
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # 处理零除情况
        rs = pd.Series(0.0, index=price_series.index)
        valid_idx = avg_loss > 0
        rs[valid_idx] = avg_gain[valid_idx] / avg_loss[valid_idx]
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rsi_momentum(self, rsi: pd.Series, period: int = 5) -> pd.Series:
        """
        计算RSI动量
        
        Args:
            rsi: RSI序列
            period: 计算周期
            
        Returns:
            pd.Series: RSI动量序列
        """
        # 计算RSI变化率
        rsi_change = rsi - rsi.shift(period)
        
        # 标准化动量
        rsi_momentum = rsi_change / period
        
        return rsi_momentum
    
    def _calculate_dynamic_thresholds(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算动态超买超卖阈值
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, pd.Series]: 包含超买超卖阈值的字典
        """
        # 正确且高效的向量化实现
        # 直接在原始数据帧上计算波动率，让pandas处理索引对齐
        data['volatility'] = data['close'].pct_change().rolling(window=20).std(ddof=0) * np.sqrt(252)

        # 设置默认阈值
        oversold = pd.Series(30.0, index=data.index)
        overbought = pd.Series(70.0, index=data.index)

        # 根据波动率向量化地调整阈值
        overbought = overbought.where(data['volatility'] <= 0.3, 75.0)
        overbought = overbought.where(data['volatility'] >= 0.15, 65.0)
        
        oversold = oversold.where(data['volatility'] <= 0.3, 25.0)
        oversold = oversold.where(data['volatility'] >= 0.15, 35.0)
        
        # 考虑市场环境调整阈值（避免递归调用）
        market_env_str = self._get_simple_market_environment(data)

        # 将字符串转换为MarketEnvironment枚举
        from indicators.base_indicator import MarketEnvironment
        market_env = MarketEnvironment.SIDEWAYS_MARKET  # 默认值

        if market_env_str == 'bullish':
            market_env = MarketEnvironment.BULL_MARKET
        elif market_env_str == 'bearish':
            market_env = MarketEnvironment.BEAR_MARKET
        elif market_env_str == 'volatile':
            market_env = MarketEnvironment.VOLATILE_MARKET
        else:
            market_env = MarketEnvironment.SIDEWAYS_MARKET
        
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市中提高超买阈值，降低超卖阈值
            overbought = overbought + 5
            oversold = oversold - 2
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市中降低超买阈值，提高超卖阈值
            overbought = overbought - 5
            oversold = oversold + 2
        
        return {"oversold": oversold, "overbought": overbought}
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别RSI指标形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        if not self.has_result():
            self.calculate(data)
            
        patterns = []
        rsi = self._result[f"rsi_{self.period}"]
        
        # 1. 超买超卖区域判断
        oversold = self._result["oversold_threshold"]
        overbought = self._result["overbought_threshold"]
        
        if rsi.iloc[-1] < oversold.iloc[-1]:
            patterns.append("超卖区域")
        elif rsi.iloc[-1] > overbought.iloc[-1]:
            patterns.append("超买区域")
            
        # 2. W底/M顶形态检测
        if len(rsi) >= 20:
            if self._detect_w_bottom(rsi.iloc[-20:]):
                patterns.append("W底形态")
            if self._detect_m_top(rsi.iloc[-20:]):
                patterns.append("M顶形态")
                
        # 3. 背离检测
        if len(data) >= 30:
            divergence = self._detect_divergence(data["close"], rsi)
            if divergence == "bullish":
                patterns.append("RSI正背离")
            elif divergence == "bearish":
                patterns.append("RSI负背离")
            
        # 4. 失败摆动形态
        if len(rsi) >= 15:
            if self._detect_failure_swing(rsi.iloc[-15:], "bullish"):
                patterns.append("看涨失败摆动")
            elif self._detect_failure_swing(rsi.iloc[-15:], "bearish"):
                patterns.append("看跌失败摆动")
        
        return patterns
    
    def _detect_w_bottom(self, rsi_series: pd.Series) -> bool:
        """
        检测W底形态
        
        Args:
            rsi_series: RSI序列
            
        Returns:
            bool: 是否检测到W底
        """
        # W底要求：两次探底，第二次低点不低于第一次，中间有反弹
        if len(rsi_series) < 5:
            return False
            
        # 寻找两个低点和一个高点
        low1_idx = rsi_series.idxmin()
        high_idx = rsi_series.loc[low1_idx:].idxmax()
        low2_idx = rsi_series.loc[high_idx:].idxmin()
        
        if low1_idx is None or high_idx is None or low2_idx is None:
            return False
            
        low1 = rsi_series[low1_idx]
        high = rsi_series[high_idx]
        low2 = rsi_series[low2_idx]
        
        # 条件：两个低点均在超卖区，第二个低点高于或等于第一个低点
        # 中间高点不应过高（例如不超过50）
        if low1 < 30 and low2 < 30 and low2 >= low1 and high < 50:
            return True
            
        return False

    def _detect_m_top(self, rsi_series: pd.Series) -> bool:
        """
        检测M顶形态
        
        Args:
            rsi_series: RSI序列
            
        Returns:
            bool: 是否检测到M顶
        """
        # M顶要求：两次冲顶，第二次高点不高于第一次，中间有回调
        if len(rsi_series) < 5:
            return False
            
        # 寻找两个高点和一个低点
        high1_idx = rsi_series.idxmax()
        low_idx = rsi_series.loc[high1_idx:].idxmin()
        high2_idx = rsi_series.loc[low_idx:].idxmax()
        
        if high1_idx is None or low_idx is None or high2_idx is None:
            return False
            
        high1 = rsi_series[high1_idx]
        low = rsi_series[low_idx]
        high2 = rsi_series[high2_idx]
        
        # 条件：两个高点均在超买区，第二个高点低于或等于第一个高点
        # 中间低点不应过低（例如不低于50）
        if high1 > 70 and high2 > 70 and high2 <= high1 and low > 50:
            return True
            
        return False
        
    def _detect_failure_swing(self, rsi_series: pd.Series, direction: str) -> bool:
        """
        检测失败摆动形态
        
        Args:
            rsi_series: RSI序列
            direction: 'bullish' (看涨) or 'bearish' (看跌)
            
        Returns:
            bool: 是否检测到失败摆动
        """
        if direction == "bullish":
            # 看涨失败摆动：RSI跌破超卖线，反弹后再下跌，但未创新低
            if len(rsi_series) < 4:
                return False
                
            # 找到第一个低于30的点
            oversold_points = rsi_series[rsi_series < 30]
            if oversold_points.empty:
                return False
                
            first_dip_idx = oversold_points.index[0]
            
            # 在第一个低点之后寻找反弹点
            rebound_slice = rsi_series.loc[first_dip_idx:]
            rebound_high_idx = rebound_slice.idxmax()
            
            # 在反弹高点之后寻找第二个低点
            second_dip_slice = rebound_slice.loc[rebound_high_idx:]
            if second_dip_slice.empty:
                return False
                
            second_dip_idx = second_dip_slice.idxmin()
            
            # 条件：第一个低点<30, 第二个低点>30
            if rsi_series[first_dip_idx] < 30 and rsi_series[second_dip_idx] > 30:
                return True
                
        elif direction == "bearish":
            # 看跌失败摆动：RSI突破超买线，回调后再上涨，但未创新高
            if len(rsi_series) < 4:
                return False
                
            # 找到第一个高于70的点
            overbought_points = rsi_series[rsi_series > 70]
            if overbought_points.empty:
                return False
                
            first_peak_idx = overbought_points.index[0]
            
            # 在第一个高点之后寻找回调点
            pullback_slice = rsi_series.loc[first_peak_idx:]
            pullback_low_idx = pullback_slice.idxmin()
            
            # 在回调低点之后寻找第二个高点
            second_peak_slice = pullback_slice.loc[pullback_low_idx:]
            if second_peak_slice.empty:
                return False
                
            second_peak_idx = second_peak_slice.idxmax()
            
            # 条件：第一个高点>70, 第二个高点<70
            if rsi_series[first_peak_idx] > 70 and rsi_series[second_peak_idx] < 70:
                return True
                
        return False

    def _detect_divergence(self, price: pd.Series, rsi: pd.Series) -> Optional[str]:
        """
        检测价格与RSI的背离
        
        Args:
            price: 价格序列
            rsi: RSI序列
            
        Returns:
            Optional[str]: 'bullish' (正背离), 'bearish' (负背离), or None
        """
        # 寻找最近的两个价格低点和RSI低点
        price_low1_idx = price.iloc[-30:-15].idxmin()
        price_low2_idx = price.iloc[-15:].idxmin()
        
        rsi_low1 = rsi[price_low1_idx]
        rsi_low2 = rsi[price_low2_idx]
        
        # 正背离：价格创新低，RSI未创新低
        if price[price_low2_idx] < price[price_low1_idx] and rsi_low2 > rsi_low1:
            return "bullish"
            
        # 寻找最近的两个价格高点和RSI高点
        price_high1_idx = price.iloc[-30:-15].idxmax()
        price_high2_idx = price.iloc[-15:].idxmax()
        
        rsi_high1 = rsi[price_high1_idx]
        rsi_high2 = rsi[price_high2_idx]
        
        # 负背离：价格创新高，RSI未创新高
        if price[price_high2_idx] > price[price_high1_idx] and rsi_high2 < rsi_high1:
            return "bearish"
            
        return None

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI的原始评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: RSI评分 (0-100)
        """
        if not self.has_result():
            self.calculate(data)
            
        result = self._result
        
        # 1. RSI位置评分
        rsi = result[f"rsi_{self.period}"]
        position_score = rsi.copy()
        
        # 2. 多周期一致性评分
        consistency_score = self._calculate_multi_period_consistency()
        
        # 3. 形态评分
        pattern_score = self._calculate_pattern_score(data)
        
        # 4. RSI动量评分
        momentum_score = result["rsi_momentum"].apply(lambda x: 50 + x * 10).clip(0, 100)
        
        # 综合评分
        # 权重可以根据策略调整
        weights = {
            "position": 0.4,
            "consistency": 0.3,
            "pattern": 0.2,
            "momentum": 0.1
        }
        
        raw_score = (
            position_score * weights["position"] +
            consistency_score * weights["consistency"] +
            pattern_score * weights["pattern"] +
            momentum_score * weights["momentum"]
        )
        
        return raw_score.clip(0, 100)

    def _calculate_multi_period_consistency(self) -> pd.Series:
        """
        计算多周期RSI的一致性
        
        Returns:
            pd.Series: 一致性评分
        """
        result = self._result
        
        # 获取所有RSI列
        rsi_cols = [f"rsi_{p}" for p in self.multi_periods]
        
        # 计算短期和长期RSI的差异
        short_rsi = result[rsi_cols[0]]
        long_rsi = result[rsi_cols[-1]]
        
        # 趋势方向一致性
        short_trend = short_rsi.diff().apply(np.sign)
        long_trend = long_rsi.diff().apply(np.sign)
        
        trend_consistency = (short_trend == long_trend)
        
        # 位置一致性
        short_pos = (short_rsi > 50)
        long_pos = (long_rsi > 50)
        
        pos_consistency = (short_pos == long_pos)
        
        # 综合一致性
        consistency = (trend_consistency & pos_consistency).astype(int) * 50 + 25
        
        return consistency

    def _calculate_pattern_score(self, data: pd.DataFrame) -> pd.Series:
        """
        根据识别的形态计算评分
        
        Args:
            data: 输入数据
        
        Returns:
            pd.Series: 形态评分
        """
        patterns = self.identify_patterns(data)
        score = pd.Series(50.0, index=data.index) # 默认中性评分
        
        if not patterns:
            return score
            
        last_date = data.index[-1]
        
        # 根据形态调整分数
        if "RSI正背离" in patterns or "W底形态" in patterns:
            score.loc[last_date] = 80.0
        elif "RSI负背离" in patterns or "M顶形态" in patterns:
            score.loc[last_date] = 20.0
        elif "看涨失败摆动" in patterns:
            score.loc[last_date] = 70.0
        elif "看跌失败摆动" in patterns:
            score.loc[last_date] = 30.0
            
        return score

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成RSI交易信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含买入和卖出信号的DataFrame
        """
        if not self.has_result():
            self.calculate(data)
            
        result = self._result
        rsi = result[f"rsi_{self.period}"]
        
        # 动态阈值
        oversold = result["oversold_threshold"]
        overbought = result["overbought_threshold"]
        
        # 基本信号：RSI上穿超卖线为买入，下穿超买线为卖出
        buy_signal = (rsi.shift(1) < oversold.shift(1)) & (rsi > oversold)
        sell_signal = (rsi.shift(1) > overbought.shift(1)) & (rsi < overbought)
        
        # 形态信号
        patterns = self.identify_patterns(data)
        last_date = data.index[-1]
        
        # 如果有强烈的看涨形态，生成买入信号
        if "RSI正背离" in patterns or "W底形态" in patterns or "看涨失败摆动" in patterns:
            buy_signal.loc[last_date] = True
            
        # 如果有强烈的看跌形态，生成卖出信号
        if "RSI负背离" in patterns or "M顶形态" in patterns or "看跌失败摆动" in patterns:
            sell_signal.loc[last_date] = True
            
        signals = pd.DataFrame({
            "buy_signal": buy_signal,
            "sell_signal": sell_signal
        }, index=data.index)
        
        # 计算信号强度
        signals["strength"] = self._calculate_signal_strength(result, signals)
        
        return signals

    def _calculate_signal_strength(self, result: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        计算信号强度
        
        Args:
            result: 指标计算结果
            signals: 交易信号
            
        Returns:
            pd.Series: 信号强度
        """
        strength = pd.Series(SignalStrength.NEUTRAL.value, index=result.index)
        
        # 买入信号强度
        buy_indices = signals[signals["buy_signal"]].index
        
        for idx in buy_indices:
            rsi_val = result.loc[idx, f"rsi_{self.period}"]
            
            # RSI越低，买入信号越强
            if rsi_val < 15:
                strength.loc[idx] = SignalStrength.STRONG_BUY.value
            elif rsi_val < 25:
                strength.loc[idx] = SignalStrength.BUY.value
            else:
                strength.loc[idx] = SignalStrength.WEAK_BUY.value
                
        # 卖出信号强度
        sell_indices = signals[signals["sell_signal"]].index
        
        for idx in sell_indices:
            rsi_val = result.loc[idx, f"rsi_{self.period}"]
            
            # RSI越高，卖出信号越强
            if rsi_val > 85:
                strength.loc[idx] = SignalStrength.STRONG_SELL.value
            elif rsi_val > 75:
                strength.loc[idx] = SignalStrength.SELL.value
            else:
                strength.loc[idx] = SignalStrength.WEAK_SELL.value

        return strength

    def _get_simple_market_environment(self, data: pd.DataFrame) -> str:
        """
        简单的市场环境评估（避免递归调用）

        Args:
            data: 输入数据

        Returns:
            str: 市场环境类型
        """
        try:
            if 'close' not in data.columns or len(data) < 20:
                return 'neutral'

            # 使用价格数据简单判断市场环境
            close_prices = data['close'].tail(20)
            price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
            price_volatility = close_prices.pct_change().std()

            if price_change > 0.05:
                return 'bullish'
            elif price_change < -0.05:
                return 'bearish'
            elif price_volatility > 0.03:
                return 'volatile'
            else:
                return 'neutral'

        except Exception:
            return 'neutral'

    def get_market_environment(self, data: pd.DataFrame) -> str:
        """
        获取市场环境评估

        Args:
            data: 输入数据

        Returns:
            str: 市场环境类型 ('bullish', 'bearish', 'neutral', 'volatile')
        """
        try:
            # 计算RSI指标
            result = self.calculate(data)

            if result.empty or len(result) < 20:
                return 'neutral'

            # 获取最近的RSI值
            rsi_col = f"rsi_{self.period}"
            if rsi_col not in result.columns:
                return 'neutral'

            recent_rsi = result[rsi_col].tail(20)
            latest_rsi = recent_rsi.iloc[-1]

            # 计算RSI趋势
            rsi_trend = recent_rsi.diff().tail(10).mean()

            # 计算RSI波动性
            rsi_volatility = recent_rsi.std()

            # 市场环境判断逻辑
            if latest_rsi > 70 and rsi_trend > 0:
                return 'bullish'  # 强势上涨市场
            elif latest_rsi < 30 and rsi_trend < 0:
                return 'bearish'  # 弱势下跌市场
            elif rsi_volatility > 15:
                return 'volatile'  # 高波动市场
            elif 40 <= latest_rsi <= 60 and abs(rsi_trend) < 0.5:
                return 'neutral'  # 中性市场
            elif rsi_trend > 1:
                return 'bullish'  # 上升趋势
            elif rsi_trend < -1:
                return 'bearish'  # 下降趋势
            else:
                return 'neutral'  # 默认中性

        except Exception as e:
            logger.warning(f"EnhancedRSI获取市场环境时出错: {e}")
            return 'neutral'

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
