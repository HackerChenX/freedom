#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
真实技术指标计算模块

实现RSI、MACD、KDJ、BOLL、MA、EMA等技术指标的真实计算
以及相应的形态识别算法，用于反向验证测试
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class TechnicalIndicators_Indicators:
    """技术指标计算器"""

    def __init__(self):
        """初始化计算器"""
        pass

    def calculate_rsi_Indicators(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算RSI指标

        Args:
            data: 包含close价格的DataFrame
            period: 计算周期，默认14

        Returns:
            RSI值序列
        """
        close = data['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd_Indicators(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD指标

        Args:
            data: 包含close价格的DataFrame
            fast: 快线周期，默认12
            slow: 慢线周期，默认26
            signal: 信号线周期，默认9

        Returns:
            包含DIF、DEA、MACD的字典
        """
        close = data['close']

        # 计算EMA
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()

        # 计算DIF (快线-慢线)
        dif = ema_fast - ema_slow

        # 计算DEA (DIF的EMA)
        dea = dif.ewm(span=signal).mean()

        # 计算MACD柱状图
        macd = (dif - dea) * 2

        return {
            'DIF': dif,
            'DEA': dea,
            'MACD': macd
        }

    def calculate_kdj_Indicators(self, data: pd.DataFrame, period: int = 9, k_period: int = 3, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        计算KDJ指标

        Args:
            data: 包含high、low、close价格的DataFrame
            period: RSV计算周期，默认9
            k_period: K值平滑周期，默认3
            d_period: D值平滑周期，默认3

        Returns:
            包含K、D、J的字典
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算RSV
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

        # 计算K值 (RSV的移动平均)
        k = rsv.ewm(alpha=1/k_period).mean()

        # 计算D值 (K值的移动平均)
        d = k.ewm(alpha=1/d_period).mean()

        # 计算J值
        j = 3 * k - 2 * d

        return {
            'K': k,
            'D': d,
            'J': j
        }

    def calculate_bollinger_bands_Indicators(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带指标

        Args:
            data: 包含close价格的DataFrame
            period: 移动平均周期，默认20
            std_dev: 标准差倍数，默认2.0

        Returns:
            包含UPPER、MIDDLE、LOWER的字典
        """
        close = data['close']

        # 计算中轨 (移动平均)
        middle = close.rolling(window=period).mean()

        # 计算标准差
        std = close.rolling(window=period).std()

        # 计算上轨和下轨
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'UPPER': upper,
            'MIDDLE': middle,
            'LOWER': lower
        }

    def calculate_ma_Indicators(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> Dict[str, pd.Series]:
        """
        计算移动平均线

        Args:
            data: 包含close价格的DataFrame
            periods: 计算周期列表，默认[5, 10, 20, 60]

        Returns:
            包含各周期MA的字典
        """
        close = data['close']
        ma_dict = {}

        for period in periods:
            ma_dict[f'MA{period}'] = close.rolling(window=period).mean()

        return ma_dict

    def calculate_ema_Indicators(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> Dict[str, pd.Series]:
        """
        计算指数移动平均线

        Args:
            data: 包含close价格的DataFrame
            periods: 计算周期列表，默认[5, 10, 20, 60]

        Returns:
            包含各周期EMA的字典
        """
        close = data['close']
        ema_dict = {}

        for period in periods:
            ema_dict[f'EMA{period}'] = close.ewm(span=period).mean()

        return ema_dict


class PatternRecognizer:
    """技术形态识别器"""

    def __init__(self):
        """初始化识别器"""
        self.indicators = TechnicalIndicators_Indicators()

    def detect_rsi_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        检测RSI形态

        Args:
            data: 价格数据

        Returns:
            检测到的形态字典
        """
        rsi = self.indicators.calculate_rsi_Indicators(data)
        patterns = {}

        if len(rsi) < 2:
            return {pattern: False for pattern in ['RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'RSI_GOLDEN_CROSS', 'RSI_DEATH_CROSS', 'RSI_DIVERGENCE']}

        # RSI超买 (RSI > 70)
        patterns['RSI_OVERBOUGHT'] = rsi.iloc[-1] > 70

        # RSI超卖 (RSI < 30)
        patterns['RSI_OVERSOLD'] = rsi.iloc[-1] < 30

        # RSI金叉 (RSI从下方突破50)
        patterns['RSI_GOLDEN_CROSS'] = self._detect_golden_cross_Technical_Indicators(rsi, 50)

        # RSI死叉 (RSI从上方跌破50)
        patterns['RSI_DEATH_CROSS'] = self._detect_death_cross_Technical_Indicators(rsi, 50)

        # RSI背离 (价格创新高但RSI不创新高，或价格创新低但RSI不创新低)
        patterns['RSI_DIVERGENCE'] = self._detect_rsi_divergence_Technical_Indicators(data, rsi)

        return patterns

    def detect_macd_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        检测MACD形态

        Args:
            data: 价格数据

        Returns:
            检测到的形态字典
        """
        macd_data = self.indicators.calculate_macd_Indicators(data)
        dif = macd_data['DIF']
        dea = macd_data['DEA']
        macd = macd_data['MACD']
        patterns = {}

        if len(dif) < 2:
            return {pattern: False for pattern in ['MACD_GOLDEN_CROSS', 'MACD_DEATH_CROSS', 'MACD_ABOVE_ZERO_GOLDEN', 'MACD_BELOW_ZERO_DEATH', 'MACD_HISTOGRAM_DIVERGENCE']}

        # MACD金叉 (DIF上穿DEA)
        patterns['MACD_GOLDEN_CROSS'] = self._detect_line_cross(dif, dea, 'golden')

        # MACD死叉 (DIF下穿DEA)
        patterns['MACD_DEATH_CROSS'] = self._detect_line_cross(dif, dea, 'death')

        # MACD零轴上金叉
        patterns['MACD_ABOVE_ZERO_GOLDEN'] = patterns['MACD_GOLDEN_CROSS'] and dif.iloc[-1] > 0

        # MACD零轴下死叉
        patterns['MACD_BELOW_ZERO_DEATH'] = patterns['MACD_DEATH_CROSS'] and dif.iloc[-1] < 0

        # MACD柱状图背离
        patterns['MACD_HISTOGRAM_DIVERGENCE'] = self._detect_macd_histogram_divergence(data, macd)

        return patterns

    def detect_kdj_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        检测KDJ形态

        Args:
            data: 价格数据

        Returns:
            检测到的形态字典
        """
        kdj_data = self.indicators.calculate_kdj_Indicators(data)
        k = kdj_data['K']
        d = kdj_data['D']
        j = kdj_data['J']
        patterns = {}

        if len(k) < 2:
            return {pattern: False for pattern in ['KDJ_GOLDEN_CROSS', 'KDJ_DEATH_CROSS', 'KDJ_OVERBOUGHT', 'KDJ_OVERSOLD', 'KDJ_BLUNT']}

        # KDJ金叉 (K线上穿D线)
        patterns['KDJ_GOLDEN_CROSS'] = self._detect_line_cross(k, d, 'golden')

        # KDJ死叉 (K线下穿D线)
        patterns['KDJ_DEATH_CROSS'] = self._detect_line_cross(k, d, 'death')

        # KDJ超买 (K、D、J都大于80)
        patterns['KDJ_OVERBOUGHT'] = k.iloc[-1] > 80 and d.iloc[-1] > 80 and j.iloc[-1] > 80

        # KDJ超卖 (K、D、J都小于20)
        patterns['KDJ_OVERSOLD'] = k.iloc[-1] < 20 and d.iloc[-1] < 20 and j.iloc[-1] < 20

        # KDJ钝化 (在高位或低位长期横盘)
        patterns['KDJ_BLUNT'] = self._detect_kdj_blunt(k, d, j)

        return patterns

    def detect_boll_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        检测BOLL形态

        Args:
            data: 价格数据

        Returns:
            检测到的形态字典
        """
        boll_data = self.indicators.calculate_bollinger_bands_Indicators(data)
        upper = boll_data['UPPER']
        middle = boll_data['MIDDLE']
        lower = boll_data['LOWER']
        close = data['close']
        patterns = {}

        if len(close) < 2:
            return {pattern: False for pattern in ['BOLL_UPPER_BREAKOUT', 'BOLL_LOWER_BREAKOUT', 'BOLL_SQUEEZE', 'BOLL_EXPANSION', 'BOLL_MIDDLE_SUPPORT']}

        # BOLL上轨突破
        patterns['BOLL_UPPER_BREAKOUT'] = close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2]

        # BOLL下轨突破
        patterns['BOLL_LOWER_BREAKOUT'] = close.iloc[-1] < lower.iloc[-1] and close.iloc[-2] >= lower.iloc[-2]

        # BOLL收口 (上下轨距离缩小)
        patterns['BOLL_SQUEEZE'] = self._detect_boll_squeeze(upper, lower)

        # BOLL开口 (上下轨距离扩大)
        patterns['BOLL_EXPANSION'] = self._detect_boll_expansion(upper, lower)

        # BOLL中轨支撑/阻力
        patterns['BOLL_MIDDLE_SUPPORT'] = self._detect_boll_middle_support(close, middle)

        return patterns

    def detect_ma_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        检测MA形态

        Args:
            data: 价格数据

        Returns:
            检测到的形态字典
        """
        ma_data = self.indicators.calculate_ma_Indicators(data, [5, 20])
        ma5 = ma_data['MA5']
        ma20 = ma_data['MA20']
        patterns = {}

        if len(ma5) < 2:
            return {pattern: False for pattern in ['MA_GOLDEN_CROSS', 'MA_DEATH_CROSS', 'MA_BULLISH_ALIGNMENT', 'MA_BEARISH_ALIGNMENT', 'MA_SUPPORT']}

        # MA金叉 (短期MA上穿长期MA)
        patterns['MA_GOLDEN_CROSS'] = self._detect_line_cross(ma5, ma20, 'golden')

        # MA死叉 (短期MA下穿长期MA)
        patterns['MA_DEATH_CROSS'] = self._detect_line_cross(ma5, ma20, 'death')

        # MA多头排列
        patterns['MA_BULLISH_ALIGNMENT'] = ma5.iloc[-1] > ma20.iloc[-1]

        # MA空头排列
        patterns['MA_BEARISH_ALIGNMENT'] = ma5.iloc[-1] < ma20.iloc[-1]

        # MA支撑
        patterns['MA_SUPPORT'] = self._detect_ma_support(data['close'], ma20)

        return patterns

    def detect_ema_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        检测EMA形态

        Args:
            data: 价格数据

        Returns:
            检测到的形态字典
        """
        ema_data = self.indicators.calculate_ema_Indicators(data, [5, 20])
        ema5 = ema_data['EMA5']
        ema20 = ema_data['EMA20']
        patterns = {}

        if len(ema5) < 2:
            return {pattern: False for pattern in ['EMA_GOLDEN_CROSS', 'EMA_DEATH_CROSS', 'EMA_TREND_CONFIRMATION', 'EMA_DIVERGENCE', 'EMA_SUPPORT_RESISTANCE']}

        # EMA金叉
        patterns['EMA_GOLDEN_CROSS'] = self._detect_line_cross(ema5, ema20, 'golden')

        # EMA死叉
        patterns['EMA_DEATH_CROSS'] = self._detect_line_cross(ema5, ema20, 'death')

        # EMA趋势确认
        patterns['EMA_TREND_CONFIRMATION'] = self._detect_ema_trend_confirmation(ema5, ema20)

        # EMA背离
        patterns['EMA_DIVERGENCE'] = self._detect_ema_divergence(data, ema5)

        # EMA支撑阻力
        patterns['EMA_SUPPORT_RESISTANCE'] = self._detect_ema_support_resistance(data['close'], ema20)

        return patterns

    # 辅助检测方法
    def _detect_golden_cross_Technical_Indicators(self, series: pd.Series, threshold: float) -> bool:
        """检测金叉（从下方突破阈值）"""
        if len(series) < 5:
            return False

        # 检查最近几天是否有从下方突破的情况
        for i in range(-3, 0):  # 检查最近3天
            if (series.iloc[i-1] <= threshold and series.iloc[i] > threshold):
                return True

        # 检查当前是否刚好在阈值附近且呈上升趋势
        if (series.iloc[-1] > threshold and
            series.iloc[-2] <= threshold and
            series.iloc[-1] > series.iloc[-2]):
            return True

        return False

    def _detect_death_cross_Technical_Indicators(self, series: pd.Series, threshold: float) -> bool:
        """检测死叉（从上方跌破阈值）"""
        if len(series) < 5:
            return False

        # 检查最近几天是否有从上方跌破的情况
        for i in range(-3, 0):  # 检查最近3天
            if (series.iloc[i-1] >= threshold and series.iloc[i] < threshold):
                return True

        # 检查当前是否刚好在阈值附近且呈下降趋势
        if (series.iloc[-1] < threshold and
            series.iloc[-2] >= threshold and
            series.iloc[-1] < series.iloc[-2]):
            return True

        return False

    def _detect_line_cross(self, fast_line: pd.Series, slow_line: pd.Series, cross_type: str) -> bool:
        """检测两线交叉"""
        if len(fast_line) < 2 or len(slow_line) < 2:
            return False

        if cross_type == 'golden':
            # 金叉：快线从下方穿越慢线
            return (fast_line.iloc[-2] <= slow_line.iloc[-2] and
                   fast_line.iloc[-1] > slow_line.iloc[-1])
        elif cross_type == 'death':
            # 死叉：快线从上方穿越慢线
            return (fast_line.iloc[-2] >= slow_line.iloc[-2] and
                   fast_line.iloc[-1] < slow_line.iloc[-1])
        return False

    def _detect_rsi_divergence_Technical_Indicators(self, data: pd.DataFrame, rsi: pd.Series) -> bool:
        """检测RSI背离"""
        if len(data) < 20 or len(rsi) < 20:
            return False

        # 寻找价格和RSI的峰值
        close_prices = data['close'].iloc[-20:]
        rsi_values = rsi.iloc[-20:]

        # 找到价格的两个峰值
        price_peaks = []
        for i in range(2, len(close_prices)-2):
            if (close_prices.iloc[i] > close_prices.iloc[i-1] and
                close_prices.iloc[i] > close_prices.iloc[i+1] and
                close_prices.iloc[i] > close_prices.iloc[i-2] and
                close_prices.iloc[i] > close_prices.iloc[i+2]):
                price_peaks.append((i, close_prices.iloc[i]))

        # 找到RSI的对应峰值
        rsi_peaks = []
        for i in range(2, len(rsi_values)-2):
            if (rsi_values.iloc[i] > rsi_values.iloc[i-1] and
                rsi_values.iloc[i] > rsi_values.iloc[i+1]):
                rsi_peaks.append((i, rsi_values.iloc[i]))

        # 检查背离：如果有两个价格峰值，第二个更高，但对应的RSI峰值更低
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            # 取最近的两个峰值
            price_peak1 = price_peaks[-2]
            price_peak2 = price_peaks[-1]

            # 找到对应时间的RSI值
            rsi_at_peak1 = rsi_values.iloc[price_peak1[0]]
            rsi_at_peak2 = rsi_values.iloc[price_peak2[0]]

            # 顶背离：价格创新高，RSI不创新高
            if price_peak2[1] > price_peak1[1] and rsi_at_peak2 < rsi_at_peak1:
                return True

        # 简化检测：价格总体上涨但RSI下降
        price_trend = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        rsi_trend = rsi_values.iloc[-1] - rsi_values.iloc[0]

        return (price_trend > 0.1 and rsi_trend < -10) or (price_trend < -0.1 and rsi_trend > 10)

    def _detect_macd_histogram_divergence(self, data: pd.DataFrame, macd: pd.Series) -> bool:
        """检测MACD柱状图背离"""
        if len(data) < 10 or len(macd) < 10:
            return False

        # 简化的背离检测
        price_recent = data['close'].iloc[-10:]
        macd_recent = macd.iloc[-10:]

        price_trend = (price_recent.iloc[-1] - price_recent.iloc[0]) / price_recent.iloc[0]
        macd_trend = macd_recent.iloc[-1] - macd_recent.iloc[0]

        return (price_trend > 0.05 and macd_trend < -0.1) or (price_trend < -0.05 and macd_trend > 0.1)

    def _detect_kdj_blunt(self, k: pd.Series, d: pd.Series, j: pd.Series) -> bool:
        """检测KDJ钝化"""
        if len(k) < 5:
            return False

        # 检测是否在高位或低位长期横盘
        recent_k = k.iloc[-5:]
        recent_d = d.iloc[-5:]

        # 高位钝化：K、D值都在80以上且变化不大
        high_blunt = (recent_k.mean() > 80 and recent_d.mean() > 80 and
                     recent_k.std() < 5 and recent_d.std() < 5)

        # 低位钝化：K、D值都在20以下且变化不大
        low_blunt = (recent_k.mean() < 20 and recent_d.mean() < 20 and
                    recent_k.std() < 5 and recent_d.std() < 5)

        return high_blunt or low_blunt

    def _detect_boll_squeeze(self, upper: pd.Series, lower: pd.Series) -> bool:
        """检测布林带收口"""
        if len(upper) < 5:
            return False

        # 计算最近5天的带宽变化
        recent_width = (upper - lower).iloc[-5:]

        # 收口：带宽在缩小
        return recent_width.iloc[-1] < recent_width.iloc[0] * 0.9

    def _detect_boll_expansion(self, upper: pd.Series, lower: pd.Series) -> bool:
        """检测布林带开口"""
        if len(upper) < 5:
            return False

        # 计算最近5天的带宽变化
        recent_width = (upper - lower).iloc[-5:]

        # 开口：带宽在扩大
        return recent_width.iloc[-1] > recent_width.iloc[0] * 1.1

    def _detect_boll_middle_support(self, close: pd.Series, middle: pd.Series) -> bool:
        """检测布林带中轨支撑"""
        if len(close) < 3:
            return False

        # 价格从下方接近中轨并获得支撑
        return (close.iloc[-3] < middle.iloc[-3] and
               close.iloc[-2] <= middle.iloc[-2] and
               close.iloc[-1] > middle.iloc[-1])

    def _detect_ma_support(self, close: pd.Series, ma: pd.Series) -> bool:
        """检测MA支撑"""
        if len(close) < 3:
            return False

        # 价格回调到MA附近获得支撑
        return (close.iloc[-3] > ma.iloc[-3] and
               close.iloc[-2] <= ma.iloc[-2] * 1.02 and
               close.iloc[-1] > ma.iloc[-1])

    def _detect_ema_trend_confirmation(self, ema5: pd.Series, ema20: pd.Series) -> bool:
        """检测EMA趋势确认"""
        if len(ema5) < 3:
            return False

        # 短期EMA持续在长期EMA上方，确认上升趋势
        return (ema5.iloc[-3] > ema20.iloc[-3] and
               ema5.iloc[-2] > ema20.iloc[-2] and
               ema5.iloc[-1] > ema20.iloc[-1])

    def _detect_ema_divergence(self, data: pd.DataFrame, ema: pd.Series) -> bool:
        """检测EMA背离"""
        if len(data) < 10 or len(ema) < 10:
            return False

        price_recent = data['close'].iloc[-10:]
        ema_recent = ema.iloc[-10:]

        price_trend = (price_recent.iloc[-1] - price_recent.iloc[0]) / price_recent.iloc[0]
        ema_trend = (ema_recent.iloc[-1] - ema_recent.iloc[0]) / ema_recent.iloc[0]

        # 背离：价格和EMA趋势相反
        return (price_trend > 0.05 and ema_trend < -0.02) or (price_trend < -0.05 and ema_trend > 0.02)

    def _detect_ema_support_resistance(self, close: pd.Series, ema: pd.Series) -> bool:
        """检测EMA支撑阻力"""
        if len(close) < 3:
            return False

        # 价格在EMA附近获得支撑或遇到阻力
        return (abs(close.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1] < 0.02 and
               abs(close.iloc[-2] - ema.iloc[-2]) / ema.iloc[-2] < 0.02)