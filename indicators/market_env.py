"""
市场环境检测模块

提供市场环境识别和相关功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class MarketEnvironment(Enum):
    """市场环境枚举"""
    BULL_MARKET = "牛市"
    BEAR_MARKET = "熊市"
    SIDEWAYS_MARKET = "震荡市"
    VOLATILE_MARKET = "高波动市"
    BREAKOUT_MARKET = "突破市场"


class MarketDetector:
    """
    市场环境检测器
    
    用于分析市场当前所处的环境，支持多种检测方法
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        初始化市场环境检测器
        
        Args:
            lookback_periods: 各类分析的回看周期，格式为 {'trend': 60, 'volatility': 20, ...}
        """
        self.lookback_periods = lookback_periods or {
            'trend': 60,       # 趋势分析回看周期
            'volatility': 20,  # 波动率分析回看周期
            'volume': 10,      # 成交量分析回看周期
            'momentum': 20,    # 动量分析回看周期
            'breakout': 30     # 突破分析回看周期
        }
    
    def detect_environment(self, data: pd.DataFrame) -> MarketEnvironment:
        """
        检测市场环境
        
        Args:
            data: 输入数据，包含 OHLCV 数据
            
        Returns:
            MarketEnvironment: 检测到的市场环境
        """
        # 确保数据足够
        min_required = max(self.lookback_periods.values())
        if len(data) < min_required:
            logger.warning(f"数据不足，需要至少 {min_required} 条数据，但只有 {len(data)} 条")
            return MarketEnvironment.SIDEWAYS_MARKET
        
        # 计算各项指标
        trend_score = self.calculate_trend_strength(data)
        volatility = self.calculate_volatility(data)
        volume_trend = self.analyze_volume_trend(data) if 'volume' in data.columns else 0
        momentum = self.calculate_momentum(data)
        breakout_score = self.calculate_breakout_score(data)
        
        # 记录关键指标
        logger.debug(f"市场环境分析 - 趋势得分: {trend_score:.2f}, 波动率: {volatility:.2f}, "
                    f"成交量趋势: {volume_trend:.2f}, 动量: {momentum:.2f}, 突破得分: {breakout_score:.2f}")
        
        # 综合判断市场环境
        if trend_score > 0.6 and momentum > 0.5:
            return MarketEnvironment.BULL_MARKET
        elif trend_score < -0.6 and momentum < -0.5:
            return MarketEnvironment.BEAR_MARKET
        elif volatility > 0.25 and abs(trend_score) < 0.3:
            return MarketEnvironment.VOLATILE_MARKET
        elif breakout_score > 0.7:
            return MarketEnvironment.BREAKOUT_MARKET
        else:
            return MarketEnvironment.SIDEWAYS_MARKET
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        计算趋势强度
        
        Args:
            data: 输入数据
            
        Returns:
            float: 趋势强度，范围 [-1, 1]，正值表示上升趋势，负值表示下降趋势
        """
        lookback = self.lookback_periods['trend']
        if len(data) < lookback:
            return 0
        
        recent_data = data.tail(lookback)
        
        # 计算价格变化
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # 计算线性回归斜率
        x = np.arange(len(recent_data))
        y = recent_data['close'].values
        slope, _ = np.polyfit(x, y, 1)
        normalized_slope = slope / recent_data['close'].mean()
        
        # 计算多个时间窗口的移动平均线排列情况
        ma_alignment = self.calculate_ma_alignment(recent_data['close'])
        
        # 综合评分，标准化到 [-1, 1] 区间
        trend_strength = (np.tanh(price_change * 5) * 0.5 + 
                          np.tanh(normalized_slope * 100) * 0.3 + 
                          ma_alignment * 0.2)
        
        return trend_strength
    
    def calculate_ma_alignment(self, price_series: pd.Series) -> float:
        """
        计算均线排列情况
        
        Args:
            price_series: 价格序列
            
        Returns:
            float: 均线排列得分，范围 [-1, 1]，1 表示完美多头排列，-1 表示完美空头排列
        """
        # 计算不同周期的移动平均线
        ma_periods = [5, 10, 20, 60]
        ma_series = {}
        
        for period in ma_periods:
            if len(price_series) >= period:
                ma_series[period] = price_series.rolling(window=period).mean()
        
        if len(ma_series) < 2:
            return 0
        
        # 检查均线排列情况
        last_values = {period: series.iloc[-1] for period, series in ma_series.items()}
        periods = sorted(last_values.keys())
        
        # 多头排列：短期均线在上，长期均线在下
        is_bullish = True
        for i in range(len(periods) - 1):
            if last_values[periods[i]] <= last_values[periods[i+1]]:
                is_bullish = False
                break
        
        # 空头排列：长期均线在上，短期均线在下
        is_bearish = True
        for i in range(len(periods) - 1):
            if last_values[periods[i]] >= last_values[periods[i+1]]:
                is_bearish = False
                break
        
        # 计算排列完整度得分
        if is_bullish:
            # 计算多头排列的完美度
            score = 0
            for i in range(len(periods) - 1):
                # 相邻均线的距离占价格的比例
                distance = (last_values[periods[i]] - last_values[periods[i+1]]) / last_values[periods[i]]
                score += distance
            return min(1, score * 10)  # 标准化到 [0, 1]
            
        elif is_bearish:
            # 计算空头排列的完美度
            score = 0
            for i in range(len(periods) - 1):
                # 相邻均线的距离占价格的比例
                distance = (last_values[periods[i+1]] - last_values[periods[i]]) / last_values[periods[i]]
                score += distance
            return max(-1, -score * 10)  # 标准化到 [-1, 0]
            
        else:
            # 无明确排列，计算倾向性
            short_ma = last_values[periods[0]]  # 最短周期均线
            long_ma = last_values[periods[-1]]  # 最长周期均线
            
            # 短期均线高于长期均线，轻微看涨；反之轻微看跌
            return (short_ma - long_ma) / long_ma * 3  # 弱化的得分
    
    def calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        计算市场波动率
        
        Args:
            data: 输入数据
            
        Returns:
            float: 波动率，值越大表示波动越大
        """
        lookback = self.lookback_periods['volatility']
        if len(data) < lookback:
            return 0
        
        recent_data = data.tail(lookback)
        
        # 日波动率
        daily_returns = recent_data['close'].pct_change().dropna()
        daily_volatility = daily_returns.std()
        
        # 振幅波动率
        amplitude = (recent_data['high'] - recent_data['low']) / recent_data['low']
        amplitude_volatility = amplitude.mean()
        
        # 归一化波动率（基于历史数据）
        return (daily_volatility * 16 + amplitude_volatility) / 2  # 调整为典型范围内的值
    
    def analyze_volume_trend(self, data: pd.DataFrame) -> float:
        """
        分析成交量趋势
        
        Args:
            data: 输入数据
            
        Returns:
            float: 成交量趋势得分，范围 [-1, 1]，正值表示放量，负值表示缩量
        """
        if 'volume' not in data.columns:
            return 0
        
        lookback = self.lookback_periods['volume']
        if len(data) < lookback:
            return 0
        
        recent_data = data.tail(lookback)
        
        # 计算成交量趋势
        volume = recent_data['volume']
        volume_ma5 = volume.rolling(window=5).mean()
        volume_ma10 = volume.rolling(window=10).mean() if len(volume) >= 10 else volume_ma5
        
        # 最近5日平均成交量相对于10日平均的变化
        if len(volume) >= 10:
            volume_change = (volume_ma5.iloc[-1] / volume_ma10.iloc[-1] - 1)
        else:
            # 用当前成交量与5日均量对比
            volume_change = (volume.iloc[-1] / volume_ma5.iloc[-1] - 1)
        
        # 价量配合分析
        price_change = recent_data['close'].pct_change(5).iloc[-1]
        
        # 价升量增、价跌量增得分较高
        volume_price_synergy = 0
        if price_change > 0 and volume_change > 0:
            # 价升量增，看涨
            volume_price_synergy = min(price_change, volume_change) * 2
        elif price_change < 0 and volume_change > 0:
            # 价跌量增，可能是出货，也可能是超跌反弹前兆
            volume_price_synergy = -min(abs(price_change), volume_change)
        
        # 综合评分
        volume_trend_score = np.tanh(volume_change * 3) * 0.7 + volume_price_synergy * 0.3
        
        return volume_trend_score
    
    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """
        计算市场动量
        
        Args:
            data: 输入数据
            
        Returns:
            float: 动量得分，范围 [-1, 1]，正值表示上升动量，负值表示下降动量
        """
        lookback = self.lookback_periods['momentum']
        if len(data) < lookback:
            return 0
        
        recent_data = data.tail(lookback)
        
        # 使用多个时间窗口计算价格变化率
        windows = [5, 10, 20]
        momentum_scores = []
        
        for window in windows:
            if len(recent_data) < window:
                continue
                
            # 计算窗口的价格变化率
            change_rate = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-window]) / recent_data['close'].iloc[-window]
            # 标准化变化率
            norm_change = np.tanh(change_rate * 5)
            momentum_scores.append(norm_change)
        
        # 使用RSI类指标评估动量
        if len(recent_data) >= 14:
            delta = recent_data['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=14).mean().iloc[-1]
            avg_loss = loss.rolling(window=14).mean().iloc[-1]
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                # 将RSI转换为[-1, 1]范围
                rsi_score = (rsi - 50) / 50
                momentum_scores.append(rsi_score)
        
        # 返回平均动量得分
        return np.mean(momentum_scores) if momentum_scores else 0
    
    def calculate_breakout_score(self, data: pd.DataFrame) -> float:
        """
        计算突破特征得分
        
        Args:
            data: 输入数据
            
        Returns:
            float: 突破特征得分，范围 [0, 1]，越高表示突破可能性越大
        """
        lookback = self.lookback_periods['breakout']
        if len(data) < lookback:
            return 0
        
        recent_data = data.tail(lookback)
        
        # 计算近期价格区间
        recent_high = recent_data['high'].tail(5).max()
        recent_low = recent_data['low'].tail(5).min()
        recent_close = recent_data['close'].iloc[-1]
        
        # 计算前期价格区间
        previous_high = recent_data['high'].iloc[:-5].max() if len(recent_data) > 5 else recent_high
        previous_low = recent_data['low'].iloc[:-5].min() if len(recent_data) > 5 else recent_low
        
        # 计算前期区间宽度
        range_width = (previous_high - previous_low) / previous_low
        
        # 盘整检测 - 如果区间窄，更可能是盘整突破
        is_consolidation = range_width < 0.1
        
        # 突破得分
        breakout_score = 0
        
        # 向上突破
        if recent_close > previous_high:
            # 计算突破幅度
            breakout_magnitude = (recent_close - previous_high) / previous_high
            # 突破得分，盘整突破权重更高
            breakout_score = min(breakout_magnitude * (15 if is_consolidation else 10), 1)
            
        # 向下突破
        elif recent_close < previous_low:
            # 计算突破幅度
            breakout_magnitude = (previous_low - recent_close) / previous_low
            # 突破得分，盘整突破权重更高
            breakout_score = min(breakout_magnitude * (15 if is_consolidation else 10), 1)
        
        # 成交量确认，如果有成交量数据
        if 'volume' in recent_data.columns:
            recent_volume = recent_data['volume'].tail(5).mean()
            previous_volume = recent_data['volume'].iloc[:-5].mean() if len(recent_data) > 5 else recent_volume
            
            # 成交量放大确认突破
            if recent_volume > previous_volume * 1.5 and breakout_score > 0:
                volume_confirm = min((recent_volume / previous_volume - 1) * 0.5, 0.5)
                breakout_score = min(breakout_score + volume_confirm, 1)
        
        return breakout_score 