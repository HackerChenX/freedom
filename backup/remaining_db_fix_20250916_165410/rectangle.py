"""
矩形形态识别指标
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class Rectangle(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    矩形形态识别指标
    
    矩形形态是一种整理形态，特征是：
    1. 价格在水平的支撑线和阻力线之间震荡
    2. 成交量逐渐萎缩
    3. 突破时成交量放大
    """
    
    def __init__(self, period: int = 20, tolerance: float = 0.02):
        """
        初始化矩形形态识别指标

        Args:
            period: 计算周期
            tolerance: 价格容忍度
        """
        self.name = "RECTANGLE"
        self.period = period
        self.tolerance = tolerance
        self._parameters = {
            'period': period,
            'tolerance': tolerance,
            'price_col': 'close'
        }
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算矩形形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含矩形形态信号的DataFrame
        """
        try:
            if len(data) < self.period:
                return pd.DataFrame()
            
            result = data.copy()
            
            # 计算支撑线和阻力线
            result['support'] = result['low'].rolling(window=self.period).min()
            result['resistance'] = result['high'].rolling(window=self.period).max()
            
            # 计算矩形形态
            result['rectangle'] = self._identify_rectangle(result)
            
            # 计算突破信号
            result['breakout_up'] = self._identify_breakout_up(result)
            result['breakout_down'] = self._identify_breakout_down(result)
            
            return result
            
        except Exception as e:
            logger.error(f"矩形形态计算失败: {e}")
            return pd.DataFrame()
    
    def _identify_rectangle(self, data: pd.DataFrame) -> pd.Series:
        """识别矩形形态"""
        try:
            rectangle = pd.Series(False, index=data.index)
            
            for i in range(self.period, len(data)):
                window = data.iloc[i-self.period:i]
                
                # 计算支撑线和阻力线的稳定性
                support_level = window['low'].min()
                resistance_level = window['high'].max()
                
                # 检查价格是否在矩形范围内震荡
                price_range = resistance_level - support_level
                if price_range > 0:
                    # 计算价格在范围内的比例
                    in_range_count = 0
                    for j in range(len(window)):
                        low = window['low'].iloc[j]
                        high = window['high'].iloc[j]
                        
                        if (abs(low - support_level) <= price_range * self.tolerance or
                            abs(high - resistance_level) <= price_range * self.tolerance or
                            (low >= support_level and high <= resistance_level)):
                            in_range_count += 1
                    
                    # 如果大部分价格都在矩形范围内，则认为是矩形形态
                    if in_range_count / len(window) >= 0.7:
                        rectangle.iloc[i] = True
            
            return rectangle
            
        except Exception as e:
            logger.error(f"矩形形态识别失败: {e}")
            return pd.Series(False, index=data.index)
    
    def _identify_breakout_up(self, data: pd.DataFrame) -> pd.Series:
        """识别向上突破"""
        try:
            breakout_up = pd.Series(False, index=data.index)
            
            for i in range(1, len(data)):
                if (data['rectangle'].iloc[i-1] and 
                    data['close'].iloc[i] > data['resistance'].iloc[i-1] * (1 + self.tolerance)):
                    breakout_up.iloc[i] = True
            
            return breakout_up
            
        except Exception as e:
            logger.error(f"向上突破识别失败: {e}")
            return pd.Series(False, index=data.index)
    
    def _identify_breakout_down(self, data: pd.DataFrame) -> pd.Series:
        """识别向下突破"""
        try:
            breakout_down = pd.Series(False, index=data.index)
            
            for i in range(1, len(data)):
                if (data['rectangle'].iloc[i-1] and 
                    data['close'].iloc[i] < data['support'].iloc[i-1] * (1 - self.tolerance)):
                    breakout_down.iloc[i] = True
            
            return breakout_down
            
        except Exception as e:
            logger.error(f"向下突破识别失败: {e}")
            return pd.Series(False, index=data.index)
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取矩形形态交易信号
        
        Args:
            data: 计算结果数据
            
        Returns:
            包含交易信号的字典
        """
        try:
            if data.empty:
                return {'signal': 'HOLD', 'strength': 0, 'message': '数据不足'}
            
            # 获取最新信号
            latest_rectangle = data['rectangle'].iloc[-1] if 'rectangle' in data.columns else False
            latest_breakout_up = data['breakout_up'].iloc[-1] if 'breakout_up' in data.columns else False
            latest_breakout_down = data['breakout_down'].iloc[-1] if 'breakout_down' in data.columns else False
            
            if latest_breakout_up:
                return {
                    'signal': 'BUY',
                    'strength': 0.8,
                    'message': '矩形形态向上突破，建议买入'
                }
            elif latest_breakout_down:
                return {
                    'signal': 'SELL',
                    'strength': 0.8,
                    'message': '矩形形态向下突破，建议卖出'
                }
            elif latest_rectangle:
                return {
                    'signal': 'HOLD',
                    'strength': 0.6,
                    'message': '处于矩形整理形态，等待突破'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'strength': 0.5,
                    'message': '未检测到矩形形态'
                }
                
        except Exception as e:
            logger.error(f"矩形形态信号生成失败: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'message': f'信号生成失败: {e}'}
    
    def minimum_periods(self) -> int:
        """
        矩形形态指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return self.period + 5

    # ==================== BaseIndicator抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self.calculate(data, *args, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        if len(data) < self.minimum_periods():
            return pd.Series(50.0, index=data.index)

        # 计算矩形形态识别结果
        result = self.calculate(data)

        # 基于矩形形态计算评分
        score = pd.Series(50.0, index=data.index)  # 默认中性评分

        if isinstance(result, pd.DataFrame) and not result.empty:
            if 'breakout_up' in result.columns:
                score[result['breakout_up']] = 80.0  # 向上突破，看涨
            if 'breakout_down' in result.columns:
                score[result['breakout_down']] = 20.0  # 向下突破，看跌
            if 'rectangle' in result.columns:
                score[result['rectangle']] = 60.0  # 矩形整理，轻微看涨

        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        if len(data) < self.minimum_periods():
            return pd.DataFrame(index=data.index)

        return self.calculate(data)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """抽象基类要求的置信度方法"""
        base_confidence = 0.7  # 矩形形态相对可靠

        # 基于形态数量调整置信度
        if patterns:
            pattern_bonus = min(0.2, len(patterns) * 0.05)
            base_confidence += pattern_bonus

        return min(1.0, base_confidence)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value
                if key == 'period':
                    self.period = value
                elif key == 'tolerance':
                    self.tolerance = value
