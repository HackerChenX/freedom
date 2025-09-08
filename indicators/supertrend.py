#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超级趋势(SuperTrend)指标

SuperTrend是一个基于ATR的趋势跟踪指标，它在价格图表上显示动态的支撑和阻力线。
该指标结合了平均真实波幅(ATR)和价格的中位数来确定趋势方向。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class SuperTrend(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    超级趋势(SuperTrend)指标
    
    分类：趋势指标
    描述：基于ATR的动态支撑阻力线，用于趋势跟踪
    
    计算公式：
    1. HL2 = (High + Low) / 2
    2. ATR = Average True Range
    3. Upper Band = HL2 + (multiplier * ATR)
    4. Lower Band = HL2 - (multiplier * ATR)
    5. SuperTrend = 根据价格与带线的关系确定
    
    信号解释：
    - 价格在SuperTrend线上方：上升趋势
    - 价格在SuperTrend线下方：下降趋势
    - SuperTrend线颜色变化：趋势转换信号
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0, **kwargs):
        """
        初始化SuperTrend指标
        
        Args:
            period: ATR计算周期，默认10
            multiplier: ATR乘数，默认3.0
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.period = period
        self.multiplier = multiplier
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'period': 10,
            'multiplier': 3.0
        }
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        self.period = kwargs.get('period', self.period)
        self.multiplier = kwargs.get('multiplier', self.multiplier)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算SuperTrend指标
        
        Args:
            data: 包含high、low、close列的DataFrame
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含SuperTrend指标的DataFrame
        """
        try:
            # 验证数据
            if not self._validate_data(data):
                return pd.DataFrame()
            
            df = data.copy()
            
            # 计算HL2（高低价中位数）
            hl2 = (df['high'] + df['low']) / 2
            
            # 计算ATR
            atr = self._calculate_atr(df)
            
            # 计算基础上下轨
            upper_band = hl2 + (self.multiplier * atr)
            lower_band = hl2 - (self.multiplier * atr)
            
            # 计算最终上下轨（考虑前一期的值）
            final_upper_band = pd.Series(index=df.index, dtype=float)
            final_lower_band = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df)):
                if i == 0:
                    final_upper_band.iloc[i] = upper_band.iloc[i]
                    final_lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    # 上轨：如果当前上轨小于前一期上轨或前一期收盘价大于前一期上轨，则使用当前上轨
                    if upper_band.iloc[i] < final_upper_band.iloc[i-1] or df['close'].iloc[i-1] > final_upper_band.iloc[i-1]:
                        final_upper_band.iloc[i] = upper_band.iloc[i]
                    else:
                        final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
                    
                    # 下轨：如果当前下轨大于前一期下轨或前一期收盘价小于前一期下轨，则使用当前下轨
                    if lower_band.iloc[i] > final_lower_band.iloc[i-1] or df['close'].iloc[i-1] < final_lower_band.iloc[i-1]:
                        final_lower_band.iloc[i] = lower_band.iloc[i]
                    else:
                        final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
            
            # 计算SuperTrend线
            supertrend = pd.Series(index=df.index, dtype=float)
            trend_direction = pd.Series(index=df.index, dtype=int)  # 1为上升趋势，-1为下降趋势
            
            for i in range(len(df)):
                if i == 0:
                    if df['close'].iloc[i] <= final_lower_band.iloc[i]:
                        supertrend.iloc[i] = final_upper_band.iloc[i]
                        trend_direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = final_lower_band.iloc[i]
                        trend_direction.iloc[i] = 1
                else:
                    if trend_direction.iloc[i-1] == 1:
                        if df['close'].iloc[i] <= final_lower_band.iloc[i]:
                            supertrend.iloc[i] = final_upper_band.iloc[i]
                            trend_direction.iloc[i] = -1
                        else:
                            supertrend.iloc[i] = final_lower_band.iloc[i]
                            trend_direction.iloc[i] = 1
                    else:  # trend_direction.iloc[i-1] == -1
                        if df['close'].iloc[i] >= final_upper_band.iloc[i]:
                            supertrend.iloc[i] = final_lower_band.iloc[i]
                            trend_direction.iloc[i] = 1
                        else:
                            supertrend.iloc[i] = final_upper_band.iloc[i]
                            trend_direction.iloc[i] = -1
            
            # 添加到结果DataFrame
            df['hl2'] = hl2
            df['atr'] = atr
            df['upper_band'] = upper_band
            df['lower_band'] = lower_band
            df['final_upper_band'] = final_upper_band
            df['final_lower_band'] = final_lower_band
            df['supertrend'] = supertrend
            df['trend_direction'] = trend_direction
            
            # 计算信号
            df['st_signal'] = self._generate_signals(df)
            
            return df
            
        except Exception as e:
            logger.error(f"SuperTrend指标计算失败: {e}")
            return pd.DataFrame()
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        计算平均真实波幅(ATR)
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            pd.Series: ATR序列
        """
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR（使用简单移动平均）
        atr = true_range.rolling(window=self.period).mean()
        
        return atr
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含SuperTrend数据的DataFrame
            
        Returns:
            pd.Series: 交易信号
        """
        signals = pd.Series(0, index=df.index)
        trend_direction = df['trend_direction']
        
        # 趋势转换信号
        trend_change = trend_direction.diff()
        
        # 买入信号：趋势从下降转为上升
        buy_signal = trend_change == 2  # 从-1变为1
        signals[buy_signal] = 1
        
        # 卖出信号：趋势从上升转为下降
        sell_signal = trend_change == -2  # 从1变为-1
        signals[sell_signal] = -1
        
        return signals
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取最新的交易信号
        
        Args:
            data: 计算后的数据
            
        Returns:
            Dict[str, Any]: 信号信息
        """
        if data.empty or 'st_signal' not in data.columns:
            return {'signal': 0, 'strength': 0, 'description': '无信号'}
        
        latest_signal = data['st_signal'].iloc[-1]
        latest_trend = data['trend_direction'].iloc[-1] if 'trend_direction' in data.columns else 0
        latest_close = data['close'].iloc[-1]
        latest_supertrend = data['supertrend'].iloc[-1] if 'supertrend' in data.columns else 0
        
        # 计算价格与SuperTrend线的距离作为信号强度
        if latest_supertrend != 0:
            distance_ratio = abs(latest_close - latest_supertrend) / latest_supertrend
            strength = min(distance_ratio * 10, 1.0)  # 标准化强度
        else:
            strength = 0
        
        trend_desc = "上升趋势" if latest_trend == 1 else "下降趋势" if latest_trend == -1 else "未知趋势"
        
        if latest_signal == 1:
            return {
                'signal': 1,
                'strength': strength,
                'description': f'买入信号：趋势转为上升，当前{trend_desc}'
            }
        elif latest_signal == -1:
            return {
                'signal': -1,
                'strength': strength,
                'description': f'卖出信号：趋势转为下降，当前{trend_desc}'
            }
        else:
            return {
                'signal': 0,
                'strength': 0,
                'description': f'趋势持续，当前{trend_desc}'
            }
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """获取指标模式信息"""
        return {
            'name': 'SUPERTREND',
            'description': '超级趋势指标',
            'type': 'trend',
            'parameters': {
                'period': self.period,
                'multiplier': self.multiplier
            }
        }

    # 实现BaseIndicator的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """核心计算逻辑"""
        return self.calculate(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if 'trend_direction' not in data.columns:
            return pd.Series(50.0, index=data.index)

        # 基于趋势方向计算评分
        trend_direction = data['trend_direction']
        score = pd.Series(50.0, index=data.index)
        score[trend_direction == 1] = 75.0  # 上升趋势
        score[trend_direction == -1] = 25.0  # 下降趋势

        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取技术形态"""
        patterns = pd.DataFrame(index=data.index)

        if 'trend_direction' in data.columns:
            trend = data['trend_direction']
            # 趋势状态形态
            patterns['ST_上升趋势'] = trend == 1
            patterns['ST_下降趋势'] = trend == -1
            # 趋势转换形态
            trend_change = trend.diff()
            patterns['ST_转为上升'] = trend_change == 2  # 从-1变为1
            patterns['ST_转为下降'] = trend_change == -2  # 从1变为-1
            # 趋势持续形态
            patterns['ST_趋势延续'] = (trend == trend.shift(1)) & (trend != 0)

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.0

        # 基于趋势稳定性计算置信度
        if len(score) >= 5:
            trend_stability = (score.rolling(5).std().iloc[-1] < 10)  # 趋势稳定
            confidence = 0.8 if trend_stability else 0.5
        else:
            confidence = 0.5

        pattern_strength = min(len(patterns) * 0.1, 0.3)
        return min(confidence + pattern_strength, 1.0)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return self.period + 10
