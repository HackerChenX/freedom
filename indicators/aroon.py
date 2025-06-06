#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
阿隆指标(Aroon)

阿隆指标用于识别趋势的开始和结束，通过计算最高价和最低价距离当前的时间来判断趋势强度
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator, PatternResult
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class Aroon(BaseIndicator):
    """
    阿隆指标(Aroon) (Aroon)
    
    分类：趋势类指标
    描述：阿隆指标用于识别趋势的开始和结束，通过计算最高价和最低价距离当前的时间来判断趋势强度
    """
    
    def __init__(self, period: int = 14):
        """
        初始化阿隆指标(Aroon)
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__(name="Aroon", description="阿隆指标，用于识别趋势的开始和结束")
        self.period = period
        
        # 注册Aroon形态
        self._register_aroon_patterns()
        
    def _register_aroon_patterns(self):
        """
        注册Aroon指标形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册Aroon交叉形态
        registry.register(
            pattern_id="AROON_BULLISH_CROSS",
            display_name="Aroon多头交叉",
            description="Aroon Up上穿Aroon Down，表明上升趋势可能开始",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="AROON_BEARISH_CROSS",
            display_name="Aroon空头交叉",
            description="Aroon Down上穿Aroon Up，表明下降趋势可能开始",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        # 注册Aroon极值形态
        registry.register(
            pattern_id="AROON_STRONG_UPTREND",
            display_name="Aroon强势上涨",
            description="Aroon Up > 70 且 Aroon Down < 30，表明强势上升趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="AROON_STRONG_DOWNTREND",
            display_name="Aroon强势下跌",
            description="Aroon Down > 70 且 Aroon Up < 30，表明强势下降趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
        
        # 注册Aroon震荡器穿越形态
        registry.register(
            pattern_id="AROON_OSC_CROSS_ABOVE_ZERO",
            display_name="Aroon震荡器上穿零轴",
            description="Aroon震荡器从下方穿越零轴，表明趋势转为向上",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="AROON_OSC_CROSS_BELOW_ZERO",
            display_name="Aroon震荡器下穿零轴",
            description="Aroon震荡器从上方穿越零轴，表明趋势转为向下",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册Aroon震荡器极值形态
        registry.register(
            pattern_id="AROON_OSC_EXTREME_BULLISH",
            display_name="Aroon震荡器极度看涨",
            description="Aroon震荡器值 > 50，表明极强的上升趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=25.0
        )
        
        registry.register(
            pattern_id="AROON_OSC_EXTREME_BEARISH",
            display_name="Aroon震荡器极度看跌",
            description="Aroon震荡器值 < -50，表明极强的下降趋势",
            indicator_id="AROON",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=-25.0
        )
        
        # 注册Aroon盘整形态
        registry.register(
            pattern_id="AROON_CONSOLIDATION",
            display_name="Aroon盘整",
            description="Aroon Up和Aroon Down都在30以下，表明市场处于盘整状态",
            indicator_id="AROON",
            pattern_type=PatternType.CONSOLIDATION,
            default_strength=PatternStrength.MEDIUM,
            score_impact=0.0
        )
        
        # 注册Aroon转折形态
        registry.register(
            pattern_id="AROON_REVERSAL_BULLISH",
            display_name="Aroon看涨反转",
            description="Aroon Down从高位快速下降，同时Aroon Up从低位快速上升",
            indicator_id="AROON",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="AROON_REVERSAL_BEARISH",
            display_name="Aroon看跌反转",
            description="Aroon Up从高位快速下降，同时Aroon Down从低位快速上升",
            indicator_id="AROON",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0
        )
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Aroon指标
        
        Args:
            df: 包含OHLC数据的DataFrame
                
        Returns:
            包含Aroon指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算阿隆指标(Aroon)
        
        Args:
            df: 包含OHLC数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了Aroon指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算Aroon Up和Aroon Down
        aroon_up = []
        aroon_down = []
        
        for i in range(len(df_copy)):
            if i < self.period - 1:
                aroon_up.append(np.nan)
                aroon_down.append(np.nan)
            else:
                # 获取当前周期内的数据
                period_high = df_copy['high'].iloc[i-self.period+1:i+1]
                period_low = df_copy['low'].iloc[i-self.period+1:i+1]
                
                # 找到最高价和最低价的位置（相对位置）
                high_max_pos = period_high.idxmax()
                low_min_pos = period_low.idxmin()
                
                # 计算距离当前的天数（使用相对位置）
                high_max_relative_idx = period_high.index.get_loc(high_max_pos)
                low_min_relative_idx = period_low.index.get_loc(low_min_pos)
                
                days_since_high = len(period_high) - 1 - high_max_relative_idx
                days_since_low = len(period_low) - 1 - low_min_relative_idx
                
                # 计算Aroon值
                aroon_up_val = ((self.period - days_since_high) / self.period) * 100
                aroon_down_val = ((self.period - days_since_low) / self.period) * 100
                
                aroon_up.append(aroon_up_val)
                aroon_down.append(aroon_down_val)
        
        df_copy['aroon_up'] = aroon_up
        df_copy['aroon_down'] = aroon_down
        
        # 计算Aroon震荡器
        df_copy['aroon_oscillator'] = df_copy['aroon_up'] - df_copy['aroon_down']
        
        # 存储结果
        self._result = df_copy[['aroon_up', 'aroon_down', 'aroon_oscillator']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成阿隆指标(Aroon)交易信号
        
        Args:
            df: 包含价格数据和Aroon指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - aroon_buy_signal: 1=买入信号, 0=无信号
            - aroon_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['aroon_up', 'aroon_down', 'aroon_oscillator']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['aroon_buy_signal'] = 0
        df_copy['aroon_sell_signal'] = 0
        
        # 生成交易信号
        for i in range(1, len(df_copy)):
            # 1. Aroon Up上穿Aroon Down
            if (df_copy['aroon_up'].iloc[i-1] <= df_copy['aroon_down'].iloc[i-1] and 
                df_copy['aroon_up'].iloc[i] > df_copy['aroon_down'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('aroon_buy_signal')] = 1
            
            # 2. Aroon Down上穿Aroon Up
            elif (df_copy['aroon_down'].iloc[i-1] <= df_copy['aroon_up'].iloc[i-1] and 
                  df_copy['aroon_down'].iloc[i] > df_copy['aroon_up'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('aroon_sell_signal')] = 1
            
            # 3. Aroon震荡器穿越零轴
            elif (df_copy['aroon_oscillator'].iloc[i-1] <= 0 and 
                  df_copy['aroon_oscillator'].iloc[i] > 0):
                df_copy.iloc[i, df_copy.columns.get_loc('aroon_buy_signal')] = 1
            
            elif (df_copy['aroon_oscillator'].iloc[i-1] >= 0 and 
                  df_copy['aroon_oscillator'].iloc[i] < 0):
                df_copy.iloc[i, df_copy.columns.get_loc('aroon_sell_signal')] = 1
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Aroon原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算Aroon
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. Aroon上下线交叉评分
        cross_score = self._calculate_aroon_cross_score()
        score += cross_score
        
        # 2. Aroon极值评分
        extreme_score = self._calculate_aroon_extreme_score()
        score += extreme_score
        
        # 3. Aroon趋势评分
        trend_score = self._calculate_aroon_trend_score()
        score += trend_score
        
        # 4. Aroon震荡器评分
        oscillator_score = self._calculate_aroon_oscillator_score()
        score += oscillator_score
        
        # 5. Aroon强度评分
        strength_score = self._calculate_aroon_strength_score()
        score += strength_score
        
        return np.clip(score, 0, 100)
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取AROON指标的所有形态信息
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 包含形态信息的字典列表
        """
        if not self.has_result():
            self.calculate(data)
            
        result = []
        
        # 检查是否有足够的数据
        if len(self._result) < 2:
            return result
        
        aroon_up_col = 'aroon_up'
        aroon_down_col = 'aroon_down'
        aroon_oscillator_col = 'aroon_oscillator'
        
        current_aroon_up = self._result[aroon_up_col].iloc[-1]
        current_aroon_down = self._result[aroon_down_col].iloc[-1]
        previous_aroon_up = self._result[aroon_up_col].iloc[-2]
        previous_aroon_down = self._result[aroon_down_col].iloc[-2]
        
        # 检测阿隆交叉模式
        if len(self._result) >= 2:
            # 阿隆金叉（Aroon_Up上穿Aroon_Down）
            if previous_aroon_up <= previous_aroon_down and current_aroon_up > current_aroon_down:
                strength = 0.8  # 强势信号
                
                pattern_data = {
                    'pattern_id': "AROON_GOLDEN_CROSS",
                    'display_name': f"阿隆金叉（看涨信号）：{current_aroon_up:.1f} > {current_aroon_down:.1f}",
                    'indicator_id': self.name,
                    'strength': strength,
                    'duration': 2,
                    'details': {
                        'aroon_up': float(current_aroon_up),
                        'aroon_down': float(current_aroon_down),
                        'aroon_diff': float(current_aroon_up - current_aroon_down)
                    }
                }
                result.append(pattern_data)
            # 阿隆死叉（Aroon_Up下穿Aroon_Down）
            elif previous_aroon_up >= previous_aroon_down and current_aroon_up < current_aroon_down:
                strength = -0.8  # 强势下跌信号
                
                pattern_data = {
                    'pattern_id': "AROON_DEATH_CROSS",
                    'display_name': f"阿隆死叉（看跌信号）：{current_aroon_up:.1f} < {current_aroon_down:.1f}",
                    'indicator_id': self.name,
                    'strength': strength,
                    'duration': 2,
                    'details': {
                        'aroon_up': float(current_aroon_up),
                        'aroon_down': float(current_aroon_down),
                        'aroon_diff': float(current_aroon_up - current_aroon_down)
                    }
                }
                result.append(pattern_data)
        
        # 检测极端值模式
        if current_aroon_up >= 90:
            strength = 0.9  # 极端强势信号
            
            pattern_data = {
                'pattern_id': "AROON_EXTREME_UP",
                'display_name': f"阿隆上值极端高位（{current_aroon_up:.1f}）",
                'indicator_id': self.name,
                'strength': strength,
                'duration': 3,
                'details': {
                    'aroon_up': float(current_aroon_up),
                    'period': self.period
                }
            }
            result.append(pattern_data)
        elif current_aroon_up >= 80:
            strength = 0.7  # 强势信号
            
            pattern_data = {
                'pattern_id': "AROON_HIGH_UP",
                'display_name': f"阿隆上值高位（{current_aroon_up:.1f}）",
                'indicator_id': self.name,
                'strength': strength,
                'duration': 2,
                'details': {
                    'aroon_up': float(current_aroon_up),
                    'period': self.period
                }
            }
            result.append(pattern_data)
        
        if current_aroon_down >= 90:
            strength = -0.9  # 极端弱势信号
            
            pattern_data = {
                'pattern_id': "AROON_EXTREME_DOWN",
                'display_name': f"阿隆下值极端高位（{current_aroon_down:.1f}）",
                'indicator_id': self.name,
                'strength': strength,
                'duration': 3,
                'details': {
                    'aroon_down': float(current_aroon_down),
                    'period': self.period
                }
            }
            result.append(pattern_data)
        elif current_aroon_down >= 80:
            strength = -0.7  # 弱势信号
            
            pattern_data = {
                'pattern_id': "AROON_HIGH_DOWN",
                'display_name': f"阿隆下值高位（{current_aroon_down:.1f}）",
                'indicator_id': self.name,
                'strength': strength,
                'duration': 2,
                'details': {
                    'aroon_down': float(current_aroon_down),
                    'period': self.period
                }
            }
            result.append(pattern_data)
        
        # 检测趋势延续模式
        if len(self._result) >= 5:
            recent_values = self._result.iloc[-5:]
            
            # 连续上升的趋势
            if all(recent_values[aroon_up_col].iloc[i+1] > recent_values[aroon_up_col].iloc[i] 
                   for i in range(len(recent_values)-1)):
                strength = 0.6  # 中等强度信号
                
                pattern_data = {
                    'pattern_id': "AROON_UP_TREND_CONTINUATION",
                    'display_name': f"阿隆上值连续上升趋势（{current_aroon_up:.1f}）",
                    'indicator_id': self.name,
                    'strength': strength,
                    'duration': 3,
                    'details': {
                        'aroon_up': float(current_aroon_up),
                        'previous_aroon_up': float(previous_aroon_up),
                        'trend_duration': 5
                    }
                }
                result.append(pattern_data)
            # 连续下降的趋势
            elif all(recent_values[aroon_down_col].iloc[i+1] > recent_values[aroon_down_col].iloc[i] 
                   for i in range(len(recent_values)-1)):
                strength = -0.6  # 中等强度下跌信号
                
                pattern_data = {
                    'pattern_id': "AROON_DOWN_TREND_CONTINUATION",
                    'display_name': f"阿隆下值连续上升趋势（{current_aroon_down:.1f}）",
                    'indicator_id': self.name,
                    'strength': strength,
                    'duration': 3,
                    'details': {
                        'aroon_down': float(current_aroon_down),
                        'previous_aroon_down': float(previous_aroon_down),
                        'trend_duration': 5
                    }
                }
                result.append(pattern_data)
        
        # 检测趋势反转模式
        if len(self._result) >= 3:
            current_oscillator = self._result[aroon_oscillator_col].iloc[-1]
            previous_oscillator = self._result[aroon_oscillator_col].iloc[-2]
            pre_previous_oscillator = self._result[aroon_oscillator_col].iloc[-3]
            
            # 正向反转（可能预示上涨趋势开始）
            if pre_previous_oscillator < previous_oscillator < current_oscillator and current_oscillator < 0:
                strength = 0.7  # 强势信号
                
                pattern_data = {
                    'pattern_id': "AROON_POSITIVE_REVERSAL",
                    'display_name': f"阿隆正向反转（{current_oscillator:.1f}）",
                    'indicator_id': self.name,
                    'strength': strength,
                    'duration': 2,
                    'details': {
                        'current_oscillator': float(current_oscillator),
                        'previous_oscillator': float(previous_oscillator),
                        'pre_previous_oscillator': float(pre_previous_oscillator)
                    }
                }
                result.append(pattern_data)
            # 负向反转（可能预示下跌趋势开始）
            elif pre_previous_oscillator > previous_oscillator > current_oscillator and current_oscillator > 0:
                strength = -0.7  # 强势下跌信号
                
                pattern_data = {
                    'pattern_id': "AROON_NEGATIVE_REVERSAL",
                    'display_name': f"阿隆负向反转（{current_oscillator:.1f}）",
                    'indicator_id': self.name,
                    'strength': strength,
                    'duration': 2,
                    'details': {
                        'current_oscillator': float(current_oscillator),
                        'previous_oscillator': float(previous_oscillator),
                        'pre_previous_oscillator': float(pre_previous_oscillator)
                    }
                }
                result.append(pattern_data)
        
        # 检测趋势强度变化
        if current_aroon_up > 50 and previous_aroon_up <= 50:
            strength = 0.5  # 中等强度信号
            
            pattern_data = {
                'pattern_id': "AROON_UP_BREAK_50",
                'display_name': f"阿隆上值突破50（{current_aroon_up:.1f}）",
                'indicator_id': self.name,
                'strength': strength,
                'duration': 2,
                'details': {
                    'aroon_up': float(current_aroon_up),
                    'previous_aroon_up': float(previous_aroon_up)
                }
            }
            result.append(pattern_data)
        elif current_aroon_down > 50 and previous_aroon_down <= 50:
            strength = -0.5  # 中等强度下跌信号
            
            pattern_data = {
                'pattern_id': "AROON_DOWN_BREAK_50",
                'display_name': f"阿隆下值突破50（{current_aroon_down:.1f}）",
                'indicator_id': self.name,
                'strength': strength,
                'duration': 2,
                'details': {
                    'aroon_down': float(current_aroon_down),
                    'previous_aroon_down': float(previous_aroon_down)
                }
            }
            result.append(pattern_data)
        
        return result
    
    def _calculate_aroon_cross_score(self) -> pd.Series:
        """
        计算Aroon上下线交叉评分
        
        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=self._result.index)
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        
        # Aroon Up上穿Aroon Down+25分
        aroon_up_cross_up = crossover(aroon_up, aroon_down)
        cross_score += aroon_up_cross_up * 25
        
        # Aroon Down上穿Aroon Up-25分
        aroon_down_cross_up = crossover(aroon_down, aroon_up)
        cross_score -= aroon_down_cross_up * 25
        
        # Aroon Up在Aroon Down上方+10分
        aroon_up_above = aroon_up > aroon_down
        cross_score += aroon_up_above * 10
        
        # Aroon Down在Aroon Up上方-10分
        aroon_down_above = aroon_down > aroon_up
        cross_score -= aroon_down_above * 10
        
        return cross_score
    
    def _calculate_aroon_extreme_score(self) -> pd.Series:
        """
        计算Aroon极值评分
        
        Returns:
            pd.Series: 极值评分
        """
        extreme_score = pd.Series(0.0, index=self._result.index)
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        
        # Aroon Up极高值（>80）+15分
        aroon_up_high = aroon_up > 80
        extreme_score += aroon_up_high * 15
        
        # Aroon Down极高值（>80）-15分
        aroon_down_high = aroon_down > 80
        extreme_score -= aroon_down_high * 15
        
        # Aroon Up极低值（<20）-10分
        aroon_up_low = aroon_up < 20
        extreme_score -= aroon_up_low * 10
        
        # Aroon Down极低值（<20）+10分
        aroon_down_low = aroon_down < 20
        extreme_score += aroon_down_low * 10
        
        # 双方都在极值区域（都>70或都<30）额外评分
        both_high = (aroon_up > 70) & (aroon_down > 70)
        extreme_score += both_high * 5  # 强势震荡
        
        both_low = (aroon_up < 30) & (aroon_down < 30)
        extreme_score -= both_low * 5  # 弱势震荡
        
        return extreme_score
    
    def _calculate_aroon_trend_score(self) -> pd.Series:
        """
        计算Aroon趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        
        # Aroon Up上升趋势+8分
        aroon_up_rising = aroon_up > aroon_up.shift(1)
        trend_score += aroon_up_rising * 8
        
        # Aroon Up下降趋势-5分
        aroon_up_falling = aroon_up < aroon_up.shift(1)
        trend_score -= aroon_up_falling * 5
        
        # Aroon Down上升趋势-8分
        aroon_down_rising = aroon_down > aroon_down.shift(1)
        trend_score -= aroon_down_rising * 8
        
        # Aroon Down下降趋势+5分
        aroon_down_falling = aroon_down < aroon_down.shift(1)
        trend_score += aroon_down_falling * 5
        
        return trend_score
    
    def _calculate_aroon_oscillator_score(self) -> pd.Series:
        """
        计算Aroon震荡器评分
        
        Returns:
            pd.Series: 震荡器评分
        """
        oscillator_score = pd.Series(0.0, index=self._result.index)
        
        aroon_oscillator = self._result['aroon_oscillator']
        
        # 震荡器上穿零轴+20分
        oscillator_cross_up_zero = crossover(aroon_oscillator, 0)
        oscillator_score += oscillator_cross_up_zero * 20
        
        # 震荡器下穿零轴-20分
        oscillator_cross_down_zero = crossunder(aroon_oscillator, 0)
        oscillator_score -= oscillator_cross_down_zero * 20
        
        # 震荡器在零轴上方+8分
        oscillator_above_zero = aroon_oscillator > 0
        oscillator_score += oscillator_above_zero * 8
        
        # 震荡器在零轴下方-8分
        oscillator_below_zero = aroon_oscillator < 0
        oscillator_score -= oscillator_below_zero * 8
        
        # 震荡器极值评分
        oscillator_extreme_high = aroon_oscillator > 50
        oscillator_score += oscillator_extreme_high * 12
        
        oscillator_extreme_low = aroon_oscillator < -50
        oscillator_score -= oscillator_extreme_low * 12
        
        return oscillator_score
    
    def _calculate_aroon_strength_score(self) -> pd.Series:
        """
        计算Aroon强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        aroon_oscillator = self._result['aroon_oscillator']
        
        # 计算Aroon值的变化幅度
        aroon_up_change = aroon_up.diff()
        aroon_down_change = aroon_down.diff()
        
        # Aroon Up大幅上升+10分
        aroon_up_large_rise = aroon_up_change > 20
        strength_score += aroon_up_large_rise * 10
        
        # Aroon Down大幅上升-10分
        aroon_down_large_rise = aroon_down_change > 20
        strength_score -= aroon_down_large_rise * 10
        
        # 震荡器快速变化评分
        oscillator_change = aroon_oscillator.diff()
        rapid_oscillator_change = np.abs(oscillator_change) > 30
        oscillator_direction = np.sign(oscillator_change)
        strength_score += rapid_oscillator_change * oscillator_direction * 8
        
        return strength_score
    
    def _detect_aroon_cross_patterns(self) -> List[str]:
        """
        检测Aroon交叉形态
        
        Returns:
            List[str]: 交叉形态列表
        """
        patterns = []
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        
        # 检查最近的交叉
        recent_periods = min(5, len(aroon_up))
        recent_up = aroon_up.tail(recent_periods)
        recent_down = aroon_down.tail(recent_periods)
        
        if crossover(recent_up, recent_down).any():
            patterns.append("Aroon Up上穿Down")
        
        if crossover(recent_down, recent_up).any():
            patterns.append("Aroon Down上穿Up")
        
        # 检查当前位置关系
        if len(aroon_up) > 0 and len(aroon_down) > 0:
            current_up = aroon_up.iloc[-1]
            current_down = aroon_down.iloc[-1]
            
            if not pd.isna(current_up) and not pd.isna(current_down):
                if current_up > current_down:
                    patterns.append("Aroon Up主导")
                elif current_down > current_up:
                    patterns.append("Aroon Down主导")
                else:
                    patterns.append("Aroon平衡")
        
        return patterns
    
    def _detect_aroon_extreme_patterns(self) -> List[str]:
        """
        检测Aroon极值形态
        
        Returns:
            List[str]: 极值形态列表
        """
        patterns = []
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        
        if len(aroon_up) > 0 and len(aroon_down) > 0:
            current_up = aroon_up.iloc[-1]
            current_down = aroon_down.iloc[-1]
            
            if not pd.isna(current_up) and not pd.isna(current_down):
                # 极值检测
                if current_up > 80:
                    patterns.append("Aroon Up极高")
                elif current_up < 20:
                    patterns.append("Aroon Up极低")
                
                if current_down > 80:
                    patterns.append("Aroon Down极高")
                elif current_down < 20:
                    patterns.append("Aroon Down极低")
                
                # 双方状态
                if current_up > 70 and current_down > 70:
                    patterns.append("双方强势")
                elif current_up < 30 and current_down < 30:
                    patterns.append("双方弱势")
        
        return patterns
    
    def _detect_aroon_trend_patterns(self) -> List[str]:
        """
        检测Aroon趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        
        # 检查趋势
        if len(aroon_up) >= 3 and len(aroon_down) >= 3:
            recent_up = aroon_up.tail(3)
            recent_down = aroon_down.tail(3)
            
            if not recent_up.isna().any() and not recent_down.isna().any():
                # Aroon Up趋势
                if (recent_up.iloc[2] > recent_up.iloc[1] > recent_up.iloc[0]):
                    patterns.append("Aroon Up连续上升")
                elif (recent_up.iloc[2] < recent_up.iloc[1] < recent_up.iloc[0]):
                    patterns.append("Aroon Up连续下降")
                
                # Aroon Down趋势
                if (recent_down.iloc[2] > recent_down.iloc[1] > recent_down.iloc[0]):
                    patterns.append("Aroon Down连续上升")
                elif (recent_down.iloc[2] < recent_down.iloc[1] < recent_down.iloc[0]):
                    patterns.append("Aroon Down连续下降")
        
        return patterns
    
    def _detect_aroon_oscillator_patterns(self) -> List[str]:
        """
        检测Aroon震荡器形态
        
        Returns:
            List[str]: 震荡器形态列表
        """
        patterns = []
        
        aroon_oscillator = self._result['aroon_oscillator']
        
        # 检查最近的零轴穿越
        recent_periods = min(5, len(aroon_oscillator))
        recent_oscillator = aroon_oscillator.tail(recent_periods)
        
        if crossover(recent_oscillator, 0).any():
            patterns.append("震荡器上穿零轴")
        
        if crossunder(recent_oscillator, 0).any():
            patterns.append("震荡器下穿零轴")
        
        # 检查当前位置
        if len(aroon_oscillator) > 0:
            current_oscillator = aroon_oscillator.iloc[-1]
            
            if not pd.isna(current_oscillator):
                if current_oscillator > 50:
                    patterns.append("震荡器强势区域")
                elif current_oscillator > 0:
                    patterns.append("震荡器零轴上方")
                elif current_oscillator < -50:
                    patterns.append("震荡器弱势区域")
                elif current_oscillator < 0:
                    patterns.append("震荡器零轴下方")
                else:
                    patterns.append("震荡器零轴位置")
        
        return patterns
    
    def _detect_aroon_strength_patterns(self) -> List[str]:
        """
        检测Aroon强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        aroon_oscillator = self._result['aroon_oscillator']
        
        if len(aroon_up) >= 2 and len(aroon_down) >= 2:
            up_change = aroon_up.iloc[-1] - aroon_up.iloc[-2]
            down_change = aroon_down.iloc[-1] - aroon_down.iloc[-2]
            
            if not pd.isna(up_change) and not pd.isna(down_change):
                # 大幅变化检测
                if up_change > 20:
                    patterns.append("Aroon Up急升")
                elif up_change < -20:
                    patterns.append("Aroon Up急跌")
                
                if down_change > 20:
                    patterns.append("Aroon Down急升")
                elif down_change < -20:
                    patterns.append("Aroon Down急跌")
        
        # 震荡器强度
        if len(aroon_oscillator) >= 2:
            oscillator_change = aroon_oscillator.iloc[-1] - aroon_oscillator.iloc[-2]
            
            if not pd.isna(oscillator_change):
                if oscillator_change > 30:
                    patterns.append("震荡器急速上升")
                elif oscillator_change < -30:
                    patterns.append("震荡器急速下降")
        
        return patterns
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制阿隆指标(Aroon)图表
        
        Args:
            df: 包含Aroon指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['aroon_up', 'aroon_down', 'aroon_oscillator']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        else:
            ax1 = ax
            ax2 = None
        
        # 绘制Aroon Up和Aroon Down
        ax1.plot(df.index, df['aroon_up'], label='Aroon Up', color='green', linewidth=2)
        ax1.plot(df.index, df['aroon_down'], label='Aroon Down', color='red', linewidth=2)
        
        # 添加水平线
        ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax1.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_title(f"阿隆指标(Aroon, {self.period}日)")
        ax1.set_ylabel('Aroon值')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 如果有第二个子图，绘制震荡器
        if ax2 is not None:
            ax2.plot(df.index, df['aroon_oscillator'], label='Aroon震荡器', color='blue', linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(y=-50, color='gray', linestyle='--', alpha=0.5)
            
            ax2.set_ylabel('震荡器值')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        return ax1 if ax2 is None else (ax1, ax2)

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成Aroon指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算Aroon指标
        if not self.has_result():
            self.calculate(data)
        
        # 获取Aroon值
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        aroon_oscillator = self._result['aroon_oscillator']
        
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
        signals['market_env'] = '中性'
        signals['volume_confirmation'] = False
        
        # 计算ATR用于止损设置
        try:
            from indicators.atr import ATR
            atr_indicator = ATR()
            atr_data = atr_indicator.calculate(data)
            atr_values = atr_data['atr']
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            atr_values = pd.Series(0, index=data.index)
        
        # 生成信号
        for i in range(1, len(data)):
            # 确保是有效的索引位置
            if i < 1 or i >= len(aroon_up) or pd.isna(aroon_up.iloc[i]) or pd.isna(aroon_down.iloc[i]):
                continue
                
            # 1. Aroon Up上穿Aroon Down信号
            if aroon_up.iloc[i-1] <= aroon_down.iloc[i-1] and aroon_up.iloc[i] > aroon_down.iloc[i]:
                signals.loc[data.index[i], 'buy_signal'] = True
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = 1
                signals.loc[data.index[i], 'score'] += 20
                signals.loc[data.index[i], 'signal_type'] = 'Aroon多头交叉'
                signals.loc[data.index[i], 'signal_desc'] = f'Aroon Up({aroon_up.iloc[i]:.2f})上穿Aroon Down({aroon_down.iloc[i]:.2f})'
                signals.loc[data.index[i], 'confidence'] = 65.0
                
            # 2. Aroon Down上穿Aroon Up信号
            elif aroon_down.iloc[i-1] <= aroon_up.iloc[i-1] and aroon_down.iloc[i] > aroon_up.iloc[i]:
                signals.loc[data.index[i], 'sell_signal'] = True
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = -1
                signals.loc[data.index[i], 'score'] -= 20
                signals.loc[data.index[i], 'signal_type'] = 'Aroon空头交叉'
                signals.loc[data.index[i], 'signal_desc'] = f'Aroon Down({aroon_down.iloc[i]:.2f})上穿Aroon Up({aroon_up.iloc[i]:.2f})'
                signals.loc[data.index[i], 'confidence'] = 65.0
                
            # 3. Aroon震荡器穿越零轴信号
            elif aroon_oscillator.iloc[i-1] <= 0 and aroon_oscillator.iloc[i] > 0:
                signals.loc[data.index[i], 'buy_signal'] = True
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = 1
                signals.loc[data.index[i], 'score'] += 15
                signals.loc[data.index[i], 'signal_type'] = 'Aroon震荡器上穿零轴'
                signals.loc[data.index[i], 'signal_desc'] = f'Aroon震荡器由负转正: {aroon_oscillator.iloc[i]:.2f}'
                signals.loc[data.index[i], 'confidence'] = 60.0
                
            elif aroon_oscillator.iloc[i-1] >= 0 and aroon_oscillator.iloc[i] < 0:
                signals.loc[data.index[i], 'sell_signal'] = True
                signals.loc[data.index[i], 'neutral_signal'] = False
                signals.loc[data.index[i], 'trend'] = -1
                signals.loc[data.index[i], 'score'] -= 15
                signals.loc[data.index[i], 'signal_type'] = 'Aroon震荡器下穿零轴'
                signals.loc[data.index[i], 'signal_desc'] = f'Aroon震荡器由正转负: {aroon_oscillator.iloc[i]:.2f}'
                signals.loc[data.index[i], 'confidence'] = 60.0
                
            # 4. Aroon极值信号
            if aroon_up.iloc[i] > 90 and aroon_down.iloc[i] < 30:
                signals.loc[data.index[i], 'score'] += 15
                signals.loc[data.index[i], 'trend'] = 1
                if not signals.loc[data.index[i], 'buy_signal']:
                    signals.loc[data.index[i], 'signal_desc'] = f'Aroon指示强劲上升趋势: Up={aroon_up.iloc[i]:.2f}, Down={aroon_down.iloc[i]:.2f}'
                    signals.loc[data.index[i], 'signal_type'] = 'Aroon强势上涨'
                signals.loc[data.index[i], 'confidence'] += 10.0
                
            elif aroon_down.iloc[i] > 90 and aroon_up.iloc[i] < 30:
                signals.loc[data.index[i], 'score'] -= 15
                signals.loc[data.index[i], 'trend'] = -1
                if not signals.loc[data.index[i], 'sell_signal']:
                    signals.loc[data.index[i], 'signal_desc'] = f'Aroon指示强劲下降趋势: Down={aroon_down.iloc[i]:.2f}, Up={aroon_up.iloc[i]:.2f}'
                    signals.loc[data.index[i], 'signal_type'] = 'Aroon强势下跌'
                signals.loc[data.index[i], 'confidence'] += 10.0
                
            # 5. Aroon盘整信号
            if aroon_up.iloc[i] < 30 and aroon_down.iloc[i] < 30:
                signals.loc[data.index[i], 'signal_type'] = 'Aroon盘整'
                signals.loc[data.index[i], 'signal_desc'] = f'Aroon指示市场盘整: Up={aroon_up.iloc[i]:.2f}, Down={aroon_down.iloc[i]:.2f}'
                
            # 6. Aroon趋势强度评分
            if aroon_up.iloc[i] > 70:
                signals.loc[data.index[i], 'score'] += (aroon_up.iloc[i] - 70) / 3
                signals.loc[data.index[i], 'market_env'] = '强势'
                
            elif aroon_down.iloc[i] > 70:
                signals.loc[data.index[i], 'score'] -= (aroon_down.iloc[i] - 70) / 3
                signals.loc[data.index[i], 'market_env'] = '弱势'
                
            # 设置仓位大小和止损位
            if signals.loc[data.index[i], 'buy_signal']:
                signals.loc[data.index[i], 'position_size'] = min(0.3, (signals.loc[data.index[i], 'score'] - 50) / 100)
                signals.loc[data.index[i], 'stop_loss'] = data['close'].iloc[i] - 2.5 * atr_values.iloc[i]
                
                # 判断成交量确认
                if i > 0 and 'volume' in data.columns and data['volume'].iloc[i] > data['volume'].iloc[i-1] * 1.1:
                    signals.loc[data.index[i], 'volume_confirmation'] = True
                    signals.loc[data.index[i], 'confidence'] += 5.0
                
            elif signals.loc[data.index[i], 'sell_signal']:
                signals.loc[data.index[i], 'position_size'] = min(0.3, (50 - signals.loc[data.index[i], 'score']) / 100)
                signals.loc[data.index[i], 'stop_loss'] = data['close'].iloc[i] + 2.5 * atr_values.iloc[i]
                
                # 判断成交量确认
                if i > 0 and 'volume' in data.columns and data['volume'].iloc[i] > data['volume'].iloc[i-1] * 1.1:
                    signals.loc[data.index[i], 'volume_confirmation'] = True
                    signals.loc[data.index[i], 'confidence'] += 5.0
            
            # 风险水平设置
            if signals.loc[data.index[i], 'score'] >= 75 or signals.loc[data.index[i], 'score'] <= 25:
                signals.loc[data.index[i], 'risk_level'] = '高'
            elif 40 <= signals.loc[data.index[i], 'score'] <= 60:
                signals.loc[data.index[i], 'risk_level'] = '低'
            
            # 限制评分范围
            signals.loc[data.index[i], 'score'] = np.clip(signals.loc[data.index[i], 'score'], 0, 100)
            
            # 限制置信度范围
            signals.loc[data.index[i], 'confidence'] = np.clip(signals.loc[data.index[i], 'confidence'], 0, 100)
        
        return signals 