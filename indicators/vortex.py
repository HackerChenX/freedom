#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
涡流指标(Vortex Indicator)

涡流指标用于识别趋势的开始和确认现有趋势，由两条振荡线组成：VI+和VI-
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
# import talib  # 移除talib依赖

from indicators.base_indicator import BaseIndicator
from utils.indicator_utils import crossover, crossunder
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

logger = get_logger(__name__)


class Vortex(BaseIndicator):
    """
    涡流指标(Vortex Indicator) (Vortex)
    
    分类：趋势类指标
    描述：涡流指标用于识别趋势的开始和确认现有趋势，由两条振荡线组成：VI+和VI-
    """
    
    def __init__(self, period: int = 14):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化涡流指标(Vortex Indicator)
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__(name="Vortex", description="涡流指标，用于识别趋势的开始和确认现有趋势")
        self.period = period

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Vortex指标

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Vortex指标的DataFrame
        """
        return self._calculate(data)

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含period等
        """
        if 'period' in kwargs:
            self.period = kwargs['period']
        
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
        计算Vortex指标
        
        Args:
            df: 包含OHLC数据的DataFrame
                
        Returns:
            包含Vortex指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算涡流指标(Vortex Indicator)
        
        Args:
            df: 包含OHLC数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                
        Returns:
            添加了Vortex指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['high', 'low', 'close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算正向和负向涡流运动
        # VM+ = |当前最高价 - 前一最低价|
        # VM- = |当前最低价 - 前一最高价|
        vm_plus = np.abs(df_copy['high'] - df_copy['low'].shift(1))
        vm_minus = np.abs(df_copy['low'] - df_copy['high'].shift(1))
        
        # 计算真实波幅(True Range)
        # TR = max(|高-低|, |高-前收|, |低-前收|)
        high_low = df_copy['high'] - df_copy['low']
        high_close = np.abs(df_copy['high'] - df_copy['close'].shift(1))
        low_close = np.abs(df_copy['low'] - df_copy['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # 计算周期内的累计值
        vm_plus_sum = vm_plus.rolling(window=self.period).sum()
        vm_minus_sum = vm_minus.rolling(window=self.period).sum()
        tr_sum = true_range.rolling(window=self.period).sum()
        
        # 计算VI+和VI-
        df_copy['vi_plus'] = vm_plus_sum / tr_sum
        df_copy['vi_minus'] = vm_minus_sum / tr_sum
        
        # 计算VI差值
        df_copy['vi_diff'] = df_copy['vi_plus'] - df_copy['vi_minus']
        
        # 存储结果
        self._result = df_copy[['vi_plus', 'vi_minus', 'vi_diff']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成涡流指标(Vortex Indicator)交易信号
        
        Args:
            df: 包含价格数据和Vortex指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - vortex_buy_signal: 1=买入信号, 0=无信号
            - vortex_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['vi_plus', 'vi_minus']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['vortex_buy_signal'] = 0
        df_copy['vortex_sell_signal'] = 0
        
        # 生成交易信号
        for i in range(1, len(df_copy)):
            # 1. VI+上穿VI-（买入信号）
            if (df_copy['vi_plus'].iloc[i-1] <= df_copy['vi_minus'].iloc[i-1] and 
                df_copy['vi_plus'].iloc[i] > df_copy['vi_minus'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_buy_signal')] = 1
            
            # 2. VI-上穿VI+（卖出信号）
            elif (df_copy['vi_minus'].iloc[i-1] <= df_copy['vi_plus'].iloc[i-1] and 
                  df_copy['vi_minus'].iloc[i] > df_copy['vi_plus'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_sell_signal')] = 1
            
            # 3. VI+突破1.1阈值（强买入信号）
            elif (df_copy['vi_plus'].iloc[i-1] <= 1.1 and 
                  df_copy['vi_plus'].iloc[i] > 1.1):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_buy_signal')] = 1
            
            # 4. VI-突破1.1阈值（强卖出信号）
            elif (df_copy['vi_minus'].iloc[i-1] <= 1.1 and 
                  df_copy['vi_minus'].iloc[i] > 1.1):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_sell_signal')] = 1
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Vortex原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算Vortex
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. Vortex交叉评分
        cross_score = self._calculate_vortex_cross_score()
        score += cross_score
        
        # 2. Vortex阈值评分
        threshold_score = self._calculate_vortex_threshold_score()
        score += threshold_score
        
        # 3. Vortex趋势评分
        trend_score = self._calculate_vortex_trend_score()
        score += trend_score
        
        # 4. Vortex背离评分
        divergence_score = self._calculate_vortex_divergence_score(data)
        score += divergence_score
        
        # 5. Vortex强度评分
        strength_score = self._calculate_vortex_strength_score()
        score += strength_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别Vortex技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算Vortex
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测Vortex交叉形态
        cross_patterns = self._detect_vortex_cross_patterns()
        patterns.extend(cross_patterns)
        
        # 2. 检测Vortex阈值形态
        threshold_patterns = self._detect_vortex_threshold_patterns()
        patterns.extend(threshold_patterns)
        
        # 3. 检测Vortex趋势形态
        trend_patterns = self._detect_vortex_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 4. 检测Vortex背离形态
        divergence_patterns = self._detect_vortex_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        # 5. 检测Vortex强度形态
        strength_patterns = self._detect_vortex_strength_patterns()
        patterns.extend(strength_patterns)
        
        return patterns
    
    def _calculate_vortex_cross_score(self) -> pd.Series:
        """
        计算Vortex交叉评分
        
        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=self._result.index)
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        
        # VI+上穿VI-+25分
        vi_plus_cross_up = crossover(vi_plus, vi_minus)
        cross_score += vi_plus_cross_up * 25
        
        # VI-上穿VI+-25分
        vi_minus_cross_up = crossover(vi_minus, vi_plus)
        cross_score -= vi_minus_cross_up * 25
        
        # VI+在VI-上方+10分
        vi_plus_above = vi_plus > vi_minus
        cross_score += vi_plus_above * 10
        
        # VI-在VI+上方-10分
        vi_minus_above = vi_minus > vi_plus
        cross_score -= vi_minus_above * 10
        
        return cross_score
    
    def _calculate_vortex_threshold_score(self) -> pd.Series:
        """
        计算Vortex阈值评分
        
        Returns:
            pd.Series: 阈值评分
        """
        threshold_score = pd.Series(0.0, index=self._result.index)
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        
        # VI+突破1.1阈值+20分
        vi_plus_break_high = crossover(vi_plus, 1.1)
        threshold_score += vi_plus_break_high * 20
        
        # VI-突破1.1阈值-20分
        vi_minus_break_high = crossover(vi_minus, 1.1)
        threshold_score -= vi_minus_break_high * 20
        
        # VI+跌破0.9阈值-15分
        vi_plus_break_low = crossunder(vi_plus, 0.9)
        threshold_score -= vi_plus_break_low * 15
        
        # VI-跌破0.9阈值+15分
        vi_minus_break_low = crossunder(vi_minus, 0.9)
        threshold_score += vi_minus_break_low * 15
        
        # VI+在1.0上方+8分
        vi_plus_above_one = vi_plus > 1.0
        threshold_score += vi_plus_above_one * 8
        
        # VI-在1.0上方-8分
        vi_minus_above_one = vi_minus > 1.0
        threshold_score -= vi_minus_above_one * 8
        
        return threshold_score
    
    def _calculate_vortex_trend_score(self) -> pd.Series:
        """
        计算Vortex趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        vi_diff = self._result['vi_diff']
        
        # VI差值上升趋势+12分
        vi_diff_rising = vi_diff > vi_diff.shift(1)
        trend_score += vi_diff_rising * 12
        
        # VI差值下降趋势-12分
        vi_diff_falling = vi_diff < vi_diff.shift(1)
        trend_score -= vi_diff_falling * 12
        
        # VI+连续上升+15分
        vi_plus_consecutive_rising = (
            (vi_plus > vi_plus.shift(1)) &
            (vi_plus.shift(1) > vi_plus.shift(2)) &
            (vi_plus.shift(2) > vi_plus.shift(3))
        )
        trend_score += vi_plus_consecutive_rising * 15
        
        # VI-连续上升-15分
        vi_minus_consecutive_rising = (
            (vi_minus > vi_minus.shift(1)) &
            (vi_minus.shift(1) > vi_minus.shift(2)) &
            (vi_minus.shift(2) > vi_minus.shift(3))
        )
        trend_score -= vi_minus_consecutive_rising * 15
        
        return trend_score
    
    def _calculate_vortex_divergence_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算Vortex背离评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return divergence_score
        
        close_price = data['close']
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        
        # 简化的背离检测
        if len(close_price) >= 20:
            # 检查最近20个周期的价格和VI趋势
            recent_periods = 20
            
            for i in range(recent_periods, len(close_price)):
                # 寻找最近的价格和VI峰值/谷值
                price_window = close_price.iloc[i-recent_periods:i+1]
                vi_plus_window = vi_plus.iloc[i-recent_periods:i+1]
                vi_minus_window = vi_minus.iloc[i-recent_periods:i+1]
                
                # 检查是否为价格新高/新低
                current_price = close_price.iloc[i]
                current_vi_plus = vi_plus.iloc[i]
                current_vi_minus = vi_minus.iloc[i]
                
                price_is_high = current_price >= price_window.max()
                price_is_low = current_price <= price_window.min()
                vi_plus_is_high = current_vi_plus >= vi_plus_window.max()
                vi_plus_is_low = current_vi_plus <= vi_plus_window.min()
                
                # 正背离：价格创新低但VI+未创新低
                if price_is_low and not vi_plus_is_low:
                    divergence_score.iloc[i] += 30
                
                # 负背离：价格创新高但VI+未创新高
                elif price_is_high and not vi_plus_is_high:
                    divergence_score.iloc[i] -= 30
        
        return divergence_score
    
    def _calculate_vortex_strength_score(self) -> pd.Series:
        """
        计算Vortex强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        vi_diff = self._result['vi_diff']
        
        # 计算VI值的变化幅度
        vi_plus_change = vi_plus.diff()
        vi_minus_change = vi_minus.diff()
        vi_diff_change = vi_diff.diff()
        
        # VI+大幅上升+10分
        vi_plus_large_rise = vi_plus_change > 0.1
        strength_score += vi_plus_large_rise * 10
        
        # VI-大幅上升-10分
        vi_minus_large_rise = vi_minus_change > 0.1
        strength_score -= vi_minus_large_rise * 10
        
        # VI差值快速变化评分
        rapid_diff_change = np.abs(vi_diff_change) > 0.15
        diff_direction = np.sign(vi_diff_change)
        strength_score += rapid_diff_change * diff_direction * 8
        
        # VI极值评分
        vi_plus_extreme_high = vi_plus > 1.3
        strength_score += vi_plus_extreme_high * 12
        
        vi_minus_extreme_high = vi_minus > 1.3
        strength_score -= vi_minus_extreme_high * 12
        
        return strength_score

    def register_patterns(self):
        """
        注册Vortex指标的技术形态
        """
        # 注册Vortex交叉形态
        self.register_pattern_to_registry(
            pattern_id="VORTEX_BULLISH_CROSS",
            display_name="涡流看涨交叉",
            description="VI+上穿VI-，表明上升趋势开始",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VORTEX_BEARISH_CROSS",
            display_name="涡流看跌交叉",
            description="VI+下穿VI-，表明下降趋势开始",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册Vortex阈值形态
        self.register_pattern_to_registry(
            pattern_id="VORTEX_STRONG_BULLISH",
            display_name="涡流强烈看涨",
            description="VI+远高于VI-，表明强烈上升趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VORTEX_STRONG_BEARISH",
            display_name="涡流强烈看跌",
            description="VI-远高于VI+，表明强烈下降趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册Vortex趋势形态
        self.register_pattern_to_registry(
            pattern_id="VORTEX_UPTREND",
            display_name="涡流上升趋势",
            description="VI+持续高于VI-，确认上升趋势",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VORTEX_DOWNTREND",
            display_name="涡流下降趋势",
            description="VI-持续高于VI+，确认下降趋势",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册Vortex背离形态
        self.register_pattern_to_registry(
            pattern_id="VORTEX_BULLISH_DIVERGENCE",
            display_name="涡流看涨背离",
            description="价格创新低但Vortex未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VORTEX_BEARISH_DIVERGENCE",
            display_name="涡流看跌背离",
            description="价格创新高但Vortex未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

        # 注册Vortex状态形态（从centralized mapping迁移）
        self.register_pattern_to_registry(
            pattern_id="VORTEX_RISING",
            display_name="涡流上升",
            description="涡流指标呈上升趋势",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VORTEX_FALLING",
            display_name="涡流下降",
            description="涡流指标呈下降趋势",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )
    
    def _detect_vortex_cross_patterns(self) -> List[str]:
        """
        检测Vortex交叉形态
        
        Returns:
            List[str]: 交叉形态列表
        """
        patterns = []
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        
        # 检查最近的交叉
        recent_periods = min(5, len(vi_plus))
        recent_plus = vi_plus.tail(recent_periods)
        recent_minus = vi_minus.tail(recent_periods)
        
        if crossover(recent_plus, recent_minus).any():
            patterns.append("VI+上穿VI-")
        
        if crossover(recent_minus, recent_plus).any():
            patterns.append("VI-上穿VI+")
        
        # 检查当前位置关系
        if len(vi_plus) > 0 and len(vi_minus) > 0:
            current_plus = vi_plus.iloc[-1]
            current_minus = vi_minus.iloc[-1]
            
            if not pd.isna(current_plus) and not pd.isna(current_minus):
                if current_plus > current_minus:
                    patterns.append("VI+主导")
                elif current_minus > current_plus:
                    patterns.append("VI-主导")
                else:
                    patterns.append("VI平衡")
        
        return patterns
    
    def _detect_vortex_threshold_patterns(self) -> List[str]:
        """
        检测Vortex阈值形态
        
        Returns:
            List[str]: 阈值形态列表
        """
        patterns = []
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        
        # 检查最近的阈值穿越
        recent_periods = min(5, len(vi_plus))
        recent_plus = vi_plus.tail(recent_periods)
        recent_minus = vi_minus.tail(recent_periods)
        
        if crossover(recent_plus, 1.1).any():
            patterns.append("VI+突破1.1")
        
        if crossover(recent_minus, 1.1).any():
            patterns.append("VI-突破1.1")
        
        if crossunder(recent_plus, 0.9).any():
            patterns.append("VI+跌破0.9")
        
        if crossunder(recent_minus, 0.9).any():
            patterns.append("VI-跌破0.9")
        
        # 检查当前阈值位置
        if len(vi_plus) > 0 and len(vi_minus) > 0:
            current_plus = vi_plus.iloc[-1]
            current_minus = vi_minus.iloc[-1]
            
            if not pd.isna(current_plus) and not pd.isna(current_minus):
                if current_plus > 1.1:
                    patterns.append("VI+强势区域")
                elif current_plus > 1.0:
                    patterns.append("VI+中性偏强")
                elif current_plus < 0.9:
                    patterns.append("VI+弱势区域")
                
                if current_minus > 1.1:
                    patterns.append("VI-强势区域")
                elif current_minus > 1.0:
                    patterns.append("VI-中性偏强")
                elif current_minus < 0.9:
                    patterns.append("VI-弱势区域")
        
        return patterns
    
    def _detect_vortex_trend_patterns(self) -> List[str]:
        """
        检测Vortex趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        vi_diff = self._result['vi_diff']
        
        # 检查VI差值趋势
        if len(vi_diff) >= 3:
            recent_3 = vi_diff.tail(3)
            if len(recent_3) == 3 and not recent_3.isna().any():
                if (recent_3.iloc[2] > recent_3.iloc[1] > recent_3.iloc[0]):
                    patterns.append("VI差值连续上升")
                elif (recent_3.iloc[2] < recent_3.iloc[1] < recent_3.iloc[0]):
                    patterns.append("VI差值连续下降")
        
        # 检查VI+趋势
        if len(vi_plus) >= 3:
            recent_plus = vi_plus.tail(3)
            if not recent_plus.isna().any():
                if (recent_plus.iloc[2] > recent_plus.iloc[1] > recent_plus.iloc[0]):
                    patterns.append("VI+连续上升")
                elif (recent_plus.iloc[2] < recent_plus.iloc[1] < recent_plus.iloc[0]):
                    patterns.append("VI+连续下降")
        
        # 检查VI-趋势
        if len(vi_minus) >= 3:
            recent_minus = vi_minus.tail(3)
            if not recent_minus.isna().any():
                if (recent_minus.iloc[2] > recent_minus.iloc[1] > recent_minus.iloc[0]):
                    patterns.append("VI-连续上升")
                elif (recent_minus.iloc[2] < recent_minus.iloc[1] < recent_minus.iloc[0]):
                    patterns.append("VI-连续下降")
        
        return patterns
    
    def _detect_vortex_divergence_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测Vortex背离形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        
        if len(close_price) >= 20:
            # 检查最近20个周期的趋势
            recent_price = close_price.tail(20)
            recent_vi_plus = vi_plus.tail(20)
            recent_vi_minus = vi_minus.tail(20)
            
            # 简化的背离检测
            price_trend = recent_price.iloc[-1] - recent_price.iloc[0]
            vi_plus_trend = recent_vi_plus.iloc[-1] - recent_vi_plus.iloc[0]
            vi_minus_trend = recent_vi_minus.iloc[-1] - recent_vi_minus.iloc[0]
            
            # 背离检测
            if price_trend < -0.02 and vi_plus_trend > 0.05:  # 价格下跌但VI+上升
                patterns.append("VI+正背离")
            elif price_trend > 0.02 and vi_plus_trend < -0.05:  # 价格上涨但VI+下降
                patterns.append("VI+负背离")
            
            if price_trend > 0.02 and vi_minus_trend > 0.05:  # 价格上涨但VI-也上升
                patterns.append("VI-负背离")
            elif price_trend < -0.02 and vi_minus_trend < -0.05:  # 价格下跌VI-也下降
                patterns.append("VI-正背离")
        
        return patterns
    
    def _detect_vortex_strength_patterns(self) -> List[str]:
        """
        检测Vortex强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        vi_diff = self._result['vi_diff']
        
        if len(vi_plus) >= 2 and len(vi_minus) >= 2:
            plus_change = vi_plus.iloc[-1] - vi_plus.iloc[-2]
            minus_change = vi_minus.iloc[-1] - vi_minus.iloc[-2]
            
            if not pd.isna(plus_change) and not pd.isna(minus_change):
                # 大幅变化检测
                if plus_change > 0.1:
                    patterns.append("VI+急升")
                elif plus_change < -0.1:
                    patterns.append("VI+急跌")
                
                if minus_change > 0.1:
                    patterns.append("VI-急升")
                elif minus_change < -0.1:
                    patterns.append("VI-急跌")
        
        # VI极值检测
        if len(vi_plus) > 0 and len(vi_minus) > 0:
            current_plus = vi_plus.iloc[-1]
            current_minus = vi_minus.iloc[-1]
            
            if not pd.isna(current_plus) and not pd.isna(current_minus):
                if current_plus > 1.3:
                    patterns.append("VI+极强")
                elif current_plus < 0.7:
                    patterns.append("VI+极弱")
                
                if current_minus > 1.3:
                    patterns.append("VI-极强")
                elif current_minus < 0.7:
                    patterns.append("VI-极弱")
        
        # VI差值强度
        if len(vi_diff) >= 2:
            diff_change = vi_diff.iloc[-1] - vi_diff.iloc[-2]
            
            if not pd.isna(diff_change):
                if diff_change > 0.2:
                    patterns.append("VI差值急速扩大")
                elif diff_change < -0.2:
                    patterns.append("VI差值急速收缩")
        
        return patterns
        
    def _register_vortex_patterns(self):
        """
        注册Vortex指标相关形态
        """
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册Vortex交叉形态
        registry.register(
            pattern_id="VORTEX_BULLISH_CROSS",
            display_name="Vortex多头交叉",
            description="VI+上穿VI-，表明趋势由空转多",
            indicator_id="VORTEX",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="VORTEX_BEARISH_CROSS",
            display_name="Vortex空头交叉",
            description="VI-上穿VI+，表明趋势由多转空",
            indicator_id="VORTEX",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
        
        # 注册Vortex阈值形态
        registry.register(
            pattern_id="VORTEX_STRONG_UPTREND",
            display_name="Vortex强上升趋势",
            description="VI+高于1.1，表明上升趋势强烈",
            indicator_id="VORTEX",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="VORTEX_STRONG_DOWNTREND",
            display_name="Vortex强下降趋势",
            description="VI-高于1.1，表明下降趋势强烈",
            indicator_id="VORTEX",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0
        )
        
        # 注册Vortex趋势形态
        registry.register(
            pattern_id="VORTEX_UPTREND_STRENGTHENING",
            display_name="Vortex上升趋势增强",
            description="VI+与VI-的差值扩大，表明上升趋势增强",
            indicator_id="VORTEX",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="VORTEX_DOWNTREND_STRENGTHENING",
            display_name="Vortex下降趋势增强",
            description="VI-与VI+的差值扩大，表明下降趋势增强",
            indicator_id="VORTEX",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        # 注册Vortex背离形态
        registry.register(
            pattern_id="VORTEX_BULLISH_DIVERGENCE",
            display_name="Vortex底背离",
            description="价格创新低，但VI-未创新高，表明下跌动能减弱",
            indicator_id="VORTEX",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="VORTEX_BEARISH_DIVERGENCE",
            display_name="Vortex顶背离",
            description="价格创新高，但VI+未创新高，表明上涨动能减弱",
            indicator_id="VORTEX",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
        
        # 注册Vortex强度形态
        registry.register(
            pattern_id="VORTEX_HIGH_VOLATILITY",
            display_name="Vortex高波动",
            description="VI+和VI-数值同时较高，表明市场波动性增强",
            indicator_id="VORTEX",
            pattern_type=PatternType.VOLATILITY,
            default_strength=PatternStrength.MEDIUM,
            score_impact=0.0
        )
        
        registry.register(
            pattern_id="VORTEX_LOW_VOLATILITY",
            display_name="Vortex低波动",
            description="VI+和VI-数值同时较低，表明市场波动性减弱",
            indicator_id="VORTEX",
            pattern_type=PatternType.VOLATILITY,
            default_strength=PatternStrength.WEAK,
            score_impact=0.0
        )

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取Vortex相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None or 'vi_plus' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取Vortex数据
        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']
        vi_diff = self._result['vi_diff']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. Vortex交叉形态
        patterns_df['VORTEX_BULLISH_CROSS'] = crossover(vi_plus, vi_minus)
        patterns_df['VORTEX_BEARISH_CROSS'] = crossover(vi_minus, vi_plus)
        patterns_df['VORTEX_VI_PLUS_ABOVE'] = vi_plus > vi_minus
        patterns_df['VORTEX_VI_MINUS_ABOVE'] = vi_minus > vi_plus

        # 2. Vortex阈值形态
        patterns_df['VORTEX_VI_PLUS_STRONG'] = vi_plus > 1.1
        patterns_df['VORTEX_VI_MINUS_STRONG'] = vi_minus > 1.1
        patterns_df['VORTEX_VI_PLUS_WEAK'] = vi_plus < 0.9
        patterns_df['VORTEX_VI_MINUS_WEAK'] = vi_minus < 0.9
        patterns_df['VORTEX_VI_PLUS_BREAK_HIGH'] = crossover(vi_plus, 1.1)
        patterns_df['VORTEX_VI_MINUS_BREAK_HIGH'] = crossover(vi_minus, 1.1)

        # 3. Vortex趋势形态
        patterns_df['VORTEX_VI_PLUS_RISING'] = vi_plus > vi_plus.shift(1)
        patterns_df['VORTEX_VI_MINUS_RISING'] = vi_minus > vi_minus.shift(1)
        patterns_df['VORTEX_VI_DIFF_RISING'] = vi_diff > vi_diff.shift(1)
        patterns_df['VORTEX_VI_DIFF_FALLING'] = vi_diff < vi_diff.shift(1)

        # 4. Vortex连续趋势形态
        patterns_df['VORTEX_VI_PLUS_UPTREND'] = (
            (vi_plus > vi_plus.shift(1)) &
            (vi_plus.shift(1) > vi_plus.shift(2)) &
            (vi_plus.shift(2) > vi_plus.shift(3))
        )
        patterns_df['VORTEX_VI_MINUS_UPTREND'] = (
            (vi_minus > vi_minus.shift(1)) &
            (vi_minus.shift(1) > vi_minus.shift(2)) &
            (vi_minus.shift(2) > vi_minus.shift(3))
        )

        # 5. Vortex极值形态
        patterns_df['VORTEX_VI_PLUS_EXTREME_HIGH'] = vi_plus > 1.3
        patterns_df['VORTEX_VI_MINUS_EXTREME_HIGH'] = vi_minus > 1.3
        patterns_df['VORTEX_VI_PLUS_EXTREME_LOW'] = vi_plus < 0.7
        patterns_df['VORTEX_VI_MINUS_EXTREME_LOW'] = vi_minus < 0.7

        return patterns_df

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算Vortex指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查Vortex形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)

        # 3. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)

        # 4. 基于评分趋势的置信度
        if len(score) >= 3:
            recent_scores = score.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算最终评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含评分和置信度的字典
        """
        try:
            # 1. 计算原始评分序列
            raw_scores = self.calculate_raw_score(data, **kwargs)

            # 如果数据不足，返回中性评分
            if len(raw_scores) < 3:
                return {'score': 50.0, 'confidence': 0.5}

            # 取最近的评分作为最终评分，但考虑近期趋势
            recent_scores = raw_scores.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 最终评分 = 最新评分 + 趋势调整
            final_score = recent_scores.iloc[-1] + trend / 2

            # 确保评分在0-100范围内
            final_score = max(0, min(100, final_score))

            # 2. 获取形态和信号
            patterns = self.get_patterns(data, **kwargs)

            # 3. 计算置信度
            confidence = self.calculate_confidence(raw_scores, patterns, {})

            return {
                'score': final_score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {'score': 50.0, 'confidence': 0.0}

    def register_patterns(self):
        """
        注册Vortex指标的形态到全局形态注册表
        """
        # 注册Vortex交叉形态
        self.register_pattern_to_registry(
            pattern_id="VORTEX_BULLISH_CROSS",
            display_name="Vortex多头交叉",
            description="VI+上穿VI-，表明趋势由空转多",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VORTEX_BEARISH_CROSS",
            display_name="Vortex空头交叉",
            description="VI-上穿VI+，表明趋势由多转空",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册Vortex阈值形态
        self.register_pattern_to_registry(
            pattern_id="VORTEX_VI_PLUS_STRONG",
            display_name="VI+强势",
            description="VI+高于1.1，表明上升趋势强烈",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VORTEX_VI_MINUS_STRONG",
            display_name="VI-强势",
            description="VI-高于1.1，表明下降趋势强烈",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0,
            polarity="NEGATIVE"
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成Vortex交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, pd.Series]: 包含买卖信号的字典
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        vi_plus = self._result['vi_plus']
        vi_minus = self._result['vi_minus']

        # 生成信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 1. VI+上穿VI-（买入信号）
        bullish_cross = crossover(vi_plus, vi_minus)
        buy_signal |= bullish_cross
        signal_strength += bullish_cross * 0.8

        # 2. VI-上穿VI+（卖出信号）
        bearish_cross = crossover(vi_minus, vi_plus)
        sell_signal |= bearish_cross
        signal_strength += bearish_cross * 0.8

        # 3. VI+突破1.1阈值（强买入信号）
        vi_plus_break_high = crossover(vi_plus, 1.1)
        buy_signal |= vi_plus_break_high
        signal_strength += vi_plus_break_high * 0.9

        # 4. VI-突破1.1阈值（强卖出信号）
        vi_minus_break_high = crossover(vi_minus, 1.1)
        sell_signal |= vi_minus_break_high
        signal_strength += vi_minus_break_high * 0.9

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成涡流指标(Vortex Indicator)交易信号
        
        Args:
            df: 包含价格数据和Vortex指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - vortex_buy_signal: 1=买入信号, 0=无信号
            - vortex_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['vi_plus', 'vi_minus']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['vortex_buy_signal'] = 0
        df_copy['vortex_sell_signal'] = 0
        
        # 生成交易信号
        for i in range(1, len(df_copy)):
            # 1. VI+上穿VI-（买入信号）
            if (df_copy['vi_plus'].iloc[i-1] <= df_copy['vi_minus'].iloc[i-1] and 
                df_copy['vi_plus'].iloc[i] > df_copy['vi_minus'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_buy_signal')] = 1
            
            # 2. VI-上穿VI+（卖出信号）
            elif (df_copy['vi_minus'].iloc[i-1] <= df_copy['vi_plus'].iloc[i-1] and 
                  df_copy['vi_minus'].iloc[i] > df_copy['vi_plus'].iloc[i]):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_sell_signal')] = 1
            
            # 3. VI+突破1.1阈值（强买入信号）
            elif (df_copy['vi_plus'].iloc[i-1] <= 1.1 and 
                  df_copy['vi_plus'].iloc[i] > 1.1):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_buy_signal')] = 1
            
            # 4. VI-突破1.1阈值（强卖出信号）
            elif (df_copy['vi_minus'].iloc[i-1] <= 1.1 and 
                  df_copy['vi_minus'].iloc[i] > 1.1):
                df_copy.iloc[i, df_copy.columns.get_loc('vortex_sell_signal')] = 1
        
        return df_copy 
    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # Vortex指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
