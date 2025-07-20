#!/usr/bin/env python3
"""
COMPOSITE_INDICATOR 指标

自动生成的最小化指标实现
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class CompositeIndicator(BaseIndicator, PatternSignalMixin):
    """
    COMPOSITE_INDICATOR 指标
    
    自动生成的最小化实现，支持参数标准化
    """
    
    def __init__(self, **kwargs):
        """
        初始化COMPOSITE_INDICATOR指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "COMPOSITE_INDICATOR"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_compositeindicator()
        
        # 应用用户参数
        self.set_parameters_Indicator(**kwargs)
    
    def _get_default_parameters_compositeindicator(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Indicator(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('COMPOSITE_INDICATOR', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)
                    
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 14
    
    def calculate_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算COMPOSITE_INDICATOR指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了COMPOSITE_INDICATOR指标的Data_frame
        """
        result = self._calculate_compositeindicator(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_compositeindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算COMPOSITE_INDICATOR指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了COMPOSITE_INDICATOR指标的Data_frame
        """
        df = data.copy()
        
        # 最小化实现：返回原数据加上一个简单的计算列
        df[f'COMPOSITE_INDICATOR_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑：基于评分值的阈值判断
        # 对于state_type指标，使用评分阈值模式
        score_threshold = 50.0  # 默认阈值
        df.loc[:, 'buy_signal'] = df[f'COMPOSITE_INDICATOR_VALUE'] >= score_threshold
        df.loc[:, 'sell_signal'] = df[f'COMPOSITE_INDICATOR_VALUE'] < score_threshold
        df.loc[:, 'hold_signal'] = df[f'COMPOSITE_INDICATOR_VALUE'] < score_threshold

        return df
    
    def calculate_raw_score_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Indicator(data, **kwargs)
        
        # 复合指标评分：多指标综合分析
        df = data.copy()
        
        # 1. 价格动量指标
        momentum_5 = df['close'] / df['close'].shift(5) - 1
        momentum_10 = df['close'] / df['close'].shift(10) - 1
        
        # 2. 波动率指标
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # 3. 支撑阻力分析
        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        position_in_range = (df['close'] - low_20) / (high_20 - low_20)
        
        # 4. 成交量价格关系
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        price_volume_correlation = price_change * volume_change
        
        # 复合评分计算
        scores = pd.Series(50.0, index=data.index)  # 基准分
        
        # 动量信号 (30%)
        strong_momentum = momentum_5 > 0.03  # 5日涨幅>3%
        weak_momentum = momentum_5 < -0.03  # 5日跌幅>3%
        scores += np.where(strong_momentum, 20, 0)
        scores += np.where(weak_momentum, -15, 0)
        
        # 中期动量确认 (20%)
        medium_momentum = momentum_10 > 0.05  # 10日涨幅>5%
        scores += np.where(medium_momentum, 15, 0)
        
        # 位置分析 (25%)
        near_high = position_in_range > 0.8  # 接近20日高点
        near_low = position_in_range < 0.2   # 接近20日低点
        middle_range = (position_in_range >= 0.4) & (position_in_range <= 0.6)
        scores += np.where(near_low, 15, 0)  # 低位买入机会
        scores += np.where(near_high, -10, 0)  # 高位减分
        scores += np.where(middle_range, 5, 0)  # 中位加分
        
        # 量价关系 (15%)
        positive_correlation = price_volume_correlation > 0  # 量价同向
        scores += np.where(positive_correlation, 8, -3)
        
        # 波动率调整 (10%)
        low_volatility = volatility < volatility.rolling(window=60).mean()
        scores += np.where(low_volatility, 5, -2)  # 低波动率有利
        
        # 突破信号检测
        breakout_high = df['close'] > high_20.shift(1)  # 突破20日高点
        breakdown_low = df['close'] < low_20.shift(1)   # 跌破20日低点
        scores += np.where(breakout_high, 15, 0)
        scores += np.where(breakdown_low, -20, 0)
        
        # 限制评分范围
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)


# 为了向后兼容，创建别名
composite_indicator = COMPOSITE_INDICATOR