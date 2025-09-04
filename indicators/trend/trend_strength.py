#!/usr/bin/env python
from utils.dependency_injection import get_logger
# -*- coding: utf-8 -*-

"""
趋势强度指标

评估市场趋势的强度和可靠性
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class TrendStrength(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    趋势强度指标
    
    通过价格动量、波动方向一致性和趋势持续时间来评估趋势的强度
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化趋势强度指标

        Args:
            params: 参数字典，可包含：
                - lookback_period: 回溯周期，默认为20
                - min_strength: 最小强度阈值，默认为30
                - strong_threshold: 强趋势阈值，默认为70
        """
        # 🔧 Ultra Think修复：修复初始化问题
        super().__init__()
        self.name = "TREND_STRENGTH"
        self.description = "趋势强度指标"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
        # 设置默认参数
        self.params = {
            "lookback_period": 20,
            "min_strength": 30,
            "strong_threshold": 70
        }
        
        # 更新自定义参数
        if params:
            self.params.update(params)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算趋势强度指标 - 主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            添加了趋势强度指标的DataFrame
        """
        return self._calculate_trendstrength(data, **kwargs)

    def _calculate_trendstrength(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算趋势强度指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            **kwargs: 额外的参数
            
        Returns:
            添加了趋势强度指标的Data_frame
        """
        df = data.copy()
        
        # 提取参数
        lookback_period = self.params["lookback_period"]
        min_strength = self.params["min_strength"]
        strong_threshold = self.params["strong_threshold"]
        
        # 🔧 Ultra Think修复：移除提前返回，确保核心计算代码能执行
        # 确保数据有足够的长度
        if len(df) < lookback_period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({lookback_period + 1})，使用可用数据计算")
            # 不提前返回，继续执行计算

        # 🔧 Ultra Think修复：添加真实的趋势强度计算算法
        # 1. 计算价格变化率
        df['price_change'] = df['close'].pct_change()

        # 2. 计算移动平均趋势
        short_ma = df['close'].rolling(window=min(10, len(df)), min_periods=1).mean()
        long_ma = df['close'].rolling(window=min(lookback_period, len(df)), min_periods=1).mean()

        # 3. 计算趋势方向
        trend_direction = []
        for i in range(len(df)):
            if i < 1:
                trend_direction.append('neutral')
            else:
                if short_ma.iloc[i] > long_ma.iloc[i]:
                    if short_ma.iloc[i] > short_ma.iloc[i-1]:
                        trend_direction.append('uptrend')
                    else:
                        trend_direction.append('neutral')
                elif short_ma.iloc[i] < long_ma.iloc[i]:
                    if short_ma.iloc[i] < short_ma.iloc[i-1]:
                        trend_direction.append('downtrend')
                    else:
                        trend_direction.append('neutral')
                else:
                    trend_direction.append('neutral')

        df['trend_direction'] = trend_direction

        # 4. 计算趋势强度 (0-100)
        trend_strength = []
        for i in range(len(df)):
            if i < lookback_period:
                trend_strength.append(50.0)  # 默认中性强度
                continue

            # 计算价格相对于移动平均的偏离程度
            price_deviation = abs(df['close'].iloc[i] - long_ma.iloc[i]) / long_ma.iloc[i]

            # 计算价格变化的一致性
            recent_changes = df['price_change'].iloc[i-min(5, i):i+1]
            if len(recent_changes) > 0:
                consistency = abs(recent_changes.mean()) / (recent_changes.std() + 1e-8)
            else:
                consistency = 0

            # 综合计算趋势强度
            strength = min(100, max(0, (price_deviation * 1000 + consistency * 20) * 2))
            trend_strength.append(strength)

        df['trend_strength'] = trend_strength

        # 5. 计算趋势类别
        trend_category = []
        for i in range(len(df)):
            strength = trend_strength[i]
            direction = trend_direction[i]

            if strength > 70:
                if direction == 'uptrend':
                    trend_category.append('strong_bullish')
                elif direction == 'downtrend':
                    trend_category.append('strong_bearish')
                else:
                    trend_category.append('strong_neutral')
            elif strength > 40:
                if direction == 'uptrend':
                    trend_category.append('moderate_bullish')
                elif direction == 'downtrend':
                    trend_category.append('moderate_bearish')
                else:
                    trend_category.append('moderate_neutral')
            else:
                trend_category.append('weak')

        df['trend_category'] = trend_category

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
        
        # 计算价格变化百分比
        df['price_change_pct'] = df['close'].pct_change(periods=1) * 100
        
        # 计算价格动量 (N日价格变化)
        df['price_momentum'] = df['close'].pct_change(periods=lookback_period) * 100
        
        # 计算方向一致性 (正向变化的天数比例)
        df['direction_consistency'] = df['price_change_pct'].rolling(window=lookback_period).apply(
            lambda x: np.sum(x > 0) / len(x) * 100 if len(x) > 0 else np.nan
        )
        
        # 计算波动率 (标准差)
        df['volatility'] = df['price_change_pct'].rolling(window=lookback_period).std()
        
        # 计算趋势强度得分 (0-100)
        df['trend_strength'] = 0.0
        
        # 对有足够数据的行计算趋势强度
        mask = ~df['price_momentum'].isna() & ~df['direction_consistency'].isna() & ~df['volatility'].isna()
        
        if mask.any():
            # 价格动量的绝对值 (0-100)
            momentum_abs = df.loc[mask, 'price_momentum'].abs()
            max_momentum = max(momentum_abs.max(), 20)  # 使用至少20作为最大值，避免较小的变化导致过高的分数
            momentum_score = momentum_abs / max_momentum * 40  # 贡献40%的权重
            momentum_score = momentum_score.clip(0, 40)
            
            # 方向一致性 (0-40)
            consistency_score = (df.loc[mask, 'direction_consistency'] - 50) * 0.8  # 贡献40%的权重
            consistency_score = consistency_score.clip(0, 40)
            
            # 低波动性奖励 (0-20)
            volatility_median = df.loc[mask, 'volatility'].median()
            volatility_score = 20 - (df.loc[mask, 'volatility'] / volatility_median * 10).clip(0, 20)  # 贡献20%的权重
            
            # 合并得分
            df.loc[mask, 'trend_strength'] = (momentum_score + consistency_score + volatility_score).clip(0, 100)
        
        # 确定趋势方向
        df['trend_direction'] = np.where(df['price_momentum'] > 0, 'uptrend', 
                                         np.where(df['price_momentum'] < 0, 'downtrend', 'neutral'))
        
        # 确定趋势类别
        df['trend_category'] = np.where(df['trend_strength'] >= strong_threshold, 'strong',
                                        np.where(df['trend_strength'] >= min_strength, 'moderate', 'weak'))
        
        # 清理中间计算列
        df.drop(['price_change_pct', 'volatility'], axis=1, inplace=True)
        
        # 保存结果
        self._result = df
        
        return df
    
    def calculate_raw_score_Strength(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算趋势强度指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 🔧 Ultra Think修复：移除has_result检查，直接计算
        # 确保已计算指标
        # if not self.has_result():
        result = self._calculate_trendstrength(data)
        # else:
        #     result = self._result
        
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 检查结果是否有效
        if result.empty or 'trend_strength' not in result.columns or 'trend_direction' not in result.columns:
            return score
        
        # 从结果中提取趋势强度和方向
        trend_strength = result['trend_strength']
        trend_direction = result['trend_direction']
        
        # 计算评分：
        # 1. 上升趋势：根据强度映射到50-100分
        # 2. 下降趋势：根据强度映射到0-50分
        # 3. 中性：保持50分
        
        # 上升趋势评分 (50-100)
        uptrend_mask = trend_direction == 'uptrend'
        if uptrend_mask.any():
            score.loc[uptrend_mask] = 50 + trend_strength.loc[uptrend_mask] / 2
        
        # 下降趋势评分 (0-50)
        downtrend_mask = trend_direction == 'downtrend'
        if downtrend_mask.any():
            score.loc[downtrend_mask] = 50 - trend_strength.loc[downtrend_mask] / 2
        
        # 处理可能的缺失值
        score = score.fillna(50.0)
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score 

    def get_pattern_info_Strength(self, pattern_id: str) -> dict:
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

    # 🔧 Ultra Think修复：实现BaseIndicator要求的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的_calculate_baseindicator方法"""
        return self._calculate_trendstrength(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """实现BaseIndicator要求的置信度计算方法"""
        # 基于趋势强度计算置信度
        if len(score) == 0:
            return 0.5

        # 计算评分的标准差，标准差越小置信度越高
        score_std = score.std()
        if pd.isna(score_std) or score_std == 0:
            return 0.8

        # 标准差越小，置信度越高
        confidence = max(0.3, min(0.9, 1.0 - score_std / 50.0))
        return confidence

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现BaseIndicator要求的原始评分计算方法"""
        return self.calculate_raw_score_Strength(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的形态获取方法"""
        # 基于趋势强度生成形态
        result = self._calculate_trendstrength(data, **kwargs)
        patterns = pd.DataFrame(index=data.index)

        if 'trend_strength' in result.columns and 'trend_direction' in result.columns:
            # 强趋势形态
            strong_up = (result['trend_strength'] > 70) & (result['trend_direction'] == 'uptrend')
            strong_down = (result['trend_strength'] > 70) & (result['trend_direction'] == 'downtrend')

            patterns['strong_uptrend'] = strong_up
            patterns['strong_downtrend'] = strong_down
            patterns['weak_trend'] = result['trend_strength'] < 30

        return patterns

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现BaseIndicator要求的参数设置方法"""
        if kwargs:
            self.params.update(kwargs)

    @property
    def minimum_periods(self) -> int:
        """实现MinimumPeriodsMixin要求的minimum_periods属性"""
        return self.params.get("lookback_period", 20) + 10