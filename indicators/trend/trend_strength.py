#!/usr/bin/env python
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
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class TrendStrength(BaseIndicator):
    """
    趋势强度指标
    
    通过价格动量、波动方向一致性和趋势持续时间来评估趋势的强度
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化趋势强度指标
        
        Args:
            params: 参数字典，可包含：
                - lookback_period: 回溯周期，默认为20
                - min_strength: 最小强度阈值，默认为30
                - strong_threshold: 强趋势阈值，默认为70
        """
        super().__init__(name="TrendStrength", description="趋势强度指标")
        
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
        计算趋势强度指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外的参数
            
        Returns:
            添加了趋势强度指标的DataFrame
        """
        df = data.copy()
        
        # 提取参数
        lookback_period = self.params["lookback_period"]
        min_strength = self.params["min_strength"]
        strong_threshold = self.params["strong_threshold"]
        
        # 确保数据有足够的长度
        if len(df) < lookback_period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({lookback_period + 1})，返回原始数据")
            df['trend_strength'] = np.nan
            df['trend_direction'] = np.nan
            df['trend_category'] = np.nan
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
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算趋势强度指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if not self.has_result():
            result = self.calculate(data)
        else:
            result = self._result
        
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