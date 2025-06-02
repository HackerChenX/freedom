#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
趋势分类系统

实现细粒度的趋势分类，区分上涨趋势、下跌趋势和盘整
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import logging

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class TrendType(Enum):
    """趋势类型枚举"""
    STRONG_UPTREND = 5     # 强势上涨趋势
    UPTREND = 4            # 上涨趋势
    WEAK_UPTREND = 3       # 弱势上涨趋势
    CONSOLIDATION = 2      # 盘整
    WEAK_DOWNTREND = 1     # 弱势下跌趋势
    DOWNTREND = 0          # 下跌趋势
    STRONG_DOWNTREND = -1  # 强势下跌趋势


class TrendClassification(BaseIndicator):
    """
    趋势分类系统
    
    实现细粒度的趋势分类，区分不同强度的上涨趋势、下跌趋势和盘整
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化趋势分类系统
        
        Args:
            params: 配置参数字典，可包含以下键:
                - short_period: 短期均线周期，默认为20
                - medium_period: 中期均线周期，默认为60
                - long_period: 长期均线周期，默认为120
                - trend_atr_period: 趋势波动率计算周期，默认为14
                - trend_strength_period: 趋势强度计算周期，默认为10
                - consolidation_threshold: 盘整识别阈值，默认为0.03 (3%)
        """
        super().__init__()
        self._params = params if params is not None else {}
        self._initialize_params()
        
    def _initialize_params(self):
        """初始化参数，设置默认值"""
        # 移动平均线参数
        self.short_period = self._params.get('short_period', 20)
        self.medium_period = self._params.get('medium_period', 60)
        self.long_period = self._params.get('long_period', 120)
        
        # 趋势识别参数
        self.trend_atr_period = self._params.get('trend_atr_period', 14)
        self.trend_strength_period = self._params.get('trend_strength_period', 10)
        self.consolidation_threshold = self._params.get('consolidation_threshold', 0.03)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势分类
        
        Args:
            df: 输入DataFrame，需包含OHLC数据
            
        Returns:
            添加了趋势分类结果的DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for trend classification")
            return df
        
        result_df = df.copy()
        
        # 确保DataFrame中有必要的OHLC数据
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            if 'close' not in result_df.columns:
                raise ValueError("DataFrame must contain at least 'close' column")
        
        # 计算移动平均线
        result_df[f'ma_{self.short_period}'] = result_df['close'].rolling(window=self.short_period).mean()
        result_df[f'ma_{self.medium_period}'] = result_df['close'].rolling(window=self.medium_period).mean()
        result_df[f'ma_{self.long_period}'] = result_df['close'].rolling(window=self.long_period).mean()
        
        # 计算趋势方向
        result_df = self._calculate_trend_direction(result_df)
        
        # 计算趋势强度
        result_df = self._calculate_trend_strength(result_df)
        
        # 分类趋势
        result_df = self._classify_trend(result_df)
        
        return result_df 

    def _calculate_trend_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势方向
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了趋势方向指标的DataFrame
        """
        result_df = df.copy()
        
        # 短期趋势：短期均线与价格的关系
        result_df['short_trend'] = np.where(
            result_df['close'] > result_df[f'ma_{self.short_period}'], 1,
            np.where(result_df['close'] < result_df[f'ma_{self.short_period}'], -1, 0)
        )
        
        # 中期趋势：中期均线与短期均线的关系
        result_df['medium_trend'] = np.where(
            result_df[f'ma_{self.short_period}'] > result_df[f'ma_{self.medium_period}'], 1,
            np.where(result_df[f'ma_{self.short_period}'] < result_df[f'ma_{self.medium_period}'], -1, 0)
        )
        
        # 长期趋势：长期均线的斜率
        result_df['long_ma_slope'] = result_df[f'ma_{self.long_period}'].diff(periods=5) / result_df[f'ma_{self.long_period}'].shift(5)
        result_df['long_trend'] = np.where(
            result_df['long_ma_slope'] > 0.001, 1,
            np.where(result_df['long_ma_slope'] < -0.001, -1, 0)
        )
        
        # 均线多空排列
        # 计算均线排列得分
        result_df['ma_alignment'] = 0
        
        # 多头排列：短期 > 中期 > 长期
        bull_alignment = (result_df[f'ma_{self.short_period}'] > result_df[f'ma_{self.medium_period}']) & \
                         (result_df[f'ma_{self.medium_period}'] > result_df[f'ma_{self.long_period}'])
        
        # 空头排列：短期 < 中期 < 长期
        bear_alignment = (result_df[f'ma_{self.short_period}'] < result_df[f'ma_{self.medium_period}']) & \
                         (result_df[f'ma_{self.medium_period}'] < result_df[f'ma_{self.long_period}'])
        
        # 均线排列得分
        result_df.loc[bull_alignment, 'ma_alignment'] = 1
        result_df.loc[bear_alignment, 'ma_alignment'] = -1
        
        # 计算综合趋势方向得分
        result_df['trend_direction'] = result_df['short_trend'] * 0.4 + \
                                      result_df['medium_trend'] * 0.3 + \
                                      result_df['long_trend'] * 0.3 + \
                                      result_df['ma_alignment'] * 0.5
        
        return result_df
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势强度
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了趋势强度指标的DataFrame
        """
        result_df = df.copy()
        
        # 1. 计算波动率/ATR
        # 计算True Range
        result_df['high_low'] = result_df['high'] - result_df['low']
        result_df['high_close'] = np.abs(result_df['high'] - result_df['close'].shift(1))
        result_df['low_close'] = np.abs(result_df['low'] - result_df['close'].shift(1))
        
        result_df['tr'] = result_df[['high_low', 'high_close', 'low_close']].max(axis=1)
        result_df['atr'] = result_df['tr'].rolling(window=self.trend_atr_period).mean()
        
        # 标准化ATR（相对于价格）
        result_df['atr_pct'] = result_df['atr'] / result_df['close'] * 100
        
        # 2. 计算价格变化的一致性
        # 计算收益率
        result_df['return'] = result_df['close'].pct_change()
        
        # 计算前N个周期中正收益和负收益的数量
        for i in range(self.trend_strength_period, len(result_df)):
            window = result_df.iloc[i-self.trend_strength_period:i]
            positive_days = (window['return'] > 0).sum()
            negative_days = (window['return'] < 0).sum()
            
            # 计算一致性指标
            if result_df.iloc[i]['trend_direction'] > 0:  # 上涨趋势
                consistency = positive_days / self.trend_strength_period
            elif result_df.iloc[i]['trend_direction'] < 0:  # 下跌趋势
                consistency = negative_days / self.trend_strength_period
            else:  # 盘整
                consistency = 0.5
            
            result_df.at[result_df.index[i], 'price_consistency'] = consistency
        
        # 3. 计算趋势强度
        # 价格相对于均线的位置
        result_df['price_to_ma_short'] = (result_df['close'] - result_df[f'ma_{self.short_period}']) / result_df[f'ma_{self.short_period}'] * 100
        result_df['price_to_ma_medium'] = (result_df['close'] - result_df[f'ma_{self.medium_period}']) / result_df[f'ma_{self.medium_period}'] * 100
        
        # 基于价格与均线距离的趋势强度
        result_df['ma_distance'] = np.abs(result_df['price_to_ma_short'])
        
        # 计算趋势强度分数
        # 价格一致性: 0.3, 波动率: 0.3, 均线距离: 0.4
        result_df['trend_strength'] = 0.0
        
        # 需要确保price_consistency在所有行都有值
        if 'price_consistency' in result_df.columns:
            mask = ~result_df['price_consistency'].isna()
            result_df.loc[mask, 'trend_strength'] = (
                result_df.loc[mask, 'price_consistency'] * 0.3 +
                (result_df.loc[mask, 'atr_pct'] / 5).clip(0, 1) * 0.3 +  # 标准化ATR
                (result_df.loc[mask, 'ma_distance'] / 10).clip(0, 1) * 0.4  # 标准化均线距离
            )
        
        # 调整趋势强度，使其与方向一致
        result_df['trend_strength'] = result_df['trend_strength'] * np.sign(result_df['trend_direction'])
        
        return result_df
    
    def _classify_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据趋势方向和强度分类趋势
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了趋势分类结果的DataFrame
        """
        result_df = df.copy()
        
        # 初始化趋势类型列
        result_df['trend_type'] = None
        result_df['trend_type_value'] = np.nan
        
        # 基于趋势方向和强度分类趋势
        for i in range(len(result_df)):
            direction = result_df.iloc[i]['trend_direction'] if 'trend_direction' in result_df.columns else 0
            strength = result_df.iloc[i]['trend_strength'] if 'trend_strength' in result_df.columns else 0
            
            # 检查盘整条件
            is_consolidation = False
            
            # 如果有足够的历史数据，检查价格是否在狭窄范围内震荡
            if i >= 20:
                price_range = (result_df.iloc[i-20:i]['high'].max() - result_df.iloc[i-20:i]['low'].min()) / result_df.iloc[i]['close']
                if price_range < self.consolidation_threshold:
                    is_consolidation = True
            
            # 趋势分类逻辑
            if is_consolidation or abs(direction) < 0.2:
                # 盘整
                trend_type = TrendType.CONSOLIDATION
                trend_value = 2
            elif direction > 0:
                # 上涨趋势
                if strength > 0.7:
                    trend_type = TrendType.STRONG_UPTREND
                    trend_value = 5
                elif strength > 0.3:
                    trend_type = TrendType.UPTREND
                    trend_value = 4
                else:
                    trend_type = TrendType.WEAK_UPTREND
                    trend_value = 3
            else:
                # 下跌趋势
                if strength < -0.7:
                    trend_type = TrendType.STRONG_DOWNTREND
                    trend_value = -1
                elif strength < -0.3:
                    trend_type = TrendType.DOWNTREND
                    trend_value = 0
                else:
                    trend_type = TrendType.WEAK_DOWNTREND
                    trend_value = 1
            
            # 更新趋势类型
            result_df.at[result_df.index[i], 'trend_type'] = trend_type.name
            result_df.at[result_df.index[i], 'trend_type_value'] = trend_value
        
        # 添加趋势持续天数
        result_df['trend_duration'] = 1  # 默认为1天
        
        for i in range(1, len(result_df)):
            curr_type = result_df.iloc[i]['trend_type_value']
            prev_type = result_df.iloc[i-1]['trend_type_value']
            
            # 如果当前趋势类型与前一天相同，增加持续天数
            if curr_type == prev_type:
                result_df.at[result_df.index[i], 'trend_duration'] = result_df.iloc[i-1]['trend_duration'] + 1
        
        # 添加趋势改变标志
        result_df['trend_change'] = False
        for i in range(1, len(result_df)):
            curr_type = result_df.iloc[i]['trend_type_value']
            prev_type = result_df.iloc[i-1]['trend_type_value']
            
            if curr_type != prev_type:
                result_df.at[result_df.index[i], 'trend_change'] = True
        
        return result_df
    
    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据趋势变化生成交易信号
        
        Args:
            df: 输入DataFrame，需包含趋势分类结果
            
        Returns:
            添加了交易信号的DataFrame
        """
        result_df = df.copy()
        
        # 初始化信号列
        result_df['trend_signal'] = 0
        
        # 生成趋势信号
        for i in range(1, len(result_df)):
            curr_type = result_df.iloc[i]['trend_type_value'] if 'trend_type_value' in result_df.columns else 0
            prev_type = result_df.iloc[i-1]['trend_type_value'] if 'trend_type_value' in result_df.columns else 0
            
            # 趋势改变信号
            if curr_type != prev_type:
                # 转为上升趋势（买入信号）
                if curr_type >= 3 and prev_type < 3:
                    result_df.at[result_df.index[i], 'trend_signal'] = 1
                
                # 转为下降趋势（卖出信号）
                elif curr_type <= 1 and prev_type > 1:
                    result_df.at[result_df.index[i], 'trend_signal'] = -1
        
        return result_df 

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分(0-100)
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
    
        # 在这里实现指标特定的评分逻辑
        # 此处提供默认实现
    
        return score
        
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
    
        return signals