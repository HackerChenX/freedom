#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动态止损模型

基于技术指标的动态止损系统，根据市场环境和个股特性自适应调整止损位置
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import logging

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class StopLossType(Enum):
    """止损类型枚举"""
    FIXED = 0           # 固定止损
    TRAILING = 1        # 跟踪止损
    VOLATILITY = 2      # 波动率止损
    INDICATOR = 3       # 指标止损
    SUPPORT = 4         # 支撑位止损
    COMPOSITE = 5       # 复合止损


class DynamicStopLoss:
    """
    动态止损模型
    
    基于技术指标和市场环境动态调整止损策略，提高止损有效性同时保留盈利空间
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化动态止损模型
        
        Args:
            params: 配置参数字典，可包含以下键:
                - base_percentage: 基础止损百分比，默认为3%
                - atr_multiplier: ATR乘数，默认为2
                - trailing_activation: 激活跟踪止损的收益率，默认为2%
                - volatility_lookback: 波动率计算回溯期，默认为20天
                - max_stop_loss: 最大止损百分比，默认为15%
                - min_stop_loss: 最小止损百分比，默认为1%
                - indicator_weights: 各指标权重，默认为均等权重
        """
        self._params = params or {}
        self._initialize_params()
        self._result = None
        
    def _initialize_params(self):
        """初始化参数，设置默认值"""
        # 基础参数
        self.base_percentage = self._params.get('base_percentage', 3.0)
        self.atr_multiplier = self._params.get('atr_multiplier', 2.0)
        self.trailing_activation = self._params.get('trailing_activation', 2.0)
        self.volatility_lookback = self._params.get('volatility_lookback', 20)
        self.max_stop_loss = self._params.get('max_stop_loss', 15.0)
        self.min_stop_loss = self._params.get('min_stop_loss', 1.0)
        
        # 指标权重
        default_weights = {
            'volatility': 0.3,
            'trend': 0.3,
            'support': 0.2,
            'volume': 0.1,
            'momentum': 0.1
        }
        self.indicator_weights = self._params.get('indicator_weights', default_weights)
        
    def calculate_dynamic_stop_loss(self, df: pd.DataFrame, position_type: str = 'long',
                                   stop_loss_type: StopLossType = StopLossType.COMPOSITE,
                                   entry_price: Optional[float] = None) -> pd.DataFrame:
        """
        计算动态止损位置
        
        Args:
            df: 输入DataFrame，需包含OHLC数据
            position_type: 持仓类型，'long'或'short'
            stop_loss_type: 止损类型
            entry_price: 入场价格，如果为None则使用每个K线的收盘价
            
        Returns:
            添加了止损价格和止损百分比的DataFrame
        """
        result_df = df.copy()
        
        # 确保DataFrame中有必要的OHLC和成交量数据
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            if 'close' not in result_df.columns:
                raise ValueError("DataFrame must contain at least 'close' column")
        
        # 确保ATR已计算
        if 'atr' not in result_df.columns:
            result_df = self._calculate_atr(result_df)
        
        # 基于不同的止损类型，计算止损位置
        if stop_loss_type == StopLossType.FIXED:
            result_df = self._calculate_fixed_stop_loss(result_df, position_type, entry_price)
        
        elif stop_loss_type == StopLossType.TRAILING:
            result_df = self._calculate_trailing_stop_loss(result_df, position_type, entry_price)
        
        elif stop_loss_type == StopLossType.VOLATILITY:
            result_df = self._calculate_volatility_stop_loss(result_df, position_type, entry_price)
        
        elif stop_loss_type == StopLossType.INDICATOR:
            # 计算趋势强度作为指标止损的依据
            if 'trend_score' not in result_df.columns:
                result_df = self._calculate_trend_strength(result_df)
            
            result_df = self._calculate_indicator_stop_loss(result_df, position_type, entry_price)
        
        elif stop_loss_type == StopLossType.SUPPORT:
            # 计算支撑位和阻力位
            if not any(col.startswith('support_') for col in result_df.columns):
                result_df = self._calculate_support_resistance(result_df)
            
            result_df = self._calculate_support_stop_loss(result_df, position_type, entry_price)
        
        elif stop_loss_type == StopLossType.COMPOSITE:
            # 计算市场环境和动量指标作为复合止损的依据
            if 'trend_score' not in result_df.columns:
                result_df = self._calculate_trend_strength(result_df)
            
            if not any(col.startswith('support_') for col in result_df.columns):
                result_df = self._calculate_support_resistance(result_df)
            
            if 'momentum_score' not in result_df.columns:
                result_df = self._calculate_momentum(result_df)
            
            if 'volume_score' not in result_df.columns:
                result_df = self._calculate_volume_profile(result_df)
            
            result_df = self._calculate_composite_stop_loss(result_df, position_type, entry_price)
        
        # 添加额外的诊断信息
        result_df['stop_loss_type'] = stop_loss_type.name
        result_df['position_type'] = position_type
        
        # 计算离止损的距离百分比（相对于当前价格）
        if position_type.lower() == 'long':
            result_df['distance_to_stop'] = (result_df['close'] - result_df['stop_loss_price']) / result_df['close'] * 100
        else:
            result_df['distance_to_stop'] = (result_df['stop_loss_price'] - result_df['close']) / result_df['close'] * 100
        
        # 标记止损距离异常值
        mean_distance = result_df['distance_to_stop'].mean()
        std_distance = result_df['distance_to_stop'].std()
        
        result_df['distance_zscore'] = (result_df['distance_to_stop'] - mean_distance) / std_distance
        result_df['stop_loss_flag'] = np.where(abs(result_df['distance_zscore']) > 2, 'Warning', 'Normal')
        
        return result_df
    
    def _calculate_fixed_stop_loss(self, df: pd.DataFrame, position_type: str, entry_price: Optional[float]) -> pd.DataFrame:
        """
        计算固定百分比止损
        
        Args:
            df: 输入DataFrame
            position_type: 持仓类型，'long'或'short'
            entry_price: 入场价格，如果为None则使用每个K线的收盘价
            
        Returns:
            添加了固定止损价格的DataFrame
        """
        result_df = df.copy()
        
        # 添加止损距离列
        result_df['stop_loss_pct'] = self.base_percentage
        
        # 计算动态风险系数，基于波动率和趋势调整基础止损比例
        if 'atr_ratio' in result_df.columns and 'trend_score' in result_df.columns:
            # 高波动率和强趋势需要更宽的止损
            volatility_factor = result_df['atr_ratio'].clip(0.5, 2.0)
            trend_factor = result_df['trend_score'].apply(lambda x: 0.8 + (x / 100) * 0.4)  # 0.8-1.2范围
            
            # 调整止损百分比
            result_df['stop_loss_pct'] = result_df['stop_loss_pct'] * volatility_factor * trend_factor
        
        # 限制止损在允许范围内
        result_df['stop_loss_pct'] = result_df['stop_loss_pct'].clip(self.min_stop_loss, self.max_stop_loss)
        
        # 计算具体止损价格
        if entry_price is not None:
            # 使用固定入场价
            if position_type.lower() == 'long':
                result_df['stop_loss_price'] = entry_price * (1 - result_df['stop_loss_pct'] / 100)
            else:  # short
                result_df['stop_loss_price'] = entry_price * (1 + result_df['stop_loss_pct'] / 100)
        else:
            # 使用收盘价作为假设的入场价
            if position_type.lower() == 'long':
                result_df['stop_loss_price'] = result_df['close'] * (1 - result_df['stop_loss_pct'] / 100)
            else:  # short
                result_df['stop_loss_price'] = result_df['close'] * (1 + result_df['stop_loss_pct'] / 100)
        
        return result_df

    def _calculate_trailing_stop_loss(self, df: pd.DataFrame, position_type: str, entry_price: Optional[float]) -> pd.DataFrame:
        """
        计算跟踪止损
        
        Args:
            df: 输入DataFrame
            position_type: 持仓类型，'long'或'short'
            entry_price: 入场价格，如果为None则使用每个K线的收盘价
            
        Returns:
            添加了跟踪止损价格的DataFrame
        """
        result_df = df.copy()
        
        # 计算激活跟踪止损所需的利润率
        activation_threshold = self.trailing_activation
        
        # 初始化高点/低点和止损价格
        result_df['highest_price'] = np.nan
        result_df['lowest_price'] = np.nan
        result_df['trailing_stop_price'] = np.nan
        
        # 复制固定止损作为基础
        fixed_stop_df = self._calculate_fixed_stop_loss(df, position_type, entry_price)
        result_df['stop_loss_pct'] = fixed_stop_df['stop_loss_pct']
        
        # 特别调整跟踪止损幅度
        # 在强趋势中可以收紧跟踪止损，在震荡市场中需要放宽
        if 'trend_score' in result_df.columns:
            trend_adjustment = result_df['trend_score'].apply(lambda x: 1 - (x / 200))  # 高趋势分数意味着收紧止损
            result_df['stop_loss_pct'] = result_df['stop_loss_pct'] * trend_adjustment
            
        # 限制止损在允许范围内
        result_df['stop_loss_pct'] = result_df['stop_loss_pct'].clip(self.min_stop_loss, self.max_stop_loss)
        
        # 计算跟踪止损
        # 对于多头：从最高价回撤指定百分比
        # 对于空头：从最低价上涨指定百分比
        if entry_price is not None:
            # 使用固定入场价
            if position_type.lower() == 'long':
                for i in range(len(result_df)):
                    if i == 0:
                        result_df.at[result_df.index[i], 'highest_price'] = max(entry_price, result_df.iloc[i]['high'])
                    else:
                        # 更新历史最高价
                        result_df.at[result_df.index[i], 'highest_price'] = max(
                            result_df.iloc[i-1]['highest_price'],
                            result_df.iloc[i]['high']
                        )
                    
                    # 计算当前收益率
                    current_price = result_df.iloc[i]['close']
                    profit_pct = (current_price - entry_price) / entry_price * 100
                    
                    # 如果收益率超过激活阈值，使用跟踪止损，否则使用固定止损
                    if profit_pct >= activation_threshold:
                        stop_pct = result_df.iloc[i]['stop_loss_pct']
                        result_df.at[result_df.index[i], 'trailing_stop_price'] = result_df.iloc[i]['highest_price'] * (1 - stop_pct / 100)
                    else:
                        result_df.at[result_df.index[i], 'trailing_stop_price'] = fixed_stop_df.iloc[i]['stop_loss_price']
            
            else:  # short
                for i in range(len(result_df)):
                    if i == 0:
                        result_df.at[result_df.index[i], 'lowest_price'] = min(entry_price, result_df.iloc[i]['low'])
                    else:
                        # 更新历史最低价
                        result_df.at[result_df.index[i], 'lowest_price'] = min(
                            result_df.iloc[i-1]['lowest_price'],
                            result_df.iloc[i]['low']
                        )
                    
                    # 计算当前收益率 (对于空头，价格下跌意味着盈利)
                    current_price = result_df.iloc[i]['close']
                    profit_pct = (entry_price - current_price) / entry_price * 100
                    
                    # 如果收益率超过激活阈值，使用跟踪止损，否则使用固定止损
                    if profit_pct >= activation_threshold:
                        stop_pct = result_df.iloc[i]['stop_loss_pct']
                        result_df.at[result_df.index[i], 'trailing_stop_price'] = result_df.iloc[i]['lowest_price'] * (1 + stop_pct / 100)
                    else:
                        result_df.at[result_df.index[i], 'trailing_stop_price'] = fixed_stop_df.iloc[i]['stop_loss_price']
        
        else:
            # 使用每个周期的收盘价作为假设入场价
            if position_type.lower() == 'long':
                # 对于无入场价的情况，直接使用ATR倍数来计算跟踪止损
                if 'atr' in result_df.columns:
                    # 使用当前ATR的倍数作为止损距离
                    result_df['trailing_stop_price'] = result_df['close'] - self.atr_multiplier * result_df['atr']
                else:
                    # 无ATR时回退到固定百分比止损
                    result_df['trailing_stop_price'] = result_df['close'] * (1 - result_df['stop_loss_pct'] / 100)
            
            else:  # short
                if 'atr' in result_df.columns:
                    result_df['trailing_stop_price'] = result_df['close'] + self.atr_multiplier * result_df['atr']
                else:
                    result_df['trailing_stop_price'] = result_df['close'] * (1 + result_df['stop_loss_pct'] / 100)
        
        # 最终止损价格 = 跟踪止损价格
        result_df['stop_loss_price'] = result_df['trailing_stop_price']
        
        # 删除临时列
        result_df = result_df.drop(['trailing_stop_price'], axis=1)
        
        return result_df
    
    def _calculate_volatility_stop_loss(self, df: pd.DataFrame, position_type: str, entry_price: Optional[float]) -> pd.DataFrame:
        """
        计算基于波动率的止损
        
        Args:
            df: 输入DataFrame
            position_type: 持仓类型，'long'或'short'
            entry_price: 入场价格，如果为None则使用每个K线的收盘价
            
        Returns:
            添加了波动率止损价格的DataFrame
        """
        result_df = df.copy()
        
        # 确保ATR已经计算
        if 'atr' not in result_df.columns:
            result_df = self._calculate_atr(result_df)
        
        # 动态ATR乘数，根据趋势强度和市场环境调整
        atr_multiplier = self.atr_multiplier
        
        # 根据趋势强度调整ATR乘数
        if 'trend_score' in result_df.columns:
            # 强趋势下可以使用较大的乘数以避免被震出
            trend_adjustment = result_df['trend_score'].apply(lambda x: 0.8 + (x / 100) * 0.6)  # 0.8-1.4范围
            atr_multiplier = atr_multiplier * trend_adjustment
        
        # 根据市场波动性调整ATR乘数
        if 'atr_ratio' in result_df.columns:
            # 高波动率市场需要更大的乘数
            volatility_adjustment = result_df['atr_ratio'].clip(0.8, 1.5)
            atr_multiplier = atr_multiplier * volatility_adjustment
        
        # 计算波动率止损
        if position_type.lower() == 'long':
            if entry_price is not None:
                # 使用ATR设置止损位
                result_df['stop_loss_price'] = result_df.apply(
                    lambda row: min(
                        entry_price * (1 - self.max_stop_loss / 100),  # 最大止损限制
                        entry_price - row['atr'] * atr_multiplier[row.name] if isinstance(atr_multiplier, pd.Series) else entry_price - row['atr'] * atr_multiplier
                    ),
                    axis=1
                )
            else:
                # 使用当前收盘价设置止损位
                result_df['stop_loss_price'] = result_df.apply(
                    lambda row: row['close'] - row['atr'] * atr_multiplier[row.name] if isinstance(atr_multiplier, pd.Series) else row['close'] - row['atr'] * atr_multiplier,
                    axis=1
                )
        else:  # short
            if entry_price is not None:
                # 使用ATR设置止损位
                result_df['stop_loss_price'] = result_df.apply(
                    lambda row: max(
                        entry_price * (1 + self.max_stop_loss / 100),  # 最大止损限制
                        entry_price + row['atr'] * atr_multiplier[row.name] if isinstance(atr_multiplier, pd.Series) else entry_price + row['atr'] * atr_multiplier
                    ),
                    axis=1
                )
            else:
                # 使用当前收盘价设置止损位
                result_df['stop_loss_price'] = result_df.apply(
                    lambda row: row['close'] + row['atr'] * atr_multiplier[row.name] if isinstance(atr_multiplier, pd.Series) else row['close'] + row['atr'] * atr_multiplier,
                    axis=1
                )
        
        # 计算止损百分比
        if entry_price is not None:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (entry_price - result_df['stop_loss_price']) / entry_price * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['stop_loss_price'] - entry_price) / entry_price * 100
        else:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (result_df['close'] - result_df['stop_loss_price']) / result_df['close'] * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['stop_loss_price'] - result_df['close']) / result_df['close'] * 100
        
        # 限制止损百分比在允许范围内
        result_df['stop_loss_pct'] = result_df['stop_loss_pct'].clip(self.min_stop_loss, self.max_stop_loss)
        
        return result_df

    def _calculate_indicator_stop_loss(self, df: pd.DataFrame, position_type: str, entry_price: Optional[float]) -> pd.DataFrame:
        """
        计算基于技术指标的止损
        
        Args:
            df: 输入DataFrame
            position_type: 持仓类型，'long'或'short'
            entry_price: 入场价格，如果为None则使用每个K线的收盘价
            
        Returns:
            添加了指标止损价格的DataFrame
        """
        result_df = df.copy()
        
        # 使用均线作为止损参考
        ma_periods = [20, 50, 100]
        for period in ma_periods:
            ma_col = f'ma_{period}'
            if ma_col not in result_df.columns and 'close' in result_df.columns:
                result_df[ma_col] = result_df['close'].rolling(window=period).mean()
        
        # 初始化指标止损价格
        result_df['indicator_stop_price'] = np.nan
        
        # 根据持仓方向确定指标止损逻辑
        if position_type.lower() == 'long':
            # 多头使用支撑均线作为止损位
            # 首选短期均线(MA20)，如果价格已低于短期均线，则使用中期均线(MA50)
            for i in range(len(result_df)):
                current_price = result_df.iloc[i]['close']
                ma20 = result_df.iloc[i]['ma_20'] if 'ma_20' in result_df.columns else np.nan
                ma50 = result_df.iloc[i]['ma_50'] if 'ma_50' in result_df.columns else np.nan
                ma100 = result_df.iloc[i]['ma_100'] if 'ma_100' in result_df.columns else np.nan
                
                # 决定使用哪个均线作为止损
                if not pd.isna(ma20) and current_price > ma20:
                    # 价格在MA20上方，使用MA20作为止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = ma20
                elif not pd.isna(ma50) and current_price > ma50:
                    # 价格在MA50上方，使用MA50作为止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = ma50
                elif not pd.isna(ma100) and current_price > ma100:
                    # 价格在MA100上方，使用MA100作为止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = ma100
                else:
                    # 无合适均线，回退到波动率止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = np.nan
        
        else:  # short
            # 空头使用阻力均线作为止损位
            # 首选短期均线(MA20)，如果价格已高于短期均线，则使用中期均线(MA50)
            for i in range(len(result_df)):
                current_price = result_df.iloc[i]['close']
                ma20 = result_df.iloc[i]['ma_20'] if 'ma_20' in result_df.columns else np.nan
                ma50 = result_df.iloc[i]['ma_50'] if 'ma_50' in result_df.columns else np.nan
                ma100 = result_df.iloc[i]['ma_100'] if 'ma_100' in result_df.columns else np.nan
                
                # 决定使用哪个均线作为止损
                if not pd.isna(ma20) and current_price < ma20:
                    # 价格在MA20下方，使用MA20作为止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = ma20
                elif not pd.isna(ma50) and current_price < ma50:
                    # 价格在MA50下方，使用MA50作为止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = ma50
                elif not pd.isna(ma100) and current_price < ma100:
                    # 价格在MA100下方，使用MA100作为止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = ma100
                else:
                    # 无合适均线，回退到波动率止损
                    result_df.at[result_df.index[i], 'indicator_stop_price'] = np.nan
        
        # 对于缺失的止损价格，使用波动率止损作为备选
        vol_stop_df = self._calculate_volatility_stop_loss(df, position_type, entry_price)
        result_df.loc[result_df['indicator_stop_price'].isna(), 'indicator_stop_price'] = vol_stop_df.loc[result_df['indicator_stop_price'].isna(), 'stop_loss_price']
        
        # 如果有入场价，确保止损价格合理
        if entry_price is not None:
            if position_type.lower() == 'long':
                # 确保止损价格不高于入场价（防止亏损）
                result_df['indicator_stop_price'] = result_df['indicator_stop_price'].clip(upper=entry_price)
                
                # 确保止损不会太远（最大止损限制）
                min_allowed_price = entry_price * (1 - self.max_stop_loss / 100)
                result_df['indicator_stop_price'] = result_df['indicator_stop_price'].clip(lower=min_allowed_price)
            
            else:  # short
                # 确保止损价格不低于入场价（防止亏损）
                result_df['indicator_stop_price'] = result_df['indicator_stop_price'].clip(lower=entry_price)
                
                # 确保止损不会太远（最大止损限制）
                max_allowed_price = entry_price * (1 + self.max_stop_loss / 100)
                result_df['indicator_stop_price'] = result_df['indicator_stop_price'].clip(upper=max_allowed_price)
        
        # 计算止损百分比
        if entry_price is not None:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (entry_price - result_df['indicator_stop_price']) / entry_price * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['indicator_stop_price'] - entry_price) / entry_price * 100
        else:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (result_df['close'] - result_df['indicator_stop_price']) / result_df['close'] * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['indicator_stop_price'] - result_df['close']) / result_df['close'] * 100
        
        # 限制止损百分比在允许范围内
        result_df['stop_loss_pct'] = result_df['stop_loss_pct'].clip(self.min_stop_loss, self.max_stop_loss)
        
        # 最终止损价格 = 指标止损价格
        result_df['stop_loss_price'] = result_df['indicator_stop_price']
        
        # 删除临时列
        result_df = result_df.drop(['indicator_stop_price'], axis=1)
        
        return result_df
    
    def _calculate_support_stop_loss(self, df: pd.DataFrame, position_type: str, entry_price: Optional[float]) -> pd.DataFrame:
        """
        计算基于支撑位/阻力位的止损
        
        Args:
            df: 输入DataFrame
            position_type: 持仓类型，'long'或'short'
            entry_price: 入场价格，如果为None则使用每个K线的收盘价
            
        Returns:
            添加了支撑位止损价格的DataFrame
        """
        result_df = df.copy()
        
        # 确保支撑阻力位已计算
        support_cols = [col for col in result_df.columns if col.startswith('support_')]
        resistance_cols = [col for col in result_df.columns if col.startswith('resistance_')]
        
        if not support_cols or not resistance_cols:
            result_df = self._calculate_support_resistance(result_df)
            support_cols = [col for col in result_df.columns if col.startswith('support_')]
            resistance_cols = [col for col in result_df.columns if col.startswith('resistance_')]
        
        # 初始化支撑位止损价格
        result_df['support_stop_price'] = np.nan
        
        # 根据持仓方向确定支撑位止损逻辑
        if position_type.lower() == 'long':
            # 多头使用支撑位作为止损
            # 尝试找到最近的关键支撑位
            if 'key_support' in result_df.columns:
                # 优先使用基于成交量的关键支撑位
                result_df['support_stop_price'] = result_df['key_support']
            
            # 如果没有关键支撑位，使用最强的历史支撑位
            for i in range(len(result_df)):
                if pd.isna(result_df.iloc[i]['support_stop_price']):
                    # 收集所有可用的支撑位
                    supports = []
                    for col in support_cols:
                        value = result_df.iloc[i][col]
                        if not pd.isna(value):
                            supports.append(value)
                    
                    if supports:
                        # 使用距离当前价格最近的支撑位
                        current_price = result_df.iloc[i]['close']
                        closest_support = max(supports)  # 默认最高支撑位
                        min_distance = float('inf')
                        
                        for support in supports:
                            if support < current_price:  # 只考虑低于当前价格的支撑位
                                distance = current_price - support
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_support = support
                        
                        result_df.at[result_df.index[i], 'support_stop_price'] = closest_support
        
        else:  # short
            # 空头使用阻力位作为止损
            # 尝试找到最近的关键阻力位
            if 'key_resistance' in result_df.columns:
                # 优先使用基于成交量的关键阻力位
                result_df['support_stop_price'] = result_df['key_resistance']
            
            # 如果没有关键阻力位，使用最强的历史阻力位
            for i in range(len(result_df)):
                if pd.isna(result_df.iloc[i]['support_stop_price']):
                    # 收集所有可用的阻力位
                    resistances = []
                    for col in resistance_cols:
                        value = result_df.iloc[i][col]
                        if not pd.isna(value):
                            resistances.append(value)
                    
                    if resistances:
                        # 使用距离当前价格最近的阻力位
                        current_price = result_df.iloc[i]['close']
                        closest_resistance = min(resistances)  # 默认最低阻力位
                        min_distance = float('inf')
                        
                        for resistance in resistances:
                            if resistance > current_price:  # 只考虑高于当前价格的阻力位
                                distance = resistance - current_price
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_resistance = resistance
                        
                        result_df.at[result_df.index[i], 'support_stop_price'] = closest_resistance
        
        # 对于缺失的止损价格，使用ATR止损作为备选
        vol_stop_df = self._calculate_volatility_stop_loss(df, position_type, entry_price)
        result_df.loc[result_df['support_stop_price'].isna(), 'support_stop_price'] = vol_stop_df.loc[result_df['support_stop_price'].isna(), 'stop_loss_price']
        
        # 如果有入场价，确保止损价格合理
        if entry_price is not None:
            if position_type.lower() == 'long':
                # 确保止损价格不高于入场价（防止亏损）
                result_df['support_stop_price'] = result_df['support_stop_price'].clip(upper=entry_price)
                
                # 确保止损不会太远（最大止损限制）
                min_allowed_price = entry_price * (1 - self.max_stop_loss / 100)
                result_df['support_stop_price'] = result_df['support_stop_price'].clip(lower=min_allowed_price)
            
            else:  # short
                # 确保止损价格不低于入场价（防止亏损）
                result_df['support_stop_price'] = result_df['support_stop_price'].clip(lower=entry_price)
                
                # 确保止损不会太远（最大止损限制）
                max_allowed_price = entry_price * (1 + self.max_stop_loss / 100)
                result_df['support_stop_price'] = result_df['support_stop_price'].clip(upper=max_allowed_price)
        
        # 计算止损百分比
        if entry_price is not None:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (entry_price - result_df['support_stop_price']) / entry_price * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['support_stop_price'] - entry_price) / entry_price * 100
        else:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (result_df['close'] - result_df['support_stop_price']) / result_df['close'] * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['support_stop_price'] - result_df['close']) / result_df['close'] * 100
        
        # 限制止损百分比在允许范围内
        result_df['stop_loss_pct'] = result_df['stop_loss_pct'].clip(self.min_stop_loss, self.max_stop_loss)
        
        # 最终止损价格 = 支撑位止损价格
        result_df['stop_loss_price'] = result_df['support_stop_price']
        
        # 删除临时列
        result_df = result_df.drop(['support_stop_price'], axis=1)
        
        return result_df

    def _calculate_composite_stop_loss(self, df: pd.DataFrame, position_type: str, entry_price: Optional[float]) -> pd.DataFrame:
        """
        计算复合止损（综合多种止损方式）
        
        Args:
            df: 输入DataFrame
            position_type: 持仓类型，'long'或'short'
            entry_price: 入场价格，如果为None则使用每个K线的收盘价
            
        Returns:
            添加了复合止损价格的DataFrame
        """
        result_df = df.copy()
        
        # 获取各种止损计算结果
        fixed_stop_df = self._calculate_fixed_stop_loss(df, position_type, entry_price)
        trailing_stop_df = self._calculate_trailing_stop_loss(df, position_type, entry_price)
        volatility_stop_df = self._calculate_volatility_stop_loss(df, position_type, entry_price)
        indicator_stop_df = self._calculate_indicator_stop_loss(df, position_type, entry_price)
        support_stop_df = self._calculate_support_stop_loss(df, position_type, entry_price)
        
        # 为各种止损策略分配权重（根据市场环境动态调整）
        weights = {
            'fixed': 0.05,       # 固定止损权重
            'trailing': 0.20,    # 跟踪止损权重
            'volatility': 0.25,  # 波动率止损权重
            'indicator': 0.25,   # 指标止损权重
            'support': 0.25      # 支撑位止损权重
        }
        
        # 基于趋势调整权重
        if 'trend_score' in result_df.columns:
            # 强趋势市场下，增加跟踪止损和指标止损权重
            trend_adjustment = result_df['trend_score'] / 100  # 0-1范围
            
            for i in range(len(result_df)):
                adj = trend_adjustment.iloc[i]
                
                # 趋势越强，越重视跟踪止损和指标止损
                local_weights = weights.copy()
                local_weights['trailing'] += adj * 0.10
                local_weights['indicator'] += adj * 0.10
                
                # 减少固定止损和支撑位止损的权重
                local_weights['fixed'] -= adj * 0.05
                local_weights['support'] -= adj * 0.10
                local_weights['volatility'] -= adj * 0.05
                
                # 确保权重非负且总和为1
                for k in local_weights:
                    local_weights[k] = max(0.01, local_weights[k])
                weight_sum = sum(local_weights.values())
                for k in local_weights:
                    local_weights[k] /= weight_sum
                
                # 加权平均各种止损价格
                if position_type.lower() == 'long':
                    # 多头取加权平均
                    stop_price = (
                        local_weights['fixed'] * fixed_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['trailing'] * trailing_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['volatility'] * volatility_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['indicator'] * indicator_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['support'] * support_stop_df['stop_loss_price'].iloc[i]
                    )
                    result_df.at[result_df.index[i], 'stop_loss_price'] = stop_price
                else:  # short
                    # 空头取加权平均
                    stop_price = (
                        local_weights['fixed'] * fixed_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['trailing'] * trailing_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['volatility'] * volatility_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['indicator'] * indicator_stop_df['stop_loss_price'].iloc[i] +
                        local_weights['support'] * support_stop_df['stop_loss_price'].iloc[i]
                    )
                    result_df.at[result_df.index[i], 'stop_loss_price'] = stop_price
        else:
            # 无趋势数据，使用固定权重
            if position_type.lower() == 'long':
                # 多头使用各止损价格的加权平均
                result_df['stop_loss_price'] = (
                    weights['fixed'] * fixed_stop_df['stop_loss_price'] +
                    weights['trailing'] * trailing_stop_df['stop_loss_price'] +
                    weights['volatility'] * volatility_stop_df['stop_loss_price'] +
                    weights['indicator'] * indicator_stop_df['stop_loss_price'] +
                    weights['support'] * support_stop_df['stop_loss_price']
                )
            else:  # short
                # 空头使用各止损价格的加权平均
                result_df['stop_loss_price'] = (
                    weights['fixed'] * fixed_stop_df['stop_loss_price'] +
                    weights['trailing'] * trailing_stop_df['stop_loss_price'] +
                    weights['volatility'] * volatility_stop_df['stop_loss_price'] +
                    weights['indicator'] * indicator_stop_df['stop_loss_price'] +
                    weights['support'] * support_stop_df['stop_loss_price']
                )
        
        # 计算最终止损百分比
        if entry_price is not None:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (entry_price - result_df['stop_loss_price']) / entry_price * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['stop_loss_price'] - entry_price) / entry_price * 100
        else:
            if position_type.lower() == 'long':
                result_df['stop_loss_pct'] = (result_df['close'] - result_df['stop_loss_price']) / result_df['close'] * 100
            else:  # short
                result_df['stop_loss_pct'] = (result_df['stop_loss_price'] - result_df['close']) / result_df['close'] * 100
        
        # 限制止损百分比在允许范围内
        result_df['stop_loss_pct'] = result_df['stop_loss_pct'].clip(self.min_stop_loss, self.max_stop_loss)
        
        return result_df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算真实波动幅度（ATR）
        
        Args:
            df: 输入DataFrame
            period: ATR计算周期
            
        Returns:
            添加了ATR的DataFrame
        """
        result_df = df.copy()
        
        # 计算真实范围 (True Range)
        result_df['tr1'] = abs(result_df['high'] - result_df['low'])
        result_df['tr2'] = abs(result_df['high'] - result_df['close'].shift())
        result_df['tr3'] = abs(result_df['low'] - result_df['close'].shift())
        
        result_df['tr'] = result_df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR
        result_df['atr'] = result_df['tr'].rolling(window=period).mean()
        
        # 计算相对ATR (ATR/收盘价 的百分比)
        result_df['atr_pct'] = result_df['atr'] / result_df['close'] * 100
        
        # 计算ATR相对于N日平均ATR的比率
        for window in [10, 20, 50]:
            if len(result_df) >= window:
                result_df[f'atr_ratio_{window}'] = result_df['atr'] / result_df['atr'].rolling(window=window).mean()
        
        # 使用20日窗口的比率作为主要ATR比率
        if 'atr_ratio_20' in result_df.columns:
            result_df['atr_ratio'] = result_df['atr_ratio_20']
        elif 'atr_ratio_10' in result_df.columns:
            result_df['atr_ratio'] = result_df['atr_ratio_10']
        else:
            result_df['atr_ratio'] = 1.0  # 默认值
        
        # 删除临时列
        result_df = result_df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return result_df
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势强度指标
        
        Args:
            df: 输入DataFrame，需包含收盘价数据
            
        Returns:
            添加了趋势强度指标的DataFrame
        """
        result_df = df.copy()
        
        # 计算不同周期的移动平均线
        ma_periods = [5, 10, 20, 50, 100]
        for period in ma_periods:
            result_df[f'ma_{period}'] = result_df['close'].rolling(window=period).mean()
        
        # 计算短期、中期、长期趋势方向
        # 1表示上涨趋势，-1表示下跌趋势，0表示盘整
        result_df['short_trend'] = np.where(result_df['ma_5'] > result_df['ma_10'], 1, 
                                     np.where(result_df['ma_5'] < result_df['ma_10'], -1, 0))
        
        result_df['medium_trend'] = np.where(result_df['ma_10'] > result_df['ma_20'], 1,
                                      np.where(result_df['ma_10'] < result_df['ma_20'], -1, 0))
        
        result_df['long_trend'] = np.where(result_df['ma_20'] > result_df['ma_50'], 1,
                                    np.where(result_df['ma_20'] < result_df['ma_50'], -1, 0))
        
        # 计算趋势强度 (1表示最强上涨趋势，-1表示最强下跌趋势)
        result_df['trend_strength'] = (result_df['short_trend'] * 0.5 + 
                                       result_df['medium_trend'] * 0.3 + 
                                       result_df['long_trend'] * 0.2)
        
        # 趋势一致性：判断短中长期趋势是否一致
        result_df['trend_consistency'] = np.where(
            (result_df['short_trend'] == result_df['medium_trend']) & 
            (result_df['medium_trend'] == result_df['long_trend']),
            1, 0
        )
        
        # 计算ADX指标以评估趋势强度
        result_df = self._calculate_adx(result_df)
        
        # 综合趋势评分 (0-100)
        result_df['trend_score'] = 50.0  # 初始为中性
        
        # 如果是一致的上涨趋势，增加分数
        result_df.loc[result_df['trend_consistency'] == 1, 'trend_score'] += 20
        
        # 基于趋势方向调整分数
        result_df['trend_score'] += result_df['trend_strength'] * 30
        
        # 基于ADX调整分数
        result_df.loc[result_df['adx'] > 25, 'trend_score'] += 10
        result_df.loc[result_df['adx'] > 40, 'trend_score'] += 10
        
        # 确保分数在0-100范围内
        result_df['trend_score'] = result_df['trend_score'].clip(0, 100)
        
        return result_df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算平均趋向指数(ADX)
        
        Args:
            df: 输入DataFrame
            period: ADX计算周期
            
        Returns:
            添加了ADX指标的DataFrame
        """
        result_df = df.copy()
        
        # 计算+DM和-DM
        result_df['up_move'] = result_df['high'].diff()
        result_df['down_move'] = result_df['low'].diff(-1).abs()
        
        # 计算+DM
        result_df['plus_dm'] = np.where(
            (result_df['up_move'] > result_df['down_move']) & (result_df['up_move'] > 0),
            result_df['up_move'],
            0
        )
        
        # 计算-DM
        result_df['minus_dm'] = np.where(
            (result_df['down_move'] > result_df['up_move']) & (result_df['down_move'] > 0),
            result_df['down_move'],
            0
        )
        
        # 使用True Range计算+DI和-DI
        if 'atr' not in result_df.columns:
            result_df = self._calculate_atr(result_df)
        
        # 平滑+DM和-DM
        result_df['smooth_plus_dm'] = result_df['plus_dm'].rolling(window=period).sum()
        result_df['smooth_minus_dm'] = result_df['minus_dm'].rolling(window=period).sum()
        
        # 计算+DI和-DI
        result_df['plus_di'] = 100 * result_df['smooth_plus_dm'] / (result_df['atr'] * period)
        result_df['minus_di'] = 100 * result_df['smooth_minus_dm'] / (result_df['atr'] * period)
        
        # 计算DI差和DI和
        result_df['di_diff'] = (result_df['plus_di'] - result_df['minus_di']).abs()
        result_df['di_sum'] = result_df['plus_di'] + result_df['minus_di']
        
        # 计算DX
        result_df['dx'] = 100 * result_df['di_diff'] / result_df['di_sum'].replace(0, np.nan)
        
        # 计算ADX
        result_df['adx'] = result_df['dx'].rolling(window=period).mean()
        
        # 删除临时列
        result_df = result_df.drop(['up_move', 'down_move', 'plus_dm', 'minus_dm', 
                                   'smooth_plus_dm', 'smooth_minus_dm', 'di_diff', 'di_sum', 'dx'], axis=1)
        
        return result_df
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量指标
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了动量指标的DataFrame
        """
        result_df = df.copy()
        
        # 计算RSI
        rsi_periods = [6, 14, 21]
        for period in rsi_periods:
            delta = result_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss.replace(0, np.nan)
            result_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = result_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = result_df['close'].ewm(span=26, adjust=False).mean()
        result_df['macd'] = exp1 - exp2
        result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
        result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
        
        # 计算近期涨跌幅
        for period in [5, 10, 20]:
            result_df[f'return_{period}'] = result_df['close'].pct_change(periods=period) * 100
        
        # 计算综合动量评分 (0-100)
        result_df['momentum_score'] = 50.0  # 初始为中性
        
        # RSI分量
        # RSI > 70表示超买，< 30表示超卖
        result_df.loc[result_df['rsi_14'] > 70, 'momentum_score'] += 20
        result_df.loc[result_df['rsi_14'] < 30, 'momentum_score'] -= 20
        
        # MACD分量
        # MACD柱状图为正表示上升动量，为负表示下降动量
        result_df.loc[result_df['macd_hist'] > 0, 'momentum_score'] += 10
        result_df.loc[result_df['macd_hist'] < 0, 'momentum_score'] -= 10
        
        # 价格动量分量
        result_df.loc[result_df['return_5'] > 0, 'momentum_score'] += 5
        result_df.loc[result_df['return_5'] < 0, 'momentum_score'] -= 5
        
        result_df.loc[result_df['return_10'] > 0, 'momentum_score'] += 5
        result_df.loc[result_df['return_10'] < 0, 'momentum_score'] -= 5
        
        # 确保分数在0-100范围内
        result_df['momentum_score'] = result_df['momentum_score'].clip(0, 100)
        
        # 动量强度分类
        result_df['momentum_strength'] = pd.cut(
            result_df['momentum_score'],
            bins=[0, 30, 45, 55, 70, 100],
            labels=['Strong Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong Bullish']
        )
        
        return result_df

    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了成交量特征的DataFrame
        """
        result_df = df.copy()
        
        # 计算相对成交量 (相对于N日平均)
        for period in [5, 10, 20]:
            result_df[f'rel_volume_{period}'] = result_df['volume'] / result_df['volume'].rolling(window=period).mean()
        
        # 计算成交量变化率
        result_df['volume_change'] = result_df['volume'].pct_change() * 100
        
        # 计算放量程度 (当前成交量 / 前N日最大成交量)
        for period in [5, 10, 20]:
            result_df[f'volume_surge_{period}'] = result_df['volume'] / result_df['volume'].rolling(window=period).max()
        
        # 计算量价背离 (价格上涨但成交量下降，或价格下跌但成交量上升)
        result_df['price_up'] = result_df['close'] > result_df['close'].shift(1)
        result_df['volume_up'] = result_df['volume'] > result_df['volume'].shift(1)
        
        # 创建量价背离指标 (1=背离，0=一致)
        result_df['vol_price_divergence'] = np.where(
            (result_df['price_up'] & ~result_df['volume_up']) | (~result_df['price_up'] & result_df['volume_up']),
            1, 0
        )
        
        # 连续背离计数
        result_df['divergence_count'] = 0
        
        for i in range(1, len(result_df)):
            if result_df['vol_price_divergence'].iloc[i] == 1:
                result_df['divergence_count'].iloc[i] = result_df['divergence_count'].iloc[i-1] + 1
        
        # 成交量趋势分数 (0-100)
        result_df['volume_score'] = 50.0  # 初始为中性
        
        # 基于相对成交量调整分数
        result_df.loc[result_df['rel_volume_5'] > 2.0, 'volume_score'] += 15  # 剧烈放量
        result_df.loc[result_df['rel_volume_5'] > 1.5, 'volume_score'] += 10  # 明显放量
        result_df.loc[result_df['rel_volume_5'] < 0.5, 'volume_score'] -= 10  # 明显缩量
        
        # 基于量价背离调整分数
        result_df.loc[result_df['vol_price_divergence'] == 1, 'volume_score'] -= 5  # 单日背离
        result_df.loc[result_df['divergence_count'] >= 3, 'volume_score'] -= 10  # 连续背离
        
        # 上涨放量加分
        result_df.loc[(result_df['price_up']) & (result_df['volume_up']), 'volume_score'] += 5
        
        # 确保分数在0-100范围内
        result_df['volume_score'] = result_df['volume_score'].clip(0, 100)
        
        # 清理临时列
        result_df = result_df.drop(['price_up', 'volume_up'], axis=1)
        
        return result_df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算关键支撑阻力位
        
        Args:
            df: 输入DataFrame，需包含OHLC数据
            
        Returns:
            添加了支撑阻力位的DataFrame
        """
        result_df = df.copy()
        
        # 计算过去N个周期的极值点
        lookback_periods = [10, 20, 50]
        
        for period in lookback_periods:
            # 确保数据长度足够
            if len(result_df) < period:
                continue
                
            # 找出过去N个周期的低点作为支撑位
            result_df[f'support_{period}'] = result_df['low'].rolling(window=period, min_periods=1).min()
            
            # 找出过去N个周期的高点作为阻力位
            result_df[f'resistance_{period}'] = result_df['high'].rolling(window=period, min_periods=1).max()
        
        # 计算支撑位强度
        result_df['support_strength'] = 0.0
        
        # 如果收盘价接近支撑位(在5%范围内)，则增加支撑强度
        for period in lookback_periods:
            support_col = f'support_{period}'
            if support_col in result_df.columns:
                # 距离最近支撑位的百分比
                distance_to_support = (result_df['close'] - result_df[support_col]) / result_df[support_col] * 100
                
                # 距离支撑位越近，强度越高
                close_to_support = (distance_to_support > -5) & (distance_to_support <= 0)
                very_close_to_support = (distance_to_support > -2) & (distance_to_support <= 0)
                
                # 更新支撑强度
                result_df.loc[close_to_support, 'support_strength'] += 1
                result_df.loc[very_close_to_support, 'support_strength'] += 2
        
        # 归一化支撑强度
        max_strength = len(lookback_periods) * 3  # 最大可能强度
        result_df['support_strength'] = result_df['support_strength'] / max_strength
        
        # 增加高级支撑位识别 - 识别主要的支撑位
        result_df = self._identify_key_levels(result_df)
        
        return result_df
        
    def _identify_key_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        识别关键价格水平
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了关键价格水平的DataFrame
        """
        result_df = df.copy()
        window_size = min(100, len(df))
        
        if window_size < 20:
            return result_df
        
        # 初始化关键水平列
        result_df['key_support'] = np.nan
        result_df['key_resistance'] = np.nan
        
        # 循环计算每个时间点的关键水平
        for i in range(window_size, len(result_df)):
            # 取过去window_size个周期的数据计算关键水平
            window_data = result_df.iloc[i-window_size:i]
            
            # 创建价格区间，计算各区间的成交量
            price_range = np.linspace(window_data['low'].min(), window_data['high'].max(), 100)
            volume_profile = np.zeros(len(price_range) - 1)
            
            # 使用成交量加权计算支撑阻力
            for j in range(len(window_data)):
                low, high, volume = window_data.iloc[j][['low', 'high', 'volume']]
                volume_per_price = volume / (high - low) if high > low else 0
                
                # 为该K线范围内的所有价格区间添加成交量
                for k in range(len(price_range) - 1):
                    if (price_range[k] >= low and price_range[k+1] <= high) or \
                       (price_range[k] <= low and price_range[k+1] >= low) or \
                       (price_range[k] <= high and price_range[k+1] >= high):
                        overlap = min(price_range[k+1], high) - max(price_range[k], low)
                        if overlap > 0:
                            volume_profile[k] += volume_per_price * overlap
            
            # 找出成交量峰值对应的价格区间作为关键水平
            peaks = []
            for k in range(1, len(volume_profile) - 1):
                if volume_profile[k] > volume_profile[k-1] and volume_profile[k] > volume_profile[k+1]:
                    peaks.append((k, volume_profile[k]))
            
            # 按成交量排序
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # 取前3个峰值
            top_peaks = peaks[:3] if len(peaks) >= 3 else peaks
            
            if top_peaks:
                current_price = result_df.iloc[i]['close']
                
                # 找出当前价格下方最近的关键支撑位
                support_peaks = [(k, v) for k, v in top_peaks if price_range[k] < current_price]
                if support_peaks:
                    # 取最近的支撑位
                    support_peak = max(support_peaks, key=lambda x: price_range[x[0]])
                    result_df.at[result_df.index[i], 'key_support'] = price_range[support_peak[0]]
                
                # 找出当前价格上方最近的关键阻力位
                resistance_peaks = [(k, v) for k, v in top_peaks if price_range[k] > current_price]
                if resistance_peaks:
                    # 取最近的阻力位
                    resistance_peak = min(resistance_peaks, key=lambda x: price_range[x[0]])
                    result_df.at[result_df.index[i], 'key_resistance'] = price_range[resistance_peak[0]]
        
        return result_df 