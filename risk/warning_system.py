#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风险预警系统

细化风险预警级别，分为多级预警，提供提前预警能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import logging

from utils.logger import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    """风险级别枚举"""
    SAFE = 0         # 安全
    ATTENTION = 1    # 需要关注
    CAUTION = 2      # 警惕
    WARNING = 3      # 警告
    DANGER = 4       # 危险
    EXTREME = 5      # 极端风险


class RiskWarningSystem:
    """
    风险预警系统
    
    基于技术指标和市场环境的多级风险预警系统，提供提前预警能力
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化风险预警系统
        
        Args:
            params: 配置参数字典，可包含以下键:
                - warning_lookback: 预警检查回溯期，默认为20
                - sensitivity: 敏感度系数，范围0.5-2.0，默认为1.0
                - indicator_weights: 各指标权重，默认为均等权重
        """
        self._params = params or {}
        self._initialize_params()
        
    def _initialize_params(self):
        """初始化参数，设置默认值"""
        # 基础参数
        self.warning_lookback = self._params.get('warning_lookback', 20)
        self.sensitivity = self._params.get('sensitivity', 1.0)
        
        # 限制敏感度范围
        self.sensitivity = max(0.5, min(2.0, self.sensitivity))
        
        # 指标权重
        default_weights = {
            'price_action': 0.25,    # 价格行为
            'volatility': 0.20,      # 波动率
            'volume': 0.15,          # 成交量
            'momentum': 0.20,        # 动量
            'liquidity': 0.10,       # 流动性
            'divergence': 0.10       # 背离
        }
        self.indicator_weights = self._params.get('indicator_weights', default_weights) 

    def classify_risk_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对股票进行风险分级
        
        Args:
            df: 输入DataFrame，需包含OHLC数据和成交量
            
        Returns:
            添加了风险评估结果的DataFrame
        """
        result_df = df.copy()
        
        # 确保DataFrame中有必要的OHLC和成交量数据
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            if 'close' not in result_df.columns:
                raise ValueError("DataFrame must contain at least 'close' column")
        
        # 计算风险维度指标
        result_df = self._calculate_price_action_risk(result_df)
        result_df = self._calculate_volatility_risk(result_df)
        result_df = self._calculate_volume_risk(result_df)
        result_df = self._calculate_momentum_risk(result_df)
        result_df = self._calculate_liquidity_risk(result_df)
        result_df = self._calculate_divergence_risk(result_df)
        
        # 计算综合风险评分 (0-100)
        result_df['risk_score'] = (
            self.indicator_weights['price_action'] * result_df['price_action_risk'] +
            self.indicator_weights['volatility'] * result_df['volatility_risk'] +
            self.indicator_weights['volume'] * result_df['volume_risk'] +
            self.indicator_weights['momentum'] * result_df['momentum_risk'] +
            self.indicator_weights['liquidity'] * result_df['liquidity_risk'] +
            self.indicator_weights['divergence'] * result_df['divergence_risk']
        )
        
        # 根据风险评分分级
        risk_bins = [0, 20, 40, 60, 75, 90, 100]
        risk_labels = [level.name for level in RiskLevel]
        
        result_df['risk_level'] = pd.cut(
            result_df['risk_score'], 
            bins=risk_bins, 
            labels=risk_labels, 
            include_lowest=True
        )
        
        # 计算当前趋势下的预期风险变化
        result_df = self._calculate_risk_trend(result_df)
        
        # 给出详细风险原因
        result_df = self._generate_risk_details(result_df)
        
        return result_df
    
    def _calculate_risk_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算风险趋势
        
        Args:
            df: 输入DataFrame，包含风险评分
            
        Returns:
            添加了风险趋势评估的DataFrame
        """
        result_df = df.copy()
        
        # 计算风险评分的移动平均和方向
        result_df['risk_score_ma5'] = result_df['risk_score'].rolling(window=5).mean()
        result_df['risk_score_ma10'] = result_df['risk_score'].rolling(window=10).mean()
        
        # 风险分数变化趋势（正值表示风险上升）
        result_df['risk_score_change'] = result_df['risk_score'] - result_df['risk_score'].shift(5)
        result_df['risk_score_change_pct'] = result_df['risk_score_change'] / result_df['risk_score'].shift(5) * 100
        
        # 风险加速度（二阶导数）
        result_df['risk_acceleration'] = result_df['risk_score_change'] - result_df['risk_score_change'].shift(5)
        
        # 风险趋势评估
        result_df['risk_trend'] = 'Stable'  # 默认为稳定
        
        # 风险上升趋势
        rising_mask = (result_df['risk_score_ma5'] > result_df['risk_score_ma10']) & (result_df['risk_score_change'] > 0)
        result_df.loc[rising_mask, 'risk_trend'] = 'Rising'
        
        # 风险快速上升趋势
        fast_rising_mask = rising_mask & (result_df['risk_score_change_pct'] > 15) & (result_df['risk_acceleration'] > 0)
        result_df.loc[fast_rising_mask, 'risk_trend'] = 'Fast Rising'
        
        # 风险下降趋势
        falling_mask = (result_df['risk_score_ma5'] < result_df['risk_score_ma10']) & (result_df['risk_score_change'] < 0)
        result_df.loc[falling_mask, 'risk_trend'] = 'Falling'
        
        # 风险快速下降趋势
        fast_falling_mask = falling_mask & (result_df['risk_score_change_pct'] < -15) & (result_df['risk_acceleration'] < 0)
        result_df.loc[fast_falling_mask, 'risk_trend'] = 'Fast Falling'
        
        # 预警级别调整（考虑风险趋势）
        result_df['adjusted_risk_level'] = result_df['risk_level']
        
        # 如果风险快速上升，风险级别上调一级
        risk_levels = [level.name for level in RiskLevel]
        
        for i in range(len(result_df)):
            current_level = result_df.iloc[i]['risk_level']
            current_trend = result_df.iloc[i]['risk_trend']
            
            if pd.notna(current_level) and current_level in risk_levels:
                current_idx = risk_levels.index(current_level)
                
                if current_trend == 'Fast Rising' and current_idx < len(risk_levels) - 1:
                    # 上调一级
                    result_df.at[result_df.index[i], 'adjusted_risk_level'] = risk_levels[current_idx + 1]
                elif current_trend == 'Fast Falling' and current_idx > 0:
                    # 下调一级
                    result_df.at[result_df.index[i], 'adjusted_risk_level'] = risk_levels[current_idx - 1]
        
        return result_df
    
    def _generate_risk_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成风险详细信息
        
        Args:
            df: 输入DataFrame，包含风险评分和各维度评分
            
        Returns:
            添加了风险详情的DataFrame
        """
        result_df = df.copy()
        
        # 初始化风险详情
        result_df['risk_details'] = ''
        
        # 风险指标阈值（高于这个值视为该维度风险显著）
        risk_threshold = 70
        
        # 为每行数据生成风险详情
        for i in range(len(result_df)):
            details = []
            
            # 检查各个风险维度
            if result_df.iloc[i]['price_action_risk'] >= risk_threshold:
                details.append('价格行为异常')
            
            if result_df.iloc[i]['volatility_risk'] >= risk_threshold:
                details.append('波动率过高')
            
            if result_df.iloc[i]['volume_risk'] >= risk_threshold:
                details.append('成交量异常')
            
            if result_df.iloc[i]['momentum_risk'] >= risk_threshold:
                details.append('动量过强')
            
            if result_df.iloc[i]['liquidity_risk'] >= risk_threshold:
                details.append('流动性不足')
            
            if result_df.iloc[i]['divergence_risk'] >= risk_threshold:
                details.append('存在指标背离')
            
            # 检查风险趋势
            trend = result_df.iloc[i]['risk_trend']
            if trend == 'Rising':
                details.append('风险水平上升')
            elif trend == 'Fast Rising':
                details.append('风险水平快速上升')
            
            # 合并所有风险详情
            if details:
                result_df.at[result_df.index[i], 'risk_details'] = '，'.join(details)
            else:
                result_df.at[result_df.index[i], 'risk_details'] = '无明显风险'
        
        return result_df 

    def _calculate_price_action_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格行为风险
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了价格行为风险评分的DataFrame
        """
        result_df = df.copy()
        
        # 初始化价格行为风险分数
        result_df['price_action_risk'] = 0.0
        
        # 1. 极端价格变动（单日振幅）
        result_df['daily_range_pct'] = (result_df['high'] - result_df['low']) / result_df['close'].shift(1) * 100
        
        # 计算历史振幅分位数
        for i in range(len(result_df)):
            if i >= self.warning_lookback:
                # 计算过去N天的振幅分位数
                historical_range = result_df.iloc[i-self.warning_lookback:i]['daily_range_pct']
                if not historical_range.empty:
                    current_range = result_df.iloc[i]['daily_range_pct']
                    percentile = sum(historical_range <= current_range) / len(historical_range) * 100
                    
                    # 根据分位数增加风险分数
                    if percentile > 95:  # 极端大振幅
                        result_df.at[result_df.index[i], 'price_action_risk'] += 40
                    elif percentile > 90:  # 很大振幅
                        result_df.at[result_df.index[i], 'price_action_risk'] += 30
                    elif percentile > 80:  # 较大振幅
                        result_df.at[result_df.index[i], 'price_action_risk'] += 20
        
        # 2. 跳空缺口
        result_df['gap_up'] = result_df['low'] > result_df['high'].shift(1)
        result_df['gap_down'] = result_df['high'] < result_df['low'].shift(1)
        result_df['gap_size'] = np.where(
            result_df['gap_up'],
            (result_df['low'] - result_df['high'].shift(1)) / result_df['high'].shift(1) * 100,
            np.where(
                result_df['gap_down'],
                (result_df['high'] - result_df['low'].shift(1)) / result_df['low'].shift(1) * 100,
                0
            )
        )
        
        # 检测大缺口
        result_df.loc[result_df['gap_up'] & (result_df['gap_size'] > 3), 'price_action_risk'] += 20
        result_df.loc[result_df['gap_down'] & (result_df['gap_size'] < -3), 'price_action_risk'] += 25
        
        # 3. 连续单向走势
        # 计算连续上涨或下跌的天数
        result_df['price_up'] = result_df['close'] > result_df['close'].shift(1)
        result_df['consecutive_up'] = 0
        result_df['consecutive_down'] = 0
        
        for i in range(1, len(result_df)):
            if result_df.iloc[i]['price_up']:
                result_df.at[result_df.index[i], 'consecutive_up'] = result_df.iloc[i-1]['consecutive_up'] + 1
                result_df.at[result_df.index[i], 'consecutive_down'] = 0
            else:
                result_df.at[result_df.index[i], 'consecutive_up'] = 0
                result_df.at[result_df.index[i], 'consecutive_down'] = result_df.iloc[i-1]['consecutive_down'] + 1
        
        # 评估连续走势风险
        result_df.loc[result_df['consecutive_up'] >= 7, 'price_action_risk'] += 15
        result_df.loc[result_df['consecutive_up'] >= 10, 'price_action_risk'] += 15  # 累计30
        result_df.loc[result_df['consecutive_down'] >= 7, 'price_action_risk'] += 20
        result_df.loc[result_df['consecutive_down'] >= 10, 'price_action_risk'] += 20  # 累计40
        
        # 4. 价格远离均线
        # 计算与20日均线的偏离度
        if 'ma_20' not in result_df.columns:
            result_df['ma_20'] = result_df['close'].rolling(window=20).mean()
        
        result_df['ma_deviation'] = (result_df['close'] - result_df['ma_20']) / result_df['ma_20'] * 100
        
        # 评估均线偏离风险
        result_df.loc[result_df['ma_deviation'].abs() > 15, 'price_action_risk'] += 15
        result_df.loc[result_df['ma_deviation'].abs() > 25, 'price_action_risk'] += 20  # 累计35
        
        # 5. 长上影线或下影线（看涨后继续下跌或看跌后继续上涨的风险）
        result_df['upper_shadow'] = (result_df['high'] - result_df[['open', 'close']].max(axis=1)) / result_df['close'] * 100
        result_df['lower_shadow'] = (result_df[['open', 'close']].min(axis=1) - result_df['low']) / result_df['close'] * 100
        
        # 长上/下影线风险
        result_df.loc[result_df['upper_shadow'] > 3, 'price_action_risk'] += 10
        result_df.loc[result_df['lower_shadow'] > 3, 'price_action_risk'] += 10
        
        # 确保风险分数在0-100范围内
        result_df['price_action_risk'] = result_df['price_action_risk'].clip(0, 100)
        
        # 应用敏感度系数
        result_df['price_action_risk'] = result_df['price_action_risk'] * self.sensitivity
        
        # 再次确保风险分数在0-100范围内
        result_df['price_action_risk'] = result_df['price_action_risk'].clip(0, 100)
        
        return result_df
    
    def _calculate_volatility_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率风险
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了波动率风险评分的DataFrame
        """
        result_df = df.copy()
        
        # 初始化波动率风险分数
        result_df['volatility_risk'] = 0.0
        
        # 1. 计算历史波动率
        # 计算日收益率
        result_df['daily_return'] = result_df['close'].pct_change() * 100
        
        # 计算N日波动率（收益率标准差）
        volatility_windows = [10, 20, 60]
        for window in volatility_windows:
            result_df[f'volatility_{window}d'] = result_df['daily_return'].rolling(window=window).std()
        
        # 2. 波动率突变
        # 计算短期波动率相对于长期波动率的比率
        result_df['vol_ratio_10_60'] = result_df['volatility_10d'] / result_df['volatility_60d']
        
        # 评估波动率风险
        result_df.loc[result_df['vol_ratio_10_60'] > 1.5, 'volatility_risk'] += 20
        result_df.loc[result_df['vol_ratio_10_60'] > 2.0, 'volatility_risk'] += 20  # 累计40
        
        # 3. 历史波动率分位数
        # 计算波动率的历史分位数
        for i in range(len(result_df)):
            if i >= self.warning_lookback:
                # 计算过去N天的波动率分位数
                historical_vol = result_df.iloc[i-self.warning_lookback:i]['volatility_20d']
                if not historical_vol.empty and not historical_vol.isna().all():
                    current_vol = result_df.iloc[i]['volatility_20d']
                    if not pd.isna(current_vol):
                        percentile = sum(historical_vol <= current_vol) / len(historical_vol) * 100
                        
                        # 根据分位数增加风险分数
                        if percentile > 95:  # 极高波动率
                            result_df.at[result_df.index[i], 'volatility_risk'] += 40
                        elif percentile > 90:  # 高波动率
                            result_df.at[result_df.index[i], 'volatility_risk'] += 30
                        elif percentile > 80:  # 较高波动率
                            result_df.at[result_df.index[i], 'volatility_risk'] += 20
        
        # 4. 波动率加速度（二阶导数）
        result_df['volatility_change'] = result_df['volatility_20d'].diff()
        result_df['volatility_acceleration'] = result_df['volatility_change'].diff()
        
        # 评估波动率加速度
        result_df.loc[result_df['volatility_acceleration'] > 0, 'volatility_risk'] += 10
        result_df.loc[result_df['volatility_acceleration'] > result_df['volatility_20d'] * 0.1, 'volatility_risk'] += 20  # 累计30
        
        # 5. 极端日内波动（真实波动幅度/ATR）
        # 计算真实波动幅度
        result_df['tr1'] = abs(result_df['high'] - result_df['low'])
        result_df['tr2'] = abs(result_df['high'] - result_df['close'].shift())
        result_df['tr3'] = abs(result_df['low'] - result_df['close'].shift())
        
        result_df['tr'] = result_df[['tr1', 'tr2', 'tr3']].max(axis=1)
        result_df['atr_20'] = result_df['tr'].rolling(window=20).mean()
        
        # 真实波动幅度相对于ATR的比率
        result_df['tr_atr_ratio'] = result_df['tr'] / result_df['atr_20']
        
        # 评估极端波动风险
        result_df.loc[result_df['tr_atr_ratio'] > 2.0, 'volatility_risk'] += 20
        result_df.loc[result_df['tr_atr_ratio'] > 3.0, 'volatility_risk'] += 20  # 累计40
        
        # 确保风险分数在0-100范围内
        result_df['volatility_risk'] = result_df['volatility_risk'].clip(0, 100)
        
        # 应用敏感度系数
        result_df['volatility_risk'] = result_df['volatility_risk'] * self.sensitivity
        
        # 再次确保风险分数在0-100范围内
        result_df['volatility_risk'] = result_df['volatility_risk'].clip(0, 100)
        
        # 删除临时列
        result_df = result_df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return result_df 

    def _calculate_volume_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量风险
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了成交量风险评分的DataFrame
        """
        result_df = df.copy()
        
        # 初始化成交量风险分数
        result_df['volume_risk'] = 0.0
        
        # 1. 成交量激增/萎缩
        # 计算相对成交量 (当前成交量相对于N日平均成交量)
        for window in [5, 10, 20]:
            result_df[f'rel_volume_{window}'] = result_df['volume'] / result_df['volume'].rolling(window=window).mean()
        
        # 评估成交量激增风险
        result_df.loc[result_df['rel_volume_5'] > 3.0, 'volume_risk'] += 30  # 成交量激增
        result_df.loc[result_df['rel_volume_5'] > 5.0, 'volume_risk'] += 20  # 成交量极度激增，累计50
        
        # 评估成交量萎缩风险
        result_df.loc[result_df['rel_volume_5'] < 0.3, 'volume_risk'] += 20  # 成交量萎缩
        
        # 2. 量价关系异常
        # 计算价格和成交量的变化方向
        result_df['price_up'] = result_df['close'] > result_df['close'].shift(1)
        result_df['volume_up'] = result_df['volume'] > result_df['volume'].shift(1)
        
        # 创建量价背离指标
        result_df['vol_price_divergence'] = np.where(
            (result_df['price_up'] & ~result_df['volume_up']) | (~result_df['price_up'] & result_df['volume_up']),
            1, 0
        )
        
        # 计算连续背离的天数
        result_df['divergence_count'] = 0
        for i in range(1, len(result_df)):
            if result_df.iloc[i]['vol_price_divergence'] == 1:
                result_df.at[result_df.index[i], 'divergence_count'] = result_df.iloc[i-1]['divergence_count'] + 1
        
        # 评估量价背离风险
        result_df.loc[result_df['divergence_count'] >= 3, 'volume_risk'] += 20  # 连续背离
        result_df.loc[result_df['divergence_count'] >= 5, 'volume_risk'] += 20  # 长期背离，累计40
        
        # 3. 成交量分布异常
        # 计算价格上涨日的成交量比例
        window = 20
        if len(result_df) >= window:
            for i in range(window, len(result_df)):
                window_data = result_df.iloc[i-window:i]
                
                # 上涨日和下跌日的总成交量
                up_volume = window_data.loc[window_data['price_up'], 'volume'].sum()
                down_volume = window_data.loc[~window_data['price_up'], 'volume'].sum()
                
                # 计算上涨日成交量占比
                total_volume = up_volume + down_volume
                if total_volume > 0:
                    up_volume_ratio = up_volume / total_volume
                    
                    # 成交量分布异常风险评估
                    if up_volume_ratio < 0.3:  # 下跌日成交量过大
                        result_df.at[result_df.index[i], 'volume_risk'] += 25
                    elif up_volume_ratio > 0.8:  # 上涨日成交量过大
                        result_df.at[result_df.index[i], 'volume_risk'] += 20
        
        # 4. 放量滞涨/放量下跌
        # 放量滞涨：成交量明显放大但价格涨幅很小
        big_volume_small_price = (result_df['rel_volume_5'] > 2.0) & \
                                 (result_df['price_up']) & \
                                 (result_df['close'] / result_df['close'].shift(1) - 1 < 0.01)
        
        # 放量下跌：成交量明显放大且价格大幅下跌
        big_volume_big_drop = (result_df['rel_volume_5'] > 2.0) & \
                              (~result_df['price_up']) & \
                              (result_df['close'].shift(1) / result_df['close'] - 1 > 0.02)
        
        # 评估放量滞涨/放量下跌风险
        result_df.loc[big_volume_small_price, 'volume_risk'] += 20
        result_df.loc[big_volume_big_drop, 'volume_risk'] += 30
        
        # 确保风险分数在0-100范围内
        result_df['volume_risk'] = result_df['volume_risk'].clip(0, 100)
        
        # 应用敏感度系数
        result_df['volume_risk'] = result_df['volume_risk'] * self.sensitivity
        
        # 再次确保风险分数在0-100范围内
        result_df['volume_risk'] = result_df['volume_risk'].clip(0, 100)
        
        return result_df
    
    def _calculate_momentum_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量风险
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了动量风险评分的DataFrame
        """
        result_df = df.copy()
        
        # 初始化动量风险分数
        result_df['momentum_risk'] = 0.0
        
        # 1. 超买/超卖风险评估
        # 计算RSI指标
        rsi_periods = [6, 14, 21]
        for period in rsi_periods:
            delta = result_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss.replace(0, np.nan)
            result_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 评估RSI超买/超卖风险
        result_df.loc[result_df['rsi_14'] > 85, 'momentum_risk'] += 40  # 极度超买
        result_df.loc[(result_df['rsi_14'] > 75) & (result_df['rsi_14'] <= 85), 'momentum_risk'] += 30  # 显著超买
        result_df.loc[(result_df['rsi_14'] > 70) & (result_df['rsi_14'] <= 75), 'momentum_risk'] += 20  # 超买
        
        result_df.loc[result_df['rsi_14'] < 15, 'momentum_risk'] += 40  # 极度超卖
        result_df.loc[(result_df['rsi_14'] < 25) & (result_df['rsi_14'] >= 15), 'momentum_risk'] += 30  # 显著超卖
        result_df.loc[(result_df['rsi_14'] < 30) & (result_df['rsi_14'] >= 25), 'momentum_risk'] += 20  # 超卖
        
        # 2. MACD背离风险
        # 计算MACD
        exp1 = result_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = result_df['close'].ewm(span=26, adjust=False).mean()
        result_df['macd'] = exp1 - exp2
        result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
        result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
        
        # 检测MACD柱状图顶底背离
        # 需要足够的数据来识别背离
        if len(result_df) >= 20:
            for i in range(20, len(result_df)):
                # 寻找最近的高点
                recent_high_idx = result_df['close'].iloc[i-20:i].idxmax()
                recent_high_price = result_df.loc[recent_high_idx, 'close']
                recent_high_macd = result_df.loc[recent_high_idx, 'macd_hist']
                
                # 寻找前一个高点
                prev_window = result_df.iloc[:i-20]
                if not prev_window.empty:
                    prev_high_idx = prev_window['close'].idxmax()
                    prev_high_price = result_df.loc[prev_high_idx, 'close']
                    prev_high_macd = result_df.loc[prev_high_idx, 'macd_hist']
                    
                    # 检测顶背离（价格创新高但MACD未创新高）
                    if recent_high_price > prev_high_price and recent_high_macd < prev_high_macd:
                        result_df.at[result_df.index[i], 'momentum_risk'] += 35
                
                # 寻找最近的低点
                recent_low_idx = result_df['close'].iloc[i-20:i].idxmin()
                recent_low_price = result_df.loc[recent_low_idx, 'close']
                recent_low_macd = result_df.loc[recent_low_idx, 'macd_hist']
                
                # 寻找前一个低点
                prev_window = result_df.iloc[:i-20]
                if not prev_window.empty:
                    prev_low_idx = prev_window['close'].idxmin()
                    prev_low_price = result_df.loc[prev_low_idx, 'close']
                    prev_low_macd = result_df.loc[prev_low_idx, 'macd_hist']
                    
                    # 检测底背离（价格创新低但MACD未创新低）
                    if recent_low_price < prev_low_price and recent_low_macd > prev_low_macd:
                        result_df.at[result_df.index[i], 'momentum_risk'] += 35
        
        # 3. 短期动量过热/过冷
        # 计算短期收益率
        for period in [3, 5, 10]:
            result_df[f'return_{period}'] = result_df['close'].pct_change(periods=period) * 100
        
        # 评估短期收益率风险
        result_df.loc[result_df['return_5'] > 15, 'momentum_risk'] += 25  # 短期涨幅过大
        result_df.loc[result_df['return_5'] < -15, 'momentum_risk'] += 25  # 短期跌幅过大
        
        # 4. 动量趋势逆转信号
        # 计算动量指标的变化趋势
        result_df['rsi_change'] = result_df['rsi_14'].diff()
        result_df['macd_hist_change'] = result_df['macd_hist'].diff()
        
        # 评估动量逆转风险
        # RSI从高位开始下降
        rsi_reversal_high = (result_df['rsi_14'] > 70) & (result_df['rsi_change'] < 0)
        # RSI从低位开始上升
        rsi_reversal_low = (result_df['rsi_14'] < 30) & (result_df['rsi_change'] > 0)
        # MACD柱状图由正转负
        macd_reversal_down = (result_df['macd_hist'].shift(1) > 0) & (result_df['macd_hist'] < 0)
        # MACD柱状图由负转正
        macd_reversal_up = (result_df['macd_hist'].shift(1) < 0) & (result_df['macd_hist'] > 0)
        
        # 评估动量逆转风险
        result_df.loc[rsi_reversal_high, 'momentum_risk'] += 20
        result_df.loc[rsi_reversal_low, 'momentum_risk'] += 20
        result_df.loc[macd_reversal_down, 'momentum_risk'] += 25
        result_df.loc[macd_reversal_up, 'momentum_risk'] += 25
        
        # 确保风险分数在0-100范围内
        result_df['momentum_risk'] = result_df['momentum_risk'].clip(0, 100)
        
        # 应用敏感度系数
        result_df['momentum_risk'] = result_df['momentum_risk'] * self.sensitivity
        
        # 再次确保风险分数在0-100范围内
        result_df['momentum_risk'] = result_df['momentum_risk'].clip(0, 100)
        
        return result_df
    
    def _calculate_liquidity_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算流动性风险
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了流动性风险评分的DataFrame
        """
        result_df = df.copy()
        
        # 初始化流动性风险分数
        result_df['liquidity_risk'] = 0.0
        
        # 1. 计算换手率
        result_df['turnover_rate'] = result_df['volume'] / result_df['volume'].rolling(window=20).mean()
        
        # 评估换手率风险
        result_df.loc[result_df['turnover_rate'] > 0.5, 'liquidity_risk'] += 20  # 高换手率
        result_df.loc[result_df['turnover_rate'] > 1.0, 'liquidity_risk'] += 20  # 极高换手率，累计40
        
        # 2. 计算买卖价差
        result_df['bid_ask_spread'] = (result_df['high'] - result_df['low']) / result_df['close'] * 100
        
        # 评估买卖价差风险
        result_df.loc[result_df['bid_ask_spread'] > 1.0, 'liquidity_risk'] += 10  # 高买卖价差
        result_df.loc[result_df['bid_ask_spread'] > 2.0, 'liquidity_risk'] += 10  # 极高买卖价差，累计20
        
        # 3. 计算流动性比率
        result_df['liquidity_ratio'] = result_df['volume'] / result_df['volume'].rolling(window=20).mean()
        
        # 评估流动性比率风险
        result_df.loc[result_df['liquidity_ratio'] < 0.5, 'liquidity_risk'] += 20  # 低流动性比率
        result_df.loc[result_df['liquidity_ratio'] < 0.3, 'liquidity_risk'] += 20  # 极低流动性比率，累计40
        
        # 确保风险分数在0-100范围内
        result_df['liquidity_risk'] = result_df['liquidity_risk'].clip(0, 100)
        
        # 应用敏感度系数
        result_df['liquidity_risk'] = result_df['liquidity_risk'] * self.sensitivity
        
        # 再次确保风险分数在0-100范围内
        result_df['liquidity_risk'] = result_df['liquidity_risk'].clip(0, 100)
        
        return result_df
    
    def _calculate_divergence_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算背离风险
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加了背离风险评分的DataFrame
        """
        result_df = df.copy()
        
        # 初始化背离风险分数
        result_df['divergence_risk'] = 0.0
        
        # 1. 计算价格和成交量之间的背离
        result_df['price_up'] = result_df['close'] > result_df['close'].shift(1)
        result_df['volume_up'] = result_df['volume'] > result_df['volume'].shift(1)
        
        # 创建价格和成交量背离指标
        result_df['price_volume_divergence'] = np.where(
            (result_df['price_up'] & ~result_df['volume_up']) | (~result_df['price_up'] & result_df['volume_up']),
            1, 0
        )
        
        # 计算连续背离的天数
        result_df['divergence_count'] = 0
        for i in range(1, len(result_df)):
            if result_df.iloc[i]['price_volume_divergence'] == 1:
                result_df.at[result_df.index[i], 'divergence_count'] = result_df.iloc[i-1]['divergence_count'] + 1
        
        # 评估价格和成交量背离风险
        result_df.loc[result_df['divergence_count'] >= 3, 'divergence_risk'] += 20  # 连续背离
        result_df.loc[result_df['divergence_count'] >= 5, 'divergence_risk'] += 20  # 长期背离，累计40
        
        # 2. 计算价格和均线之间的背离
        if 'ma_20' not in result_df.columns:
            result_df['ma_20'] = result_df['close'].rolling(window=20).mean()
        
        result_df['price_ma_divergence'] = (result_df['close'] - result_df['ma_20']) / result_df['ma_20'] * 100
        
        # 评估价格和均线背离风险
        result_df.loc[result_df['price_ma_divergence'].abs() > 15, 'divergence_risk'] += 15
        result_df.loc[result_df['price_ma_divergence'].abs() > 25, 'divergence_risk'] += 20  # 累计35
        
        # 3. 计算价格和成交量加速度之间的背离
        result_df['price_volume_acceleration'] = result_df['price_ma_divergence'].diff()
        
        # 评估价格和成交量加速度背离风险
        result_df.loc[result_df['price_volume_acceleration'] > 0, 'divergence_risk'] += 10
        result_df.loc[result_df['price_volume_acceleration'] > result_df['price_ma_divergence'] * 0.1, 'divergence_risk'] += 20  # 累计30
        
        # 确保风险分数在0-100范围内
        result_df['divergence_risk'] = result_df['divergence_risk'].clip(0, 100)
        
        # 应用敏感度系数
        result_df['divergence_risk'] = result_df['divergence_risk'] * self.sensitivity
        
        # 再次确保风险分数在0-100范围内
        result_df['divergence_risk'] = result_df['divergence_risk'].clip(0, 100)
        
        return result_df 

    def generate_risk_alerts(self, df: pd.DataFrame, alert_threshold: str = 'CAUTION') -> pd.DataFrame:
        """
        生成风险预警信号
        
        Args:
            df: 包含风险评估结果的DataFrame
            alert_threshold: 触发预警的最小风险级别，默认为'CAUTION'
            
        Returns:
            预警信号DataFrame
        """
        # 确保已经进行风险分级
        if 'adjusted_risk_level' not in df.columns:
            df = self.classify_risk_level(df)
        
        # 初始化预警结果
        alerts = pd.DataFrame(index=df.index)
        alerts['alert_triggered'] = False
        alerts['alert_level'] = None
        alerts['alert_message'] = None
        alerts['suggested_action'] = None
        alerts['alert_score'] = 0
        alerts['early_warning'] = False
        alerts['persistent_warning'] = False
        
        # 确定预警阈值
        threshold_level = RiskLevel[alert_threshold].value
        
        for i, date in enumerate(alerts.index):
            if i < 10:  # 确保有足够的历史数据
                continue
                
            current_level_name = df.loc[date, 'adjusted_risk_level']
            
            if pd.isna(current_level_name):
                continue
                
            current_level = RiskLevel[current_level_name].value
            current_trend = df.loc[date, 'risk_trend']
            current_score = df.loc[date, 'risk_score']
            
            # 判断是否触发预警
            if current_level >= threshold_level:
                alerts.loc[date, 'alert_triggered'] = True
                alerts.loc[date, 'alert_level'] = current_level_name
                alerts.loc[date, 'alert_score'] = current_score
                
                # 根据不同风险级别生成预警消息和建议操作
                if current_level == RiskLevel.EXTREME.value:
                    alerts.loc[date, 'alert_message'] = "极端风险警报！市场可能处于崩溃边缘"
                    alerts.loc[date, 'suggested_action'] = "立即平仓，暂停交易，全面保护资金安全"
                
                elif current_level == RiskLevel.DANGER.value:
                    alerts.loc[date, 'alert_message'] = "危险风险警报！市场出现严重恶化"
                    alerts.loc[date, 'suggested_action'] = "大幅减仓，保持高度防御，避免新的交易"
                
                elif current_level == RiskLevel.WARNING.value:
                    alerts.loc[date, 'alert_message'] = "警告风险警报！市场走势恶化明显"
                    alerts.loc[date, 'suggested_action'] = "减轻持仓，收紧止损，暂停新的多头交易"
                
                elif current_level == RiskLevel.CAUTION.value:
                    alerts.loc[date, 'alert_message'] = "警惕风险提示！市场出现不利趋势"
                    alerts.loc[date, 'suggested_action'] = "谨慎操作，提高选股标准，缩小仓位"
                
                elif current_level == RiskLevel.ATTENTION.value:
                    alerts.loc[date, 'alert_message'] = "需要关注！市场表现不佳"
                    alerts.loc[date, 'suggested_action'] = "保持警惕，准备好应对风险，审视持仓"
            
            # 生成早期预警信号 (趋势变化但尚未达到阈值)
            elif current_trend == 'Fast Rising' and current_level == threshold_level - 1:
                alerts.loc[date, 'early_warning'] = True
                alerts.loc[date, 'alert_message'] = f"早期风险预警！风险评分正在快速上升，接近{RiskLevel(threshold_level).name}级别"
                alerts.loc[date, 'suggested_action'] = "密切关注市场变化，准备调整策略"
                alerts.loc[date, 'alert_score'] = current_score
            
            # 检测持续性预警（连续多日高风险）
            lookback = 5
            if i >= lookback:
                # 检查过去多日的风险级别
                persistent_high_risk = True
                for j in range(1, lookback+1):
                    prev_date = alerts.index[i-j]
                    prev_level_name = df.loc[prev_date, 'adjusted_risk_level']
                    if pd.isna(prev_level_name):
                        persistent_high_risk = False
                        break
                    prev_level = RiskLevel[prev_level_name].value
                    if prev_level < threshold_level:
                        persistent_high_risk = False
                        break
                
                if persistent_high_risk:
                    alerts.loc[date, 'persistent_warning'] = True
                    if not alerts.loc[date, 'alert_triggered']:
                        alerts.loc[date, 'alert_message'] = f"持续性风险警报！市场连续{lookback}日保持高风险状态"
                        alerts.loc[date, 'suggested_action'] = "考虑进一步降低风险敞口，耐心等待市场转向"
                    else:
                        alerts.loc[date, 'alert_message'] = alerts.loc[date, 'alert_message'] + f"（持续{lookback}日）"
                        
        # 添加风险因素分析
        alerts = self._add_risk_factor_analysis(df, alerts)
        
        return alerts
    
    def _add_risk_factor_analysis(self, risk_df: pd.DataFrame, alert_df: pd.DataFrame) -> pd.DataFrame:
        """
        向预警信号添加风险因素分析
        
        Args:
            risk_df: 风险评估DataFrame
            alert_df: 预警信号DataFrame
            
        Returns:
            添加了风险因素分析的预警DataFrame
        """
        result_df = alert_df.copy()
        result_df['top_risk_factors'] = None
        
        # 仅对触发预警的行分析风险因素
        alert_mask = result_df['alert_triggered'] | result_df['early_warning']
        
        for date in result_df.index[alert_mask]:
            # 获取各风险维度得分
            risk_factors = {
                'price_action': risk_df.loc[date, 'price_action_risk'],
                'volatility': risk_df.loc[date, 'volatility_risk'],
                'volume': risk_df.loc[date, 'volume_risk'],
                'momentum': risk_df.loc[date, 'momentum_risk'],
                'liquidity': risk_df.loc[date, 'liquidity_risk'],
                'divergence': risk_df.loc[date, 'divergence_risk']
            }
            
            # 按风险得分排序
            sorted_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
            
            # 提取前3个主要风险因素
            top_factors = sorted_factors[:3]
            top_factors_text = ", ".join([f"{factor[0]}({factor[1]:.1f})" for factor in top_factors])
            
            result_df.loc[date, 'top_risk_factors'] = top_factors_text
            
            # 更新预警消息，添加风险因素
            if result_df.loc[date, 'alert_message'] is not None:
                result_df.loc[date, 'alert_message'] += f" | 主要风险因素: {top_factors_text}"
        
        return result_df 