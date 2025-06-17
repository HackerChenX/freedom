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
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
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
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # result_df['trend_strength'] = result_df['trend_strength'] * np.sign(result_df['trend_direction'])
        
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
        
        # 定义趋势分类条件
        conditions = [
            # 强势上涨
            (result_df['trend_direction'] > 1.0) & (result_df['trend_strength'] > 0.7),
            # 普通上涨
            (result_df['trend_direction'] > 0.3),
            # 弱势上涨
            (result_df['trend_direction'] > 0.1),
            # 强势下跌
            (result_df['trend_direction'] < -1.0) & (result_df['trend_strength'] > 0.7),
            # 普通下跌
            (result_df['trend_direction'] < -0.3),
            # 弱势下跌
            (result_df['trend_direction'] < -0.1)
        ]
        
        choices_text = [
            'strong_uptrend', 'uptrend', 'weak_uptrend',
            'strong_downtrend', 'downtrend', 'weak_downtrend'
        ]
        
        choices_value = [
            TrendType.STRONG_UPTREND.value, TrendType.UPTREND.value, TrendType.WEAK_UPTREND.value,
            TrendType.STRONG_DOWNTREND.value, TrendType.DOWNTREND.value, TrendType.WEAK_DOWNTREND.value
        ]
        
        # 应用分类
        result_df['trend_type'] = np.select(conditions, choices_text, default='sideways')
        result_df['trend_type_value'] = np.select(conditions, choices_value, default=TrendType.CONSOLIDATION.value)
        
        # 将细分类映射为简化的三大类，以满足大多数上层用例
        result_df['trend_type'] = result_df['trend_type'].replace({
            'strong_uptrend': 'uptrend',
            'weak_uptrend': 'uptrend',
            'strong_downtrend': 'downtrend',
            'weak_downtrend': 'downtrend'
        })
        
        return result_df
    
    def get_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含趋势分类结果的DataFrame
            
        Returns:
            包含交易信号的DataFrame
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # 趋势反转信号
        # 从盘整/下跌转为上涨
        signals.loc[
            (df['trend_type_value'].shift(1) <= TrendType.CONSOLIDATION.value) &
            (df['trend_type_value'] >= TrendType.WEAK_UPTREND.value),
            'signal'
        ] = 1  # 买入信号
        
        # 从上涨转为盘整/下跌
        signals.loc[
            (df['trend_type_value'].shift(1) >= TrendType.WEAK_UPTREND.value) &
            (df['trend_type_value'] <= TrendType.CONSOLIDATION.value),
            'signal'
        ] = -1  # 卖出信号
        
        # 趋势延续信号
        # 上涨趋势持续
        signals.loc[
            (df['trend_type_value'] >= TrendType.WEAK_UPTREND.value) &
            (df['trend_type_value'].shift(1) >= TrendType.WEAK_UPTREND.value),
            'signal'
        ] = 2  # 持有/加仓
        
        # 下跌趋势持续
        signals.loc[
            (df['trend_type_value'] <= TrendType.WEAK_DOWNTREND.value) &
            (df['trend_type_value'].shift(1) <= TrendType.WEAK_DOWNTREND.value),
            'signal'
        ] = -2  # 空仓/减仓
        
        return signals

    def get_patterns(self, data: pd.DataFrame = None) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        识别趋势形态。
        
        目前返回一个空的DataFrame以满足接口要求。
        
        Args:
            data: 输入数据，可选。
            
        Returns:
            一个空的DataFrame。
        """
        return pd.DataFrame(columns=['pattern_id', 'start_index', 'end_index'])

    def register_patterns(self):
        """
        注册TrendClassification指标的形态到全局形态注册表
        """
        # 注册强势趋势形态
        self.register_pattern_to_registry(
            pattern_id="STRONG_UPTREND",
            display_name="强势上升趋势",
            description="价格处于强势上升趋势，多头力量强劲",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="STRONG_DOWNTREND",
            display_name="强势下降趋势",
            description="价格处于强势下降趋势，空头力量强劲",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

        # 注册一般趋势形态
        self.register_pattern_to_registry(
            pattern_id="UPTREND",
            display_name="上升趋势",
            description="价格处于上升趋势，多头占优",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="DOWNTREND",
            display_name="下降趋势",
            description="价格处于下降趋势，空头占优",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册弱势趋势形态
        self.register_pattern_to_registry(
            pattern_id="WEAK_UPTREND",
            display_name="弱势上升趋势",
            description="价格处于弱势上升趋势，上涨动能有限",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="WEAK_DOWNTREND",
            display_name="弱势下降趋势",
            description="价格处于弱势下降趋势，下跌动能有限",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        # 注册横盘整理形态
        self.register_pattern_to_registry(
            pattern_id="SIDEWAYS",
            display_name="横盘整理",
            description="价格处于横盘整理状态，多空力量均衡",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册趋势转换形态
        self.register_pattern_to_registry(
            pattern_id="TREND_REVERSAL_BULLISH",
            display_name="趋势转为看涨",
            description="趋势从下降转为上升，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TREND_REVERSAL_BEARISH",
            display_name="趋势转为看跌",
            description="趋势从上升转为下降，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册趋势加速形态
        self.register_pattern_to_registry(
            pattern_id="TREND_ACCELERATION_UP",
            display_name="上升趋势加速",
            description="上升趋势加速，多头力量增强",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=28.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TREND_ACCELERATION_DOWN",
            display_name="下降趋势加速",
            description="下降趋势加速，空头力量增强",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-28.0,
            polarity="NEGATIVE"
        )

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分
        
        评分逻辑：
        - 强势上涨：100
        - 上涨：80
        - 弱势上涨：60
        - 盘整：40
        - 弱势下跌：20
        - 下跌：10
        - 强势下跌：0
        """
        if self.indicator_data is None:
            self.indicator_data = self.calculate(data)
        
        score_map = {
            'strong_uptrend': 100,
            'uptrend': 80,
            'weak_uptrend': 60,
            'sideways': 40,
            'weak_downtrend': 20,
            'downtrend': 10,
            'strong_downtrend': 0
        }
        
        raw_score = self.indicator_data['trend_type'].map(score_map).fillna(40)
        return raw_score
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号字典
        
        Args:
            data: 输入数据
            
        Returns:
            信号字典
        """
        if self.indicator_data is None:
            self.indicator_data = self.calculate(data)
            
        signals_df = self.get_signals(self.indicator_data)
        
        return {
            'buy_signal': signals_df['signal'] == 1,
            'sell_signal': signals_df['signal'] == -1,
            'hold_signal': signals_df['signal'] == 2,
            'short_signal': signals_df['signal'] == -2
        }

    def get_pattern_info(self, pattern_id: str) -> dict:
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
