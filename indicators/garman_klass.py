#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Garman-Klass波动率指标

Garman-Klass波动率是一种更精确的波动率估计方法，它利用开盘价、最高价、最低价和收盘价
来计算波动率，比仅使用收盘价的方法更准确。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class GarmanKlass(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    Garman-Klass波动率指标
    
    分类：波动性指标
    描述：利用OHLC数据计算更精确的波动率
    
    计算公式：
    GK = ln(H/L) * ln(H/L) - (2*ln(2)-1) * ln(C/O) * ln(C/O)
    其中：H=最高价, L=最低价, C=收盘价, O=开盘价
    
    信号解释：
    - 数值越大：波动率越高
    - 数值越小：波动率越低
    - 可用于风险管理和仓位调整
    """
    
    def __init__(self, period: int = 20, annualize: bool = True, **kwargs):
        """
        初始化Garman-Klass波动率指标
        
        Args:
            period: 计算周期，默认20
            annualize: 是否年化，默认True
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.period = period
        self.annualize = annualize
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'period': 20,
            'annualize': True
        }
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        self.period = kwargs.get('period', self.period)
        self.annualize = kwargs.get('annualize', self.annualize)
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据的有效性"""
        if data is None or len(data) == 0:
            return False

        # 检查必需列
        for col in self.REQUIRED_COLUMNS:
            if col not in data.columns:
                logger.error(f"数据缺少必需列: {col}")
                return False

        return True

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Garman-Klass波动率

        Args:
            data: 包含OHLC列的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含Garman-Klass波动率的DataFrame
        """
        try:
            # 验证数据
            if not self._validate_data(data):
                return pd.DataFrame()

            df = data.copy()
            
            # 避免除零和对数计算错误
            df = df[(df['high'] > 0) & (df['low'] > 0) & 
                   (df['open'] > 0) & (df['close'] > 0)].copy()
            
            if df.empty:
                logger.warning("数据中包含非正数，无法计算Garman-Klass波动率")
                return pd.DataFrame()
            
            # 计算对数比率
            ln_hl = np.log(df['high'] / df['low'])
            ln_co = np.log(df['close'] / df['open'])
            
            # 计算Garman-Klass估计量
            gk_estimator = ln_hl * ln_hl - (2 * np.log(2) - 1) * ln_co * ln_co
            
            # 计算滚动平均波动率
            gk_volatility = gk_estimator.rolling(window=self.period).mean()
            
            # 年化处理（假设252个交易日）
            if self.annualize:
                gk_volatility_annualized = np.sqrt(gk_volatility * 252)
            else:
                gk_volatility_annualized = np.sqrt(gk_volatility)
            
            # 添加到结果DataFrame
            df['gk_estimator'] = gk_estimator
            df['gk_volatility'] = gk_volatility
            df['gk_volatility_annualized'] = gk_volatility_annualized
            
            # 计算波动率等级
            df['volatility_percentile'] = self._calculate_percentile(gk_volatility_annualized)
            df['volatility_regime'] = self._classify_volatility_regime(df['volatility_percentile'])
            
            # 计算信号
            df['gk_signal'] = self._generate_signals(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Garman-Klass波动率计算失败: {e}")
            return pd.DataFrame()
    
    def _calculate_percentile(self, volatility: pd.Series) -> pd.Series:
        """
        计算波动率的历史分位数
        
        Args:
            volatility: 波动率序列
            
        Returns:
            pd.Series: 分位数序列
        """
        rolling_window = min(252, len(volatility))  # 最多使用一年的数据
        if rolling_window < 20:
            return pd.Series(50, index=volatility.index)  # 默认中位数
        
        percentiles = volatility.rolling(rolling_window).apply(
            lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) > 0 else 50
        )
        
        return percentiles.fillna(50)
    
    def _classify_volatility_regime(self, percentiles: pd.Series) -> pd.Series:
        """
        分类波动率状态
        
        Args:
            percentiles: 分位数序列
            
        Returns:
            pd.Series: 波动率状态
        """
        regimes = pd.Series('中等', index=percentiles.index)
        regimes[percentiles >= 80] = '极高'
        regimes[(percentiles >= 60) & (percentiles < 80)] = '高'
        regimes[(percentiles >= 40) & (percentiles < 60)] = '中等'
        regimes[(percentiles >= 20) & (percentiles < 40)] = '低'
        regimes[percentiles < 20] = '极低'
        
        return regimes
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含波动率数据的DataFrame
            
        Returns:
            pd.Series: 交易信号
        """
        signals = pd.Series(0, index=df.index)
        
        if 'volatility_percentile' not in df.columns:
            return signals
        
        percentiles = df['volatility_percentile']
        
        # 波动率突破信号
        # 从低波动率突破到高波动率（趋势可能开始）
        low_to_high = (percentiles > 70) & (percentiles.shift(1) < 30)
        signals[low_to_high] = 1
        
        # 从高波动率回落到低波动率（趋势可能结束）
        high_to_low = (percentiles < 30) & (percentiles.shift(1) > 70)
        signals[high_to_low] = -1
        
        return signals
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取最新的交易信号
        
        Args:
            data: 计算后的数据
            
        Returns:
            Dict[str, Any]: 信号信息
        """
        if data.empty or 'gk_signal' not in data.columns:
            return {'signal': 0, 'strength': 0, 'description': '无信号'}
        
        latest_signal = data['gk_signal'].iloc[-1]
        latest_volatility = data['gk_volatility_annualized'].iloc[-1] if 'gk_volatility_annualized' in data.columns else 0
        latest_regime = data['volatility_regime'].iloc[-1] if 'volatility_regime' in data.columns else '未知'
        latest_percentile = data['volatility_percentile'].iloc[-1] if 'volatility_percentile' in data.columns else 50
        
        if latest_signal == 1:
            return {
                'signal': 1,
                'strength': min(latest_percentile / 100, 1.0),
                'description': f'波动率突破信号，当前状态：{latest_regime}（{latest_percentile:.1f}%分位）'
            }
        elif latest_signal == -1:
            return {
                'signal': -1,
                'strength': min((100 - latest_percentile) / 100, 1.0),
                'description': f'波动率回落信号，当前状态：{latest_regime}（{latest_percentile:.1f}%分位）'
            }
        else:
            return {
                'signal': 0,
                'strength': 0,
                'description': f'波动率正常，当前状态：{latest_regime}（{latest_percentile:.1f}%分位）'
            }
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """获取指标模式信息"""
        return {
            'name': 'GARMAN_KLASS',
            'description': 'Garman-Klass波动率指标',
            'type': 'volatility',
            'parameters': {
                'period': self.period,
                'annualize': self.annualize
            }
        }

    # 实现BaseIndicator的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """核心计算逻辑"""
        return self.calculate(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if 'volatility_percentile' not in data.columns:
            return pd.Series(50.0, index=data.index)

        # 直接使用波动率分位数作为评分
        return data['volatility_percentile']

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取技术形态"""
        patterns = pd.DataFrame(index=data.index)

        if 'volatility_percentile' in data.columns:
            percentiles = data['volatility_percentile']
            # 波动率状态形态
            patterns['GK_极高波动'] = percentiles >= 80
            patterns['GK_高波动'] = (percentiles >= 60) & (percentiles < 80)
            patterns['GK_低波动'] = (percentiles >= 20) & (percentiles < 40)
            patterns['GK_极低波动'] = percentiles < 20
            # 波动率转换形态
            patterns['GK_波动率突破'] = (percentiles > 70) & (percentiles.shift(1) < 30)
            patterns['GK_波动率回落'] = (percentiles < 30) & (percentiles.shift(1) > 70)

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.0

        # 基于波动率分位数的稳定性计算置信度
        score_range = score.rolling(10).max().iloc[-1] - score.rolling(10).min().iloc[-1] if len(score) >= 10 else 50
        stability = 1.0 - (score_range / 100.0)
        pattern_strength = min(len(patterns) * 0.15, 1.0)

        return (stability + pattern_strength) / 2

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return max(self.period + 20, 50)  # 需要足够的历史数据计算分位数
