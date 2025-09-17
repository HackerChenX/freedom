#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
力量指数(Force Index)指标

力量指数是由Alexander Elder开发的技术指标，结合价格变化和成交量来衡量市场的力量。
它通过计算价格变化与成交量的乘积来反映买卖双方的力量对比。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ForceIndex(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    力量指数(Force Index)指标
    
    分类：成交量指标
    描述：结合价格变化和成交量来衡量市场的力量
    
    计算公式：
    Force Index = (Close - Previous Close) * Volume
    
    信号解释：
    - 正值：买方力量强于卖方
    - 负值：卖方力量强于买方
    - 数值大小：反映力量的强弱程度
    """
    
    def __init__(self, period: int = 13, **kwargs):
        """
        初始化力量指数指标
        
        Args:
            period: 平滑周期，默认13
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.period = period
        self.REQUIRED_COLUMNS = ['close', 'volume']
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'period': 13
        }
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        self.period = kwargs.get('period', self.period)
    
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
        计算力量指数

        Args:
            data: 包含close和volume列的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含力量指数的DataFrame
        """
        try:
            # 验证数据
            if not self._validate_data(data):
                return pd.DataFrame()

            df = data.copy()
            
            # 计算价格变化
            price_change = df['close'].diff()
            
            # 计算原始力量指数
            raw_force_index = price_change * df['volume']
            
            # 计算平滑的力量指数（使用EMA）
            force_index = raw_force_index.ewm(span=self.period, adjust=False).mean()
            
            # 添加到结果DataFrame
            df['force_index'] = force_index
            df['raw_force_index'] = raw_force_index
            
            # 计算信号
            df['fi_signal'] = self._generate_signals(df)
            
            return df
            
        except Exception as e:
            logger.error(f"力量指数计算失败: {e}")
            return pd.DataFrame()
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含力量指数的DataFrame
            
        Returns:
            pd.Series: 交易信号
        """
        signals = pd.Series(0, index=df.index)
        
        # 力量指数穿越零轴信号
        force_index = df['force_index']
        
        # 买入信号：力量指数从负转正
        buy_signal = (force_index > 0) & (force_index.shift(1) <= 0)
        signals[buy_signal] = 1
        
        # 卖出信号：力量指数从正转负
        sell_signal = (force_index < 0) & (force_index.shift(1) >= 0)
        signals[sell_signal] = -1
        
        return signals
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取最新的交易信号
        
        Args:
            data: 计算后的数据
            
        Returns:
            Dict[str, Any]: 信号信息
        """
        if data.empty or 'fi_signal' not in data.columns:
            return {'signal': 0, 'strength': 0, 'description': '无信号'}
        
        latest_signal = data['fi_signal'].iloc[-1]
        latest_fi = data['force_index'].iloc[-1]
        
        if latest_signal == 1:
            return {
                'signal': 1,
                'strength': min(abs(latest_fi) / 1000, 1.0),  # 标准化强度
                'description': '买入信号：力量指数转正'
            }
        elif latest_signal == -1:
            return {
                'signal': -1,
                'strength': min(abs(latest_fi) / 1000, 1.0),
                'description': '卖出信号：力量指数转负'
            }
        else:
            return {
                'signal': 0,
                'strength': 0,
                'description': '无明确信号'
            }
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """获取指标模式信息"""
        return {
            'name': 'FORCE_INDEX',
            'description': '力量指数指标',
            'type': 'volume',
            'parameters': {
                'period': self.period
            }
        }

    # 实现BaseIndicator的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """核心计算逻辑"""
        return self.calculate(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if 'force_index' not in data.columns:
            return pd.Series(50.0, index=data.index)

        # 基于力量指数的强度计算评分
        fi = data['force_index'].abs()
        fi_normalized = fi / (fi.rolling(20).mean() + 1e-8)
        score = np.clip(fi_normalized * 50 + 50, 0, 100)
        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取技术形态"""
        patterns = pd.DataFrame(index=data.index)

        if 'force_index' in data.columns:
            fi = data['force_index']
            # 力量指数穿越零轴形态
            patterns['FI_零轴突破'] = (fi > 0) & (fi.shift(1) <= 0)
            patterns['FI_零轴跌破'] = (fi < 0) & (fi.shift(1) >= 0)
            # 力量指数强势形态
            patterns['FI_强势上涨'] = fi > fi.rolling(10).quantile(0.8)
            patterns['FI_强势下跌'] = fi < fi.rolling(10).quantile(0.2)

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.0

        # 基于评分稳定性和形态数量计算置信度
        score_stability = 1.0 - (score.rolling(5).std().iloc[-1] / 100.0) if len(score) >= 5 else 0.5
        pattern_strength = min(len(patterns) * 0.2, 1.0)

        return (score_stability + pattern_strength) / 2

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return self.period + 5
