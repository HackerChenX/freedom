#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
终极振荡器(Ultimate Oscillator)指标

终极振荡器是由Larry Williams开发的动量振荡器，它结合了三个不同时间周期的价格动量，
以减少虚假信号并提供更可靠的买卖信号。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class Ultimate(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    终极振荡器(Ultimate Oscillator)指标
    
    分类：振荡器指标
    描述：结合三个不同周期的动量指标，减少虚假信号
    
    计算公式：
    1. BP = Close - min(Low, Previous Close)
    2. TR = max(High, Previous Close) - min(Low, Previous Close)
    3. Average7 = sum(BP, 7) / sum(TR, 7)
    4. Average14 = sum(BP, 14) / sum(TR, 14)
    5. Average28 = sum(BP, 28) / sum(TR, 28)
    6. UO = 100 * (4*Average7 + 2*Average14 + Average28) / (4+2+1)
    
    信号解释：
    - 超买：UO > 70
    - 超卖：UO < 30
    - 买入信号：从超卖区域向上突破
    - 卖出信号：从超买区域向下突破
    """
    
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28, **kwargs):
        """
        初始化终极振荡器指标
        
        Args:
            period1: 短期周期，默认7
            period2: 中期周期，默认14
            period3: 长期周期，默认28
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'period1': 7,
            'period2': 14,
            'period3': 28
        }
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        self.period1 = kwargs.get('period1', self.period1)
        self.period2 = kwargs.get('period2', self.period2)
        self.period3 = kwargs.get('period3', self.period3)
    
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
        计算终极振荡器

        Args:
            data: 包含high、low、close列的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含终极振荡器的DataFrame
        """
        try:
            # 验证数据
            if not self._validate_data(data):
                return pd.DataFrame()

            df = data.copy()
            
            # 计算前一日收盘价
            prev_close = df['close'].shift(1)
            
            # 计算买压(BP)和真实波幅(TR)
            bp = df['close'] - np.minimum(df['low'], prev_close)
            tr = np.maximum(df['high'], prev_close) - np.minimum(df['low'], prev_close)
            
            # 计算三个周期的平均值
            bp_sum1 = bp.rolling(window=self.period1).sum()
            tr_sum1 = tr.rolling(window=self.period1).sum()
            avg1 = bp_sum1 / tr_sum1
            
            bp_sum2 = bp.rolling(window=self.period2).sum()
            tr_sum2 = tr.rolling(window=self.period2).sum()
            avg2 = bp_sum2 / tr_sum2
            
            bp_sum3 = bp.rolling(window=self.period3).sum()
            tr_sum3 = tr.rolling(window=self.period3).sum()
            avg3 = bp_sum3 / tr_sum3
            
            # 计算终极振荡器
            ultimate_oscillator = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
            
            # 添加到结果DataFrame
            df['bp'] = bp
            df['tr'] = tr
            df['avg1'] = avg1
            df['avg2'] = avg2
            df['avg3'] = avg3
            df['ultimate_oscillator'] = ultimate_oscillator
            
            # 计算信号
            df['uo_signal'] = self._generate_signals(df)
            
            # 计算超买超卖状态
            df['uo_status'] = self._calculate_status(ultimate_oscillator)
            
            return df
            
        except Exception as e:
            logger.error(f"终极振荡器计算失败: {e}")
            return pd.DataFrame()
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含终极振荡器的DataFrame
            
        Returns:
            pd.Series: 交易信号
        """
        signals = pd.Series(0, index=df.index)
        uo = df['ultimate_oscillator']
        
        # 超买超卖阈值
        overbought = 70
        oversold = 30
        
        # 买入信号：从超卖区域向上突破30
        buy_signal = (uo > oversold) & (uo.shift(1) <= oversold) & (uo.shift(1) < uo)
        signals[buy_signal] = 1
        
        # 卖出信号：从超买区域向下突破70
        sell_signal = (uo < overbought) & (uo.shift(1) >= overbought) & (uo.shift(1) > uo)
        signals[sell_signal] = -1
        
        # 强买入信号：连续上升且突破50中线
        strong_buy = (uo > 50) & (uo.shift(1) <= 50) & (uo > uo.shift(1)) & (uo.shift(1) > uo.shift(2))
        signals[strong_buy] = 2
        
        # 强卖出信号：连续下降且跌破50中线
        strong_sell = (uo < 50) & (uo.shift(1) >= 50) & (uo < uo.shift(1)) & (uo.shift(1) < uo.shift(2))
        signals[strong_sell] = -2
        
        return signals
    
    def _calculate_status(self, uo: pd.Series) -> pd.Series:
        """
        计算超买超卖状态
        
        Args:
            uo: 终极振荡器序列
            
        Returns:
            pd.Series: 状态序列
        """
        status = pd.Series('中性', index=uo.index)
        status[uo >= 70] = '超买'
        status[uo <= 30] = '超卖'
        status[(uo > 50) & (uo < 70)] = '偏强'
        status[(uo > 30) & (uo < 50)] = '偏弱'
        
        return status
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取最新的交易信号
        
        Args:
            data: 计算后的数据
            
        Returns:
            Dict[str, Any]: 信号信息
        """
        if data.empty or 'uo_signal' not in data.columns:
            return {'signal': 0, 'strength': 0, 'description': '无信号'}
        
        latest_signal = data['uo_signal'].iloc[-1]
        latest_uo = data['ultimate_oscillator'].iloc[-1]
        latest_status = data['uo_status'].iloc[-1] if 'uo_status' in data.columns else '未知'
        
        # 计算信号强度
        if latest_uo >= 70:
            strength = min((latest_uo - 70) / 30, 1.0)
        elif latest_uo <= 30:
            strength = min((30 - latest_uo) / 30, 1.0)
        else:
            strength = abs(latest_uo - 50) / 50
        
        signal_descriptions = {
            2: f'强买入信号，UO值：{latest_uo:.2f}，状态：{latest_status}',
            1: f'买入信号，UO值：{latest_uo:.2f}，状态：{latest_status}',
            0: f'无明确信号，UO值：{latest_uo:.2f}，状态：{latest_status}',
            -1: f'卖出信号，UO值：{latest_uo:.2f}，状态：{latest_status}',
            -2: f'强卖出信号，UO值：{latest_uo:.2f}，状态：{latest_status}'
        }
        
        return {
            'signal': latest_signal,
            'strength': strength,
            'description': signal_descriptions.get(latest_signal, '未知信号')
        }
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """获取指标模式信息"""
        return {
            'name': 'ULTIMATE',
            'description': '终极振荡器指标',
            'type': 'oscillator',
            'parameters': {
                'period1': self.period1,
                'period2': self.period2,
                'period3': self.period3
            }
        }

    # 实现BaseIndicator的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """核心计算逻辑"""
        return self.calculate(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if 'ultimate_oscillator' not in data.columns:
            return pd.Series(50.0, index=data.index)

        # 直接使用终极振荡器值作为评分
        return data['ultimate_oscillator']

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取技术形态"""
        patterns = pd.DataFrame(index=data.index)

        if 'ultimate_oscillator' in data.columns:
            uo = data['ultimate_oscillator']
            # 超买超卖形态
            patterns['UO_超买'] = uo >= 70
            patterns['UO_超卖'] = uo <= 30
            patterns['UO_偏强'] = (uo > 50) & (uo < 70)
            patterns['UO_偏弱'] = (uo > 30) & (uo < 50)
            # 突破形态
            patterns['UO_超卖突破'] = (uo > 30) & (uo.shift(1) <= 30)
            patterns['UO_超买跌破'] = (uo < 70) & (uo.shift(1) >= 70)
            patterns['UO_中线突破'] = (uo > 50) & (uo.shift(1) <= 50)
            patterns['UO_中线跌破'] = (uo < 50) & (uo.shift(1) >= 50)

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.0

        # 基于振荡器位置和趋势计算置信度
        latest_score = score.iloc[-1]
        if latest_score >= 70 or latest_score <= 30:
            # 在极值区域，置信度较高
            confidence = 0.8
        else:
            # 在中间区域，置信度较低
            confidence = 0.4

        pattern_strength = min(len(patterns) * 0.1, 0.3)
        return min(confidence + pattern_strength, 1.0)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return max(self.period1, self.period2, self.period3) + 10
