#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抛物线转向系统(SAR)

判断价格趋势反转信号，提供买卖点
"""

import numpy as np
import pandas as pd
import talib
from typing import Union, List, Dict, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator
from utils.signal_utils import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class SAR(BaseIndicator):
    """
    抛物线转向系统(SAR) (SAR)
    
    分类：趋势跟踪指标
    描述：判断价格趋势反转信号，提供买卖点
    """
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化抛物线转向系统(SAR)指标
        
        Args:
            acceleration: 加速因子，默认为0.02
            maximum: 加速因子最大值，默认为0.2
        """
        super().__init__()
        self.acceleration = acceleration
        self.maximum = maximum
        self.name = "SAR"
        
        # 注册SAR形态
        # self._register_sar_patterns()
        
    def set_parameters(self, acceleration: float = None, maximum: float = None):
        """
        设置指标参数
        """
        if acceleration is not None:
            self.acceleration = acceleration
        if maximum is not None:
            self.maximum = maximum
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 数据帧
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少所需的列: {missing_columns}")
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算抛物线转向系统(SAR)指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值的DataFrame
        """
        self._validate_dataframe(df, ['high', 'low', 'close'])
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        length = len(df)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1为上升趋势，-1为下降趋势
        ep = np.zeros(length)     # 极点价格
        af = np.zeros(length)     # 加速因子
        
        # 初始化
        trend[0] = 1  # 假设初始为上升趋势
        ep[0] = high[0]  # 极点价格初始为第一个最高价
        sar[0] = low[0]  # SAR初始为第一个最低价
        af[0] = self.acceleration
        
        # 计算SAR
        for i in range(1, length):
            # 上一个周期是上升趋势
            if trend[i-1] == 1:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不高于前两个周期的最低价
                if i >= 2:
                    sar[i] = min(sar[i], min(low[i-1], low[i-2]))
                
                # 判断趋势是否反转
                if low[i] < sar[i]:
                    # 趋势反转为下降
                    trend[i] = -1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = low[i]     # 极点设为当前最低价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续上升趋势
                    trend[i] = 1
                    # 更新极点和加速因子
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            
            # 上一个周期是下降趋势
            else:
                # 计算当前SAR值
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # 确保SAR不低于前两个周期的最高价
                if i >= 2:
                    sar[i] = max(sar[i], max(high[i-1], high[i-2]))
                
                # 判断趋势是否反转
                if high[i] > sar[i]:
                    # 趋势反转为上升
                    trend[i] = 1
                    sar[i] = ep[i-1]  # SAR值设为前期极点
                    ep[i] = high[i]    # 极点设为当前最高价
                    af[i] = self.acceleration  # 加速因子重置
                else:
                    # 继续下降趋势
                    trend[i] = -1
                    # 更新极点和加速因子
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.acceleration, self.maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df.index)
        result['sar'] = sar
        result['trend'] = trend
        result['ep'] = ep
        result['af'] = af
        
        # 存储结果
        self._result = result
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据SAR生成买卖信号
        
        Args:
            df: 包含SAR值的DataFrame
            
        Returns:
            包含买卖信号的DataFrame
        """
        signals = pd.DataFrame(index=df.index)
        trend = df['trend'].values
        
        buy_signals = np.zeros(len(df))
        sell_signals = np.zeros(len(df))
        
        # 寻找趋势反转点作为信号
        for i in range(1, len(df)):
            # 趋势由下降转为上升，买入信号
            if trend[i] == 1 and trend[i-1] == -1:
                buy_signals[i] = 1
            
            # 趋势由上升转为下降，卖出信号
            if trend[i] == -1 and trend[i-1] == 1:
                sell_signals[i] = 1
        
        signals['buy'] = buy_signals
        signals['sell'] = sell_signals
        
        return signals
    
    def get_patterns(self):
        patterns = {
            "description": "收盘价下穿SAR，是卖出信号。",
        }
        return patterns

    def calculate_raw_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算SAR原始评分
        """
        if self._result is None:
            self.calculate(data)
        
        # 评分逻辑...
        score = pd.Series(50.0, index=data.index)
        return score

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算SAR指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            包含SAR值和信号的DataFrame
        """
        try:
            result = self.calculate(df)
            signals = self.generate_signals(result)
            # 合并结果
            result['buy_signal'] = signals['buy']
            result['sell_signal'] = signals['sell']
            
            return result
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise

    def _register_sar_patterns(self):
        """
        注册SAR形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册趋势反转形态
        registry.register(
            pattern_id="SAR_BULLISH_REVERSAL",
            display_name="SAR做多信号",
            description="SAR由下降趋势转为上升趋势，产生做多信号",
            indicator_id="SAR",
            pattern_type=PatternType.REVERSAL,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="SAR_BEARISH_REVERSAL",
            display_name="SAR做空信号",
            description="SAR由上升趋势转为下降趋势，产生做空信号",
            indicator_id="SAR",
            pattern_type=PatternType.REVERSAL,
            score_impact=-15.0
        )
        
        # 注册趋势持续形态
        registry.register(
            pattern_id="SAR_STRONG_UPTREND",
            display_name="SAR强势上升趋势",
            description="SAR长期保持在价格下方，表示强势上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="SAR_UPTREND",
            display_name="SAR上升趋势",
            description="SAR保持在价格下方，表示上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=7.0
        )
        
        registry.register(
            pattern_id="SAR_SHORT_UPTREND",
            display_name="SAR短期上升趋势",
            description="SAR刚刚转为上升趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_STRONG_DOWNTREND",
            display_name="SAR强势下降趋势",
            description="SAR长期保持在价格上方，表示强势下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-10.0
        )
        
        registry.register(
            pattern_id="SAR_DOWNTREND",
            display_name="SAR下降趋势",
            description="SAR保持在价格上方，表示下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-7.0
        )
        
        registry.register(
            pattern_id="SAR_SHORT_DOWNTREND",
            display_name="SAR短期下降趋势",
            description="SAR刚刚转为下降趋势",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=-5.0
        )
        
        # 注册SAR距离形态
        registry.register(
            pattern_id="SAR_CLOSE_TO_PRICE",
            display_name="SAR接近价格",
            description="SAR与价格距离较近，可能即将反转",
            indicator_id="SAR",
            pattern_type=PatternType.WARNING,
            score_impact=0.0
        )
        
        registry.register(
            pattern_id="SAR_MODERATE_DISTANCE",
            display_name="SAR与价格中等距离",
            description="SAR与价格保持中等距离",
            indicator_id="SAR",
            pattern_type=PatternType.CONTINUATION,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_FAR_FROM_PRICE",
            display_name="SAR远离价格",
            description="SAR与价格距离较远，趋势强劲",
            indicator_id="SAR",
            pattern_type=PatternType.TREND,
            score_impact=8.0
        )
        
        # 注册加速因子形态
        registry.register(
            pattern_id="SAR_HIGH_ACCELERATION",
            display_name="SAR高加速趋势",
            description="SAR加速因子较高，趋势强劲",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="SAR_MEDIUM_ACCELERATION",
            display_name="SAR中等加速趋势",
            description="SAR加速因子中等，趋势稳定",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="SAR_LOW_ACCELERATION",
            display_name="SAR低加速趋势",
            description="SAR加速因子较低，趋势刚开始或较弱",
            indicator_id="SAR",
            pattern_type=PatternType.MOMENTUM,
            score_impact=2.0
        )
        
        # 注册趋势稳定性形态
        registry.register(
            pattern_id="SAR_STABLE_TREND",
            display_name="SAR稳定趋势",
            description="SAR趋势稳定，没有频繁转向",
            indicator_id="SAR",
            pattern_type=PatternType.STABILITY,
            score_impact=8.0
        )
        
        registry.register(
            pattern_id="SAR_VOLATILE_TREND",
            display_name="SAR波动趋势",
            description="SAR趋势不稳定，频繁转向",
            indicator_id="SAR",
            pattern_type=PatternType.STABILITY,
            score_impact=-5.0
        )
        
        # 注册支撑/阻力形态
        registry.register(
            pattern_id="SAR_AS_SUPPORT",
            display_name="SAR支撑",
            description="SAR作为价格支撑位",
            indicator_id="SAR",
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="SAR_AS_RESISTANCE",
            display_name="SAR阻力",
            description="SAR作为价格阻力位",
            indicator_id="SAR",
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            score_impact=-12.0
        )

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
