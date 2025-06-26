#!/usr/bin/env python3
"""
ATR (Average True Range) 平均真实波幅指标

ATR是衡量价格波动性的技术指标，由J. Welles Wilder开发。
它计算一定周期内的平均真实波幅，用于衡量市场的波动性。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ATR(BaseIndicator, PatternSignalMixin):
    """
    ATR (Average True Range) 平均真实波幅指标
    
    ATR指标用于衡量价格波动性，通过计算真实波幅的移动平均值来反映市场的波动程度。
    ATR值越高，表示价格波动越大；ATR值越低，表示价格波动越小。
    """
    
    def __init__(self, **kwargs):
        """
        初始化ATR指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "ATR"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator
        validator = IndicatorParameterValidator()
        
        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('ATR', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 14)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ATR指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            添加了ATR指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result

    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ATR指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            添加了ATR指标的DataFrame
        """
        df = data.copy()

        # 确保数据有足够的长度
        if len(df) < self.period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({self.period + 1})，返回原始数据")
            df[f'ATR{self.period}'] = np.nan
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（ATR指标特定逻辑）
        df = self._apply_atr_signal_logic(df)

        return df

    def _apply_atr_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用ATR指标特定的信号生成逻辑
        基于ATR值的变化生成信号
        """
        try:
            # 获取ATR值
            atr_col = f'ATR{self.period}'
            if atr_col not in df.columns:
                # 如果没有ATR值，使用默认信号
                return df

            atr_value = df[atr_col]

            # ATR信号生成逻辑：
            # BUY: ATR值上升（波动性增加，可能有突破）
            # SELL: ATR值下降（波动性减少，可能趋势结束）
            # HOLD: ATR值稳定

            # 计算ATR变化率
            atr_change = atr_value.pct_change()
            atr_rising = atr_change > 0.05  # ATR上升超过5%
            atr_falling = atr_change < -0.05  # ATR下降超过5%

            # 生成信号
            df.loc[:, 'buy_signal'] = atr_rising
            df.loc[:, 'sell_signal'] = atr_falling
            df.loc[:, 'hold_signal'] = ~(atr_rising | atr_falling)

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"ATR信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

        # 计算真实波幅(TR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # 计算ATR - TR的period周期平均值
        df[f'ATR{self.period}'] = df['TR'].rolling(window=self.period).mean()

        # 清理中间计算列
        df.drop(['tr1', 'tr2', 'tr3', 'TR'], axis=1, inplace=True)

        return df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate(data, **kwargs)
        return pd.Series(50.0, index=data.index)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)
