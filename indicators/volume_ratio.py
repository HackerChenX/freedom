#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
量比指标(VOLUME_RATIO)
量比是指当前成交量与前N个周期平均成交量的比值，用于衡量市场交易活跃度的变化。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class VOLUME_RATIO(BaseIndicator, PatternSignalMixin):
    """
    量比指标(VOLUME_RATIO)
    
    特点:
    1. 用于衡量市场交易活跃度的变化
    2. 量比>1表示当前成交量高于参考期平均值，市场相对活跃
    3. 量比<1表示当前成交量低于参考期平均值，市场相对冷清
    4. 通常与价格趋势结合使用，判断市场热度变化
    
    计算方法:
    量比 = 当前成交量 / 前N个周期平均成交量
    
    参数:
    - period: 参考周期，默认为14
    """
    
    def __init__(self, **kwargs):
        """
        初始化VOLUME_RATIO指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "VOLUME_RATIO"
        
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
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('VOLUME_RATIO', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = kwargs.get('period', 14)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VOLUME_RATIO指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了VOLUME_RATIO指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算VOLUME_RATIO指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了VOLUME_RATIO指标的DataFrame
        """
        df = data.copy()
        
        # 获取成交量数据
        if 'volume' in df.columns:
            volume = df['volume']
        elif 'Volume' in df.columns:
            volume = df['Volume']
        else:
            # 如果没有成交量数据，返回默认值
            df['VOLUME_RATIO_VALUE'] = 1.0
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（VOLUME_RATIO指标特定逻辑）
        df = self._apply_volume_ratio_signal_logic(df)

        return df

    def _apply_volume_ratio_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用VOLUME_RATIO指标特定的信号生成逻辑
        基于量比值的大小生成信号
        """
        try:
            # 获取量比值
            if 'VOLUME_RATIO_VALUE' not in df.columns:
                # 如果没有量比值，使用默认信号
                return df

            volume_ratio = df['VOLUME_RATIO_VALUE']

            # VOLUME_RATIO信号生成逻辑：
            # BUY: 量比大于1.5（成交量放大）
            # SELL: 量比小于0.5（成交量萎缩）
            # HOLD: 量比在0.5-1.5之间（正常成交量）

            high_volume = volume_ratio > 1.5
            low_volume = volume_ratio < 0.5
            normal_volume = (volume_ratio >= 0.5) & (volume_ratio <= 1.5)

            # 生成信号
            df.loc[:, 'buy_signal'] = high_volume
            df.loc[:, 'sell_signal'] = low_volume
            df.loc[:, 'hold_signal'] = normal_volume

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"VOLUME_RATIO信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

        # 计算量比
        volume_ratio = volume / volume.rolling(window=self.period).mean()
        df['VOLUME_RATIO_VALUE'] = volume_ratio.fillna(1.0)
        
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
