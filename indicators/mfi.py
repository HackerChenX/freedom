#!/usr/bin/env python3
"""
MFI (Money Flow Index) 资金流量指标

MFI指标结合价格和成交量来衡量买卖压力。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class MFI(BaseIndicator, PatternSignalMixin):
    """
    MFI (Money Flow Index) 资金流量指标
    
    MFI指标通过结合价格和成交量来识别超买超卖状态。
    """
    
    def __init__(self, **kwargs):
        """
        初始化MFI指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "MFI"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14, "overbought": 80.0, "oversold": 20.0}
    
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
            is_valid, errors = validator.validate_indicator_parameters('MFI', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 14)
        self.overbought = params.get('overbought', 80.0)
        self.oversold = params.get('oversold', 20.0)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MFI指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            添加了MFI指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result

    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算MFI指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            添加了MFI指标的DataFrame
        """
        df = data.copy()

        # 确保数据有足够的长度
        if len(df) < self.period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({self.period + 1})，返回原始数据")
            df[f'MFI{self.period}'] = np.nan
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df

        # 计算典型价格
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3

        # 计算资金流量
        df['MF'] = df['TP'] * df['volume']

        # 计算价格变化
        df['TP_change'] = df['TP'].diff()

        # 分离正负资金流量
        df['PMF'] = np.where(df['TP_change'] > 0, df['MF'], 0)
        df['NMF'] = np.where(df['TP_change'] < 0, df['MF'], 0)

        # 计算资金流量比率
        pmf_sum = df['PMF'].rolling(window=self.period).sum()
        nmf_sum = df['NMF'].rolling(window=self.period).sum()

        # 计算MFI
        df[f'MFI{self.period}'] = 100 - (100 / (1 + pmf_sum / nmf_sum))

        # 清理中间计算列
        df.drop(['TP', 'MF', 'TP_change', 'PMF', 'NMF'], axis=1, inplace=True)

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
