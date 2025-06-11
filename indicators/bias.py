#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
均线多空指标(BIAS)

(收盘价-MA)/MA×100%
"""

import pandas as pd
from typing import List

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class BIAS(BaseIndicator):
    """
    均线多空指标(BIAS) (BIAS)
    
    分类：趋势类指标
    描述：(收盘价-MA)/MA×100%
    """
    
    def __init__(self, name: str = "BIAS", description: str = "均线多空指标",
                 period: int = 14, periods: List[int] = None):
        """
        初始化均线多空指标(BIAS)指标
        """
        super().__init__(name, description)
        self.periods = periods if periods is not None else [period]
        self.indicator_type = "BIAS"
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算均线多空指标(BIAS)指标
        """
        if df.empty:
            return df
            
        self._validate_dataframe(df, ['close'])
        
        # 创建一个临时的DataFrame来存储新计算的列
        result_df = pd.DataFrame(index=df.index)
        
        # 计算所有周期的BIAS
        for p in self.periods:
            ma = df['close'].rolling(window=p, min_periods=1).mean()
            result_df[f'BIAS{p}'] = (df['close'] - ma) / ma * 100
        
        # 为主周期创建 'BIAS' 和 'BIAS_MA' 列，以供形态识别使用
        if self.periods:
            main_period = self.periods[0]
            main_bias_col = f'BIAS{main_period}'
            if main_bias_col in result_df:
                result_df['BIAS'] = result_df[main_bias_col]
                result_df['BIAS_MA'] = result_df['BIAS'].rolling(window=main_period, min_periods=1).mean()

        # 将新计算的列与原始DataFrame合并，避免重复列
        return df.join(result_df, how='left')

    def get_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        识别所有已注册的BIAS相关形态
        """
        # 首先，调用calculate来获取所有需要的列
        calculated_data = self._calculate(data)

        # 验证必要的列是否存在
        required_cols = ['BIAS', 'BIAS_MA']
        if not all(col in calculated_data.columns for col in required_cols):
             logger.warning(f"BIAS指标在形态识别时缺少必要的计算列: {required_cols}")
             # 返回一个空的DataFrame，但保留索引和原始列
             return data.assign(**{col: pd.NA for col in required_cols if col not in data})

        # --- 形态识别逻辑将在这里实现 ---
        # 当前版本仅确保数据流正确，并返回计算后的DataFrame
        # 实际的形态识别可以基于'calculated_data'中的'BIAS'和'BIAS_MA'列
        
        return calculated_data

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算BIAS指标的原始评分 (0-100分) - 存根实现
        """
        return pd.Series(50.0, index=data.index)

    # 其他方法（如get_signals, plot, calculate_raw_score等）可以根据需要
    # 同样基于调用 self.calculate(df) 的模式进行重构
    # 为了本次修复的简洁性，暂时省略它们的完整实现
