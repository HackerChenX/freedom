#!/usr/bin/env python3
"""
公式指标模块
包含各种技术分析公式指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class FORMULA_INDICATORS(BaseIndicator, PatternSignalMixin):
    """
    公式指标基类
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "FORMULA_INDICATORS"
        self._default_parameters = self._get_default_parameters()
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        return {"period": 14}
    
    def set_parameters(self, **kwargs):
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            params = self._default_parameters.copy()
            params.update(kwargs)
            is_valid, errors = validator.validate_indicator_parameters('FORMULA_INDICATORS', params)
            if not is_valid:
                params = self._default_parameters.copy()
            self.period = params.get('period', 14)
        except Exception:
            self.period = 14
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df[f'FORMULA_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if not self.has_result():
            self.calculate(data, **kwargs)
        return pd.Series(50.0, index=data.index)
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        return 0.5
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)


class CrossOver(FORMULA_INDICATORS):
    """交叉指标"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CrossOver"


class KDJCondition(FORMULA_INDICATORS):
    """KDJ条件指标"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "KDJCondition"


class MACDCondition(FORMULA_INDICATORS):
    """MACD条件指标"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MACDCondition"


class MACondition(FORMULA_INDICATORS):
    """MA条件指标"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MACondition"


class GenericCondition(FORMULA_INDICATORS):
    """通用条件指标"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GenericCondition"


# 为了向后兼容，创建别名
FormulaIndicators = FORMULA_INDICATORS
