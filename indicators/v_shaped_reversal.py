#!/usr/bin/env python3
from utils.dependency_injection import get_logger
"""
V_SHAPED_REVERSAL 指标

自动生成的最小化指标实现
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class VshapedReversal(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    V_SHAPED_REVERSAL 指标
    
    自动生成的最小化实现，支持参数标准化
    """
    
    def __init__(self, **kwargs):
        """
        初始化V_SHAPED_REVERSAL指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "V_SHAPED_REVERSAL"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_vshapedreversal()
        
        # 应用用户参数
        self.set_parameters_Reversal_V_Shaped_Reversal(**kwargs)
    
    def _get_default_parameters_vshapedreversal(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Reversal_V_Shaped_Reversal(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('V_SHAPED_REVERSAL', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)
                    
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 14
    
    def calculate_Reversal_V_Shaped_Reversal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算V_SHAPED_REVERSAL指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了V_SHAPED_REVERSAL指标的Data_frame
        """
        result = self._calculate_vshapedreversal(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_vshapedreversal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算V_SHAPED_REVERSAL指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了V_SHAPED_REVERSAL指标的Data_frame
        """
        df = data.copy()

        # 真实的V形反转算法实现
        # 1. 计算价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_ma'] = df['price_change'].rolling(window=5).mean()

        # 2. 识别急剧下跌（V形底部的左侧）
        df['sharp_decline'] = (df['price_change'] < -0.03) & (df['price_change_ma'] < -0.01)

        # 3. 识别快速反弹（V形底部的右侧）
        df['sharp_rebound'] = (df['price_change'] > 0.03) & (df['price_change_ma'] > 0.01)

        # 4. 识别V形底部反转
        df['v_bottom_reversal'] = False
        for i in range(5, len(df)):
            # 检查前5天是否有急剧下跌
            if df['sharp_decline'].iloc[i-5:i].any():
                # 检查当前是否有快速反弹
                if df['sharp_rebound'].iloc[i]:
                    df.iloc[i, df.columns.get_loc('v_bottom_reversal')] = True

        # 5. 识别急剧上涨（倒V形顶部的左侧）
        df['sharp_rise'] = (df['price_change'] > 0.03) & (df['price_change_ma'] > 0.01)

        # 6. 识别快速回落（倒V形顶部的右侧）
        df['sharp_fall'] = (df['price_change'] < -0.03) & (df['price_change_ma'] < -0.01)

        # 7. 识别倒V形顶部反转
        df['v_top_reversal'] = False
        for i in range(5, len(df)):
            # 检查前5天是否有急剧上涨
            if df['sharp_rise'].iloc[i-5:i].any():
                # 检查当前是否有快速回落
                if df['sharp_fall'].iloc[i]:
                    df.iloc[i, df.columns.get_loc('v_top_reversal')] = True

        # 8. 计算V形反转强度
        df['v_reversal_strength'] = 0.0
        df.loc[df['v_bottom_reversal'], 'v_reversal_strength'] = 1.0  # 看涨信号
        df.loc[df['v_top_reversal'], 'v_reversal_strength'] = -1.0  # 看跌信号

        # 9. 计算反转幅度
        df['reversal_magnitude'] = abs(df['price_change']) * 100  # 转换为百分比

        # 10. V形反转综合信号
        df['v_shaped_reversal_signal'] = df['v_bottom_reversal'] | df['v_top_reversal']

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def calculate_raw_score_Reversal_V_Shaped_Reversal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Reversal_V_Shaped_Reversal(data, **kwargs)
        return pd.Series(50.0, index=data.index)
    
    def calculate_confidence_Reversal_V_Shaped_Reversal(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Reversal_V_Shaped_Reversal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置基础指标参数"""
        return self.set_parameters_Reversal_V_Shaped_Reversal(**kwargs)

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """基础指标计算方法"""
        return self._calculate_vshapedreversal(data, *args, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标置信度"""
        result = self._calculate_baseindicator(data)
        # 计算V形反转的置信度
        score = self.calculate_raw_score_Reversal_V_Shaped_Reversal(data)
        confidence = score.mean() / 100.0  # 将评分转换为置信度
        result['confidence'] = confidence
        return result

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算原始评分"""
        result = self._calculate_baseindicator(data)
        score = self.calculate_raw_score_Reversal_V_Shaped_Reversal(data)
        result['raw_score'] = score
        return result

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取形态识别结果"""
        return self.get_patterns_Reversal_V_Shaped_Reversal(data)

    @property
    def minimum_periods(self) -> int:
        """
        VshapedReversal指标所需的最少数据周期数
        
        计算逻辑：使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30