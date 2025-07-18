"""
模拟指标计算器实现

用于测试环境中的简单指标计算器实现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union

from db.interfaces.indicator_calculator_interface import IindicatorCalculator
from enums.indicator_types import Indicatortype_indicator_types as IndicatorType


class MockIndicatorCalculator(IindicatorCalculator):
    """
    模拟指标计算器
    
    提供基本的技术指标计算功能用于测试
    """
    
    def calculate_Indicator_Calculator_Interface_indicator_calculator_interface(self, 
                  data: pd.DataFrame, 
                  params: Optional[Dict[str, Any]] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        计算指标（模拟实现）
        
        Args:
            data: 输入数据
            params: 计算参数
            
        Returns:
            Union[pd.Series, pd.DataFrame]: 计算结果
        """
        if data.empty:
            return pd.DataFrame()
        
        # 简单返回收盘价作为指标值
        return data.get('close', pd.Series())
    
    def get_indicator_type_Indicator_Calculator_Interface_indicator_calculator_interface(self) -> IndicatorType:
        """
        获取指标类型
        
        Returns:
            IndicatorType: 指标类型枚举
        """
        return IndicatorType.TREND
    
    def get_required_columns_indicator_calculator_interface(self) -> List[str]:
        """
        获取必需列
        
        Returns:
            List[str]: 必需的数据列名
        """
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_default_params_indicator_calculator_interface(self) -> Dict[str, Any]:
        """
        获取默认参数
        
        Returns:
            Dict[str, Any]: 默认参数字典
        """
        return {}
    
    def validate_data_indicator_calculator_interface(self, data: pd.DataFrame) -> bool:
        """
        验证数据有效性
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        required_cols = self.get_required_columns_indicator_calculator_interface()
        return all(col in data.columns for col in required_cols) 