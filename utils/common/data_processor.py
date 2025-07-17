#!/usr/bin/env python3
"""
通用数据处理工具
统一项目中的各种数据处理逻辑
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

class DataProcessor:
    """通用数据处理器"""
    
    @staticmethod
    def clean_stock_data(data: pd.DataFrame) -> pd.DataFrame:
        """清理股票数据"""
        if data.empty:
            return data
        
        # 移除重复行
        data = data.drop_duplicates()
        
        # 移除缺失值过多的行
        data = data.dropna(thresh=len(data.columns) * 0.8)
        
        # 数值列处理
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 移除异常值
            q1 = data[col].quantile(0.01)
            q99 = data[col].quantile(0.99)
            data = data[(data[col] >= q1) & (data[col] <= q99)]
        
        return data
    
    @staticmethod
    def calculate_basic_statistics(data: pd.Series) -> Dict[str, float]:
        """计算基础统计指标"""
        if data.empty:
            return {}
        
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'count': len(data)
        }
    
    @staticmethod
    def normalize_data(data: pd.Series, method: str = 'minmax') -> pd.Series:
        """数据标准化"""
        if data.empty:
            return data
        
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'zscore':
            return (data - data.mean()) / data.std()
        else:
            return data
