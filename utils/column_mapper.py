#!/usr/bin/env python3
"""
通用数据列名映射工具类
用于统一处理不同数据源的列名格式差异
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)


class ColumnMapper:
    """
    通用数据列名映射工具类
    
    支持多种常见的股票数据列名格式，自动识别和映射到标准格式
    """
    
    # 标准列名映射表
    COLUMN_MAPPINGS = {
        'open': ['open', 'Open', 'OPEN', 'o', 'O', 'opening_price', 'Opening_Price'],
        'high': ['high', 'High', 'HIGH', 'h', 'H', 'highest_price', 'Highest_Price'],
        'low': ['low', 'Low', 'LOW', 'l', 'L', 'lowest_price', 'Lowest_Price'],
        'close': ['close', 'Close', 'CLOSE', 'c', 'C', 'closing_price', 'Closing_Price', 'price'],
        'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'v', 'V', 
                  'amount', 'Amount', 'AMOUNT', 'turnover', 'Turnover', 'TURNOVER',
                  'trading_volume', 'Trading_Volume', 'trade_volume'],
        'adj_close': ['adj_close', 'Adj_Close', 'ADJ_CLOSE', 'adjusted_close', 'Adjusted_Close'],
        'date': ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'DATETIME', 'time', 'Time', 'timestamp'],
        'code': ['code', 'Code', 'CODE', 'symbol', 'Symbol', 'SYMBOL', 'stock_code', 'Stock_Code']
    }
    
    @classmethod
    def find_column(cls, df: pd.DataFrame, column_type: str) -> Optional[str]:
        """
        在DataFrame中查找指定类型的列名
        
        Args:
            df: 数据DataFrame
            column_type: 列类型 ('open', 'high', 'low', 'close', 'volume', etc.)
            
        Returns:
            str: 实际的列名，如果找不到返回None
        """
        if column_type not in cls.COLUMN_MAPPINGS:
            logger.warning(f"不支持的列类型: {column_type}")
            return None
        
        possible_names = cls.COLUMN_MAPPINGS[column_type]
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        return None
    
    @classmethod
    def get_column_safe(cls, df: pd.DataFrame, column_type: str, default_value=None) -> pd.Series:
        """
        安全地获取指定类型的列数据
        
        Args:
            df: 数据DataFrame
            column_type: 列类型
            default_value: 如果找不到列时的默认值
            
        Returns:
            pd.Series: 列数据，如果找不到返回默认值填充的Series
        """
        column_name = cls.find_column(df, column_type)
        
        if column_name is not None:
            return df[column_name]
        else:
            logger.warning(f"无法找到{column_type}列，使用默认值: {default_value}")
            if default_value is not None:
                return pd.Series(default_value, index=df.index)
            else:
                return pd.Series(index=df.index, dtype=float)
    
    @classmethod
    def standardize_columns(cls, df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
        """
        标准化DataFrame的列名
        
        Args:
            df: 输入DataFrame
            required_columns: 必需的列类型列表，默认为['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            pd.DataFrame: 标准化后的DataFrame
        """
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        result_df = df.copy()
        mapping_log = []
        
        for column_type in required_columns:
            original_column = cls.find_column(df, column_type)
            
            if original_column is not None:
                # 如果原列名不是标准名，则重命名
                if original_column != column_type:
                    result_df = result_df.rename(columns={original_column: column_type})
                    mapping_log.append(f"{original_column} -> {column_type}")
            else:
                logger.warning(f"缺少必需的列: {column_type}")
                # 可以选择添加默认值列或抛出异常
                # result_df[column_type] = 0.0  # 添加默认值
        
        if mapping_log:
            logger.info(f"列名映射: {', '.join(mapping_log)}")
        
        return result_df
    
    @classmethod
    def validate_data(cls, df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Union[bool, List[str]]]:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入DataFrame
            required_columns: 必需的列类型列表
            
        Returns:
            dict: 验证结果，包含is_valid和missing_columns
        """
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        missing_columns = []
        found_columns = []
        
        for column_type in required_columns:
            column_name = cls.find_column(df, column_type)
            if column_name is not None:
                found_columns.append(f"{column_type}({column_name})")
            else:
                missing_columns.append(column_type)
        
        result = {
            'is_valid': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'found_columns': found_columns,
            'available_columns': list(df.columns)
        }
        
        return result
    
    @classmethod
    def get_ohlcv_data(cls, df: pd.DataFrame, strict: bool = False) -> Optional[pd.DataFrame]:
        """
        获取标准化的OHLCV数据
        
        Args:
            df: 输入DataFrame
            strict: 是否严格模式（缺少任何列都返回None）
            
        Returns:
            pd.DataFrame: 标准化的OHLCV数据，如果无法获取返回None
        """
        required_columns = ['open', 'high', 'low', 'close']
        optional_columns = ['volume']
        
        result_df = pd.DataFrame(index=df.index)
        
        # 处理必需列
        for column_type in required_columns:
            column_name = cls.find_column(df, column_type)
            if column_name is not None:
                result_df[column_type] = df[column_name]
            else:
                if strict:
                    logger.error(f"严格模式下缺少必需列: {column_type}")
                    return None
                else:
                    logger.warning(f"缺少必需列: {column_type}，使用默认值")
                    result_df[column_type] = 0.0
        
        # 处理可选列
        for column_type in optional_columns:
            column_name = cls.find_column(df, column_type)
            if column_name is not None:
                result_df[column_type] = df[column_name]
            else:
                if not strict:
                    logger.info(f"缺少可选列: {column_type}，使用默认值")
                    result_df[column_type] = 0.0
        
        return result_df
    
    @classmethod
    def add_column_mapping_support(cls, indicator_class):
        """
        为指标类添加列名映射支持的装饰器
        
        Args:
            indicator_class: 指标类
            
        Returns:
            装饰后的指标类
        """
        original_calculate = indicator_class._calculate
        
        def wrapped_calculate(self, data: pd.DataFrame, *args, **kwargs):
            # 在计算前标准化列名
            try:
                standardized_data = cls.standardize_columns(data)
                return original_calculate(self, standardized_data, *args, **kwargs)
            except Exception as e:
                logger.warning(f"列名标准化失败，使用原始数据: {e}")
                return original_calculate(self, data, *args, **kwargs)
        
        indicator_class._calculate = wrapped_calculate
        return indicator_class


# 便捷函数
def get_column_safe(df: pd.DataFrame, column_type: str, default_value=None) -> pd.Series:
    """便捷函数：安全地获取列数据"""
    return ColumnMapper.get_column_safe(df, column_type, default_value)


def find_column(df: pd.DataFrame, column_type: str) -> Optional[str]:
    """便捷函数：查找列名"""
    return ColumnMapper.find_column(df, column_type)


def standardize_columns(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
    """便捷函数：标准化列名"""
    return ColumnMapper.standardize_columns(df, required_columns)


def validate_data(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Union[bool, List[str]]]:
    """便捷函数：验证数据"""
    return ColumnMapper.validate_data(df, required_columns)
