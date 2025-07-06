#!/usr/bin/env python3
"""
SQL查询管理系统

提供统一的SQL语句管理、模板化查询和参数化查询功能。
遵循架构规范，避免SQL语句分散在业务代码中。
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import pandas as pd
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    STOCK_DATA = "stock_data"
    STOCK_LIST = "stock_list"
    STOCK_INFO = "stock_info"
    INDICATOR_DATA = "indicator_data"
    STRATEGY_CONFIG = "strategy_config"
    INDUSTRY_LIST = "industry_list"
    DATE_RANGE = "date_range"
    STOCK_COUNT = "stock_count"
    LATEST_DATA = "latest_data"
    BATCH_STOCK_DATA = "batch_stock_data"
    PERFORMANCE_DATA = "performance_data"
    VALIDATION_DATA = "validation_data"

class SQLManager:
    """SQL查询管理器
    
    统一管理系统中的所有SQL查询语句，提供模板化查询和参数验证功能。
    """
    
    def __init__(self):
        self.queries = self._initialize_queries()
        self.required_params = self._initialize_required_params()
    
    def _initialize_queries(self) -> Dict[QueryType, str]:
        """初始化查询语句模板"""
        return {
            QueryType.STOCK_DATA: """
                SELECT code, name, date, open, high, low, close, volume, 
                       turnover_rate, price_change, price_range, industry
                FROM stock_info 
                WHERE code = %(code)s 
                AND date BETWEEN %(start_date)s AND %(end_date)s
                AND level = %(level)s
                ORDER BY date ASC
            """,
            
            QueryType.BATCH_STOCK_DATA: """
                SELECT code, name, date, open, high, low, close, volume, 
                       turnover_rate, price_change, price_range, industry
                FROM stock_info 
                WHERE code IN %(codes)s
                AND date BETWEEN %(start_date)s AND %(end_date)s
                AND level = %(level)s
                ORDER BY code, date ASC
            """,
            
            QueryType.STOCK_LIST: """
                SELECT DISTINCT code, name, industry
                FROM stock_info 
                WHERE date = (SELECT MAX(date) FROM stock_info)
                AND level = %(level)s
                ORDER BY code
            """,
            
            QueryType.STOCK_INFO: """
                SELECT code, name, industry, 
                       MAX(date) as latest_date,
                       COUNT(*) as record_count
                FROM stock_info 
                WHERE code = %(code)s
                AND level = %(level)s
                GROUP BY code, name, industry
            """,
            
            QueryType.INDICATOR_DATA: """
                SELECT * FROM %(table_name)s
                WHERE code = %(code)s 
                AND date BETWEEN %(start_date)s AND %(end_date)s
                ORDER BY date ASC
            """,
            
            QueryType.STRATEGY_CONFIG: """
                SELECT config FROM strategy_definitions 
                WHERE strategy_id = %(strategy_id)s 
                LIMIT 1
            """,
            
            QueryType.INDUSTRY_LIST: """
                SELECT DISTINCT industry, COUNT(*) as stock_count
                FROM stock_info 
                WHERE industry IS NOT NULL 
                AND date = (SELECT MAX(date) FROM stock_info)
                AND level = %(level)s
                GROUP BY industry
                ORDER BY industry
            """,
            
            QueryType.DATE_RANGE: """
                SELECT MIN(date) as start_date, MAX(date) as end_date
                FROM stock_info 
                WHERE code = %(code)s
                AND level = %(level)s
            """,
            
            QueryType.STOCK_COUNT: """
                SELECT COUNT(DISTINCT code) as total_stocks
                FROM stock_info 
                WHERE date = (SELECT MAX(date) FROM stock_info)
                AND level = %(level)s
            """,
            
            QueryType.LATEST_DATA: """
                SELECT code, name, date, open, high, low, close, volume, 
                       turnover_rate, price_change, price_range, industry
                FROM stock_info 
                WHERE code = %(code)s
                AND level = %(level)s
                ORDER BY date DESC
                LIMIT %(limit)s
            """,
            
            QueryType.PERFORMANCE_DATA: """
                SELECT code, date, close,
                       LAG(close, 1) OVER (PARTITION BY code ORDER BY date) as prev_close,
                       (close - LAG(close, 1) OVER (PARTITION BY code ORDER BY date)) / 
                       LAG(close, 1) OVER (PARTITION BY code ORDER BY date) * 100 as return_pct
                FROM stock_info 
                WHERE code IN %(codes)s
                AND date BETWEEN %(start_date)s AND %(end_date)s
                AND level = %(level)s
                ORDER BY code, date
            """,
            
            QueryType.VALIDATION_DATA: """
                SELECT code, date, open, high, low, close, volume
                FROM stock_info 
                WHERE code = %(code)s
                AND date = %(date)s
                AND level = %(level)s
                LIMIT 1
            """
        }
    
    def _initialize_required_params(self) -> Dict[QueryType, List[str]]:
        """初始化每种查询类型的必需参数"""
        return {
            QueryType.STOCK_DATA: ['code', 'start_date', 'end_date', 'level'],
            QueryType.BATCH_STOCK_DATA: ['codes', 'start_date', 'end_date', 'level'],
            QueryType.STOCK_LIST: ['level'],
            QueryType.STOCK_INFO: ['code', 'level'],
            QueryType.INDICATOR_DATA: ['table_name', 'code', 'start_date', 'end_date'],
            QueryType.STRATEGY_CONFIG: ['strategy_id'],
            QueryType.INDUSTRY_LIST: ['level'],
            QueryType.DATE_RANGE: ['code', 'level'],
            QueryType.STOCK_COUNT: ['level'],
            QueryType.LATEST_DATA: ['code', 'level', 'limit'],
            QueryType.PERFORMANCE_DATA: ['codes', 'start_date', 'end_date', 'level'],
            QueryType.VALIDATION_DATA: ['code', 'date', 'level']
        }
    
    def get_query(self, query_type: QueryType) -> str:
        """获取查询语句模板
        
        Args:
            query_type: 查询类型
            
        Returns:
            str: SQL查询语句模板
            
        Raises:
            ValueError: 查询类型不存在
        """
        if query_type not in self.queries:
            raise ValueError(f"查询类型 {query_type} 不存在")
        
        return self.queries[query_type].strip()
    
    def validate_params(self, query_type: QueryType, params: Dict[str, Any]) -> bool:
        """验证查询参数
        
        Args:
            query_type: 查询类型
            params: 查询参数
            
        Returns:
            bool: 参数是否有效
            
        Raises:
            ValueError: 参数验证失败
        """
        if query_type not in self.required_params:
            raise ValueError(f"查询类型 {query_type} 不存在")
        
        required = self.required_params[query_type]
        missing_params = [param for param in required if param not in params]
        
        if missing_params:
            raise ValueError(f"缺少必需参数: {missing_params}")
        
        # 特殊参数验证
        if 'level' in params and params['level'] not in ['日线', '周线', '月线']:
            raise ValueError(f"无效的level参数: {params['level']}")
        
        if 'limit' in params and (not isinstance(params['limit'], int) or params['limit'] <= 0):
            raise ValueError(f"limit参数必须是正整数: {params['limit']}")
        
        return True
    
    def build_query(self, query_type: QueryType, params: Dict[str, Any]) -> str:
        """构建完整的SQL查询语句
        
        Args:
            query_type: 查询类型
            params: 查询参数
            
        Returns:
            str: 完整的SQL查询语句
        """
        # 验证参数
        self.validate_params(query_type, params)
        
        # 获取查询模板
        query_template = self.get_query(query_type)
        
        # 处理特殊参数格式
        formatted_params = self._format_params(params)
        
        try:
            # 构建查询语句
            query = query_template % formatted_params
            logger.debug(f"构建查询语句: {query_type.value}")
            return query
        except (KeyError, TypeError) as e:
            raise ValueError(f"查询参数格式错误: {e}")
    
    def _format_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """格式化查询参数
        
        Args:
            params: 原始参数
            
        Returns:
            Dict[str, Any]: 格式化后的参数
        """
        formatted = {}
        
        for key, value in params.items():
            if key == 'codes' and isinstance(value, (list, tuple)):
                # 处理代码列表
                codes_str = "','".join(str(code) for code in value)
                formatted[key] = f"('{codes_str}')"
            elif isinstance(value, str):
                formatted[key] = f"'{value}'"
            else:
                formatted[key] = value
        
        return formatted
    
    def add_custom_query(self, query_type: str, query_template: str, 
                        required_params: List[str]) -> None:
        """添加自定义查询模板
        
        Args:
            query_type: 查询类型名称
            query_template: 查询模板
            required_params: 必需参数列表
        """
        # 创建动态枚举值（仅用于自定义查询）
        custom_type = f"CUSTOM_{query_type.upper()}"
        
        # 存储自定义查询
        if not hasattr(self, '_custom_queries'):
            self._custom_queries = {}
            self._custom_required_params = {}
        
        self._custom_queries[custom_type] = query_template
        self._custom_required_params[custom_type] = required_params
        
        logger.info(f"添加自定义查询类型: {custom_type}")
    
    def get_custom_query(self, query_type: str) -> str:
        """获取自定义查询模板
        
        Args:
            query_type: 自定义查询类型
            
        Returns:
            str: 查询模板
        """
        custom_type = f"CUSTOM_{query_type.upper()}"
        
        if not hasattr(self, '_custom_queries') or custom_type not in self._custom_queries:
            raise ValueError(f"自定义查询类型 {custom_type} 不存在")
        
        return self._custom_queries[custom_type]
    
    def list_query_types(self) -> List[str]:
        """列出所有可用的查询类型
        
        Returns:
            List[str]: 查询类型列表
        """
        standard_types = [qt.value for qt in QueryType]
        custom_types = []
        
        if hasattr(self, '_custom_queries'):
            custom_types = list(self._custom_queries.keys())
        
        return standard_types + custom_types

# 全局SQL管理器实例
sql_manager = SQLManager()

def get_sql_manager() -> SQLManager:
    """获取SQL管理器实例
    
    Returns:
        SQLManager: SQL管理器实例
    """
    return sql_manager

class QueryBuilder:
    """查询构建器"""
    
    def from_template(self, query_type: str) -> 'QueryBuilder':
        """从模板开始构建查询"""
        return self
    
    def where(self, field: str, value: Any) -> 'QueryBuilder':
        """添加WHERE条件"""
        return self
    
    def order_by(self, *fields: str) -> 'QueryBuilder':
        """添加排序"""
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """添加限制"""
        return self
    
    def build(self) -> tuple[str, Dict[str, Any]]:
        """构建最终查询"""
        return "", {}

# 异常类
class QueryNotFoundException(Exception):
    """查询不存在异常"""
    pass

class QueryValidationException(Exception):
    """查询验证异常"""
    pass

# 工厂函数
def create_sql_manager() -> SQLManager:
    """创建SQL管理器实例"""
    return SQLManager()

def create_query_builder(sql_manager: SQLManager) -> QueryBuilder:
    """创建查询构建器实例"""
    return QueryBuilder() 