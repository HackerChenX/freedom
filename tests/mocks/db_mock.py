#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
数据库模拟模块

提供模拟数据库服务，用于测试环境中模拟数据库交互
"""

import os
import json
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable

class MockClickHouseDB:
    """模拟ClickHouse数据库服务"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化模拟数据库
        
        Args:
            data_path: 模拟响应数据路径，如果提供则自动加载响应数据
        """
        self.responses = {}
        self.query_history = []
        
        if data_path:
            self.load_responses(data_path)
    
    def load_responses(self, path: str) -> None:
        """
        从文件加载模拟响应
        
        Args:
            path: 模拟响应数据文件或目录路径
        """
        if os.path.isdir(path):
            # 如果是目录，加载目录下所有JSON文件
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    file_path = os.path.join(path, filename)
                    self._load_response_file(file_path)
        elif os.path.isfile(path) and path.endswith('.json'):
            # 如果是JSON文件，直接加载
            self._load_response_file(path)
        else:
            raise ValueError(f"无效的模拟响应数据路径: {path}")
    
    def _load_response_file(self, file_path: str) -> None:
        """
        加载单个响应文件
        
        Args:
            file_path: 响应文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
            
            # 每个响应文件应该包含 pattern 和 response 字段
            for item in response_data:
                if 'pattern' in item and 'response' in item:
                    pattern = item['pattern']
                    response = item['response']
                    
                    # 如果响应是列表，转换为DataFrame
                    if isinstance(response, list) and response:
                        response = pd.DataFrame(response)
                    
                    self.responses[pattern] = response
    
    def add_response(self, query_pattern: str, response_data: Union[pd.DataFrame, List[Dict], Dict]) -> None:
        """
        添加模拟响应
        
        Args:
            query_pattern: SQL查询模式（正则表达式）
            response_data: 响应数据，可以是DataFrame或字典/列表
        """
        # 如果响应是列表，转换为DataFrame
        if isinstance(response_data, list) and response_data:
            response_data = pd.DataFrame(response_data)
            
        self.responses[query_pattern] = response_data
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        模拟查询执行
        
        Args:
            sql: SQL查询语句
            
        Returns:
            查询结果DataFrame
        """
        # 记录查询历史
        self.query_history.append(sql)
        
        # 根据SQL模式匹配返回预定义响应
        for pattern, response in self.responses.items():
            if re.search(pattern, sql, re.IGNORECASE):
                return response.copy() if isinstance(response, pd.DataFrame) else pd.DataFrame(response)
        
        # 如果没有匹配的响应，返回空DataFrame
        return pd.DataFrame()
    
    def execute(self, sql: str) -> None:
        """
        模拟执行非查询SQL
        
        Args:
            sql: SQL执行语句
        """
        # 记录执行历史
        self.query_history.append(sql)
    
    def get_query_history(self) -> List[str]:
        """
        获取查询历史
        
        Returns:
            查询历史列表
        """
        return self.query_history
    
    def clear_history(self) -> None:
        """清除查询历史"""
        self.query_history = []
    
    def reset(self) -> None:
        """重置模拟数据库"""
        self.responses = {}
        self.query_history = []


class MockDataManager:
    """模拟数据管理器，用于替代实际数据管理器进行测试"""
    
    def __init__(self, mock_db: Optional[MockClickHouseDB] = None):
        """
        初始化模拟数据管理器
        
        Args:
            mock_db: 模拟数据库实例，如果不提供则自动创建
        """
        self.db = mock_db or MockClickHouseDB()
        self.cache = {}
    
    def get_stock_list(self, market: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            market: 市场代码，例如 'SH', 'SZ'
            
        Returns:
            股票列表DataFrame
        """
        cache_key = f"stock_list_{market or 'ALL'}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        sql = "SELECT * FROM stock_list"
        if market:
            sql += f" WHERE market = '{market}'"
        
        result = self.db.query(sql)
        self.cache[cache_key] = result
        
        return result
    
    def get_kline_data(self, 
                       stock_code: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, 
                       period: str = 'daily') -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: K线周期，如 'daily', 'weekly', 'monthly'
            
        Returns:
            K线数据DataFrame
        """
        cache_key = f"kline_{stock_code}_{period}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        sql = f"SELECT * FROM kline_{period} WHERE code = '{stock_code}'"
        
        if start_date:
            sql += f" AND date >= '{start_date}'"
        
        if end_date:
            sql += f" AND date <= '{end_date}'"
        
        sql += " ORDER BY date"
        
        result = self.db.query(sql)
        self.cache[cache_key] = result
        
        return result
    
    def get_indicator_data(self, 
                          indicator_name: str, 
                          stock_code: str,
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取指标数据
        
        Args:
            indicator_name: 指标名称
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            指标数据DataFrame
        """
        cache_key = f"indicator_{indicator_name}_{stock_code}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        sql = f"SELECT * FROM indicator_{indicator_name} WHERE code = '{stock_code}'"
        
        if start_date:
            sql += f" AND date >= '{start_date}'"
        
        if end_date:
            sql += f" AND date <= '{end_date}'"
        
        sql += " ORDER BY date"
        
        result = self.db.query(sql)
        self.cache[cache_key] = result
        
        return result
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.cache = {}


# 用于测试的工厂函数
def create_mock_data_manager(response_data_path: Optional[str] = None) -> MockDataManager:
    """
    创建预配置的模拟数据管理器
    
    Args:
        response_data_path: 模拟响应数据路径
        
    Returns:
        配置好的模拟数据管理器
    """
    mock_db = MockClickHouseDB(response_data_path)
    return MockDataManager(mock_db) 