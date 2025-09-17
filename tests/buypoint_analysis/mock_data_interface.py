#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模拟数据接口

为买点分析测试提供模拟的数据访问接口，
使测试能够使用生成的测试数据而不依赖真实数据库
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from db.interfaces.data_access_interface import DataAccessInterface
from utils.logger import get_logger

logger = get_logger(__name__)


class MockDataInterface(DataAccessInterface):
    """模拟数据访问接口"""
    
    def __init__(self):
        """初始化模拟数据接口"""
        self.mock_data_store = {}  # 存储模拟数据
        self.logger = get_logger(__name__)
        
    def register_mock_data(self, stock_code: str, data: pd.DataFrame):
        """
        注册模拟数据
        
        Args:
            stock_code: 股票代码
            data: 股票数据
        """
        self.mock_data_store[stock_code] = data
        self.logger.debug(f"注册模拟数据: {stock_code}, {len(data)} 行")
    
    def clear_mock_data(self):
        """清除所有模拟数据"""
        self.mock_data_store.clear()
        self.logger.debug("清除所有模拟数据")
    
    def get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            pd.DataFrame: 股票数据
        """
        if stock_code not in self.mock_data_store:
            self.logger.warning(f"未找到股票代码的模拟数据: {stock_code}")
            return None
        
        data = self.mock_data_store[stock_code].copy()
        
        # 转换日期格式进行过滤
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            
            # 确保date列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            
            # 过滤日期范围
            mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
            filtered_data = data[mask].copy()
            
            if len(filtered_data) == 0:
                self.logger.warning(f"在指定日期范围内未找到数据: {stock_code} ({start_date} - {end_date})")
                return None
            
            self.logger.debug(f"返回股票数据: {stock_code}, {len(filtered_data)} 行")
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败: {stock_code}, {e}")
            return None
    
    def get_stock_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 股票信息
        """
        if stock_code not in self.mock_data_store:
            return None
        
        data = self.mock_data_store[stock_code]
        if len(data) == 0:
            return None
        
        # 从数据中提取基本信息
        return {
            'code': stock_code,
            'name': data['name'].iloc[0] if 'name' in data.columns else f'测试股票_{stock_code}',
            'industry': data['industry'].iloc[0] if 'industry' in data.columns else '测试行业',
            'market': 'TEST',
            'list_date': data['date'].min().strftime('%Y%m%d') if 'date' in data.columns else '20240101'
        }
    
    def get_stock_list(self, market: str = None) -> List[Dict[str, Any]]:
        """
        获取股票列表
        
        Args:
            market: 市场代码
            
        Returns:
            List[Dict[str, Any]]: 股票列表
        """
        stock_list = []
        
        for stock_code in self.mock_data_store.keys():
            stock_info = self.get_stock_info(stock_code)
            if stock_info:
                stock_list.append(stock_info)
        
        return stock_list
    
    def get_latest_trading_date(self) -> str:
        """
        获取最新交易日期
        
        Returns:
            str: 最新交易日期 (YYYYMMDD)
        """
        latest_date = datetime.now()
        
        # 从所有模拟数据中找到最新日期
        for data in self.mock_data_store.values():
            if 'date' in data.columns and len(data) > 0:
                data_latest = pd.to_datetime(data['date']).max()
                if data_latest > latest_date:
                    latest_date = data_latest
        
        return latest_date.strftime('%Y%m%d')
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        执行SQL查询（模拟实现）
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            List[Dict[str, Any]]: 查询结果
        """
        # 简单的模拟实现
        self.logger.warning("模拟数据接口不支持SQL查询")
        return []
    
    def insert_data(self, table: str, data: Dict[str, Any]) -> bool:
        """
        插入数据（模拟实现）
        
        Args:
            table: 表名
            data: 数据
            
        Returns:
            bool: 是否成功
        """
        self.logger.debug(f"模拟插入数据到表 {table}")
        return True
    
    def update_data(self, table: str, data: Dict[str, Any], condition: str) -> bool:
        """
        更新数据（模拟实现）
        
        Args:
            table: 表名
            data: 数据
            condition: 条件
            
        Returns:
            bool: 是否成功
        """
        self.logger.debug(f"模拟更新表 {table} 的数据")
        return True
    
    def delete_data(self, table: str, condition: str) -> bool:
        """
        删除数据（模拟实现）
        
        Args:
            table: 表名
            condition: 条件
            
        Returns:
            bool: 是否成功
        """
        self.logger.debug(f"模拟删除表 {table} 的数据")
        return True
    
    def close(self):
        """关闭连接（模拟实现）"""
        self.logger.debug("关闭模拟数据接口连接")
        self.clear_mock_data()

    # ==================== 实现抽象接口方法 ====================

    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取股票数据（接口方法）"""
        data = self.get_stock_data(code, start_date, end_date)
        if data is None:
            return pd.DataFrame()

        if columns:
            available_columns = [col for col in columns if col in data.columns]
            if available_columns:
                return data[available_columns]

        return data

    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """批量获取多只股票数据（接口方法）"""
        all_data = []

        for code in codes:
            data = self.get_stock_data_data_access_interface(code, start_date, end_date, columns)
            if not data.empty:
                all_data.append(data)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_indicator_data_data_access_interface(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """获取指标数据（接口方法）"""
        # 模拟实现：返回基础股票数据
        return self.get_stock_data_data_access_interface(code, start_date, end_date)

    def get_stock_list_data_access_interface(self, industry: Optional[str] = None,
                      market: Optional[str] = None) -> List[str]:
        """获取股票列表（接口方法）"""
        stock_codes = list(self.mock_data_store.keys())

        if industry:
            # 根据行业过滤
            filtered_codes = []
            for code in stock_codes:
                stock_info = self.get_stock_info(code)
                if stock_info and stock_info.get('industry') == industry:
                    filtered_codes.append(code)
            return filtered_codes

        return stock_codes

    def get_industry_list_data_access_interface(self) -> List[str]:
        """获取行业列表（接口方法）"""
        industries = set()

        for code in self.mock_data_store.keys():
            stock_info = self.get_stock_info(code)
            if stock_info and 'industry' in stock_info:
                industries.add(stock_info['industry'])

        return list(industries)

    def execute_query_data_access_interface(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """执行查询（接口方法）"""
        self.logger.warning("模拟数据接口不支持SQL查询")
        return pd.DataFrame()

    def check_data_exists_data_access_interface(self, table: str, conditions: Dict) -> bool:
        """检查数据是否存在（接口方法）"""
        # 简单的模拟实现
        if 'code' in conditions:
            return conditions['code'] in self.mock_data_store
        return len(self.mock_data_store) > 0

    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """获取最新数据（接口方法）"""
        if code not in self.mock_data_store:
            return None

        data = self.mock_data_store[code]
        if len(data) == 0:
            return None

        # 获取最新一行数据
        latest_row = data.iloc[-1]

        if columns:
            result = {}
            for col in columns:
                if col in latest_row:
                    result[col] = latest_row[col]
            return result
        else:
            return latest_row.to_dict()

    # ==================== 买点分析器专用方法 ====================

    def get_stock_info(self, code: str, level: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Optional[List[List]]:
        """
        获取股票信息（兼容买点分析器接口）

        Args:
            code: 股票代码
            level: K线级别（忽略）
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            List[List]: 股票数据行列表，格式兼容买点分析器
        """
        self.logger.info(f"模拟数据接口被调用: code={code}, level={level}, start_date={start_date}, end_date={end_date}")
        self.logger.info(f"当前存储的股票代码: {list(self.mock_data_store.keys())}")

        if code not in self.mock_data_store:
            self.logger.warning(f"未找到股票代码的模拟数据: {code}")
            return None

        data = self.mock_data_store[code].copy()

        # 转换日期格式进行过滤
        try:
            if start_date and end_date:
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')

                # 确保date列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    data['date'] = pd.to_datetime(data['date'])

                # 过滤日期范围
                mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
                data = data[mask].copy()

            if len(data) == 0:
                self.logger.warning(f"在指定日期范围内未找到数据: {code} ({start_date} - {end_date})")
                return None

            # 转换为买点分析器期望的格式
            rows = []
            for _, row in data.iterrows():
                # 格式: [code, name, date, level, open, close, high, low, volume, turnover_rate, price_change, price_range, industry]
                formatted_row = [
                    row['code'],
                    row['name'],
                    row['date'].strftime('%Y%m%d') if pd.api.types.is_datetime64_any_dtype(row['date']) else str(row['date']),
                    level or '日线',
                    float(row['open']),
                    float(row['close']),
                    float(row['high']),
                    float(row['low']),
                    int(row['volume']),
                    0.05,  # turnover_rate
                    0.0,   # price_change
                    0.0,   # price_range
                    row.get('industry', '测试行业')
                ]
                rows.append(formatted_row)

            self.logger.debug(f"返回股票信息: {code}, {len(rows)} 行")
            return rows

        except Exception as e:
            self.logger.error(f"获取股票信息失败: {code}, {e}")
            return None


class MockDataAccessManager:
    """模拟数据管理器"""
    
    def __init__(self):
        """初始化模拟数据管理器"""
        self.mock_interface = MockDataInterface()
        self.original_interface = None
        
    def setup_mock_data(self, test_data_dict: Dict[str, pd.DataFrame]):
        """
        设置模拟数据
        
        Args:
            test_data_dict: 测试数据字典 {stock_code: data}
        """
        for stock_code, data in test_data_dict.items():
            self.mock_interface.register_mock_data(stock_code, data)
    
    def inject_mock_interface(self):
        """注入模拟数据接口到依赖注入系统"""
        try:
            from utils.dependency_injection import get_container
from db.sql_manager import SQLManager, QueryType
            
            container = get_container()
            
            # 保存原始接口（如果存在）
            try:
                self.original_interface = container.resolve(DataAccessInterface)
            except:
                self.original_interface = None
            
            # 注册模拟接口
            container.register(DataAccessInterface, self.mock_interface)
            
            logger.info("成功注入模拟数据接口")
            return True
            
        except Exception as e:
            logger.error(f"注入模拟数据接口失败: {e}")
            return False
    
    def restore_original_interface(self):
        """恢复原始数据接口"""
        try:
            if self.original_interface:
                from utils.dependency_injection import get_container
from db.sql_manager import SQLManager, QueryType
                container = get_container()
                container.register(DataAccessInterface, self.original_interface)
                logger.info("恢复原始数据接口")
            
        except Exception as e:
            logger.error(f"恢复原始数据接口失败: {e}")
    
    def cleanup(self):
        """清理模拟数据"""
        self.mock_interface.clear_mock_data()
        self.restore_original_interface()


# 全局模拟数据管理器实例
_mock_data_manager = None

def get_mock_data_manager() -> MockDataAccessManager:
    """获取模拟数据管理器单例"""
    global _mock_data_manager
    if _mock_data_manager is None:
        _mock_data_manager = MockDataAccessManager()
    return _mock_data_manager

def setup_test_data(test_data_dict: Dict[str, pd.DataFrame]) -> MockDataAccessManager:
    """
    设置测试数据的便捷函数
    
    Args:
        test_data_dict: 测试数据字典
        
    Returns:
        MockDataAccessManager: 模拟数据管理器
    """
    manager = get_mock_data_manager()
    manager.setup_mock_data(test_data_dict)
    manager.inject_mock_interface()
    return manager

def cleanup_test_data():
    """清理测试数据的便捷函数"""
    global _mock_data_manager
    if _mock_data_manager:
        _mock_data_manager.cleanup()
        _mock_data_manager = None
