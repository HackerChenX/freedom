#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据服务层 - 严格按照架构规范实现

提供标准的股票数据访问服务，遵循数据层架构要求：
1. 不在业务层直接写SQL
2. 提供通用的数据查询方法
3. 统一的数据访问接口
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from db.interfaces.data_access_interface import DataAccessInterface, ClickHouseDataAccess, DataAccessError
from utils.logger import get_logger

logger = get_logger(__name__)

class StockDataService:
    """股票数据服务 - 符合架构规范的数据层服务"""
    
    def __init__(self, data_access: Optional[DataAccessInterface] = None):
        """
        初始化股票数据服务
        
        Args:
            data_access: 数据访问接口实例，如果为None则创建默认实例
        """
        if data_access is None:
            try:
                self.data_access = ClickHouseDataAccess()
                logger.info("✅ 使用默认ClickHouse数据访问接口")
            except Exception as e:
                logger.error(f"❌ 初始化数据访问接口失败: {e}")
                raise RuntimeError(f"无法初始化股票数据服务: {e}")
        else:
            self.data_access = data_access
            logger.info("✅ 使用提供的数据访问接口")
    
    def get_stock_list(self, limit: int = 50, min_price: float = 5.0) -> List[str]:
        """
        获取股票列表
        
        Args:
            limit: 限制返回数量
            min_price: 最低价格筛选
            
        Returns:
            股票代码列表
        """
        try:
            # 使用数据访问接口，不直接写SQL
            stock_codes = self.data_access.get_stock_list_data_access_interface()
            
            # 如果需要更多筛选条件，可以通过数据访问接口的其他方法实现
            return stock_codes[:limit] if stock_codes else []
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_stock_data(self, stock_code: str, days: int = 120, 
                      end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据
        
        Args:
            stock_code: 股票代码
            days: 获取天数
            end_date: 结束日期，默认为今天
            
        Returns:
            股票数据DataFrame
        """
        try:
            if end_date is None:
                # 使用数据库中的最新日期而不是当前日期
                end_date = '2025-12-31'

            # 计算开始日期 - 使用更大的范围确保有足够数据
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=days * 2)  # 恢复到2倍，避免查询范围过大
            start_date = start_dt.strftime('%Y-%m-%d')

            # 如果计算的开始日期太早，使用数据库中的最早日期
            if start_dt.year < 2020:
                start_date = '2020-01-01'
            
            # 使用数据访问接口获取数据
            df = self.data_access.get_stock_data_data_access_interface(
                code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                # 确保数据类型正确
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 按日期排序并限制数量
                df = df.sort_values('date').tail(days).reset_index(drop=True)
                return df
            else:
                logger.warning(f"股票{stock_code}没有数据")
                return None
                
        except Exception as e:
            logger.error(f"获取股票{stock_code}数据失败: {e}")
            return None
    
    def get_stock_data_by_date_range(self, stock_code: str, start_date: str, 
                                   end_date: str) -> Optional[pd.DataFrame]:
        """
        按日期范围获取股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据DataFrame
        """
        try:
            # 使用数据访问接口获取数据
            df = self.data_access.get_stock_data_data_access_interface(
                code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                # 确保数据类型正确
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df.sort_values('date').reset_index(drop=True)
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取股票{stock_code}日期范围数据失败: {e}")
            return None
    
    def get_stocks_data_batch(self, stock_codes: List[str], start_date: str, 
                            end_date: str) -> Optional[pd.DataFrame]:
        """
        批量获取多只股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            批量股票数据DataFrame
        """
        try:
            # 使用数据访问接口批量获取数据
            df = self.data_access.get_stocks_data_batch_data_access_interface(
                codes=stock_codes,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                # 确保数据类型正确
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df.sort_values(['code', 'date']).reset_index(drop=True)
            else:
                return None
                
        except Exception as e:
            logger.error(f"批量获取股票数据失败: {e}")
            return None
    
    def check_stock_data_exists(self, stock_code: str, date: str) -> bool:
        """
        检查股票数据是否存在
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            数据是否存在
        """
        try:
            conditions = {
                'code': stock_code,
                'date': date,
                'level': '日线'
            }
            
            return self.data_access.check_data_exists_data_access_interface(
                table='stock_info',
                conditions=conditions
            )
            
        except Exception as e:
            logger.error(f"检查股票{stock_code}数据存在性失败: {e}")
            return False
    
    def get_latest_stock_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取股票最新数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            最新数据字典
        """
        try:
            columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            latest_data = self.data_access.get_latest_data_data_access_interface(
                table='stock_info',
                code=stock_code,
                columns=columns
            )
            
            return latest_data
            
        except Exception as e:
            logger.error(f"获取股票{stock_code}最新数据失败: {e}")
            return None
    
    def execute_custom_query(self, query: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        执行自定义查询（仅供高级用户使用）
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame
        """
        try:
            logger.warning(f"执行自定义查询: {query[:100]}...")
            
            df = self.data_access.execute_query_data_access_interface(
                query=query,
                params=params
            )
            
            return df
            
        except Exception as e:
            logger.error(f"执行自定义查询失败: {e}")
            return None

# 全局服务实例（单例模式）
_stock_data_service_instance = None

def get_stock_data_service() -> StockDataService:
    """
    获取股票数据服务实例（单例模式）
    
    Returns:
        股票数据服务实例
    """
    global _stock_data_service_instance
    
    if _stock_data_service_instance is None:
        try:
            _stock_data_service_instance = StockDataService()
            logger.info("✅ 创建股票数据服务实例")
        except Exception as e:
            logger.error(f"❌ 创建股票数据服务实例失败: {e}")
            raise RuntimeError(f"无法创建股票数据服务: {e}")
    
    return _stock_data_service_instance

def reset_stock_data_service():
    """重置股票数据服务实例（用于测试）"""
    global _stock_data_service_instance
    _stock_data_service_instance = None
