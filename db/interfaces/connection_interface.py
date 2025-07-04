"""
连接管理接口定义

定义数据库连接管理的标准接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, ContextManager
from datetime import datetime
import pandas as pd


class IConnectionManager(ABC):
    """
    连接管理器接口
    
    定义数据库连接管理的标准接口
    """
    
    @abstractmethod
    def get_connection(self, config: Optional[Dict[str, Any]] = None) -> ContextManager:
        """
        获取数据库连接
        
        Args:
            config: 连接配置
            
        Returns:
            ContextManager: 连接上下文管理器
        """
        pass
    
    @abstractmethod
    def release_connection(self, connection_id: str) -> None:
        """
        释放数据库连接
        
        Args:
            connection_id: 连接ID
        """
        pass
    
    @abstractmethod
    def test_connection(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        测试数据库连接
        
        Args:
            config: 连接配置
            
        Returns:
            bool: 连接成功返回True
        """
        pass
    
    @abstractmethod
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        获取连接统计信息
        
        Returns:
            Dict[str, Any]: 连接统计信息
        """
        pass
    
    @abstractmethod
    def close_all_connections(self) -> None:
        """
        关闭所有连接
        """
        pass


class IConnectionPool(ABC):
    """
    连接池接口
    """
    
    @abstractmethod
    def get_connection(self) -> ContextManager:
        """
        从连接池获取连接
        
        Returns:
            ContextManager: 连接上下文管理器
        """
        pass
    
    @abstractmethod
    def return_connection(self, connection: Any) -> None:
        """
        归还连接到连接池
        
        Args:
            connection: 数据库连接对象
        """
        pass
    
    @abstractmethod
    def get_pool_size(self) -> int:
        """
        获取连接池大小
        
        Returns:
            int: 连接池大小
        """
        pass
    
    @abstractmethod
    def get_active_connections(self) -> int:
        """
        获取活跃连接数
        
        Returns:
            int: 活跃连接数
        """
        pass
    
    @abstractmethod
    def get_idle_connections(self) -> int:
        """
        获取空闲连接数
        
        Returns:
            int: 空闲连接数
        """
        pass
    
    @abstractmethod
    def resize_pool(self, new_size: int) -> None:
        """
        调整连接池大小
        
        Args:
            new_size: 新的连接池大小
        """
        pass


class IConnection(ABC):
    """
    数据库连接接口
    """
    
    @abstractmethod
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        执行SQL语句
        
        Args:
            query: SQL语句
            params: 查询参数
        """
        pass
    
    @abstractmethod
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回结果
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        pass
    
    @abstractmethod
    def query_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        pass
    
    @abstractmethod
    def is_alive(self) -> bool:
        """
        检查连接是否有效
        
        Returns:
            bool: 连接有效返回True
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        关闭连接
        """
        pass


class ITransactionManager(ABC):
    """
    事务管理器接口
    """
    
    @abstractmethod
    def begin_transaction(self) -> ContextManager:
        """
        开始事务
        
        Returns:
            ContextManager: 事务上下文管理器
        """
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """
        提交事务
        """
        pass
    
    @abstractmethod
    def rollback(self) -> None:
        """
        回滚事务
        """
        pass
    
    @abstractmethod
    def in_transaction(self) -> bool:
        """
        检查是否在事务中
        
        Returns:
            bool: 在事务中返回True
        """
        pass


class IHealthChecker(ABC):
    """
    健康检查接口
    """
    
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """
        检查数据库健康状态
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        检查是否健康
        
        Returns:
            bool: 健康返回True
        """
        pass
    
    @abstractmethod
    def get_last_check_time(self) -> datetime:
        """
        获取最后检查时间
        
        Returns:
            datetime: 最后检查时间
        """
        pass 