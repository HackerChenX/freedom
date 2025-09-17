"""
连接管理器 - 兼容版本
基于标准连接池实现，提供向后兼容的API
严格遵循L2存储访问层规范
"""

import threading
from typing import Dict, List, Optional, Any, ContextManager

from db.enhanced_connection_pool import get_connection_pool
from utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    连接管理器 - 兼容实现
    基于标准连接池，提供统一的连接管理接口
    """
    
    def __init__(self):
        """初始化连接管理器"""
        self.logger = logger
        self._pool = None
        self.logger.info("连接管理器初始化完成")
    
    @property
    def pool(self):
        """获取连接池实例"""
        if self._pool is None:
            self._pool = get_connection_pool()
        return self._pool
    
    def get_connection(self):
        """获取数据库连接"""
        return self.pool.get_connection()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行查询"""
        return self.pool.query_dataframe(query, params)
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            test_query = "SELECT 1 as test"
            result = self.execute_query(test_query)
            return not result.empty
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False


# 向后兼容的全局实例
_connection_manager = None
_manager_lock = threading.Lock()


def get_connection_manager() -> ConnectionManager:
    """获取全局连接管理器实例（单例模式）"""
    global _connection_manager

    if _connection_manager is None:
        with _manager_lock:
            if _connection_manager is None:
                _connection_manager = ConnectionManager()
                logger.info("全局连接管理器已创建")

    return _connection_manager


# 向后兼容的别名
connection_manager = get_connection_manager()
