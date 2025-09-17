"""
数据库配置管理器 - 兼容版本
基于统一数据库配置实现，提供向后兼容的API
严格遵循L2存储访问层单一配置源原则
"""

from config.unified_database_config import get_unified_database_config
from typing import Dict, Any, Optional


class DatabaseConfigManager:
    """数据库配置管理器 - 兼容类"""

    def __init__(self):
        """初始化数据库配置管理器"""
        self.unified_config = get_unified_database_config()

    def get_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return self.unified_config.get_clickhouse_config()

    def get_connection_config(self) -> Dict[str, Any]:
        """获取连接配置"""
        return self.unified_config.get_connection_config()

    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return self.get_config()

    def get_clickhouse_config(self) -> Dict[str, Any]:
        """获取ClickHouse配置"""
        return self.get_config()


# 向后兼容的函数
def get_clickhouse_connection_config() -> Dict[str, Any]:
    """获取ClickHouse连接配置的便捷函数"""
    return get_unified_database_config().get_connection_config()


def get_database_config() -> Dict[str, Any]:
    """获取数据库配置的便捷函数"""
    return get_unified_database_config().get_clickhouse_config()


# 向后兼容的实例
database_config_manager = DatabaseConfigManager()
