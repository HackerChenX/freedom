"""
统一数据库配置管理器
L2存储访问层 - 单一配置源原则实现

严格遵循：
1. 单一配置源：只从 config/database.yaml 读取
2. 统一配置入口：只提供一个获取配置的函数
3. 废弃分散配置：不再使用其他配置文件
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedDatabaseConfig:
    """
    统一数据库配置管理器
    L2存储访问层的标准配置管理实现
    """
    
    _instance = None
    _config = None
    _config_file = "config/database.yaml"
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """
        从标准配置文件加载数据库配置
        只从 config/database.yaml 读取，严格执行单一配置源原则
        """
        try:
            # 确保路径正确
            if not os.path.isabs(self._config_file):
                config_path = Path.cwd() / self._config_file
            else:
                config_path = Path(self._config_file)

            if not config_path.exists():
                logger.error(f"数据库配置文件不存在: {config_path}")
                self._config = self._get_default_config()
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            
            logger.info(f"数据库配置已从 {self._config_file} 加载")
            
            # 验证配置完整性
            self._validate_config()
            
        except Exception as e:
            logger.error(f"加载数据库配置失败: {e}")
            self._config = self._get_default_config()
    
    def _validate_config(self) -> None:
        """验证配置完整性"""
        required_keys = ['clickhouse']
        clickhouse_required = ['host', 'port', 'database', 'user']
        
        if not self._config:
            raise ValueError("配置为空")
        
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"缺少必需的配置项: {key}")
        
        clickhouse_config = self._config['clickhouse']
        for key in clickhouse_required:
            if key not in clickhouse_config:
                raise ValueError(f"缺少必需的ClickHouse配置项: {key}")
        
        # 验证端口必须是9000（原生端口）
        if clickhouse_config['port'] != 9000:
            logger.warning(f"ClickHouse端口应为9000（原生端口），当前为: {clickhouse_config['port']}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'clickhouse': {
                'host': 'localhost',
                'port': 9000,  # 强制使用原生端口
                'database': 'stock',
                'user': 'default',
                'password': '123456',
                'timeout': 30,
                'pool': {
                    'min_size': 5,
                    'max_size': 20,
                    'max_overflow': 10,
                    'timeout': 30
                },
                'cache': {
                    'enabled': True,
                    'max_size': 1000,
                    'ttl': 3600
                },
                'query': {
                    'timeout': 30,
                    'max_rows': 1000000,
                    'max_memory_usage': 20000000000
                },
                'compression': True
            },
            'logging': {
                'level': 'INFO',
                'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
                'path': 'data/result/logs/database.log',
                'rotation': '1 day',
                'retention': '7 days',
                'compression': True
            }
        }
    
    def get_clickhouse_config(self) -> Dict[str, Any]:
        """
        获取ClickHouse配置
        L2存储访问层的标准配置获取接口
        
        Returns:
            Dict[str, Any]: ClickHouse配置字典
        """
        if not self._config:
            self._load_config()
        
        return self._config.get('clickhouse', {})
    
    def get_connection_config(self) -> Dict[str, Any]:
        """
        获取数据库连接配置
        专门为连接池提供的配置接口
        
        Returns:
            Dict[str, Any]: 连接配置字典
        """
        clickhouse_config = self.get_clickhouse_config()
        
        return {
            'host': clickhouse_config.get('host', 'localhost'),
            'port': clickhouse_config.get('port', 9000),
            'database': clickhouse_config.get('database', 'stock'),
            'user': clickhouse_config.get('user', 'default'),
            'password': clickhouse_config.get('password', '123456'),
            'timeout': clickhouse_config.get('timeout', 30),
            'compression': clickhouse_config.get('compression', True)
        }
    
    def get_pool_config(self) -> Dict[str, Any]:
        """
        获取连接池配置
        
        Returns:
            Dict[str, Any]: 连接池配置字典
        """
        clickhouse_config = self.get_clickhouse_config()
        pool_config = clickhouse_config.get('pool', {})
        
        return {
            'min_size': pool_config.get('min_size', 5),
            'max_size': pool_config.get('max_size', 20),
            'max_overflow': pool_config.get('max_overflow', 10),
            'timeout': pool_config.get('timeout', 30)
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """
        获取缓存配置
        
        Returns:
            Dict[str, Any]: 缓存配置字典
        """
        clickhouse_config = self.get_clickhouse_config()
        cache_config = clickhouse_config.get('cache', {})
        
        return {
            'enabled': cache_config.get('enabled', True),
            'max_size': cache_config.get('max_size', 1000),
            'ttl': cache_config.get('ttl', 3600)
        }
    
    def reload_config(self) -> None:
        """重新加载配置"""
        self._config = None
        self._load_config()
        logger.info("数据库配置已重新加载")


# 全局单例实例
_unified_db_config = None


def get_unified_database_config() -> UnifiedDatabaseConfig:
    """
    获取统一数据库配置管理器实例
    L2存储访问层的标准配置获取入口
    
    Returns:
        UnifiedDatabaseConfig: 配置管理器实例
    """
    global _unified_db_config
    if _unified_db_config is None:
        _unified_db_config = UnifiedDatabaseConfig()
    return _unified_db_config


def get_database_config() -> Dict[str, Any]:
    """
    获取数据库配置的便捷函数
    向后兼容接口
    
    Returns:
        Dict[str, Any]: 数据库配置字典
    """
    return get_unified_database_config().get_clickhouse_config()


def get_clickhouse_connection_config() -> Dict[str, Any]:
    """
    获取ClickHouse连接配置的便捷函数
    向后兼容接口
    
    Returns:
        Dict[str, Any]: 连接配置字典
    """
    return get_unified_database_config().get_connection_config()


# 向后兼容的别名
get_clickhouse_config = get_database_config
database_config = get_unified_database_config()
