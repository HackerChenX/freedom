"""
统一配置管理入口
提供向后兼容的API，统一所有配置访问
基于 unified_config_manager 实现
"""

import os
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

# 导入统一配置管理器
from config.unified_config_manager import UnifiedConfigManager

logger = logging.getLogger(__name__)

# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> UnifiedConfigManager:
    """获取配置管理器实例（单例模式）"""
    global _config_manager
    if _config_manager is None:
        _config_manager = UnifiedConfigManager()
        logger.info("统一配置管理器已初始化")
    return _config_manager


def get_config(key: str = None, default: Any = None) -> Any:
    """
    获取配置值 - 兼容旧API
    
    Args:
        key: 配置键，支持点分隔符（如 'db.host'）
        default: 默认值
        
    Returns:
        配置值
    """
    manager = get_config_manager()
    
    if key is None:
        # 返回所有配置
        return manager.get_all_config()
    
    return manager.get(key, default)


def get_database_config() -> Dict[str, Any]:
    """获取数据库配置 - 兼容旧API"""
    return get_config('database', {
        'host': 'localhost',
        'port': 9000,
        'database': 'stock',
        'user': 'default',
        'password': '123456',
        'timeout': 30,
        'compression': True,
        'pool': {
            'min_size': 5,
            'max_size': 20,
            'timeout': 30
        }
    })


# 向后兼容的别名
db_config = get_database_config
config = get_config
