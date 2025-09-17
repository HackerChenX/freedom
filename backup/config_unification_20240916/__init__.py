#!/usr/bin/env python3
"""
统一配置管理器

提供统一的配置访问接口，支持环境变量覆盖、配置验证等功能。
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config: Optional[Dict] = None
        self._config_file = Path(__file__).parent / 'unified_config.json'
    
    @lru_cache(maxsize=1)
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self._config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self._config_file}")
        
        with open(self._config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 环境变量覆盖
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        # 数据库配置
        if 'CLICKHOUSE_HOST' in os.environ:
            config['database']['host'] = os.environ['CLICKHOUSE_HOST']
        if 'CLICKHOUSE_PORT' in os.environ:
            config['database']['port'] = int(os.environ['CLICKHOUSE_PORT'])
        if 'CLICKHOUSE_USER' in os.environ:
            config['database']['user'] = os.environ['CLICKHOUSE_USER']
        if 'CLICKHOUSE_PASSWORD' in os.environ:
            config['database']['password'] = os.environ['CLICKHOUSE_PASSWORD']
        if 'CLICKHOUSE_DATABASE' in os.environ:
            config['database']['name'] = os.environ['CLICKHOUSE_DATABASE']
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套访问，如 'database.host'
            default: 默认值
        
        Returns:
            配置值
        """
        if self._config is None:
            self._config = self._load_config()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self):
        """重新加载配置"""
        self._config = None
        self._load_config.cache_clear()


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_config(key: str = None, default: Any = None) -> Any:
    """获取配置
    
    Args:
        key: 配置键，如果为None则返回整个配置
        default: 默认值
    
    Returns:
        配置值
    """
    if key is None:
        return _config_manager._load_config()
    return _config_manager.get(key, default)


def reload_config():
    """重新加载配置"""
    _config_manager.reload()
