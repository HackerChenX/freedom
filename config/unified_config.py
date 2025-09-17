"""
统一配置管理系统

提供统一的配置管理功能，支持环境变量、配置文件和默认值的层次化配置。
解决系统中的硬编码配置问题。
"""

import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import logging
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = 'localhost'
    port: int = 9000
    database: str = 'stock_data'
    username: str = 'default'
    password: str = ''
    timeout: int = 30
    pool_size: int = 5
    
    @classmethod
    def from_env(cls, prefix: str = 'DB_') -> 'DatabaseConfig':
        """从环境变量创建配置"""
        return cls(
            host=os.getenv(f'{prefix}HOST', 'localhost'),
            port=int(os.getenv(f'{prefix}PORT', '9000')),
            database=os.getenv(f'{prefix}DATABASE', 'stock_data'),
            username=os.getenv(f'{prefix}USERNAME', 'default'),
            password=os.getenv(f'{prefix}PASSWORD', ''),
            timeout=int(os.getenv(f'{prefix}TIMEOUT', '30')),
            pool_size=int(os.getenv(f'{prefix}POOL_SIZE', '5'))
        )

@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = 'localhost'
    port: int = 6379
    database: int = 0
    password: str = ''
    timeout: int = 5
    max_connections: int = 10
    
    @classmethod
    def from_env(cls, prefix: str = 'REDIS_') -> 'RedisConfig':
        """从环境变量创建配置"""
        return cls(
            host=os.getenv(f'{prefix}HOST', 'localhost'),
            port=int(os.getenv(f'{prefix}PORT', '6379')),
            database=int(os.getenv(f'{prefix}DATABASE', '0')),
            password=os.getenv(f'{prefix}PASSWORD', ''),
            timeout=int(os.getenv(f'{prefix}TIMEOUT', '5')),
            max_connections=int(os.getenv(f'{prefix}MAX_CONNECTIONS', '10'))
        )

@dataclass
class CrawlerConfig:
    """爬虫配置"""
    # 反爬虫配置
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ''
    
    # 请求配置
    request_timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    
    # 并发配置
    max_workers: int = 5
    rate_limit: float = 1.0  # 每秒请求数
    
    # 代理配置
    proxy_enabled: bool = False
    proxy_list: list = field(default_factory=list)
    
    @classmethod
    def from_env(cls, prefix: str = 'CRAWLER_') -> 'CrawlerConfig':
        """从环境变量创建配置"""
        return cls(
            redis_host=os.getenv(f'{prefix}REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv(f'{prefix}REDIS_PORT', '6379')),
            redis_db=int(os.getenv(f'{prefix}REDIS_DB', '0')),
            redis_password=os.getenv(f'{prefix}REDIS_PASSWORD', ''),
            request_timeout=int(os.getenv(f'{prefix}REQUEST_TIMEOUT', '30')),
            retry_count=int(os.getenv(f'{prefix}RETRY_COUNT', '3')),
            retry_delay=int(os.getenv(f'{prefix}RETRY_DELAY', '1')),
            max_workers=int(os.getenv(f'{prefix}MAX_WORKERS', '5')),
            rate_limit=float(os.getenv(f'{prefix}RATE_LIMIT', '1.0')),
            proxy_enabled=os.getenv(f'{prefix}PROXY_ENABLED', 'false').lower() == 'true'
        )

@dataclass
class TestConfig:
    """测试配置"""
    # 测试数据库配置
    test_db_host: str = 'localhost'
    test_db_port: int = 9000
    test_db_database: str = 'test_stock_data'
    test_db_username: str = 'default'
    test_db_password: str = ''
    
    # 测试参数
    test_timeout: int = 60
    test_stock_codes: list = field(default_factory=lambda: ['000001', '000002', '600000'])
    test_date_range: int = 30  # 测试数据天数
    
    @classmethod
    def from_env(cls, prefix: str = 'TEST_') -> 'TestConfig':
        """从环境变量创建配置"""
        return cls(
            test_db_host=os.getenv(f'{prefix}DB_HOST', 'localhost'),
            test_db_port=int(os.getenv(f'{prefix}DB_PORT', '9000')),
            test_db_database=os.getenv(f'{prefix}DB_DATABASE', 'test_stock_data'),
            test_db_username=os.getenv(f'{prefix}DB_USERNAME', 'default'),
            test_db_password=os.getenv(f'{prefix}DB_PASSWORD', ''),
            test_timeout=int(os.getenv(f'{prefix}TIMEOUT', '60')),
            test_date_range=int(os.getenv(f'{prefix}DATE_RANGE', '30'))
        )

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_enabled: bool = True
    file_path: str = 'logs/app.log'
    file_max_size: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    console_enabled: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = 'LOG_') -> 'LoggingConfig':
        """从环境变量创建配置"""
        return cls(
            level=os.getenv(f'{prefix}LEVEL', 'INFO'),
            format=os.getenv(f'{prefix}FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            file_enabled=os.getenv(f'{prefix}FILE_ENABLED', 'true').lower() == 'true',
            file_path=os.getenv(f'{prefix}FILE_PATH', 'logs/app.log'),
            file_max_size=int(os.getenv(f'{prefix}FILE_MAX_SIZE', str(10 * 1024 * 1024))),
            file_backup_count=int(os.getenv(f'{prefix}FILE_BACKUP_COUNT', '5')),
            console_enabled=os.getenv(f'{prefix}CONSOLE_ENABLED', 'true').lower() == 'true'
        )

@dataclass
class SystemConfig:
    """系统配置"""
    # 环境配置
    environment: str = 'development'
    debug: bool = False
    
    # 性能配置
    max_workers: int = 4
    batch_size: int = 100
    cache_size: int = 1000
    
    # 路径配置
    data_dir: str = 'data'
    log_dir: str = 'logs'
    config_dir: str = 'config'
    
    @classmethod
    def from_env(cls, prefix: str = 'SYS_') -> 'SystemConfig':
        """从环境变量创建配置"""
        return cls(
            environment=os.getenv(f'{prefix}ENVIRONMENT', 'development'),
            debug=os.getenv(f'{prefix}DEBUG', 'false').lower() == 'true',
            max_workers=int(os.getenv(f'{prefix}MAX_WORKERS', '4')),
            batch_size=int(os.getenv(f'{prefix}BATCH_SIZE', '100')),
            cache_size=int(os.getenv(f'{prefix}CACHE_SIZE', '1000')),
            data_dir=os.getenv(f'{prefix}DATA_DIR', 'data'),
            log_dir=os.getenv(f'{prefix}LOG_DIR', 'logs'),
            config_dir=os.getenv(f'{prefix}CONFIG_DIR', 'config')
        )

class UnifiedConfig:
    """统一配置管理器
    
    提供统一的配置管理功能，支持环境变量、配置文件和默认值的层次化配置。
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self._config_file = config_file
        self._config_data = {}
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        # 1. 加载默认配置
        self._load_default_config()
        
        # 2. 加载配置文件
        if self._config_file and os.path.exists(self._config_file):
            self._load_config_file()
        
        # 3. 加载环境变量
        self._load_env_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.crawler = CrawlerConfig()
        self.test = TestConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
    
    def _load_config_file(self):
        """加载配置文件"""
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                if self._config_file.endswith('.json'):
                    self._config_data = json.load(f)
                elif self._config_file.endswith(('.yml', '.yaml')):
                    self._config_data = yaml.safe_load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {self._config_file}")
                    return
            
            # 更新配置对象
            self._update_config_from_dict(self._config_data)
            logger.info(f"加载配置文件: {self._config_file}")
        
        except Exception as e:
            logger.error(f"加载配置文件失败 {self._config_file}: {e}")
    
    def _load_env_config(self):
        """加载环境变量配置"""
        try:
            # 从环境变量更新配置
            self.database = DatabaseConfig.from_env()
            self.redis = RedisConfig.from_env()
            self.crawler = CrawlerConfig.from_env()
            self.test = TestConfig.from_env()
            self.logging = LoggingConfig.from_env()
            self.system = SystemConfig.from_env()
            
            logger.debug("加载环境变量配置完成")
        
        except Exception as e:
            logger.error(f"加载环境变量配置失败: {e}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def get_database_config(self, name: str = 'default') -> DatabaseConfig:
        """获取数据库配置
        
        Args:
            name: 数据库配置名称
            
        Returns:
            DatabaseConfig: 数据库配置
        """
        if name == 'test':
            return DatabaseConfig(
                host=self.test.test_db_host,
                port=self.test.test_db_port,
                database=self.test.test_db_database,
                username=self.test.test_db_username,
                password=self.test.test_db_password
            )
        return self.database
    
    def get_redis_config(self, name: str = 'default') -> RedisConfig:
        """获取Redis配置
        
        Args:
            name: Redis配置名称
            
        Returns:
            RedisConfig: Redis配置
        """
        if name == 'crawler':
            return RedisConfig(
                host=self.crawler.redis_host,
                port=self.crawler.redis_port,
                database=self.crawler.redis_db,
                password=self.crawler.redis_password
            )
        return self.redis
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    def set_config_value(self, key: str, value: Any):
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config_obj = self
        
        # 导航到父对象
        for k in keys[:-1]:
            if hasattr(config_obj, k):
                config_obj = getattr(config_obj, k)
            else:
                return
        
        # 设置值
        if hasattr(config_obj, keys[-1]):
            setattr(config_obj, keys[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'crawler': self.crawler.__dict__,
            'test': self.test.__dict__,
            'logging': self.logging.__dict__,
            'system': self.system.__dict__
        }
    
    def save_config(self, file_path: str):
        """保存配置到文件
        
        Args:
            file_path: 文件路径
        """
        try:
            config_dict = self.to_dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                elif file_path.endswith(('.yml', '.yaml')):
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                else:
                    raise ValueError(f"不支持的文件格式: {file_path}")
            
            logger.info(f"配置已保存到: {file_path}")
        
        except Exception as e:
            logger.error(f"保存配置失败 {file_path}: {e}")
            raise

# 全局配置实例
_unified_config = None

def get_unified_config(config_file: Optional[str] = None) -> UnifiedConfig:
    """获取统一配置实例
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        UnifiedConfig: 统一配置实例
    """
    global _unified_config
    if _unified_config is None:
        _unified_config = UnifiedConfig(config_file)
    return _unified_config

def reload_config(config_file: Optional[str] = None):
    """重新加载配置
    
    Args:
        config_file: 配置文件路径
    """
    global _unified_config
    _unified_config = UnifiedConfig(config_file)

# 便捷函数
def get_database_config(name: str = 'default') -> DatabaseConfig:
    """获取数据库配置"""
    return get_unified_config().get_database_config(name)

def get_redis_config(name: str = 'default') -> RedisConfig:
    """获取Redis配置"""
    return get_unified_config().get_redis_config(name)

def get_config_value(key: str, default: Any = None) -> Any:
    """获取配置值"""
    return get_unified_config().get_config_value(key, default) 