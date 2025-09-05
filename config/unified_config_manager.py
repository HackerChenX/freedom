"""
统一配置管理系统
提供分层配置、环境变量支持、配置验证和热重载功能
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """配置源"""
    name: str
    path: str
    priority: int
    format: str  # json, yaml, env
    loaded: bool = False
    last_modified: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.validation_rules = {}
    
    def add_rule(self, key: str, validator: callable, required: bool = False):
        """添加验证规则"""
        self.validation_rules[key] = {
            'validator': validator,
            'required': required
        }
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """验证配置"""
        errors = []
        
        for key, rule in self.validation_rules.items():
            value = self._get_nested_value(config, key)
            
            if value is None:
                if rule['required']:
                    errors.append(f"必需配置项缺失: {key}")
                continue
            
            try:
                if not rule['validator'](value):
                    errors.append(f"配置项验证失败: {key} = {value}")
            except Exception as e:
                errors.append(f"配置项验证错误: {key} - {e}")
        
        return errors
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """获取嵌套配置值"""
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None


class UnifiedConfigManager:
    """
    统一配置管理器
    
    支持多配置源、分层配置、环境变量覆盖和配置验证
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.sources: List[ConfigSource] = []
        self.merged_config: Dict[str, Any] = {}
        self.validator = ConfigValidator()
        self._lock = threading.RLock()
        self._watchers = []
        
        # 默认配置
        self.default_config = {
            'system': {
                'name': 'stock-analysis-system',
                'version': '1.0.0',
                'environment': 'development'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_enabled': True,
                'console_enabled': True
            },
            'performance': {
                'monitoring_enabled': True,
                'threshold_seconds': 2.0,
                'memory_tracking': True
            },
            'database': {
                'host': 'localhost',
                'port': 8123,
                'user': 'default',
                'password': '',
                'database': 'stock_data',
                'connection_pool': {
                    'min_connections': 5,
                    'max_connections': 20,
                    'timeout': 30
                }
            },
            'indicators': {
                'cache_enabled': True,
                'cache_ttl': 300,
                'calculation_timeout': 10
            },
            'security': {
                'encrypt_sensitive': True,
                'audit_enabled': True
            }
        }
        
        # 注册默认验证规则
        self._register_default_validation_rules()
        
        # 初始化配置源
        self._initialize_config_sources()
        
        # 加载配置
        self.reload_config()
    
    def _register_default_validation_rules(self):
        """注册默认验证规则"""
        # 数据库配置验证
        self.validator.add_rule('database.host', lambda x: isinstance(x, str) and len(x) > 0, True)
        self.validator.add_rule('database.port', lambda x: isinstance(x, int) and 1 <= x <= 65535, True)
        self.validator.add_rule('database.user', lambda x: isinstance(x, str), True)
        self.validator.add_rule('database.database', lambda x: isinstance(x, str) and len(x) > 0, True)
        
        # 连接池配置验证
        self.validator.add_rule('database.connection_pool.min_connections', 
                               lambda x: isinstance(x, int) and x > 0, True)
        self.validator.add_rule('database.connection_pool.max_connections', 
                               lambda x: isinstance(x, int) and x > 0, True)
        
        # 性能配置验证
        self.validator.add_rule('performance.threshold_seconds', 
                               lambda x: isinstance(x, (int, float)) and x > 0)
        
        # 日志配置验证
        self.validator.add_rule('logging.level', 
                               lambda x: x in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    def _initialize_config_sources(self):
        """初始化配置源"""
        # 1. 默认配置（最低优先级）
        self.sources.append(ConfigSource(
            name="default",
            path="",
            priority=0,
            format="dict",
            loaded=True,
            data=self.default_config
        ))
        
        # 2. 基础配置文件
        base_config_files = [
            ("base.yaml", "yaml", 10),
            ("base.json", "json", 10),
            ("config.yaml", "yaml", 20),
            ("config.json", "json", 20)
        ]
        
        for filename, format_type, priority in base_config_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                self.sources.append(ConfigSource(
                    name=filename,
                    path=str(config_path),
                    priority=priority,
                    format=format_type
                ))
        
        # 3. 环境特定配置
        env = os.getenv('ENVIRONMENT', 'development')
        env_config_files = [
            (f"{env}.yaml", "yaml", 30),
            (f"{env}.json", "json", 30)
        ]
        
        for filename, format_type, priority in env_config_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                self.sources.append(ConfigSource(
                    name=filename,
                    path=str(config_path),
                    priority=priority,
                    format=format_type
                ))
        
        # 4. 本地配置文件（最高优先级）
        local_config_files = [
            ("local.yaml", "yaml", 40),
            ("local.json", "json", 40)
        ]
        
        for filename, format_type, priority in local_config_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                self.sources.append(ConfigSource(
                    name=filename,
                    path=str(config_path),
                    priority=priority,
                    format=format_type
                ))
        
        # 5. 环境变量（最高优先级）
        self.sources.append(ConfigSource(
            name="environment",
            path="",
            priority=50,
            format="env",
            loaded=True,
            data=self._load_env_config()
        ))
        
        # 按优先级排序
        self.sources.sort(key=lambda x: x.priority)
    
    def _load_env_config(self) -> Dict[str, Any]:
        """加载环境变量配置"""
        env_config = {}
        
        # 数据库配置
        if os.getenv('DB_HOST'):
            env_config.setdefault('database', {})['host'] = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            env_config.setdefault('database', {})['port'] = int(os.getenv('DB_PORT'))
        if os.getenv('DB_USER'):
            env_config.setdefault('database', {})['user'] = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            env_config.setdefault('database', {})['password'] = os.getenv('DB_PASSWORD')
        if os.getenv('DB_DATABASE'):
            env_config.setdefault('database', {})['database'] = os.getenv('DB_DATABASE')
        
        # 日志配置
        if os.getenv('LOG_LEVEL'):
            env_config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
        
        # 性能配置
        if os.getenv('PERFORMANCE_THRESHOLD'):
            env_config.setdefault('performance', {})['threshold_seconds'] = float(os.getenv('PERFORMANCE_THRESHOLD'))
        
        return env_config
    
    def _load_file_config(self, source: ConfigSource) -> Dict[str, Any]:
        """加载文件配置"""
        try:
            with open(source.path, 'r', encoding='utf-8') as f:
                if source.format == 'json':
                    return json.load(f)
                elif source.format == 'yaml':
                    return yaml.safe_load(f) or {}
                else:
                    return {}
        except Exception as e:
            logger.error(f"加载配置文件失败 {source.path}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def reload_config(self) -> bool:
        """重新加载配置"""
        with self._lock:
            try:
                # 加载所有配置源
                for source in self.sources:
                    if source.format in ['json', 'yaml'] and source.path:
                        source.data = self._load_file_config(source)
                        source.loaded = True
                        source.last_modified = datetime.now()
                    elif source.format == 'env':
                        source.data = self._load_env_config()
                        source.loaded = True
                        source.last_modified = datetime.now()
                
                # 合并配置
                self.merged_config = {}
                for source in self.sources:
                    if source.loaded:
                        self.merged_config = self._merge_configs(self.merged_config, source.data)
                
                # 验证配置
                validation_errors = self.validator.validate(self.merged_config)
                if validation_errors:
                    logger.warning(f"配置验证警告: {validation_errors}")
                
                logger.info("配置重新加载成功")
                return True
                
            except Exception as e:
                logger.error(f"配置重新加载失败: {e}")
                return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            keys = key.split('.')
            value = self.merged_config
            
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any, persist: bool = False) -> bool:
        """设置配置值"""
        with self._lock:
            keys = key.split('.')
            config = self.merged_config
            
            # 导航到父级
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # 设置值
            config[keys[-1]] = value
            
            # 持久化到本地配置文件
            if persist:
                return self._persist_to_local_config(key, value)
            
            return True
    
    def _persist_to_local_config(self, key: str, value: Any) -> bool:
        """持久化到本地配置文件"""
        try:
            local_config_path = self.config_dir / "local.yaml"
            
            # 读取现有本地配置
            local_config = {}
            if local_config_path.exists():
                with open(local_config_path, 'r', encoding='utf-8') as f:
                    local_config = yaml.safe_load(f) or {}
            
            # 设置新值
            keys = key.split('.')
            config = local_config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            
            # 保存到文件
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(local_config, f, default_flow_style=False, allow_unicode=True)
            
            return True
            
        except Exception as e:
            logger.error(f"持久化配置失败: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            return self.merged_config.copy()
    
    def get_sources_info(self) -> List[Dict[str, Any]]:
        """获取配置源信息"""
        return [
            {
                'name': source.name,
                'path': source.path,
                'priority': source.priority,
                'format': source.format,
                'loaded': source.loaded,
                'last_modified': source.last_modified.isoformat() if source.last_modified else None
            }
            for source in self.sources
        ]


# 全局配置管理器实例
_config_manager: Optional[UnifiedConfigManager] = None
_config_lock = threading.Lock()


def get_config_manager() -> UnifiedConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    
    if _config_manager is None:
        with _config_lock:
            if _config_manager is None:
                _config_manager = UnifiedConfigManager()
    
    return _config_manager


def get_config(key: str = None, default: Any = None) -> Any:
    """
    获取配置值
    
    Args:
        key: 配置键，支持点号分隔的嵌套访问
        default: 默认值
        
    Returns:
        配置值
    """
    manager = get_config_manager()
    if key is None:
        return manager.get_all()
    return manager.get(key, default)


def set_config(key: str, value: Any, persist: bool = False) -> bool:
    """
    设置配置值
    
    Args:
        key: 配置键
        value: 配置值
        persist: 是否持久化
        
    Returns:
        bool: 设置成功
    """
    manager = get_config_manager()
    return manager.set(key, value, persist)


def reload_config() -> bool:
    """重新加载配置"""
    manager = get_config_manager()
    return manager.reload_config()


# 导出主要类和函数
__all__ = [
    'ConfigSource',
    'ConfigValidator',
    'UnifiedConfigManager',
    'get_config_manager',
    'get_config',
    'set_config',
    'reload_config'
]
