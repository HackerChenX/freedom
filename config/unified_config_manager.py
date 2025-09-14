"""
统一配置管理系统 - 标准化增强版
提供分层配置、环境变量支持、配置验证和热重载功能
支持JSON/YAML格式标准化、Schema验证、性能优化和缓存机制

版本: v2.0 (任务4标准化增强)
更新时间: 2025-09-14
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
from functools import lru_cache
import time

# 可选依赖处理
try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    # 简单的TTL缓存实现
    class TTLCache:
        def __init__(self, maxsize=100, ttl=300):
            self.maxsize = maxsize
            self.ttl = ttl
            self.data = {}
            self.timestamps = {}

        def __contains__(self, key):
            if key in self.data:
                if time.time() - self.timestamps[key] < self.ttl:
                    return True
                else:
                    del self.data[key]
                    del self.timestamps[key]
            return False

        def __getitem__(self, key):
            if key in self:
                return self.data[key]
            raise KeyError(key)

        def __setitem__(self, key, value):
            self.data[key] = value
            self.timestamps[key] = time.time()
            if len(self.data) > self.maxsize:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.data[oldest_key]
                del self.timestamps[oldest_key]

        def __len__(self):
            return len(self.data)

        def clear(self):
            self.data.clear()
            self.timestamps.clear()

try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    # 简单的验证实现
    class ValidationError(Exception):
        def __init__(self, message):
            self.message = message
            super().__init__(message)

    def validate(instance, schema):
        # 简单的类型检查
        if 'type' in schema:
            expected_type = schema['type']
            if expected_type == 'object' and not isinstance(instance, dict):
                raise ValidationError(f"Expected object, got {type(instance).__name__}")
            elif expected_type == 'array' and not isinstance(instance, list):
                raise ValidationError(f"Expected array, got {type(instance).__name__}")
            elif expected_type == 'string' and not isinstance(instance, str):
                raise ValidationError(f"Expected string, got {type(instance).__name__}")
            elif expected_type == 'number' and not isinstance(instance, (int, float)):
                raise ValidationError(f"Expected number, got {type(instance).__name__}")
        return True

# 导入项目模块
try:
    from utils.logger import get_logger
    from utils.performance_monitor import performance_monitor
    from utils.exception_handler import exception_handler
    logger = get_logger(__name__)
except ImportError:
    # 兼容性处理
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
    
    def __init__(self, config_dir: str = "config", cache_ttl: int = 300):
        self.config_dir = Path(config_dir)
        self.sources: List[ConfigSource] = []
        self.merged_config: Dict[str, Any] = {}
        self.validator = ConfigValidator()
        self._lock = threading.RLock()
        self._watchers = []

        # 标准化增强功能
        self.cache = TTLCache(maxsize=100, ttl=cache_ttl)
        self.schema_cache = TTLCache(maxsize=50, ttl=600)
        self.supported_formats = {'.json', '.yaml', '.yml'}
        self.standardization_enabled = True

        # 性能监控
        self.load_stats = {
            'total_loads': 0,
            'cache_hits': 0,
            'validation_errors': 0,
            'last_load_time': None
        }
        
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


# 标准化增强功能
@performance_monitor(threshold_seconds=2.0) if 'performance_monitor' in globals() else lambda f: f
@exception_handler(reraise=True) if 'exception_handler' in globals() else lambda f: f
def load_standardized_config(config_path: str, schema_type: Optional[str] = None,
                           validate_config: bool = True) -> Dict[str, Any]:
    """
    标准化配置加载函数

    Args:
        config_path: 配置文件路径
        schema_type: Schema类型
        validate_config: 是否验证配置

    Returns:
        Dict[str, Any]: 配置数据
    """
    manager = get_config_manager()

    # 检查缓存
    cache_key = f"std_{config_path}_{schema_type}_{validate_config}"
    if hasattr(manager, 'cache') and cache_key in manager.cache:
        manager.load_stats['cache_hits'] += 1
        logger.debug(f"从缓存加载标准化配置: {config_path}")
        return manager.cache[cache_key]

    # 加载配置文件
    full_path = manager.config_dir / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {full_path}")

    # 根据文件扩展名加载
    suffix = full_path.suffix.lower()
    if suffix == '.json':
        with open(full_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    elif suffix in {'.yaml', '.yml'}:
        with open(full_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {suffix}")

    # Schema验证
    if validate_config and schema_type:
        _validate_standardized_config(config_data, schema_type)

    # 缓存配置
    if hasattr(manager, 'cache'):
        manager.cache[cache_key] = config_data

    # 更新统计
    manager.load_stats['total_loads'] += 1
    manager.load_stats['last_load_time'] = time.time()

    logger.info(f"标准化配置加载成功: {config_path}")
    return config_data


def _validate_standardized_config(config_data: Dict[str, Any], schema_type: str):
    """验证标准化配置"""
    # 基础Schema定义
    schemas = {
        'strategy': {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "version": {"type": "string"},
                        "conditions": {"type": "array"},
                        "filters": {"type": "object"},
                        "parameters": {"type": "object"}
                    },
                    "required": ["id", "name", "conditions"]
                }
            },
            "required": ["strategy"]
        },
        'buypoints': {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "properties": {
                        "indicators": {"type": "array"},
                        "weights": {"type": "object"},
                        "thresholds": {"type": "object"}
                    },
                    "required": ["indicators"]
                }
            },
            "required": ["analysis"]
        }
    }

    if schema_type in schemas:
        try:
            validate(instance=config_data, schema=schemas[schema_type])
            logger.debug(f"标准化配置验证通过: {schema_type}")
        except ValidationError as e:
            manager = get_config_manager()
            manager.load_stats['validation_errors'] += 1
            logger.error(f"标准化配置验证失败: {e.message}")
            raise


def get_standardization_stats() -> Dict[str, Any]:
    """获取标准化统计信息"""
    manager = get_config_manager()
    stats = {
        "load_stats": getattr(manager, 'load_stats', {}),
        "cache_stats": {},
        "supported_formats": getattr(manager, 'supported_formats', set()),
        "standardization_enabled": getattr(manager, 'standardization_enabled', False)
    }

    if hasattr(manager, 'cache'):
        stats["cache_stats"] = {
            "size": len(manager.cache),
            "maxsize": manager.cache.maxsize,
            "ttl": getattr(manager.cache, 'ttl', 0)
        }

    return stats


# 导出主要类和函数
__all__ = [
    'ConfigSource',
    'ConfigValidator',
    'UnifiedConfigManager',
    'get_config_manager',
    'get_config',
    'set_config',
    'reload_config',
    'load_standardized_config',
    'get_standardization_stats'
]
