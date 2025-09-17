"""
缓存配置模块

定义统一缓存层的配置参数和策略。
"""

from typing import Dict, Any, List
from enum import Enum

from config.unified_config_manager import get_config


class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"                    # 最近最少使用
    LFU = "lfu"                    # 最少使用频率
    FIFO = "fifo"                  # 先进先出
    TTL = "ttl"                    # 基于时间过期


class CacheProfile(Enum):
    """缓存配置文件"""
    DEVELOPMENT = "development"     # 开发环境
    PRODUCTION = "production"      # 生产环境
    TESTING = "testing"           # 测试环境
    HIGH_PERFORMANCE = "high_performance"  # 高性能模式


def get_cache_config(profile: CacheProfile = CacheProfile.PRODUCTION) -> Dict[str, Any]:
    """
    获取缓存配置
    
    Args:
        profile: 配置文件类型
        
    Returns:
        Dict[str, Any]: 缓存配置
    """
    base_config = {
        'write_through': True,
        'read_through': True,
        'preload_enabled': True,
        'cleanup_interval': 300,  # 5分钟
        'metrics_enabled': True,
        'compression_enabled': False,
        'encryption_enabled': False
    }
    
    if profile == CacheProfile.DEVELOPMENT:
        return {
            **base_config,
            'memory': {
                'enabled': True,
                'max_size': 1000,
                'default_ttl': 1800,  # 30分钟
                'strategy': CacheStrategy.LRU.value
            },
            'disk': {
                'enabled': True,
                'cache_dir': './data/.cache/dev',
                'default_ttl': 7200,  # 2小时
                'max_size_mb': 100,
                'compression_enabled': False
            },
            'cleanup_interval': 60,  # 1分钟
            'debug_enabled': True
        }
    
    elif profile == CacheProfile.TESTING:
        return {
            **base_config,
            'memory': {
                'enabled': True,
                'max_size': 100,
                'default_ttl': 300,  # 5分钟
                'strategy': CacheStrategy.FIFO.value
            },
            'disk': {
                'enabled': False
            },
            'cleanup_interval': 30,  # 30秒
            'preload_enabled': False,
            'debug_enabled': True
        }
    
    elif profile == CacheProfile.HIGH_PERFORMANCE:
        return {
            **base_config,
            'memory': {
                'enabled': True,
                'max_size': 50000,
                'default_ttl': 7200,  # 2小时
                'strategy': CacheStrategy.LFU.value
            },
            'disk': {
                'enabled': True,
                'cache_dir': './data/.cache/prod',
                'default_ttl': 86400,  # 24小时
                'max_size_mb': 2048,  # 2GB
                'compression_enabled': True
            },
            'cleanup_interval': 600,  # 10分钟
            'compression_enabled': True,
            'parallel_loading': True,
            'batch_size': 1000
        }
    
    else:  # PRODUCTION
        return {
            **base_config,
            'memory': {
                'enabled': True,
                'max_size': 10000,
                'default_ttl': 3600,  # 1小时
                'strategy': CacheStrategy.LRU.value
            },
            'disk': {
                'enabled': True,
                'cache_dir': './data/.cache/prod',
                'default_ttl': 86400,  # 24小时
                'max_size_mb': 1024,  # 1GB
                'compression_enabled': True
            },
            'cleanup_interval': 300,  # 5分钟
            'compression_enabled': True
        }


def get_stock_cache_config() -> Dict[str, Any]:
    """获取股票数据专用缓存配置"""
    return {
        'stock_basic': {
            'ttl': 86400,  # 24小时，基础信息变化不频繁
            'levels': ['memory', 'disk'],
            'preload': True
        },
        'stock_daily': {
            'ttl': 3600,   # 1小时，日线数据
            'levels': ['memory', 'disk'],
            'preload': False
        },
        'stock_minute': {
            'ttl': 300,    # 5分钟，分钟数据
            'levels': ['memory'],
            'preload': False
        },
        'indicators': {
            'ttl': 1800,   # 30分钟，技术指标
            'levels': ['memory', 'disk'],
            'preload': False
        },
        'market_data': {
            'ttl': 600,    # 10分钟，市场数据
            'levels': ['memory'],
            'preload': True
        },
        'industry_data': {
            'ttl': 86400,  # 24小时，行业数据
            'levels': ['memory', 'disk'],
            'preload': True
        }
    }


def get_cache_key_patterns() -> Dict[str, str]:
    """获取缓存键模式"""
    return {
        'stock_basic': 'stock:basic:{code}',
        'stock_daily': 'stock:daily:{code}:{start_date}:{end_date}',
        'stock_minute': 'stock:minute:{code}:{date}',
        'indicator': 'indicator:{type}:{code}:{period}:{params_hash}',
        'market_overview': 'market:overview:{date}',
        'industry_list': 'industry:list',
        'industry_stocks': 'industry:stocks:{industry_code}',
        'strategy_result': 'strategy:{name}:{params_hash}:{date}',
        'backtest_result': 'backtest:{strategy}:{start_date}:{end_date}:{params_hash}'
    }


def get_preload_keys() -> List[str]:
    """获取需要预热的缓存键"""
    return [
        'industry:list',
        'market:overview:latest',
        'stock:basic:all'
    ]


def get_cache_cleanup_rules() -> Dict[str, Any]:
    """获取缓存清理规则"""
    return {
        'max_memory_usage_mb': 512,
        'max_disk_usage_mb': 2048,
        'cleanup_threshold': 0.8,  # 使用率达到80%时开始清理
        'cleanup_batch_size': 100,
        'preserve_patterns': [
            'stock:basic:*',      # 保留基础股票信息
            'industry:*',         # 保留行业信息
            'market:overview:*'   # 保留市场概览
        ],
        'priority_eviction_patterns': [
            'stock:minute:*',     # 优先清理分钟数据
            'indicator:*'         # 其次清理指标数据
        ]
    }


# 默认配置
DEFAULT_CACHE_CONFIG = get_cache_config(CacheProfile.PRODUCTION)
STOCK_CACHE_CONFIG = get_stock_cache_config()
CACHE_KEY_PATTERNS = get_cache_key_patterns()
PRELOAD_KEYS = get_preload_keys()
CLEANUP_RULES = get_cache_cleanup_rules() 