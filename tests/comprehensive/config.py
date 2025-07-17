#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合测试配置管理器

提供测试环境配置管理，包括数据库连接、性能阈值、测试数据等配置。
严格遵循系统架构规范，使用统一的配置管理机制。
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from utils.logger import getLogger
from config.database_config_manager import DatabaseConfigManager

logger = getLogger(__name__)


@dataclass
class PerformanceThresholds:
    """性能阈值配置"""
    single_stock_query: float = 2.0      # 单股查询时间限制（秒）
    batch_100_stocks: float = 30.0       # 100股批量处理时间限制（秒）
    batch_1000_stocks: float = 300.0     # 1000股批量处理时间限制（秒）
    full_market_scan: float = 1200.0     # 全市场扫描时间限制（秒）
    max_memory_usage: float = 8.0        # 最大内存使用限制（GB）
    max_connections: int = 50             # 最大数据库连接数
    cache_hit_rate_threshold: float = 0.8 # 缓存命中率阈值


@dataclass
class TestDataConfig:
    """测试数据配置"""
    sample_stock_codes: List[str]
    test_date_range_start: str
    test_date_range_end: str
    market_conditions: List[str]
    test_indicators: List[str]
    test_strategies: List[str]


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_real_time_monitoring: bool = True
    cpu_usage_threshold: float = 80.0
    memory_usage_threshold: float = 6.0
    query_timeout_threshold: float = 10.0
    log_level: str = "INFO"


@dataclass
class ReportingConfig:
    """报告配置"""
    output_directory: str = "test_results"
    generate_html: bool = True
    generate_json: bool = True
    generate_charts: bool = True
    enable_email_notifications: bool = False


class TestEnvironmentConfig:
    """测试环境配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化测试环境配置
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.config_file = config_file or self._get_default_config_file()
        self.config = self._load_config()
        self.db_config_manager = DatabaseConfigManager()
        
        logger.info(f"测试环境配置已加载: {self.config_file}")
    
    def _get_default_config_file(self) -> str:
        """获取默认配置文件路径"""
        current_dir = Path(__file__).parent
        return str(current_dir / "test_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {self.config_file}")
                return config
            else:
                logger.warning(f"配置文件不存在，使用默认配置: {self.config_file}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'database': {
                'host': os.getenv('CLICKHOUSE_HOST', 'localhost'),
                'port': int(os.getenv('CLICKHOUSE_PORT', 9000)),
                'database': os.getenv('CLICKHOUSE_DB', 'stock'),
                'user': os.getenv('CLICKHOUSE_USER', 'default'),
                'password': os.getenv('CLICKHOUSE_PASSWORD', ''),
                'timeout': 30,
                'pool_size': 10,
                'max_overflow': 20
            },
            'performance_thresholds': {
                'single_stock_query': 2.0,
                'batch_100_stocks': 30.0,
                'batch_1000_stocks': 300.0,
                'full_market_scan': 1200.0,
                'max_memory_usage': 8.0,
                'max_connections': 50,
                'cache_hit_rate_threshold': 0.8
            },
            'test_data': {
                'sample_stock_codes': ['000001', '000002', '600000', '600036', '000858'],
                'test_date_range_start': '2023-01-01',
                'test_date_range_end': '2024-12-31',
                'market_conditions': ['bull_market', 'bear_market', 'sideways_market'],
                'test_indicators': [
                    'MA', 'MACD', 'RSI', 'KDJ', 'BOLL', 
                    'DMI', 'TRIX', 'AROON', 'ROC', 'MOMENTUM'
                ],
                'test_strategies': [
                    'dual_ma_strategy', 'institutional_strategy', 'rebound_strategy'
                ]
            },
            'monitoring': {
                'enable_real_time_monitoring': True,
                'cpu_usage_threshold': 80.0,
                'memory_usage_threshold': 6.0,
                'query_timeout_threshold': 10.0,
                'log_level': 'INFO'
            },
            'reporting': {
                'output_directory': 'test_results',
                'generate_html': True,
                'generate_json': True,
                'generate_charts': True,
                'enable_email_notifications': False
            }
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return self.config.get('database', {})
    
    def get_performance_thresholds(self) -> PerformanceThresholds:
        """获取性能阈值配置"""
        thresholds_config = self.config.get('performance_thresholds', {})
        return PerformanceThresholds(**thresholds_config)
    
    def get_test_data_config(self) -> TestDataConfig:
        """获取测试数据配置"""
        test_data_config = self.config.get('test_data', {})
        return TestDataConfig(
            sample_stock_codes=test_data_config.get('sample_stock_codes', []),
            test_date_range_start=test_data_config.get('test_date_range_start', '2023-01-01'),
            test_date_range_end=test_data_config.get('test_date_range_end', '2024-12-31'),
            market_conditions=test_data_config.get('market_conditions', []),
            test_indicators=test_data_config.get('test_indicators', []),
            test_strategies=test_data_config.get('test_strategies', [])
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        monitoring_config = self.config.get('monitoring', {})
        return MonitoringConfig(**monitoring_config)
    
    def get_reporting_config(self) -> ReportingConfig:
        """获取报告配置"""
        reporting_config = self.config.get('reporting', {})
        return ReportingConfig(**reporting_config)
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证数据库配置
            db_config = self.get_database_config()
            required_db_fields = ['host', 'port', 'database']
            for field in required_db_fields:
                if field not in db_config:
                    logger.error(f"数据库配置缺少必要字段: {field}")
                    return False
            
            # 验证性能阈值
            thresholds = self.get_performance_thresholds()
            if thresholds.single_stock_query <= 0:
                logger.error("单股查询时间阈值必须大于0")
                return False
            
            # 验证测试数据配置
            test_data = self.get_test_data_config()
            if not test_data.sample_stock_codes:
                logger.error("测试股票代码列表不能为空")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """保存配置到文件"""
        try:
            output_file = config_file or self.config_file
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {output_file}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            """深度更新字典"""
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        logger.info("配置已更新")


# 全局配置实例
_test_config = None

def get_test_config() -> TestEnvironmentConfig:
    """获取测试配置实例"""
    global _test_config
    if _test_config is None:
        _test_config = TestEnvironmentConfig()
    return _test_config


def initialize_test_environment() -> bool:
    """初始化测试环境"""
    try:
        config = get_test_config()
        
        # 验证配置
        if not config.validate_config():
            logger.error("测试环境配置验证失败")
            return False
        
        # 创建输出目录
        reporting_config = config.get_reporting_config()
        output_dir = Path(reporting_config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("测试环境初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"测试环境初始化失败: {e}")
        return False