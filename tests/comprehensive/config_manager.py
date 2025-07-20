#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置管理器

管理选股测试系统的配置参数，支持从YAML文件加载和验证
"""

import os
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from utils.logger import getLogger

logger = getLogger(__name__)


@dataclass
class DateRange:
    """日期范围"""
    start_date: str
    end_date: str


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 9000
    user: str = "default"
    password: str = ""
    database: str = "stock"
    max_connections: int = 50
    query_timeout: int = 30
    compression: bool = True


@dataclass
class CacheConfig:
    """缓存配置"""
    memory_capacity: int = 1000
    memory_ttl: int = 3600
    disk_cache_dir: str = "cache"
    disk_ttl: int = 86400
    enable_disk_cache: bool = True


@dataclass
class StockSelectionConfig:
    """选股配置"""
    min_volume: float = 1000000
    min_price: float = 1.0
    max_price: float = 1000.0
    exclude_st: bool = True
    exclude_suspended: bool = True
    sample_stock_codes: List[str] = field(default_factory=list)


@dataclass
class VerificationConfig:
    """验证配置"""
    confidence_threshold: float = 0.7
    pattern_match_threshold: float = 0.8
    require_at_least_one_stock: bool = True
    max_verification_attempts: int = 1000


@dataclass
class PerformanceConfig:
    """性能配置"""
    timeout_seconds: int = 300
    performance_threshold: float = 0.8
    max_workers: int = 20
    batch_size: int = 1000
    memory_limit_mb: int = 8192
    enable_early_stopping: bool = True


@dataclass
class ReportingConfig:
    """报告配置"""
    output_formats: List[str] = field(default_factory=lambda: ["json", "html", "csv"])
    report_level: str = "comprehensive"
    include_diagnostics: bool = True
    export_selected_stocks: bool = True
    export_verification_details: bool = True
    output_dir: str = "test_reports"


@dataclass
class TestConfig:
    """测试配置"""
    date_range: DateRange
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    stock_selection: StockSelectionConfig = field(default_factory=StockSelectionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    indicators_to_test: Optional[List[str]] = None
    patterns_to_test: Optional[List[str]] = None


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，None表示使用默认配置
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        logger.info(f"配置管理器初始化完成，配置文件: {config_path or '默认配置'}")
    
    def _load_config(self) -> TestConfig:
        """
        加载配置
        
        Returns:
            TestConfig: 测试配置
        """
        if not self.config_path or not os.path.exists(self.config_path):
            logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            return self._parse_config(config_data)
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._create_default_config()
    
    def _create_default_config(self) -> TestConfig:
        """
        创建默认配置
        
        Returns:
            TestConfig: 默认测试配置
        """
        # 默认日期范围：当前年份
        current_year = datetime.now().year
        start_date = f"{current_year}0101"
        end_date = f"{current_year}1231"
        
        return TestConfig(
            date_range=DateRange(start_date=start_date, end_date=end_date)
        )
    
    def _parse_config(self, config_data: Dict[str, Any]) -> TestConfig:
        """
        解析配置数据
        
        Args:
            config_data: 配置数据
            
        Returns:
            TestConfig: 测试配置
        """
        test_config = config_data.get('test_config', {})
        
        # 解析日期范围
        date_range_data = test_config.get('date_range', {})
        date_range = DateRange(
            start_date=date_range_data.get('start_date', '20240101'),
            end_date=date_range_data.get('end_date', '20241231')
        )
        
        # 解析数据库配置
        db_config_data = test_config.get('database', {})
        db_config = DatabaseConfig(
            host=db_config_data.get('host', 'localhost'),
            port=db_config_data.get('port', 9000),
            user=db_config_data.get('user', 'default'),
            password=db_config_data.get('password', ''),
            database=db_config_data.get('database', 'stock'),
            max_connections=db_config_data.get('max_connections', 50),
            query_timeout=db_config_data.get('query_timeout', 30),
            compression=db_config_data.get('compression', True)
        )
        
        # 解析缓存配置
        cache_config_data = test_config.get('cache', {})
        cache_config = CacheConfig(
            memory_capacity=cache_config_data.get('memory_capacity', 1000),
            memory_ttl=cache_config_data.get('memory_ttl', 3600),
            disk_cache_dir=cache_config_data.get('disk_cache_dir', 'cache'),
            disk_ttl=cache_config_data.get('disk_ttl', 86400),
            enable_disk_cache=cache_config_data.get('enable_disk_cache', True)
        )
        
        # 解析选股配置
        stock_config_data = test_config.get('stock_selection', {})
        stock_config = StockSelectionConfig(
            min_volume=stock_config_data.get('min_volume', 1000000),
            min_price=stock_config_data.get('min_price', 1.0),
            max_price=stock_config_data.get('max_price', 1000.0),
            exclude_st=stock_config_data.get('exclude_st', True),
            exclude_suspended=stock_config_data.get('exclude_suspended', True),
            sample_stock_codes=stock_config_data.get('sample_stock_codes', [])
        )
        
        # 解析验证配置
        verification_data = test_config.get('verification', {})
        verification_config = VerificationConfig(
            confidence_threshold=verification_data.get('confidence_threshold', 0.7),
            pattern_match_threshold=verification_data.get('pattern_match_threshold', 0.8),
            require_at_least_one_stock=verification_data.get('require_at_least_one_stock', True),
            max_verification_attempts=verification_data.get('max_verification_attempts', 1000)
        )
        
        # 解析性能配置
        performance_data = test_config.get('performance', {})
        performance_config = PerformanceConfig(
            timeout_seconds=performance_data.get('timeout_seconds', 300),
            performance_threshold=performance_data.get('performance_threshold', 0.8),
            max_workers=performance_data.get('max_workers', 20),
            batch_size=performance_data.get('batch_size', 1000),
            memory_limit_mb=performance_data.get('memory_limit_mb', 8192),
            enable_early_stopping=performance_data.get('enable_early_stopping', True)
        )
        
        # 解析报告配置
        reporting_data = test_config.get('reporting', {})
        reporting_config = ReportingConfig(
            output_formats=reporting_data.get('output_formats', ["json", "html", "csv"]),
            report_level=reporting_data.get('report_level', "comprehensive"),
            include_diagnostics=reporting_data.get('include_diagnostics', True),
            export_selected_stocks=reporting_data.get('export_selected_stocks', True),
            export_verification_details=reporting_data.get('export_verification_details', True),
            output_dir=reporting_data.get('output_dir', "test_reports")
        )
        
        # 解析指标和形态过滤
        indicators_to_test = test_config.get('indicators_to_test')
        patterns_to_test = test_config.get('patterns_to_test')
        
        return TestConfig(
            date_range=date_range,
            database=db_config,
            cache=cache_config,
            stock_selection=stock_config,
            verification=verification_config,
            performance=performance_config,
            reporting=reporting_config,
            indicators_to_test=indicators_to_test,
            patterns_to_test=patterns_to_test
        )
    
    def get_config(self) -> TestConfig:
        """
        获取配置
        
        Returns:
            TestConfig: 测试配置
        """
        return self.config
    
    def validate_config(self) -> List[str]:
        """
        验证配置
        
        Returns:
            List[str]: 验证错误列表，空列表表示验证通过
        """
        errors = []
        
        # 验证日期范围
        try:
            start_date = datetime.strptime(self.config.date_range.start_date, '%Y%m%d')
            end_date = datetime.strptime(self.config.date_range.end_date, '%Y%m%d')
            
            if start_date > end_date:
                errors.append(f"开始日期 {self.config.date_range.start_date} 晚于结束日期 {self.config.date_range.end_date}")
                
        except ValueError:
            errors.append(f"日期格式错误: {self.config.date_range.start_date} 或 {self.config.date_range.end_date}，应为YYYYMMDD格式")
        
        # 验证性能配置
        if self.config.performance.timeout_seconds <= 0:
            errors.append(f"超时时间必须大于0: {self.config.performance.timeout_seconds}")
        
        if self.config.performance.max_workers <= 0:
            errors.append(f"最大工作线程数必须大于0: {self.config.performance.max_workers}")
        
        if self.config.performance.batch_size <= 0:
            errors.append(f"批处理大小必须大于0: {self.config.performance.batch_size}")
        
        # 验证验证配置
        if not 0 <= self.config.verification.confidence_threshold <= 1:
            errors.append(f"置信度阈值必须在0-1之间: {self.config.verification.confidence_threshold}")
        
        if not 0 <= self.config.verification.pattern_match_threshold <= 1:
            errors.append(f"形态匹配阈值必须在0-1之间: {self.config.verification.pattern_match_threshold}")
        
        return errors
    
    def save_config(self, config_path: str) -> bool:
        """
        保存配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 转换为字典
            config_dict = {
                'test_config': {
                    'date_range': {
                        'start_date': self.config.date_range.start_date,
                        'end_date': self.config.date_range.end_date
                    },
                    'database': {
                        'host': self.config.database.host,
                        'port': self.config.database.port,
                        'user': self.config.database.user,
                        'password': self.config.database.password,
                        'database': self.config.database.database,
                        'max_connections': self.config.database.max_connections,
                        'query_timeout': self.config.database.query_timeout,
                        'compression': self.config.database.compression
                    },
                    'cache': {
                        'memory_capacity': self.config.cache.memory_capacity,
                        'memory_ttl': self.config.cache.memory_ttl,
                        'disk_cache_dir': self.config.cache.disk_cache_dir,
                        'disk_ttl': self.config.cache.disk_ttl,
                        'enable_disk_cache': self.config.cache.enable_disk_cache
                    },
                    'stock_selection': {
                        'min_volume': self.config.stock_selection.min_volume,
                        'min_price': self.config.stock_selection.min_price,
                        'max_price': self.config.stock_selection.max_price,
                        'exclude_st': self.config.stock_selection.exclude_st,
                        'exclude_suspended': self.config.stock_selection.exclude_suspended,
                        'sample_stock_codes': self.config.stock_selection.sample_stock_codes
                    },
                    'verification': {
                        'confidence_threshold': self.config.verification.confidence_threshold,
                        'pattern_match_threshold': self.config.verification.pattern_match_threshold,
                        'require_at_least_one_stock': self.config.verification.require_at_least_one_stock,
                        'max_verification_attempts': self.config.verification.max_verification_attempts
                    },
                    'performance': {
                        'timeout_seconds': self.config.performance.timeout_seconds,
                        'performance_threshold': self.config.performance.performance_threshold,
                        'max_workers': self.config.performance.max_workers,
                        'batch_size': self.config.performance.batch_size,
                        'memory_limit_mb': self.config.performance.memory_limit_mb,
                        'enable_early_stopping': self.config.performance.enable_early_stopping
                    },
                    'reporting': {
                        'output_formats': self.config.reporting.output_formats,
                        'report_level': self.config.reporting.report_level,
                        'include_diagnostics': self.config.reporting.include_diagnostics,
                        'export_selected_stocks': self.config.reporting.export_selected_stocks,
                        'export_verification_details': self.config.reporting.export_verification_details,
                        'output_dir': self.config.reporting.output_dir
                    }
                }
            }
            
            # 添加指标和形态过滤（如果有）
            if self.config.indicators_to_test:
                config_dict['test_config']['indicators_to_test'] = self.config.indicators_to_test
            
            if self.config.patterns_to_test:
                config_dict['test_config']['patterns_to_test'] = self.config.patterns_to_test
            
            # 保存到文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"配置已保存到: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_updates: 配置更新
        """
        # 更新日期范围
        if 'date_range' in config_updates:
            date_range = config_updates['date_range']
            if isinstance(date_range, dict):
                if 'start_date' in date_range:
                    self.config.date_range.start_date = date_range['start_date']
                if 'end_date' in date_range:
                    self.config.date_range.end_date = date_range['end_date']
        
        # 更新数据库配置
        if 'database' in config_updates:
            db_config = config_updates['database']
            if isinstance(db_config, dict):
                for key, value in db_config.items():
                    if hasattr(self.config.database, key):
                        setattr(self.config.database, key, value)
        
        # 更新缓存配置
        if 'cache' in config_updates:
            cache_config = config_updates['cache']
            if isinstance(cache_config, dict):
                for key, value in cache_config.items():
                    if hasattr(self.config.cache, key):
                        setattr(self.config.cache, key, value)
        
        # 更新选股配置
        if 'stock_selection' in config_updates:
            stock_config = config_updates['stock_selection']
            if isinstance(stock_config, dict):
                for key, value in stock_config.items():
                    if hasattr(self.config.stock_selection, key):
                        setattr(self.config.stock_selection, key, value)
        
        # 更新验证配置
        if 'verification' in config_updates:
            verification_config = config_updates['verification']
            if isinstance(verification_config, dict):
                for key, value in verification_config.items():
                    if hasattr(self.config.verification, key):
                        setattr(self.config.verification, key, value)
        
        # 更新性能配置
        if 'performance' in config_updates:
            performance_config = config_updates['performance']
            if isinstance(performance_config, dict):
                for key, value in performance_config.items():
                    if hasattr(self.config.performance, key):
                        setattr(self.config.performance, key, value)
        
        # 更新报告配置
        if 'reporting' in config_updates:
            reporting_config = config_updates['reporting']
            if isinstance(reporting_config, dict):
                for key, value in reporting_config.items():
                    if hasattr(self.config.reporting, key):
                        setattr(self.config.reporting, key, value)
        
        # 更新指标和形态过滤
        if 'indicators_to_test' in config_updates:
            self.config.indicators_to_test = config_updates['indicators_to_test']
        
        if 'patterns_to_test' in config_updates:
            self.config.patterns_to_test = config_updates['patterns_to_test']
        
        logger.info("配置已更新")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要
        
        Returns:
            Dict[str, Any]: 配置摘要
        """
        return {
            'date_range': {
                'start_date': self.config.date_range.start_date,
                'end_date': self.config.date_range.end_date
            },
            'performance': {
                'timeout_seconds': self.config.performance.timeout_seconds,
                'max_workers': self.config.performance.max_workers,
                'batch_size': self.config.performance.batch_size
            },
            'verification': {
                'confidence_threshold': self.config.verification.confidence_threshold,
                'pattern_match_threshold': self.config.verification.pattern_match_threshold
            },
            'stock_selection': {
                'min_volume': self.config.stock_selection.min_volume,
                'min_price': self.config.stock_selection.min_price,
                'exclude_st': self.config.stock_selection.exclude_st
            },
            'indicators_filter': self.config.indicators_to_test is not None,
            'patterns_filter': self.config.patterns_to_test is not None
        }


# 全局配置管理器实例
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_test_config(config_path: Optional[str] = None) -> TestConfig:
    """
    获取测试配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        TestConfig: 测试配置
    """
    return get_config_manager(config_path).get_config()


def main():
    """测试配置管理器"""
    print("测试配置管理器...")
    
    # 创建默认配置
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print("\nfrom config.config import get_config\n默认配置:")
    print(f"日期范围: {config.date_range.start_date} - {config.date_range.end_date}")
    print(f"超时时间: {config.performance.timeout_seconds}秒")
    print(f"最大工作线程: {config.performance.max_workers}")
    print(f"批处理大小: {config.performance.batch_size}")
    
    # 验证配置
    errors = config_manager.validate_config()
    if errors:
        print("\n配置验证错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n配置验证通过")
    
    # 更新配置
    config_updates = {
        'date_range': {
            'start_date': '20240101',
            'end_date': '20240630'
        },
        'performance': {
            'timeout_seconds': 600,
            'max_workers': 30
        }
    }
    
    config_manager.update_config(config_updates)
    config = config_manager.get_config()
    
    print("\n更新后的配置:")
    print(f"日期范围: {config.date_range.start_date} - {config.date_range.end_date}")
    print(f"超时时间: {config.performance.timeout_seconds}秒")
    print(f"最大工作线程: {config.performance.max_workers}")
    
    # 保存配置
    config_path = "test_config.yaml"
    if config_manager.save_config(config_path):
        print(f"\n配置已保存到: {config_path}")
    
    # 加载保存的配置
    loaded_config_manager = ConfigManager(config_path)
    loaded_config = loaded_config_manager.get_config()
    
    print("\n加载的配置:")
    print(f"日期范围: {loaded_config.date_range.start_date} - {loaded_config.date_range.end_date}")
    print(f"超时时间: {loaded_config.performance.timeout_seconds}秒")
    
    # 清理测试文件
    import os
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"\n已删除测试配置文件: {config_path}")


if __name__ == "__main__":
    main()