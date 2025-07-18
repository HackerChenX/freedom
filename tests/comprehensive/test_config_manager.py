#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置管理器

专门针对选股测试的配置管理，提供配置加载、验证和访问功能
"""

import os
import yaml
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import logging

from utils.logger import getLogger

logger = getLogger(__name__)


@dataclass
class DateRangeConfig:
    """日期范围配置"""
    start_date: str
    end_date: str


@dataclass
class DatabaseConfig:
    """数据库配置"""
    connection_pool_size: int = 50
    query_timeout: int = 30
    batch_size: int = 1000
    compression: bool = True


@dataclass
class TestScopeConfig:
    """测试范围配置"""
    date_range: DateRangeConfig
    stock_universe: str = "ALL"
    min_volume: int = 1000000
    min_price: float = 1.0


@dataclass
class VerificationConfig:
    """验证配置"""
    confidence_threshold: float = 0.7
    pattern_match_threshold: float = 0.8
    require_at_least_one_stock: bool = True


@dataclass
class ReportingConfig:
    """报告配置"""
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv", "html"])
    detail_level: str = "comprehensive"
    include_diagnostics: bool = True
    export_selected_stocks: bool = True
    export_verification_details: bool = True


@dataclass
class OptimizationConfig:
    """优化配置"""
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_early_stopping: bool = True
    memory_limit_mb: int = 8192
    gc_threshold: float = 0.8


@dataclass
class ExecutionConfig:
    """执行配置"""
    timeout_seconds: int = 300
    performance_threshold: float = 0.8
    max_workers: int = 20
    batch_size: int = 1000


@dataclass
class TestConfig:
    """测试配置"""
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    test_scope: TestScopeConfig = None
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    indicators_to_test: Optional[List[str]] = None
    patterns_to_test: Optional[List[str]] = None


class TestConfigManager:
    """测试配置管理器"""
    
    def __init__(self, config_path: str = "tests/comprehensive/test_config.yaml"):
        """
        初始化测试配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = None
        self.validation_errors = []
        
        # 加载配置
        self._load_config()
        
        logger.info(f"测试配置管理器初始化完成，配置文件: {config_path}")
    
    def _load_config(self) -> None:
        """加载配置"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                self._create_default_config()
                return
            
            # 加载YAML配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 解析配置
            self._parse_config(config_data)
            
            logger.info(f"配置加载成功: {self.config_path}")
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """创建默认配置"""
        # 创建默认日期范围
        date_range = DateRangeConfig(
            start_date="20240101",
            end_date="20241231"
        )
        
        # 创建默认测试范围
        test_scope = TestScopeConfig(
            date_range=date_range,
            stock_universe="ALL",
            min_volume=1000000,
            min_price=1.0
        )
        
        # 创建默认配置
        self.config = TestConfig(
            test_scope=test_scope
        )
        
        logger.info("已创建默认配置")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> None:
        """
        解析配置数据
        
        Args:
            config_data: 配置数据
        """
        test_config = config_data.get('test_config', {})
        
        # 解析执行配置
        execution_config = test_config.get('execution', {})
        execution = ExecutionConfig(
            timeout_seconds=execution_config.get('timeout_seconds', 300),
            performance_threshold=execution_config.get('performance_threshold', 0.8),
            max_workers=execution_config.get('max_workers', 20),
            batch_size=execution_config.get('batch_size', 1000)
        )
        
        # 解析数据库配置
        db_config = test_config.get('database', {}).get('clickhouse', {})
        database = DatabaseConfig(
            connection_pool_size=db_config.get('connection_pool_size', 50),
            query_timeout=db_config.get('query_timeout', 30),
            batch_size=db_config.get('batch_size', 1000),
            compression=db_config.get('compression', True)
        )
        
        # 解析测试范围配置
        scope_config = test_config.get('test_scope', {})
        date_range_config = scope_config.get('date_range', {})
        date_range = DateRangeConfig(
            start_date=date_range_config.get('start_date', '20240101'),
            end_date=date_range_config.get('end_date', '20241231')
        )
        
        test_scope = TestScopeConfig(
            date_range=date_range,
            stock_universe=scope_config.get('stock_universe', 'ALL'),
            min_volume=scope_config.get('min_volume', 1000000),
            min_price=scope_config.get('min_price', 1.0)
        )
        
        # 解析验证配置
        verification_config = test_config.get('verification', {})
        verification = VerificationConfig(
            confidence_threshold=verification_config.get('confidence_threshold', 0.7),
            pattern_match_threshold=verification_config.get('pattern_match_threshold', 0.8),
            require_at_least_one_stock=verification_config.get('require_at_least_one_stock', True)
        )
        
        # 解析报告配置
        reporting_config = test_config.get('reporting', {})
        reporting = ReportingConfig(
            output_formats=reporting_config.get('output_formats', ['json', 'csv', 'html']),
            detail_level=reporting_config.get('detail_level', 'comprehensive'),
            include_diagnostics=reporting_config.get('include_diagnostics', True),
            export_selected_stocks=reporting_config.get('export_selected_stocks', True),
            export_verification_details=reporting_config.get('export_verification_details', True)
        )
        
        # 解析优化配置
        optimization_config = test_config.get('optimization', {})
        optimization = OptimizationConfig(
            enable_caching=optimization_config.get('enable_caching', True),
            cache_ttl=optimization_config.get('cache_ttl', 3600),
            enable_early_stopping=optimization_config.get('enable_early_stopping', True),
            memory_limit_mb=optimization_config.get('memory_limit_mb', 8192),
            gc_threshold=optimization_config.get('gc_threshold', 0.8)
        )
        
        # 创建配置对象
        self.config = TestConfig(
            execution=execution,
            database=database,
            test_scope=test_scope,
            verification=verification,
            reporting=reporting,
            optimization=optimization,
            indicators_to_test=test_config.get('indicators_to_test'),
            patterns_to_test=test_config.get('patterns_to_test')
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
            List[str]: 验证错误列表
        """
        self.validation_errors = []
        
        # 验证日期范围
        self._validate_date_range()
        
        # 验证执行配置
        self._validate_execution_config()
        
        # 验证数据库配置
        self._validate_database_config()
        
        # 验证测试范围配置
        self._validate_test_scope_config()
        
        # 验证验证配置
        self._validate_verification_config()
        
        return self.validation_errors
    
    def _validate_date_range(self) -> None:
        """验证日期范围"""
        if not self.config or not self.config.test_scope or not self.config.test_scope.date_range:
            self.validation_errors.append("缺少日期范围配置")
            return
        
        date_range = self.config.test_scope.date_range
        
        # 验证日期格式
        if not self._is_valid_date_format(date_range.start_date):
            self.validation_errors.append(f"开始日期格式无效: {date_range.start_date}，应为YYYYMMDD格式")
        
        if not self._is_valid_date_format(date_range.end_date):
            self.validation_errors.append(f"结束日期格式无效: {date_range.end_date}，应为YYYYMMDD格式")
        
        # 验证日期范围
        if date_range.start_date > date_range.end_date:
            self.validation_errors.append(f"开始日期 {date_range.start_date} 晚于结束日期 {date_range.end_date}")
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """
        验证日期格式
        
        Args:
            date_str: 日期字符串
            
        Returns:
            bool: 是否有效
        """
        if not date_str or not isinstance(date_str, str):
            return False
        
        if len(date_str) != 8:
            return False
        
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            if not (1900 <= year <= 2100):
                return False
            
            if not (1 <= month <= 12):
                return False
            
            if not (1 <= day <= 31):
                return False
            
            return True
            
        except ValueError:
            return False
    
    def _validate_execution_config(self) -> None:
        """验证执行配置"""
        if not self.config or not self.config.execution:
            self.validation_errors.append("缺少执行配置")
            return
        
        execution = self.config.execution
        
        # 验证超时时间
        if execution.timeout_seconds <= 0:
            self.validation_errors.append(f"超时时间无效: {execution.timeout_seconds}，应大于0")
        
        # 验证性能阈值
        if not (0 < execution.performance_threshold <= 1):
            self.validation_errors.append(f"性能阈值无效: {execution.performance_threshold}，应在(0,1]范围内")
        
        # 验证最大工作线程数
        if execution.max_workers <= 0:
            self.validation_errors.append(f"最大工作线程数无效: {execution.max_workers}，应大于0")
        
        # 验证批处理大小
        if execution.batch_size <= 0:
            self.validation_errors.append(f"批处理大小无效: {execution.batch_size}，应大于0")
    
    def _validate_database_config(self) -> None:
        """验证数据库配置"""
        if not self.config or not self.config.database:
            self.validation_errors.append("缺少数据库配置")
            return
        
        database = self.config.database
        
        # 验证连接池大小
        if database.connection_pool_size <= 0:
            self.validation_errors.append(f"连接池大小无效: {database.connection_pool_size}，应大于0")
        
        # 验证查询超时
        if database.query_timeout <= 0:
            self.validation_errors.append(f"查询超时无效: {database.query_timeout}，应大于0")
        
        # 验证批处理大小
        if database.batch_size <= 0:
            self.validation_errors.append(f"批处理大小无效: {database.batch_size}，应大于0")
    
    def _validate_test_scope_config(self) -> None:
        """验证测试范围配置"""
        if not self.config or not self.config.test_scope:
            self.validation_errors.append("缺少测试范围配置")
            return
        
        test_scope = self.config.test_scope
        
        # 验证股票池
        if test_scope.stock_universe not in ["ALL", "INDEX", "CUSTOM"]:
            self.validation_errors.append(f"股票池无效: {test_scope.stock_universe}，应为ALL、INDEX或CUSTOM")
        
        # 验证最小成交量
        if test_scope.min_volume < 0:
            self.validation_errors.append(f"最小成交量无效: {test_scope.min_volume}，应大于等于0")
        
        # 验证最小价格
        if test_scope.min_price < 0:
            self.validation_errors.append(f"最小价格无效: {test_scope.min_price}，应大于等于0")
    
    def _validate_verification_config(self) -> None:
        """验证验证配置"""
        if not self.config or not self.config.verification:
            self.validation_errors.append("缺少验证配置")
            return
        
        verification = self.config.verification
        
        # 验证置信度阈值
        if not (0 <= verification.confidence_threshold <= 1):
            self.validation_errors.append(f"置信度阈值无效: {verification.confidence_threshold}，应在[0,1]范围内")
        
        # 验证形态匹配阈值
        if not (0 <= verification.pattern_match_threshold <= 1):
            self.validation_errors.append(f"形态匹配阈值无效: {verification.pattern_match_threshold}，应在[0,1]范围内")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        保存配置
        
        Args:
            config_path: 配置文件路径，None表示使用当前路径
        """
        if config_path is None:
            config_path = self.config_path
        
        try:
            # 创建配置目录
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # 将配置转换为字典
            config_dict = self._config_to_dict()
            
            # 保存为YAML
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump({'test_config': config_dict}, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"配置已保存到: {config_path}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        if not self.config:
            return {}
        
        # 执行配置
        execution = {
            'timeout_seconds': self.config.execution.timeout_seconds,
            'performance_threshold': self.config.execution.performance_threshold,
            'max_workers': self.config.execution.max_workers,
            'batch_size': self.config.execution.batch_size
        }
        
        # 数据库配置
        database = {
            'clickhouse': {
                'connection_pool_size': self.config.database.connection_pool_size,
                'query_timeout': self.config.database.query_timeout,
                'batch_size': self.config.database.batch_size,
                'compression': self.config.database.compression
            }
        }
        
        # 测试范围配置
        test_scope = {
            'date_range': {
                'start_date': self.config.test_scope.date_range.start_date,
                'end_date': self.config.test_scope.date_range.end_date
            },
            'stock_universe': self.config.test_scope.stock_universe,
            'min_volume': self.config.test_scope.min_volume,
            'min_price': self.config.test_scope.min_price
        }
        
        # 验证配置
        verification = {
            'confidence_threshold': self.config.verification.confidence_threshold,
            'pattern_match_threshold': self.config.verification.pattern_match_threshold,
            'require_at_least_one_stock': self.config.verification.require_at_least_one_stock
        }
        
        # 报告配置
        reporting = {
            'output_formats': self.config.reporting.output_formats,
            'detail_level': self.config.reporting.detail_level,
            'include_diagnostics': self.config.reporting.include_diagnostics,
            'export_selected_stocks': self.config.reporting.export_selected_stocks,
            'export_verification_details': self.config.reporting.export_verification_details
        }
        
        # 优化配置
        optimization = {
            'enable_caching': self.config.optimization.enable_caching,
            'cache_ttl': self.config.optimization.cache_ttl,
            'enable_early_stopping': self.config.optimization.enable_early_stopping,
            'memory_limit_mb': self.config.optimization.memory_limit_mb,
            'gc_threshold': self.config.optimization.gc_threshold
        }
        
        # 组合配置
        config_dict = {
            'execution': execution,
            'database': database,
            'test_scope': test_scope,
            'verification': verification,
            'reporting': reporting,
            'optimization': optimization
        }
        
        # 添加可选配置
        if self.config.indicators_to_test:
            config_dict['indicators_to_test'] = self.config.indicators_to_test
        
        if self.config.patterns_to_test:
            config_dict['patterns_to_test'] = self.config.patterns_to_test
        
        return config_dict
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_updates: 配置更新
        """
        if not self.config:
            self._create_default_config()
        
        # 更新执行配置
        if 'execution' in config_updates:
            execution = config_updates['execution']
            if 'timeout_seconds' in execution:
                self.config.execution.timeout_seconds = execution['timeout_seconds']
            if 'performance_threshold' in execution:
                self.config.execution.performance_threshold = execution['performance_threshold']
            if 'max_workers' in execution:
                self.config.execution.max_workers = execution['max_workers']
            if 'batch_size' in execution:
                self.config.execution.batch_size = execution['batch_size']
        
        # 更新数据库配置
        if 'database' in config_updates and 'clickhouse' in config_updates['database']:
            db = config_updates['database']['clickhouse']
            if 'connection_pool_size' in db:
                self.config.database.connection_pool_size = db['connection_pool_size']
            if 'query_timeout' in db:
                self.config.database.query_timeout = db['query_timeout']
            if 'batch_size' in db:
                self.config.database.batch_size = db['batch_size']
            if 'compression' in db:
                self.config.database.compression = db['compression']
        
        # 更新测试范围配置
        if 'test_scope' in config_updates:
            scope = config_updates['test_scope']
            if 'date_range' in scope:
                date_range = scope['date_range']
                if 'start_date' in date_range:
                    self.config.test_scope.date_range.start_date = date_range['start_date']
                if 'end_date' in date_range:
                    self.config.test_scope.date_range.end_date = date_range['end_date']
            if 'stock_universe' in scope:
                self.config.test_scope.stock_universe = scope['stock_universe']
            if 'min_volume' in scope:
                self.config.test_scope.min_volume = scope['min_volume']
            if 'min_price' in scope:
                self.config.test_scope.min_price = scope['min_price']
        
        # 更新验证配置
        if 'verification' in config_updates:
            verification = config_updates['verification']
            if 'confidence_threshold' in verification:
                self.config.verification.confidence_threshold = verification['confidence_threshold']
            if 'pattern_match_threshold' in verification:
                self.config.verification.pattern_match_threshold = verification['pattern_match_threshold']
            if 'require_at_least_one_stock' in verification:
                self.config.verification.require_at_least_one_stock = verification['require_at_least_one_stock']
        
        # 更新可选配置
        if 'indicators_to_test' in config_updates:
            self.config.indicators_to_test = config_updates['indicators_to_test']
        
        if 'patterns_to_test' in config_updates:
            self.config.patterns_to_test = config_updates['patterns_to_test']
        
        logger.info("配置已更新")


def get_config_manager(config_path: str = "tests/comprehensive/test_config.yaml") -> TestConfigManager:
    """
    获取配置管理器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        TestConfigManager: 配置管理器
    """
    return TestConfigManager(config_path)


def main():
    """测试配置管理器"""
    print("测试配置管理器...")
    
    # 创建配置管理器
    config_manager = TestConfigManager()
    
    # 获取配置
    config = config_manager.get_config()
    
    # 验证配置
    errors = config_manager.validate_config()
    if errors:
        print("配置验证错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过")
    
    # 打印配置
    print("\n配置信息:")
    print(f"超时时间: {config.execution.timeout_seconds}秒")
    print(f"性能阈值: {config.execution.performance_threshold}")
    print(f"最大工作线程数: {config.execution.max_workers}")
    print(f"批处理大小: {config.execution.batch_size}")
    print(f"日期范围: {config.test_scope.date_range.start_date} - {config.test_scope.date_range.end_date}")
    print(f"股票池: {config.test_scope.stock_universe}")
    print(f"最小成交量: {config.test_scope.min_volume}")
    print(f"最小价格: {config.test_scope.min_price}")
    
    # 更新配置
    config_updates = {
        'execution': {
            'timeout_seconds': 600,
            'max_workers': 30
        },
        'test_scope': {
            'date_range': {
                'start_date': '20240201',
                'end_date': '20240331'
            }
        }
    }
    
    config_manager.update_config(config_updates)
    
    # 保存配置
    config_manager.save_config("tests/comprehensive/test_config_updated.yaml")
    
    print("\n配置已更新并保存")


if __name__ == "__main__":
    main()