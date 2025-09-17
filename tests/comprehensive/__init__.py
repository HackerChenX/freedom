"""
股票选股策略系统综合测试模块

提供全面的测试功能，包括：
- 真实数据验证
- 选股功能测试
- 性能基准测试
- 架构合规性检查
- 监控和可观测性测试
"""

__version__ = "1.0.0"
__author__ = "Stock Analysis System"

from .stock_selection_tester import ComprehensiveStockSelectionTester as StockSelectionTester
from .performance_tester import PerformanceBenchmarkTester
from .data_consistency_validator import DataConsistencyValidator as RealDataValidator
from .architecture_checker import ArchitectureComplianceChecker
from db.sql_manager import SQLManager, QueryType

__all__ = [
    'StockSelectionTester',
    'PerformanceBenchmarkTester', 
    'RealDataValidator',
    'ArchitectureComplianceChecker'
]