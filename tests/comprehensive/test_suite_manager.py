#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票选股策略系统测试套件管理器

负责管理和组织不同类型的测试套件，包括数据验证、选股功能、性能基准、架构合规性等。
遵循L5业务应用层规范，协调各个测试模块的执行。
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import importlib
import inspect

from utils.logger import get_logger
from .config import get_test_config
from .logging_config import get_test_logger
from .test_infrastructure import ComprehensiveTestEngine, TestPriority

logger = get_test_logger('test_suite_manager')


@dataclass
class TestSuiteInfo:
    """测试套件信息"""
    name: str
    description: str
    priority: TestPriority
    module_path: str
    class_name: str
    dependencies: List[str]
    estimated_time: float  # 预估执行时间（秒）
    category: str  # 测试类别


class TestSuiteManager:
    """测试套件管理器"""
    
    def __init__(self, engine: ComprehensiveTestEngine):
        """
        初始化测试套件管理器
        
        Args:
            engine: 综合测试引擎
        """
        self.engine = engine
        self.config = get_test_config()
        self.suite_registry = {}
        self.loaded_suites = {}
        
        # 注册内置测试套件
        self._register_builtin_suites()
        
        logger.info("测试套件管理器初始化完成")
    
    def _register_builtin_suites(self) -> None:
        """注册内置测试套件"""
        
        # 1. 真实数据验证测试套件
        self.register_suite(TestSuiteInfo(
            name="data_validation",
            description="真实数据验证测试套件",
            priority=TestPriority.CRITICAL,
            module_path="tests.comprehensive.validators",
            class_name="RealDataValidator",
            dependencies=[],
            estimated_time=120.0,  # 2分钟
            category="基础验证"
        ))
        
        # 2. 选股功能测试套件
        self.register_suite(TestSuiteInfo(
            name="stock_selection",
            description="选股功能测试套件",
            priority=TestPriority.HIGH,
            module_path="tests.comprehensive.stock_selection_tester",
            class_name="StockSelectionTester",
            dependencies=["data_validation"],
            estimated_time=600.0,  # 10分钟
            category="功能测试"
        ))
        
        # 3. 性能基准测试套件
        self.register_suite(TestSuiteInfo(
            name="performance_benchmark",
            description="性能基准测试套件",
            priority=TestPriority.HIGH,
            module_path="tests.comprehensive.performance_tester",
            class_name="PerformanceBenchmarkTester",
            dependencies=["data_validation"],
            estimated_time=1800.0,  # 30分钟
            category="性能测试"
        ))
        
        # 4. 架构合规性测试套件
        self.register_suite(TestSuiteInfo(
            name="architecture_compliance",
            description="架构合规性检查测试套件",
            priority=TestPriority.MEDIUM,
            module_path="tests.comprehensive.architecture_checker",
            class_name="ArchitectureComplianceChecker",
            dependencies=[],
            estimated_time=300.0,  # 5分钟
            category="架构验证"
        ))
        
        # 5. 指标和形态测试套件
        self.register_suite(TestSuiteInfo(
            name="indicator_pattern",
            description="指标和形态测试套件",
            priority=TestPriority.HIGH,
            module_path="tests.comprehensive.indicator_pattern_tester",
            class_name="IndicatorPatternTester",
            dependencies=["data_validation"],
            estimated_time=3600.0,  # 60分钟
            category="指标测试"
        ))
        
        # 6. 集成测试套件
        self.register_suite(TestSuiteInfo(
            name="integration",
            description="集成测试套件",
            priority=TestPriority.MEDIUM,
            module_path="tests.comprehensive.integration_tester",
            class_name="IntegrationTester",
            dependencies=["data_validation", "stock_selection"],
            estimated_time=900.0,  # 15分钟
            category="集成测试"
        ))
        
        # 7. 监控和可观测性测试套件
        self.register_suite(TestSuiteInfo(
            name="monitoring",
            description="监控和可观测性测试套件",
            priority=TestPriority.LOW,
            module_path="tests.comprehensive.monitoring_tester",
            class_name="MonitoringTester",
            dependencies=[],
            estimated_time=180.0,  # 3分钟
            category="监控测试"
        ))
        
        # 8. 安全性测试套件
        self.register_suite(TestSuiteInfo(
            name="security",
            description="数据安全和完整性测试套件",
            priority=TestPriority.MEDIUM,
            module_path="tests.comprehensive.security_tester",
            class_name="SecurityTester",
            dependencies=["data_validation"],
            estimated_time=240.0,  # 4分钟
            category="安全测试"
        ))
    
    def register_suite(self, suite_info: TestSuiteInfo) -> None:
        """
        注册测试套件
        
        Args:
            suite_info: 测试套件信息
        """
        self.suite_registry[suite_info.name] = suite_info
        logger.debug(f"注册测试套件: {suite_info.name} - {suite_info.description}")
    
    def load_test_suite(self, suite_name: str) -> Optional[Dict[str, Callable]]:
        """
        加载测试套件
        
        Args:
            suite_name: 测试套件名称
            
        Returns:
            Optional[Dict[str, Callable]]: 测试套件字典，如果加载失败则返回None
        """
        if suite_name in self.loaded_suites:
            return self.loaded_suites[suite_name]
        
        if suite_name not in self.suite_registry:
            logger.error(f"未注册的测试套件: {suite_name}")
            return None
        
        suite_info = self.suite_registry[suite_name]
        
        try:
            # 动态导入模块
            module = importlib.import_module(suite_info.module_path)
            tester_class = getattr(module, suite_info.class_name)
            
            # 创建测试器实例
            tester_instance = tester_class()
            
            # 获取测试方法
            test_methods = {}
            for method_name, method in inspect.getmembers(tester_instance, inspect.ismethod):
                if method_name.startswith('test_'):
                    test_methods[method_name] = method
            
            if not test_methods:
                logger.warning(f"测试套件 {suite_name} 中没有找到测试方法")
                return None
            
            self.loaded_suites[suite_name] = test_methods
            logger.info(f"成功加载测试套件: {suite_name}, 包含 {len(test_methods)} 个测试")
            
            return test_methods
            
        except Exception as e:
            logger.error(f"加载测试套件失败: {suite_name}, 错误: {e}")
            return None
    
    def load_all_suites(self) -> Dict[str, Dict[str, Callable]]:
        """
        加载所有注册的测试套件
        
        Returns:
            Dict[str, Dict[str, Callable]]: 所有测试套件字典
        """
        all_suites = {}
        
        for suite_name in self.suite_registry:
            suite = self.load_test_suite(suite_name)
            if suite:
                all_suites[suite_name] = suite
        
        logger.info(f"成功加载 {len(all_suites)} 个测试套件")
        return all_suites
    
    def get_execution_order(self) -> List[str]:
        """
        根据依赖关系和优先级获取测试套件执行顺序
        
        Returns:
            List[str]: 测试套件执行顺序
        """
        # 拓扑排序解决依赖关系
        def topological_sort(suites_info: Dict[str, TestSuiteInfo]) -> List[str]:
            visited = set()
            temp_visited = set()
            result = []
            
            def visit(name: str):
                if name in temp_visited:
                    raise ValueError(f"检测到循环依赖: {name}")
                if name in visited:
                    return
                
                temp_visited.add(name)
                
                # 先处理依赖
                suite_info = suites_info.get(name)
                if suite_info:
                    for dep in suite_info.dependencies:
                        if dep in suites_info:
                            visit(dep)
                
                temp_visited.remove(name)
                visited.add(name)
                result.append(name)
            
            for suite_name in suites_info:
                if suite_name not in visited:
                    visit(suite_name)
            
            return result
        
        try:
            # 按依赖关系排序
            dependency_order = topological_sort(self.suite_registry)
            
            # 按优先级进一步排序
            priority_values = {
                TestPriority.CRITICAL: 0,
                TestPriority.HIGH: 1,
                TestPriority.MEDIUM: 2,
                TestPriority.LOW: 3
            }
            
            sorted_suites = sorted(
                dependency_order,
                key=lambda name: priority_values.get(
                    self.suite_registry[name].priority,
                    999
                )
            )
            
            logger.info(f"测试套件执行顺序: {' -> '.join(sorted_suites)}")
            return sorted_suites
            
        except Exception as e:
            logger.error(f"计算执行顺序失败: {e}")
            # 返回简单的按优先级排序
            return sorted(
                self.suite_registry.keys(),
                key=lambda name: priority_values.get(
                    self.suite_registry[name].priority,
                    999
                )
            )
    
    def estimate_total_time(self, suite_names: Optional[List[str]] = None) -> float:
        """
        估算总执行时间
        
        Args:
            suite_names: 要执行的测试套件名称列表，如果为None则估算所有套件
            
        Returns:
            float: 预估总执行时间（秒）
        """
        if suite_names is None:
            suite_names = list(self.suite_registry.keys())
        
        total_time = 0.0
        for suite_name in suite_names:
            if suite_name in self.suite_registry:
                total_time += self.suite_registry[suite_name].estimated_time
        
        return total_time
    
    def get_suite_info(self, suite_name: str) -> Optional[TestSuiteInfo]:
        """
        获取测试套件信息
        
        Args:
            suite_name: 测试套件名称
            
        Returns:
            Optional[TestSuiteInfo]: 测试套件信息，如果不存在则返回None
        """
        return self.suite_registry.get(suite_name)
    
    def list_suites(self, category: Optional[str] = None, 
                   priority: Optional[TestPriority] = None) -> List[TestSuiteInfo]:
        """
        列出测试套件
        
        Args:
            category: 按类别筛选
            priority: 按优先级筛选
            
        Returns:
            List[TestSuiteInfo]: 测试套件信息列表
        """
        suites = list(self.suite_registry.values())
        
        if category:
            suites = [s for s in suites if s.category == category]
        
        if priority:
            suites = [s for s in suites if s.priority == priority]
        
        return suites
    
    def run_suites(self, suite_names: Optional[List[str]] = None, 
                  parallel: bool = False) -> Dict[str, Any]:
        """
        运行测试套件
        
        Args:
            suite_names: 要运行的测试套件名称列表，如果为None则运行所有套件
            parallel: 是否并行执行（注意依赖关系）
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        if suite_names is None:
            suite_names = self.get_execution_order()
        else:
            # 重新排序以满足依赖关系
            all_ordered = self.get_execution_order()
            suite_names = [name for name in all_ordered if name in suite_names]
        
        # 加载测试套件
        for suite_name in suite_names:
            test_suite = self.load_test_suite(suite_name)
            if test_suite:
                self.engine.register_test_suite(suite_name, test_suite)
            else:
                logger.warning(f"跳过无法加载的测试套件: {suite_name}")
        
        # 执行测试
        logger.info(f"开始执行 {len(suite_names)} 个测试套件")
        estimated_time = self.estimate_total_time(suite_names)
        logger.info(f"预估总执行时间: {estimated_time/60:.1f} 分钟")
        
        return self.engine.run_all_tests()
    
    def run_by_category(self, category: str) -> Dict[str, Any]:
        """
        按类别运行测试套件
        
        Args:
            category: 测试类别
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        suites = self.list_suites(category=category)
        suite_names = [s.name for s in suites]
        
        logger.info(f"运行类别 '{category}' 的测试套件: {suite_names}")
        return self.run_suites(suite_names)
    
    def run_by_priority(self, priority: TestPriority) -> Dict[str, Any]:
        """
        按优先级运行测试套件
        
        Args:
            priority: 测试优先级
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        suites = self.list_suites(priority=priority)
        suite_names = [s.name for s in suites]
        
        logger.info(f"运行优先级 '{priority.value}' 的测试套件: {suite_names}")
        return self.run_suites(suite_names)
    
    def print_suite_summary(self) -> None:
        """打印测试套件摘要"""
        print("\n" + "=" * 80)
        print("股票选股策略系统综合测试套件摘要")
        print("=" * 80)
        
        categories = {}
        for suite_info in self.suite_registry.values():
            if suite_info.category not in categories:
                categories[suite_info.category] = []
            categories[suite_info.category].append(suite_info)
        
        total_estimated_time = 0.0
        
        for category, suites in categories.items():
            print(f"\n📂 {category}")
            print("-" * 40)
            
            for suite in sorted(suites, key=lambda s: s.priority.value):
                priority_icon = {
                    TestPriority.CRITICAL: "🔥",
                    TestPriority.HIGH: "⭐",
                    TestPriority.MEDIUM: "📊",
                    TestPriority.LOW: "🔧"
                }
                
                icon = priority_icon.get(suite.priority, "📋")
                deps = f" (依赖: {', '.join(suite.dependencies)})" if suite.dependencies else ""
                
                print(f"  {icon} {suite.name}: {suite.description}")
                print(f"     优先级: {suite.priority.value} | 预估时间: {suite.estimated_time/60:.1f}分钟{deps}")
                
                total_estimated_time += suite.estimated_time
        
        print(f"\n📊 总体统计:")
        print(f"   测试套件数量: {len(self.suite_registry)}")
        print(f"   预估总执行时间: {total_estimated_time/60:.1f} 分钟")
        print(f"   测试类别数量: {len(categories)}")
        print("=" * 80)


def create_test_suite_manager(engine: ComprehensiveTestEngine) -> TestSuiteManager:
    """
    创建测试套件管理器实例
    
    Args:
        engine: 综合测试引擎
        
    Returns:
        TestSuiteManager: 测试套件管理器实例
    """
    return TestSuiteManager(engine)


if __name__ == "__main__":
    # 测试套件管理器功能演示
    from .test_infrastructure import initialize_test_infrastructure
    
    # 初始化测试基础设施
    engine = initialize_test_infrastructure()
    
    # 创建测试套件管理器
    manager = create_test_suite_manager(engine)
    
    # 打印测试套件摘要
    manager.print_suite_summary()
    
    # 清理资源
    engine.cleanup() 