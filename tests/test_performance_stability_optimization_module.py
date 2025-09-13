"""
性能优化和稳定性提升模块测试
测试性能优化器、稳定性增强器和主控制器的功能
"""

import sys
import os
import unittest
import time
import threading
from datetime import datetime, timedelta
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPerformanceStabilityOptimizationModule(unittest.TestCase):
    """性能优化和稳定性提升模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        logger.info("=" * 60)
        logger.info(f"开始测试: {self._testMethodName}")
        logger.info("=" * 60)
        self.start_time = time.time()
    
    def tearDown(self):
        """测试后清理"""
        execution_time = time.time() - self.start_time
        logger.info(f"测试 {self._testMethodName} 完成，耗时: {execution_time:.2f}秒")
        logger.info("-" * 60)
    
    def test_01_performance_stability_optimizer(self):
        """测试性能稳定性优化器"""
        logger.info("测试性能稳定性优化器...")
        
        try:
            from strategy.performance_stability_optimizer import (
                PerformanceStabilityOptimizer,
                PerformanceOptimizationConfig,
                get_performance_optimizer
            )
            
            # 测试配置创建
            config = PerformanceOptimizationConfig(
                max_memory_usage_mb=1024,
                max_workers=4,
                performance_threshold_seconds=1.0
            )
            self.assertIsNotNone(config)
            logger.info("✓ 性能优化配置创建成功")
            
            # 测试优化器创建
            optimizer = PerformanceStabilityOptimizer(config)
            self.assertIsNotNone(optimizer)
            logger.info("✓ 性能优化器创建成功")
            
            # 测试启动优化系统
            start_result = optimizer.start_optimization_system()
            self.assertEqual(start_result["status"], "started")
            logger.info("✓ 优化系统启动成功")
            
            # 测试内存优化
            memory_result = optimizer.memory_optimizer.optimize_memory_usage()
            self.assertTrue(memory_result.success)
            logger.info("✓ 内存优化执行成功")
            
            # 测试并发优化
            concurrency_result = optimizer.concurrency_optimizer.optimize_concurrency()
            self.assertTrue(concurrency_result.success)
            logger.info("✓ 并发优化执行成功")
            
            # 测试获取优化报告
            report = optimizer.get_optimization_report()
            # 检查报告内容（可能是no_optimizations状态）
            self.assertTrue("status" in report or "total_optimizations" in report)
            logger.info("✓ 优化报告生成成功")
            
            # 测试停止优化系统
            stop_result = optimizer.stop_optimization_system()
            self.assertEqual(stop_result["status"], "stopped")
            logger.info("✓ 优化系统停止成功")
            
            logger.info("性能稳定性优化器测试通过 ✓")
            
        except Exception as e:
            logger.error(f"性能稳定性优化器测试失败: {e}")
            raise
    
    def test_02_system_stability_enhancer(self):
        """测试系统稳定性增强器"""
        logger.info("测试系统稳定性增强器...")
        
        try:
            from strategy.system_stability_enhancer import (
                SystemStabilityEnhancer,
                StabilityConfig,
                CircuitBreaker,
                get_stability_enhancer
            )
            
            # 测试配置创建
            config = StabilityConfig(
                max_retry_attempts=2,
                circuit_breaker_threshold=3,
                health_check_interval=10
            )
            self.assertIsNotNone(config)
            logger.info("✓ 稳定性配置创建成功")
            
            # 测试增强器创建
            enhancer = SystemStabilityEnhancer(config)
            self.assertIsNotNone(enhancer)
            logger.info("✓ 稳定性增强器创建成功")
            
            # 测试启动稳定性系统
            start_result = enhancer.start_stability_system()
            self.assertEqual(start_result["status"], "started")
            logger.info("✓ 稳定性系统启动成功")
            
            # 测试熔断器
            circuit_breaker = enhancer.get_circuit_breaker("test_operation")
            self.assertIsNotNone(circuit_breaker)
            logger.info("✓ 熔断器创建成功")
            
            # 测试健康检查
            health_status = enhancer.health_checker.get_health_status()
            self.assertIn("overall_status", health_status)
            logger.info("✓ 健康检查执行成功")
            
            # 测试稳定性报告
            stability_report = enhancer.get_stability_report()
            self.assertIn("stability_score", stability_report)
            logger.info("✓ 稳定性报告生成成功")
            
            # 测试停止稳定性系统
            stop_result = enhancer.stop_stability_system()
            self.assertEqual(stop_result["status"], "stopped")
            logger.info("✓ 稳定性系统停止成功")
            
            logger.info("系统稳定性增强器测试通过 ✓")
            
        except Exception as e:
            logger.error(f"系统稳定性增强器测试失败: {e}")
            raise
    
    def test_03_performance_stability_controller(self):
        """测试性能稳定性主控制器"""
        logger.info("测试性能稳定性主控制器...")
        
        try:
            from strategy.performance_stability_controller import (
                PerformanceStabilityController,
                PerformanceStabilityReport,
                get_performance_stability_controller
            )
            from strategy.performance_stability_optimizer import PerformanceOptimizationConfig
            from strategy.system_stability_enhancer import StabilityConfig
            
            # 测试控制器创建
            perf_config = PerformanceOptimizationConfig(max_workers=2)
            stability_config = StabilityConfig(health_check_interval=5)
            
            controller = PerformanceStabilityController(perf_config, stability_config)
            self.assertIsNotNone(controller)
            logger.info("✓ 主控制器创建成功")
            
            # 测试启动系统
            start_result = controller.start_system()
            self.assertEqual(start_result["status"], "started")
            logger.info("✓ 系统启动成功")
            
            # 等待系统稳定
            time.sleep(2)
            
            # 测试生成综合报告
            report = controller.generate_comprehensive_report()
            self.assertIsInstance(report, PerformanceStabilityReport)
            self.assertGreaterEqual(report.overall_score, 0)
            logger.info(f"✓ 综合报告生成成功，总体评分: {report.overall_score:.1f}")
            
            # 测试获取系统状态
            status = controller.get_system_status()
            self.assertEqual(status["status"], "running")
            logger.info("✓ 系统状态获取成功")
            
            # 测试停止系统
            stop_result = controller.stop_system()
            self.assertEqual(stop_result["status"], "stopped")
            logger.info("✓ 系统停止成功")
            
            logger.info("性能稳定性主控制器测试通过 ✓")
            
        except Exception as e:
            logger.error(f"性能稳定性主控制器测试失败: {e}")
            raise
    
    def test_04_circuit_breaker_functionality(self):
        """测试熔断器功能"""
        logger.info("测试熔断器功能...")
        
        try:
            from strategy.system_stability_enhancer import CircuitBreaker, StabilityConfig
            
            # 创建熔断器
            config = StabilityConfig(circuit_breaker_threshold=2)
            circuit_breaker = CircuitBreaker("test_circuit", config)
            
            # 测试正常调用
            with circuit_breaker.call():
                pass
            logger.info("✓ 熔断器正常调用成功")
            
            # 测试失败调用
            failure_count = 0
            for i in range(3):
                try:
                    with circuit_breaker.call():
                        raise Exception("模拟失败")
                except Exception:
                    failure_count += 1
            
            self.assertEqual(failure_count, 3)
            logger.info("✓ 熔断器失败处理成功")
            
            # 检查熔断器状态
            state = circuit_breaker.get_state()
            self.assertEqual(state["state"], "open")
            logger.info("✓ 熔断器状态检查成功")
            
            logger.info("熔断器功能测试通过 ✓")
            
        except Exception as e:
            logger.error(f"熔断器功能测试失败: {e}")
            raise
    
    def test_05_retry_mechanism(self):
        """测试重试机制"""
        logger.info("测试重试机制...")
        
        try:
            from strategy.system_stability_enhancer import RetryManager, StabilityConfig
            
            # 创建重试管理器
            config = StabilityConfig(max_retry_attempts=3, retry_delay_base=0.1)
            retry_manager = RetryManager(config)
            
            # 测试成功重试
            attempt_count = 0
            def test_operation():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 2:
                    raise Exception("模拟失败")
                return "成功"

            result = retry_manager.retry_operation("test_operation", test_operation)
            self.assertEqual(attempt_count, 2)
            self.assertEqual(result, "成功")
            logger.info("✓ 重试机制成功测试通过")
            
            # 测试重试统计
            stats = retry_manager.get_stats()
            self.assertIn("test_operation", stats)
            logger.info("✓ 重试统计获取成功")
            
            logger.info("重试机制测试通过 ✓")
            
        except Exception as e:
            logger.error(f"重试机制测试失败: {e}")
            raise
    
    def test_06_health_checker(self):
        """测试健康检查器"""
        logger.info("测试健康检查器...")
        
        try:
            from strategy.system_stability_enhancer import HealthChecker, StabilityConfig
            
            # 创建健康检查器
            config = StabilityConfig(health_check_interval=1)
            health_checker = HealthChecker(config)
            
            # 注册健康检查
            def test_health_check():
                return True
            
            health_checker.register_health_check("test_check", test_health_check)
            logger.info("✓ 健康检查注册成功")
            
            # 启动监控
            health_checker.start_monitoring()
            time.sleep(2)  # 等待检查执行
            
            # 获取健康状态
            status = health_checker.get_health_status()
            self.assertIn("overall_status", status)
            logger.info("✓ 健康状态获取成功")
            
            # 停止监控
            health_checker.stop_monitoring()
            logger.info("✓ 健康监控停止成功")
            
            logger.info("健康检查器测试通过 ✓")
            
        except Exception as e:
            logger.error(f"健康检查器测试失败: {e}")
            raise
    
    def test_07_memory_optimizer(self):
        """测试内存优化器"""
        logger.info("测试内存优化器...")
        
        try:
            from strategy.performance_stability_optimizer import (
                MemoryOptimizer, 
                PerformanceOptimizationConfig
            )
            
            # 创建内存优化器
            config = PerformanceOptimizationConfig()
            memory_optimizer = MemoryOptimizer(config)
            
            # 执行内存优化
            result = memory_optimizer.optimize_memory_usage()
            self.assertTrue(result.success)
            self.assertEqual(result.optimization_type, "memory_optimization")
            logger.info("✓ 内存优化执行成功")
            
            # 检查优化结果
            self.assertIn("memory_mb", result.before_metrics)
            self.assertIn("memory_mb", result.after_metrics)
            logger.info("✓ 内存优化结果检查成功")
            
            logger.info("内存优化器测试通过 ✓")
            
        except Exception as e:
            logger.error(f"内存优化器测试失败: {e}")
            raise
    
    def test_08_concurrency_optimizer(self):
        """测试并发优化器"""
        logger.info("测试并发优化器...")
        
        try:
            from strategy.performance_stability_optimizer import (
                ConcurrencyOptimizer,
                PerformanceOptimizationConfig
            )
            
            # 创建并发优化器
            config = PerformanceOptimizationConfig(thread_pool_size=4, process_pool_size=2)
            concurrency_optimizer = ConcurrencyOptimizer(config)
            
            # 执行并发优化
            result = concurrency_optimizer.optimize_concurrency()
            self.assertTrue(result.success)
            self.assertEqual(result.optimization_type, "concurrency_optimization")
            logger.info("✓ 并发优化执行成功")
            
            # 测试线程池上下文
            with concurrency_optimizer.get_thread_pool() as thread_pool:
                self.assertIsNotNone(thread_pool)
            logger.info("✓ 线程池上下文测试成功")
            
            # 清理资源
            concurrency_optimizer.cleanup()
            logger.info("✓ 并发优化器清理成功")
            
            logger.info("并发优化器测试通过 ✓")
            
        except Exception as e:
            logger.error(f"并发优化器测试失败: {e}")
            raise
    
    def test_09_integration_workflow(self):
        """测试集成工作流"""
        logger.info("测试集成工作流...")
        
        try:
            from strategy.performance_stability_controller import get_performance_stability_controller
            
            # 获取全局控制器
            controller = get_performance_stability_controller()
            self.assertIsNotNone(controller)
            logger.info("✓ 全局控制器获取成功")
            
            # 启动系统
            start_result = controller.start_system()
            self.assertEqual(start_result["status"], "started")
            logger.info("✓ 系统启动成功")
            
            # 等待系统稳定
            time.sleep(3)
            
            # 运行全面优化
            optimization_result = controller.run_comprehensive_optimization()
            self.assertIn("optimization_time", optimization_result)
            logger.info("✓ 全面优化执行成功")
            
            # 检查改善效果
            improvement = optimization_result.get("improvement", {})
            self.assertIn("overall_score_improvement", improvement)
            logger.info("✓ 改善效果检查成功")
            
            # 获取历史趋势
            trends = controller.get_historical_trends(hours=1)
            if trends.get("status") != "no_recent_data":
                self.assertIn("overall_score_trend", trends)
                logger.info("✓ 历史趋势获取成功")
            else:
                logger.info("✓ 历史趋势数据不足（正常）")
            
            # 停止系统
            stop_result = controller.stop_system()
            self.assertEqual(stop_result["status"], "stopped")
            logger.info("✓ 系统停止成功")
            
            logger.info("集成工作流测试通过 ✓")
            
        except Exception as e:
            logger.error(f"集成工作流测试失败: {e}")
            raise


def main():
    """主函数"""
    print("=" * 80)
    print("性能优化和稳定性提升模块测试")
    print("=" * 80)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceStabilityOptimizationModule)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    print("\n" + "=" * 80)
    print("测试结果汇总:")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
