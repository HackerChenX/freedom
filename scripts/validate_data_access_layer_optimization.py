#!/usr/bin/env python3
"""
数据访问层优化验证脚本

验证任务1.4数据访问层优化功能是否正常工作
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from db.unified_data_manager import (
    get_unified_data_manager, 
    get_production_data_access_layer,
    initialize_production_data_access_layer
)

logger = get_logger(__name__)


class DataAccessLayerValidator:
    """数据访问层验证器"""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
    
    def run_all_tests(self) -> bool:
        """运行所有验证测试"""
        logger.info("🚀 开始数据访问层优化验证...")
        
        tests = [
            self.test_unified_data_manager,
            self.test_production_data_access_layer,
            self.test_health_check,
            self.test_performance_monitoring,
            self.test_cache_optimization,
            self.test_backward_compatibility
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                test_name = test.__name__
                logger.info(f"📋 运行测试: {test_name}")
                
                start_time = time.time()
                result = test()
                execution_time = time.time() - start_time
                
                if result:
                    logger.info(f"  ✅ {test_name} 通过 ({execution_time:.2f}s)")
                    passed_tests += 1
                else:
                    logger.error(f"  ❌ {test_name} 失败 ({execution_time:.2f}s)")
                
                self.test_results.append({
                    'test': test_name,
                    'passed': result,
                    'execution_time': execution_time
                })
                
            except Exception as e:
                logger.error(f"  💥 {test.__name__} 异常: {e}")
                self.errors.append(f"{test.__name__}: {str(e)}")
                self.test_results.append({
                    'test': test.__name__,
                    'passed': False,
                    'error': str(e)
                })
        
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"📊 验证完成: {passed_tests}/{total_tests} 通过 ({success_rate:.1f}%)")
        
        return success_rate >= 80.0
    
    def test_unified_data_manager(self) -> bool:
        """测试统一数据管理器"""
        try:
            # 获取统一数据管理器实例
            data_manager = get_unified_data_manager()
            
            # 检查基本属性
            assert hasattr(data_manager, 'connection_pool'), "缺少连接池属性"
            assert hasattr(data_manager, 'cache_enabled'), "缺少缓存配置属性"
            assert hasattr(data_manager, 'stats'), "缺少统计属性"
            
            # 检查统计信息结构
            required_stats = [
                'total_queries', 'cache_hits', 'cache_misses', 
                'query_errors', 'total_execution_time'
            ]
            for stat in required_stats:
                assert stat in data_manager.stats, f"缺少统计项: {stat}"
            
            logger.info("  ✓ 统一数据管理器基本功能正常")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 统一数据管理器测试失败: {e}")
            return False
    
    def test_production_data_access_layer(self) -> bool:
        """测试生产级数据访问层"""
        try:
            # 初始化生产级数据访问层
            initialize_production_data_access_layer()
            
            # 获取实例
            prod_layer = get_production_data_access_layer()
            
            # 检查是否继承自UnifiedDataManager
            from db.unified_data_manager import UnifiedDataManager
            assert isinstance(prod_layer, UnifiedDataManager), "生产级数据访问层应继承UnifiedDataManager"
            
            # 检查生产级特有方法
            assert hasattr(prod_layer, 'health_check'), "缺少健康检查方法"
            assert hasattr(prod_layer, 'get_performance_stats'), "缺少性能统计方法"
            assert hasattr(prod_layer, 'optimize_performance'), "缺少性能优化方法"
            
            logger.info("  ✓ 生产级数据访问层基本功能正常")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 生产级数据访问层测试失败: {e}")
            return False
    
    def test_health_check(self) -> bool:
        """测试健康检查功能"""
        try:
            prod_layer = get_production_data_access_layer()
            
            # 执行健康检查
            health_result = prod_layer.health_check()
            
            # 检查健康检查结果结构
            required_fields = ['timestamp', 'status', 'components', 'metrics']
            for field in required_fields:
                assert field in health_result, f"健康检查结果缺少字段: {field}"
            
            # 检查状态值
            valid_statuses = ['healthy', 'degraded', 'unhealthy', 'unknown']
            assert health_result['status'] in valid_statuses, f"无效的健康状态: {health_result['status']}"
            
            # 检查组件状态
            assert 'database' in health_result['components'], "缺少数据库组件状态"
            assert 'connection_pool' in health_result['components'], "缺少连接池组件状态"
            
            logger.info(f"  ✓ 健康检查功能正常，状态: {health_result['status']}")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 健康检查测试失败: {e}")
            return False
    
    def test_performance_monitoring(self) -> bool:
        """测试性能监控功能"""
        try:
            prod_layer = get_production_data_access_layer()
            
            # 获取性能统计
            perf_stats = prod_layer.get_performance_stats()
            
            # 检查性能统计结构
            required_metrics = [
                'total_queries', 'cache_hit_rate', 'error_rate', 'avg_query_time'
            ]
            for metric in required_metrics:
                assert metric in perf_stats, f"性能统计缺少指标: {metric}"
            
            # 检查指标类型
            assert isinstance(perf_stats['cache_hit_rate'], (int, float)), "缓存命中率应为数值"
            assert isinstance(perf_stats['error_rate'], (int, float)), "错误率应为数值"
            assert isinstance(perf_stats['avg_query_time'], (int, float)), "平均查询时间应为数值"
            
            logger.info("  ✓ 性能监控功能正常")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 性能监控测试失败: {e}")
            return False
    
    def test_cache_optimization(self) -> bool:
        """测试缓存优化功能"""
        try:
            prod_layer = get_production_data_access_layer()
            
            # 获取缓存统计
            cache_stats = prod_layer.get_cache_stats()
            
            # 检查缓存统计结构
            required_fields = [
                'total_entries', 'expired_entries', 'active_entries', 'hit_rate'
            ]
            for field in required_fields:
                assert field in cache_stats, f"缓存统计缺少字段: {field}"
            
            # 执行性能优化
            optimization_result = prod_layer.optimize_performance()
            
            # 检查优化结果结构
            assert 'actions_taken' in optimization_result, "优化结果缺少执行动作"
            assert 'recommendations' in optimization_result, "优化结果缺少建议"
            assert 'before_stats' in optimization_result, "优化结果缺少优化前统计"
            
            logger.info("  ✓ 缓存优化功能正常")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 缓存优化测试失败: {e}")
            return False
    
    def test_backward_compatibility(self) -> bool:
        """测试向后兼容性"""
        try:
            # 测试向后兼容的获取函数
            from db.unified_data_manager import (
                get_data_manager, 
                get_enhanced_data_manager,
                get_data_manager_adapter
            )
            
            # 所有函数都应该返回UnifiedDataManager实例
            dm1 = get_data_manager()
            dm2 = get_enhanced_data_manager()
            dm3 = get_data_manager_adapter()
            
            from db.unified_data_manager import UnifiedDataManager
            assert isinstance(dm1, UnifiedDataManager), "get_data_manager应返回UnifiedDataManager"
            assert isinstance(dm2, UnifiedDataManager), "get_enhanced_data_manager应返回UnifiedDataManager"
            assert isinstance(dm3, UnifiedDataManager), "get_data_manager_adapter应返回UnifiedDataManager"
            
            logger.info("  ✓ 向后兼容性正常")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 向后兼容性测试失败: {e}")
            return False
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("📋 数据访问层优化验证报告")
        report.append("=" * 60)
        
        passed_count = sum(1 for result in self.test_results if result.get('passed', False))
        total_count = len(self.test_results)
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        report.append(f"总测试数: {total_count}")
        report.append(f"通过测试: {passed_count}")
        report.append(f"失败测试: {total_count - passed_count}")
        report.append(f"成功率: {success_rate:.1f}%")
        report.append("")
        
        # 详细测试结果
        report.append("📝 详细测试结果:")
        for result in self.test_results:
            status = "✅ 通过" if result.get('passed', False) else "❌ 失败"
            time_info = f"({result.get('execution_time', 0):.2f}s)" if 'execution_time' in result else ""
            report.append(f"  {status} {result['test']} {time_info}")
            
            if 'error' in result:
                report.append(f"    错误: {result['error']}")
        
        if self.errors:
            report.append("")
            report.append("❌ 错误详情:")
            for error in self.errors:
                report.append(f"  - {error}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """主函数"""
    validator = DataAccessLayerValidator()
    
    # 运行所有测试
    success = validator.run_all_tests()
    
    # 生成并显示报告
    report = validator.generate_report()
    print("\n" + report)
    
    # 保存报告
    report_path = project_root / "docs" / "data_access_layer_optimization_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📄 验证报告已保存: {report_path}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
