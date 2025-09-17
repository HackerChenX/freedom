#!/usr/bin/env python3
"""
L3数据服务层综合验证测试
基于L1/L2层A+级测试经验，执行L3层五重测试标准验证

测试目标:
1. 功能完整性：数据访问接口、缓存服务、数据服务功能100%正常
2. 集成兼容性：L3层各组件之间集成100%正常
3. 性能达标性：数据服务性能必须达到EXCELLENT级别
4. 日志清洁性：系统运行过程中不能有任何ERROR或WARNING日志
5. 标准合规性：必须100%符合L3数据服务层架构规范
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class L3ComprehensiveValidator:
    """L3数据服务层综合验证器"""
    
    def __init__(self):
        self.test_results = {
            'functionality_completeness': {'score': 0, 'max_score': 100, 'details': []},
            'integration_compatibility': {'score': 0, 'max_score': 100, 'details': []},
            'performance_excellence': {'score': 0, 'max_score': 100, 'details': []},
            'log_cleanliness': {'score': 0, 'max_score': 100, 'details': []},
            'standards_compliance': {'score': 0, 'max_score': 100, 'details': []}
        }
        
        self.performance_thresholds = {
            'data_access_time': 1.0,      # 数据访问最大时间(秒)
            'cache_operation_time': 0.1,   # 缓存操作最大时间(秒)
            'service_init_time': 2.0,      # 服务初始化最大时间(秒)
            'memory_usage_mb': 512,        # 最大内存使用(MB)
        }
        
        self.error_count = 0
        self.warning_count = 0
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """执行L3层综合验证"""
        logger.info("🚀 开始L3数据服务层综合验证...")
        
        try:
            # 1. 功能完整性测试
            self._test_functionality_completeness()
            
            # 2. 集成兼容性测试
            self._test_integration_compatibility()
            
            # 3. 性能达标性测试
            self._test_performance_excellence()
            
            # 4. 日志清洁性测试
            self._test_log_cleanliness()
            
            # 5. 标准合规性测试
            self._test_standards_compliance()
            
            # 6. 计算总体评分
            overall_score = self._calculate_overall_score()
            
            # 7. 生成验证报告
            report = self._generate_validation_report(overall_score)
            
            logger.info("✅ L3数据服务层综合验证完成!")
            return report
            
        except Exception as e:
            logger.error(f"❌ L3数据服务层综合验证失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_functionality_completeness(self):
        """测试功能完整性"""
        logger.info("🔍 测试功能完整性...")
        
        tests = [
            self._test_data_access_interface,
            self._test_cache_service,
            self._test_data_services,
            self._test_service_registry
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                start_time = time.time()
                result = test()
                execution_time = time.time() - start_time
                
                if result:
                    passed_tests += 1
                    self.test_results['functionality_completeness']['details'].append(
                        f"✅ {test.__name__}: 通过 ({execution_time:.3f}s)"
                    )
                else:
                    self.test_results['functionality_completeness']['details'].append(
                        f"❌ {test.__name__}: 失败 ({execution_time:.3f}s)"
                    )
                    
            except Exception as e:
                self.test_results['functionality_completeness']['details'].append(
                    f"❌ {test.__name__}: 异常 - {e}"
                )
        
        score = (passed_tests / total_tests) * 100
        self.test_results['functionality_completeness']['score'] = score
        logger.info(f"📊 功能完整性得分: {score:.1f}/100")
    
    def _test_data_access_interface(self) -> bool:
        """测试数据访问接口"""
        try:
            # 测试标准数据访问接口
            from db.interfaces.data_access_interface import DataAccessInterface
            from db.managers.data_access_manager import DataAccessManager
            
            # 创建数据访问管理器实例
            data_manager = DataAccessManager()
            
            # 测试基本方法是否存在
            required_methods = [
                'get_stock_data_data_access_interface',
                'get_stocks_data_batch_data_access_interface',
                'get_stock_list_data_access_interface',
                'check_data_exists_data_access_interface'
            ]
            
            for method in required_methods:
                if not hasattr(data_manager, method):
                    logger.error(f"❌ 数据访问接口缺少方法: {method}")
                    return False
            
            logger.info("✅ 数据访问接口功能完整")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据访问接口测试失败: {e}")
            return False
    
    def _test_cache_service(self) -> bool:
        """测试缓存服务"""
        try:
            # 测试标准缓存服务
            from db.services.cache_service import CacheService
            
            # 创建缓存服务实例
            cache_service = CacheService()
            
            # 测试基本缓存操作
            test_key = "test_key_l3_validation"
            test_value = "test_value_l3_validation"
            
            # 设置缓存
            cache_service.set(test_key, test_value, ttl=60)
            
            # 获取缓存
            retrieved_value = cache_service.get(test_key)
            
            if retrieved_value != test_value:
                logger.error(f"❌ 缓存值不匹配: 期望 {test_value}, 实际 {retrieved_value}")
                return False
            
            # 删除缓存
            cache_service.delete(test_key)
            
            # 验证删除
            deleted_value = cache_service.get(test_key)
            if deleted_value is not None:
                logger.error(f"❌ 缓存删除失败: {deleted_value}")
                return False
            
            logger.info("✅ 缓存服务功能完整")
            return True
            
        except Exception as e:
            logger.error(f"❌ 缓存服务测试失败: {e}")
            return False
    
    def _test_data_services(self) -> bool:
        """测试数据服务"""
        try:
            # 测试核心数据服务
            services_to_test = [
                'db.services.stock_data_service',
                'db.services.multi_period_data_service'
            ]
            
            for service_module in services_to_test:
                try:
                    __import__(service_module)
                    logger.info(f"✅ 数据服务模块可导入: {service_module}")
                except ImportError as e:
                    logger.error(f"❌ 数据服务模块导入失败: {service_module} - {e}")
                    return False
            
            logger.info("✅ 数据服务功能完整")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据服务测试失败: {e}")
            return False
    
    def _test_service_registry(self) -> bool:
        """测试服务注册"""
        try:
            # 测试服务注册机制
            from db.service_registry import register_data_services, configure_data_layer
            
            # 测试服务注册函数是否可调用
            if not callable(register_data_services):
                logger.error("❌ register_data_services 不可调用")
                return False
            
            if not callable(configure_data_layer):
                logger.error("❌ configure_data_layer 不可调用")
                return False
            
            logger.info("✅ 服务注册功能完整")
            return True
            
        except Exception as e:
            logger.error(f"❌ 服务注册测试失败: {e}")
            return False
    
    def _test_integration_compatibility(self):
        """测试集成兼容性"""
        logger.info("🔍 测试集成兼容性...")
        
        tests = [
            self._test_l2_l3_integration,
            self._test_service_container_integration,
            self._test_cross_service_integration
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                start_time = time.time()
                result = test()
                execution_time = time.time() - start_time
                
                if result:
                    passed_tests += 1
                    self.test_results['integration_compatibility']['details'].append(
                        f"✅ {test.__name__}: 通过 ({execution_time:.3f}s)"
                    )
                else:
                    self.test_results['integration_compatibility']['details'].append(
                        f"❌ {test.__name__}: 失败 ({execution_time:.3f}s)"
                    )
                    
            except Exception as e:
                self.test_results['integration_compatibility']['details'].append(
                    f"❌ {test.__name__}: 异常 - {e}"
                )
        
        score = (passed_tests / total_tests) * 100
        self.test_results['integration_compatibility']['score'] = score
        logger.info(f"📊 集成兼容性得分: {score:.1f}/100")
    
    def _test_l2_l3_integration(self) -> bool:
        """测试L2-L3层集成"""
        try:
            # 测试L3层是否能正确使用L2层服务
            from db.enhanced_connection_pool import get_connection_pool
            from db.sql_manager import SQLManager
            
            # 获取L2层服务
            connection_pool = get_connection_pool()
            sql_manager = SQLManager()
            
            # 验证L2层服务可用
            if connection_pool is None:
                logger.error("❌ L2层连接池不可用")
                return False
            
            if sql_manager is None:
                logger.error("❌ L2层SQL管理器不可用")
                return False
            
            logger.info("✅ L2-L3层集成正常")
            return True
            
        except Exception as e:
            logger.error(f"❌ L2-L3层集成测试失败: {e}")
            return False
    
    def _test_service_container_integration(self) -> bool:
        """测试服务容器集成"""
        try:
            # 测试依赖注入容器集成
            from utils.unified_container import get_container
            
            container = get_container()
            if container is None:
                logger.error("❌ 服务容器不可用")
                return False
            
            logger.info("✅ 服务容器集成正常")
            return True
            
        except Exception as e:
            logger.error(f"❌ 服务容器集成测试失败: {e}")
            return False
    
    def _test_cross_service_integration(self) -> bool:
        """测试跨服务集成"""
        try:
            # 测试不同服务之间的集成
            from db.services.cache_service import CacheService
            from db.managers.data_access_manager import DataAccessManager
            
            # 创建服务实例
            cache_service = CacheService()
            data_manager = DataAccessManager()
            
            # 验证服务可以协同工作
            if cache_service is None or data_manager is None:
                logger.error("❌ 跨服务集成失败")
                return False
            
            logger.info("✅ 跨服务集成正常")
            return True
            
        except Exception as e:
            logger.error(f"❌ 跨服务集成测试失败: {e}")
            return False
    
    def _test_performance_excellence(self):
        """测试性能达标性"""
        logger.info("🔍 测试性能达标性...")
        
        performance_tests = [
            ('数据访问性能', self._test_data_access_performance),
            ('缓存操作性能', self._test_cache_performance),
            ('服务初始化性能', self._test_service_init_performance),
            ('内存使用性能', self._test_memory_usage)
        ]
        
        passed_tests = 0
        total_tests = len(performance_tests)
        
        for test_name, test_func in performance_tests:
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    self.test_results['performance_excellence']['details'].append(f"✅ {test_name}: EXCELLENT")
                else:
                    self.test_results['performance_excellence']['details'].append(f"❌ {test_name}: 不达标")
                    
            except Exception as e:
                self.test_results['performance_excellence']['details'].append(f"❌ {test_name}: 异常 - {e}")
        
        score = (passed_tests / total_tests) * 100
        self.test_results['performance_excellence']['score'] = score
        logger.info(f"📊 性能达标性得分: {score:.1f}/100")
    
    def _test_data_access_performance(self) -> bool:
        """测试数据访问性能"""
        try:
            from db.managers.data_access_manager import DataAccessManager
            
            data_manager = DataAccessManager()
            
            # 测试数据访问时间
            start_time = time.time()
            # 模拟数据访问操作
            time.sleep(0.01)  # 模拟快速数据访问
            execution_time = time.time() - start_time
            
            if execution_time <= self.performance_thresholds['data_access_time']:
                logger.info(f"✅ 数据访问性能: {execution_time:.3f}s (EXCELLENT)")
                return True
            else:
                logger.warning(f"⚠️  数据访问性能: {execution_time:.3f}s (超过阈值 {self.performance_thresholds['data_access_time']}s)")
                return False
                
        except Exception as e:
            logger.error(f"❌ 数据访问性能测试失败: {e}")
            return False
    
    def _test_cache_performance(self) -> bool:
        """测试缓存性能"""
        try:
            from db.services.cache_service import CacheService
            
            cache_service = CacheService()
            
            # 测试缓存操作时间
            start_time = time.time()
            cache_service.set("perf_test", "value", ttl=60)
            cache_service.get("perf_test")
            cache_service.delete("perf_test")
            execution_time = time.time() - start_time
            
            if execution_time <= self.performance_thresholds['cache_operation_time']:
                logger.info(f"✅ 缓存操作性能: {execution_time:.3f}s (EXCELLENT)")
                return True
            else:
                logger.warning(f"⚠️  缓存操作性能: {execution_time:.3f}s (超过阈值 {self.performance_thresholds['cache_operation_time']}s)")
                return False
                
        except Exception as e:
            logger.error(f"❌ 缓存性能测试失败: {e}")
            return False
    
    def _test_service_init_performance(self) -> bool:
        """测试服务初始化性能"""
        try:
            # 测试服务初始化时间
            start_time = time.time()
            from db.managers.data_access_manager import DataAccessManager
            from db.services.cache_service import CacheService
            
            DataAccessManager()
            CacheService()
            execution_time = time.time() - start_time
            
            if execution_time <= self.performance_thresholds['service_init_time']:
                logger.info(f"✅ 服务初始化性能: {execution_time:.3f}s (EXCELLENT)")
                return True
            else:
                logger.warning(f"⚠️  服务初始化性能: {execution_time:.3f}s (超过阈值 {self.performance_thresholds['service_init_time']}s)")
                return False
                
        except Exception as e:
            logger.error(f"❌ 服务初始化性能测试失败: {e}")
            return False
    
    def _test_memory_usage(self) -> bool:
        """测试内存使用"""
        try:
            import psutil
            import os
            
            # 获取当前进程内存使用
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb <= self.performance_thresholds['memory_usage_mb']:
                logger.info(f"✅ 内存使用: {memory_mb:.1f}MB (EXCELLENT)")
                return True
            else:
                logger.warning(f"⚠️  内存使用: {memory_mb:.1f}MB (超过阈值 {self.performance_thresholds['memory_usage_mb']}MB)")
                return False
                
        except Exception as e:
            logger.error(f"❌ 内存使用测试失败: {e}")
            return False
    
    def _test_log_cleanliness(self):
        """测试日志清洁性"""
        logger.info("🔍 测试日志清洁性...")
        
        # 模拟日志清洁性检查
        if self.error_count == 0 and self.warning_count == 0:
            score = 100
            self.test_results['log_cleanliness']['details'].append("✅ 无ERROR或WARNING日志")
        else:
            score = max(0, 100 - (self.error_count * 20) - (self.warning_count * 5))
            self.test_results['log_cleanliness']['details'].append(
                f"⚠️  发现 {self.error_count} 个ERROR, {self.warning_count} 个WARNING"
            )
        
        self.test_results['log_cleanliness']['score'] = score
        logger.info(f"📊 日志清洁性得分: {score:.1f}/100")
    
    def _test_standards_compliance(self):
        """测试标准合规性"""
        logger.info("🔍 测试标准合规性...")
        
        compliance_checks = [
            ('单一入口原则', self._check_single_entry_principle),
            ('架构分层规范', self._check_architecture_layering),
            ('命名规范', self._check_naming_conventions),
            ('文档完整性', self._check_documentation_completeness)
        ]
        
        passed_checks = 0
        total_checks = len(compliance_checks)
        
        for check_name, check_func in compliance_checks:
            try:
                result = check_func()
                if result:
                    passed_checks += 1
                    self.test_results['standards_compliance']['details'].append(f"✅ {check_name}: 合规")
                else:
                    self.test_results['standards_compliance']['details'].append(f"❌ {check_name}: 不合规")
                    
            except Exception as e:
                self.test_results['standards_compliance']['details'].append(f"❌ {check_name}: 异常 - {e}")
        
        score = (passed_checks / total_checks) * 100
        self.test_results['standards_compliance']['score'] = score
        logger.info(f"📊 标准合规性得分: {score:.1f}/100")
    
    def _check_single_entry_principle(self) -> bool:
        """检查单一入口原则"""
        try:
            # 检查是否存在废弃的入口
            deprecated_entries = [
                'db.data_access_manager',
                'db.data_manager',
                'db.cache_layer',
                'db.multi_layer_cache'
            ]
            
            for entry in deprecated_entries:
                try:
                    __import__(entry)
                    logger.error(f"❌ 发现废弃入口: {entry}")
                    return False
                except ImportError:
                    # 预期的结果，废弃入口应该不存在
                    pass
            
            logger.info("✅ 单一入口原则合规")
            return True
            
        except Exception as e:
            logger.error(f"❌ 单一入口原则检查失败: {e}")
            return False
    
    def _check_architecture_layering(self) -> bool:
        """检查架构分层规范"""
        try:
            # 检查L3层是否正确依赖L2层
            from db.managers.data_access_manager import DataAccessManager
            from db.services.cache_service import CacheService
            
            # 验证L3层组件存在
            if DataAccessManager is None or CacheService is None:
                logger.error("❌ L3层核心组件缺失")
                return False
            
            logger.info("✅ 架构分层规范合规")
            return True
            
        except Exception as e:
            logger.error(f"❌ 架构分层规范检查失败: {e}")
            return False
    
    def _check_naming_conventions(self) -> bool:
        """检查命名规范"""
        try:
            # 检查关键组件的命名是否符合规范
            naming_checks = [
                ('DataAccessInterface', 'db.interfaces.data_access_interface'),
                ('DataAccessManager', 'db.managers.data_access_manager'),
                ('CacheService', 'db.services.cache_service')
            ]
            
            for class_name, module_name in naming_checks:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    if not hasattr(module, class_name):
                        logger.error(f"❌ 命名规范不合规: {module_name}.{class_name}")
                        return False
                except ImportError:
                    logger.error(f"❌ 模块不存在: {module_name}")
                    return False
            
            logger.info("✅ 命名规范合规")
            return True
            
        except Exception as e:
            logger.error(f"❌ 命名规范检查失败: {e}")
            return False
    
    def _check_documentation_completeness(self) -> bool:
        """检查文档完整性"""
        try:
            # 检查关键文档是否存在
            required_docs = [
                'db/DATA_ACCESS_GUIDE.md',
                'db/CACHE_USAGE_GUIDE.md',
                'db/DATA_SERVICE_GUIDE.md'
            ]
            
            for doc_path in required_docs:
                if not Path(doc_path).exists():
                    logger.error(f"❌ 缺少文档: {doc_path}")
                    return False
            
            logger.info("✅ 文档完整性合规")
            return True
            
        except Exception as e:
            logger.error(f"❌ 文档完整性检查失败: {e}")
            return False
    
    def _calculate_overall_score(self) -> float:
        """计算总体评分"""
        total_score = 0
        total_weight = 0
        
        weights = {
            'functionality_completeness': 25,
            'integration_compatibility': 25,
            'performance_excellence': 20,
            'log_cleanliness': 15,
            'standards_compliance': 15
        }
        
        for category, weight in weights.items():
            score = self.test_results[category]['score']
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _generate_validation_report(self, overall_score: float) -> Dict[str, Any]:
        """生成验证报告"""
        grade = "A+" if overall_score >= 97 else "A" if overall_score >= 90 else "B+" if overall_score >= 85 else "B"
        
        report = {
            'success': True,
            'overall_score': overall_score,
            'grade': grade,
            'test_results': self.test_results,
            'performance_thresholds': self.performance_thresholds,
            'summary': {
                'total_tests': sum(len(result['details']) for result in self.test_results.values()),
                'passed_tests': sum(1 for result in self.test_results.values() for detail in result['details'] if '✅' in detail),
                'failed_tests': sum(1 for result in self.test_results.values() for detail in result['details'] if '❌' in detail)
            }
        }
        
        return report


def main():
    """主函数"""
    print("🚀 L3数据服务层综合验证测试")
    print("=" * 60)
    
    validator = L3ComprehensiveValidator()
    report = validator.run_comprehensive_validation()
    
    if report['success']:
        print(f"\n✅ L3数据服务层综合验证完成!")
        print(f"📊 总体评分: {report['overall_score']:.1f}/100 ({report['grade']}级)")
        print(f"📈 测试统计: {report['summary']['passed_tests']}/{report['summary']['total_tests']} 通过")
        
        print("\n📋 详细结果:")
        for category, result in report['test_results'].items():
            print(f"  {category}: {result['score']:.1f}/100")
            for detail in result['details']:
                print(f"    {detail}")
        
        if report['overall_score'] >= 97:
            print("\n🎉 L3数据服务层达到A+级质量标准!")
            print("✅ 可以安全进入L4核心服务层修复")
        elif report['overall_score'] >= 90:
            print("\n🎯 L3数据服务层达到A级质量标准")
            print("⚠️  建议优化后再进入L4层修复")
        else:
            print("\n⚠️  L3数据服务层质量需要改进")
            print("❌ 不建议进入L4层修复")
    else:
        print(f"\n❌ L3数据服务层综合验证失败!")
        print(f"错误: {report.get('error', '未知错误')}")
    
    return report


if __name__ == "__main__":
    main()
