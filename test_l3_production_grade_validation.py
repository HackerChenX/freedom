#!/usr/bin/env python3
"""
L3数据服务层生产级验证脚本

验证所有HIGH和MEDIUM优先级问题的修复情况
严格按照生产级标准进行验证，不允许简化逻辑规避问题
"""

import sys
import os
import time
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor

logger = get_logger(__name__)


class L3ProductionGradeValidator:
    """L3数据服务层生产级验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.test_results = {}
        self.performance_metrics = {}
        self.error_logs = []
        self.start_time = time.time()
        
        logger.info("=== L3数据服务层生产级验证开始 ===")
    
    @exception_handler(reraise=False, default_return=False)
    def test_cache_service_real_implementation(self) -> bool:
        """测试缓存服务真实实现（HIGH优先级问题1）"""
        logger.info("测试1: 缓存服务真实实现")
        
        try:
            from db.services.cache_service import CacheService
            from db.interfaces.cache_interface import ICacheService
            
            # 验证继承关系
            cache_service = CacheService()
            if not isinstance(cache_service, ICacheService):
                self.error_logs.append("缓存服务未正确继承ICacheService接口")
                return False
            
            # 验证真实缓存层实现
            if hasattr(cache_service, 'cache_layer') and cache_service.cache_layer is None:
                self.error_logs.append("缓存服务仍然使用空实现 (cache_layer = None)")
                return False
            
            # 测试基础缓存操作
            test_key = "test:cache:key"
            test_value = {"test": "data", "timestamp": time.time()}
            
            # 测试设置
            if not cache_service.set_9(test_key, test_value, 60):
                self.error_logs.append("缓存设置操作失败")
                return False
            
            # 测试获取
            retrieved_value = cache_service.get_9(test_key)
            if retrieved_value != test_value:
                self.error_logs.append(f"缓存获取操作失败: 期望 {test_value}, 实际 {retrieved_value}")
                return False
            
            # 测试存在检查
            if not cache_service.exists_Interface(test_key):
                self.error_logs.append("缓存存在检查失败")
                return False
            
            # 测试删除
            if not cache_service.delete_Interface(test_key):
                self.error_logs.append("缓存删除操作失败")
                return False
            
            # 测试业务方法
            stock_data = {"code": "000001", "name": "平安银行", "price": 10.5}
            if not cache_service.set_stock_basic_Interface("000001", stock_data):
                self.error_logs.append("股票基础信息缓存设置失败")
                return False
            
            retrieved_stock = cache_service.get_stock_basic_Interface("000001")
            if retrieved_stock != stock_data:
                self.error_logs.append("股票基础信息缓存获取失败")
                return False
            
            # 测试统计信息
            stats = cache_service.get_cache_stats_Interface()
            if not isinstance(stats, dict) or 'hits' not in stats:
                self.error_logs.append("缓存统计信息获取失败")
                return False
            
            logger.info("✅ 缓存服务真实实现测试通过")
            return True
            
        except Exception as e:
            self.error_logs.append(f"缓存服务测试异常: {e}")
            logger.error(f"缓存服务测试失败: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_data_access_manager_fixes(self) -> bool:
        """测试数据访问管理器修复（HIGH优先级问题2-4）"""
        logger.info("测试2: 数据访问管理器修复")
        
        try:
            from db.managers.data_access_manager import DataAccessManager
            from db.interfaces.data_access_interface import IDataAccess
            
            # 验证继承关系
            data_manager = DataAccessManager()
            if not isinstance(data_manager, IDataAccess):
                self.error_logs.append("数据访问管理器未正确继承IDataAccess接口")
                return False
            
            # 测试方法重复定义修复
            # 检查get_latest_data_data_access_interface方法是否正确实现
            try:
                result = data_manager.get_latest_data_data_access_interface("stock_info", "000001")
                # 这里应该返回None或Dict，不应该抛出异常
                logger.info(f"get_latest_data_data_access_interface 返回: {type(result)}")
            except Exception as e:
                self.error_logs.append(f"get_latest_data_data_access_interface方法调用失败: {e}")
                return False
            
            # 测试SQL注入防护
            try:
                # 测试安全的表名
                safe_result = data_manager.check_data_exists("stock_info", {"code": "000001"})
                logger.info(f"安全表名测试通过: {safe_result}")
                
                # 测试不安全的表名（应该抛出异常）
                try:
                    unsafe_result = data_manager.check_data_exists("stock_info; DROP TABLE users;", {"code": "000001"})
                    self.error_logs.append("SQL注入防护失败：不安全的表名未被拒绝")
                    return False
                except ValueError as ve:
                    logger.info(f"SQL注入防护正常工作: {ve}")
                
                # 测试不安全的列名（应该抛出异常）
                try:
                    unsafe_result = data_manager.check_data_exists("stock_info", {"code'; DROP TABLE users; --": "000001"})
                    self.error_logs.append("SQL注入防护失败：不安全的列名未被拒绝")
                    return False
                except ValueError as ve:
                    logger.info(f"列名SQL注入防护正常工作: {ve}")
                    
            except Exception as e:
                self.error_logs.append(f"SQL注入防护测试失败: {e}")
                return False
            
            logger.info("✅ 数据访问管理器修复测试通过")
            return True
            
        except Exception as e:
            self.error_logs.append(f"数据访问管理器测试异常: {e}")
            logger.error(f"数据访问管理器测试失败: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_service_registry_error_handling(self) -> bool:
        """测试服务注册错误处理（HIGH优先级问题6）"""
        logger.info("测试3: 服务注册错误处理")
        
        try:
            from db.service_registry import ServiceRegistry
            
            registry = ServiceRegistry()
            
            # 测试关键服务注册失败抛出异常
            try:
                registry.register_service("", None, critical=True)
                self.error_logs.append("关键服务注册失败未抛出异常")
                return False
            except (ValueError, RuntimeError) as e:
                logger.info(f"关键服务注册失败正确抛出异常: {e}")
            
            # 测试非关键服务注册失败返回False
            result = registry.register_service("", None, critical=False)
            if result is not False:
                self.error_logs.append("非关键服务注册失败未返回False")
                return False
            
            # 测试正常服务注册
            class TestService:
                pass
            
            result = registry.register_service("test_service", TestService, critical=True)
            if not result:
                self.error_logs.append("正常服务注册失败")
                return False
            
            logger.info("✅ 服务注册错误处理测试通过")
            return True
            
        except Exception as e:
            self.error_logs.append(f"服务注册测试异常: {e}")
            logger.error(f"服务注册测试失败: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_architecture_compliance(self) -> bool:
        """测试架构合规性（HIGH优先级问题4）"""
        logger.info("测试4: 架构合规性检查")
        
        try:
            # 检查L3层是否有跨层导入
            import ast
            import os
            
            l3_files = []
            for root, dirs, files in os.walk("db"):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        l3_files.append(os.path.join(root, file))
            
            violations = []
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否有违规导入
                    if 'from analysis' in content or 'from strategy' in content or 'from indicators' in content:
                        violations.append(f"{file_path}: 违规跨层导入L4/L5层组件")
                    
                except Exception as e:
                    logger.warning(f"无法检查文件 {file_path}: {e}")
            
            if violations:
                for violation in violations:
                    self.error_logs.append(violation)
                return False
            
            logger.info("✅ 架构合规性检查通过")
            return True
            
        except Exception as e:
            self.error_logs.append(f"架构合规性检查异常: {e}")
            logger.error(f"架构合规性检查失败: {e}")
            return False
    
    @performance_monitor(threshold_seconds=10.0)
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行全面验证"""
        logger.info("开始L3数据服务层生产级全面验证")
        
        # 定义测试用例
        test_cases = [
            ("缓存服务真实实现", self.test_cache_service_real_implementation),
            ("数据访问管理器修复", self.test_data_access_manager_fixes),
            ("服务注册错误处理", self.test_service_registry_error_handling),
            ("架构合规性检查", self.test_architecture_compliance),
        ]
        
        # 执行测试
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_name, test_func in test_cases:
            logger.info(f"执行测试: {test_name}")
            start_time = time.time()
            
            try:
                result = test_func()
                execution_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    'passed': result,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                if result:
                    passed_tests += 1
                    logger.info(f"✅ {test_name} - 通过 ({execution_time:.3f}s)")
                else:
                    logger.error(f"❌ {test_name} - 失败 ({execution_time:.3f}s)")
                    
            except Exception as e:
                execution_time = time.time() - start_time
                self.test_results[test_name] = {
                    'passed': False,
                    'execution_time': execution_time,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"❌ {test_name} - 异常: {e}")
        
        # 计算总体结果
        pass_rate = (passed_tests / total_tests) * 100
        total_time = time.time() - self.start_time
        
        # 评级计算
        if pass_rate >= 97:
            grade = "A+"
            score = 97 + (pass_rate - 97) * 3
        elif pass_rate >= 90:
            grade = "A"
            score = 90 + (pass_rate - 90) * 7 / 7
        elif pass_rate >= 80:
            grade = "B"
            score = 80 + (pass_rate - 80) * 10 / 10
        elif pass_rate >= 70:
            grade = "C"
            score = 70 + (pass_rate - 70) * 10 / 10
        else:
            grade = "D"
            score = pass_rate
        
        # 生成验证报告
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': pass_rate,
            'grade': grade,
            'score': round(score, 1),
            'total_execution_time': round(total_time, 3),
            'test_results': self.test_results,
            'error_logs': self.error_logs,
            'performance_metrics': self.performance_metrics,
            'validation_status': 'PASSED' if pass_rate == 100 else 'FAILED',
            'recommendation': self._get_recommendation(pass_rate, grade)
        }
        
        return validation_report
    
    def _get_recommendation(self, pass_rate: float, grade: str) -> str:
        """获取推荐建议"""
        if pass_rate == 100:
            return "✅ 强烈推荐批准进入L4核心服务层修复"
        elif pass_rate >= 90:
            return "⚠️ 建议修复剩余问题后进入L4层修复"
        else:
            return "❌ 强烈不建议进入L4层，必须先修复所有HIGH优先级问题"


def main():
    """主函数"""
    try:
        validator = L3ProductionGradeValidator()
        report = validator.run_comprehensive_validation()
        
        # 输出验证报告
        print("\n" + "="*80)
        print("🎯 L3数据服务层生产级验证报告")
        print("="*80)
        print(f"验证时间: {report['timestamp']}")
        print(f"总测试数: {report['total_tests']}")
        print(f"通过测试: {report['passed_tests']}")
        print(f"失败测试: {report['failed_tests']}")
        print(f"通过率: {report['pass_rate']:.1f}%")
        print(f"评级: {report['grade']} ({report['score']}/100)")
        print(f"执行时间: {report['total_execution_time']}s")
        print(f"验证状态: {report['validation_status']}")
        print(f"推荐建议: {report['recommendation']}")
        
        if report['error_logs']:
            print("\n❌ 错误日志:")
            for i, error in enumerate(report['error_logs'], 1):
                print(f"  {i}. {error}")
        
        print("\n📊 详细测试结果:")
        for test_name, result in report['test_results'].items():
            status = "✅ 通过" if result['passed'] else "❌ 失败"
            print(f"  {test_name}: {status} ({result['execution_time']:.3f}s)")
        
        print("="*80)
        
        # 返回适当的退出码
        return 0 if report['pass_rate'] == 100 else 1
        
    except Exception as e:
        logger.error(f"验证过程发生异常: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
