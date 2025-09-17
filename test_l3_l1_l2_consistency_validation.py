#!/usr/bin/env python3
"""
L3数据服务层与L1/L2标准一致性验证脚本

基于L1/L2架构合规审计标准，验证L3层是否严格遵循已确立的A+级质量标准
参考文档: docs/system_optimization_2024/L1_L2_architecture_compliance_audit.md
"""

import sys
import os
import time
import ast
import traceback
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import importlib.util

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor

logger = get_logger(__name__)


class L3L1L2ConsistencyValidator:
    """L3数据服务层与L1/L2标准一致性验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.test_results = {}
        self.consistency_issues = []
        self.architecture_violations = []
        self.standard_compliance_score = {}
        self.start_time = time.time()
        
        # L1/L2标准入口定义（基于已确立的A+级标准）
        self.l1_standard_entries = {
            'config': 'config.unified_config_manager',
            'container': 'utils.unified_container', 
            'logger': 'utils.logger',
            'exception_handler': 'utils.enhanced_exception_handler',
            'performance_monitor': 'utils.enhanced_performance_monitor'
        }
        
        self.l2_standard_entries = {
            'connection_pool': 'db.enhanced_connection_pool',
            'database_config': 'config.database_config_manager',
            'sql_manager': 'db.sql_manager',
            'data_access_interface': 'db.interfaces.data_access_interface'
        }
        
        logger.info("=== L3数据服务层与L1/L2标准一致性验证开始 ===")
    
    @exception_handler(reraise=False, default_return=False)
    def test_l1_config_management_consistency(self) -> bool:
        """测试L1配置管理标准一致性"""
        logger.info("测试1: L1配置管理标准一致性")
        
        try:
            # 获取L3层所有Python文件
            l3_files = self._get_l3_python_files()
            
            config_violations = []
            deprecated_imports = []
            correct_imports = []
            
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否使用了废弃的配置入口
                    if 'from config.config import' in content or 'import config.config' in content:
                        deprecated_imports.append(f"{file_path}: 使用废弃的config.config")
                    
                    if 'from config import get_config' in content:
                        deprecated_imports.append(f"{file_path}: 使用废弃的config.__init__")
                    
                    # 检查是否使用了标准配置入口
                    if 'from config.unified_config_manager import get_config' in content:
                        correct_imports.append(f"{file_path}: 正确使用统一配置管理")
                    
                    # 检查硬编码配置
                    if self._has_hardcoded_config(content):
                        config_violations.append(f"{file_path}: 存在硬编码配置")
                        
                except Exception as e:
                    logger.warning(f"无法检查文件 {file_path}: {e}")
            
            # 评估结果
            total_files = len(l3_files)
            violation_count = len(deprecated_imports) + len(config_violations)
            
            if deprecated_imports:
                for violation in deprecated_imports:
                    self.consistency_issues.append(f"配置管理违规: {violation}")
            
            if config_violations:
                for violation in config_violations:
                    self.consistency_issues.append(f"硬编码配置: {violation}")
            
            compliance_rate = ((total_files - violation_count) / total_files * 100) if total_files > 0 else 100
            self.standard_compliance_score['l1_config'] = compliance_rate
            
            logger.info(f"L1配置管理一致性: {compliance_rate:.1f}% ({len(correct_imports)}个正确导入, {violation_count}个违规)")
            
            return compliance_rate >= 95  # A+级标准要求95%以上合规
            
        except Exception as e:
            self.consistency_issues.append(f"L1配置管理一致性检查异常: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_l1_dependency_injection_consistency(self) -> bool:
        """测试L1依赖注入标准一致性"""
        logger.info("测试2: L1依赖注入标准一致性")
        
        try:
            # 检查L3层是否正确使用统一容器
            from utils.unified_container import get_container
            
            l3_files = self._get_l3_python_files()
            di_violations = []
            correct_di_usage = []
            
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否使用了标准依赖注入
                    if 'from utils.unified_container import' in content or 'from utils.dependency_injection import' in content:
                        correct_di_usage.append(file_path)
                    
                    # 检查是否有硬编码依赖
                    if self._has_hardcoded_dependencies(content):
                        di_violations.append(f"{file_path}: 存在硬编码依赖")
                        
                except Exception as e:
                    logger.warning(f"无法检查依赖注入 {file_path}: {e}")
            
            # 测试实际的依赖注入功能
            container = get_container()
            
            # 验证L3层服务是否正确注册
            try:
                from db.services.cache_service import CacheService
                cache_service = CacheService()
                logger.info("✅ L3缓存服务可以正常实例化")
            except Exception as e:
                di_violations.append(f"L3缓存服务实例化失败: {e}")
            
            try:
                from db.managers.data_access_manager import DataAccessManager
                data_manager = DataAccessManager()
                logger.info("✅ L3数据访问管理器可以正常实例化")
            except Exception as e:
                di_violations.append(f"L3数据访问管理器实例化失败: {e}")
            
            violation_count = len(di_violations)
            compliance_rate = max(0, 100 - violation_count * 10)  # 每个违规扣10分
            self.standard_compliance_score['l1_di'] = compliance_rate
            
            if di_violations:
                for violation in di_violations:
                    self.consistency_issues.append(f"依赖注入违规: {violation}")
            
            logger.info(f"L1依赖注入一致性: {compliance_rate:.1f}% ({len(correct_di_usage)}个正确使用, {violation_count}个违规)")
            
            return compliance_rate >= 95
            
        except Exception as e:
            self.consistency_issues.append(f"L1依赖注入一致性检查异常: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_l1_logging_exception_consistency(self) -> bool:
        """测试L1日志和异常处理标准一致性"""
        logger.info("测试3: L1日志和异常处理标准一致性")
        
        try:
            l3_files = self._get_l3_python_files()
            logging_violations = []
            exception_violations = []
            correct_usage = []
            
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查日志使用
                    if 'from utils.logger import get_logger' in content:
                        correct_usage.append(f"{file_path}: 正确使用标准日志")
                    elif 'import logging' in content and 'get_logger' not in content:
                        logging_violations.append(f"{file_path}: 使用原生logging而非标准日志入口")
                    
                    # 检查异常处理装饰器使用
                    if '@exception_handler' in content:
                        if 'from utils.enhanced_exception_handler import exception_handler' in content:
                            correct_usage.append(f"{file_path}: 正确使用标准异常处理")
                        else:
                            exception_violations.append(f"{file_path}: 使用异常处理装饰器但导入不标准")
                    
                    # 检查性能监控装饰器使用
                    if '@performance_monitor' in content:
                        if 'from utils.enhanced_performance_monitor import performance_monitor' in content:
                            correct_usage.append(f"{file_path}: 正确使用标准性能监控")
                        else:
                            exception_violations.append(f"{file_path}: 使用性能监控装饰器但导入不标准")
                            
                except Exception as e:
                    logger.warning(f"无法检查日志异常处理 {file_path}: {e}")
            
            violation_count = len(logging_violations) + len(exception_violations)
            total_checks = len(l3_files)
            compliance_rate = ((total_checks - violation_count) / total_checks * 100) if total_checks > 0 else 100
            self.standard_compliance_score['l1_logging_exception'] = compliance_rate
            
            if logging_violations:
                for violation in logging_violations:
                    self.consistency_issues.append(f"日志标准违规: {violation}")
            
            if exception_violations:
                for violation in exception_violations:
                    self.consistency_issues.append(f"异常处理标准违规: {violation}")
            
            logger.info(f"L1日志异常处理一致性: {compliance_rate:.1f}% ({len(correct_usage)}个正确使用, {violation_count}个违规)")
            
            return compliance_rate >= 95
            
        except Exception as e:
            self.consistency_issues.append(f"L1日志异常处理一致性检查异常: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_l2_storage_access_consistency(self) -> bool:
        """测试L2存储访问层标准一致性"""
        logger.info("测试4: L2存储访问层标准一致性")
        
        try:
            l3_files = self._get_l3_python_files()
            storage_violations = []
            correct_usage = []
            
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否使用标准连接池
                    if 'from db.enhanced_connection_pool import' in content:
                        correct_usage.append(f"{file_path}: 正确使用标准连接池")
                    elif 'from db.clickhouse_db import get_clickhouse_db' in content:
                        storage_violations.append(f"{file_path}: 使用废弃的直接数据库连接")
                    
                    # 检查是否使用标准数据库配置
                    if 'from config.database_config_manager import' in content:
                        correct_usage.append(f"{file_path}: 正确使用标准数据库配置")
                    elif 'from config.unified_database_config import' in content:
                        correct_usage.append(f"{file_path}: 正确使用统一数据库配置")
                    
                    # 检查是否使用标准SQL管理
                    if 'from db.sql_manager import' in content:
                        correct_usage.append(f"{file_path}: 正确使用标准SQL管理")
                    
                    # 检查是否有直接SQL拼接（安全风险）
                    if self._has_sql_injection_risk(content):
                        storage_violations.append(f"{file_path}: 存在SQL注入风险")
                        
                except Exception as e:
                    logger.warning(f"无法检查存储访问 {file_path}: {e}")
            
            # 测试实际的L2层集成
            try:
                from db.enhanced_connection_pool import ClickHouseConnectionPool
                pool = ClickHouseConnectionPool()
                logger.info("✅ L2连接池可以正常实例化")
            except Exception as e:
                storage_violations.append(f"L2连接池实例化失败: {e}")
            
            try:
                from config.database_config_manager import get_database_config
                config = get_database_config()
                logger.info("✅ L2数据库配置可以正常获取")
            except Exception as e:
                storage_violations.append(f"L2数据库配置获取失败: {e}")
            
            violation_count = len(storage_violations)
            compliance_rate = max(0, 100 - violation_count * 5)  # 每个违规扣5分
            self.standard_compliance_score['l2_storage'] = compliance_rate
            
            if storage_violations:
                for violation in storage_violations:
                    self.consistency_issues.append(f"存储访问违规: {violation}")
            
            logger.info(f"L2存储访问一致性: {compliance_rate:.1f}% ({len(correct_usage)}个正确使用, {violation_count}个违规)")
            
            return compliance_rate >= 95
            
        except Exception as e:
            self.consistency_issues.append(f"L2存储访问一致性检查异常: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def test_architecture_layering_compliance(self) -> bool:
        """测试架构分层合规性"""
        logger.info("测试5: 架构分层合规性")
        
        try:
            l3_files = self._get_l3_python_files()
            layering_violations = []
            
            for file_path in l3_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查跨层导入违规（L3不能导入L4/L5/L6）
                    forbidden_imports = [
                        'from analysis',
                        'from strategy', 
                        'from indicators',
                        'from api',
                        'from bin'
                    ]
                    
                    for forbidden in forbidden_imports:
                        if forbidden in content:
                            layering_violations.append(f"{file_path}: 违规跨层导入 {forbidden}")
                    
                    # 检查是否绕过标准接口
                    if self._bypasses_standard_interfaces(content):
                        layering_violations.append(f"{file_path}: 绕过标准接口")
                        
                except Exception as e:
                    logger.warning(f"无法检查架构分层 {file_path}: {e}")
            
            violation_count = len(layering_violations)
            total_files = len(l3_files)
            compliance_rate = ((total_files - violation_count) / total_files * 100) if total_files > 0 else 100
            self.standard_compliance_score['architecture_layering'] = compliance_rate
            
            if layering_violations:
                for violation in layering_violations:
                    self.architecture_violations.append(violation)
            
            logger.info(f"架构分层合规性: {compliance_rate:.1f}% ({total_files - violation_count}/{total_files}个文件合规)")
            
            return compliance_rate >= 100  # 架构分层必须100%合规
            
        except Exception as e:
            self.consistency_issues.append(f"架构分层合规性检查异常: {e}")
            return False
    
    def _get_l3_python_files(self) -> List[str]:
        """获取L3层所有Python文件"""
        l3_files = []
        l3_directories = ['db/services', 'db/managers', 'db/interfaces']
        
        for directory in l3_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            l3_files.append(os.path.join(root, file))
        
        # 添加其他L3相关文件
        additional_files = [
            'db/service_registry.py',
            'db/parallel_processor.py'
        ]
        
        for file_path in additional_files:
            if os.path.exists(file_path):
                l3_files.append(file_path)
        
        return l3_files
    
    def _has_hardcoded_config(self, content: str) -> bool:
        """检查是否有硬编码配置"""
        hardcoded_patterns = [
            'localhost:9000',
            'password="',
            'host="localhost"',
            'port=9000',
            'database="stock"'
        ]
        
        for pattern in hardcoded_patterns:
            if pattern in content and 'config' not in content.lower():
                return True
        return False
    
    def _has_hardcoded_dependencies(self, content: str) -> bool:
        """检查是否有硬编码依赖"""
        hardcoded_patterns = [
            'get_clickhouse_db()',
            'ClickHouseClient(',
            'clickhouse_driver.Client'
        ]
        
        for pattern in hardcoded_patterns:
            if pattern in content:
                return True
        return False
    
    def _has_sql_injection_risk(self, content: str) -> bool:
        """检查是否有SQL注入风险"""
        risky_patterns = [
            'f"SELECT * FROM {',
            'f"INSERT INTO {',
            'f"UPDATE {',
            'f"DELETE FROM {',
            '+ "WHERE"',
            '% "SELECT"'
        ]
        
        for pattern in risky_patterns:
            if pattern in content:
                return True
        return False
    
    def _bypasses_standard_interfaces(self, content: str) -> bool:
        """检查是否绕过标准接口"""
        bypass_patterns = [
            'clickhouse_driver.Client',
            'direct_db_connection',
            'raw_sql_execute'
        ]
        
        for pattern in bypass_patterns:
            if pattern in content:
                return True
        return False
    
    @performance_monitor(threshold_seconds=30.0)
    def run_comprehensive_consistency_validation(self) -> Dict[str, Any]:
        """运行全面一致性验证"""
        logger.info("开始L3数据服务层与L1/L2标准一致性验证")
        
        # 定义测试用例（基于L1/L2 A+级标准）
        test_cases = [
            ("L1配置管理标准一致性", self.test_l1_config_management_consistency),
            ("L1依赖注入标准一致性", self.test_l1_dependency_injection_consistency),
            ("L1日志异常处理标准一致性", self.test_l1_logging_exception_consistency),
            ("L2存储访问层标准一致性", self.test_l2_storage_access_consistency),
            ("架构分层合规性", self.test_architecture_layering_compliance),
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
        
        # 计算总体一致性评分
        overall_compliance = sum(self.standard_compliance_score.values()) / len(self.standard_compliance_score) if self.standard_compliance_score else 0
        pass_rate = (passed_tests / total_tests) * 100
        total_time = time.time() - self.start_time
        
        # 评级计算（基于L1/L2 A+级标准）
        if overall_compliance >= 98 and pass_rate == 100:
            grade = "A+"
            score = 98 + (overall_compliance - 98) * 2
        elif overall_compliance >= 95 and pass_rate >= 80:
            grade = "A"
            score = 95 + (overall_compliance - 95) * 3 / 3
        elif overall_compliance >= 85:
            grade = "B"
            score = 85 + (overall_compliance - 85) * 10 / 10
        else:
            grade = "C"
            score = overall_compliance
        
        # 生成一致性验证报告
        consistency_report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': pass_rate,
            'overall_compliance_score': round(overall_compliance, 1),
            'grade': grade,
            'score': round(score, 1),
            'total_execution_time': round(total_time, 3),
            'standard_compliance_scores': self.standard_compliance_score,
            'test_results': self.test_results,
            'consistency_issues': self.consistency_issues,
            'architecture_violations': self.architecture_violations,
            'validation_status': 'CONSISTENT' if pass_rate == 100 and overall_compliance >= 95 else 'INCONSISTENT',
            'l1_l2_compatibility': self._assess_l1_l2_compatibility(),
            'recommendation': self._get_consistency_recommendation(overall_compliance, pass_rate)
        }
        
        return consistency_report
    
    def _assess_l1_l2_compatibility(self) -> Dict[str, Any]:
        """评估与L1/L2的兼容性"""
        return {
            'l1_config_compatibility': self.standard_compliance_score.get('l1_config', 0) >= 95,
            'l1_di_compatibility': self.standard_compliance_score.get('l1_di', 0) >= 95,
            'l1_logging_compatibility': self.standard_compliance_score.get('l1_logging_exception', 0) >= 95,
            'l2_storage_compatibility': self.standard_compliance_score.get('l2_storage', 0) >= 95,
            'architecture_compliance': self.standard_compliance_score.get('architecture_layering', 0) == 100
        }
    
    def _get_consistency_recommendation(self, compliance_score: float, pass_rate: float) -> str:
        """获取一致性推荐建议"""
        if compliance_score >= 98 and pass_rate == 100:
            return "✅ L3层与L1/L2标准完全一致，达到A+级质量标准"
        elif compliance_score >= 95 and pass_rate >= 80:
            return "⚠️ L3层基本符合L1/L2标准，建议修复剩余不一致问题"
        else:
            return "❌ L3层与L1/L2标准存在重大不一致，必须修复后才能确保架构完整性"


def main():
    """主函数"""
    try:
        validator = L3L1L2ConsistencyValidator()
        report = validator.run_comprehensive_consistency_validation()
        
        # 输出一致性验证报告
        print("\n" + "="*80)
        print("🎯 L3数据服务层与L1/L2标准一致性验证报告")
        print("="*80)
        print(f"验证时间: {report['timestamp']}")
        print(f"总测试数: {report['total_tests']}")
        print(f"通过测试: {report['passed_tests']}")
        print(f"失败测试: {report['failed_tests']}")
        print(f"测试通过率: {report['pass_rate']:.1f}%")
        print(f"整体合规评分: {report['overall_compliance_score']}/100")
        print(f"评级: {report['grade']} ({report['score']}/100)")
        print(f"执行时间: {report['total_execution_time']}s")
        print(f"一致性状态: {report['validation_status']}")
        print(f"推荐建议: {report['recommendation']}")
        
        print("\n📊 标准合规性详细评分:")
        for standard, score in report['standard_compliance_scores'].items():
            status = "✅" if score >= 95 else "⚠️" if score >= 85 else "❌"
            print(f"  {standard}: {status} {score:.1f}/100")
        
        print("\n🔗 L1/L2兼容性评估:")
        compatibility = report['l1_l2_compatibility']
        for component, is_compatible in compatibility.items():
            status = "✅ 兼容" if is_compatible else "❌ 不兼容"
            print(f"  {component}: {status}")
        
        if report['consistency_issues']:
            print("\n⚠️ 一致性问题:")
            for i, issue in enumerate(report['consistency_issues'], 1):
                print(f"  {i}. {issue}")
        
        if report['architecture_violations']:
            print("\n❌ 架构违规:")
            for i, violation in enumerate(report['architecture_violations'], 1):
                print(f"  {i}. {violation}")
        
        print("\n📋 详细测试结果:")
        for test_name, result in report['test_results'].items():
            status = "✅ 通过" if result['passed'] else "❌ 失败"
            print(f"  {test_name}: {status} ({result['execution_time']:.3f}s)")
        
        print("="*80)
        
        # 返回适当的退出码
        return 0 if report['validation_status'] == 'CONSISTENT' else 1
        
    except Exception as e:
        logger.error(f"一致性验证过程发生异常: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
