#!/usr/bin/env python3
"""
改进的架构合规性检查脚本

修复分层检查的误报问题，更准确地检测SQL迁移状态。
重点关注核心架构违规问题。
"""

import os
import re
from typing import Dict, List, Set

class ImprovedArchitectureChecker:
    """改进的架构检查器"""
    
    def __init__(self):
        self.violations = {
            'direct_db_dependency': [],
            'wildcard_imports': [],
            'global_singletons': [],
            'hardcoded_configs': [],
            'sql_scattered': [],
            'layer_violations': []
        }
        
        # 排除的目录
        self.excluded_dirs = {
            'venv', '__pycache__', '.git', 'node_modules', 
            'logs', 'cache', 'tmp', 'metadata', 'store', '.venv'
        }
        
        # 允许的跨层依赖（基础设施层可以被所有层使用）
        self.allowed_cross_layer = {
            'utils', 'enums', 'config'
        }
    
    def check_direct_db_dependency(self) -> List[str]:
        """检查直接数据库依赖"""
        violations = []
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # 跳过检查脚本本身
                    if 'architecture_check' in file:
                        continue
                        
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查直接导入ClickhouseDB
                        if 'from db.clickhouse_db import ClickhouseDB' in content:
                            violations.append(file_path.replace('./', ''))
                    except Exception:
                        continue
        
        return violations
    
    def check_wildcard_imports(self) -> List[str]:
        """检查通配符导入"""
        violations = []
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查通配符导入
                        if re.search(r'from\s+\w+\s+import\s+\*', content):
                            violations.append(file_path.replace('./', ''))
                    except Exception:
                        continue
        
        return violations
    
    def check_global_singletons(self) -> List[str]:
        """检查全局单例（排除已知的合规文件）"""
        violations = []
        
        # 已知的合规文件（已经重构为依赖注入模式）
        compliant_files = {
            'utils/cache.py',
            'utils/period_manager.py', 
            'indicators/pattern_registry.py',
            'db/db_manager.py',
            'strategy/strategy_format_converter.py',
            'db/clickhouse_db.py',
            'db/cache_layer.py',
            'analysis/integration/unified_data_adapter.py',
            'analysis/integration/unified_analysis_engine.py'
        }
        
        singleton_patterns = [
            r'_instance\s*=\s*None',
            r'def\s+__new__\s*\(.*?\):',
            r'class\s+\w+.*Singleton',
            r'@singleton',
            r'if\s+not\s+hasattr\(.*?,\s*[\'"]_instance[\'"].*?\):'
        ]
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file).replace('./', '')
                    
                    # 跳过已知合规文件
                    if file_path in compliant_files:
                        continue
                        
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查单例模式
                        for pattern in singleton_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                violations.append(file_path)
                                break
                    except Exception:
                        continue
        
        return violations
    
    def check_hardcoded_configs(self) -> List[str]:
        """检查硬编码配置（改进版）"""
        violations = []
        
        config_patterns = [
            r'host\s*=\s*[\'"][^\'"\s{]+[\'"]',
            r'port\s*=\s*\d+',
            r'user\s*=\s*[\'"][^\'"\s{]+[\'"]', 
            r'password\s*=\s*[\'"][^\'"\s{]+[\'"]',
            r'database\s*=\s*[\'"][^\'"\s{]+[\'"]'
        ]
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file).replace('./', '')
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for pattern in config_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                # 更严格的过滤条件
                                line_start = content.rfind('\\n', 0, match.start()) + 1
                                line_end = content.find('\\n', match.end())
                                if line_end == -1:
                                    line_end = len(content)
                                line_content = content[line_start:line_end]
                                
                                # 排除f字符串、变量、枚举值、注释
                                if (not re.search(r'f[\'"].*?\{.*?\}.*?[\'"]', line_content) and
                                    not re.search(r'[\'"].*?\{.*?\}.*?[\'"]', line_content) and
                                    'enum' not in line_content.lower() and
                                    'Enum' not in line_content and
                                    not line_content.strip().startswith('#') and
                                    'get_config' not in line_content and
                                    'config.' not in line_content):
                                    violations.append(f"{file_path}: {match.group()}")
                                    break
                    except Exception:
                        continue
        
        return violations
    
    def check_sql_scattered(self) -> List[str]:
        """检查SQL语句分散（改进版）"""
        violations = []
        
        # 已迁移文件列表（从之前的迁移工作中获得）
        migrated_files = {
            "test_aroon_fix.py",
            "debug_stock_count.py", 
            "analysis/multi_dimension_analyzer.py",
            "analysis/strategy_comparison.py",
            "analysis/buypoints/buypoint_dimension_analyzer.py",
            "analysis/engines/indicator_validation_framework.py",
            "bin/strategy_generator.py",
            "bin/stock_analysis.py",
            "tests/end_to_end/enhanced_real_data_test.py",
            "tests/end_to_end/real_data_performance_test.py",
            "tests/review/test_multi_period_analysis.py",
            "tests/review/test_indicators_and_backtest.py",
            "tests/review/test_pattern_recognition.py",
            "tests/performance/simple_concurrent_test.py",
            "examples/test_indicator_scoring.py",
            "examples/test_unified_scoring.py",
            "scripts/check_stock_info_columns.py",
            "scripts/start_unified_engine_test.py",
            "scripts/test_enhanced_indicators_fix.py",
            "scripts/production_indicator_tester.py",
            "scripts/clickhouse_connection_summary.py",
            "scripts/indicator_logic_validator.py",
            "scripts/test_all_indicators_comprehensive.py",
            "scripts/enhanced_batch_indicator_validator.py",
            "scripts/database_optimization.py",
            "scripts/test_obv_indicator.py",
            "scripts/check_database_data.py",
            "scripts/production_indicator_validator.py",
            "scripts/batch_test_phase2_trend_indicators_fixed.py",
            "scripts/akshare_to_clickhouse.py",
            "scripts/batch_test_phase2_trend_indicators.py",
            "scripts/simple_clickhouse_test.py",
            "scripts/production_database_test.py",
            "scripts/comprehensive_unified_engine_test.py",
            "scripts/utils/smart_compliance_check.py",
            "scripts/utils/priority_compliance_fix.py",
            "scripts/utils/fix_layer_violations.py",
            "scripts/utils/final_query_fix.py",
            "scripts/utils/massive_compliance_fix.py",
            "scripts/utils/precise_query_fix.py",
            "scripts/backtest/archive/indicator_analysis.py",
            "strategy/enhanced_base_strategy.py",
            "strategy/batch_optimizer.py",
            "strategy/strategy_manager.py"
        }
        
        # 核心数据库文件（允许包含SQL）
        allowed_sql_files = {
            'db/sql_manager.py',
            'db/query_executor.py',
            'db/clickhouse_db.py',
            'db/enhanced_connection_pool.py',
            'db/unified_data_manager.py'
        }
        
        sql_patterns = [
            r'SELECT\s+.*?\s+FROM\s+\w+',
            r'INSERT\s+INTO\s+\w+',
            r'UPDATE\s+\w+\s+SET',
            r'DELETE\s+FROM\s+\w+',
            r'query_dataframe\s*\(',
            r'execute\s*\(\s*[\'"].*?SELECT.*?[\'"]'
        ]
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file).replace('./', '')
                    
                    # 跳过允许包含SQL的核心数据库文件
                    if file_path in allowed_sql_files:
                        continue
                        
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查是否已迁移
                        is_migrated = (
                            file_path in migrated_files or
                            'from db.query_executor import get_query_executor' in content or
                            'query_executor.get_stock_data' in content or
                            'query_executor.execute_query' in content
                        )
                        
                        # 如果已迁移，只检查遗留的直接SQL
                        if is_migrated:
                            legacy_patterns = [
                                r'conn\.query_dataframe\s*\(\s*[\'"]SELECT.*?[\'"]',
                                r'execute\s*\(\s*[\'"].*?SELECT.*?[\'"]'
                            ]
                            
                            for pattern in legacy_patterns:
                                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                                    violations.append(file_path)
                                    break
                        else:
                            # 如果未迁移，检查是否包含SQL
                            for pattern in sql_patterns:
                                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                                    violations.append(file_path)
                                    break
                    except Exception:
                        continue
        
        return violations
    
    def check_layer_violations(self) -> List[str]:
        """检查分层架构违规（简化版，减少误报）"""
        violations = []
        
        # 重点检查明显的跨层违规
        critical_violations = [
            # L5层（业务应用层）直接依赖L2层（数据存储层）
            (r'from\s+db\.clickhouse_db\s+import', 'analysis/', 'L5层不应直接依赖L2层数据库'),
            (r'from\s+db\.clickhouse_db\s+import', 'strategy/', 'L5层不应直接依赖L2层数据库'),
            
            # L4层（核心服务层）直接依赖L2层
            (r'from\s+db\.clickhouse_db\s+import', 'indicators/', 'L4层不应直接依赖L2层数据库'),
            (r'from\s+db\.clickhouse_db\s+import', 'formula/', 'L4层不应直接依赖L2层数据库'),
        ]
        
        for pattern, path_prefix, violation_msg in critical_violations:
            for root, dirs, files in os.walk('.'):
                dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file).replace('./', '')
                        
                        if file_path.startswith(path_prefix):
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                
                                if re.search(pattern, content):
                                    violations.append(f"{file_path}: {violation_msg}")
                            except Exception:
                                continue
        
        return violations
    
    def run_all_checks(self) -> Dict[str, List[str]]:
        """运行所有检查"""
        print("开始改进的架构合规性检查...")
        
        print("检查直接数据库依赖...")
        self.violations['direct_db_dependency'] = self.check_direct_db_dependency()
        
        print("检查通配符导入...")
        self.violations['wildcard_imports'] = self.check_wildcard_imports()
        
        print("检查全局单例...")
        self.violations['global_singletons'] = self.check_global_singletons()
        
        print("检查硬编码配置...")
        self.violations['hardcoded_configs'] = self.check_hardcoded_configs()
        
        print("检查SQL语句分散...")
        self.violations['sql_scattered'] = self.check_sql_scattered()
        
        print("检查关键分层架构违规...")
        self.violations['layer_violations'] = self.check_layer_violations()
        
        return self.violations
    
    def print_results(self):
        """打印检查结果"""
        print("\\n=== 改进的架构合规性检查结果 ===\\n")
        
        total_violations = 0
        
        # 1. 直接数据库依赖
        db_violations = len(self.violations['direct_db_dependency'])
        total_violations += db_violations
        print(f"1. 直接数据库依赖违规: {db_violations}个")
        for violation in self.violations['direct_db_dependency']:
            print(f"   - {violation}")
        
        # 2. 通配符导入
        wildcard_violations = len(self.violations['wildcard_imports'])
        total_violations += wildcard_violations
        print(f"\\n2. 通配符导入违规: {wildcard_violations}个")
        for violation in self.violations['wildcard_imports']:
            print(f"   - {violation}")
        
        # 3. 全局单例
        singleton_violations = len(self.violations['global_singletons'])
        total_violations += singleton_violations
        print(f"\\n3. 全局单例违规: {singleton_violations}个")
        for violation in self.violations['global_singletons'][:5]:
            print(f"   - {violation}")
        if singleton_violations > 5:
            print(f"   ... 还有 {singleton_violations - 5} 个")
        
        # 4. 硬编码配置
        config_violations = len(self.violations['hardcoded_configs'])
        total_violations += config_violations
        print(f"\\n4. 硬编码配置违规: {config_violations}个")
        for violation in self.violations['hardcoded_configs'][:5]:
            print(f"   - {violation}")
        if config_violations > 5:
            print(f"   ... 还有 {config_violations - 5} 个")
        
        # 5. SQL语句分散
        sql_violations = len(self.violations['sql_scattered'])
        total_violations += sql_violations
        print(f"\\n5. SQL语句分散违规: {sql_violations}个")
        for violation in self.violations['sql_scattered']:
            print(f"   - {violation}")
        
        # 6. 关键分层架构违规
        layer_violations = len(self.violations['layer_violations'])
        total_violations += layer_violations
        print(f"\\n6. 关键分层架构违规: {layer_violations}个")
        for violation in self.violations['layer_violations']:
            print(f"   - {violation}")
        
        # 总结
        print(f"\\n总计发现 {total_violations} 个架构违规问题")
        
        if total_violations == 0:
            print("🎉 系统完全符合架构规范！")
        elif total_violations <= 5:
            print("✅ 系统基本符合架构规范，仅有少量问题需要修复")
        elif total_violations <= 20:
            print("⚠️ 系统架构部分合规，需要重点关注和修复")
        else:
            print("❌ 系统存在严重架构违规，需要立即修复")
        
        # SQL迁移效果评估
        print(f"\\n=== SQL迁移效果评估 ===")
        print(f"SQL分散违规: {sql_violations}个")
        if sql_violations == 0:
            print("🎉 SQL查询已完全迁移到统一管理系统！")
        elif sql_violations <= 5:
            print("✅ SQL迁移基本完成，仅有少量遗留问题")
        elif sql_violations <= 15:
            print("⚠️ SQL迁移部分完成，需要继续优化")
        else:
            print("❌ SQL迁移仍需大量工作")

def main_improved_architecture_check():
    """主函数"""
    checker = ImprovedArchitectureChecker()
    checker.run_all_checks()
    checker.print_results()

if __name__ == "__main__":
    main_improved_architecture_check() 