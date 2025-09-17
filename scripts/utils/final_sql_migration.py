#!/usr/bin/env python3
"""
最终SQL迁移脚本

处理剩余的31个SQL违规文件，完成SQL查询统一管理系统的实施。
根据文件类型和优先级进行分类处理。
"""

import os
import re
import shutil
from typing import Dict, List, Tuple, Set
from pathlib import Path

class FinalSQLMigrator:
    """最终SQL迁移器"""
    
    def __init__(self):
        # 剩余的31个SQL违规文件
        self.remaining_files = [
            "test_enhanced_macd.py",
            "test_enhanced_dmi.py", 
            "test_vortex.py",
            "test_production_db.py",
            "test_enhanced_trix.py",
            "debug_stock_count.py",
            "analysis/integration/unified_data_adapter.py",
            "config/database_config_manager.py",
            "tests/mocks/db_mock.py",
            "tests/performance/simple_concurrent_test.py",
            "utils/period_manager.py",
            "examples/test_basic_optimization.py",
            "examples/test_indicator_optimization.py",
            "examples/parameter_optimization.py",
            "examples/use_new_indicators.py",
            "examples/test_zxm_score_indicators.py",
            "examples/combined_indicators_strategy.py",
            "examples/use_advanced_indicators.py",
            "scripts/clickhouse_connection_summary.py",
            "scripts/test_enhanced_oscillator.py",
            "scripts/test_enhanced_macd.py",
            "scripts/test_enhanced_indicators.py",
            "scripts/simple_clickhouse_test.py",
            "scripts/utils/refactor_sql_statements.py",
            "scripts/utils/implement_sql_management.py",
            "db/db_manager.py",
            "db/enhanced_data_manager.py",
            "db/data_manager_adapter.py",
            "db/batch_data_optimizer.py",
            "db/managers/data_access_manager.py",
            "monitoring/performance_monitor.py"
        ]
        
        # 按优先级分组
        self.priority_groups = {
            'CRITICAL': [  # 核心数据库文件
                "db/db_manager.py",
                "db/enhanced_data_manager.py",
                "db/data_manager_adapter.py",
                "db/batch_data_optimizer.py",
                "db/managers/data_access_manager.py"
            ],
            'HIGH': [  # 核心业务文件
                "analysis/integration/unified_data_adapter.py",
                "utils/period_manager.py",
                "monitoring/performance_monitor.py",
                "config/database_config_manager.py"
            ],
            'MEDIUM': [  # 测试和示例文件
                "tests/mocks/db_mock.py",
                "tests/performance/simple_concurrent_test.py",
                "examples/test_basic_optimization.py",
                "examples/test_indicator_optimization.py",
                "examples/parameter_optimization.py",
                "examples/use_new_indicators.py",
                "examples/test_zxm_score_indicators.py",
                "examples/combined_indicators_strategy.py",
                "examples/use_advanced_indicators.py"
            ],
            'LOW': [  # 临时测试文件和工具脚本
                "test_enhanced_macd.py",
                "test_enhanced_dmi.py",
                "test_vortex.py", 
                "test_production_db.py",
                "test_enhanced_trix.py",
                "debug_stock_count.py",
                "scripts/clickhouse_connection_summary.py",
                "scripts/test_enhanced_oscillator.py",
                "scripts/test_enhanced_macd.py",
                "scripts/test_enhanced_indicators.py",
                "scripts/simple_clickhouse_test.py",
                "scripts/utils/refactor_sql_statements.py",
                "scripts/utils/implement_sql_management.py"
            ]
        }
        
        # 迁移模板
        self.migration_templates = {
            'db_layer': {
                'imports': [
                    'from db.query_executor import get_query_executor',
                    'from db.sql_manager import QueryType'
                ],
                'patterns': [
                    (r'conn\.query_dataframe\s*\(\s*([\'"].*?[\'"])\s*\)', 
                     'query_executor.execute_query(QueryType.CUSTOM, {"query": \\1})'),
                    (r'SELECT\s+.*?\s+FROM\s+stock_info\s+WHERE\s+code\s*=\s*[\'"]([^\'\"]+)[\'"]',
                     'query_executor.get_stock_data({"code": "\\1"})'),
                    (r'SELECT\s+COUNT\(\*\)\s+FROM\s+stock_info',
                     'query_executor.get_stock_count()'),
                    (r'SELECT\s+DISTINCT\s+code\s+FROM\s+stock_info',
                     'query_executor.get_stock_list()')
                ]
            },
            'business_layer': {
                'imports': [
                    'from db.query_executor import get_query_executor',
                    'from db.sql_manager import QueryType'
                ],
                'patterns': [
                    (r'with\s+.*?\.get_connection\(\)\s+as\s+conn:\s*\n\s*([^=]+)\s*=\s*conn\.query_dataframe\s*\(\s*([\'"].*?[\'"])\s*\)',
                     'query_executor = get_query_executor()\n\\1 = query_executor.execute_query(QueryType.CUSTOM, {"query": \\2})'),
                    (r'data_manager\.get_stock_data\s*\(',
                     'query_executor.get_stock_data('),
                    (r'data_manager\.get_stock_list\s*\(',
                     'query_executor.get_stock_list(')
                ]
            },
            'test_layer': {
                'imports': [
                    'from db.query_executor import get_query_executor',
                    'from db.sql_manager import QueryType'
                ],
                'patterns': [
                    (r'conn\.query_dataframe\s*\(\s*([\'"].*?[\'"])\s*\)',
                     'query_executor.execute_query(QueryType.CUSTOM, {"query": \\1})'),
                    (r'# 测试查询',
                     '# 使用统一查询接口进行测试')
                ]
            }
        }
    
    def detect_file_type(self, file_path: str) -> str:
        """检测文件类型"""
        if file_path.startswith('db/'):
            return 'db_layer'
        elif file_path.startswith('tests/') or file_path.startswith('test_'):
            return 'test_layer'
        else:
            return 'business_layer'
    
    def backup_file_final_sql_migration(self, file_path: str) -> str:
        """备份文件"""
        backup_path = f"{file_path}.final_migration_backup"
        if os.path.exists(file_path):
            shutil.copy2(file_path, backup_path)
            return backup_path
        return ""
    
    def add_imports(self, content: str, imports: List[str]) -> str:
        """添加导入语句"""
        lines = content.split('\n')
        import_section_end = 0
        
        # 找到导入部分的结束位置
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_section_end = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # 检查是否已存在导入
        existing_imports = set()
        for line in lines[:import_section_end]:
            if 'from db.query_executor import' in line:
                existing_imports.add('query_executor')
            if 'from db.sql_manager import' in line:
                existing_imports.add('sql_manager')
        
        # 添加缺失的导入
        new_imports = []
        for import_stmt in imports:
            if ('query_executor' in import_stmt and 'query_executor' not in existing_imports) or \
               ('sql_manager' in import_stmt and 'sql_manager' not in existing_imports):
                new_imports.append(import_stmt)
        
        if new_imports:
            lines.insert(import_section_end, '\n'.join(new_imports))
        
        return '\n'.join(lines)
    
    def apply_patterns(self, content: str, patterns: List[Tuple[str, str]]) -> str:
        """应用迁移模式"""
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL)
        return content
    
    def add_fallback_mechanism(self, content: str, file_path: str) -> str:
        """添加降级处理机制"""
        if 'query_executor' in content and 'get_query_executor' in content:
            fallback_code = '''
# 降级处理机制
try:
    query_executor = get_query_executor()
except Exception as e:
    print(f"警告：统一查询接口不可用，使用传统方式: {e}")
    query_executor = None
'''
            # 在第一个query_executor使用前添加降级处理
            content = content.replace(
                'query_executor = get_query_executor()',
                fallback_code.strip()
            )
        
        return content
    
    def migrate_file_final_sql_migration(self, file_path: str) -> Tuple[bool, str]:
        """迁移单个文件"""
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        try:
            # 备份文件
            backup_path = self.backup_file(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已经迁移
            if 'from db.query_executor import get_query_executor' in content:
from db.sql_manager import SQLManager, QueryType
                return True, f"文件已迁移: {file_path}"
            
            # 确定文件类型和模板
            file_type = self.detect_file_type(file_path)
            template = self.migration_templates[file_type]
            
            # 应用迁移
            original_content = content
            
            # 1. 添加导入
            content = self.add_imports(content, template['imports'])
            
            # 2. 应用模式替换
            content = self.apply_patterns(content, template['patterns'])
            
            # 3. 添加降级处理
            content = self.add_fallback_mechanism(content, file_path)
            
            # 检查是否有实际变化
            if content != original_content:
                # 写入迁移后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, f"成功迁移: {file_path}"
            else:
                return True, f"无需迁移: {file_path}"
                
        except Exception as e:
            return False, f"迁移失败 {file_path}: {str(e)}"
    
    def migrate_by_priority(self) -> Dict[str, List[Tuple[str, bool, str]]]:
        """按优先级迁移文件"""
        results = {}
        
        for priority, files in self.priority_groups.items():
            print(f"\n开始迁移 {priority} 优先级文件...")
            priority_results = []
            
            for file_path in files:
                success, message = self.migrate_file(file_path)
                priority_results.append((file_path, success, message))
                print(f"  {'✅' if success else '❌'} {message}")
            
            results[priority] = priority_results
        
        return results
    
    def generate_migration_report_final_sql_migration(self, results: Dict[str, List[Tuple[str, bool, str]]]) -> str:
        """生成迁移报告"""
        report = []
        report.append("# 最终SQL迁移报告")
        report.append(f"生成时间: {os.popen('date').read().strip()}")
        report.append("")
        
        total_files = 0
        total_success = 0
        
        for priority, file_results in results.items():
            report.append(f"## {priority} 优先级文件")
            report.append("")
            
            success_count = sum(1 for _, success, _ in file_results if success)
            total_count = len(file_results)
            
            report.append(f"- 总文件数: {total_count}")
            report.append(f"- 成功迁移: {success_count}")
            report.append(f"- 失败数: {total_count - success_count}")
            report.append(f"- 成功率: {success_count/total_count*100:.1f}%")
            report.append("")
            
            report.append("### 详细结果")
            for file_path, success, message in file_results:
                status = "✅" if success else "❌"
                report.append(f"- {status} {file_path}: {message}")
            report.append("")
            
            total_files += total_count
            total_success += success_count
        
        # 总结
        report.append("## 总体统计")
        report.append(f"- 总文件数: {total_files}")
        report.append(f"- 成功迁移: {total_success}")
        report.append(f"- 失败数: {total_files - total_success}")
        report.append(f"- 总体成功率: {total_success/total_files*100:.1f}%")
        
        return '\n'.join(report)
    
    def run_migration(self):
        """运行迁移"""
        print("开始最终SQL迁移...")
        print(f"目标文件数: {len(self.remaining_files)}")
        
        # 按优先级迁移
        results = self.migrate_by_priority()
        
        # 生成报告
        report = self.generate_migration_report(results)
        
        # 保存报告
        report_path = "final_sql_migration_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n迁移完成！报告已保存到: {report_path}")
        
        return results

def main_final_sql_migration():
    """主函数"""
    migrator = FinalSQLMigrator()
    results = migrator.run_migration()
    
    # 打印简要统计
    total_files = sum(len(file_results) for file_results in results.values())
    total_success = sum(sum(1 for _, success, _ in file_results if success) 
                       for file_results in results.values())
    
    print(f"\n=== 最终迁移统计 ===")
    print(f"总文件数: {total_files}")
    print(f"成功迁移: {total_success}")
    print(f"失败数: {total_files - total_success}")
    print(f"成功率: {total_success/total_files*100:.1f}%")

if __name__ == "__main__":
    main_final_sql_migration() 