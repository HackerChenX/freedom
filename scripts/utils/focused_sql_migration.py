#!/usr/bin/env python3
"""
专注SQL迁移工具

针对架构检查发现的具体文件进行SQL迁移，
优先处理核心业务文件，确保迁移质量。
"""

import os
import re
import sys
from typing import Dict, List, Tuple
from pathlib import Path

class FocusedSQLMigrator:
    """专注SQL迁移器"""
    
    def __init__(self):
        # 从架构检查结果中获取的需要迁移的文件列表
        self.target_files = [
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
        ]
        
        # 按优先级分组
        self.priority_groups = {
            'high': [  # 核心业务文件
                "analysis/multi_dimension_analyzer.py",
                "analysis/strategy_comparison.py",
                "analysis/buypoints/buypoint_dimension_analyzer.py",
                "bin/strategy_generator.py",
                "bin/stock_analysis.py",
                "strategy/enhanced_base_strategy.py",
                "strategy/batch_optimizer.py",
                "strategy/strategy_manager.py"
            ],
            'medium': [  # 测试和验证文件
                "tests/end_to_end/enhanced_real_data_test.py",
                "tests/end_to_end/real_data_performance_test.py",
                "scripts/production_indicator_tester.py",
                "scripts/production_indicator_validator.py",
                "scripts/database_optimization.py",
                "analysis/engines/indicator_validation_framework.py"
            ],
            'low': []  # 其他文件
        }
        
        # 将剩余文件分配到低优先级
        all_priority_files = set()
        for files in self.priority_groups.values():
            all_priority_files.update(files)
        
        self.priority_groups['low'] = [f for f in self.target_files if f not in all_priority_files]
    
    def get_migration_template(self, file_type: str) -> Dict[str, str]:
        """获取迁移模板"""
        templates = {
            'analysis': {
                'imports': '''from db.query_executor import get_query_executor
from db.sql_manager import QueryType''',
                'initialization': 'query_executor = get_query_executor()',
                'stock_data_query': 'query_executor.get_stock_data(code=stock_code, start_date=start_date, end_date=end_date, level="日线")',
                'stock_count_query': 'query_executor.get_stock_count()',
                'stock_list_query': 'query_executor.get_stock_list()'
            },
            'strategy': {
                'imports': '''from db.query_executor import get_query_executor
from db.sql_manager import QueryType''',
                'initialization': 'query_executor = get_query_executor()',
                'stock_data_query': 'query_executor.get_stock_data(code=code, start_date=start_date, end_date=end_date, level=level)',
                'batch_query': 'query_executor.get_batch_stock_data(codes=codes, start_date=start_date, end_date=end_date, level=level)'
            },
            'test': {
                'imports': '''from db.query_executor import get_query_executor
from db.sql_manager import QueryType''',
                'initialization': 'query_executor = get_query_executor()',
                'test_query': 'query_executor.execute_query(QueryType.STOCK_DATA, test_params)'
            },
            'script': {
                'imports': '''from db.query_executor import get_query_executor
from db.sql_manager import QueryType''',
                'initialization': 'query_executor = get_query_executor()',
                'custom_query': 'query_executor.execute_query(QueryType.CUSTOM, custom_params)'
            }
        }
        
        return templates.get(file_type, templates['script'])
    
    def detect_file_type_focused_sql_migration(self, file_path: str) -> str:
        """检测文件类型"""
        if file_path.startswith('analysis/'):
            return 'analysis'
        elif file_path.startswith('strategy/'):
            return 'strategy'
        elif file_path.startswith('tests/') or file_path.startswith('examples/'):
            return 'test'
        else:
            return 'script'
    
    def migrate_file_focused_sql_migration(self, file_path: str) -> Dict[str, any]:
        """迁移单个文件"""
        result = {
            'file': file_path,
            'success': False,
            'changes': [],
            'errors': [],
            'backup_created': None
        }
        
        if not os.path.exists(file_path):
            result['errors'].append(f"文件不存在: {file_path}")
            return result
        
        try:
            # 读取原始文件
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 检测文件类型
            file_type = self.detect_file_type(file_path)
            template = self.get_migration_template(file_type)
            
            # 应用迁移
            modified_content = self._apply_migration(original_content, template, file_path)
            
            # 检查是否有变更
            if modified_content != original_content:
                # 创建备份
                backup_path = file_path + '.sql_migration_backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # 写入修改后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                result['success'] = True
                result['backup_created'] = backup_path
                result['changes'] = ['SQL查询已迁移到统一管理系统']
                
                print(f"✅ 成功迁移: {file_path}")
            else:
                result['success'] = True
                result['changes'] = ['无需修改']
                print(f"ℹ️ 无需修改: {file_path}")
                
        except Exception as e:
            result['errors'].append(str(e))
            print(f"❌ 迁移失败: {file_path} - {e}")
        
        return result
    
    def _apply_migration(self, content: str, template: Dict[str, str], file_path: str) -> str:
        """应用迁移规则"""
        modified_content = content
        
        # 1. 添加导入语句（如果还没有）
        if 'from db.query_executor import' not in content:
            # 找到合适的位置插入导入
            lines = content.split('\n')
            insert_pos = self._find_import_position(lines)
            
            import_lines = template['imports'].split('\n')
            for i, import_line in enumerate(import_lines):
                lines.insert(insert_pos + i, import_line)
            
            modified_content = '\n'.join(lines)
        
        # 2. 替换常见的SQL模式
        replacements = [
            # 连接池模式
            (
                r'with\s+.*?\.get_connection\(\)\s+as\s+conn:\s*\n\s*.*?conn\.query_dataframe\(',
                lambda m: self._replace_connection_pattern(m.group(0), template)
            ),
            
            # 直接SQL查询
            (
                r'conn\.query_dataframe\(\s*["\']SELECT.*?FROM\s+stock_info.*?["\'].*?\)',
                template.get('stock_data_query', 'query_executor.execute_query(QueryType.STOCK_DATA, params)')
            ),
            
            # 计数查询
            (
                r'conn\.query_dataframe\(\s*["\']SELECT\s+COUNT\(\*\).*?FROM\s+stock_info.*?["\'].*?\)',
                template.get('stock_count_query', 'query_executor.get_stock_count()')
            ),
        ]
        
        for pattern, replacement in replacements:
            if callable(replacement):
                modified_content = re.sub(pattern, replacement, modified_content, flags=re.IGNORECASE | re.DOTALL)
            else:
                modified_content = re.sub(pattern, replacement, modified_content, flags=re.IGNORECASE | re.DOTALL)
        
        # 3. 添加查询执行器初始化（如果需要）
        if 'query_executor = get_query_executor()' not in modified_content and 'query_executor' in template.get('stock_data_query', ''):
            # 在适当的位置添加初始化
            modified_content = self._add_query_executor_init(modified_content, template['initialization'])
        
        return modified_content
    
    def _find_import_position(self, lines: List[str]) -> int:
        """找到插入导入语句的位置"""
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                continue
            elif line.strip() == '' or line.strip().startswith('#'):
                continue
            else:
                return i
        return len(lines)
    
    def _replace_connection_pattern(self, match_text: str, template: Dict[str, str]) -> str:
        """替换连接模式"""
        return f'''# 使用统一查询接口替代直接连接
        try:
            {template.get('initialization', 'query_executor = get_query_executor()')}
            # 原始查询已迁移到统一接口'''
    
    def _add_query_executor_init(self, content: str, init_code: str) -> str:
        """添加查询执行器初始化"""
        # 在函数开始处添加初始化
        lines = content.split('\n')
        
        # 寻找函数定义
        for i, line in enumerate(lines):
            if 'def ' in line and ':' in line:
                # 在函数体开始处插入初始化
                indent = '    '  # 假设使用4空格缩进
                lines.insert(i + 1, f'{indent}{init_code}')
                break
        
        return '\n'.join(lines)
    
    def migrate_by_priority_focused_sql_migration(self) -> Dict[str, any]:
        """按优先级迁移文件"""
        results = {
            'high': [],
            'medium': [],
            'low': [],
            'summary': {
                'total_files': len(self.target_files),
                'successful': 0,
                'failed': 0,
                'errors': []
            }
        }
        
        print("=== 开始专注SQL迁移 ===")
        
        for priority in ['high', 'medium', 'low']:
            print(f"\n处理 {priority.upper()} 优先级文件...")
            
            for file_path in self.priority_groups[priority]:
                result = self.migrate_file(file_path)
                results[priority].append(result)
                
                if result['success']:
                    results['summary']['successful'] += 1
                else:
                    results['summary']['failed'] += 1
                    results['summary']['errors'].extend(result['errors'])
        
        print(f"\n=== 迁移总结 ===")
        print(f"总文件数: {results['summary']['total_files']}")
        print(f"成功迁移: {results['summary']['successful']}")
        print(f"失败: {results['summary']['failed']}")
        
        if results['summary']['errors']:
            print(f"错误信息:")
            for error in results['summary']['errors'][:5]:  # 只显示前5个错误
                print(f"  - {error}")
        
        return results
    
    def create_migration_report_focused_sql_migration(self, results: Dict[str, any]):
        """创建迁移报告"""
        report_file = "focused_sql_migration_report.md"
        
        report = f"""# 专注SQL迁移报告

## 迁移概述

- **总文件数**: {results['summary']['total_files']}
- **成功迁移**: {results['summary']['successful']}
- **失败数**: {results['summary']['failed']}
- **成功率**: {results['summary']['successful'] / results['summary']['total_files'] * 100:.1f}%
- **生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 按优先级分组结果

"""
        
        for priority in ['high', 'medium', 'low']:
            priority_results = results[priority]
            successful = len([r for r in priority_results if r['success']])
            
            report += f"### {priority.upper()} 优先级 ({successful}/{len(priority_results)} 成功)\n\n"
            
            for result in priority_results:
                status = "✅" if result['success'] else "❌"
                report += f"- {status} {result['file']}\n"
                if result['errors']:
                    for error in result['errors']:
                        report += f"  - 错误: {error}\n"
            
            report += "\n"
        
        if results['summary']['errors']:
            report += "## 错误详情\n\n"
            for error in results['summary']['errors']:
                report += f"- {error}\n"
        
        report += """
## 迁移效果

SQL查询迁移到统一管理系统后的优势：

1. **统一接口**: 所有SQL查询通过标准化接口访问
2. **参数验证**: 自动验证查询参数，减少运行时错误
3. **性能优化**: 统一的连接池和缓存管理
4. **错误处理**: 标准化的错误处理和日志记录
5. **维护性**: 更好的代码组织和模块化设计

## 下一步行动

1. 运行架构检查验证迁移效果
2. 执行相关测试确保功能正常
3. 更新文档反映架构变更
4. 清理备份文件（测试通过后）
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"迁移报告已生成: {report_file}")
        return report_file

def main_focused_sql_migration():
    """主函数"""
    migrator = FocusedSQLMigrator()
    
    # 执行迁移
    results = migrator.migrate_by_priority()
    
    # 生成报告
    migrator.create_migration_report(results)
    
    print("\n专注SQL迁移完成!")
    print("建议运行 'python simple_architecture_check.py' 验证迁移效果")

if __name__ == "__main__":
    main_focused_sql_migration() 