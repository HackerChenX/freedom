#!/usr/bin/env python3
"""
批量SQL迁移工具

自动识别和替换常见的SQL查询模式，将其迁移到统一的SQL管理系统。
"""

import os
import re
import sys
from typing import Dict, List, Tuple, Set
from pathlib import Path

class BatchSQLMigrator:
    """批量SQL迁移器"""
    
    def __init__(self):
        self.migration_patterns = {
            # 股票数据查询模式
            'stock_data_query': {
                'pattern': r'SELECT\s+.*?\s+FROM\s+stock_info\s+WHERE.*?code\s*=.*?AND.*?date.*?',
                'replacement': 'query_executor.get_stock_data(code={code}, start_date={start_date}, end_date={end_date}, level={level})',
                'imports': ['from db.query_executor import get_query_executor']
            },
            
            # 股票计数查询模式
            'stock_count_query': {
                'pattern': r'SELECT\s+COUNT\(\*\)\s+.*?FROM\s+stock_info',
                'replacement': 'query_executor.get_stock_count()',
                'imports': ['from db.query_executor import get_query_executor']
            },
            
            # 股票列表查询模式
            'stock_list_query': {
                'pattern': r'SELECT\s+DISTINCT\s+code\s+FROM\s+stock_info',
                'replacement': 'query_executor.get_stock_list()',
                'imports': ['from db.query_executor import get_query_executor']
            },
            
            # 直接连接查询模式
            'direct_connection': {
                'pattern': r'with\s+.*?\.get_connection\(\)\s+as\s+conn:.*?conn\.query_dataframe\(',
                'replacement': 'query_executor.execute_query(QueryType.CUSTOM, params)',
                'imports': ['from db.query_executor import get_query_executor', 'from db.sql_manager import QueryType']
            }
        }
        
        self.files_to_migrate = []
        self.migration_stats = {
            'files_processed': 0,
            'patterns_replaced': 0,
            'errors': []
        }
    
    def scan_files(self, directory: str = ".") -> List[str]:
        """扫描需要迁移的文件"""
        files_with_sql = []
        
        for root, dirs, files in os.walk(directory):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in {'venv', '__pycache__', '.git', 'logs', 'cache', 'tmp', 'metadata', 'store', '.venv'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if self._file_contains_sql(file_path):
                        files_with_sql.append(file_path)
        
        self.files_to_migrate = files_with_sql
        return files_with_sql
    
    def _file_contains_sql(self, file_path: str) -> bool:
        """检查文件是否包含SQL查询"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 检查是否包含SQL关键字
            sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM stock_info', 'query_dataframe']
            return any(keyword in content.upper() for keyword in sql_keywords)
        except Exception:
            return False
    
    def migrate_file(self, file_path: str) -> Dict[str, any]:
        """迁移单个文件"""
        result = {
            'file': file_path,
            'success': False,
            'changes': [],
            'errors': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            modified_content = original_content
            imports_to_add = set()
            
            # 应用各种迁移模式
            for pattern_name, pattern_info in self.migration_patterns.items():
                matches = re.finditer(pattern_info['pattern'], modified_content, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    # 记录变更
                    result['changes'].append({
                        'pattern': pattern_name,
                        'original': match.group(0)[:100] + '...' if len(match.group(0)) > 100 else match.group(0),
                        'replacement': pattern_info['replacement']
                    })
                    
                    # 添加需要的导入
                    imports_to_add.update(pattern_info['imports'])
            
            # 添加导入语句
            if imports_to_add:
                modified_content = self._add_imports(modified_content, imports_to_add)
            
            # 应用具体的替换规则
            modified_content = self._apply_specific_replacements(modified_content)
            
            # 如果有变更，写回文件
            if modified_content != original_content:
                # 创建备份
                backup_path = file_path + '.backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # 写入修改后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                result['success'] = True
                result['backup_created'] = backup_path
                self.migration_stats['patterns_replaced'] += len(result['changes'])
            else:
                result['success'] = True
                result['message'] = 'No changes needed'
            
        except Exception as e:
            result['errors'].append(str(e))
            self.migration_stats['errors'].append(f"{file_path}: {e}")
        
        self.migration_stats['files_processed'] += 1
        return result
    
    def _add_imports(self, content: str, imports: Set[str]) -> str:
        """添加导入语句"""
        lines = content.split('\n')
        import_section_end = 0
        
        # 找到导入区域的结束位置
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')) or line.strip() == '':
                import_section_end = i
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # 检查哪些导入是新的
        existing_imports = set()
        for line in lines[:import_section_end + 5]:  # 检查前几行
            if 'from db.query_executor import' in line:
                existing_imports.add('from db.query_executor import get_query_executor')
            if 'from db.sql_manager import' in line:
                existing_imports.add('from db.sql_manager import QueryType')
from db.sql_manager import SQLManager, QueryType
        
        new_imports = imports - existing_imports
        
        if new_imports:
            # 在适当位置插入新的导入
            insert_position = import_section_end + 1
            for imp in sorted(new_imports):
                lines.insert(insert_position, imp)
                insert_position += 1
        
        return '\n'.join(lines)
    
    def _apply_specific_replacements(self, content: str) -> str:
        """应用具体的替换规则"""
        
        # 替换直接SQL查询为统一接口调用
        replacements = [
            # 股票数据查询
            (
                r'conn\.query_dataframe\(\s*["\']SELECT.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?["\'].*?\)',
                'query_executor.get_stock_data(code=stock_code, start_date=start_date, end_date=end_date, level="日线")'
            ),
            
            # 股票计数查询
            (
                r'conn\.query_dataframe\(\s*["\']SELECT\s+COUNT\(\*\).*?FROM\s+stock_info.*?["\'].*?\)',
                'query_executor.get_stock_count()'
            ),
            
            # 连接池获取
            (
                r'with\s+.*?\.get_connection\(\)\s+as\s+conn:',
                '# 使用统一查询接口替代直接连接\n        try:'
            ),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL)
        
        return content
    
    def migrate_all(self, directory: str = ".") -> Dict[str, any]:
        """批量迁移所有文件"""
        print("开始批量SQL迁移...")
        
        # 扫描文件
        files = self.scan_files(directory)
        print(f"发现 {len(files)} 个包含SQL的文件")
        
        # 迁移文件
        results = []
        for file_path in files:
            print(f"迁移文件: {file_path}")
            result = self.migrate_file(file_path)
            results.append(result)
            
            if result['success']:
                if result['changes']:
                    print(f"  ✅ 成功迁移 {len(result['changes'])} 个SQL模式")
                else:
                    print(f"  ✅ 无需更改")
            else:
                print(f"  ❌ 迁移失败: {result['errors']}")
        
        # 生成总结报告
        summary = {
            'total_files': len(files),
            'successful_migrations': len([r for r in results if r['success']]),
            'total_patterns_replaced': self.migration_stats['patterns_replaced'],
            'errors': self.migration_stats['errors'],
            'results': results
        }
        
        print(f"\n迁移总结:")
        print(f"  总文件数: {summary['total_files']}")
        print(f"  成功迁移: {summary['successful_migrations']}")
        print(f"  替换模式数: {summary['total_patterns_replaced']}")
        print(f"  错误数: {len(summary['errors'])}")
        
        return summary
    
    def create_migration_report(self, summary: Dict[str, any], output_file: str = "sql_migration_report.md"):
        """生成迁移报告"""
        report = f"""# SQL迁移报告

## 迁移概述

- **总文件数**: {summary['total_files']}
- **成功迁移**: {summary['successful_migrations']}
- **替换模式数**: {summary['total_patterns_replaced']}
- **错误数**: {len(summary['errors'])}
- **生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 迁移详情

"""
        
        for result in summary['results']:
            if result['changes']:
                report += f"### {result['file']}\n"
                report += f"- 迁移状态: {'✅ 成功' if result['success'] else '❌ 失败'}\n"
                report += f"- 变更数量: {len(result['changes'])}\n\n"
                
                for change in result['changes']:
                    report += f"**{change['pattern']}**:\n"
                    report += f"- 原始: `{change['original']}`\n"
                    report += f"- 替换: `{change['replacement']}`\n\n"
        
        if summary['errors']:
            report += "## 错误信息\n\n"
            for error in summary['errors']:
                report += f"- {error}\n"
        
        report += """
## 迁移后的优势

1. **统一查询接口**: 所有SQL查询通过统一的接口管理
2. **参数验证**: 自动验证查询参数的有效性
3. **错误处理**: 统一的错误处理和日志记录
4. **性能优化**: 查询缓存和连接池管理
5. **维护性**: 更好的代码组织和模块化

## 下一步

1. 测试迁移后的功能
2. 更新相关文档
3. 进行性能测试
4. 清理备份文件（如果测试通过）
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"迁移报告已生成: {output_file}")
        return output_file

def main_batch_sql_migration_tool():
    """主函数"""
    print("=== 批量SQL迁移工具 ===")
    
    migrator = BatchSQLMigrator()
    
    # 执行批量迁移
    summary = migrator.migrate_all()
    
    # 生成报告
    migrator.create_migration_report(summary)
    
    print("\n批量迁移完成!")

if __name__ == "__main__":
    main_batch_sql_migration_tool() 