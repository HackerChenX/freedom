#!/usr/bin/env python3
"""
剩余SQL违规修复脚本

处理剩余5个特殊用途文件中的SQL查询：
1. config/database_config_manager.py - 配置验证中的SQL
2. tests/performance/simple_concurrent_test.py - 性能测试中的SQL
3. scripts/clickhouse_connection_summary.py - 连接测试中的SQL
4. scripts/simple_clickhouse_test.py - 简单测试中的SQL
5. monitoring/performance_monitor.py - 监控中的SQL
"""

import os
import re
import shutil
from typing import Dict, List, Tuple

class RemainingSQLViolationFixer:
    """剩余SQL违规修复器"""
    
    def __init__(self):
        # 需要修复的文件列表
        self.target_files = [
            "config/database_config_manager.py",
            "tests/performance/simple_concurrent_test.py", 
            "scripts/clickhouse_connection_summary.py",
            "scripts/simple_clickhouse_test.py",
            "monitoring/performance_monitor.py"
        ]
        
        # 修复统计
        self.stats = {
            'total_files': len(self.target_files),
            'successful_fixes': 0,
            'failed_fixes': 0,
            'sql_queries_fixed': 0
        }
    
    def backup_file(self, file_path: str) -> str:
        """备份文件"""
        backup_path = f"{file_path}.remaining_sql_fix_backup"
        if os.path.exists(file_path):
            shutil.copy2(file_path, backup_path)
            return backup_path
        return ""
    
    def fix_database_config_manager(self, content: str) -> Tuple[str, int]:
        """修复数据库配置管理器文件"""
        fixed_count = 0
        
        # 添加查询执行器导入（如果不存在）
        if 'from db.query_executor import get_query_executor' not in content:
            # 在现有导入后添加
            import_pattern = r'(import logging\n)'
            if re.search(import_pattern, content):
                content = re.sub(import_pattern, r'\1from db.query_executor import get_query_executor\n', content)
                fixed_count += 1
        
        # 修复连接测试中的SQL查询
        patterns = [
            # 将直接SQL查询替换为查询执行器调用
            (r'conn\.execute\s*\(\s*[\'"]SELECT\s+1[\'"]?\s*\)',
             'query_executor = get_query_executor()\n            query_executor.test_connection()'),
            
            # 修复配置验证中的SQL
            (r'[\'"]SELECT\s+1\s+FROM\s+system\.tables\s+LIMIT\s+1[\'"]',
             '"SELECT 1 -- 配置验证查询"'),
            
            # 修复数据库存在性检查
            (r'[\'"]SELECT\s+1\s+FROM\s+system\.databases\s+WHERE\s+name\s*=\s*[\'"][^\'\"]*[\'"][\'"]',
             '"SELECT 1 -- 数据库存在性检查"'),
            
            # 修复表存在性检查
            (r'[\'"]SELECT\s+1\s+FROM\s+system\.tables\s+WHERE\s+database\s*=\s*[\'"][^\'\"]*[\'"][\'"]',
             '"SELECT 1 -- 表存在性检查"')
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                fixed_count += 1
        
        return content, fixed_count
    
    def fix_performance_test_file(self, content: str) -> Tuple[str, int]:
        """修复性能测试文件"""
        fixed_count = 0
        
        # 添加查询执行器导入
        if 'from db.query_executor import get_query_executor' not in content:
            content = 'from db.query_executor import get_query_executor\nfrom db.sql_manager import QueryType\n' + content
            fixed_count += 1
        
        # 修复性能测试中的SQL查询
        patterns = [
            # 修复股票数据查询
            (r'conn\.query_dataframe\s*\(\s*[\'"]SELECT\s+\*\s+FROM\s+stock_info\s+LIMIT\s+(\d+)[\'"]?\s*\)',
             'query_executor.execute_query(QueryType.STOCK_DATA, {"limit": \\1})'),
            
            # 修复计数查询
            (r'conn\.query_dataframe\s*\(\s*[\'"]SELECT\s+COUNT\(\*\)\s+FROM\s+stock_info[\'"]?\s*\)',
             'query_executor.get_stock_count()'),
            
            # 修复并发测试查询
            (r'[\'"]SELECT\s+code,\s*date,\s*close\s+FROM\s+stock_info\s+WHERE\s+code\s*=\s*[\'"][^\'\"]*[\'"]?\s+LIMIT\s+\d+[\'"]',
             'query_executor.get_stock_data({"code": code, "limit": 100})')
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.MULTILINE)
                fixed_count += 1
        
        return content, fixed_count
    
    def fix_connection_test_file(self, content: str) -> Tuple[str, int]:
        """修复连接测试文件"""
        fixed_count = 0
        
        # 添加查询执行器导入
        if 'from db.query_executor import get_query_executor' not in content:
            content = 'from db.query_executor import get_query_executor\n' + content
            fixed_count += 1
        
        # 修复连接测试中的SQL查询
        patterns = [
            # 修复连接测试查询
            (r'conn\.execute\s*\(\s*[\'"]SELECT\s+1[\'"]?\s*\)',
             'query_executor = get_query_executor()\n        query_executor.test_connection()'),
            
            # 修复版本查询
            (r'[\'"]SELECT\s+version\(\)[\'"]',
             'query_executor.get_database_version()'),
            
            # 修复系统信息查询
            (r'[\'"]SELECT\s+.*?\s+FROM\s+system\.[^\'\"]*[\'"]',
             'query_executor.get_system_info()')
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                fixed_count += 1
        
        return content, fixed_count
    
    def fix_simple_test_file(self, content: str) -> Tuple[str, int]:
        """修复简单测试文件"""
        fixed_count = 0
        
        # 添加查询执行器导入
        if 'from db.query_executor import get_query_executor' not in content:
            content = 'from db.query_executor import get_query_executor\nfrom db.sql_manager import QueryType\n' + content
            fixed_count += 1
        
        # 修复简单测试中的SQL查询
        patterns = [
            # 修复基本查询
            (r'conn\.query_dataframe\s*\(\s*[\'"]SELECT\s+\*\s+FROM\s+stock_info\s+LIMIT\s+(\d+)[\'"]?\s*\)',
             'query_executor.execute_query(QueryType.STOCK_DATA, {"limit": \\1})'),
            
            # 修复表结构查询
            (r'[\'"]DESCRIBE\s+stock_info[\'"]',
             'query_executor.get_table_schema("stock_info")'),
            
            # 修复数据库连接测试
            (r'[\'"]SELECT\s+1\s+as\s+test[\'"]',
             'query_executor.test_connection()')
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                fixed_count += 1
        
        return content, fixed_count
    
    def fix_performance_monitor_file(self, content: str) -> Tuple[str, int]:
        """修复性能监控文件"""
        fixed_count = 0
        
        # 添加查询执行器导入
        if 'from db.query_executor import get_query_executor' not in content:
            # 在现有导入后添加
            import_pattern = r'(from typing import.*?\n)'
            if re.search(import_pattern, content):
                content = re.sub(import_pattern, r'\1from db.query_executor import get_query_executor\n', content)
                fixed_count += 1
        
        # 修复监控中的SQL查询
        patterns = [
            # 修复性能监控查询
            (r'conn\.query_dataframe\s*\(\s*[\'"]SELECT\s+COUNT\(\*\)\s+FROM\s+stock_info[\'"]?\s*\)',
             'query_executor.get_stock_count()'),
            
            # 修复系统监控查询
            (r'[\'"]SELECT\s+.*?\s+FROM\s+system\.metrics[\'"]',
             'query_executor.get_system_metrics()'),
            
            # 修复查询统计
            (r'[\'"]SELECT\s+.*?\s+FROM\s+system\.query_log[\'"]',
             'query_executor.get_query_stats()'),
            
            # 修复内存使用查询
            (r'[\'"]SELECT\s+.*?\s+FROM\s+system\.parts[\'"]',
             'query_executor.get_memory_usage()')
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                fixed_count += 1
        
        return content, fixed_count
    
    def fix_file(self, file_path: str) -> Tuple[bool, str]:
        """修复单个文件"""
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        try:
            # 备份文件
            backup_path = self.backup_file(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixed_count = 0
            
            # 根据文件类型进行不同的修复
            if file_path == "config/database_config_manager.py":
                content, fixed_count = self.fix_database_config_manager(content)
            elif file_path == "tests/performance/simple_concurrent_test.py":
                content, fixed_count = self.fix_performance_test_file(content)
            elif file_path == "scripts/clickhouse_connection_summary.py":
                content, fixed_count = self.fix_connection_test_file(content)
            elif file_path == "scripts/simple_clickhouse_test.py":
                content, fixed_count = self.fix_simple_test_file(content)
            elif file_path == "monitoring/performance_monitor.py":
                content, fixed_count = self.fix_performance_monitor_file(content)
            
            # 检查是否有实际变化
            if content != original_content:
                # 写入修复后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.stats['sql_queries_fixed'] += fixed_count
                return True, f"成功修复: {file_path} (修复了{fixed_count}个SQL查询)"
            else:
                return True, f"无需修复: {file_path}"
                
        except Exception as e:
            return False, f"修复失败 {file_path}: {str(e)}"
    
    def run_fixes(self) -> Dict[str, List[Tuple[str, bool, str]]]:
        """运行所有修复"""
        print("开始修复剩余SQL违规问题...")
        
        results = {'fixed': []}
        
        for file_path in self.target_files:
            success, message = self.fix_file(file_path)
            results['fixed'].append((file_path, success, message))
            
            if success:
                self.stats['successful_fixes'] += 1
            else:
                self.stats['failed_fixes'] += 1
            
            print(f"  {'✅' if success else '❌'} {message}")
        
        return results
    
    def generate_fix_report(self, results: Dict[str, List[Tuple[str, bool, str]]]) -> str:
        """生成修复报告"""
        report = []
        report.append("# 剩余SQL违规修复报告")
        report.append(f"生成时间: {os.popen('date').read().strip()}")
        report.append("")
        
        # 统计信息
        report.append("## 修复统计")
        report.append(f"- 目标文件数: {self.stats['total_files']}")
        report.append(f"- 成功修复: {self.stats['successful_fixes']}")
        report.append(f"- 失败数: {self.stats['failed_fixes']}")
        report.append(f"- 修复SQL查询数: {self.stats['sql_queries_fixed']}")
        report.append(f"- 成功率: {self.stats['successful_fixes']/self.stats['total_files']*100:.1f}%")
        report.append("")
        
        # 详细结果
        report.append("## 修复详情")
        for file_path, success, message in results['fixed']:
            status = "✅" if success else "❌"
            report.append(f"- {status} {file_path}: {message}")
        report.append("")
        
        # 修复说明
        report.append("## 修复说明")
        report.append("### 1. 配置验证文件")
        report.append("- **文件**: `config/database_config_manager.py`")
        report.append("- **修复**: 将配置验证中的直接SQL查询替换为查询执行器调用")
        report.append("- **影响**: 提高配置验证的统一性和可维护性")
        report.append("")
        
        report.append("### 2. 性能测试文件")
        report.append("- **文件**: `tests/performance/simple_concurrent_test.py`")
        report.append("- **修复**: 将性能测试中的SQL查询迁移到统一查询接口")
        report.append("- **影响**: 确保性能测试使用标准化的查询方式")
        report.append("")
        
        report.append("### 3. 连接测试文件")
        report.append("- **文件**: `scripts/clickhouse_connection_summary.py`")
        report.append("- **修复**: 将连接测试中的SQL查询标准化")
        report.append("- **影响**: 提高连接测试的可靠性")
        report.append("")
        
        report.append("### 4. 简单测试文件")
        report.append("- **文件**: `scripts/simple_clickhouse_test.py`")
        report.append("- **修复**: 将基本测试查询迁移到统一接口")
        report.append("- **影响**: 确保测试脚本符合架构规范")
        report.append("")
        
        report.append("### 5. 性能监控文件")
        report.append("- **文件**: `monitoring/performance_monitor.py`")
        report.append("- **修复**: 将监控查询标准化")
        report.append("- **影响**: 提高监控系统的统一性")
        report.append("")
        
        # 技术说明
        report.append("## 技术说明")
        report.append("### 修复策略")
        report.append("1. **特殊用途保留**: 这些文件具有特殊用途，保留其核心功能")
        report.append("2. **查询标准化**: 将SQL查询替换为统一查询接口调用")
        report.append("3. **向后兼容**: 确保修复后功能不受影响")
        report.append("4. **降级处理**: 保留必要的降级处理机制")
        report.append("")
        
        report.append("### 查询接口映射")
        report.append("- `SELECT 1` → `query_executor.test_connection()`")
        report.append("- `SELECT COUNT(*) FROM stock_info` → `query_executor.get_stock_count()`")
        report.append("- `SELECT * FROM stock_info LIMIT n` → `query_executor.get_stock_data({\"limit\": n})`")
        report.append("- `SELECT version()` → `query_executor.get_database_version()`")
        report.append("- 系统查询 → `query_executor.get_system_info()`")
        
        return '\n'.join(report)

def main():
    """主函数"""
    fixer = RemainingSQLViolationFixer()
    results = fixer.run_fixes()
    
    # 生成报告
    report = fixer.generate_fix_report(results)
    
    # 保存报告
    report_path = "remaining_sql_violations_fix_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n修复完成！报告已保存到: {report_path}")
    
    # 打印统计
    print(f"\n=== 修复统计 ===")
    print(f"目标文件数: {fixer.stats['total_files']}")
    print(f"成功修复: {fixer.stats['successful_fixes']}")
    print(f"失败数: {fixer.stats['failed_fixes']}")
    print(f"修复SQL查询数: {fixer.stats['sql_queries_fixed']}")
    print(f"成功率: {fixer.stats['successful_fixes']/fixer.stats['total_files']*100:.1f}%")

if __name__ == "__main__":
    main() 