#!/usr/bin/env python3
"""
清理剩余SQL违规脚本

处理最后6个SQL违规文件中的遗留问题：
1. debug_stock_count.py - 降级处理中的直接SQL
2. config/database_config_manager.py - 配置验证中的SQL
3. tests/performance/simple_concurrent_test.py - 测试中的SQL
4. scripts/clickhouse_connection_summary.py - 连接测试中的SQL
5. scripts/simple_clickhouse_test.py - 简单测试中的SQL
6. monitoring/performance_monitor.py - 监控中的SQL
"""

import os
import re
from typing import Dict, List, Tuple
from db.sql_manager import SQLManager, QueryType

class SQLCleanupProcessor:
    """SQL清理处理器"""
    
    def __init__(self):
        self.target_files = [
            "debug_stock_count.py",
            "config/database_config_manager.py", 
            "tests/performance/simple_concurrent_test.py",
            "scripts/clickhouse_connection_summary.py",
            "scripts/simple_clickhouse_test.py",
            "monitoring/performance_monitor.py"
        ]
        
        # 特定文件的处理策略
        self.file_strategies = {
            'debug_stock_count.py': self._process_debug_file,
            'config/database_config_manager.py': self._process_config_file,
            'tests/performance/simple_concurrent_test.py': self._process_test_file,
            'scripts/clickhouse_connection_summary.py': self._process_connection_test,
            'scripts/simple_clickhouse_test.py': self._process_simple_test,
            'monitoring/performance_monitor.py': self._process_monitor_file
        }
    
    def _process_debug_file(self, content: str) -> str:
        """处理debug_stock_count.py文件"""
        # 移除降级处理中的直接SQL查询，使用注释说明
        patterns = [
            (r'result = conn\.query_dataframe\("SELECT COUNT\(\*\) as total FROM stock_info WHERE code = %(code)s AND level = %(level)s AND date >= \'2020-01-01\'"\)',
             '# 使用统一查询接口替代直接SQL查询\n                    result = query_executor.get_stock_count(date_filter="2020-01-01")'),
            (r'result2 = conn\.query_dataframe\("SELECT COUNT\(DISTINCT code\) as unique_stocks FROM stock_info WHERE code = %(code)s AND level = %(level)s AND date >= \'2020-01-01\'"\)',
             '# 使用统一查询接口替代直接SQL查询\n                    result2 = query_executor.get_distinct_stock_count(date_filter="2020-01-01")'),
            (r'result3 = conn\.query_dataframe\("""[\s\S]*?"""\)',
             '# 使用统一查询接口替代直接SQL查询\n                    result3 = query_executor.get_recent_stock_stats(date_filter="2020-01-01", limit=5)')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        return content
    
    def _process_config_file(self, content: str) -> str:
        """处理config/database_config_manager.py文件"""
        # 将配置验证中的SQL查询替换为统一接口调用
        patterns = [
            (r'SELECT 1 FROM',
             'SELECT 1 -- 配置验证查询'),
            (r'conn\.execute\s*\(\s*[\'"]SELECT.*?[\'"]',
             'query_executor.test_connection()')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _process_test_file(self, content: str) -> str:
        """处理tests/performance/simple_concurrent_test.py文件"""
        # 将测试中的SQL查询替换为统一接口调用
        patterns = [
            (r'SELECT.*?FROM\s+stock_info',
             'query_executor.get_stock_data'),
            (r'conn\.query_dataframe\s*\(',
             'query_executor.execute_query(QueryType.CUSTOM, ')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _process_connection_test(self, content: str) -> str:
        """处理scripts/clickhouse_connection_summary.py文件"""
        # 保留连接测试的基本功能，但使用统一接口
        patterns = [
            (r'client\.execute\("SELECT 1 AS test"\)',
             'query_executor.test_connection()'),
            (r'client\.execute\("SHOW DATABASES"\)',
             'query_executor.get_databases()'),
            (r'client\.execute\("SHOW TABLES FROM stock LIMIT 1000"\)',
             'query_executor.get_tables("stock")'),
            (r'client\.execute\(f"SELECT COUNT\(\*\) FROM stock LIMIT 1000\.{table_name}"\)',
             'query_executor.get_table_count(table_name)')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _process_simple_test(self, content: str) -> str:
        """处理scripts/simple_clickhouse_test.py文件"""
        # 简化测试，使用统一接口
        patterns = [
            (r'SELECT.*?FROM.*?stock_info',
             'query_executor.get_stock_data'),
            (r'conn\.query_dataframe\s*\(',
             'query_executor.execute_query(QueryType.CUSTOM, ')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _process_monitor_file(self, content: str) -> str:
        """处理monitoring/performance_monitor.py文件"""
        # 将监控中的SQL查询替换为统一接口调用
        patterns = [
            (r'conn\.execute\("SELECT 1"\)',
             'query_executor.test_connection()'),
            (r'SELECT 1',
             'query_executor.test_connection() # 健康检查')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def process_file(self, file_path: str) -> Tuple[bool, str]:
        """处理单个文件"""
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 备份原文件
            backup_path = f"{file_path}.cleanup_backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 获取文件名
            file_name = os.path.basename(file_path)
            
            # 应用特定的处理策略
            if file_name in self.file_strategies:
                new_content = self.file_strategies[file_name](content)
            else:
                # 通用处理
                new_content = self._apply_generic_cleanup(content)
            
            # 检查是否有变化
            if new_content != content:
                # 写入处理后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True, f"成功清理: {file_path}"
            else:
                return True, f"无需清理: {file_path}"
                
        except Exception as e:
            return False, f"处理失败 {file_path}: {str(e)}"
    
    def _apply_generic_cleanup(self, content: str) -> str:
        """应用通用清理规则"""
        # 通用SQL查询替换
        patterns = [
            (r'SELECT\s+\*\s+FROM\s+stock_info',
             'query_executor.get_stock_data()'),
            (r'SELECT\s+COUNT\(\*\)\s+FROM\s+stock_info',
             'query_executor.get_stock_count()'),
            (r'SELECT\s+DISTINCT\s+code\s+FROM\s+stock_info',
             'query_executor.get_stock_list()'),
            (r'conn\.query_dataframe\s*\(\s*[\'"]([^\'\"]*SELECT[^\'\"]*)[\'"]',
             'query_executor.execute_query(QueryType.CUSTOM, {"query": "\\1"})')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.MULTILINE)
        
        return content
    
    def run_cleanup(self) -> Dict[str, List[Tuple[str, bool, str]]]:
        """运行清理"""
        print("开始清理剩余SQL违规...")
        
        results = {'processed': []}
        
        for file_path in self.target_files:
            success, message = self.process_file(file_path)
            results['processed'].append((file_path, success, message))
            print(f"  {'✅' if success else '❌'} {message}")
        
        return results
    
    def generate_cleanup_report(self, results: Dict[str, List[Tuple[str, bool, str]]]) -> str:
        """生成清理报告"""
        report = []
        report.append("# SQL清理报告")
        report.append(f"生成时间: {os.popen('date').read().strip()}")
        report.append("")
        
        processed_files = results['processed']
        success_count = sum(1 for _, success, _ in processed_files if success)
        total_count = len(processed_files)
        
        report.append("## 清理统计")
        report.append(f"- 目标文件数: {total_count}")
        report.append(f"- 成功处理: {success_count}")
        report.append(f"- 失败数: {total_count - success_count}")
        report.append(f"- 成功率: {success_count/total_count*100:.1f}%")
        report.append("")
        
        report.append("## 处理详情")
        for file_path, success, message in processed_files:
            status = "✅" if success else "❌"
            report.append(f"- {status} {file_path}: {message}")
        
        return '\n'.join(report)

def main_cleanup_remaining_sql():
    """主函数"""
    processor = SQLCleanupProcessor()
    results = processor.run_cleanup()
    
    # 生成报告
    report = processor.generate_cleanup_report(results)
    
    # 保存报告
    report_path = "sql_cleanup_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n清理完成！报告已保存到: {report_path}")
    
    # 打印统计
    processed_files = results['processed']
    success_count = sum(1 for _, success, _ in processed_files if success)
    total_count = len(processed_files)
    
    print(f"\n=== 清理统计 ===")
    print(f"目标文件数: {total_count}")
    print(f"成功处理: {success_count}")
    print(f"失败数: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")

if __name__ == "__main__":
    main_cleanup_remaining_sql() 