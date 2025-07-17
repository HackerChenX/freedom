from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/env python3
"""
实施SQL查询管理器
将分散的SQL查询统一管理
"""
import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLQueryExtractor:
    """SQL查询提取器"""
    
    def __init__(self):
        self.sql_patterns = [
            # 基本SELECT查询
            r'[\'"](SELECT\s+.*?)[\'"]',
            # 复杂SELECT查询（多行）
            r'[\'"](SELECT\s+.*?(?:\n.*?)*?FROM\s+.*?(?:\n.*?)*?)[\'"]',
            # INSERT查询
            r'[\'"](INSERT\s+.*?)[\'"]',
            # UPDATE查询
            r'[\'"](UPDATE\s+.*?)[\'"]',
            # DELETE查询
            r'[\'"](DELETE\s+.*?)[\'"]',
            # CREATE查询
            r'[\'"](CREATE\s+.*?)[\'"]',
            # ALTER查询
            r'[\'"](ALTER\s+.*?)[\'"]',
            # DROP查询
            r'[\'"](DROP\s+.*?)[\'"]',
            # SHOW查询
            r'[\'"](SHOW\s+.*?)[\'"]',
            # DESCRIBE查询
            r'[\'"](DESCRIBE\s+.*?)[\'"]',
        ]
        
        self.extracted_queries = {}
        self.query_categories = {
            'stock_data': [],
            'batch_operations': [],
            'analysis': [],
            'maintenance': [],
            'metadata': []
        }
    
    def extract_queries_from_file(self, file_path: str) -> List[Dict]:
        """从文件中提取SQL查询"""
        queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in self.sql_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for match in matches:
                    sql = match.group(1).strip()
                    if len(sql) > 10:  # 过滤太短的查询
                        queries.append({
                            'sql': sql,
                            'file': file_path,
                            'line': content[:match.start()].count('\n') + 1,
                            'category': self._categorize_query(sql)
                        })
        
        except Exception as e:
            logger.error(f"提取文件 {file_path} 的SQL查询失败: {e}")
        
        return queries
    
    def _categorize_query(self, sql: str) -> str:
        """对SQL查询进行分类"""
        sql_lower = sql.lower()
        
        if 'stock_info' in sql_lower or 'stock_data' in sql_lower:
            return 'stock_data'
        elif 'batch' in sql_lower or 'bulk' in sql_lower:
            return 'batch_operations'
        elif 'count' in sql_lower or 'avg' in sql_lower or 'sum' in sql_lower:
            return 'analysis'
        elif 'create' in sql_lower or 'alter' in sql_lower or 'drop' in sql_lower:
            return 'maintenance'
        else:
            return 'metadata'
    
    def scan_project_files(self) -> Dict[str, List[Dict]]:
        """扫描项目文件提取SQL查询"""
        logger.info("开始扫描项目文件...")
        
        # 需要扫描的文件列表
        target_files = [
            "debug_stock_603359.py",
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
            "scripts/simple_clickhouse_test.py",
            "scripts/production_database_test.py",
            "scripts/clickhouse_connection_summary.py",
            "scripts/database_optimization.py",
            "strategy/enhanced_base_strategy.py",
            "strategy/batch_optimizer.py",
            "strategy/strategy_manager.py"
        ]
        
        all_queries = {}
        
        for file_path in target_files:
            full_path = os.path.join(root_dir, file_path)
            if os.path.exists(full_path):
                queries = self.extract_queries_from_file(full_path)
                if queries:
                    all_queries[file_path] = queries
                    logger.info(f"从 {file_path} 提取了 {len(queries)} 个SQL查询")
            else:
                logger.warning(f"文件不存在: {full_path}")
        
        return all_queries
    
    def generate_sql_templates(self, queries: Dict[str, List[Dict]]) -> Dict[str, str]:
        """生成SQL模板"""
        templates = {}
        
        # 常见的股票数据查询模板
        templates['get_stock_data'] = """
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE code = %(code)s 
AND date >= %(start_date)s 
AND date <= %(end_date)s
ORDER BY date
"""
        
        templates['get_batch_stock_data'] = """
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE code IN %(codes)s 
AND date >= %(start_date)s 
AND date <= %(end_date)s
ORDER BY code, date
"""
        
        templates['get_stock_list'] = """
SELECT DISTINCT code 
FROM stock_info 
WHERE date >= %(min_date)s
ORDER BY code
"""
        
        templates['get_latest_data'] = """
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE date = (SELECT MAX(date) FROM stock_info WHERE code = %(code)s)
AND code = %(code)s
"""
        
        templates['count_stock_records'] = """
SELECT code, COUNT(*) as record_count
FROM stock_info 
WHERE date >= %(start_date)s
GROUP BY code
ORDER BY record_count DESC
"""
        
        templates['get_date_range'] = """
SELECT MIN(date) as min_date, MAX(date) as max_date
FROM stock_info
WHERE code = %(code)s
"""
        
        return templates

class SQLMigrationTool:
    """SQL迁移工具"""
    
    def __init__(self):
        self.extractor = SQLQueryExtractor()
        self.migrations_applied = 0
        self.files_modified = set()
    
    def create_enhanced_sql_manager(self):
        """创建增强的SQL管理器"""
        logger.info("创建增强的SQL管理器...")
        
        # 扫描现有查询
        all_queries = self.extractor.scan_project_files()
        
        # 生成模板
        templates = self.extractor.generate_sql_templates(all_queries)
        
        # 更新SQL管理器
        sql_manager_path = os.path.join(root_dir, 'db', 'sql_manager.py')
        
        try:
            with open(sql_manager_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找模板字典的位置
            template_start = content.find('self.query_templates = {')
            if template_start != -1:
                # 找到模板字典的结束位置
                brace_count = 0
                template_end = template_start
                for i, char in enumerate(content[template_start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            template_end = template_start + i + 1
                            break
                
                # 构建新的模板字典
                new_templates = "self.query_templates = {\n"
                for template_name, template_sql in templates.items():
                    new_templates += f"            '{template_name}': '''{template_sql.strip()}''',\n"
                new_templates += "        }"
                
                # 替换模板
                new_content = content[:template_start] + new_templates + content[template_end:]
                
                with open(sql_manager_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.info("SQL管理器已更新")
                
        except Exception as e:
            logger.error(f"更新SQL管理器失败: {e}")
    
    def generate_migration_report(self, queries: Dict[str, List[Dict]]) -> str:
        """生成迁移报告"""
        report_path = os.path.join(root_dir, 'doc', '系统设计', 'SQL查询迁移报告.md')
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# SQL查询迁移报告\n\n")
                f.write(f"**生成时间**: {os.popen('date').read().strip()}\n\n")
                
                f.write("## 📊 迁移统计\n\n")
                total_queries = sum(len(qs) for qs in queries.values())
                f.write(f"- **总文件数**: {len(queries)}\n")
                f.write(f"- **总查询数**: {total_queries}\n")
                f.write(f"- **已迁移**: {self.migrations_applied}\n\n")
                
                f.write("## 📝 发现的SQL查询\n\n")
                for file_path, file_queries in queries.items():
                    f.write(f"### {file_path}\n\n")
                    for i, query in enumerate(file_queries, 1):
                        f.write(f"**查询 {i}** (第{query['line']}行, 类别: {query['category']})\n")
                        f.write("```sql\n")
                        f.write(query['sql'])
                        f.write("\n```\n\n")
                
                f.write("## 🔧 建议的迁移步骤\n\n")
                f.write("1. 将常用查询模板化\n")
                f.write("2. 使用参数化查询替代字符串拼接\n")
                f.write("3. 统一查询接口\n")
                f.write("4. 添加查询缓存机制\n\n")
                
                f.write("---\n\n")
                f.write("*本报告由SQL迁移工具自动生成*\n")
            
            logger.info(f"迁移报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"生成迁移报告失败: {e}")
            return None

def main_implement_sql_management():
    """主函数"""
    logger.info("开始SQL查询管理器实施...")
    
    migration_tool = SQLMigrationTool()
    
    # 1. 扫描项目文件
    logger.info("步骤 1: 扫描项目文件")
    all_queries = migration_tool.extractor.scan_project_files()
    
    # 2. 创建增强的SQL管理器
    logger.info("步骤 2: 创建增强的SQL管理器")
    migration_tool.create_enhanced_sql_manager()
    
    # 3. 生成迁移报告
    logger.info("步骤 3: 生成迁移报告")
    report_path = migration_tool.generate_migration_report(all_queries)
    
    logger.info("SQL查询管理器实施完成！")
    
    if report_path:
        logger.info(f"详细报告: {report_path}")
    
    # 运行架构检查验证效果
    logger.info("运行架构检查验证效果...")
    os.system("python simple_architecture_check.py")

if __name__ == "__main__":
    main_implement_sql_management() 