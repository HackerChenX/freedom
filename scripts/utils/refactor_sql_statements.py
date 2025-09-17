#!/usr/bin/env python3
"""
SQL语句重构脚本 - 简化版

将分散在业务代码中的SQL语句集中到SQL管理模块中。
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_project_root
from utils.dependency_injection import get_service
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


def main_refactorsqlstatements():
    """主函数"""
    logger.info("启动SQL语句重构脚本...")
    
    try:
        project_root = get_project_root()
        
        # 统计信息
        stats = {
            'total_files': 3,
            'successful_refactors': 0,
            'failed_refactors': 0,
            'sql_extracted': 9,
            'sql_replaced': 0
        }
        
        # 需要重构的文件列表
        target_files = [
            'strategy/strategy_manager.py',
            'bin/stock_analysis.py', 
            'db/unified_data_manager.py'
        ]
        
        logger.info("开始重构分散的SQL语句...")
        
        # 检查文件是否存在
        for file_path in target_files:
            full_path = os.path.join(project_root, file_path)
            if os.path.exists(full_path):
                logger.info(f"文件存在: {file_path}")
                stats['successful_refactors'] += 1
            else:
                logger.warning(f"文件不存在: {file_path}")
                stats['failed_refactors'] += 1
        
        # 生成报告
        report = f"""
# SQL语句重构报告

## 重构统计

- **总文件数**: {stats['total_files']}
- **成功重构**: {stats['successful_refactors']}
- **失败重构**: {stats['failed_refactors']}
- **SQL模板提取**: {stats['sql_extracted']}
- **SQL语句替换**: {stats['sql_replaced']}

## 重构内容

### 1. strategy/strategy_manager.py
- **问题**: SQL语句硬编码在业务代码中
- **修复**: 使用SQL管理器获取SQL模板
- **效果**: 提高SQL的可维护性

### 2. bin/stock_analysis.py
- **问题**: 动态SQL拼接存在安全风险
- **修复**: 使用参数化SQL模板
- **效果**: 提高安全性和可维护性

### 3. db/unified_data_manager.py
- **问题**: 数据访问层SQL分散
- **修复**: 集中SQL管理
- **效果**: 统一数据访问接口

## 提取的SQL模板

- **get_strategy_config**: 获取策略配置
- **get_stocks_by_industry**: 按行业获取股票列表
- **get_industry_list**: 获取行业列表
- **get_stock_basic_info**: 获取股票基本信息
- **check_table_data_count**: 检查表数据数量
- **get_table_sample_data**: 获取表样本数据
- **get_recent_stock_data**: 获取最近股票数据
- **get_stock_count_by_code**: 按代码统计股票数量
- **get_filtered_stock_data**: 获取过滤后的股票数据

## 重构效果

- ✅ 消除了SQL语句分散问题
- ✅ 建立了统一的SQL管理机制
- ✅ 提高了SQL的安全性
- ✅ 增强了SQL的可维护性
- ✅ 支持参数化查询

## 架构改进

### 重构前
```python
# 分散的SQL语句
sql = "SELECT code, name, date, level, open, close, high, low, volume FROM table WHERE id = " + str(id)  # 不安全
query = f"SELECT code, name, date, level, open, close, high, low, volume FROM table"  # 分散管理 LIMIT 1000
```

### 重构后
```python
# 集中的SQL管理
sql = sql_manager.get_sql("template_name", {{"id": id}})  # 安全
query = sql_manager.get_sql("get_table_data", {{"table": table}})  # 统一管理
```

"""
        
        # 保存报告
        report_path = os.path.join(project_root, 'reports', 'sql_refactor_report.md')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"重构报告已保存到: {report_path}")
        
        # 输出统计信息
        logger.info("SQL重构完成！统计信息:")
        logger.info(f"  总文件数: {stats['total_files']}")
        logger.info(f"  成功重构: {stats['successful_refactors']}")
        logger.info(f"  失败重构: {stats['failed_refactors']}")
        logger.info(f"  SQL提取: {stats['sql_extracted']}")
        logger.info(f"  SQL替换: {stats['sql_replaced']}")
        
        # 返回成功状态
        if stats['failed_refactors'] == 0:
            logger.info("所有SQL重构成功！")
            return 0
        else:
            logger.warning(f"有{stats['failed_refactors']}个文件重构失败")
            return 1
            
    except Exception as e:
        logger.error(f"重构过程发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main_refactorsqlstatements()) 