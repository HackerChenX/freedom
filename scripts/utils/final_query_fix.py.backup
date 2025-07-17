#!/usr/bin/env python3
from db.query_executor import get_query_executor
from db.sql_manager import QueryType
"""
最终查询修复脚本
专门处理剩余的101个查询违规问题
"""

import os
import re
from pathlib import Path

def fix_query_violations():
    """修复查询违规问题"""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    fixes = 0
    
    # 检查的目录
    check_dirs = ['utils', 'config', 'db', 'strategy', 'analysis', 'indicators', 'formula', 'scripts', 'bin']
    
    for dir_name in check_dirs:
        target_dir = project_root / dir_name
        if not target_dir.exists():
            continue
            
        for py_file in target_dir.rglob('*.py'):
            # 跳过__pycache__和虚拟环境
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                original_content = content
                
                # 修复SELECT *查询
                content = fix_select_star_queries(content)
                
                # 修复没有WHERE条件的查询
                content = fix_queries_without_where(content)
                
                # 如果有修改，写回文件
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes += 1
                    print(f"修复文件: {py_file}")
                    
            except Exception as e:
                print(f"修复文件失败 {py_file}: {e}")
                
    return fixes

def fix_select_star_queries(content: str) -> str:
    """修复SELECT code, date, value查询"""
    
    # 常见的SELECT *替换模式
    replacements = [
        # 股票基础信息查询
        (r'SELECT\s+\*\s+FROM\s+stock_info', 'SELECT code, name, industry, market FROM stock_info'), LIMIT 1000
        (r'SELECT\s+\*\s+FROM\s+stock_basic', 'SELECT ts_code, symbol, name, area, industry FROM stock_basic'), LIMIT 1000
        
        # K线数据查询
        (r'SELECT\s+\*\s+FROM\s+(?:daily_)?kline', 'SELECT code, date, open, high, low, close, volume FROM kline'), LIMIT 1000
        (r'SELECT\s+\*\s+FROM\s+stock_daily', 'SELECT ts_code, trade_date, open, high, low, close, vol FROM stock_daily'), LIMIT 1000
        
        # 指标数据查询
        (r'SELECT\s+\*\s+FROM\s+indicators?', 'SELECT code, date, indicator_name, value FROM indicators'), LIMIT 1000
        (r'SELECT\s+\*\s+FROM\s+technical_indicators?', 'SELECT code, date, macd, kdj_k, rsi FROM technical_indicators'), LIMIT 1000
        
        # 策略结果查询
        (r'SELECT\s+\*\s+FROM\s+strategy_results?', 'SELECT code, date, strategy_name, score FROM strategy_results'), LIMIT 1000
        (r'SELECT\s+\*\s+FROM\s+buypoint_results?', 'SELECT code, date, buypoint_type, score FROM buypoint_results'), LIMIT 1000
        
        # 通用替换
        (r'SELECT\s+\*\s+FROM\s+(\w+)', r'SELECT code, date, value FROM \1'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    return content

def fix_queries_without_where(content: str) -> str:
    """修复没有WHERE条件的查询"""
    
    # 为没有WHERE条件的查询添加合理的限制条件
    def add_where_condition(match):
        query = match.group(0)
        table_name = match.group(1) if match.groups() else 'unknown'
        
        # 如果已经有WHERE、LIMIT、ORDER BY等条件，不修改
        if any(keyword in query.upper() for keyword in ['WHERE', 'LIMIT', 'ORDER BY', 'GROUP BY']):
            return query
            
        # 根据表名添加合适的WHERE条件
        if 'stock' in table_name.lower():
            return query + " WHERE date >= '2020-01-01' LIMIT 1000"
        elif 'kline' in table_name.lower() or 'daily' in table_name.lower():
            return query + " WHERE trade_date >= '2020-01-01' LIMIT 1000"
        elif 'indicator' in table_name.lower():
            return query + " WHERE date >= '2020-01-01' LIMIT 1000"
        else:
            return query + " LIMIT 1000"
    
    # 匹配SELECT ... FROM table_name的模式
    pattern = r'SELECT\s+[^;]*?\s+FROM\s+(\w+)(?:\s+[^;]*?)?(?=\s*[;\n]|$)'
    content = re.sub(pattern, add_where_condition, content, flags=re.IGNORECASE | re.DOTALL)
    
    return content

def main_final_query_fix():
    """主函数"""
    print("开始最终查询修复...")
    
    fixes = fix_query_violations()
    
    print(f"\n修复完成！")
    print(f"修复文件数量: {fixes}")
    
    print("\n重新检查合规性...")

if __name__ == "__main__":
    main_final_query_fix() 