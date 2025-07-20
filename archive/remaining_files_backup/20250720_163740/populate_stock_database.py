#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量填充股票数据库

为支持4000+只股票的测试，批量插入股票基本信息。

Author: AI Assistant
Date: 2025-07-19
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def generate_stock_data():
    """生成股票数据"""
    stocks = []
    
    # 深交所主板 (000001-002999)
    for i in range(1, 3000):
        code = f"{i:06d}"
        if i <= 999:
            market = "深交所主板"
            industry = "综合"
        else:
            market = "深交所中小板"
            industry = "制造业"
        
        stocks.append({
            'code': code,
            'name': f'深股{code}',
            'market': market,
            'industry': industry,
            'list_date': '2020-01-01'
        })
    
    # 创业板 (300001-301999)
    for i in range(300001, 302000):
        code = f"{i:06d}"
        stocks.append({
            'code': code,
            'name': f'创业板{code}',
            'market': '深交所创业板',
            'industry': '高新技术',
            'list_date': '2020-01-01'
        })
    
    # 上交所主板 (600000-603999)
    for i in range(600000, 604000):
        code = f"{i:06d}"
        stocks.append({
            'code': code,
            'name': f'沪股{code}',
            'market': '上交所主板',
            'industry': '传统行业',
            'list_date': '2020-01-01'
        })
    
    # 科创板 (688001-688999)
    for i in range(688001, 689000):
        code = f"{i:06d}"
        stocks.append({
            'code': code,
            'name': f'科创板{code}',
            'market': '上交所科创板',
            'industry': '科技创新',
            'list_date': '2020-01-01'
        })
    
    return stocks

def insert_stocks_batch(stocks, batch_size=1000):
    """批量插入股票数据"""
    import subprocess
    
    total_batches = (len(stocks) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(stocks))
        batch_stocks = stocks[start_idx:end_idx]
        
        # 构建INSERT语句
        values = []
        for stock in batch_stocks:
            values.append(f"('{stock['code']}', '{stock['name']}', '{stock['market']}', '{stock['industry']}', '{stock['list_date']}', 1)")
        
        values_str = ',\n'.join(values)
        
        sql = f"""
        INSERT INTO stock.stock_info (stock_code, stock_name, market, industry, list_date, is_active) VALUES
        {values_str}
        """
        
        try:
            # 执行插入
            result = subprocess.run([
                'clickhouse-client', 
                '--password=123456', 
                '--query', sql
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ 批次 {batch_idx + 1}/{total_batches} 插入成功: {len(batch_stocks)} 只股票")
            else:
                print(f"❌ 批次 {batch_idx + 1}/{total_batches} 插入失败: {result.stderr}")
                
        except Exception as e:
            print(f"❌ 批次 {batch_idx + 1}/{total_batches} 执行失败: {e}")

def main():
    """主函数"""
    print("🚀 开始批量填充股票数据库")
    print("=" * 60)
    
    # 生成股票数据
    print("📊 生成股票数据...")
    stocks = generate_stock_data()
    print(f"生成了 {len(stocks)} 只股票数据")
    
    # 批量插入
    print("💾 开始批量插入...")
    insert_stocks_batch(stocks, batch_size=500)
    
    # 验证插入结果
    print("🔍 验证插入结果...")
    import subprocess
    result = subprocess.run([
        'clickhouse-client', 
        '--password=123456', 
        '--query', 'SELECT COUNT(*) FROM stock.stock_info'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        count = result.stdout.strip()
        print(f"✅ 数据库中共有 {count} 只股票")
        
        if int(count) >= 4000:
            print("🎉 股票数据已满足4000+的测试要求！")
        else:
            print(f"⚠️ 股票数量不足4000只，当前: {count}")
    else:
        print(f"❌ 验证失败: {result.stderr}")
    
    print("=" * 60)
    print("✅ 股票数据库填充完成")

if __name__ == "__main__":
    main()
