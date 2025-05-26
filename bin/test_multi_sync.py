#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
多线程同步测试脚本
-----------------
本脚本用于测试股票数据的多线程同步功能，支持多个数据源和自动切换。

使用方法:
python test_multi_sync.py [--csv 股票代码文件] [--threads 线程数] 
                         [--batch 批次大小] [--limit 限制数量] 
                         [--sleep 休眠秒数] [--force] [--code 股票代码]
                         [--source 数据源] [--debug]

支持的数据源: efinance, akshare, baostock (按优先级排序)
"""

import os
import sys
import time
import argparse
import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.akshare_to_clickhouse import AKShareToClickHouse, DataSourceManager

def main():
    """测试多线程同步功能"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='测试多线程同步功能')
    parser.add_argument('--csv', type=str, default='data/reference/stock_code_name.csv',
                        help='股票代码CSV文件路径')
    parser.add_argument('--threads', type=int, default=5,
                        help='最大工作线程数')
    parser.add_argument('--batch', type=int, default=10,
                        help='每批处理的股票数量')
    parser.add_argument('--limit', type=int, default=0,
                        help='最多处理的股票数量，用于测试。0表示不限制')
    parser.add_argument('--sleep', type=int, default=1,
                        help='批次间休眠秒数')
    parser.add_argument('--force', action='store_true',
                        help='强制同步所有股票，忽略最新日期检查')
    parser.add_argument('--code', type=str, default=None,
                        help='只同步指定股票代码，多个代码用逗号分隔，如：000001,600000')
    parser.add_argument('--source', type=str, default='efinance', choices=['efinance', 'akshare', 'baostock'],
                        help='初始数据源类型，可选值：efinance, akshare, baostock')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式，显示更详细的日志')
    parser.add_argument('--no-auto-switch', action='store_true',
                        help='禁用数据源自动切换功能')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print(f"最大线程数: {args.threads}")
    print(f"每批处理股票数: {args.batch}")
    print(f"股票代码文件: {args.csv}")
    print(f"批次间休眠秒数: {args.sleep}")
    print(f"强制同步所有股票: {args.force}")
    print(f"数据源类型: {args.source}")
    print(f"自动切换数据源: {not args.no_auto_switch}")
    
    if args.limit > 0:
        print(f"最多处理股票数: {args.limit}")
    
    if args.code:
        print(f"仅同步指定股票: {args.code}")
        # 将指定的股票代码转换为列表
        stock_codes = args.code.split(',')
        # 创建一个临时CSV文件
        import pandas as pd
        temp_df = pd.DataFrame({'code': stock_codes, 'name': [f'股票{code}' for code in stock_codes]})
        temp_csv = 'temp_stock_codes.csv'
        temp_df.to_csv(temp_csv, index=False)
        csv_path = temp_csv
    else:
        csv_path = args.csv
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化同步器
    synchronizer = AKShareToClickHouse(max_workers=args.threads, batch_size=args.batch, 
                                     force_sync=args.force, data_source=args.source)
    
    # 如果禁用自动切换，修改数据源管理器的最大失败次数为一个非常大的值
    if args.no_auto_switch:
        synchronizer.data_source_manager.max_failures = 999999
    
    # 获取并显示最新交易日期
    print(f"当前最新交易日期: {synchronizer.latest_trade_date}")
    print(f"只同步尚未同步到最新日期的股票" if not args.force else "强制同步所有股票")
    print(f"初始数据源: {args.source}，自动切换顺序: {' > '.join(DataSourceManager.SOURCES)}")
    
    try:
        # 开始同步
        print(f"开始多线程同步股票数据，详细日志请查看 logs/akshare_sync.log")
        
        # 直接调用同步器的sync_all_stocks方法
        success_count = synchronizer.sync_all_stocks(csv_path)
        
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"同步完成，总耗时: {elapsed_time:.2f} 秒，成功处理股票数量: {success_count}")
    except KeyboardInterrupt:
        print("\n用户中断，停止同步...")
    except Exception as e:
        print(f"同步过程中发生错误: {e}")
    finally:
        # 如果创建了临时文件，清理它
        if args.code:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

if __name__ == "__main__":
    main() 