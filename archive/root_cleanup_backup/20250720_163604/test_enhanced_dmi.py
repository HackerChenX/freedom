from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/env python3
"""
测试Enhanced DMI指标修复效果
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from indicators.trend.enhanced_dmi import Enhanced_dMI

def test_enhanced_dmi():
    """测试Enhanced DMI指标"""
    print("=" * 60)
    print("测试Enhanced DMI指标")
    print("=" * 60)
    
    # 获取数据库连接
    data_access = get_container().resolve(DataAccessInterface)
    
    # 测试股票代码
    test_codes = ['000001', '000002', '000858']
    
    # 创建指标实例
    indicator = Enhanced_dMI(period=14, adx_period=14, adaptive=True)
    
    total_signals = 0
    total_tests = 0
    
    for code in test_codes:
        print(f"\n测试股票: {code}")
        print("-" * 40)
        
        try:
            # 获取股票数据
            sql = f"""
            SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info WHERE 1=1
            WHERE code = '{code}' 
            AND date >= '2025-03-01' 
            AND date <= '2025-07-31'
            ORDER BY date
            """
            
            data = data_access.execute_query(sql)
            
            if data.empty:
                print(f"  警告: 股票 {code} 没有数据")
                continue
                
            # 设置索引
            data.set_index('date', inplace=True)
            
            print(f"  数据行数: {len(data)}")
            
            # 计算指标
            result = indicator.calculate(data)
            
            # 计算评分
            scores = indicator.calculate_raw_score(data)
            
            # 统计买入信号
            buy_signals = scores > 60  # 评分大于60认为是买入信号
            buy_count = buy_signals.sum()
            
            # 计算信号率
            signal_rate = buy_count / len(scores) * 100 if len(scores) > 0 else 0
            
            # 计算平均置信度
            avg_confidence = scores.mean() if not scores.empty else 0
            
            print(f"  指标计算成功: {not result.empty}")
            print(f"  总评分数: {len(scores)}")
            print(f"  买入信号数: {buy_count}")
            print(f"  信号率: {signal_rate:.1f}%")
            print(f"  平均置信度: {avg_confidence:.1f}")
            
            # 显示最近几个信号
            if buy_count > 0:
                recent_signals = buy_signals.tail(10)
                signal_dates = recent_signals[recent_signals].index
                if len(signal_dates) > 0:
                    print(f"  最近信号日期: {signal_dates[-1]}")
            
            total_signals += buy_count
            total_tests += 1
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("Enhanced DMI测试总结")
    print("=" * 60)
    print(f"成功测试股票数: {total_tests}")
    print(f"总买入信号数: {total_signals}")
    if total_tests > 0:
        print(f"平均每股信号数: {total_signals / total_tests:.1f}")
    
    # 测试结果判断
    success = total_tests > 0 and total_signals > 0
    print(f"\n测试结果: {'成功' if success else '失败'}")
    
    return success

if __name__ == "__main__":
    test_enhanced_dmi() 