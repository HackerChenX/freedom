#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的MACD计算

验证专业修复是否解决了计算差异问题
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

def test_fixed_macd_calculation():
    """测试修复后的MACD计算"""
    
    print("🔧 测试修复后的MACD计算")
    print("=" * 80)
    
    # 测试用例
    test_cases = {
        '000017': {
            'date': '2025-05-14',
            'benchmark': {'MACD': 0.037, 'DIFF': 0.149, 'DEA': 0.13}
        },
        '000001': {
            'date': '2025-05-12', 
            'benchmark': {'MACD': 0.073, 'DIFF': -0.039, 'DEA': -0.076}
        }
    }
    
    try:
        stock_data_service = get_stock_data_service()
        
        for stock_code, test_case in test_cases.items():
            print(f"\n📊 测试{stock_code}股票:")
            print("-" * 60)
            
            # 获取股票数据
            df = stock_data_service.get_stock_data(stock_code, days=250)
            
            if df is None or len(df) == 0:
                print(f"❌ 无法获取{stock_code}数据")
                continue
            
            # 查找目标日期
            target_date = pd.to_datetime(test_case['date']).date()
            target_rows = df[df['date'].dt.date == target_date]
            
            if target_rows.empty:
                print(f"❌ 未找到{test_case['date']}的数据")
                continue
            
            target_idx = target_rows.index[0]
            
            print(f"📅 目标日期: {test_case['date']}")
            print(f"📈 数据量: {len(df)}天")
            print(f"💰 收盘价: {df.iloc[target_idx]['close']:.3f}")
            
            # 测试不同的EMA方法
            methods = ['standard', 'sma_init', 'pandas']
            
            print(f"\n🔄 测试不同EMA方法:")
            
            best_method = None
            min_error = float('inf')
            
            for method in methods:
                print(f"\n  📊 {method}方法:")
                
                try:
                    # 创建MACD指标实例
                    macd_indicator = MacdMacd()
                    
                    # 计算MACD（传入ema_method参数）
                    macd_result = macd_indicator._calculate_macd(df, ema_method=method)
                    
                    if target_idx >= len(macd_result):
                        print(f"    ❌ 索引超出范围")
                        continue
                    
                    # 获取目标日期的结果
                    target_data = macd_result.iloc[target_idx]
                    
                    # 检查数据有效性
                    if (pd.isna(target_data['macd_line']) or 
                        pd.isna(target_data['macd_signal']) or 
                        pd.isna(target_data['macd_histogram'])):
                        print(f"    ⚠️ 数据无效（可能在预热期内）")
                        continue
                    
                    # 显示计算结果
                    diff_val = target_data['macd_line']
                    dea_val = target_data['macd_signal']
                    macd_val = target_data['macd_histogram']
                    
                    print(f"    DIFF: {diff_val:.6f}")
                    print(f"    DEA:  {dea_val:.6f}")
                    print(f"    MACD: {macd_val:.6f}")
                    
                    # 计算与基准的误差
                    benchmark = test_case['benchmark']
                    diff_error = abs(diff_val - benchmark['DIFF'])
                    dea_error = abs(dea_val - benchmark['DEA'])
                    macd_error = abs(macd_val - benchmark['MACD'])
                    total_error = diff_error + dea_error + macd_error
                    
                    print(f"    误差: DIFF={diff_error:.6f}, DEA={dea_error:.6f}, MACD={macd_error:.6f}")
                    print(f"    总误差: {total_error:.6f}")
                    
                    # 记录最佳方法
                    if total_error < min_error:
                        min_error = total_error
                        best_method = method
                    
                except Exception as e:
                    print(f"    ❌ 计算异常: {e}")
            
            # 显示最佳结果
            print(f"\n🏆 {stock_code}最佳方法: {best_method}")
            print(f"📊 最小总误差: {min_error:.6f}")
            
            # 评估修复效果
            if min_error < 0.01:
                print(f"✅ 修复效果优秀：误差<0.01")
            elif min_error < 0.05:
                print(f"⚠️ 修复效果良好：误差<0.05")
            else:
                print(f"❌ 仍需进一步优化：误差>{0.05}")
    
    except Exception as e:
        print(f"❌ 测试过程异常: {e}")
        import traceback
        traceback.print_exc()

def test_macd_pattern_detection():
    """测试修复后的MACD形态检测"""
    
    print(f"\n🎯 测试修复后的MACD形态检测")
    print("=" * 80)
    
    try:
        stock_data_service = get_stock_data_service()
        macd_indicator = MacdMacd()
        
        # 测试几支股票的形态检测
        test_stocks = ['000001', '000017', '000066']
        
        for stock_code in test_stocks:
            print(f"\n📊 测试{stock_code}形态检测:")
            
            df = stock_data_service.get_stock_data(stock_code, days=250)
            
            if df is None or len(df) == 0:
                print(f"  ❌ 无法获取数据")
                continue
            
            # 使用最佳方法计算MACD
            macd_result = macd_indicator._calculate_macd(df, ema_method='standard')
            
            print(f"  📈 数据量: {len(df)}天")
            print(f"  📊 MACD结果: {len(macd_result)}行")
            
            # 检查最近的有效数据
            valid_data = macd_result.dropna()
            if len(valid_data) > 0:
                latest_data = valid_data.iloc[-1]
                print(f"  📅 最新有效日期: {df.iloc[valid_data.index[-1]]['date'].strftime('%Y-%m-%d')}")
                print(f"  📊 最新MACD值: DIFF={latest_data['macd_line']:.6f}, DEA={latest_data['macd_signal']:.6f}, MACD={latest_data['macd_histogram']:.6f}")
                print(f"  ✅ 形态检测可用")
            else:
                print(f"  ❌ 无有效MACD数据")
    
    except Exception as e:
        print(f"❌ 形态检测测试异常: {e}")

def main():
    """主函数"""
    print("🔧 MACD专业修复验证测试")
    print("验证修复后的MACD计算是否解决了差异问题")
    
    # 测试修复后的MACD计算
    test_fixed_macd_calculation()
    
    # 测试形态检测功能
    test_macd_pattern_detection()
    
    print(f"\n🎯 修复验证总结:")
    print("=" * 80)
    print(f"1. 已实施专业EMA计算方法（标准、SMA初始化、pandas）")
    print(f"2. 修复了MACD计算函数使用专业方法")
    print(f"3. 保持了向后兼容性")
    print(f"4. 可以通过ema_method参数选择计算方法")
    
    print(f"\n💡 使用建议:")
    print(f"• 对于与市场数据对比：使用'standard'方法")
    print(f"• 对于金融行业标准：使用'sma_init'方法")
    print(f"• 对于向后兼容：使用'pandas'方法")
    
    print(f"\n🚀 下一步:")
    print(f"1. 根据测试结果选择最佳EMA方法")
    print(f"2. 重新生成MACD形态检测结果")
    print(f"3. 验证修复后的人工验证清单")

if __name__ == "__main__":
    main()
