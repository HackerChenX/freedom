#!/usr/bin/env python3
"""
选股结果分析脚本
分析所有批次测试结果，统计每个指标的选股表现
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Any

def analyze_selection_results():
    """分析所有测试结果文件的选股情况"""
    
    # 获取所有测试结果文件
    result_files = glob.glob("results/comprehensive_test/comprehensive_test_*.json")
    result_files.sort()
    
    # 用于存储所有指标的选股结果
    all_indicators = {}
    batch_summary = {}
    
    print("📊 分析88个技术指标的选股表现")
    print("=" * 80)
    
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取批次信息
            if 'batch_results' in data and data['batch_results']:
                batch_name = data['batch_results'][0]['batch_name']
                batch_summary[batch_name] = {
                    'total_indicators': data['batch_results'][0]['total_indicators'],
                    'success_rate': data['batch_results'][0]['success_rate']
                }
                
                print(f"\n🔍 分析 {batch_name}")
                
                # 分析每个指标的选股结果
                for result in data['detailed_results']:
                    indicator_name = result['indicator_name']
                    selected_stocks = result.get('selected_stocks', [])
                    stock_count = len(selected_stocks) if selected_stocks else 0
                    execution_time = result.get('execution_time', 0)
                    
                    all_indicators[indicator_name] = {
                        'batch': batch_name,
                        'selected_count': stock_count,
                        'selected_stocks': selected_stocks,
                        'execution_time': execution_time,
                        'success': result.get('success', False)
                    }
                    
                    # 显示选股情况
                    if stock_count > 0:
                        print(f"   ✅ {indicator_name}: 选中 {stock_count} 只股票 {selected_stocks}")
                    else:
                        print(f"   ❌ {indicator_name}: 未选中任何股票")
                        
        except Exception as e:
            print(f"❌ 处理文件 {file_path} 时出错: {e}")
    
    # 生成统计报告
    print("\n" + "=" * 80)
    print("📈 选股结果统计报告")
    print("=" * 80)
    
    # 按选股能力分类
    with_selections = {}
    without_selections = {}
    
    for indicator, data in all_indicators.items():
        if data['selected_count'] > 0:
            with_selections[indicator] = data
        else:
            without_selections[indicator] = data
    
    print(f"\n🎯 有选股结果的指标: {len(with_selections)} 个")
    print("-" * 50)
    for indicator, data in sorted(with_selections.items(), key=lambda x: x[1]['selected_count'], reverse=True):
        print(f"   {indicator}: {data['selected_count']} 只股票 ({data['batch']})")
        print(f"      股票代码: {data['selected_stocks']}")
    
    print(f"\n❌ 无选股结果的指标: {len(without_selections)} 个")
    print("-" * 50)
    for indicator, data in sorted(without_selections.items()):
        print(f"   {indicator} ({data['batch']})")
    
    # 按批次统计
    print(f"\n📊 按批次统计选股情况")
    print("-" * 50)
    batch_stats = defaultdict(lambda: {'with_selection': 0, 'without_selection': 0, 'total': 0})
    
    for indicator, data in all_indicators.items():
        batch = data['batch']
        batch_stats[batch]['total'] += 1
        if data['selected_count'] > 0:
            batch_stats[batch]['with_selection'] += 1
        else:
            batch_stats[batch]['without_selection'] += 1
    
    for batch, stats in sorted(batch_stats.items()):
        selection_rate = (stats['with_selection'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"   {batch}: {stats['with_selection']}/{stats['total']} ({selection_rate:.1f}%) 有选股")
    
    # 总体统计
    total_indicators = len(all_indicators)
    indicators_with_selection = len(with_selections)
    overall_selection_rate = (indicators_with_selection / total_indicators * 100) if total_indicators > 0 else 0
    
    print(f"\n🎉 总体选股统计")
    print("-" * 50)
    print(f"   总指标数: {total_indicators}")
    print(f"   有选股的指标: {indicators_with_selection}")
    print(f"   无选股的指标: {len(without_selections)}")
    print(f"   选股覆盖率: {overall_selection_rate:.1f}%")
    
    # 分析原因
    print(f"\n🤔 无选股结果的可能原因分析")
    print("-" * 50)
    print("   1. 指标条件过于严格，当前市场环境下无股票满足条件")
    print("   2. 指标主要用于分析展示，不是选股类指标")
    print("   3. 需要特定市场条件或更大股票池才能触发选股")
    print("   4. 指标参数可能需要根据当前市场调优")
    
    return all_indicators, batch_stats

if __name__ == "__main__":
    analyze_selection_results() 