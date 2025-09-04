#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追踪MACD计算不一致问题

发现问题：
1. 精确验证结果：DIFF=-0.136158, DEA=-0.246368, MACD=0.220420
2. validation_result.json：macd_line=0.2517, signal_line=0.20136, histogram=0.05034

需要找出为什么同一支股票同一日期会有完全不同的MACD计算结果
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

def trace_macd_inconsistency():
    """追踪MACD计算不一致问题"""
    
    print("🔍 追踪MACD计算不一致问题")
    print("=" * 80)
    
    stock_code = "000028"
    target_date = "2025-05-12"
    
    # 读取validation_result.json中的数据
    json_file = Path("validation/results/MACD_validation_result.json")
    json_data = None
    
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            print(f"✅ 成功读取 {json_file}")
        except Exception as e:
            print(f"❌ 读取JSON文件失败: {e}")
    else:
        print(f"❌ 文件不存在: {json_file}")
        return
    
    # 查找000028的数据
    target_stock_data = None
    if json_data:
        # 遍历所有形态查找000028
        for pattern_id, pattern_data in json_data.get('patterns_validation', {}).items():
            for stock_info in pattern_data.get('matching_stocks', []):
                if (stock_info.get('stock_code') == stock_code and 
                    stock_info.get('detection_date') == target_date):
                    target_stock_data = stock_info
                    print(f"✅ 在{pattern_id}形态中找到{stock_code}的数据")
                    break
            if target_stock_data:
                break
    
    if not target_stock_data:
        print(f"❌ 在JSON文件中未找到{stock_code}在{target_date}的数据")
        return
    
    print(f"\n📊 JSON文件中的数据:")
    indicator_values = target_stock_data.get('indicator_values', {})
    print(f"  macd_line: {indicator_values.get('macd_line', 'N/A')}")
    print(f"  signal_line: {indicator_values.get('signal_line', 'N/A')}")
    print(f"  histogram: {indicator_values.get('histogram', 'N/A')}")
    
    # 现在重新计算MACD
    print(f"\n🔄 重新计算MACD进行对比...")
    
    try:
        macd_indicator = MacdMacd()
        stock_data_service = get_stock_data_service()
        
        # 获取股票数据
        df = stock_data_service.get_stock_data(stock_code, days=200)
        
        if df is None or len(df) == 0:
            print(f"❌ 无法获取{stock_code}数据")
            return
        
        print(f"✅ 获取到{len(df)}天数据")
        
        # 查找目标日期
        target_date_obj = pd.to_datetime(target_date).date()
        target_rows = df[df['date'].dt.date == target_date_obj]
        
        if target_rows.empty:
            print(f"❌ 未找到{target_date}的数据")
            return
        
        target_idx = target_rows.index[0]
        print(f"✅ 找到目标日期，索引: {target_idx}")
        
        # 显示价格数据
        price_data = df.loc[target_idx]
        print(f"\n📈 {target_date}的价格数据:")
        print(f"  开盘: {price_data['open']:.3f}")
        print(f"  收盘: {price_data['close']:.3f}")
        print(f"  最高: {price_data['high']:.3f}")
        print(f"  最低: {price_data['low']:.3f}")
        print(f"  成交量: {price_data['volume']:,.0f}")
        
        # 计算MACD
        macd_result = macd_indicator.calculate(df)
        
        if macd_result is None or macd_result.empty:
            print("❌ MACD计算失败")
            return
        
        if target_idx >= len(macd_result):
            print(f"❌ 目标索引{target_idx}超出MACD结果范围")
            return
        
        current_data = macd_result.iloc[target_idx]
        
        print(f"\n📊 当前计算结果:")
        print(f"  DIFF (macd_line): {current_data['macd_line']:.6f}")
        print(f"  DEA (macd_signal): {current_data['macd_signal']:.6f}")
        print(f"  MACD (macd_histogram): {current_data['macd_histogram']:.6f}")
        
        # 详细对比
        print(f"\n📋 数据对比分析:")
        print(f"{'数据源':<15} {'DIFF':<15} {'DEA':<15} {'MACD':<15}")
        print("-" * 65)
        print(f"{'JSON文件':<15} {indicator_values.get('macd_line', 0):<15.6f} {indicator_values.get('signal_line', 0):<15.6f} {indicator_values.get('histogram', 0):<15.6f}")
        print(f"{'当前计算':<15} {current_data['macd_line']:<15.6f} {current_data['macd_signal']:<15.6f} {current_data['macd_histogram']:<15.6f}")
        
        # 计算差异
        diff_macd_line = abs(indicator_values.get('macd_line', 0) - current_data['macd_line'])
        diff_signal_line = abs(indicator_values.get('signal_line', 0) - current_data['macd_signal'])
        diff_histogram = abs(indicator_values.get('histogram', 0) - current_data['macd_histogram'])
        
        print(f"{'绝对差异':<15} {diff_macd_line:<15.6f} {diff_signal_line:<15.6f} {diff_histogram:<15.6f}")
        
        # 分析差异程度
        print(f"\n🔍 差异分析:")
        
        if diff_macd_line > 0.1 or diff_signal_line > 0.1 or diff_histogram > 0.1:
            print(f"  ❌ 发现重大差异（>0.1），可能原因:")
            print(f"    1. 使用了不同的MACD计算方法")
            print(f"    2. 数据源完全不同")
            print(f"    3. 参数设置不同")
            print(f"    4. 计算时间点不同")
        elif diff_macd_line > 0.01 or diff_signal_line > 0.01 or diff_histogram > 0.01:
            print(f"  ⚠️ 发现中等差异（>0.01），可能原因:")
            print(f"    1. 数据源有差异")
            print(f"    2. 计算精度不同")
            print(f"    3. 历史数据长度不同")
        else:
            print(f"  ✅ 差异很小（<0.01），在可接受范围内")
        
        # 检查MACD参数
        print(f"\n⚙️ MACD参数检查:")
        if hasattr(macd_indicator, '_parameters'):
            params = macd_indicator._parameters
            print(f"  快线周期: {params.get('fast_period', 'N/A')}")
            print(f"  慢线周期: {params.get('slow_period', 'N/A')}")
            print(f"  信号线周期: {params.get('signal_period', 'N/A')}")
        
        # 检查数据时间戳
        print(f"\n📅 数据时间戳检查:")
        print(f"  数据最新日期: {df['date'].max()}")
        print(f"  目标日期: {target_date}")
        print(f"  数据是否最新: {'是' if df['date'].max().date() >= pd.to_datetime(target_date).date() else '否'}")
        
        # 尝试找出JSON数据的来源
        print(f"\n🔍 JSON数据来源分析:")
        if json_data:
            validation_timestamp = json_data.get('validation_timestamp', 'N/A')
            data_source = json_data.get('data_source', 'N/A')
            print(f"  验证时间戳: {validation_timestamp}")
            print(f"  数据源: {data_source}")
            
            # 检查是否有其他MACD相关信息
            if 'macd_calculation_method' in json_data:
                print(f"  计算方法: {json_data['macd_calculation_method']}")
        
        # 建议
        print(f"\n💡 问题解决建议:")
        if diff_macd_line > 0.1:
            print(f"  1. 重新生成validation_result.json文件")
            print(f"  2. 检查是否使用了不同的MACD指标类")
            print(f"  3. 确认数据源一致性")
            print(f"  4. 验证计算参数设置")
        else:
            print(f"  1. 差异较小，可能是正常的精度差异")
            print(f"  2. 建议更新JSON文件以保持一致性")
        
    except Exception as e:
        print(f"❌ 追踪过程异常: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    trace_macd_inconsistency()

if __name__ == "__main__":
    main()
