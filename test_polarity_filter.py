#!/usr/bin/env python3
"""
测试极性过滤功能

验证买点分析系统是否正确过滤了负面模式
"""

import sys
import os
sys.path.append('.')

from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
import pandas as pd

def test_polarity_filter():
    """测试极性过滤功能"""
    print("🧪 测试极性过滤功能")
    print("=" * 50)
    
    # 创建分析器
    analyzer = BuyPointBatchAnalyzer()
    
    # 读取实际数据
    df = pd.read_csv('data/buypoints.csv')
    print(f"加载了 {len(df)} 个买点")
    
    # 测试前3个买点
    test_data = df.head(3).to_dict('records')
    print(f"测试 {len(test_data)} 个买点...")
    
    try:
        # 分析买点
        results = analyzer.analyze_batch_buypoints(pd.DataFrame(test_data))
        print(f"分析完成，结果数量: {len(results)}")
        
        if not results:
            print("❌ 没有分析结果")
            return
        
        # 测试不过滤负面模式
        print("\n📊 不过滤负面模式的结果:")
        common_indicators_unfiltered = analyzer.extract_common_indicators(
            results, 
            min_hit_ratio=0.5, 
            filter_negative_patterns=False
        )
        
        total_unfiltered = sum(len(indicators) for indicators in common_indicators_unfiltered.values())
        print(f"总共性指标数量: {total_unfiltered}")
        
        # 统计负面模式
        negative_count = 0
        for period, indicators in common_indicators_unfiltered.items():
            for indicator in indicators:
                indicator_name = indicator['name']
                pattern_name = indicator.get('pattern', '')
                display_name = indicator.get('display_name', '')
                
                if analyzer.polarity_filter.is_negative_pattern(indicator_name, pattern_name, display_name):
                    negative_count += 1
                    print(f"  负面模式: {indicator_name} - {display_name}")
        
        print(f"检测到负面模式数量: {negative_count}")
        
        # 测试过滤负面模式
        print("\n🔍 过滤负面模式的结果:")
        common_indicators_filtered = analyzer.extract_common_indicators(
            results, 
            min_hit_ratio=0.5, 
            filter_negative_patterns=True
        )
        
        total_filtered = sum(len(indicators) for indicators in common_indicators_filtered.values())
        print(f"总共性指标数量: {total_filtered}")
        
        # 验证过滤效果
        filtered_negative_count = 0
        for period, indicators in common_indicators_filtered.items():
            for indicator in indicators:
                indicator_name = indicator['name']
                pattern_name = indicator.get('pattern', '')
                display_name = indicator.get('display_name', '')
                
                if analyzer.polarity_filter.is_negative_pattern(indicator_name, pattern_name, display_name):
                    filtered_negative_count += 1
                    print(f"  ⚠️ 漏过的负面模式: {indicator_name} - {display_name}")
        
        print(f"过滤后剩余负面模式: {filtered_negative_count}")
        
        # 统计结果
        filtered_out = total_unfiltered - total_filtered
        print(f"\n📈 过滤统计:")
        print(f"过滤前总数: {total_unfiltered}")
        print(f"过滤后总数: {total_filtered}")
        print(f"被过滤数量: {filtered_out}")
        print(f"过滤率: {filtered_out/total_unfiltered*100:.1f}%" if total_unfiltered > 0 else "0%")
        
        # 验证结果
        if filtered_negative_count == 0:
            print("✅ 极性过滤功能正常工作，所有负面模式已被过滤")
        else:
            print(f"⚠️ 仍有 {filtered_negative_count} 个负面模式未被过滤")
        
        if filtered_out > 0:
            print("✅ 过滤功能已生效，成功过滤了一些模式")
        else:
            print("⚠️ 过滤功能可能未生效，没有模式被过滤")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_specific_patterns():
    """测试特定的负面模式"""
    print("\n🎯 测试特定负面模式识别")
    print("=" * 50)
    
    analyzer = BuyPointBatchAnalyzer()
    filter_obj = analyzer.polarity_filter
    
    # 测试案例
    test_cases = [
        ("ZXMBSAbsorb", "无吸筹信号", "应该被识别为负面"),
        ("ZXMWeeklyKDJDTrendUp", "周线KDJ死叉后持续下行", "应该被识别为负面"),
        ("TrendDetector", "虚弱上升趋势", "应该被识别为负面"),
        ("Chaikin", "CHAIKIN_FALLING", "应该被识别为负面"),
        ("MACD", "MACD_HIST_NEGATIVE", "应该被识别为负面"),
        ("MA", "MA_BULLISH_ARRANGEMENT", "应该被识别为正面"),
        ("MACD", "MACD_GOLDEN_CROSS", "应该被识别为正面"),
        ("Volume", "VOLUME_SURGE", "应该被识别为正面"),
    ]
    
    for indicator_name, pattern_name, expected in test_cases:
        is_negative = filter_obj.is_negative_pattern(indicator_name, pattern_name)
        status = "负面" if is_negative else "正面/中性"
        result = "✅" if (is_negative and "负面" in expected) or (not is_negative and "正面" in expected) else "❌"
        
        print(f"{result} {indicator_name} - {pattern_name}: {status} ({expected})")


if __name__ == "__main__":
    test_polarity_filter()
    test_specific_patterns()
