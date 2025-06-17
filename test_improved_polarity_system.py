#!/usr/bin/env python3
"""
测试改进后的极性分类系统

验证基于注册信息的极性过滤是否正确工作，确保买点分析只包含正面模式。
"""

import sys
import os
sys.path.append('.')

from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from indicators.pattern_registry import PatternRegistry, PatternPolarity
import pandas as pd

def test_improved_polarity_system():
    """测试改进后的极性分类系统"""
    print("🧪 测试改进后的极性分类系统")
    print("=" * 60)
    
    # 创建分析器
    analyzer = BuyPointBatchAnalyzer()
    
    # 初始化指标以注册模式
    from indicators.macd import MACD
    from indicators.ma import MA
    from indicators.rsi import RSI
    from indicators.kdj import KDJ
    from indicators.boll import BOLL
    
    indicators = [MACD(), MA(periods=[5, 10, 20]), RSI(), KDJ(), BOLL()]
    print(f"已初始化 {len(indicators)} 个指标")
    
    # 获取注册表
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    print(f"注册表中共有 {len(all_patterns)} 个模式")
    
    # 统计各极性的模式数量
    polarity_stats = {}
    for pattern_id, pattern_info in all_patterns.items():
        polarity = pattern_info.get('polarity')
        if polarity:
            polarity_value = polarity.value
            polarity_stats[polarity_value] = polarity_stats.get(polarity_value, 0) + 1
    
    print(f"\n📊 极性分布:")
    for polarity, count in polarity_stats.items():
        print(f"  {polarity}: {count} 个模式")
    
    # 测试极性过滤器
    print(f"\n🔍 测试极性过滤器:")
    filter_obj = analyzer.polarity_filter
    
    # 测试一些具体的模式
    test_cases = [
        ("MACD", "MACD_GOLDEN_CROSS", "MACD金叉"),
        ("MACD", "MACD_DEATH_CROSS", "MACD死叉"),
        ("MA", "MA_BULLISH_ARRANGEMENT", "MA多头排列"),
        ("MA", "MA_BEARISH_ARRANGEMENT", "MA空头排列"),
        ("KDJ", "KDJ_OVERBOUGHT", "KDJ超买"),
        ("KDJ", "KDJ_OVERSOLD", "KDJ超卖"),
        ("BOLL", "PRICE_BREAK_LOWER", "布林带价格突破下轨"),
        ("RSI", "RSI_EXTREME_OVERSOLD", "RSI极度超卖"),
    ]
    
    for indicator_name, pattern_name, display_name in test_cases:
        is_negative = filter_obj.is_negative_pattern(indicator_name, pattern_name, display_name)
        status = "❌ 负面" if is_negative else "✅ 正面/中性"
        print(f"  {indicator_name}_{pattern_name}: {status}")
    
    # 读取实际数据进行测试
    print(f"\n📈 测试实际买点分析:")
    df = pd.read_csv('data/buypoints.csv')
    test_data = df.head(3).to_dict('records')
    print(f"使用 {len(test_data)} 个买点进行测试")
    
    try:
        # 分析买点
        results = analyzer.analyze_batch_buypoints(pd.DataFrame(test_data))
        print(f"分析完成，结果数量: {len(results)}")
        
        if not results:
            print("❌ 没有分析结果")
            return
        
        # 测试不过滤负面模式
        print(f"\n📊 不过滤负面模式的结果:")
        common_indicators_unfiltered = analyzer.extract_common_indicators(
            results, 
            min_hit_ratio=0.5, 
            filter_negative_patterns=False
        )
        
        total_unfiltered = sum(len(indicators) for indicators in common_indicators_unfiltered.values())
        print(f"总共性指标数量: {total_unfiltered}")
        
        # 统计负面模式（基于注册信息）
        negative_count_registry = 0
        negative_patterns_registry = []
        
        for period, indicators in common_indicators_unfiltered.items():
            for indicator in indicators:
                indicator_name = indicator['name']
                pattern_name = indicator.get('pattern', '')
                
                # 从注册表获取极性
                pattern_id = f"{indicator_name}_{pattern_name}"
                pattern_info = registry.get_pattern(pattern_id)
                
                if pattern_info and pattern_info.get('polarity') == PatternPolarity.NEGATIVE:
                    negative_count_registry += 1
                    negative_patterns_registry.append(f"{indicator_name}_{pattern_name}")
        
        print(f"基于注册信息检测到的负面模式: {negative_count_registry}")
        if negative_patterns_registry:
            print("负面模式列表:")
            for pattern in negative_patterns_registry[:5]:  # 显示前5个
                print(f"  - {pattern}")
        
        # 测试过滤负面模式
        print(f"\n🔍 过滤负面模式的结果:")
        common_indicators_filtered = analyzer.extract_common_indicators(
            results, 
            min_hit_ratio=0.5, 
            filter_negative_patterns=True
        )
        
        total_filtered = sum(len(indicators) for indicators in common_indicators_filtered.values())
        print(f"总共性指标数量: {total_filtered}")
        
        # 验证过滤后是否还有负面模式
        remaining_negative = 0
        remaining_negative_patterns = []
        
        for period, indicators in common_indicators_filtered.items():
            for indicator in indicators:
                indicator_name = indicator['name']
                pattern_name = indicator.get('pattern', '')
                
                # 从注册表获取极性
                pattern_id = f"{indicator_name}_{pattern_name}"
                pattern_info = registry.get_pattern(pattern_id)
                
                if pattern_info and pattern_info.get('polarity') == PatternPolarity.NEGATIVE:
                    remaining_negative += 1
                    remaining_negative_patterns.append(f"{indicator_name}_{pattern_name}")
        
        print(f"过滤后剩余负面模式: {remaining_negative}")
        if remaining_negative_patterns:
            print("剩余负面模式:")
            for pattern in remaining_negative_patterns:
                print(f"  ⚠️ {pattern}")
        
        # 统计结果
        filtered_out = total_unfiltered - total_filtered
        print(f"\n📈 过滤统计:")
        print(f"过滤前总数: {total_unfiltered}")
        print(f"过滤后总数: {total_filtered}")
        print(f"被过滤数量: {filtered_out}")
        print(f"过滤率: {filtered_out/total_unfiltered*100:.1f}%" if total_unfiltered > 0 else "0%")
        print(f"负面模式过滤率: {(negative_count_registry-remaining_negative)/negative_count_registry*100:.1f}%" if negative_count_registry > 0 else "N/A")
        
        # 验证结果
        print(f"\n✅ 验证结果:")
        if remaining_negative == 0:
            print("🎉 极性过滤系统完美工作！所有负面模式已被过滤")
        else:
            print(f"⚠️ 仍有 {remaining_negative} 个负面模式未被过滤，需要进一步检查")
        
        if filtered_out > 0:
            print("✅ 过滤功能正常，成功过滤了一些模式")
        else:
            print("⚠️ 过滤功能可能未生效")
            
        # 显示保留的正面模式示例
        positive_patterns = []
        for period, indicators in common_indicators_filtered.items():
            for indicator in indicators[:3]:  # 每个周期显示前3个
                indicator_name = indicator['name']
                pattern_name = indicator.get('pattern', '')
                hit_ratio = indicator.get('hit_ratio', 0)
                positive_patterns.append(f"{indicator_name}_{pattern_name} ({hit_ratio:.1f}%)")
        
        if positive_patterns:
            print(f"\n🌟 保留的正面模式示例:")
            for pattern in positive_patterns[:8]:  # 显示前8个
                print(f"  ✅ {pattern}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_improved_polarity_system()
