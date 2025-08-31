#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查找下一个需要进行5阶段验证的指标
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

def get_all_registered_indicators():
    """获取所有已注册的指标"""
    try:
        from indicators.complete_indicator_registry import complete_registry
        all_indicators = list(complete_registry.get_all_indicators().keys())
        return sorted(all_indicators)
    except Exception as e:
        print(f"❌ 获取已注册指标失败: {e}")
        return []

def get_validated_indicators():
    """获取已完成5阶段验证的指标"""
    validated_indicators = []
    validation_reports_dir = Path("docs/finaltesting/indicators")
    
    if validation_reports_dir.exists():
        for report_file in validation_reports_dir.glob("*_validation_report.md"):
            indicator_name = report_file.stem.replace("_validation_report", "")
            validated_indicators.append(indicator_name)
    
    return sorted(validated_indicators)

def find_next_indicators_to_validate():
    """找出下一批需要验证的指标"""
    print("🔍 查找下一个需要进行5阶段验证的指标...")
    print("=" * 60)
    
    # 获取所有已注册指标
    all_indicators = get_all_registered_indicators()
    print(f"📊 系统中已注册指标总数: {len(all_indicators)}")
    
    # 获取已验证指标
    validated_indicators = get_validated_indicators()
    print(f"✅ 已完成5阶段验证的指标数: {len(validated_indicators)}")
    
    # 找出待验证指标
    pending_indicators = [ind for ind in all_indicators if ind not in validated_indicators]
    print(f"⏸️ 待验证指标数: {len(pending_indicators)}")
    
    # 计算完成率
    completion_rate = (len(validated_indicators) / len(all_indicators)) * 100 if all_indicators else 0
    print(f"📈 5阶段验证完成率: {completion_rate:.1f}%")
    
    print(f"\n📋 已完成验证的指标 ({len(validated_indicators)}个):")
    for i, indicator in enumerate(validated_indicators, 1):
        print(f"  {i:2d}. {indicator}")
    
    print(f"\n⏸️ 待验证指标列表 ({len(pending_indicators)}个):")
    
    # 按优先级分类显示待验证指标
    priority_categories = {
        'P0_核心指标': ['MA', 'EMA', 'RSI', 'MACD', 'BOLL', 'KDJ'],
        'P1_重要指标': ['DMI', 'CCI', 'STOCHRSI', 'TRIX', 'WR', 'OBV', 'MFI', 'ATR', 'SAR', 'ADX'],
        'P2_常用指标': ['ROC', 'CMO', 'AROON', 'ICHIMOKU', 'WMA', 'VORTEX', 'EMV', 'KC', 'VIX', 'VOLUME_SCORE'],
        'P3_专业指标': ['ENHANCED_CCI', 'ENHANCED_DMI', 'ENHANCED_RSI', 'ENHANCED_KDJ', 'ENHANCED_BOLL', 'ENHANCED_STOCHRSI'],
        'P4_ZXM系列': [ind for ind in pending_indicators if ind.startswith('ZXM_')],
        'P5_评分系列': [ind for ind in pending_indicators if ind.endswith('_SCORE')],
        'P7_形态识别': ['CANDLESTICK_PATTERNS', 'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI']
    }
    
    next_indicator = None
    next_priority = None
    
    for priority, category_indicators in priority_categories.items():
        category_pending = [ind for ind in category_indicators if ind in pending_indicators]
        if category_pending:
            print(f"\n{priority} 待验证指标 ({len(category_pending)}个):")
            for i, indicator in enumerate(category_pending, 1):
                print(f"  {i:2d}. {indicator}")
                if next_indicator is None:  # 选择第一个待验证指标
                    next_indicator = indicator
                    next_priority = priority
    
    # 显示其他待验证指标
    other_pending = []
    for ind in pending_indicators:
        found_in_category = False
        for category_indicators in priority_categories.values():
            if ind in category_indicators:
                found_in_category = True
                break
        if not found_in_category:
            other_pending.append(ind)
    
    if other_pending:
        print(f"\n其他待验证指标 ({len(other_pending)}个):")
        for i, indicator in enumerate(other_pending, 1):
            print(f"  {i:2d}. {indicator}")
            if next_indicator is None:  # 如果前面没有找到，选择第一个其他指标
                next_indicator = indicator
                next_priority = "其他"
    
    print(f"\n" + "=" * 60)
    print(f"🎯 建议验证的下一个指标")
    print(f"=" * 60)
    
    if next_indicator:
        print(f"指标名称: {next_indicator}")
        print(f"优先级别: {next_priority}")
        print(f"验证进度: {len(validated_indicators)}/{len(all_indicators)} ({completion_rate:.1f}%)")
        
        # 检查指标是否存在
        try:
            indicator_instance = complete_registry.create_indicator(next_indicator)
            print(f"指标状态: ✅ 可用")
            print(f"指标类型: {type(indicator_instance).__name__}")
        except Exception as e:
            print(f"指标状态: ⚠️ 需要检查 - {e}")
        
        print(f"\n📝 下一步操作:")
        print(f"1. 运行5阶段验证: python validation/{next_indicator.lower()}_stage_validation.py")
        print(f"2. 或使用通用验证器: python scripts/universal_5stage_validator.py --indicator {next_indicator}")
        print(f"3. 验证完成后更新进度表")
        
        return next_indicator, next_priority
    else:
        print("🎉 所有指标已完成5阶段验证！")
        return None, None

def main():
    """主函数"""
    next_indicator, priority = find_next_indicators_to_validate()
    
    if next_indicator:
        print(f"\n🚀 准备验证指标: {next_indicator}")
        return next_indicator
    else:
        print(f"\n🎉 验证工作已全部完成！")
        return None

if __name__ == "__main__":
    main()
