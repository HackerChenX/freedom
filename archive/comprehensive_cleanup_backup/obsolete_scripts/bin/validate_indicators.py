#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标验证执行脚本

便捷的指标验证工具，支持多种验证模式
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.indicator_validation_framework import IndicatorValidationFramework
import json
import argparse
from datetime import datetime


def validate_priority_indicators():
    """验证优先级指标"""
    print("🎯 开始验证优先级指标")
    
    # 加载配置
    config_file = os.path.join(root_dir, "config", "indicator_validation_config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    priority_indicators = config.get("priority_indicators", [])
    
    results = []
    for indicator in priority_indicators:
        print(f"\n🔍 验证指标: {indicator}")
        result = framework.validate_single_indicator(indicator)
        results.append(result)
        
        # 显示结果
        if result.get("status") == "success":
            analysis = result.get("analysis", {})
            status = analysis.get("validation_status", "unknown")
            selection_rate = analysis.get("selection_rate", 0)
            
            if status == "success":
                print(f"   ✅ 验证成功 - 选股率: {selection_rate:.1%}")
            elif status == "no_selection":
                print(f"   ⚠️ 无选股")
            elif status == "over_selection":
                print(f"   ⚠️ 过度选股 - 选股率: {selection_rate:.1%}")
            else:
                print(f"   ❌ 验证失败: {status}")
        else:
            print(f"   ❌ 执行失败: {result.get('error', '未知错误')}")
    
    # 生成摘要
    successful = sum(1 for r in results if r.get("analysis", {}).get("validation_status") == "success")
    print(f"\n🎯 优先级指标验证完成: {successful}/{len(priority_indicators)} 成功")
    
    return results


def validate_by_category(category: str):
    """按类别验证指标"""
    print(f"📋 开始验证 {category} 类别指标")
    
    # 加载配置
    config_file = os.path.join(root_dir, "config", "indicator_validation_config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    # 获取类别指标
    category_indicators = config.get("test_categories", {}).get(category, [])
    
    if not category_indicators:
        print(f"❌ 未找到 {category} 类别的指标配置")
        return []
    
    results = []
    for indicator in category_indicators:
        print(f"\n🔍 验证指标: {indicator}")
        result = framework.validate_single_indicator(indicator)
        results.append(result)
        
        # 显示结果
        if result.get("status") == "success":
            analysis = result.get("analysis", {})
            status = analysis.get("validation_status", "unknown")
            selection_rate = analysis.get("selection_rate", 0)
            
            if status == "success":
                print(f"   ✅ 验证成功 - 选股率: {selection_rate:.1%}")
            elif status == "no_selection":
                print(f"   ⚠️ 无选股")
            elif status == "over_selection":
                print(f"   ⚠️ 过度选股 - 选股率: {selection_rate:.1%}")
            else:
                print(f"   ❌ 验证失败: {status}")
        else:
            print(f"   ❌ 执行失败: {result.get('error', '未知错误')}")
    
    # 生成摘要
    successful = sum(1 for r in results if r.get("analysis", {}).get("validation_status") == "success")
    print(f"\n📋 {category} 类别验证完成: {successful}/{len(category_indicators)} 成功")
    
    return results


def quick_test():
    """快速测试几个核心指标"""
    print("⚡ 开始快速测试核心指标")
    
    framework = IndicatorValidationFramework()
    
    # 快速测试指标列表
    test_indicators = ["MA", "MACD", "RSI", "BOLL", "KDJ"]
    
    results = []
    for indicator in test_indicators:
        print(f"\n🔍 测试指标: {indicator}")
        result = framework.validate_single_indicator(indicator)
        results.append(result)
        
        # 显示结果
        if result.get("status") == "success":
            analysis = result.get("analysis", {})
            status = analysis.get("validation_status", "unknown")
            selection_rate = analysis.get("selection_rate", 0)
            
            if status == "success":
                print(f"   ✅ 测试通过 - 选股率: {selection_rate:.1%}")
            else:
                print(f"   ⚠️ 测试问题: {status}")
        else:
            print(f"   ❌ 测试失败: {result.get('error', '未知错误')}")
    
    # 生成摘要
    successful = sum(1 for r in results if r.get("analysis", {}).get("validation_status") == "success")
    print(f"\n⚡ 快速测试完成: {successful}/{len(test_indicators)} 通过")
    
    return results


def full_validation():
    """完整验证所有指标"""
    print("🚀 开始完整验证所有指标")
    
    # 加载配置
    config_file = os.path.join(root_dir, "config", "indicator_validation_config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    # 执行完整验证
    output_dir = f"results/full_indicator_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = framework.validate_all_indicators(output_dir)
    
    return result


def main_9():
    """主函数"""
    parser = argparse.ArgumentParser(description="指标验证执行脚本")
    parser.add_argument("--mode", type=str, choices=["quick", "priority", "category", "full"], 
                       default="quick", help="验证模式")
    parser.add_argument("--category", type=str, choices=["basic", "enhanced", "volume", "zxm"],
                       help="验证类别（仅在category模式下使用）")
    parser.add_argument("--indicator", type=str, help="验证单个指标")
    parser.add_argument("--output", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    print("🔧 指标验证工具")
    print("=" * 50)
    
    if args.indicator:
        # 验证单个指标
        framework = IndicatorValidationFramework()
        result = framework.validate_single_indicator(args.indicator)
        
        print(f"\n📊 {args.indicator} 验证结果:")
        if result.get("status") == "success":
            analysis = result.get("analysis", {})
            print(f"验证状态: {analysis.get('validation_status', 'unknown')}")
            print(f"选股率: {analysis.get('selection_rate', 0):.1%}")
            print(f"质量分数: {analysis.get('quality_score', 0):.2f}")
            print(f"执行时间: {result.get('execution_time', 0):.2f}秒")
        else:
            print(f"验证失败: {result.get('error', '未知错误')}")
    
    elif args.mode == "quick":
        # 快速测试
        quick_test()
    
    elif args.mode == "priority":
        # 验证优先级指标
        validate_priority_indicators()
    
    elif args.mode == "category":
        # 按类别验证
        if not args.category:
            print("❌ category模式需要指定--category参数")
            return
        validate_by_category(args.category)
    
    elif args.mode == "full":
        # 完整验证
        result = full_validation()
        print(f"\n🎯 完整验证结果:")
        print(f"总指标数: {result.get('total_indicators', 0)}")
        print(f"成功验证: {result.get('successful_count', 0)}")
        print(f"失败验证: {result.get('failed_count', 0)}")
        print(f"成功率: {result.get('successful_count', 0) / result.get('total_indicators', 1):.1%}")
        print(f"结果保存到: {result.get('output_directory', 'N/A')}")
    
    print("\n✅ 验证完成")


if __name__ == "__main__":
    main_9() 