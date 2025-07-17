#!/usr/bin/env python3
from config import get_config
"""
指标验证框架演示脚本

演示如何使用指标验证框架逐个验证每个指标
"""

import os
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import (
    Indicator_validation_framework,
    Indicator_validation_config,
    Validation_mode
)
from utils.logger import get_logger

logger = get_logger(__name__)


def demo_quick_validation():
    """演示快速验证模式"""
    print("\n" + "="*60)
    print("快速验证模式演示")
    print("="*60)
    
    # 配置快速验证
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_get_config('performance.pool_size'),
        max_selection_ratio=0.08,
        parallel_workers=1,
        output_format="json",
        save_details=True
    )
    
    # 创建验证框架
    framework = Indicator_validation_framework(config)
    
    try:
        # 执行验证
        results = framework.validate_all_indicators()
        
        # 显示结果
        print(f"验证完成！")
        print(f"总指标数: {results['summary']['total_indicators']}")
        print(f"成功验证: {results['summary']['validation_results']['success']}")
        print(f"无选股: {results['summary']['validation_results']['no_selection']}")
        print(f"过度选择: {results['summary']['validation_results']['over_selection']}")
        print(f"验证错误: {results['summary']['validation_results']['error']}")
        print(f"成功率: {results['summary']['success_rate']:.2%}")
        print(f"平均选股率: {results['summary']['average_selection_ratio']:.2%}")
        print(f"耗时: {results['stats']['duration']:.2f}秒")
        
        # 显示表现最好的指标
        if results['summary']['top_performing_indicators']:
            print(f"\n表现最好的指标:")
            for i, indicator in enumerate(results['summary']['top_performing_indicators'][:5], 1):
                print(f"  {i}. {indicator['indicator']}: 选出{indicator['selected_count']}只股票 "
                      f"(选股率{indicator['selection_ratio']:.2%})")
        
        # 显示失败的指标
        if results['summary']['failed_indicators']:
            print(f"\n验证失败的指标:")
            for indicator in results['summary']['failed_indicators'][:5]:
                print(f"  - {indicator['indicator']}: {indicator['status']} - {indicator.get('error', '未知错误')}")
        
        return results
        
    except Exception as e:
        print(f"快速验证失败: {e}")
        logger.error(f"快速验证失败: {e}")
        return None


def demo_single_indicator_validation():
    """演示单个指标验证"""
    print("\n" + "="*60)
    print("单个指标验证演示")
    print("="*60)
    
    # 配置验证
    config = Indicator_validation_config(
        stock_get_config('performance.pool_size'),
        max_selection_ratio=0.05,
        save_details=False
    )
    
    # 创建验证框架
    framework = Indicator_validation_framework(config)
    
    # 测试几个核心指标
    test_indicators = ['MA', 'MACD', 'RSI', 'KDJ', 'BOLL']
    
    for indicator in test_indicators:
        print(f"\n验证指标: {indicator}")
        print("-" * 40)
        
        try:
            result = framework.validate_single_indicator(indicator)
            
            print(f"状态: {result['status']}")
            print(f"选股数量: {result.get('selected_count', 0)}")
            print(f"选股率: {result.get('selection_ratio', 0):.2%}")
            print(f"执行时间: {result.get('execution_time', 0):.2f}秒")
            
            if result.get('selected_stocks'):
                print(f"选出的股票（前10只）: {result['selected_stocks'][:10]}")
            
            if result.get('error_message'):
                print(f"错误信息: {result['error_message']}")
                
        except Exception as e:
            print(f"验证 {indicator} 时出错: {e}")
            logger.error(f"验证 {indicator} 时出错: {e}")


def demo_category_validation():
    """演示分类验证模式"""
    print("\n" + "="*60)
    print("分类验证模式演示")
    print("="*60)
    
    # 配置分类验证
    config = Indicator_validation_config(
        mode=Validation_mode.CATEGORY,
        stock_get_config('performance.pool_size'),
        max_selection_ratio=0.06,
        parallel_workers=2,
        output_format="txt",
        save_details=True
    )
    
    # 创建验证框架
    framework = Indicator_validation_framework(config)
    
    try:
        # 执行验证
        results = framework.validate_all_indicators()
        
        # 按类别统计结果
        category_stats = {}
        for result in results['results']:
            indicator_name = result['indicator_name']
            status = result['status']
            
            # 简单分类
            if indicator_name.startswith('ZXM_'):
                category = 'ZXM指标'
            elif indicator_name.startswith('ENHANCED_'):
                category = '增强指标'
            elif indicator_name in ['MA', 'EMA', 'MACD', 'RSI', 'KDJ', 'BOLL']:
                category = '基础指标'
            else:
                category = '其他指标'
            
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'success': 0, 'failed': 0}
            
            category_stats[category]['total'] += 1
            if status == 'success':
                category_stats[category]['success'] += 1
            else:
                category_stats[category]['failed'] += 1
        
        # 显示分类统计
        print(f"\n分类验证结果:")
        for category, stats in category_stats.items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {category}: {stats['success']}/{stats['total']} "
                  f"(成功率: {success_rate:.1%})")
        
        return results
        
    except Exception as e:
        print(f"分类验证失败: {e}")
        logger.error(f"分类验证失败: {e}")
        return None


def demo_priority_validation():
    """演示优先级验证模式"""
    print("\n" + "="*60)
    print("优先级验证模式演示")
    print("="*60)
    
    # 配置优先级验证
    config = Indicator_validation_config(
        mode=Validation_mode.PRIORITY,
        stock_get_config('performance.pool_size'),
        max_selection_ratio=0.07,
        parallel_workers=1,
        save_details=False
    )
    
    # 创建验证框架
    framework = Indicator_validation_framework(config)
    
    try:
        # 只验证前10个优先级最高的指标
        indicators = framework._get_indicators_by_mode()[:10]
        
        print(f"验证前10个优先级指标: {indicators}")
        
        results = []
        for i, indicator in enumerate(indicators, 1):
            print(f"\n[{i}/10] 验证指标: {indicator}")
            
            result = framework.validate_single_indicator(indicator)
            results.append(result)
            
            print(f"  状态: {result['status']}")
            if result['status'] == 'success':
                print(f"  选股: {result.get('selected_count', 0)}只 "
                      f"(选股率: {result.get('selection_ratio', 0):.2%})")
            elif result.get('error_message'):
                print(f"  错误: {result['error_message']}")
        
        # 统计优先级验证结果
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"\n优先级验证总结:")
        print(f"  验证指标: {len(results)}")
        print(f"  成功验证: {success_count}")
        print(f"  成功率: {success_count/len(results):.1%}")
        
        return results
        
    except Exception as e:
        print(f"优先级验证失败: {e}")
        logger.error(f"优先级验证失败: {e}")
        return None


def main_indicatorvalidationdemo():
    """主函数"""
    print("指标验证框架演示")
    print("="*60)
    print("本演示将展示如何使用指标验证框架逐个验证每个指标是否能通过策略选股系统选出来")
    
    try:
        # 1. 快速验证演示
        quick_results = demo_quick_validation()
        
        # 2. 单个指标验证演示
        demo_single_indicator_validation()
        
        # 3. 分类验证演示
        category_results = demo_category_validation()
        
        # 4. 优先级验证演示
        priority_results = demo_priority_validation()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        print("您可以根据需要选择不同的验证模式：")
        print("- QUICK: 快速验证核心指标")
        print("- PRIORITY: 按重要性优先级验证")
        print("- CATEGORY: 按指标类型分类验证")
        print("- FULL: 验证所有指标")
        print("\n支持的输出格式：JSON、CSV、TXT")
        print("支持并行验证以提高效率")
        print("提供详细的验证报告和统计信息")
        
    except Exception as e:
        print(f"演示过程出错: {e}")
        logger.error(f"演示过程出错: {e}")


if __name__ == "__main__":
    main_indicatorvalidationdemo() 