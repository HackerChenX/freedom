#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速性能测试工具

简单快速的性能分析工具
"""

import os
import sys
import time
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer

logger = get_logger(__name__)


def quick_performance_test(input_csv: str, sample_size: int = 3):
    """
    快速性能测试
    
    Args:
        input_csv: 输入CSV文件路径
        sample_size: 采样大小
    """
    print("="*60)
    print("股票分析系统快速性能测试")
    print("="*60)
    
    # 加载买点数据
    print(f"1. 加载买点数据: {input_csv}")
    analyzer = BuyPointBatchAnalyzer()
    buypoints_df = analyzer.load_buypoints_from_csv(input_csv)
    
    if buypoints_df.empty:
        print("❌ 无法加载买点数据")
        return
    
    # 限制采样大小
    if len(buypoints_df) > sample_size:
        buypoints_df = buypoints_df.head(sample_size)
    
    print(f"   ✅ 成功加载 {len(buypoints_df)} 个买点")
    
    # 测试各个阶段的性能
    results = {}
    
    print(f"\n2. 开始性能测试（采样 {len(buypoints_df)} 个买点）")
    
    # 总体批量处理测试
    print("   测试批量处理性能...")
    start_time = time.time()
    batch_results = analyzer.analyze_batch_buypoints(buypoints_df)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_stock = total_time / len(buypoints_df) if len(buypoints_df) > 0 else 0
    success_rate = len(batch_results) / len(buypoints_df) if len(buypoints_df) > 0 else 0
    
    results['batch_processing'] = {
        'total_time': total_time,
        'avg_time_per_stock': avg_time_per_stock,
        'success_rate': success_rate,
        'successful_analyses': len(batch_results)
    }
    
    print(f"   ✅ 批量处理完成: {total_time:.2f}秒")
    print(f"      - 平均每股: {avg_time_per_stock:.2f}秒")
    print(f"      - 成功率: {success_rate:.1%}")
    
    # 单个买点详细分析
    if not buypoints_df.empty:
        print("\n   测试单个买点各阶段性能...")
        first_row = buypoints_df.iloc[0]
        stock_code = first_row['stock_code']
        buypoint_date = first_row['buypoint_date']
        
        # 数据加载阶段
        data_start = time.time()
        stock_data = analyzer.data_processor.get_multi_period_data(
            stock_code=stock_code,
            end_date=buypoint_date
        )
        data_end = time.time()
        data_time = data_end - data_start
        
        # 指标计算阶段
        if stock_data:
            indicator_start = time.time()
            target_rows = {period: len(df) - 1 for period, df in stock_data.items() if df is not None and not df.empty}
            indicator_results = analyzer.indicator_analyzer.analyze_all_indicators(
                stock_data, target_rows
            )
            indicator_end = time.time()
            indicator_time = indicator_end - indicator_start
            
            results['stage_analysis'] = {
                'data_loading': data_time,
                'indicator_calculation': indicator_time,
                'data_loading_percentage': (data_time / (data_time + indicator_time)) * 100,
                'indicator_calculation_percentage': (indicator_time / (data_time + indicator_time)) * 100
            }
            
            print(f"      - 数据加载: {data_time:.2f}秒 ({(data_time / (data_time + indicator_time)) * 100:.1f}%)")
            print(f"      - 指标计算: {indicator_time:.2f}秒 ({(indicator_time / (data_time + indicator_time)) * 100:.1f}%)")
    
    # 指标性能分析
    print("\n   测试指标计算性能...")
    if stock_data:
        indicator_performance = {}
        all_indicators = analyzer.indicator_analyzer.all_indicators
        
        # 测试前10个指标的性能
        test_indicators = list(all_indicators)[:10]
        
        for indicator_name in test_indicators:
            try:
                start_time = time.time()
                
                # 创建指标实例
                indicator = analyzer.indicator_analyzer.indicator_registry.create_indicator(indicator_name)
                if indicator is None:
                    indicator = analyzer.indicator_analyzer.indicator_factory.create_indicator(indicator_name)
                
                if indicator is None:
                    continue
                
                # 计算指标（只测试日线数据）
                if 'daily' in stock_data and stock_data['daily'] is not None:
                    df_copy = stock_data['daily'].copy()
                    indicator_df = indicator.calculate(df_copy)
                
                end_time = time.time()
                calc_time = end_time - start_time
                indicator_performance[indicator_name] = calc_time
                
            except Exception as e:
                indicator_performance[indicator_name] = f"Error: {str(e)}"
        
        # 排序并显示最慢的指标
        valid_indicators = {k: v for k, v in indicator_performance.items() if isinstance(v, (int, float))}
        if valid_indicators:
            sorted_indicators = sorted(valid_indicators.items(), key=lambda x: x[1], reverse=True)
            
            print(f"      测试了 {len(test_indicators)} 个指标，成功 {len(valid_indicators)} 个")
            print("      最耗时的5个指标:")
            for i, (name, time_val) in enumerate(sorted_indicators[:5], 1):
                print(f"        {i}. {name}: {time_val:.3f}秒")
            
            results['indicator_performance'] = {
                'tested_indicators': len(test_indicators),
                'successful_indicators': len(valid_indicators),
                'avg_time_per_indicator': sum(valid_indicators.values()) / len(valid_indicators),
                'slowest_indicators': dict(sorted_indicators[:5])
            }
    
    # 生成优化建议
    print("\n3. 性能分析和优化建议")
    recommendations = generate_quick_recommendations(results)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['category']} ({rec['priority']})")
        print(f"      问题: {rec['issue']}")
        print(f"      建议: {rec['recommendation']}")
        print(f"      预期改进: {rec['expected_improvement']}")
        print()
    
    # 保存结果
    output_dir = "data/result/quick_performance_test"
    os.makedirs(output_dir, exist_ok=True)
    
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'test_info': {
            'input_file': input_csv,
            'sample_size': len(buypoints_df)
        },
        'performance_results': results,
        'recommendations': recommendations
    }
    
    results_path = os.path.join(output_dir, 'quick_performance_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"4. 测试完成，结果已保存到: {results_path}")
    print("="*60)


def generate_quick_recommendations(results):
    """生成快速优化建议"""
    recommendations = []
    
    # 批量处理性能建议
    if 'batch_processing' in results:
        batch = results['batch_processing']
        avg_time = batch.get('avg_time_per_stock', 0)
        
        if avg_time > 60:
            recommendations.append({
                'category': '批量处理优化',
                'priority': 'HIGH',
                'issue': f'平均每股分析时间过长: {avg_time:.1f}秒',
                'recommendation': '实施并行处理，使用多进程',
                'expected_improvement': '50-80%性能提升'
            })
        elif avg_time > 30:
            recommendations.append({
                'category': '批量处理优化',
                'priority': 'MEDIUM',
                'issue': f'平均每股分析时间较长: {avg_time:.1f}秒',
                'recommendation': '优化指标计算和缓存机制',
                'expected_improvement': '20-40%性能提升'
            })
    
    # 阶段分析建议
    if 'stage_analysis' in results:
        stage = results['stage_analysis']
        indicator_percentage = stage.get('indicator_calculation_percentage', 0)
        
        if indicator_percentage > 80:
            recommendations.append({
                'category': '指标计算优化',
                'priority': 'HIGH',
                'issue': f'指标计算占用 {indicator_percentage:.1f}% 的时间',
                'recommendation': '优化指标算法，使用向量化计算',
                'expected_improvement': '30-60%性能提升'
            })
    
    # 指标性能建议
    if 'indicator_performance' in results:
        indicator = results['indicator_performance']
        avg_time = indicator.get('avg_time_per_indicator', 0)
        
        if avg_time > 0.1:
            recommendations.append({
                'category': '指标算法优化',
                'priority': 'MEDIUM',
                'issue': f'平均每指标计算时间: {avg_time:.3f}秒',
                'recommendation': '使用numpy向量化操作替代循环',
                'expected_improvement': '40-70%指标计算性能提升'
            })
    
    # 系统级建议
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    recommendations.append({
        'category': '并行处理',
        'priority': 'HIGH',
        'issue': f'当前串行处理，系统有 {cpu_count} 个CPU核心',
        'recommendation': f'实施 {cpu_count} 进程并行处理',
        'expected_improvement': f'理论上可获得 {cpu_count}x 性能提升'
    })
    
    return recommendations


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python quick_performance_test.py <buypoints.csv> [sample_size]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    quick_performance_test(input_csv, sample_size)
