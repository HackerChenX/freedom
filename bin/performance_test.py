#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统性能测试工具

对比原始版本和优化版本的性能差异
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger, setup_logger
from utils.path_utils import ensure_dir_exists
from analysis.performance_analyzer import PerformanceAnalyzer
from analysis.optimized_batch_analyzer import OptimizedBatchAnalyzer
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer

logger = get_logger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票分析系统性能测试工具')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入的CSV文件路径，包含股票代码和买点日期')
    
    parser.add_argument('--output', '-o', type=str, default='data/result/performance_test',
                        help='输出结果的目录路径')
    
    parser.add_argument('--sample-size', '-s', type=int, default=5,
                        help='性能测试的采样大小，默认为5')
    
    parser.add_argument('--test-type', '-t', type=str, default='all',
                        choices=['all', 'batch', 'indicator', 'data', 'memory'],
                        help='测试类型，默认为all（全部测试）')
    
    parser.add_argument('--compare-optimized', '-c', action='store_true',
                        help='是否对比优化版本的性能')
    
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别，默认为INFO')
    
    return parser.parse_args()


def run_performance_analysis(input_csv: str, 
                           output_dir: str,
                           sample_size: int,
                           test_type: str) -> dict:
    """
    运行性能分析
    
    Args:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
        sample_size: 采样大小
        test_type: 测试类型
        
    Returns:
        dict: 性能分析结果
    """
    logger.info(f"开始性能分析，测试类型: {test_type}, 采样大小: {sample_size}")
    
    analyzer = PerformanceAnalyzer()
    results = {}
    
    # 批量处理性能测试
    if test_type in ['all', 'batch']:
        logger.info("执行批量处理性能测试...")
        batch_result = analyzer.analyze_batch_processing_performance(
            buypoints_csv=input_csv,
            sample_size=sample_size
        )
        results['batch_processing'] = batch_result
        
        # 保存详细的性能分析报告
        if batch_result.get('performance_stats'):
            profile_output = batch_result['performance_stats'].get('profile_output', '')
            if profile_output:
                profile_path = os.path.join(output_dir, 'batch_processing_profile.txt')
                with open(profile_path, 'w', encoding='utf-8') as f:
                    f.write(profile_output)
                logger.info(f"性能分析报告已保存到: {profile_path}")
    
    # 指标计算性能测试
    if test_type in ['all', 'indicator']:
        logger.info("执行指标计算性能测试...")
        
        # 获取测试数据
        base_analyzer = BuyPointBatchAnalyzer()
        buypoints_df = base_analyzer.load_buypoints_from_csv(input_csv)
        
        if not buypoints_df.empty:
            # 取第一个买点的数据进行测试
            first_row = buypoints_df.iloc[0]
            stock_data = base_analyzer.data_processor.get_multi_period_data(
                stock_code=first_row['stock_code'],
                end_date=first_row['buypoint_date']
            )
            
            if stock_data:
                target_rows = {period: len(df) - 1 for period, df in stock_data.items() if df is not None and not df.empty}
                indicator_result = analyzer.analyze_indicator_calculation_performance(
                    stock_data=stock_data,
                    target_rows=target_rows
                )
                results['indicator_calculation'] = indicator_result
    
    # 数据加载性能测试
    if test_type in ['all', 'data']:
        logger.info("执行数据加载性能测试...")
        
        base_analyzer = BuyPointBatchAnalyzer()
        buypoints_df = base_analyzer.load_buypoints_from_csv(input_csv)
        
        if not buypoints_df.empty:
            # 限制测试数据量
            test_df = buypoints_df.head(sample_size)
            stock_codes = test_df['stock_code'].tolist()
            end_dates = test_df['buypoint_date'].tolist()
            
            data_result = analyzer.analyze_data_loading_performance(
                stock_codes=stock_codes,
                end_dates=end_dates
            )
            results['data_loading'] = data_result
    
    # 内存使用分析
    if test_type in ['all', 'memory']:
        logger.info("执行内存使用分析...")
        memory_result = analyzer.analyze_memory_usage_patterns(
            buypoints_csv=input_csv,
            sample_size=min(sample_size, 3)  # 内存测试使用较小样本
        )
        results['memory_usage'] = memory_result
    
    return results


def run_optimized_comparison(input_csv: str, 
                           output_dir: str,
                           sample_size: int) -> dict:
    """
    运行优化版本对比测试
    
    Args:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
        sample_size: 采样大小
        
    Returns:
        dict: 对比测试结果
    """
    logger.info("开始优化版本对比测试...")
    
    # 加载测试数据
    base_analyzer = BuyPointBatchAnalyzer()
    buypoints_df = base_analyzer.load_buypoints_from_csv(input_csv)
    
    if buypoints_df.empty:
        logger.error("无法加载测试数据")
        return {}
    
    # 限制测试数据量
    test_df = buypoints_df.head(sample_size)
    logger.info(f"使用 {len(test_df)} 个买点进行对比测试")
    
    results = {}
    
    # 测试原始版本
    logger.info("测试原始版本性能...")
    original_start = time.time()
    original_results = base_analyzer.analyze_batch_buypoints(test_df)
    original_time = time.time() - original_start
    
    results['original'] = {
        'total_time': original_time,
        'avg_time_per_stock': original_time / len(test_df),
        'successful_analyses': len(original_results),
        'success_rate': len(original_results) / len(test_df)
    }
    
    # 测试优化版本
    logger.info("测试优化版本性能...")
    optimized_analyzer = OptimizedBatchAnalyzer()
    optimized_start = time.time()
    optimized_results = optimized_analyzer.analyze_batch_buypoints_optimized(test_df)
    optimized_time = time.time() - optimized_start
    
    results['optimized'] = {
        'total_time': optimized_time,
        'avg_time_per_stock': optimized_time / len(test_df),
        'successful_analyses': len(optimized_results),
        'success_rate': len(optimized_results) / len(test_df)
    }
    
    # 计算性能提升
    if original_time > 0:
        speedup = original_time / optimized_time
        improvement = (original_time - optimized_time) / original_time * 100
    else:
        speedup = 1.0
        improvement = 0.0
    
    results['comparison'] = {
        'speedup': speedup,
        'improvement_percentage': improvement,
        'time_saved': original_time - optimized_time
    }
    
    logger.info(f"性能对比完成: 加速比 {speedup:.2f}x, 性能提升 {improvement:.1f}%")
    
    return results


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    setup_logger(level=args.log_level)
    logger.info("股票分析系统性能测试工具启动")
    
    # 确保输出目录存在
    ensure_dir_exists(args.output)
    
    try:
        # 运行性能分析
        performance_results = run_performance_analysis(
            input_csv=args.input,
            output_dir=args.output,
            sample_size=args.sample_size,
            test_type=args.test_type
        )
        
        # 运行优化版本对比（如果启用）
        comparison_results = {}
        if args.compare_optimized:
            comparison_results = run_optimized_comparison(
                input_csv=args.input,
                output_dir=args.output,
                sample_size=args.sample_size
            )
        
        # 生成优化建议
        analyzer = PerformanceAnalyzer()
        recommendations = analyzer.generate_optimization_recommendations(performance_results)
        
        # 保存结果
        final_results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'input_file': args.input,
                'sample_size': args.sample_size,
                'test_type': args.test_type
            },
            'performance_analysis': performance_results,
            'optimization_comparison': comparison_results,
            'recommendations': recommendations
        }
        
        # 保存JSON结果
        results_path = os.path.join(args.output, 'performance_test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成性能报告
        report_path = os.path.join(args.output, 'performance_report.md')
        generate_performance_report(final_results, report_path)
        
        logger.info(f"性能测试完成，结果已保存到: {args.output}")
        print(f"\n性能测试结果: {results_path}")
        print(f"性能报告: {report_path}")
        
    except Exception as e:
        logger.error(f"性能测试过程中出错: {e}", exc_info=True)
        sys.exit(1)


def generate_performance_report(results: dict, output_path: str):
    """生成性能报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 股票分析系统性能测试报告\n\n")
        f.write(f"生成时间: {results['test_info']['timestamp']}\n\n")
        
        # 测试信息
        f.write("## 测试信息\n\n")
        f.write(f"- 输入文件: {results['test_info']['input_file']}\n")
        f.write(f"- 采样大小: {results['test_info']['sample_size']}\n")
        f.write(f"- 测试类型: {results['test_info']['test_type']}\n\n")
        
        # 性能分析结果
        if 'performance_analysis' in results:
            f.write("## 性能分析结果\n\n")
            perf = results['performance_analysis']
            
            if 'batch_processing' in perf:
                batch = perf['batch_processing']
                f.write("### 批量处理性能\n\n")
                f.write(f"- 总时间: {batch.get('total_time', 0):.2f}秒\n")
                f.write(f"- 平均每股时间: {batch.get('avg_time_per_stock', 0):.2f}秒\n")
                f.write(f"- 成功率: {batch.get('success_rate', 0):.1%}\n\n")
        
        # 优化对比结果
        if 'optimization_comparison' in results and results['optimization_comparison']:
            f.write("## 优化版本对比\n\n")
            comp = results['optimization_comparison']
            
            if 'comparison' in comp:
                comparison = comp['comparison']
                f.write(f"- 性能加速比: {comparison.get('speedup', 1):.2f}x\n")
                f.write(f"- 性能提升: {comparison.get('improvement_percentage', 0):.1f}%\n")
                f.write(f"- 节省时间: {comparison.get('time_saved', 0):.2f}秒\n\n")
        
        # 优化建议
        if 'recommendations' in results:
            f.write("## 优化建议\n\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"### {i}. {rec.get('category', '未知类别')}\n\n")
                f.write(f"- **优先级**: {rec.get('priority', 'MEDIUM')}\n")
                f.write(f"- **问题**: {rec.get('issue', '')}\n")
                f.write(f"- **建议**: {rec.get('recommendation', '')}\n")
                f.write(f"- **预期改进**: {rec.get('expected_improvement', '')}\n")
                f.write(f"- **实施方案**: {rec.get('implementation', '')}\n\n")


if __name__ == "__main__":
    main()
