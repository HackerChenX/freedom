#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化的股票分析系统性能测试工具

不依赖外部库的性能测试工具
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
from analysis.simple_performance_analyzer import SimplePerformanceAnalyzer
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer

logger = get_logger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='简化的股票分析系统性能测试工具')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入的CSV文件路径，包含股票代码和买点日期')
    
    parser.add_argument('--output', '-o', type=str, default='data/result/performance_test',
                        help='输出结果的目录路径')
    
    parser.add_argument('--sample-size', '-s', type=int, default=5,
                        help='性能测试的采样大小，默认为5')
    
    parser.add_argument('--test-type', '-t', type=str, default='all',
                        choices=['all', 'batch', 'indicator', 'data'],
                        help='测试类型，默认为all（全部测试）')
    
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
    
    analyzer = SimplePerformanceAnalyzer()
    results = {}
    
    # 批量处理性能测试
    if test_type in ['all', 'batch']:
        logger.info("执行批量处理性能测试...")
        batch_result = analyzer.analyze_batch_processing_performance(
            buypoints_csv=input_csv,
            sample_size=sample_size
        )
        results['batch_processing'] = batch_result
    
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
    
    return results


def analyze_processing_stages(input_csv: str, sample_size: int = 3) -> dict:
    """
    分析处理阶段的耗时分布
    
    Args:
        input_csv: 输入CSV文件路径
        sample_size: 采样大小
        
    Returns:
        dict: 各阶段耗时分析结果
    """
    logger.info("开始分析处理阶段耗时分布...")
    
    # 加载买点数据
    analyzer = BuyPointBatchAnalyzer()
    buypoints_df = analyzer.load_buypoints_from_csv(input_csv)
    
    if buypoints_df.empty:
        return {}
    
    # 限制采样大小
    test_df = buypoints_df.head(sample_size)
    
    stage_times = {
        'data_loading': [],
        'indicator_calculation': [],
        'pattern_analysis': [],
        'total_per_stock': []
    }
    
    for _, row in test_df.iterrows():
        stock_code = row['stock_code']
        buypoint_date = row['buypoint_date']
        
        logger.info(f"分析股票 {stock_code} 各阶段耗时...")
        
        # 总计时开始
        total_start = time.time()
        
        # 1. 数据加载阶段
        data_start = time.time()
        stock_data = analyzer.data_processor.get_multi_period_data(
            stock_code=stock_code,
            end_date=buypoint_date
        )
        data_end = time.time()
        data_time = data_end - data_start
        stage_times['data_loading'].append(data_time)
        
        if not stock_data:
            continue
        
        # 2. 指标计算阶段
        indicator_start = time.time()
        target_rows = {period: len(df) - 1 for period, df in stock_data.items() if df is not None and not df.empty}
        indicator_results = analyzer.indicator_analyzer.analyze_all_indicators(
            stock_data, target_rows
        )
        indicator_end = time.time()
        indicator_time = indicator_end - indicator_start
        stage_times['indicator_calculation'].append(indicator_time)
        
        # 3. 形态分析阶段（如果有的话）
        pattern_start = time.time()
        # 这里可以添加形态分析的代码
        pattern_end = time.time()
        pattern_time = pattern_end - pattern_start
        stage_times['pattern_analysis'].append(pattern_time)
        
        # 总计时结束
        total_end = time.time()
        total_time = total_end - total_start
        stage_times['total_per_stock'].append(total_time)
        
        logger.info(f"股票 {stock_code} 各阶段耗时: 数据加载 {data_time:.2f}s, 指标计算 {indicator_time:.2f}s, 总计 {total_time:.2f}s")
    
    # 计算各阶段统计信息
    result = {}
    for stage, times in stage_times.items():
        if times:
            result[stage] = {
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'percentage': (sum(times) / sum(stage_times['total_per_stock'])) * 100 if stage_times['total_per_stock'] else 0
            }
    
    return result


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    setup_logger()
    logger.info("简化的股票分析系统性能测试工具启动")
    
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
        
        # 分析处理阶段耗时
        stage_analysis = analyze_processing_stages(
            input_csv=args.input,
            sample_size=min(args.sample_size, 3)
        )
        
        # 生成优化建议
        analyzer = SimplePerformanceAnalyzer()
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
            'stage_analysis': stage_analysis,
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
        
        # 打印关键性能指标
        print_performance_summary(final_results)
        
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
            
            if 'indicator_calculation' in perf:
                indicator = perf['indicator_calculation']
                f.write("### 指标计算性能\n\n")
                f.write(f"- 总指标数: {indicator.get('total_indicators', 0)}\n")
                f.write(f"- 成功指标数: {indicator.get('successful_indicators', 0)}\n")
                f.write(f"- 平均每指标时间: {indicator.get('avg_time_per_indicator', 0):.3f}秒\n\n")
                
                # 最慢的指标
                top_slow = indicator.get('top_10_slowest', {})
                if top_slow:
                    f.write("#### 最耗时的指标（前5个）\n\n")
                    for i, (name, data) in enumerate(list(top_slow.items())[:5], 1):
                        f.write(f"{i}. {name}: {data.get('total_time', 0):.3f}秒\n")
                    f.write("\n")
        
        # 处理阶段分析
        if 'stage_analysis' in results:
            f.write("## 处理阶段耗时分析\n\n")
            stage = results['stage_analysis']
            
            for stage_name, stage_data in stage.items():
                f.write(f"### {stage_name}\n\n")
                f.write(f"- 平均时间: {stage_data.get('avg_time', 0):.2f}秒\n")
                f.write(f"- 占总时间比例: {stage_data.get('percentage', 0):.1f}%\n\n")
        
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


def print_performance_summary(results: dict):
    """打印性能摘要"""
    print("\n" + "="*60)
    print("性能测试摘要")
    print("="*60)
    
    if 'performance_analysis' in results:
        perf = results['performance_analysis']
        
        if 'batch_processing' in perf:
            batch = perf['batch_processing']
            print(f"批量处理性能:")
            print(f"  - 平均每股时间: {batch.get('avg_time_per_stock', 0):.2f}秒")
            print(f"  - 成功率: {batch.get('success_rate', 0):.1%}")
        
        if 'indicator_calculation' in perf:
            indicator = perf['indicator_calculation']
            print(f"指标计算性能:")
            print(f"  - 总指标数: {indicator.get('total_indicators', 0)}")
            print(f"  - 平均每指标时间: {indicator.get('avg_time_per_indicator', 0):.3f}秒")
    
    print("\n优化建议数量:", len(results.get('recommendations', [])))
    print("="*60)


if __name__ == "__main__":
    main()
