#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测报告自动生成系统综合示例

演示完整的回测报告生成流程：
1. 策略性能评估
2. 多格式报告生成
3. 可视化图表生成
4. 自动化分发
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from reporting.core.report_engine import BacktestReportEngine, ReportConfig, ReportGenerationRequest
from analysis.integrated_performance_framework import PerformanceEvaluationFramework, PerformanceEvaluationRequest
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_sample_strategy_data(days: int = 252) -> pd.DataFrame:
    """
    生成示例策略数据

    Args:
        days: 生成数据的天数

    Returns:
        pd.DataFrame: 策略收益数据
    """
    np.random.seed(42)  # 确保结果可重现

    # 生成日期序列
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='B')

    # 生成策略收益率（使用改进的随机游走模型）
    base_return = 0.0008  # 基础日收益率
    volatility = 0.018    # 波动率

    # 生成带趋势的收益率
    trend = np.linspace(0, 0.0003, days)  # 轻微上升趋势
    random_returns = np.random.normal(0, volatility, days)

    # 添加一些现实特征
    returns = base_return + trend + random_returns

    # 添加一些回撤期间
    drawdown_periods = [
        (50, 70),   # 第一个回撤期
        (150, 160), # 第二个回撤期
        (200, 215)  # 第三个回撤期
    ]

    for start, end in drawdown_periods:
        if end < len(returns):
            returns[start:end] *= 0.7  # 模拟回撤

    # 计算累计收益和价格
    cumulative_returns = np.cumprod(1 + returns)
    prices = 100 * cumulative_returns

    # 构建DataFrame
    strategy_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'returns': returns,
        'cumulative_returns': cumulative_returns,
        'volume': np.random.randint(100000, 1000000, days)
    })

    strategy_data.set_index('date', inplace=True)

    logger.info(f"生成示例策略数据: {len(strategy_data)} 天")

    return strategy_data


def run_comprehensive_demo():
    """运行综合演示"""
    print("🚀 启动回测报告自动生成系统综合演示")
    print("=" * 60)

    try:
        # 1. 生成示例数据
        print("\n📊 第1步: 生成示例策略数据")
        strategy_data = generate_sample_strategy_data(days=252)
        print(f"✅ 策略数据生成完成: {len(strategy_data)} 个交易日")
        print(f"   收益率范围: {strategy_data['returns'].min():.4f} ~ {strategy_data['returns'].max():.4f}")
        print(f"   累计收益: {(strategy_data['cumulative_returns'].iloc[-1] - 1) * 100:.2f}%")

        # 2. 初始化性能评估框架
        print("\n⚙️ 第2步: 初始化性能评估框架")
        performance_framework = PerformanceEvaluationFramework()

        # 3. 执行策略性能评估
        print("\n📈 第3步: 执行策略性能评估")
        evaluation_request = PerformanceEvaluationRequest(
            strategy_name="示例量化策略",
            strategy_data=strategy_data,
            benchmark_code="000300",  # 沪深300作为基准
            evaluation_period_days=252,
            include_stress_testing=True,
            include_time_series_analysis=True,
            output_formats=['json']
        )

        evaluation_results = performance_framework.evaluate_single_strategy(evaluation_request)
        print(f"✅ 性能评估完成")

        # 显示关键指标
        if 'performance_metrics' in evaluation_results:
            pm = evaluation_results['performance_metrics']
            print(f"   总收益率: {pm.get('total_return', 0) * 100:.2f}%")
            print(f"   年化收益率: {pm.get('annualized_return', 0) * 100:.2f}%")
            print(f"   夏普比率: {pm.get('sharpe_ratio', 0):.3f}")
            print(f"   最大回撤: {pm.get('max_drawdown', 0) * 100:.2f}%")

        # 4. 配置报告生成参数
        print("\n⚙️ 第4步: 配置报告生成参数")
        report_config = ReportConfig(
            title="量化策略回测分析报告",
            subtitle="基于历史数据的策略性能评估",
            author="量化团队",
            company="投资管理公司",

            # 输出配置
            output_formats=['html', 'json', 'excel'],  # 去掉PDF避免依赖问题
            output_dir="./reports/demo",
            filename_prefix="demo_strategy_report",

            # 可视化配置
            chart_theme="professional",
            chart_dpi=300,
            parallel_chart_generation=True,
            max_workers=4,

            # 分发配置（演示用，不实际发送）
            auto_email=False,
            auto_archive=True,
            archive_retention_days=30
        )

        # 5. 初始化报告生成引擎
        print("\n🔧 第5步: 初始化报告生成引擎")
        report_engine = BacktestReportEngine(
            base_config=report_config,
            cache_dir="./cache/demo",
            template_dir="./templates/demo"
        )

        # 6. 生成报告
        print("\n📝 第6步: 生成多格式报告")
        report_request = ReportGenerationRequest(
            strategy_name="示例量化策略",
            evaluation_results=evaluation_results,
            config=report_config
        )

        generation_results = report_engine.generate_report(report_request)

        print(f"✅ 报告生成完成!")
        print(f"   生成时间: {generation_results['generation_time_seconds']:.2f} 秒")
        print(f"   请求ID: {generation_results['request_id']}")

        # 显示生成的报告文件
        print("\n📋 生成的报告文件:")
        for format_type, file_path in generation_results['report_files'].items():
            if file_path and not file_path.startswith('Error'):
                file_size = Path(file_path).stat().st_size / 1024 if Path(file_path).exists() else 0
                print(f"   {format_type.upper()}: {file_path} ({file_size:.1f} KB)")
            else:
                print(f"   {format_type.upper()}: ❌ {file_path}")

        # 显示图表文件
        if generation_results.get('chart_files'):
            print("\n🎨 生成的图表文件:")
            for chart_type, chart_path in generation_results['chart_files'].items():
                if chart_path and Path(chart_path).exists():
                    file_size = Path(chart_path).stat().st_size / 1024
                    print(f"   {chart_type}: {chart_path} ({file_size:.1f} KB)")

        # 显示性能统计
        if 'performance_stats' in generation_results:
            stats = generation_results['performance_stats']
            print(f"\n📊 生成统计:")
            print(f"   图表生成数量: {stats['total_charts_generated']}")
            print(f"   缓存命中率: {stats['chart_cache_usage']['hits']}/{stats['chart_cache_usage']['hits'] + stats['chart_cache_usage']['misses']}")

        # 7. 获取引擎性能报告
        print("\n📈 第7步: 系统性能报告")
        engine_performance = report_engine.get_performance_report()
        print(f"   引擎统计: {engine_performance['engine_stats']['total_reports']} 个报告")
        if 'system_resources' in engine_performance and 'cpu_percent' in engine_performance['system_resources']:
            print(f"   系统资源: CPU {engine_performance['system_resources']['cpu_percent']:.1f}%, "
                  f"内存 {engine_performance['system_resources']['memory_percent']:.1f}%")

        print("\n🎉 综合演示完成!")
        print("=" * 60)

        # 提供查看建议
        print("\n💡 查看建议:")
        print("1. 打开生成的HTML报告文件查看完整的可视化报告")
        print("2. 查看JSON文件了解详细的数据结构")
        print("3. 打开Excel文件查看结构化的数据表格")
        print("4. 图表文件可以直接在图片查看器中打开")

        return True

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        logger.error(f"演示失败: {e}")
        return False


def run_simple_demo():
    """运行简化演示（仅核心功能）"""
    print("🚀 启动简化演示模式")
    print("-" * 40)

    try:
        # 生成简单数据
        strategy_data = generate_sample_strategy_data(days=60)

        # 创建基本配置
        config = ReportConfig(
            title="简化策略报告",
            output_formats=['html', 'json'],
            output_dir="./reports/simple_demo"
        )

        # 模拟评估结果（简化版）
        evaluation_results = {
            'performance_metrics': {
                'total_return': (strategy_data['cumulative_returns'].iloc[-1] - 1),
                'annualized_return': (strategy_data['cumulative_returns'].iloc[-1] - 1) * 252 / len(strategy_data),
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'win_rate': 0.58,
                'volatility': strategy_data['returns'].std() * np.sqrt(252)
            },
            'risk_metrics': {
                'daily_volatility': strategy_data['returns'].std(),
                'max_drawdown_duration': 15
            }
        }

        # 快速生成报告
        from reporting.core.report_engine import generate_backtest_report
        results = generate_backtest_report(
            strategy_name="简化策略演示",
            evaluation_results=evaluation_results,
            config=config
        )

        print(f"✅ 简化报告生成完成: {results['generation_time_seconds']:.1f} 秒")
        for fmt, path in results['report_files'].items():
            if Path(path).exists():
                print(f"   {fmt}: {path}")

        return True

    except Exception as e:
        print(f"❌ 简化演示失败: {e}")
        return False


if __name__ == "__main__":
    # 确保输出目录存在
    Path("./reports").mkdir(exist_ok=True)
    Path("./cache").mkdir(exist_ok=True)

    print("回测报告自动生成系统 - 演示程序")
    print("=" * 60)

    # 选择演示模式
    demo_type = input("请选择演示模式 [1=完整演示, 2=简化演示]: ").strip()

    if demo_type == "2":
        success = run_simple_demo()
    else:
        success = run_comprehensive_demo()

    if success:
        print("\n✨ 演示成功完成!")
        print("请查看生成的报告文件以了解系统功能。")
    else:
        print("\n😞 演示未能完成，请查看错误信息。")

    print("\n感谢使用回测报告自动生成系统!")