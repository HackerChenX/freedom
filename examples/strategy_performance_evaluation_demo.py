#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略性能评估框架 - 完整使用示例

本示例演示如何使用策略性能评估框架进行：
1. 单个策略性能评估
2. 多策略批量对比评估
3. 高级配置和自定义分析
4. 与现有策略系统集成

运行此示例前请确保：
- 已安装所有依赖包
- ClickHouse数据库配置正确（可选，框架会自动使用模拟数据）
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 忽略警告
warnings.filterwarnings('ignore')

# 导入框架组件
from analysis.integrated_performance_framework import (
    PerformanceEvaluationFramework,
    PerformanceEvaluationRequest,
    BatchEvaluationRequest,
    evaluate_strategy_performance,
    batch_evaluate_strategies
)
from analysis.strategy_performance_evaluator import EvaluationConfig
from strategy.unified_base_strategy import UnifiedBaseStrategy
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class DemoStrategy(UnifiedBaseStrategy):
    """演示策略类 - 继承自统一基础策略"""

    def __init__(self, name: str, return_pattern: str = "steady"):
        """
        初始化演示策略

        Args:
            name: 策略名称
            return_pattern: 收益模式 ('steady', 'volatile', 'momentum', 'mean_revert')
        """
        super().__init__(name=name, description=f"演示策略 - {return_pattern}模式")
        self.return_pattern = return_pattern

    def select_stocks_unified_base_strategy(self, universe: List[str], start_date: str, end_date: str, **kwargs):
        """实现选股逻辑"""
        # 简化实现：返回前10只股票
        return pd.DataFrame({
            'code': universe[:10],
            'score': np.random.uniform(80, 95, min(10, len(universe))),
            'reason': [f'{self.return_pattern}信号' for _ in range(min(10, len(universe)))]
        })

    def generate_performance_data(self, days: int = 252) -> pd.DataFrame:
        """
        生成策略历史表现数据

        Args:
            days: 生成天数

        Returns:
            pd.DataFrame: 策略表现数据
        """
        # 设置随机种子以确保可重复性
        np.random.seed(hash(self.name) % 2**32)

        # 生成日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 根据不同模式生成收益率
        if self.return_pattern == "steady":
            # 稳定增长模式
            base_return = 0.0005  # 日均0.05%
            volatility = 0.01    # 1%波动率
            returns = np.random.normal(base_return, volatility, len(dates))

        elif self.return_pattern == "volatile":
            # 高波动模式
            base_return = 0.0008  # 日均0.08%
            volatility = 0.025   # 2.5%波动率
            returns = np.random.normal(base_return, volatility, len(dates))

        elif self.return_pattern == "momentum":
            # 动量模式 - 趋势跟随
            base_return = 0.0003
            volatility = 0.015
            returns = [np.random.normal(base_return, volatility)]
            for i in range(1, len(dates)):
                # 动量效应：前一日表现影响当日
                momentum = 0.3 * returns[-1]
                daily_return = base_return + momentum + np.random.normal(0, volatility)
                returns.append(daily_return)

        elif self.return_pattern == "mean_revert":
            # 均值回归模式
            base_return = 0.0004
            volatility = 0.012
            returns = [np.random.normal(base_return, volatility)]
            cumulative_excess = 0

            for i in range(1, len(dates)):
                cumulative_excess += returns[-1] - base_return
                # 均值回归：累计偏离越大，回归力度越强
                revert_force = -0.1 * cumulative_excess
                daily_return = base_return + revert_force + np.random.normal(0, volatility)
                returns.append(daily_return)

        else:
            # 默认随机游走
            returns = np.random.normal(0.0003, 0.015, len(dates))

        # 创建DataFrame
        price = 100.0
        prices = [price]

        for ret in returns[:-1]:
            price *= (1 + ret)
            prices.append(price)

        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'returns': [0] + list(returns[:-1])  # 第一天收益率为0
        })

        data.set_index('date', inplace=True)
        data['cumulative_returns'] = (1 + data['returns']).cumprod()

        # 添加其他必要字段
        data['volume'] = np.random.randint(100000, 1000000, len(data))
        data['high'] = data['close'] * (1 + np.random.uniform(0, 0.02, len(data)))
        data['low'] = data['close'] * (1 - np.random.uniform(0, 0.02, len(data)))
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])

        return data


def demonstration_single_strategy_evaluation():
    """演示单策略评估功能"""
    print("\n" + "="*60)
    print("🎯 演示1: 单策略性能评估")
    print("="*60)

    # 创建演示策略
    strategy = DemoStrategy("稳健成长策略", "steady")
    performance_data = strategy.generate_performance_data(300)

    print(f"📊 评估策略: {strategy.name}")
    print(f"📅 数据期间: {performance_data.index[0].strftime('%Y-%m-%d')} 至 {performance_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"📈 数据点数: {len(performance_data)}")

    # 执行评估
    start_time = datetime.now()

    result = evaluate_strategy_performance(
        strategy_name=strategy.name,
        strategy_data=performance_data,
        benchmark_code="000001",  # 上证指数
        output_formats=['html']  # 只生成HTML报告避免JSON序列化问题
    )

    execution_time = (datetime.now() - start_time).total_seconds()

    # 展示结果
    performance_metrics = result.get('performance_metrics', {})
    risk_metrics = result.get('risk_metrics', {})

    print(f"\n📋 评估结果:")
    print(f"   执行时间: {execution_time:.2f}秒")
    print(f"   总收益率: {performance_metrics.get('total_return', 0):.2%}")
    print(f"   年化收益率: {performance_metrics.get('annualized_return', 0):.2%}")
    print(f"   夏普比率: {performance_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"   年化波动率: {performance_metrics.get('volatility', 0):.2%}")
    print(f"   胜率: {performance_metrics.get('win_rate', 0):.1%}")
    print(f"   盈亏比: {performance_metrics.get('profit_loss_ratio', 0):.2f}")

    print(f"\n🎯 风险指标:")
    print(f"   95% VaR: {risk_metrics.get('var_1d_95', 0):.2%}")
    print(f"   95% CVaR: {risk_metrics.get('cvar_1d_95', 0):.2%}")
    print(f"   当前回撤: {risk_metrics.get('current_drawdown', 0):.2%}")

    # 显示报告文件
    report_files = result.get('report_files', {})
    if report_files:
        print(f"\n📄 生成报告:")
        for format_type, file_path in report_files.items():
            if not file_path.startswith("Error:"):
                print(f"   {format_type.upper()}: {file_path}")

    return result


def demonstration_multi_strategy_evaluation():
    """演示多策略批量评估功能"""
    print("\n" + "="*60)
    print("🎯 演示2: 多策略批量对比评估")
    print("="*60)

    # 创建多个演示策略
    strategies = [
        DemoStrategy("稳健型策略", "steady"),
        DemoStrategy("进取型策略", "volatile"),
        DemoStrategy("动量型策略", "momentum"),
        DemoStrategy("均值回归策略", "mean_revert")
    ]

    # 生成各策略历史表现数据
    strategies_data = {}
    for strategy in strategies:
        print(f"📈 生成策略数据: {strategy.name}")
        strategies_data[strategy.name] = strategy.generate_performance_data(252)

    print(f"\n🔄 开始批量评估 {len(strategies)} 个策略...")

    # 执行批量评估
    start_time = datetime.now()

    result = batch_evaluate_strategies(
        strategies_data=strategies_data,
        benchmark_code="000001",
        parallel_workers=4,
        output_formats=['html']  # 只生成HTML避免序列化问题
    )

    execution_time = (datetime.now() - start_time).total_seconds()

    # 展示结果
    evaluation_summary = result.get('evaluation_summary', {})
    strategy_rankings = result.get('strategy_rankings', {})

    print(f"\n📊 批量评估结果:")
    print(f"   执行时间: {execution_time:.2f}秒")
    print(f"   评估策略数: {evaluation_summary.get('total_strategies', 0)}")
    print(f"   成功评估: {evaluation_summary.get('successful_evaluations', 0)}")
    print(f"   成功率: {evaluation_summary.get('success_rate', 0):.1%}")
    print(f"   平均每策略耗时: {execution_time / len(strategies):.2f}秒")

    # 显示策略排名
    if 'composite_ranking' in strategy_rankings:
        print(f"\n🏆 策略综合排名:")
        ranking = strategy_rankings['composite_ranking']['ranking']
        for i, entry in enumerate(ranking[:5]):  # 显示前5名
            print(f"   {entry['rank']}. {entry['strategy']}: 综合评分 {entry['value']:.3f}")

    # 显示各类别排名
    ranking_types = [
        ('sharpe_ratio_ranking', '夏普比率'),
        ('total_return_ranking', '总收益率'),
        ('max_drawdown_ranking', '最大回撤')
    ]

    for ranking_key, ranking_name in ranking_types:
        if ranking_key in strategy_rankings:
            print(f"\n📈 {ranking_name}排名:")
            ranking = strategy_rankings[ranking_key]['ranking']
            for entry in ranking[:3]:  # 显示前3名
                print(f"   {entry['rank']}. {entry['strategy']}: {entry['value']:.3f}")

    return result


def demonstration_advanced_configuration():
    """演示高级配置和自定义分析"""
    print("\n" + "="*60)
    print("🎯 演示3: 高级配置和自定义分析")
    print("="*60)

    # 创建自定义配置
    custom_config = EvaluationConfig(
        benchmark_code="000300",           # 沪深300指数
        risk_free_rate=0.025,             # 2.5%无风险利率
        evaluation_period=504,            # 2年评估期
        rolling_window=60,                # 60日滚动窗口
        parallel_workers=8,               # 8线程并行
        confidence_levels=[0.95, 0.99, 0.999]  # 多个置信度
    )

    print("⚙️  自定义配置:")
    print(f"   基准指数: {custom_config.benchmark_code}")
    print(f"   无风险利率: {custom_config.risk_free_rate:.1%}")
    print(f"   评估期间: {custom_config.evaluation_period}天")
    print(f"   滚动窗口: {custom_config.rolling_window}天")
    print(f"   并行线程: {custom_config.parallel_workers}")
    print(f"   置信水平: {custom_config.confidence_levels}")

    # 创建自定义框架实例
    custom_framework = PerformanceEvaluationFramework(
        config=custom_config,
        cache_dir="./cache/custom_demo",
        output_dir="./reports/custom_demo"
    )

    print(f"\n🏗️  自定义框架创建完成")
    print(f"   缓存目录: {custom_framework.cache_dir}")
    print(f"   输出目录: {custom_framework.output_dir}")

    # 创建测试策略
    test_strategy = DemoStrategy("自定义配置测试策略", "momentum")
    performance_data = test_strategy.generate_performance_data(600)  # 更长的历史数据

    # 创建评估请求
    request = PerformanceEvaluationRequest(
        strategy_name=test_strategy.name,
        strategy_data=performance_data,
        benchmark_code=custom_config.benchmark_code,
        evaluation_period_days=custom_config.evaluation_period,
        include_stress_testing=True,
        include_time_series_analysis=True,
        output_formats=['html']
    )

    print(f"\n🔬 执行自定义评估...")

    # 执行评估
    start_time = datetime.now()
    result = custom_framework.evaluate_single_strategy(request)
    execution_time = (datetime.now() - start_time).total_seconds()

    # 展示结果
    print(f"\n📊 自定义评估结果:")
    print(f"   执行时间: {execution_time:.2f}秒")

    performance_metrics = result.get('performance_metrics', {})
    time_series_analysis = result.get('time_series_analysis', {})

    print(f"   策略表现:")
    print(f"     年化收益: {performance_metrics.get('annualized_return', 0):.2%}")
    print(f"     夏普比率: {performance_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"     信息比率: {performance_metrics.get('information_ratio', 0):.3f}")

    # 显示时间序列分析结果
    if 'monthly_returns' in time_series_analysis:
        print(f"\n📅 月度收益分析:")
        monthly_returns = time_series_analysis['monthly_returns']
        if monthly_returns:
            best_month = max(monthly_returns.items(), key=lambda x: x[1])
            worst_month = min(monthly_returns.items(), key=lambda x: x[1])
            print(f"     最佳月份: {best_month[0]}月 ({best_month[1]:.2%})")
            print(f"     最差月份: {worst_month[0]}月 ({worst_month[1]:.2%})")

    # 显示框架性能报告
    framework_report = custom_framework.get_framework_performance_report()
    framework_stats = framework_report.get('framework_stats', {})
    cache_stats = framework_report.get('cache_stats', {})

    print(f"\n⚡ 框架性能统计:")
    print(f"   总评估次数: {framework_stats.get('total_evaluations', 0)}")
    print(f"   平均执行时间: {framework_stats.get('avg_execution_time', 0):.3f}秒")
    print(f"   缓存命中率: {cache_stats.get('hit_rate', 0):.1%}")

    return result


def demonstration_strategy_integration():
    """演示与现有策略系统集成"""
    print("\n" + "="*60)
    print("🎯 演示4: 策略系统集成")
    print("="*60)

    class IntegratedStrategy(UnifiedBaseStrategy):
        """集成性能评估功能的策略类"""

        def __init__(self, name: str):
            super().__init__(name=name, description="集成性能评估的策略")
            self.performance_history = []

        def select_stocks_unified_base_strategy(self, universe: List[str], start_date: str, end_date: str, **kwargs):
            """选股实现"""
            # 模拟选股逻辑
            selected_stocks = pd.DataFrame({
                'code': universe[:8],
                'score': np.random.uniform(85, 98, min(8, len(universe))),
                'reason': ['技术突破' for _ in range(min(8, len(universe)))]
            })

            # 记录选股结果用于后续性能分析
            self.performance_history.append({
                'date': end_date,
                'selected_stocks': selected_stocks['code'].tolist(),
                'avg_score': selected_stocks['score'].mean()
            })

            return selected_stocks

        def evaluate_historical_performance(self, days: int = 252):
            """评估策略历史表现"""
            print(f"📈 评估策略 '{self.name}' 的历史表现...")

            # 生成模拟历史表现数据
            performance_data = self._generate_simulated_performance(days)

            # 执行性能评估
            try:
                result = evaluate_strategy_performance(
                    strategy_name=self.name,
                    strategy_data=performance_data,
                    benchmark_code="000001",
                    output_formats=['html']
                )

                # 存储评估结果
                self._store_evaluation_result(result)

                return result

            except Exception as e:
                print(f"❌ 策略性能评估失败: {e}")
                return None

        def _generate_simulated_performance(self, days: int) -> pd.DataFrame:
            """生成模拟的策略表现数据"""
            # 基于历史选股记录生成表现数据
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

            # 模拟策略收益率（基于选股历史）
            base_return = 0.0006  # 基础日收益率
            skill_factor = len(self.performance_history) * 0.0001  # 技能因子
            volatility = 0.018

            returns = np.random.normal(
                base_return + skill_factor,
                volatility,
                len(dates)
            )

            price = 100.0
            prices = [price]

            for ret in returns[:-1]:
                price *= (1 + ret)
                prices.append(price)

            return pd.DataFrame({
                'date': dates,
                'close': prices,
                'returns': [0] + list(returns[:-1])
            }).set_index('date')

        def _store_evaluation_result(self, result: Dict[str, Any]):
            """存储评估结果"""
            performance_metrics = result.get('performance_metrics', {})
            print(f"💾 存储策略评估结果:")
            print(f"   年化收益率: {performance_metrics.get('annualized_return', 0):.2%}")
            print(f"   夏普比率: {performance_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}")

        def get_performance_summary(self) -> str:
            """获取性能摘要"""
            if not hasattr(self, '_last_evaluation'):
                return "尚未进行性能评估"

            metrics = self._last_evaluation.get('performance_metrics', {})
            return f"年化收益{metrics.get('annualized_return', 0):.1%}，" \
                   f"夏普比率{metrics.get('sharpe_ratio', 0):.2f}，" \
                   f"最大回撤{metrics.get('max_drawdown', 0):.1%}"

    # 演示集成策略
    print("🏗️  创建集成策略实例...")
    integrated_strategy = IntegratedStrategy("智能选股策略V1.0")

    # 模拟选股过程
    print("\n🎯 模拟选股过程...")
    universe = [f"{i:06d}.SZ" for i in range(1, 51)]  # 50只股票池

    for i in range(5):  # 模拟5次选股
        end_date = (datetime.now() - timedelta(days=(4-i)*30)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=(5-i)*30)).strftime('%Y-%m-%d')

        result = integrated_strategy.execute(
            universe=universe,
            start_date=start_date,
            end_date=end_date
        )

        print(f"   {end_date}: 选出 {len(result)} 只股票，平均评分 {result['score'].mean():.1f}")

    # 评估历史表现
    print(f"\n📊 评估策略历史表现...")
    evaluation_result = integrated_strategy.evaluate_historical_performance(180)

    if evaluation_result:
        print(f"✅ 性能评估完成")
        report_files = evaluation_result.get('report_files', {})
        if report_files:
            for format_type, file_path in report_files.items():
                if not file_path.startswith("Error:"):
                    print(f"   报告文件: {file_path}")

    return integrated_strategy


def main():
    """主演示函数"""
    print("🚀 策略性能评估框架 - 完整功能演示")
    print("="*80)
    print("本演示将展示策略性能评估框架的四大核心功能：")
    print("1. 单策略性能评估")
    print("2. 多策略批量对比评估")
    print("3. 高级配置和自定义分析")
    print("4. 与现有策略系统集成")
    print("="*80)

    try:
        # 演示1: 单策略评估
        demo1_result = demonstration_single_strategy_evaluation()

        # 演示2: 多策略批量评估
        demo2_result = demonstration_multi_strategy_evaluation()

        # 演示3: 高级配置
        demo3_result = demonstration_advanced_configuration()

        # 演示4: 策略集成
        demo4_result = demonstration_strategy_integration()

        # 总结
        print("\n" + "="*60)
        print("🎉 演示完成总结")
        print("="*60)

        print("✅ 所有演示功能已成功执行")
        print("📊 策略性能评估框架已准备就绪")
        print("🚀 可以开始在生产环境中使用")

        print(f"\n📁 生成的报告文件位置:")
        print(f"   单策略报告: reports/performance_evaluation/")
        print(f"   批量评估报告: reports/performance_evaluation/")
        print(f"   自定义报告: reports/custom_demo/")

        print(f"\n💡 使用建议:")
        print(f"   1. 根据实际需求调整评估配置")
        print(f"   2. 定期清理缓存以释放存储空间")
        print(f"   3. 监控框架性能统计以优化使用")
        print(f"   4. 将评估结果集成到现有决策流程")

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n🔚 演示结束")
    return True


if __name__ == "__main__":
    main()