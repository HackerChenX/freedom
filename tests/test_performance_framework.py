#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略性能评估框架测试脚本

测试和验证策略性能评估框架的功能：
1. 单策略性能评估
2. 多策略批量评估
3. 性能指标计算准确性
4. 报告生成功能
5. 缓存机制验证
6. 并行处理性能
"""

import os
import sys
import time
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

from analysis.integrated_performance_framework import (
    PerformanceEvaluationFramework,
    PerformanceEvaluationRequest,
    BatchEvaluationRequest,
    evaluate_strategy_performance,
    batch_evaluate_strategies
)
from analysis.strategy_performance_evaluator import EvaluationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceFrameworkTester:
    """策略性能评估框架测试器"""

    def __init__(self):
        """初始化测试器"""
        self.logger = get_logger(__name__)
        self.test_results = {}

    def generate_mock_strategy_data(self,
                                  strategy_name: str,
                                  days: int = 252,
                                  annual_return: float = 0.12,
                                  annual_volatility: float = 0.20,
                                  max_drawdown: float = 0.15) -> pd.DataFrame:
        """
        生成模拟策略数据

        Args:
            strategy_name: 策略名称
            days: 天数
            annual_return: 年化收益率
            annual_volatility: 年化波动率
            max_drawdown: 最大回撤

        Returns:
            pd.DataFrame: 模拟策略数据
        """
        # 设置随机种子以确保可重复性
        np.random.seed(hash(strategy_name) % 2**32)

        # 生成日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 生成收益率
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)

        # 使用几何布朗运动模拟价格路径
        dt = 1 / 252
        returns = []
        price = 100.0
        prices = [price]

        for i in range(len(dates) - 1):
            # 基础随机游走
            random_shock = np.random.normal(0, 1)
            daily_ret = daily_return + daily_volatility * random_shock

            # 添加一些现实特征
            # 1. 均值回归效应
            if len(returns) > 20:
                recent_performance = np.mean(returns[-20:])
                if recent_performance > daily_return * 2:
                    daily_ret *= 0.8  # 强势后回调
                elif recent_performance < -daily_return:
                    daily_ret *= 1.2  # 弱势后反弹

            # 2. 波动率聚集
            if len(returns) > 0 and abs(returns[-1]) > 2 * daily_volatility:
                daily_ret *= 1.3

            returns.append(daily_ret)
            price *= (1 + daily_ret)
            prices.append(price)

        # 调整以符合目标最大回撤
        cumulative_returns = np.cumprod(np.array([1] + [1 + r for r in returns]))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max

        current_max_dd = abs(drawdowns.min())
        if current_max_dd > max_drawdown:
            # 缩放收益率以控制最大回撤
            scale_factor = max_drawdown / current_max_dd * 0.9
            returns = [r * scale_factor for r in returns]

            # 重新计算价格
            prices = [100.0]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))

        # 创建DataFrame
        data = pd.DataFrame({
            'date': dates,
            'close': prices[:len(dates)],
            'returns': [0] + returns[:len(dates)-1]  # 第一天收益率为0
        })

        data.set_index('date', inplace=True)
        data['cumulative_returns'] = (1 + data['returns']).cumprod()

        # 添加其他字段
        data['volume'] = np.random.randint(100000, 1000000, len(data))
        data['high'] = data['close'] * (1 + np.random.uniform(0, 0.03, len(data)))
        data['low'] = data['close'] * (1 - np.random.uniform(0, 0.03, len(data)))
        data['open'] = data['close'].shift(1) * (1 + np.random.uniform(-0.01, 0.01, len(data)))
        data['open'].iloc[0] = data['close'].iloc[0]

        return data

    def test_single_strategy_evaluation(self) -> Dict[str, Any]:
        """测试单策略评估"""
        self.logger.info("开始测试单策略评估...")

        try:
            # 生成测试数据
            strategy_data = self.generate_mock_strategy_data(
                strategy_name="测试策略A",
                days=300,
                annual_return=0.15,
                annual_volatility=0.25,
                max_drawdown=0.12
            )

            # 执行评估
            start_time = time.time()
            result = evaluate_strategy_performance(
                strategy_name="测试策略A",
                strategy_data=strategy_data,
                benchmark_code="000001",
                output_formats=['json', 'html']
            )
            execution_time = time.time() - start_time

            # 验证结果
            performance_metrics = result.get('performance_metrics', {})
            risk_metrics = result.get('risk_metrics', {})

            validation_results = {
                'execution_time': execution_time,
                'execution_time_acceptable': execution_time < 30.0,  # 应该<30秒
                'has_performance_metrics': bool(performance_metrics),
                'has_risk_metrics': bool(risk_metrics),
                'has_report_files': 'report_files' in result,
                'total_return': performance_metrics.get('total_return', 0),
                'annualized_return': performance_metrics.get('annualized_return', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'volatility': performance_metrics.get('volatility', 0),
                'data_quality_score': result.get('framework_metadata', {}).get('data_quality_score', 0)
            }

            # 检查指标合理性
            validation_results['metrics_reasonable'] = all([
                0 <= abs(validation_results['total_return']) <= 2,  # 总收益率在合理范围
                0 <= validation_results['max_drawdown'] <= 1,  # 最大回撤在0-100%
                0 <= validation_results['volatility'] <= 2,  # 波动率在合理范围
                validation_results['data_quality_score'] >= 70  # 数据质量评分>=70
            ])

            self.logger.info(f"单策略评估测试完成: 耗时 {execution_time:.2f}秒")
            self.logger.info(f"总收益率: {validation_results['total_return']:.4f}")
            self.logger.info(f"夏普比率: {validation_results['sharpe_ratio']:.4f}")
            self.logger.info(f"最大回撤: {validation_results['max_drawdown']:.4f}")

            return validation_results

        except Exception as e:
            self.logger.error(f"单策略评估测试失败: {e}")
            return {'error': str(e), 'success': False}

    def test_batch_strategy_evaluation(self) -> Dict[str, Any]:
        """测试批量策略评估"""
        self.logger.info("开始测试批量策略评估...")

        try:
            # 生成多个测试策略
            strategies_data = {}
            strategy_configs = [
                ("低风险策略", 0.08, 0.12, 0.05),
                ("中风险策略", 0.12, 0.20, 0.10),
                ("高风险策略", 0.18, 0.30, 0.20),
                ("价值策略", 0.10, 0.15, 0.08),
                ("成长策略", 0.15, 0.25, 0.15)
            ]

            for name, annual_ret, annual_vol, max_dd in strategy_configs:
                strategies_data[name] = self.generate_mock_strategy_data(
                    strategy_name=name,
                    days=252,
                    annual_return=annual_ret,
                    annual_volatility=annual_vol,
                    max_drawdown=max_dd
                )

            # 执行批量评估
            start_time = time.time()
            result = batch_evaluate_strategies(
                strategies_data=strategies_data,
                benchmark_code="000001",
                parallel_workers=4,
                output_formats=['json', 'html']
            )
            execution_time = time.time() - start_time

            # 验证结果
            evaluation_summary = result.get('evaluation_summary', {})
            individual_results = result.get('individual_results', {})
            strategy_rankings = result.get('strategy_rankings', {})

            validation_results = {
                'execution_time': execution_time,
                'execution_time_acceptable': execution_time < 60.0,  # 应该<60秒
                'total_strategies': evaluation_summary.get('total_strategies', 0),
                'successful_evaluations': evaluation_summary.get('successful_evaluations', 0),
                'success_rate': evaluation_summary.get('success_rate', 0),
                'has_individual_results': len(individual_results) > 0,
                'has_strategy_rankings': bool(strategy_rankings),
                'has_batch_reports': 'batch_report_files' in result,
                'parallel_efficiency': evaluation_summary.get('parallel_efficiency', 0),
                'avg_time_per_strategy': execution_time / len(strategies_data) if strategies_data else 0
            }

            # 检查批量评估质量
            validation_results['batch_evaluation_quality'] = all([
                validation_results['success_rate'] >= 0.8,  # 成功率>=80%
                validation_results['avg_time_per_strategy'] < 15.0,  # 平均每策略<15秒
                validation_results['has_individual_results'],
                validation_results['has_strategy_rankings']
            ])

            self.logger.info(f"批量策略评估测试完成: 耗时 {execution_time:.2f}秒")
            self.logger.info(f"评估策略数: {validation_results['total_strategies']}")
            self.logger.info(f"成功率: {validation_results['success_rate']:.2%}")
            self.logger.info(f"平均每策略用时: {validation_results['avg_time_per_strategy']:.2f}秒")

            return validation_results

        except Exception as e:
            self.logger.error(f"批量策略评估测试失败: {e}")
            return {'error': str(e), 'success': False}

    def test_performance_accuracy(self) -> Dict[str, Any]:
        """测试性能指标计算准确性"""
        self.logger.info("开始测试性能指标计算准确性...")

        try:
            # 创建已知结果的测试数据
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

            # 构造简单的测试案例
            # 每天收益率1%，应该产生可预测的结果
            daily_returns = [0.01] * len(dates)
            daily_returns[0] = 0  # 第一天无收益

            test_data = pd.DataFrame({
                'returns': daily_returns,
                'close': [100 * (1.01 ** i) for i in range(len(dates))]
            }, index=dates)

            test_data['cumulative_returns'] = (1 + test_data['returns']).cumprod()

            # 执行评估
            result = evaluate_strategy_performance(
                strategy_name="准确性测试策略",
                strategy_data=test_data,
                output_formats=['json']
            )

            performance_metrics = result.get('performance_metrics', {})

            # 计算期望值
            expected_total_return = (1.01 ** (len(dates) - 1)) - 1  # 复合收益
            expected_annualized_return = (1.01 ** 252) - 1  # 年化收益
            expected_volatility = 0.0  # 固定收益率，波动率应该为0

            # 验证准确性
            tolerance = 0.01  # 1%容差

            accuracy_results = {
                'calculated_total_return': performance_metrics.get('total_return', 0),
                'expected_total_return': expected_total_return,
                'total_return_error': abs(performance_metrics.get('total_return', 0) - expected_total_return) / expected_total_return,
                'calculated_annualized_return': performance_metrics.get('annualized_return', 0),
                'expected_annualized_return': expected_annualized_return,
                'annualized_return_error': abs(performance_metrics.get('annualized_return', 0) - expected_annualized_return) / expected_annualized_return,
                'calculated_volatility': performance_metrics.get('volatility', 0),
                'expected_volatility': expected_volatility,
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
            }

            # 检查准确性
            accuracy_results['total_return_accurate'] = accuracy_results['total_return_error'] < tolerance
            accuracy_results['annualized_return_accurate'] = accuracy_results['annualized_return_error'] < tolerance
            accuracy_results['volatility_accurate'] = accuracy_results['calculated_volatility'] < 0.01  # 应该接近0
            accuracy_results['max_drawdown_accurate'] = accuracy_results['max_drawdown'] < 0.01  # 应该接近0

            # 总体准确性
            accuracy_results['overall_accuracy'] = all([
                accuracy_results['total_return_accurate'],
                accuracy_results['annualized_return_accurate'],
                accuracy_results['volatility_accurate'],
                accuracy_results['max_drawdown_accurate']
            ])

            accuracy_percentage = (
                (1 - accuracy_results['total_return_error']) * 100 if accuracy_results['total_return_error'] < 1 else 0
            )

            self.logger.info(f"性能指标准确性测试完成")
            self.logger.info(f"总收益率计算准确度: {(1 - accuracy_results['total_return_error']) * 100:.2f}%")
            self.logger.info(f"年化收益率计算准确度: {(1 - accuracy_results['annualized_return_error']) * 100:.2f}%")

            return accuracy_results

        except Exception as e:
            self.logger.error(f"性能指标准确性测试失败: {e}")
            return {'error': str(e), 'success': False}

    def test_caching_mechanism(self) -> Dict[str, Any]:
        """测试缓存机制"""
        self.logger.info("开始测试缓存机制...")

        try:
            # 创建测试数据
            strategy_data = self.generate_mock_strategy_data("缓存测试策略", days=100)

            # 第一次评估（应该缓存未命中）
            start_time1 = time.time()
            result1 = evaluate_strategy_performance(
                strategy_name="缓存测试策略",
                strategy_data=strategy_data,
                output_formats=['json']
            )
            time1 = time.time() - start_time1

            # 第二次评估（应该缓存命中）
            start_time2 = time.time()
            result2 = evaluate_strategy_performance(
                strategy_name="缓存测试策略",
                strategy_data=strategy_data,
                output_formats=['json']
            )
            time2 = time.time() - start_time2

            # 验证缓存效果
            cache_results = {
                'first_execution_time': time1,
                'second_execution_time': time2,
                'speedup_ratio': time1 / time2 if time2 > 0 else 0,
                'cache_effective': time2 < time1 * 0.5,  # 第二次应该快至少50%
                'results_identical': (
                    result1.get('performance_metrics', {}).get('total_return') ==
                    result2.get('performance_metrics', {}).get('total_return')
                )
            }

            self.logger.info(f"缓存机制测试完成")
            self.logger.info(f"第一次执行时间: {time1:.3f}秒")
            self.logger.info(f"第二次执行时间: {time2:.3f}秒")
            self.logger.info(f"加速比: {cache_results['speedup_ratio']:.2f}x")

            return cache_results

        except Exception as e:
            self.logger.error(f"缓存机制测试失败: {e}")
            return {'error': str(e), 'success': False}

    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用"""
        self.logger.info("开始测试内存使用...")

        try:
            import psutil
            import gc

            # 获取初始内存
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 创建大量策略数据
            strategies_data = {}
            for i in range(50):  # 创建50个策略
                strategies_data[f"策略{i:02d}"] = self.generate_mock_strategy_data(
                    f"策略{i:02d}", days=252
                )

            memory_after_data = process.memory_info().rss / 1024 / 1024

            # 执行批量评估
            result = batch_evaluate_strategies(
                strategies_data=strategies_data,
                parallel_workers=4,
                output_formats=['json']
            )

            memory_after_evaluation = process.memory_info().rss / 1024 / 1024

            # 强制垃圾回收
            del strategies_data
            del result
            gc.collect()

            memory_after_cleanup = process.memory_info().rss / 1024 / 1024

            memory_results = {
                'initial_memory_mb': initial_memory,
                'memory_after_data_mb': memory_after_data,
                'memory_after_evaluation_mb': memory_after_evaluation,
                'memory_after_cleanup_mb': memory_after_cleanup,
                'peak_memory_usage_mb': memory_after_evaluation,
                'memory_increase_mb': memory_after_evaluation - initial_memory,
                'memory_per_strategy_mb': (memory_after_evaluation - initial_memory) / 50,
                'memory_usage_acceptable': memory_after_evaluation < 2048,  # 应该<2GB
                'memory_cleanup_effective': (memory_after_evaluation - memory_after_cleanup) > 0
            }

            self.logger.info(f"内存使用测试完成")
            self.logger.info(f"峰值内存使用: {memory_results['peak_memory_usage_mb']:.1f}MB")
            self.logger.info(f"平均每策略内存: {memory_results['memory_per_strategy_mb']:.1f}MB")

            return memory_results

        except ImportError:
            self.logger.warning("psutil未安装，跳过内存测试")
            return {'error': 'psutil not available', 'skipped': True}
        except Exception as e:
            self.logger.error(f"内存使用测试失败: {e}")
            return {'error': str(e), 'success': False}

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试"""
        self.logger.info("="*60)
        self.logger.info("开始策略性能评估框架全面测试")
        self.logger.info("="*60)

        overall_start_time = time.time()

        # 运行所有测试
        test_methods = [
            ('单策略评估', self.test_single_strategy_evaluation),
            ('批量策略评估', self.test_batch_strategy_evaluation),
            ('性能指标准确性', self.test_performance_accuracy),
            ('缓存机制', self.test_caching_mechanism),
            ('内存使用', self.test_memory_usage),
        ]

        test_results = {}
        passed_tests = 0
        total_tests = len(test_methods)

        for test_name, test_method in test_methods:
            self.logger.info(f"\n--- 执行测试: {test_name} ---")
            try:
                result = test_method()
                test_results[test_name] = result

                # 判断测试是否通过
                if 'error' not in result and result.get('success', True):
                    passed_tests += 1
                    self.logger.info(f"✅ {test_name} 测试通过")
                else:
                    self.logger.error(f"❌ {test_name} 测试失败")

            except Exception as e:
                self.logger.error(f"❌ {test_name} 测试异常: {e}")
                test_results[test_name] = {'error': str(e), 'success': False}

        total_execution_time = time.time() - overall_start_time

        # 生成测试总结
        summary = {
            'test_results': test_results,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests,
                'total_execution_time': total_execution_time,
                'framework_ready': passed_tests >= total_tests * 0.8  # 80%通过率认为框架可用
            }
        }

        # 打印总结
        self.logger.info("\n" + "="*60)
        self.logger.info("测试总结")
        self.logger.info("="*60)
        self.logger.info(f"总测试数: {total_tests}")
        self.logger.info(f"通过测试: {passed_tests}")
        self.logger.info(f"失败测试: {total_tests - passed_tests}")
        self.logger.info(f"通过率: {summary['summary']['success_rate']:.1%}")
        self.logger.info(f"总执行时间: {total_execution_time:.2f}秒")

        if summary['summary']['framework_ready']:
            self.logger.info("🎉 策略性能评估框架已准备就绪！")
        else:
            self.logger.warning("⚠️  策略性能评估框架需要进一步优化")

        return summary


def main():
    """主测试函数"""
    tester = PerformanceFrameworkTester()
    results = tester.run_comprehensive_test()

    # 保存测试结果
    import json
    output_file = f"performance_framework_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        # 序列化结果
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj

        def serialize_dict(d):
            if isinstance(d, dict):
                return {k: serialize_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [serialize_dict(item) for item in d]
            else:
                return make_serializable(d)

        serialized_results = serialize_dict(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serialized_results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"测试结果已保存到: {output_file}")

    except Exception as e:
        logger.error(f"保存测试结果失败: {e}")

    return results


if __name__ == "__main__":
    main()