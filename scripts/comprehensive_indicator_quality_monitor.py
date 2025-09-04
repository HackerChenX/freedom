#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全面指标质量监控测试脚本
定期执行所有已验证指标的五阶段测试，监控质量变化
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger
from db.enhanced_connection_pool import ClickHouseConnectionPool

logger = get_logger(__name__)


class ComprehensiveIndicatorQualityMonitor:
    """全面指标质量监控器"""
    
    def __init__(self):
        self.pool = ClickHouseConnectionPool()
        self.results = {}
        self.summary = {
            'total_indicators': 0,
            'passed_indicators': 0,
            'failed_indicators': 0,
            'warning_indicators': 0,
            'execution_time': 0,
            'test_timestamp': datetime.now().isoformat()
        }
        
        # 五阶段测试权重
        self.stage_weights = {
            'algorithm_correctness': 20,    # 算法正确性
            'numerical_reasonableness': 15, # 数值合理性
            'functional_completeness': 20,  # 功能完整性
            'performance': 10,              # 性能表现
            'stability': 10                 # 稳定性
        }
        
        # 质量阈值
        self.quality_thresholds = {
            'excellent': 95,  # 优秀
            'good': 85,       # 良好
            'acceptable': 75, # 可接受
            'poor': 60        # 较差
        }
    
    def get_verified_indicators(self) -> List[str]:
        """获取所有已验证的指标列表"""
        logger.info("📋 获取已验证指标列表...")
        
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            all_indicators = registry.get_all_indicators()
            
            # 过滤出已验证的指标（这里假设所有注册的指标都是已验证的）
            verified_indicators = list(all_indicators.keys())
            
            logger.info(f"  ✅ 发现 {len(verified_indicators)} 个已验证指标")
            return verified_indicators
            
        except Exception as e:
            logger.error(f"  ❌ 获取指标列表失败: {e}")
            return []
    
    def get_test_data(self, code: str = '000001', days: int = 100) -> pd.DataFrame:
        """获取测试数据"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate
            FROM stock_info 
            WHERE code = '{code}'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """
            
            with self.pool.get_connection() as conn:
                result = conn.query_dataframe(query)
                
            if result.empty:
                # 如果没有数据，生成模拟数据
                result = self._generate_mock_data(days)
                
            return result
            
        except Exception as e:
            logger.warning(f"获取真实数据失败，使用模拟数据: {e}")
            return self._generate_mock_data(days)
    
    def _generate_mock_data(self, days: int) -> pd.DataFrame:
        """生成模拟测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # 生成模拟价格数据
        base_price = 100
        price_changes = np.random.normal(0, 0.02, days)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # 确保价格为正
        
        data = pd.DataFrame({
            'date': dates,
            'code': '000001',
            'name': '测试股票',
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, days),
            'turnover_rate': np.random.uniform(0.5, 5.0, days)
        })
        
        return data
    
    def test_algorithm_correctness(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段1：算法正确性测试"""
        try:
            result = indicator.calculate(data)
            
            if result is None or result.empty:
                return 0, "计算结果为空"
            
            # 检查结果的基本特征
            score = 20
            issues = []
            
            # 检查是否包含NaN值
            if result.isnull().any().any():
                score -= 5
                issues.append("包含NaN值")
            
            # 检查数据长度合理性
            if len(result) < len(data) * 0.5:
                score -= 5
                issues.append("结果长度过短")
            
            # 检查数值范围合理性
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if result[col].abs().max() > 1e6:
                    score -= 3
                    issues.append(f"{col}列数值过大")
            
            return max(score, 0), "; ".join(issues) if issues else "算法正确"
            
        except Exception as e:
            return 0, f"计算异常: {str(e)[:100]}"
    
    def test_numerical_reasonableness(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段2：数值合理性测试"""
        try:
            result = indicator.calculate(data)
            
            if result is None or result.empty:
                return 0, "无计算结果"
            
            score = 15
            issues = []
            
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                values = result[col].dropna()
                if len(values) == 0:
                    continue
                
                # 检查极值
                if values.min() < -1000 or values.max() > 1000:
                    score -= 2
                    issues.append(f"{col}存在极值")
                
                # 检查方差
                if values.var() == 0:
                    score -= 1
                    issues.append(f"{col}无变化")
                elif values.var() > 10000:
                    score -= 1
                    issues.append(f"{col}波动过大")
            
            return max(score, 0), "; ".join(issues) if issues else "数值合理"
            
        except Exception as e:
            return 0, f"数值检查异常: {str(e)[:100]}"
    
    def test_functional_completeness(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段3：功能完整性测试"""
        try:
            score = 20
            issues = []
            
            # 测试基本方法
            required_methods = ['calculate']
            for method in required_methods:
                if not hasattr(indicator, method):
                    score -= 5
                    issues.append(f"缺少{method}方法")
            
            # 测试计算方法
            try:
                result = indicator.calculate(data)
                if result is None:
                    score -= 10
                    issues.append("calculate返回None")
            except Exception as e:
                score -= 10
                issues.append(f"calculate方法异常: {str(e)[:50]}")
            
            # 测试BaseIndicator方法
            base_methods = [
                '_calculate_baseindicator',
                'calculate_raw_score_Indicator_Base_Indicator',
                'get_patterns_Indicator_Base_Indicator',
                'calculate_confidence_Indicator_Base_Indicator'
            ]
            
            for method in base_methods:
                if hasattr(indicator, method):
                    try:
                        if method == '_calculate_baseindicator':
                            indicator._calculate_baseindicator(data)
                        elif method == 'calculate_raw_score_Indicator_Base_Indicator':
                            indicator.calculate_raw_score_Indicator_Base_Indicator(data)
                        elif method == 'get_patterns_Indicator_Base_Indicator':
                            indicator.get_patterns_Indicator_Base_Indicator(data)
                        elif method == 'calculate_confidence_Indicator_Base_Indicator':
                            indicator.calculate_confidence_Indicator_Base_Indicator(data)
                    except Exception as e:
                        score -= 2
                        issues.append(f"{method}异常")
            
            return max(score, 0), "; ".join(issues) if issues else "功能完整"
            
        except Exception as e:
            return 0, f"功能测试异常: {str(e)[:100]}"
    
    def test_performance(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段4：性能表现测试"""
        try:
            # 测试计算时间
            start_time = time.time()
            result = indicator.calculate(data)
            execution_time = time.time() - start_time
            
            score = 10
            issues = []
            
            # 性能评分
            if execution_time > 5.0:
                score = 0
                issues.append(f"执行时间过长: {execution_time:.2f}s")
            elif execution_time > 2.0:
                score = 3
                issues.append(f"执行时间较长: {execution_time:.2f}s")
            elif execution_time > 1.0:
                score = 6
                issues.append(f"执行时间一般: {execution_time:.2f}s")
            else:
                issues.append(f"执行时间良好: {execution_time:.3f}s")
            
            return score, "; ".join(issues)
            
        except Exception as e:
            return 0, f"性能测试异常: {str(e)[:100]}"
    
    def test_stability(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段5：稳定性测试"""
        try:
            score = 10
            issues = []
            
            # 多次计算一致性测试
            results = []
            for i in range(3):
                try:
                    result = indicator.calculate(data)
                    if result is not None and not result.empty:
                        results.append(result)
                except Exception as e:
                    score -= 3
                    issues.append(f"第{i+1}次计算失败")
            
            # 检查结果一致性
            if len(results) >= 2:
                try:
                    # 比较前两次结果
                    if not results[0].equals(results[1]):
                        # 检查数值差异
                        numeric_cols = results[0].select_dtypes(include=[np.number]).columns
                        max_diff = 0
                        for col in numeric_cols:
                            if col in results[1].columns:
                                diff = abs(results[0][col] - results[1][col]).max()
                                max_diff = max(max_diff, diff)
                        
                        if max_diff > 1e-10:
                            score -= 2
                            issues.append(f"结果不一致，最大差异: {max_diff}")
                except Exception as e:
                    score -= 1
                    issues.append("一致性检查异常")
            
            # 边界条件测试
            try:
                # 测试最小数据集
                min_data = data.head(max(10, getattr(indicator, 'period', 10)))
                indicator.calculate(min_data)
            except Exception as e:
                score -= 2
                issues.append("最小数据集测试失败")
            
            return max(score, 0), "; ".join(issues) if issues else "稳定性良好"
            
        except Exception as e:
            return 0, f"稳定性测试异常: {str(e)[:100]}"
    
    def test_single_indicator(self, indicator_name: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """测试单个指标的五阶段质量"""
        logger.info(f"  🔍 测试指标: {indicator_name}")
        
        try:
            # 创建指标实例
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            indicator = registry.create_indicator(indicator_name)
            
            if indicator is None:
                return {
                    'indicator_name': indicator_name,
                    'total_score': 0,
                    'status': 'FAILED',
                    'error': '无法创建指标实例',
                    'stages': {}
                }
            
            # 执行五阶段测试
            stages = {}
            total_score = 0
            
            # 阶段1：算法正确性
            score1, msg1 = self.test_algorithm_correctness(indicator_name, indicator, test_data)
            stages['algorithm_correctness'] = {'score': score1, 'message': msg1}
            total_score += score1
            
            # 阶段2：数值合理性
            score2, msg2 = self.test_numerical_reasonableness(indicator_name, indicator, test_data)
            stages['numerical_reasonableness'] = {'score': score2, 'message': msg2}
            total_score += score2
            
            # 阶段3：功能完整性
            score3, msg3 = self.test_functional_completeness(indicator_name, indicator, test_data)
            stages['functional_completeness'] = {'score': score3, 'message': msg3}
            total_score += score3
            
            # 阶段4：性能表现
            score4, msg4 = self.test_performance(indicator_name, indicator, test_data)
            stages['performance'] = {'score': score4, 'message': msg4}
            total_score += score4
            
            # 阶段5：稳定性
            score5, msg5 = self.test_stability(indicator_name, indicator, test_data)
            stages['stability'] = {'score': score5, 'message': msg5}
            total_score += score5
            
            # 确定状态
            if total_score >= self.quality_thresholds['excellent']:
                status = 'EXCELLENT'
            elif total_score >= self.quality_thresholds['good']:
                status = 'GOOD'
            elif total_score >= self.quality_thresholds['acceptable']:
                status = 'ACCEPTABLE'
            elif total_score >= self.quality_thresholds['poor']:
                status = 'POOR'
            else:
                status = 'FAILED'
            
            return {
                'indicator_name': indicator_name,
                'total_score': total_score,
                'status': status,
                'stages': stages,
                'test_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"    ❌ 测试 {indicator_name} 失败: {e}")
            return {
                'indicator_name': indicator_name,
                'total_score': 0,
                'status': 'ERROR',
                'error': str(e),
                'stages': {}
            }

    def run_comprehensive_test(self, test_codes: List[str] = None, max_indicators: int = None) -> Dict[str, Any]:
        """运行全面的指标质量测试"""
        logger.info("🚀 开始全面指标质量监控测试...")
        logger.info("=" * 80)

        start_time = time.time()

        # 获取测试数据
        test_codes = test_codes or ['000001', '000002', '600000']
        test_data_sets = {}

        for code in test_codes:
            logger.info(f"📊 获取 {code} 的测试数据...")
            test_data_sets[code] = self.get_test_data(code)

        # 使用第一个股票的数据作为主要测试数据
        primary_test_data = test_data_sets[test_codes[0]]

        # 获取已验证指标列表
        verified_indicators = self.get_verified_indicators()

        if max_indicators:
            verified_indicators = verified_indicators[:max_indicators]

        self.summary['total_indicators'] = len(verified_indicators)

        logger.info(f"📋 开始测试 {len(verified_indicators)} 个指标...")
        logger.info("=" * 80)

        # 测试每个指标
        for i, indicator_name in enumerate(verified_indicators, 1):
            logger.info(f"[{i}/{len(verified_indicators)}] 测试指标: {indicator_name}")

            result = self.test_single_indicator(indicator_name, primary_test_data)
            self.results[indicator_name] = result

            # 更新统计
            if result['status'] in ['EXCELLENT', 'GOOD', 'ACCEPTABLE']:
                self.summary['passed_indicators'] += 1
            elif result['status'] in ['POOR']:
                self.summary['warning_indicators'] += 1
            else:
                self.summary['failed_indicators'] += 1

            # 显示结果
            status_emoji = {
                'EXCELLENT': '🟢',
                'GOOD': '🟢',
                'ACCEPTABLE': '🟡',
                'POOR': '🟠',
                'FAILED': '🔴',
                'ERROR': '❌'
            }

            emoji = status_emoji.get(result['status'], '❓')
            score = result.get('total_score', 0)
            logger.info(f"    {emoji} {result['status']} - 总分: {score}/75")

        # 计算执行时间
        self.summary['execution_time'] = time.time() - start_time

        # 生成报告
        self._generate_report()

        logger.info("=" * 80)
        logger.info("🎉 全面指标质量监控测试完成！")
        logger.info(f"📊 测试结果: {self.summary['passed_indicators']}/{self.summary['total_indicators']} 通过")
        logger.info(f"⏱️ 执行时间: {self.summary['execution_time']:.2f} 秒")

        return {
            'summary': self.summary,
            'results': self.results
        }

    def _generate_report(self):
        """生成测试报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 生成JSON报告
        json_report_path = f"results/indicator_quality_monitor_{timestamp}.json"
        os.makedirs(os.path.dirname(json_report_path), exist_ok=True)

        report_data = {
            'summary': self.summary,
            'results': self.results,
            'test_config': {
                'stage_weights': self.stage_weights,
                'quality_thresholds': self.quality_thresholds
            }
        }

        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 生成Markdown报告
        md_report_path = f"results/indicator_quality_monitor_{timestamp}.md"
        self._generate_markdown_report(md_report_path)

        logger.info(f"📄 报告已生成:")
        logger.info(f"  - JSON: {json_report_path}")
        logger.info(f"  - Markdown: {md_report_path}")

    def _generate_markdown_report(self, report_path: str):
        """生成Markdown格式的报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# 指标质量监控报告

## 📊 测试概要

- **测试时间**: {self.summary['test_timestamp']}
- **测试指标数**: {self.summary['total_indicators']}
- **通过指标数**: {self.summary['passed_indicators']}
- **警告指标数**: {self.summary['warning_indicators']}
- **失败指标数**: {self.summary['failed_indicators']}
- **执行时间**: {self.summary['execution_time']:.2f} 秒
- **通过率**: {(self.summary['passed_indicators']/self.summary['total_indicators']*100):.1f}%

## 🎯 质量分布

""")

            # 按状态分组统计
            status_counts = {}
            for result in self.results.values():
                status = result['status']
                status_counts[status] = status_counts.get(status, 0) + 1

            for status, count in status_counts.items():
                percentage = count / self.summary['total_indicators'] * 100
                f.write(f"- **{status}**: {count}个 ({percentage:.1f}%)\n")

            f.write(f"""
## 📋 详细结果

| 指标名称 | 状态 | 总分 | 算法正确性 | 数值合理性 | 功能完整性 | 性能表现 | 稳定性 |
|---------|------|------|-----------|-----------|-----------|---------|-------|
""")

            # 按总分排序
            sorted_results = sorted(self.results.items(),
                                  key=lambda x: x[1].get('total_score', 0),
                                  reverse=True)

            for indicator_name, result in sorted_results:
                status = result['status']
                total_score = result.get('total_score', 0)
                stages = result.get('stages', {})

                # 获取各阶段分数
                alg_score = stages.get('algorithm_correctness', {}).get('score', 0)
                num_score = stages.get('numerical_reasonableness', {}).get('score', 0)
                func_score = stages.get('functional_completeness', {}).get('score', 0)
                perf_score = stages.get('performance', {}).get('score', 0)
                stab_score = stages.get('stability', {}).get('score', 0)

                status_emoji = {
                    'EXCELLENT': '🟢',
                    'GOOD': '🟢',
                    'ACCEPTABLE': '🟡',
                    'POOR': '🟠',
                    'FAILED': '🔴',
                    'ERROR': '❌'
                }

                emoji = status_emoji.get(status, '❓')

                f.write(f"| {indicator_name} | {emoji} {status} | {total_score}/75 | {alg_score}/20 | {num_score}/15 | {func_score}/20 | {perf_score}/10 | {stab_score}/10 |\n")

            f.write(f"""
## ⚠️ 需要关注的指标

""")

            # 列出需要关注的指标
            warning_indicators = []
            failed_indicators = []

            for indicator_name, result in self.results.items():
                if result['status'] in ['POOR']:
                    warning_indicators.append((indicator_name, result))
                elif result['status'] in ['FAILED', 'ERROR']:
                    failed_indicators.append((indicator_name, result))

            if warning_indicators:
                f.write("### 🟠 警告指标\n\n")
                for indicator_name, result in warning_indicators:
                    f.write(f"- **{indicator_name}**: 总分 {result.get('total_score', 0)}/75\n")
                    stages = result.get('stages', {})
                    for stage_name, stage_data in stages.items():
                        if stage_data.get('score', 0) < self.stage_weights.get(stage_name, 0) * 0.7:
                            f.write(f"  - {stage_name}: {stage_data.get('message', 'N/A')}\n")
                f.write("\n")

            if failed_indicators:
                f.write("### 🔴 失败指标\n\n")
                for indicator_name, result in failed_indicators:
                    f.write(f"- **{indicator_name}**: {result.get('error', '未知错误')}\n")
                f.write("\n")

            f.write(f"""
## 📈 建议

### 质量改进建议
1. 对于警告指标，建议检查具体的失败阶段并进行优化
2. 对于失败指标，建议重新审查代码实现
3. 定期运行此监控脚本，确保指标质量稳定

### 监控频率建议
- **日常监控**: 每周运行一次
- **版本发布前**: 必须运行并确保通过率 > 95%
- **重大变更后**: 立即运行验证

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='全面指标质量监控测试')
    parser.add_argument('--codes', nargs='+', default=['000001'],
                       help='测试股票代码列表')
    parser.add_argument('--max-indicators', type=int,
                       help='最大测试指标数量（用于快速测试）')
    parser.add_argument('--output-dir', default='results',
                       help='输出目录')

    args = parser.parse_args()

    try:
        monitor = ComprehensiveIndicatorQualityMonitor()
        results = monitor.run_comprehensive_test(
            test_codes=args.codes,
            max_indicators=args.max_indicators
        )

        # 返回状态码
        if results['summary']['failed_indicators'] == 0:
            return 0  # 成功
        else:
            return 1  # 有失败指标

    except Exception as e:
        logger.error(f"❌ 监控测试过程中发生异常: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        return 2  # 异常


if __name__ == "__main__":
    exit(main())
