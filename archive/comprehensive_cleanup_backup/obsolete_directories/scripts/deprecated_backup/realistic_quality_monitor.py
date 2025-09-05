#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
现实化指标质量监控脚本
基于实际业务需求的合理测试标准
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

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class RealisticQualityMonitor:
    """现实化指标质量监控器"""
    
    def __init__(self):
        self.results = {}
        self.summary = {
            'total_indicators': 0,
            'excellent_indicators': 0,
            'good_indicators': 0,
            'acceptable_indicators': 0,
            'poor_indicators': 0,
            'failed_indicators': 0,
            'execution_time': 0,
            'test_timestamp': datetime.now().isoformat()
        }
        
        # 现实化质量阈值
        self.quality_thresholds = {
            'excellent': 85,    # 优秀
            'good': 75,         # 良好  
            'acceptable': 65,   # 可接受
            'poor': 50          # 较差
        }
        
        # 指标特性配置
        self.indicator_configs = {
            # 移动平均类 - 允许更多NaN（预热期）
            'MA': {'nan_tolerance': 0.15, 'min_data_ratio': 0.8},
            'EMA': {'nan_tolerance': 0.15, 'min_data_ratio': 0.8},
            'WMA': {'nan_tolerance': 0.15, 'min_data_ratio': 0.8},
            
            # 复合指标 - 需要更多预热期
            'MACD': {'nan_tolerance': 0.20, 'min_data_ratio': 0.7},
            'KDJ': {'nan_tolerance': 0.10, 'min_data_ratio': 0.8},
            'RSI': {'nan_tolerance': 0.10, 'min_data_ratio': 0.8},
            
            # 波动性指标 - 允许一定的极值
            'BOLL': {'nan_tolerance': 0.10, 'min_data_ratio': 0.8, 'allow_large_values': True},
            'ATR': {'nan_tolerance': 0.05, 'min_data_ratio': 0.9, 'allow_large_values': True},
            
            # 默认配置
            'default': {'nan_tolerance': 0.05, 'min_data_ratio': 0.9}
        }
    
    def get_indicator_config(self, indicator_name: str) -> Dict:
        """获取指标特定配置"""
        return self.indicator_configs.get(indicator_name, self.indicator_configs['default'])
    
    def generate_realistic_test_data(self, days: int = 300) -> pd.DataFrame:
        """生成现实化的测试数据"""
        np.random.seed(42)  # 固定种子确保可重复性
        
        # 生成更长的时间序列
        dates = pd.date_range(start='2023-01-01', periods=days, freq='B')
        
        # 模拟真实股价走势
        base_price = 10.0
        trend = 0.0001  # 轻微上升趋势
        volatility = 0.02
        
        prices = [base_price]
        for i in range(1, days):
            # 加入趋势和随机波动
            daily_return = trend + np.random.normal(0, volatility)
            # 加入一些跳跃（模拟重大事件）
            if np.random.random() < 0.02:  # 2%概率的跳跃
                daily_return += np.random.normal(0, 0.05)
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.1))
        
        # 生成OHLC数据
        data_list = []
        for i, close in enumerate(prices):
            if i == 0:
                open_price = close
            else:
                # 开盘价基于前一日收盘价，加入跳空
                gap = np.random.normal(0, 0.003)
                open_price = prices[i-1] * (1 + gap)
            
            # 日内高低价
            intraday_range = abs(np.random.normal(0, 0.01))
            high = max(open_price, close) * (1 + intraday_range)
            low = min(open_price, close) * (1 - intraday_range)
            
            # 成交量（对数正态分布，更真实）
            volume = int(np.random.lognormal(13.8, 0.6))  # 约100万平均
            
            # 换手率
            turnover_rate = np.random.gamma(2, 1.5)  # 伽马分布，更真实
            
            data_list.append({
                'date': dates[i] if i < len(dates) else dates[-1] + timedelta(days=i-len(dates)+1),
                'code': '000001',
                'name': '平安银行',
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'turnover_rate': min(turnover_rate, 15.0)  # 限制最大换手率
            })
        
        df = pd.DataFrame(data_list)
        logger.info(f"✅ 生成 {len(df)} 个现实化测试数据点")
        return df
    
    def test_basic_functionality(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """基础功能测试（40分）"""
        try:
            score = 40
            issues = []
            
            # 测试基本计算能力
            try:
                result = indicator.calculate(data)
                if result is None:
                    return 0, "calculate方法返回None"
                if hasattr(result, 'empty') and result.empty:
                    return 0, "calculate方法返回空DataFrame"
            except Exception as e:
                return 0, f"calculate方法异常: {str(e)[:50]}"
            
            # 检查结果基本特征
            config = self.get_indicator_config(indicator_name)
            
            # 检查数据长度
            expected_min_length = len(data) * config['min_data_ratio']
            if len(result) < expected_min_length:
                score -= 10
                issues.append(f"结果长度不足: {len(result)}/{len(data)}")
            
            # 检查NaN比例（根据指标特性调整）
            if hasattr(result, 'isnull'):
                total_cells = len(result) * len(result.columns) if len(result.columns) > 0 else len(result)
                nan_cells = result.isnull().sum().sum() if total_cells > 0 else 0
                nan_ratio = nan_cells / total_cells if total_cells > 0 else 0
                
                if nan_ratio > config['nan_tolerance']:
                    score -= 15
                    issues.append(f"NaN比例过高: {nan_ratio:.2%}")
                elif nan_ratio > config['nan_tolerance'] * 0.5:
                    score -= 5
                    issues.append(f"NaN比例较高: {nan_ratio:.2%}")
            
            # 检查数值有效性
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                values = result[col].dropna()
                if len(values) > 0:
                    # 检查无穷值
                    if np.isinf(values).any():
                        score -= 10
                        issues.append(f"{col}包含无穷值")
                    
                    # 检查极值（根据指标特性）
                    if not config.get('allow_large_values', False):
                        if values.abs().max() > 10000:
                            score -= 8
                            issues.append(f"{col}存在极值")
            
            # 测试BaseIndicator接口
            base_methods = ['_calculate_baseindicator', 'calculate_raw_score_Indicator_Base_Indicator']
            for method in base_methods:
                if hasattr(indicator, method):
                    try:
                        if method == '_calculate_baseindicator':
                            indicator._calculate_baseindicator(data)
                        elif method == 'calculate_raw_score_Indicator_Base_Indicator':
                            score_val = indicator.calculate_raw_score_Indicator_Base_Indicator(data)
                            if not isinstance(score_val, (int, float)):
                                score -= 3
                                issues.append(f"{method}返回类型错误")
                    except Exception as e:
                        score -= 2
                        issues.append(f"{method}执行异常")
            
            return max(score, 0), "; ".join(issues) if issues else "基础功能正常"
            
        except Exception as e:
            return 0, f"基础功能测试异常: {str(e)[:100]}"
    
    def test_business_logic(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """业务逻辑测试（30分）"""
        try:
            score = 30
            issues = []
            
            result = indicator.calculate(data)
            if result is None or (hasattr(result, 'empty') and result.empty):
                return 0, "无计算结果"
            
            # 根据指标类型进行业务逻辑检查
            if indicator_name in ['RSI', 'KDJ', 'STOCHRSI']:
                # 振荡器指标应该在合理范围内
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    values = result[col].dropna()
                    if len(values) > 0:
                        if values.min() < -20 or values.max() > 120:
                            score -= 10
                            issues.append(f"{col}超出振荡器合理范围")
                        elif values.min() < 0 or values.max() > 100:
                            score -= 5
                            issues.append(f"{col}轻微超出标准范围[0,100]")
            
            elif indicator_name in ['MA', 'EMA', 'WMA']:
                # 移动平均应该跟随价格趋势
                if 'close' in data.columns and len(result) > 0:
                    price_col = 'close'
                    ma_col = result.columns[0] if len(result.columns) > 0 else None
                    
                    if ma_col is not None:
                        # 检查移动平均是否合理跟随价格
                        common_indices = result.index.intersection(data.index)
                        if len(common_indices) > 10:
                            price_values = data.loc[common_indices, price_col]
                            ma_values = result.loc[common_indices, ma_col].dropna()
                            
                            if len(ma_values) > 0:
                                # 移动平均应该在价格附近
                                price_ma_ratio = abs(ma_values.mean() / price_values.mean() - 1)
                                if price_ma_ratio > 0.5:
                                    score -= 15
                                    issues.append("移动平均偏离价格过远")
                                elif price_ma_ratio > 0.2:
                                    score -= 8
                                    issues.append("移动平均偏离价格较远")
            
            elif indicator_name == 'MACD':
                # MACD应该围绕零轴震荡
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if 'macd' in col.lower():
                        values = result[col].dropna()
                        if len(values) > 0:
                            mean_abs = abs(values.mean())
                            std_val = values.std()
                            if std_val > 0 and mean_abs > std_val * 3:
                                score -= 8
                                issues.append(f"MACD偏离零轴过远")
            
            # 检查数据连续性
            if len(result) > 1:
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    values = result[col].dropna()
                    if len(values) > 2:
                        # 检查是否有异常跳跃
                        diff = values.diff().abs()
                        if len(diff) > 0:
                            mean_diff = diff.mean()
                            max_diff = diff.max()
                            if mean_diff > 0 and max_diff > mean_diff * 20:
                                score -= 5
                                issues.append(f"{col}存在异常跳跃")
            
            return max(score, 0), "; ".join(issues) if issues else "业务逻辑合理"
            
        except Exception as e:
            return 0, f"业务逻辑测试异常: {str(e)[:100]}"
    
    def test_performance_stability(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """性能稳定性测试（30分）"""
        try:
            score = 30
            issues = []
            
            # 性能测试（15分）
            start_time = time.time()
            result1 = indicator.calculate(data)
            execution_time = time.time() - start_time
            
            if execution_time > 10.0:
                score -= 15
                issues.append(f"执行时间过长: {execution_time:.2f}s")
            elif execution_time > 5.0:
                score -= 10
                issues.append(f"执行时间较长: {execution_time:.2f}s")
            elif execution_time > 2.0:
                score -= 5
                issues.append(f"执行时间一般: {execution_time:.2f}s")
            
            # 稳定性测试（15分）
            try:
                # 多次计算一致性
                result2 = indicator.calculate(data)
                
                if result1 is not None and result2 is not None:
                    if hasattr(result1, 'equals') and hasattr(result2, 'equals'):
                        if not result1.equals(result2):
                            # 检查数值差异
                            try:
                                numeric_cols = result1.select_dtypes(include=[np.number]).columns
                                max_diff = 0
                                for col in numeric_cols:
                                    if col in result2.columns:
                                        diff = abs(result1[col] - result2[col]).max()
                                        if not np.isnan(diff):
                                            max_diff = max(max_diff, diff)
                                
                                if max_diff > 1e-6:
                                    score -= 8
                                    issues.append(f"计算结果不一致: 最大差异{max_diff}")
                                else:
                                    score -= 2
                                    issues.append("轻微计算差异")
                            except:
                                score -= 5
                                issues.append("一致性检查异常")
                
                # 边界条件测试
                if len(data) > 50:
                    min_data = data.head(max(30, getattr(indicator, 'period', 20) + 10))
                    try:
                        result3 = indicator.calculate(min_data)
                        if result3 is None:
                            score -= 5
                            issues.append("最小数据集测试失败")
                    except Exception as e:
                        score -= 3
                        issues.append("边界条件测试异常")
                        
            except Exception as e:
                score -= 10
                issues.append(f"稳定性测试异常: {str(e)[:50]}")
            
            return max(score, 0), "; ".join(issues) if issues else f"性能稳定 ({execution_time:.3f}s)"
            
        except Exception as e:
            return 0, f"性能稳定性测试异常: {str(e)[:100]}"

    def test_single_indicator(self, indicator_name: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """测试单个指标（现实化标准）"""
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

            # 执行三阶段测试
            stages = {}
            total_score = 0

            # 阶段1：基础功能测试（40分）
            score1, msg1 = self.test_basic_functionality(indicator_name, indicator, test_data)
            stages['basic_functionality'] = {'score': score1, 'message': msg1, 'weight': 40}
            total_score += score1

            # 阶段2：业务逻辑测试（30分）
            score2, msg2 = self.test_business_logic(indicator_name, indicator, test_data)
            stages['business_logic'] = {'score': score2, 'message': msg2, 'weight': 30}
            total_score += score2

            # 阶段3：性能稳定性测试（30分）
            score3, msg3 = self.test_performance_stability(indicator_name, indicator, test_data)
            stages['performance_stability'] = {'score': score3, 'message': msg3, 'weight': 30}
            total_score += score3

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

    def run_realistic_test(self, max_indicators: int = None) -> Dict[str, Any]:
        """运行现实化指标质量测试"""
        logger.info("🚀 开始现实化指标质量监控测试...")
        logger.info("=" * 80)

        start_time = time.time()

        # 生成现实化测试数据
        logger.info("📊 生成现实化测试数据...")
        test_data = self.generate_realistic_test_data(300)  # 300个交易日

        # 获取已验证指标列表
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            all_indicators = registry.get_all_indicators()
            verified_indicators = list(all_indicators.keys())

            if max_indicators:
                verified_indicators = verified_indicators[:max_indicators]

            self.summary['total_indicators'] = len(verified_indicators)
            logger.info(f"📋 开始测试 {len(verified_indicators)} 个指标...")

        except Exception as e:
            logger.error(f"❌ 获取指标列表失败: {e}")
            return {'summary': self.summary, 'results': {}}

        logger.info("=" * 80)

        # 测试每个指标
        for i, indicator_name in enumerate(verified_indicators, 1):
            logger.info(f"[{i}/{len(verified_indicators)}] 测试指标: {indicator_name}")

            result = self.test_single_indicator(indicator_name, test_data)
            self.results[indicator_name] = result

            # 更新统计
            status = result['status']
            if status == 'EXCELLENT':
                self.summary['excellent_indicators'] += 1
            elif status == 'GOOD':
                self.summary['good_indicators'] += 1
            elif status == 'ACCEPTABLE':
                self.summary['acceptable_indicators'] += 1
            elif status == 'POOR':
                self.summary['poor_indicators'] += 1
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

            emoji = status_emoji.get(status, '❓')
            score = result.get('total_score', 0)
            logger.info(f"    {emoji} {status} - 总分: {score}/100")

        # 计算执行时间
        self.summary['execution_time'] = time.time() - start_time

        # 生成报告
        self._generate_realistic_report()

        # 计算通过率
        passed = (self.summary['excellent_indicators'] +
                 self.summary['good_indicators'] +
                 self.summary['acceptable_indicators'])

        logger.info("=" * 80)
        logger.info("🎉 现实化指标质量监控测试完成！")
        logger.info(f"📊 测试结果: {passed}/{self.summary['total_indicators']} 通过")
        logger.info(f"⏱️ 执行时间: {self.summary['execution_time']:.2f} 秒")

        return {
            'summary': self.summary,
            'results': self.results
        }

    def _generate_realistic_report(self):
        """生成现实化测试报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 生成Markdown报告
        md_report_path = f"results/realistic_quality_monitor_{timestamp}.md"
        os.makedirs(os.path.dirname(md_report_path), exist_ok=True)

        with open(md_report_path, 'w', encoding='utf-8') as f:
            passed = (self.summary['excellent_indicators'] +
                     self.summary['good_indicators'] +
                     self.summary['acceptable_indicators'])

            f.write(f"""# 现实化指标质量监控报告

## 📊 测试概要

- **测试时间**: {self.summary['test_timestamp']}
- **测试指标数**: {self.summary['total_indicators']}
- **优秀指标数**: {self.summary['excellent_indicators']} (≥85分)
- **良好指标数**: {self.summary['good_indicators']} (≥75分)
- **可接受指标数**: {self.summary['acceptable_indicators']} (≥65分)
- **较差指标数**: {self.summary['poor_indicators']} (≥50分)
- **失败指标数**: {self.summary['failed_indicators']} (<50分)
- **执行时间**: {self.summary['execution_time']:.2f} 秒
- **通过率**: {(passed/self.summary['total_indicators']*100):.1f}%

## 🎯 质量分布

""")

            # 按状态分组统计
            status_counts = {
                'EXCELLENT': self.summary['excellent_indicators'],
                'GOOD': self.summary['good_indicators'],
                'ACCEPTABLE': self.summary['acceptable_indicators'],
                'POOR': self.summary['poor_indicators'],
                'FAILED': self.summary['failed_indicators']
            }

            for status, count in status_counts.items():
                if count > 0:
                    percentage = count / self.summary['total_indicators'] * 100
                    f.write(f"- **{status}**: {count}个 ({percentage:.1f}%)\n")

            f.write(f"""
## 📋 详细结果

| 指标名称 | 状态 | 总分 | 基础功能 | 业务逻辑 | 性能稳定性 |
|---------|------|------|----------|----------|-----------|
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
                basic_score = stages.get('basic_functionality', {}).get('score', 0)
                business_score = stages.get('business_logic', {}).get('score', 0)
                perf_score = stages.get('performance_stability', {}).get('score', 0)

                status_emoji = {
                    'EXCELLENT': '🟢',
                    'GOOD': '🟢',
                    'ACCEPTABLE': '🟡',
                    'POOR': '🟠',
                    'FAILED': '🔴',
                    'ERROR': '❌'
                }

                emoji = status_emoji.get(status, '❓')

                f.write(f"| {indicator_name} | {emoji} {status} | {total_score}/100 | {basic_score}/40 | {business_score}/30 | {perf_score}/30 |\n")

            f.write(f"""
## 📈 测试标准说明

### 现实化测试方法
本次测试采用更贴近实际业务需求的测试标准：

1. **基础功能测试 (40分)**
   - 指标能否正常计算
   - 结果数据完整性检查
   - 根据指标特性调整NaN容忍度
   - BaseIndicator接口完整性

2. **业务逻辑测试 (30分)**
   - 指标数值是否符合业务逻辑
   - 不同类型指标的专门检查
   - 数据连续性和合理性验证

3. **性能稳定性测试 (30分)**
   - 计算性能要求 (<10秒)
   - 多次计算一致性
   - 边界条件处理能力

### 质量等级标准
- **优秀 (≥85分)**: 生产级质量，可直接使用
- **良好 (≥75分)**: 质量良好，轻微优化后可用
- **可接受 (≥65分)**: 基本可用，需要一定优化
- **较差 (≥50分)**: 存在问题，需要重点关注
- **失败 (<50分)**: 严重问题，需要修复

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

        logger.info(f"📄 现实化测试报告已生成: {md_report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='现实化指标质量监控测试')
    parser.add_argument('--max-indicators', type=int, default=20,
                       help='最大测试指标数量（默认20个）')

    args = parser.parse_args()

    try:
        monitor = RealisticQualityMonitor()
        results = monitor.run_realistic_test(max_indicators=args.max_indicators)

        # 计算通过率
        passed = (results['summary']['excellent_indicators'] +
                 results['summary']['good_indicators'] +
                 results['summary']['acceptable_indicators'])
        total = results['summary']['total_indicators']

        if passed / total >= 0.8:  # 80%通过率
            return 0  # 成功
        else:
            return 1  # 需要关注

    except Exception as e:
        logger.error(f"❌ 现实化监控测试过程中发生异常: {e}")
        return 2  # 异常


if __name__ == "__main__":
    exit(main())
