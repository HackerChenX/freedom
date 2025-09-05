#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
技术指标系统集成测试脚本
验证所有63个指标在生产环境中的表现，确保系统集成正常
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import concurrent.futures
from threading import Lock

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class SystemIntegrationTester:
    """系统集成测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.test_lock = Lock()
        self.start_time = None
        
        # 所有63个技术指标
        self.all_indicators = [
            # BaseIndicator指标 (9个)
            'ADX', 'ROC', 'MFI', 'OBV', 'KC', 'VIX', 'MTM', 'SYNERGY', 'UNIFIED_MA',
            
            # ZXM体系指标 (35个)
            'ZXM_DAILY_MACD', 'ZXM_WEEKLY_MACD', 'ZXM_MONTHLY_MACD', 'ZXM_DAILY_KDJ', 'ZXM_WEEKLY_KDJ',
            'ZXM_DAILY_RSI', 'ZXM_WEEKLY_RSI', 'ZXM_DAILY_BOLL', 'ZXM_WEEKLY_BOLL', 'ZXM_DAILY_MA',
            'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_VOLUME_BREAKOUT', 'ZXM_VOLUME_PRICE_TREND',
            'ZXM_BUYPOINT_SCORE', 'ZXM_TREND_SCORE', 'ZXM_COMPREHENSIVE_SCORE',
            'ZXM_PRICE_POSITION', 'ZXM_TREND_STRENGTH', 'ZXM_SUPPORT_RESISTANCE', 'ZXM_BREAKOUT_SIGNAL',
            'ZXM_HOT_SPOT', 'ZXM_SECTOR_ROTATION',
            'ZXM_RISK_CONTROL', 'ZXM_POSITION_SIZING', 'ZXM_TIMING_SIGNAL', 'ZXM_STOP_LOSS',
            'ZXM_PORTFOLIO_OPTIMIZATION', 'ZXM_STRATEGY_COMBINATION', 'ZXM_PERFORMANCE_ATTRIBUTION',
            'ZXM_ALPHA_GENERATION', 'ZXM_BETA_HEDGING',
            'ZXM_MARKET_SENTIMENT', 'ZXM_LIQUIDITY_ANALYSIS', 'ZXM_VOLATILITY_FORECAST', 'ZXM_CORRELATION_MATRIX',
            
            # 形态识别指标 (19个)
            'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
            'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS',
            'V_SHAPED_REVERSAL', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
            'WEDGE', 'FLAG', 'PENNANT'
        ]
        
    def generate_production_test_data(self) -> pd.DataFrame:
        """生成生产级别测试数据"""
        logger.info("📊 生成生产级别测试数据...")
        
        # 生成300天的测试数据，模拟真实市场环境
        dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        base_volume = 2000000
        
        # 生成更复杂的市场数据
        # 长期趋势
        long_trend = np.linspace(0, 30, 300)
        # 中期周期
        medium_cycle = 15 * np.sin(np.linspace(0, 12*np.pi, 300))
        # 短期波动
        short_volatility = 5 * np.sin(np.linspace(0, 50*np.pi, 300))
        # 随机噪声
        noise = np.random.normal(0, 3, 300)
        
        price_changes = long_trend + medium_cycle + short_volatility + noise
        
        # 生成价格序列
        prices = [base_price]
        volumes = []
        
        for i in range(1, 300):
            # 价格计算
            trend_factor = price_changes[i] * 0.3
            volatility_factor = abs(price_changes[i]) * 0.05
            new_price = max(prices[-1] + trend_factor, 1.0)
            prices.append(new_price)
            
            # 成交量计算（与价格变化和波动率相关）
            price_change_pct = abs(trend_factor) / prices[-1]
            volume_factor = 1 + price_change_pct * 3 + volatility_factor
            new_volume = int(base_volume * volume_factor * np.random.uniform(0.7, 1.3))
            volumes.append(max(new_volume, 100000))
        
        volumes.append(base_volume)
        
        # 生成OHLC数据
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_volatility = abs(price_changes[i]) * 0.008
            
            # 生成更真实的OHLC
            high_factor = np.random.uniform(1.0, 1.0 + daily_volatility)
            low_factor = np.random.uniform(1.0 - daily_volatility, 1.0)
            
            high = price * high_factor
            low = price * low_factor
            
            open_price = prices[i-1] if i > 0 else price
            close = price
            
            # 确保OHLC关系正确
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # 偶尔生成特殊市场情况
            if i % 30 == 0:  # 每30天可能出现特殊情况
                if np.random.random() > 0.7:
                    # 跳空情况
                    gap_factor = np.random.uniform(0.95, 1.05)
                    open_price *= gap_factor
                    close *= gap_factor
                    high = max(high, open_price, close)
                    low = min(low, open_price, close)
            
            data.append({
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ 生成生产级别测试数据: {len(df)}行")
        return df
    
    def test_single_indicator_integration(self, indicator_name: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """测试单个指标的集成情况"""
        test_result = {
            'indicator': indicator_name,
            'start_time': time.time(),
            'status': 'TESTING',
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # 1. 指标创建测试
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            indicator = registry.create_indicator(indicator_name)
            
            if indicator is None:
                test_result['status'] = 'FAILED'
                test_result['errors'].append('指标创建失败')
                return test_result
            
            # 2. 计算功能测试
            calculation_start = time.time()
            result = indicator.calculate(test_data)
            calculation_time = time.time() - calculation_start
            
            test_result['metrics']['calculation_time'] = calculation_time
            
            if result is None:
                test_result['status'] = 'FAILED'
                test_result['errors'].append('计算返回None')
                return test_result
            
            if not isinstance(result, dict):
                test_result['status'] = 'FAILED'
                test_result['errors'].append('计算返回类型错误')
                return test_result
            
            if len(result) == 0:
                test_result['status'] = 'FAILED'
                test_result['errors'].append('计算返回空结果')
                return test_result
            
            # 3. 数据质量测试
            valid_data_count = 0
            total_data_count = 0
            
            for key, value in result.items():
                if isinstance(value, (int, float, np.number)):
                    total_data_count += 1
                    if not (np.isnan(value) or np.isinf(value)):
                        valid_data_count += 1
                elif hasattr(value, '__len__') and hasattr(value, 'notna'):
                    total_data_count += len(value)
                    valid_data_count += value.notna().sum()
                else:
                    total_data_count += 1
                    valid_data_count += 1
            
            data_quality = valid_data_count / total_data_count if total_data_count > 0 else 0
            test_result['metrics']['data_quality'] = data_quality
            
            if data_quality < 0.95:
                test_result['warnings'].append(f'数据质量{data_quality:.1%}，低于95%标准')
            
            # 4. 性能测试
            if calculation_time > 0.1:
                test_result['warnings'].append(f'计算时间{calculation_time:.3f}秒，超过100ms标准')
            
            # 5. get_patterns方法测试
            try:
                patterns = indicator.get_patterns()
                if patterns is None or not isinstance(patterns, dict):
                    test_result['warnings'].append('get_patterns方法返回异常')
                else:
                    test_result['metrics']['patterns_count'] = len(patterns)
            except Exception as e:
                test_result['warnings'].append(f'get_patterns方法异常: {e}')
            
            # 6. 内存使用测试
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            test_result['metrics']['memory_usage_mb'] = memory_usage
            
            # 7. 并发测试
            concurrent_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(indicator.calculate, test_data) for _ in range(3)]
                concurrent_results = [future.result() for future in futures]
            concurrent_time = time.time() - concurrent_start
            
            test_result['metrics']['concurrent_time'] = concurrent_time
            
            # 检查并发结果一致性
            if len(set(str(r) for r in concurrent_results)) > 1:
                test_result['warnings'].append('并发计算结果不一致')
            
            # 8. 综合评分
            score = 100
            if test_result['errors']:
                score = 0
            else:
                if calculation_time > 0.1:
                    score -= 10
                if data_quality < 0.95:
                    score -= 20
                if memory_usage > 100:  # 100MB
                    score -= 10
                if test_result['warnings']:
                    score -= len(test_result['warnings']) * 5
            
            test_result['metrics']['integration_score'] = max(0, score)
            test_result['status'] = 'PASSED' if score >= 80 else 'WARNING' if score >= 60 else 'FAILED'
            
        except Exception as e:
            test_result['status'] = 'FAILED'
            test_result['errors'].append(f'集成测试异常: {e}')
            test_result['metrics']['integration_score'] = 0
        
        test_result['end_time'] = time.time()
        test_result['total_time'] = test_result['end_time'] - test_result['start_time']
        
        return test_result
    
    def run_system_integration_test(self) -> Dict[str, Any]:
        """运行完整的系统集成测试"""
        logger.info(f"🚀 开始系统集成测试...")
        logger.info(f"📊 测试指标数量: {len(self.all_indicators)}个")
        
        self.start_time = time.time()
        
        # 生成测试数据
        test_data = self.generate_production_test_data()
        
        # 并发测试所有指标
        test_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有测试任务
            future_to_indicator = {
                executor.submit(self.test_single_indicator_integration, indicator, test_data): indicator
                for indicator in self.all_indicators
            }
            
            # 收集测试结果
            completed = 0
            for future in concurrent.futures.as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                try:
                    result = future.result()
                    test_results[indicator_name] = result
                    
                    completed += 1
                    status_icon = "✅" if result['status'] == 'PASSED' else "⚠️" if result['status'] == 'WARNING' else "❌"
                    logger.info(f"{status_icon} [{completed}/{len(self.all_indicators)}] {indicator_name}: {result['status']} ({result['metrics'].get('integration_score', 0):.1f}分)")
                    
                except Exception as e:
                    logger.error(f"❌ {indicator_name} 测试失败: {e}")
                    test_results[indicator_name] = {
                        'indicator': indicator_name,
                        'status': 'FAILED',
                        'errors': [str(e)],
                        'metrics': {'integration_score': 0}
                    }
        
        total_time = time.time() - self.start_time
        
        # 计算总体统计
        total_indicators = len(test_results)
        passed_indicators = sum(1 for r in test_results.values() if r['status'] == 'PASSED')
        warning_indicators = sum(1 for r in test_results.values() if r['status'] == 'WARNING')
        failed_indicators = sum(1 for r in test_results.values() if r['status'] == 'FAILED')
        
        pass_rate = (passed_indicators / total_indicators) * 100 if total_indicators > 0 else 0
        
        scores = [r['metrics'].get('integration_score', 0) for r in test_results.values()]
        average_score = sum(scores) / len(scores) if scores else 0
        
        avg_calculation_time = np.mean([r['metrics'].get('calculation_time', 0) for r in test_results.values()])
        avg_data_quality = np.mean([r['metrics'].get('data_quality', 0) for r in test_results.values()])
        avg_memory_usage = np.mean([r['metrics'].get('memory_usage_mb', 0) for r in test_results.values()])
        
        summary = {
            'test_type': 'SYSTEM_INTEGRATION_TEST',
            'total_indicators': total_indicators,
            'passed_indicators': passed_indicators,
            'warning_indicators': warning_indicators,
            'failed_indicators': failed_indicators,
            'pass_rate': pass_rate,
            'average_score': average_score,
            'performance_metrics': {
                'avg_calculation_time': avg_calculation_time,
                'avg_data_quality': avg_data_quality,
                'avg_memory_usage_mb': avg_memory_usage,
                'total_test_time': total_time
            },
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 系统集成测试完成!")
        logger.info(f"📊 总指标数: {total_indicators}个")
        logger.info(f"📊 通过率: {pass_rate:.1f}% (通过:{passed_indicators}, 警告:{warning_indicators}, 失败:{failed_indicators})")
        logger.info(f"📊 平均得分: {average_score:.1f}/100")
        logger.info(f"📊 平均计算时间: {avg_calculation_time:.3f}秒")
        logger.info(f"📊 平均数据质量: {avg_data_quality:.1%}")
        logger.info(f"⏱️ 总测试时间: {total_time:.2f}秒")
        
        return summary


def main():
    """主函数"""
    logger.info("🔍 系统集成测试开始...")
    
    tester = SystemIntegrationTester()
    result = tester.run_system_integration_test()
    
    # 保存集成测试报告
    report_file = f"docs/finaltesting/indicators/system_integration_test_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成集成测试报告
    passed_indicators = [name for name, data in result['test_results'].items() if data['status'] == 'PASSED']
    warning_indicators = [name for name, data in result['test_results'].items() if data['status'] == 'WARNING']
    failed_indicators = [name for name, data in result['test_results'].items() if data['status'] == 'FAILED']
    
    report_content = f"""# 系统集成测试报告

## 测试概览
- **测试类型**: 系统集成测试
- **测试时间**: {result['timestamp']}
- **测试指标数**: {result['total_indicators']}个
- **通过率**: {result['pass_rate']:.1f}%
- **平均得分**: {result['average_score']:.1f}/100

## 测试结果统计
- **✅ 通过**: {result['passed_indicators']}个
- **⚠️ 警告**: {result['warning_indicators']}个
- **❌ 失败**: {result['failed_indicators']}个

## 性能指标
- **平均计算时间**: {result['performance_metrics']['avg_calculation_time']:.3f}秒
- **平均数据质量**: {result['performance_metrics']['avg_data_quality']:.1%}
- **平均内存使用**: {result['performance_metrics']['avg_memory_usage_mb']:.1f}MB
- **总测试时间**: {result['performance_metrics']['total_test_time']:.2f}秒

## 通过的指标
{chr(10).join([f"- **{name}**: {result['test_results'][name]['metrics'].get('integration_score', 0):.1f}分" for name in passed_indicators])}

## 警告的指标
{chr(10).join([f"- **{name}**: {result['test_results'][name]['metrics'].get('integration_score', 0):.1f}分 - {', '.join(result['test_results'][name].get('warnings', []))}" for name in warning_indicators])}

## 失败的指标
{chr(10).join([f"- **{name}**: {', '.join(result['test_results'][name].get('errors', []))}" for name in failed_indicators])}

## 集成测试结论

{'### 🎉 系统集成测试通过！' if result['pass_rate'] >= 90 else '### ⚠️ 系统集成测试需要优化'}

{'所有指标都能正常集成运行，系统准备就绪。' if result['pass_rate'] >= 90 else f'有{result["failed_indicators"] + result["warning_indicators"]}个指标需要进一步优化。'}

---
*测试时间: {result['performance_metrics']['total_test_time']:.2f}秒*
*测试工具: 系统集成测试框架*
*质量保证: 生产级别标准*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 集成测试报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
