#!/usr/bin/env python3
"""
全面指标测试器 - 覆盖88个指标的所有形态
基于ClickHouse真实数据，关注性能问题，遇到错误及时停止修复
"""

import sys
import os
import time
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.production_indicator_validator import ProductionIndicatorValidator
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TestResult:
    """测试结果数据类"""
    indicator_name: str
    success: bool
    execution_time: float
    stock_count: int
    selected_stocks: int
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict] = None

@dataclass
class BatchResult:
    """批次测试结果数据类"""
    batch_name: str
    total_indicators: int
    successful_indicators: int
    failed_indicators: int
    total_time: float
    test_results: List[TestResult]

class ComprehensiveIndicatorTester:
    """全面指标测试器 - 支持88个指标的分批测试"""
    
    def __init__(self, max_stocks: int = 100, max_workers: int = 4):
        """
        初始化测试器
        
        Args:
            max_stocks: 每个指标测试的最大股票数量
            max_workers: 并发测试的最大线程数
        """
        self.max_stocks = max_stocks
        self.max_workers = max_workers
        self.validator = ProductionIndicatorValidator()
        self.test_results: List[TestResult] = []
        self.batch_results: List[BatchResult] = []
        self.stop_on_error = True
        self.performance_threshold = 30.0  # 30秒性能阈值
        
        # 获取所有可用指标
        self.all_indicators = self.validator.list_available_indicators()
        logger.info(f"📊 初始化完成，共有 {len(self.all_indicators)} 个指标待测试")
        
        # 定义指标分批策略
        self.batches = self._define_test_batches()
    
    def _define_test_batches(self) -> Dict[str, List[str]]:
        """定义测试批次，按指标类型和复杂度分组"""
        
        # 基础指标（快速验证）
        basic_indicators = [
            'ma', 'ema', 'wma', 'rsi', 'macd', 'boll', 'kdj', 'vol',
            'momentum', 'mtm', 'roc', 'bias', 'dma'
        ]
        
        # 趋势指标
        trend_indicators = [
            'sar', 'adx', 'aroon', 'dmi', 'trix', 'cci', 'psy',
            'daily_trend_up', 'weekly_trend_up', 'trend_detector', 'trend_duration'
        ]
        
        # 成交量指标
        volume_indicators = [
            'obv', 'mfi', 'pvt', 'vr', 'vosc', 'volume_ratio', 'ad', 'emv', 'chaikin',
            'volume_shrink', 'turnover'
        ]
        
        # 波动率指标
        volatility_indicators = [
            'atr', 'kc', 'vix', 'stock_vix', 'vortex'
        ]
        
        # ZXM专业指标
        zxm_indicators = [
            'daily_macd', 'bs_absorb', 'ma_callback', 'elasticity', 'bounce_detector',
            'amplitude_elasticity', 'rise_elasticity', 'elasticity_score', 'buypoint_score',
            'stock_score', 'market_breadth', 'selection_model', 'diagnostics'
        ]
        
        # 增强指标
        enhanced_indicators = [
            'enhanced_rsi', 'enhanced_cci', 'enhanced_dmi', 'enhanced_macd_trend',
            'enhanced_trix', 'enhanced_kdj_osc', 'enhanced_mfi', 'enhanced_obv',
            'enhanced_stochrsi', 'enhanced_wr', 'enhanced_macd_root'
        ]
        
        # 复合和形态指标
        complex_indicators = [
            'composite', 'unified_ma', 'chip_distribution', 'institutional_behavior',
            'candlestick_patterns', 'advanced_candlestick', 'patterns'
        ]
        
        # 工具和公式指标
        tool_indicators = [
            'fibonacci_tools', 'gann_tools', 'elliott_wave',
            'cross_over', 'kdj_condition', 'macd_condition', 'ma_condition', 'generic_condition'
        ]
        
        # 多周期指标
        multi_period_indicators = [
            'monthly_kdj_trend_up', 'monthly_macd', 'weekly_kdj_d_or_dea_trend_up',
            'weekly_kdj_d_trend_up', 'weekly_macd'
        ]
        
        # 震荡指标
        oscillator_indicators = [
            'wr', 'cmo', 'stochrsi', 'ichimoku'
        ]
        
        batches = {
            '第1批-基础指标': basic_indicators,
            '第2批-趋势指标': trend_indicators,
            '第3批-成交量指标': volume_indicators,
            '第4批-波动率指标': volatility_indicators,
            '第5批-ZXM专业指标': zxm_indicators,
            '第6批-增强指标': enhanced_indicators,
            '第7批-复合形态指标': complex_indicators,
            '第8批-工具公式指标': tool_indicators,
            '第9批-多周期指标': multi_period_indicators,
            '第10批-震荡指标': oscillator_indicators,
        }
        
        # 过滤掉不存在的指标
        filtered_batches = {}
        for batch_name, indicators in batches.items():
            available_indicators = [ind for ind in indicators if ind in self.all_indicators]
            if available_indicators:
                filtered_batches[batch_name] = available_indicators
                logger.info(f"📋 {batch_name}: {len(available_indicators)} 个指标")
        
        return filtered_batches
    
    def test_single_indicator(self, indicator_name: str, test_date: str = None) -> TestResult:
        """测试单个指标"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 开始测试指标: {indicator_name}")
            
            # 执行验证
            result = self.validator.validate_single_indicator(
                indicator_name=indicator_name,
                test_date=test_date,
                max_stocks=self.max_stocks
            )
            
            execution_time = time.time() - start_time
            
            # 检查性能阈值
            if execution_time > self.performance_threshold:
                logger.warning(f"⚠️ 指标 {indicator_name} 执行时间过长: {execution_time:.2f}秒")
            
            # 提取关键指标
            stock_count = result.get('stock_count', 0)
            selected_stocks = result.get('selected_stocks', 0)
            selection_rate = result.get('selection_rate', 0.0)
            
            logger.info(f"✅ {indicator_name}: {selected_stocks}/{stock_count} 只股票被选中 ({selection_rate:.1f}%), 耗时 {execution_time:.2f}秒")
            
            return TestResult(
                indicator_name=indicator_name,
                success=True,
                execution_time=execution_time,
                stock_count=stock_count,
                selected_stocks=selected_stocks,
                performance_metrics={
                    'selection_rate': selection_rate,
                    'avg_time_per_stock': execution_time / max(stock_count, 1)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"❌ 指标 {indicator_name} 测试失败: {error_msg}")
            
            if self.stop_on_error:
                logger.error(f"🛑 遇到错误，停止测试进行修复")
                raise
            
            return TestResult(
                indicator_name=indicator_name,
                success=False,
                execution_time=execution_time,
                stock_count=0,
                selected_stocks=0,
                error_message=error_msg
            )
    
    def test_batch(self, batch_name: str, indicators: List[str], 
                   concurrent: bool = False, test_date: str = None) -> BatchResult:
        """测试一个批次的指标"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 开始测试 {batch_name}")
        logger.info(f"📊 指标数量: {len(indicators)}")
        logger.info(f"📈 股票池大小: {self.max_stocks}")
        logger.info(f"🔄 并发模式: {'是' if concurrent else '否'}")
        logger.info(f"{'='*80}")
        
        batch_start_time = time.time()
        test_results = []
        successful_count = 0
        failed_count = 0
        
        if concurrent and len(indicators) > 1:
            # 并发测试
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_indicator = {
                    executor.submit(self.test_single_indicator, indicator, test_date): indicator 
                    for indicator in indicators
                }
                
                for future in as_completed(future_to_indicator):
                    indicator = future_to_indicator[future]
                    try:
                        result = future.result()
                        test_results.append(result)
                        
                        if result.success:
                            successful_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"❌ 并发测试指标 {indicator} 异常: {e}")
                        if self.stop_on_error:
                            raise
                        failed_count += 1
        else:
            # 串行测试
            for indicator in indicators:
                try:
                    result = self.test_single_indicator(indicator, test_date)
                    test_results.append(result)
                    
                    if result.success:
                        successful_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ 串行测试指标 {indicator} 异常: {e}")
                    if self.stop_on_error:
                        raise
                    failed_count += 1
        
        batch_time = time.time() - batch_start_time
        
        # 创建批次结果
        batch_result = BatchResult(
            batch_name=batch_name,
            total_indicators=len(indicators),
            successful_indicators=successful_count,
            failed_indicators=failed_count,
            total_time=batch_time,
            test_results=test_results
        )
        
        # 显示批次总结
        self._print_batch_summary(batch_result)
        
        return batch_result
    
    def _print_batch_summary(self, batch_result: BatchResult):
        """打印批次测试总结"""
        success_rate = (batch_result.successful_indicators / batch_result.total_indicators * 100) if batch_result.total_indicators > 0 else 0
        
        logger.info(f"\n📊 {batch_result.batch_name} 测试总结:")
        logger.info(f"   总指标数: {batch_result.total_indicators}")
        logger.info(f"   成功测试: {batch_result.successful_indicators}")
        logger.info(f"   失败测试: {batch_result.failed_indicators}")
        logger.info(f"   成功率: {success_rate:.1f}%")
        logger.info(f"   总耗时: {batch_result.total_time:.2f}秒")
        logger.info(f"   平均耗时: {batch_result.total_time/batch_result.total_indicators:.2f}秒/指标")
        
        if batch_result.failed_indicators > 0:
            failed_indicators = [r.indicator_name for r in batch_result.test_results if not r.success]
            logger.warning(f"   失败指标: {', '.join(failed_indicators)}")
    
    def run_comprehensive_test(self, selected_batches: List[str] = None, 
                             concurrent: bool = False, test_date: str = None) -> Dict[str, Any]:
        """运行全面测试"""
        
        logger.info(f"\n{'='*100}")
        logger.info(f"🎯 全面指标测试开始")
        logger.info(f"📊 总指标数: {len(self.all_indicators)}")
        logger.info(f"📈 股票池大小: {self.max_stocks}")
        logger.info(f"🔄 并发模式: {'开启' if concurrent else '关闭'}")
        logger.info(f"🛑 错误停止: {'开启' if self.stop_on_error else '关闭'}")
        logger.info(f"⏱️ 性能阈值: {self.performance_threshold}秒")
        logger.info(f"{'='*100}")
        
        overall_start_time = time.time()
        
        # 确定要测试的批次
        batches_to_test = selected_batches if selected_batches else list(self.batches.keys())
        
        total_indicators_tested = 0
        total_successful = 0
        total_failed = 0
        
        try:
            for batch_name in batches_to_test:
                if batch_name not in self.batches:
                    logger.warning(f"⚠️ 批次 {batch_name} 不存在，跳过")
                    continue
                
                indicators = self.batches[batch_name]
                
                # 执行批次测试
                batch_result = self.test_batch(
                    batch_name=batch_name,
                    indicators=indicators,
                    concurrent=concurrent,
                    test_date=test_date
                )
                
                # 累计统计
                self.batch_results.append(batch_result)
                self.test_results.extend(batch_result.test_results)
                
                total_indicators_tested += batch_result.total_indicators
                total_successful += batch_result.successful_indicators
                total_failed += batch_result.failed_indicators
                
                # 检查是否需要停止
                if batch_result.failed_indicators > 0 and self.stop_on_error:
                    logger.error(f"🛑 批次 {batch_name} 有失败指标，停止测试")
                    break
        
        except Exception as e:
            logger.error(f"❌ 全面测试过程中发生异常: {e}")
            if self.stop_on_error:
                raise
        
        overall_time = time.time() - overall_start_time
        
        # 生成综合报告
        comprehensive_result = {
            'test_summary': {
                'total_indicators': len(self.all_indicators),
                'tested_indicators': total_indicators_tested,
                'successful_indicators': total_successful,
                'failed_indicators': total_failed,
                'success_rate': (total_successful / total_indicators_tested * 100) if total_indicators_tested > 0 else 0,
                'total_time': overall_time,
                'average_time_per_indicator': overall_time / total_indicators_tested if total_indicators_tested > 0 else 0
            },
            'batch_results': [
                {
                    'batch_name': br.batch_name,
                    'total_indicators': br.total_indicators,
                    'successful_indicators': br.successful_indicators,
                    'failed_indicators': br.failed_indicators,
                    'success_rate': (br.successful_indicators / br.total_indicators * 100) if br.total_indicators > 0 else 0,
                    'total_time': br.total_time
                }
                for br in self.batch_results
            ],
            'detailed_results': [
                {
                    'indicator_name': tr.indicator_name,
                    'success': tr.success,
                    'execution_time': tr.execution_time,
                    'stock_count': tr.stock_count,
                    'selected_stocks': tr.selected_stocks,
                    'error_message': tr.error_message,
                    'performance_metrics': tr.performance_metrics
                }
                for tr in self.test_results
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # 打印最终总结
        self._print_comprehensive_summary(comprehensive_result)
        
        return comprehensive_result
    
    def _print_comprehensive_summary(self, result: Dict[str, Any]):
        """打印全面测试总结"""
        summary = result['test_summary']
        
        logger.info(f"\n{'='*100}")
        logger.info(f"🎉 全面指标测试完成")
        logger.info(f"{'='*100}")
        logger.info(f"📊 测试总结:")
        logger.info(f"   总指标数: {summary['total_indicators']}")
        logger.info(f"   已测指标: {summary['tested_indicators']}")
        logger.info(f"   成功指标: {summary['successful_indicators']}")
        logger.info(f"   失败指标: {summary['failed_indicators']}")
        logger.info(f"   成功率: {summary['success_rate']:.1f}%")
        logger.info(f"   总耗时: {summary['total_time']:.2f}秒")
        logger.info(f"   平均耗时: {summary['average_time_per_indicator']:.2f}秒/指标")
        
        # 性能分析
        slow_indicators = [tr for tr in self.test_results if tr.success and tr.execution_time > self.performance_threshold]
        if slow_indicators:
            logger.warning(f"\n⚠️ 性能较慢的指标 (>{self.performance_threshold}秒):")
            for tr in sorted(slow_indicators, key=lambda x: x.execution_time, reverse=True):
                logger.warning(f"   {tr.indicator_name}: {tr.execution_time:.2f}秒")
        
        # 失败指标分析
        failed_indicators = [tr for tr in self.test_results if not tr.success]
        if failed_indicators:
            logger.error(f"\n❌ 失败指标:")
            for tr in failed_indicators:
                logger.error(f"   {tr.indicator_name}: {tr.error_message}")
    
    def save_results(self, result: Dict[str, Any], output_dir: str = "results/comprehensive_test") -> str:
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式结果
        json_file = os.path.join(output_dir, f"comprehensive_test_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存文本格式报告
        txt_file = os.path.join(output_dir, f"comprehensive_test_report_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            self._write_text_report(f, result)
        
        logger.info(f"📄 测试结果已保存:")
        logger.info(f"   JSON: {json_file}")
        logger.info(f"   报告: {txt_file}")
        
        return json_file

    def _write_text_report(self, f, result: Dict[str, Any]):
        """写入文本格式报告"""
        summary = result['test_summary']
        
        f.write("="*100 + "\n")
        f.write("全面指标测试报告\n")
        f.write("="*100 + "\n")
        f.write(f"测试时间: {result['timestamp']}\n")
        f.write(f"总指标数: {summary['total_indicators']}\n")
        f.write(f"已测指标: {summary['tested_indicators']}\n")
        f.write(f"成功指标: {summary['successful_indicators']}\n")
        f.write(f"失败指标: {summary['failed_indicators']}\n")
        f.write(f"成功率: {summary['success_rate']:.1f}%\n")
        f.write(f"总耗时: {summary['total_time']:.2f}秒\n")
        f.write(f"平均耗时: {summary['average_time_per_indicator']:.2f}秒/指标\n")
        f.write("\n")
        
        # 批次结果
        f.write("批次测试结果:\n")
        f.write("-"*80 + "\n")
        for batch in result['batch_results']:
            f.write(f"{batch['batch_name']}: {batch['successful_indicators']}/{batch['total_indicators']} "
                   f"({batch['success_rate']:.1f}%) - {batch['total_time']:.2f}秒\n")
        f.write("\n")
        
        # 详细结果
        f.write("详细测试结果:\n")
        f.write("-"*80 + "\n")
        for tr in result['detailed_results']:
            status = "✅" if tr['success'] else "❌"
            f.write(f"{status} {tr['indicator_name']}: {tr['execution_time']:.2f}秒")
            if tr['success']:
                f.write(f" - {tr['selected_stocks']}/{tr['stock_count']} 只股票")
            else:
                f.write(f" - 错误: {tr['error_message']}")
            f.write("\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全面指标测试器 - 覆盖88个指标的所有形态')
    parser.add_argument('--batches', nargs='+', help='要测试的批次名称')
    parser.add_argument('--list-batches', action='store_true', help='列出所有可用批次')
    parser.add_argument('--max-stocks', type=int, default=100, help='每个指标测试的最大股票数量')
    parser.add_argument('--max-workers', type=int, default=4, help='并发测试的最大线程数')
    parser.add_argument('--concurrent', action='store_true', help='启用并发测试')
    parser.add_argument('--test-date', help='测试日期，格式：YYYY-MM-DD')
    parser.add_argument('--output-dir', default='results/comprehensive_test', help='输出目录')
    parser.add_argument('--continue-on-error', action='store_true', help='遇到错误继续测试')
    parser.add_argument('--performance-threshold', type=float, default=30.0, help='性能阈值（秒）')
    
    args = parser.parse_args()
    
    try:
        # 初始化测试器
        tester = ComprehensiveIndicatorTester(
            max_stocks=args.max_stocks,
            max_workers=args.max_workers
        )
        
        # 设置测试参数
        tester.stop_on_error = not args.continue_on_error
        tester.performance_threshold = args.performance_threshold
        
        if args.list_batches:
            print("可用测试批次:")
            for i, (batch_name, indicators) in enumerate(tester.batches.items(), 1):
                print(f"   {i:2d}. {batch_name}: {len(indicators)} 个指标")
            return 0
        
        # 运行全面测试
        result = tester.run_comprehensive_test(
            selected_batches=args.batches,
            concurrent=args.concurrent,
            test_date=args.test_date
        )
        
        # 保存结果
        output_file = tester.save_results(result, args.output_dir)
        
        # 返回状态码
        summary = result['test_summary']
        if summary['failed_indicators'] == 0:
            logger.info("🎉 所有指标测试成功！")
            return 0
        elif summary['success_rate'] >= 80:
            logger.info("✅ 大部分指标测试成功")
            return 0
        else:
            logger.warning("⚠️ 部分指标测试失败")
            return 1
            
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 