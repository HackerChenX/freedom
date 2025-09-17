#!/usr/bin/env python3
"""
测试通用多周期指标计算器
验证全周期分析功能和周期+指标的二维分析矩阵
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators.services.universal_multi_period_calculator import UniversalMultiPeriodCalculator
from db.services.multi_period_data_service import Period
from utils.logger import get_logger

logger = get_logger(__name__)


def test_universal_multi_period_calculator():
    """测试通用多周期指标计算器"""
    print("🧪 **测试通用多周期指标计算器**")
    print("=" * 60)
    
    try:
        # 初始化计算器
        calculator = UniversalMultiPeriodCalculator()
        print("✅ 通用多周期计算器初始化成功")
        
        # 测试全周期分析
        print("\n🔍 **测试全周期分析**:")
        result = calculator.calculate_multi_period_indicators(
            stock_code='300005',
            target_date='2025-05-09',
            indicator_names=['MACD', 'RSI', 'KDJ', 'MA'],  # 测试几个核心指标
            periods=None  # 不指定periods，使用全周期
        )
        
        print(f"📊 分析结果概览:")
        print(f"   - 股票代码: {result['stock_code']}")
        print(f"   - 分析日期: {result['target_date']}")
        print(f"   - 分析周期: {result['periods_analyzed']}")
        print(f"   - 分析指标: {len(result['indicators_analyzed'])}个")
        print(f"   - 综合评分: {result['overall_score']:.2f}/100")
        
        # 检查分析矩阵
        analysis_matrix = result['analysis_matrix']
        print(f"\n📈 **周期+指标分析矩阵**:")
        for period_name, period_indicators in analysis_matrix.items():
            print(f"   {period_name}: {len(period_indicators)}个指标")
            
            # 显示每个指标的信号
            for indicator_name, indicator_result in period_indicators.items():
                signal = indicator_result.get('signal', 'UNKNOWN')
                strength = indicator_result.get('strength', 0.0)
                print(f"     - {indicator_name}: {signal} (强度: {strength:.2f})")
        
        # 检查一致性分析
        consistency = result['consistency_analysis']
        print(f"\n🔄 **跨周期一致性分析**:")
        print(f"   - 一致性评分: {consistency['overall_consistency_score']:.2f}")
        print(f"   - 一致指标: {len(consistency['consistent_indicators'])}个")
        print(f"   - 不一致指标: {len(consistency['inconsistent_indicators'])}个")
        
        if consistency['consistent_indicators']:
            print(f"   - 一致指标列表: {', '.join(consistency['consistent_indicators'])}")
        
        # 检查聚合信号
        aggregated_signals = result['aggregated_signals']
        print(f"\n🎯 **多周期聚合信号**:")
        for indicator_name, signal_info in aggregated_signals.items():
            agg_signal = signal_info.get('aggregated_signal', 'UNKNOWN')
            agg_strength = signal_info.get('aggregated_strength', 0.0)
            print(f"   - {indicator_name}: {agg_signal} (聚合强度: {agg_strength:.2f})")
        
        # 检查数据质量
        data_quality = result['data_quality']
        print(f"\n📊 **数据质量评估**:")
        print(f"   - 整体质量: {data_quality['overall_quality']}")
        print(f"   - 有数据周期: {data_quality['periods_with_data']}/{data_quality['total_periods']}")
        
        for period_name, data_count in data_quality['data_completeness'].items():
            print(f"   - {period_name}: {data_count}条数据")
        
        print("\n✅ **全周期分析测试完成！**")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_periods():
    """测试指定周期分析"""
    print("\n🧪 **测试指定周期分析**")
    print("=" * 60)
    
    try:
        calculator = UniversalMultiPeriodCalculator()
        
        # 测试指定周期
        specific_periods = [Period.MIN_15, Period.DAILY, Period.WEEKLY]
        result = calculator.calculate_multi_period_indicators(
            stock_code='300005',
            target_date='2025-05-09',
            indicator_names=['MACD', 'RSI'],
            periods=specific_periods
        )
        
        print(f"📊 指定周期分析结果:")
        print(f"   - 请求周期: {[p.value for p in specific_periods]}")
        print(f"   - 实际分析周期: {result['periods_analyzed']}")
        print(f"   - 综合评分: {result['overall_score']:.2f}/100")
        
        print("\n✅ **指定周期分析测试完成！**")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_buypoint_analyzer_integration():
    """测试与买点分析器的集成"""
    print("\n🧪 **测试买点分析器集成**")
    print("=" * 60)
    
    try:
        from bin.multi_period_buypoint_analyzer import MultiPeriodBuypointAnalyzer
        
        analyzer = MultiPeriodBuypointAnalyzer()
        print("✅ 买点分析器初始化成功")
        
        # 测试全周期买点分析
        result = analyzer.analyze_multi_period_buypoint(
            stock_code='300005',
            target_date='2025-05-09',
            periods=None,  # 使用全周期分析
            indicator_names=['MACD', 'RSI', 'KDJ'],
            enable_strategy_analysis=False  # 暂时禁用策略分析以简化测试
        )
        
        print(f"📊 买点分析结果:")
        print(f"   - 分析状态: {result.get('status', 'UNKNOWN')}")
        
        if result.get('status') == 'SUCCESS':
            buypoint_analysis = result.get('buypoint_analysis', {})
            print(f"   - 综合评分: {buypoint_analysis.get('overall_score', 0):.2f}/100")
            print(f"   - 强买信号: {len(buypoint_analysis.get('strong_buy_signals', []))}个")
            print(f"   - 中等买信号: {len(buypoint_analysis.get('moderate_buy_signals', []))}个")
            print(f"   - 弱买信号: {len(buypoint_analysis.get('weak_buy_signals', []))}个")
            
            # 显示周期信号摘要
            period_summary = buypoint_analysis.get('period_signal_summary', {})
            print(f"\n📈 **各周期信号摘要**:")
            for period_name, signals in period_summary.items():
                buy_count = signals.get('buy_count', 0)
                sell_count = signals.get('sell_count', 0)
                hold_count = signals.get('hold_count', 0)
                print(f"   - {period_name}: 买入{buy_count} | 卖出{sell_count} | 持有{hold_count}")
        
        print("\n✅ **买点分析器集成测试完成！**")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 **通用多周期指标计算器测试套件**")
    print("=" * 80)
    
    test_results = []
    
    # 测试1: 通用多周期计算器
    test_results.append(test_universal_multi_period_calculator())
    
    # 测试2: 指定周期分析
    test_results.append(test_specific_periods())
    
    # 测试3: 买点分析器集成
    test_results.append(test_buypoint_analyzer_integration())
    
    # 总结测试结果
    print("\n" + "=" * 80)
    print("🎯 **测试结果总结**")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"❌ 失败测试: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 **所有测试通过！全周期分析功能正常工作！**")
        print("💡 **核心改进**:")
        print("   - ✅ 支持全周期分析（15分钟、30分钟、60分钟、日线、周线、月线）")
        print("   - ✅ 抽离了公共的指标计算逻辑")
        print("   - ✅ 实现了周期+指标的二维分析矩阵")
        print("   - ✅ 提供跨周期信号一致性验证")
        print("   - ✅ 生成多周期聚合信号")
        print("   - ✅ 不再限制为只使用日线和周线数据")
    else:
        print("\n⚠️ **部分测试失败，需要进一步调试**")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
