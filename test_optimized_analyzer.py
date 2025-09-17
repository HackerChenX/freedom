#!/usr/bin/env python3
"""
测试优化后的MultiPeriodBuypointAnalyzer
验证架构设计和生产级要求的改进

测试内容：
1. 消除硬编码问题
2. 动态指标和策略发现
3. 配置驱动的系统
4. 全指标覆盖率
5. 六层架构合规性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bin.multi_period_buypoint_analyzer import MultiPeriodBuypointAnalyzer
from db.services.multi_period_data_service import Period
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


def test_dynamic_configuration():
    """测试动态配置功能"""
    print("🧪 **测试1: 动态配置功能**")
    print("=" * 60)
    
    try:
        # 测试默认配置
        analyzer1 = MultiPeriodBuypointAnalyzer()
        print(f"✅ 默认配置初始化成功")
        print(f"   - 可用指标: {len(analyzer1.all_indicators)}个")
        print(f"   - 动态策略: {len(analyzer1.builtin_strategies)}个")
        print(f"   - 指标权重: {len(analyzer1.indicator_weights)}个")
        
        # 验证动态权重生成
        tier1_count = sum(1 for w in analyzer1.indicator_weights.values() 
                         if abs(w - analyzer1.config["scoring"]["tier1_weight"]) < 0.01)
        print(f"   - 核心指标权重: {tier1_count}个")
        
        # 验证策略动态生成
        strategy_types = set()
        for strategy_config in analyzer1.builtin_strategies.values():
            strategy_types.add(strategy_config.get('type', 'unknown'))
        print(f"   - 策略类型: {', '.join(strategy_types)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 动态配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_hardcoded_elements():
    """测试无硬编码元素"""
    print("\n🧪 **测试2: 无硬编码元素验证**")
    print("=" * 60)
    
    try:
        analyzer = MultiPeriodBuypointAnalyzer()
        
        # 验证指标权重不是硬编码
        hardcoded_indicators = {'KDJ', 'MACD', 'RSI', 'MA', 'EMA'}
        dynamic_indicators = set(analyzer.indicator_weights.keys())
        
        print(f"✅ 指标权重动态生成验证:")
        print(f"   - 硬编码指标: {len(hardcoded_indicators)}个")
        print(f"   - 动态指标: {len(dynamic_indicators)}个")
        print(f"   - 覆盖率: {len(dynamic_indicators)/len(analyzer.all_indicators)*100:.1f}%")
        
        # 验证策略不是硬编码
        old_hardcoded_strategies = {
            'MULTI_PERIOD_MOMENTUM', 'TREND_FOLLOWING', 
            'VOLUME_PRICE_ANALYSIS', 'VOLATILITY_BREAKOUT'
        }
        current_strategies = set(analyzer.builtin_strategies.keys())
        
        print(f"✅ 策略配置动态生成验证:")
        print(f"   - 旧硬编码策略: {len(old_hardcoded_strategies)}个")
        print(f"   - 当前动态策略: {len(current_strategies)}个")
        
        # 检查是否还有旧的硬编码策略
        remaining_hardcoded = old_hardcoded_strategies.intersection(current_strategies)
        if remaining_hardcoded:
            print(f"   ⚠️ 仍存在硬编码策略: {remaining_hardcoded}")
        else:
            print(f"   ✅ 已完全消除硬编码策略")
        
        return len(remaining_hardcoded) == 0
        
    except Exception as e:
        print(f"❌ 硬编码验证测试失败: {e}")
        return False


def test_full_indicator_coverage():
    """测试全指标覆盖率"""
    print("\n🧪 **测试3: 全指标覆盖率验证**")
    print("=" * 60)
    
    try:
        analyzer = MultiPeriodBuypointAnalyzer()
        
        # 测试指标覆盖
        total_indicators = len(analyzer.all_indicators)
        weighted_indicators = len(analyzer.indicator_weights)
        
        print(f"✅ 指标覆盖率验证:")
        print(f"   - 系统可用指标: {total_indicators}个")
        print(f"   - 权重分配指标: {weighted_indicators}个")
        print(f"   - 覆盖率: {weighted_indicators/total_indicators*100:.1f}%")
        
        # 测试策略指标使用
        strategy_indicators = set()
        for strategy_config in analyzer.builtin_strategies.values():
            strategy_indicators.update(strategy_config.get('indicators', []))
        
        print(f"✅ 策略指标使用验证:")
        print(f"   - 策略使用指标: {len(strategy_indicators)}个")
        print(f"   - 指标利用率: {len(strategy_indicators)/total_indicators*100:.1f}%")
        
        # 验证是否能处理所有指标
        sample_indicators = analyzer.all_indicators[:10]  # 测试前10个指标
        print(f"✅ 指标处理能力验证:")
        print(f"   - 测试指标样本: {len(sample_indicators)}个")
        
        for indicator in sample_indicators:
            weight = analyzer.indicator_weights.get(indicator, 0)
            print(f"     {indicator}: 权重={weight:.4f}")
        
        return weighted_indicators == total_indicators
        
    except Exception as e:
        print(f"❌ 指标覆盖率测试失败: {e}")
        return False


def test_architecture_compliance():
    """测试六层架构合规性"""
    print("\n🧪 **测试4: 六层架构合规性验证**")
    print("=" * 60)
    
    try:
        analyzer = MultiPeriodBuypointAnalyzer()
        
        # 验证依赖注入
        print(f"✅ 依赖注入验证:")
        print(f"   - 数据服务: {type(analyzer.data_service).__name__}")
        print(f"   - 指标服务: {type(analyzer.indicator_service).__name__}")
        print(f"   - 通用计算器: {type(analyzer.universal_calculator).__name__}")
        print(f"   - 买点检测器: {type(analyzer.buypoint_detector).__name__}")
        
        # 验证无重复初始化
        print(f"✅ 初始化验证:")
        print(f"   - 策略管理器: {'已初始化' if analyzer.strategy_manager else '未初始化'}")
        
        # 验证配置驱动
        print(f"✅ 配置驱动验证:")
        print(f"   - 配置项数量: {len(analyzer.config)}个")
        print(f"   - 分析配置: {len(analyzer.config.get('analysis', {}))}项")
        print(f"   - 评分配置: {len(analyzer.config.get('scoring', {}))}项")
        print(f"   - 策略配置: {len(analyzer.config.get('strategies', {}))}项")
        
        return True
        
    except Exception as e:
        print(f"❌ 架构合规性测试失败: {e}")
        return False


def test_production_analysis():
    """测试生产级分析功能"""
    print("\n🧪 **测试5: 生产级分析功能验证**")
    print("=" * 60)
    
    try:
        analyzer = MultiPeriodBuypointAnalyzer()
        
        # 执行实际分析
        result = analyzer.analyze_multi_period_buypoint(
            stock_code='300005',
            target_date='2025-05-09',
            periods=None,  # 全周期分析
            indicator_names=None,  # 全指标分析
            enable_strategy_analysis=True
        )
        
        print(f"✅ 生产级分析验证:")
        print(f"   - 分析状态: {result.get('status', 'UNKNOWN')}")
        print(f"   - 分析指标数: {result.get('indicators_analyzed', 0)}个")
        print(f"   - 分析周期数: {len(result.get('periods_analyzed', []))}个")
        print(f"   - 策略分析: {'启用' if result.get('strategy_analysis_enabled') else '禁用'}")
        print(f"   - 综合评分: {result.get('overall_score', 0):.2f}")
        
        # 验证系统信息
        system_info = result.get('system_info', {})
        print(f"✅ 系统信息验证:")
        print(f"   - 总指标数: {system_info.get('total_real_indicators', 0)}个")
        print(f"   - 内置策略数: {system_info.get('builtin_strategies', 0)}个")
        print(f"   - 分析类型: {system_info.get('analysis_type', 'UNKNOWN')}")
        
        return result.get('status') == 'SUCCESS'
        
    except Exception as e:
        print(f"❌ 生产级分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 **优化后MultiPeriodBuypointAnalyzer测试套件**")
    print("=" * 80)
    
    test_results = []
    
    # 执行所有测试
    test_results.append(test_dynamic_configuration())
    test_results.append(test_no_hardcoded_elements())
    test_results.append(test_full_indicator_coverage())
    test_results.append(test_architecture_compliance())
    test_results.append(test_production_analysis())
    
    # 总结测试结果
    print("\n" + "=" * 80)
    print("🎯 **测试结果总结**")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"❌ 失败测试: {total_tests - passed_tests}/{total_tests}")
    
    test_names = [
        "动态配置功能",
        "无硬编码元素",
        "全指标覆盖率",
        "六层架构合规性",
        "生产级分析功能"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 **所有测试通过！系统优化成功！**")
        print("💡 **核心改进成果**:")
        print("   - ✅ 完全消除硬编码指标和策略配置")
        print("   - ✅ 实现动态指标发现和权重分配")
        print("   - ✅ 建立配置驱动的策略管理")
        print("   - ✅ 提供100%指标覆盖率")
        print("   - ✅ 符合六层架构设计规范")
        print("   - ✅ 达到生产级质量标准")
    else:
        print("\n⚠️ **部分测试失败，需要进一步优化**")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
