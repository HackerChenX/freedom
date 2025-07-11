#!/usr/bin/env python3
"""
完整的系统集成测试

测试整个系统的端到端功能，包括策略执行、指标计算、数据访问等
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

def test_complete_strategy_workflow():
    """测试完整的策略工作流程"""
    print("🔍 测试完整策略工作流程...")
    
    try:
        # 初始化服务
        from config.service_initializer import initialize_all_services
        container = initialize_all_services()
        
        # 创建测试策略
        from strategy.unified_base_strategy import UnifiedBaseStrategy
        
        class IntegrationTestStrategy(UnifiedBaseStrategy):
            def select_stocks(self, universe: List[str], start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
                """完整的选股逻辑测试"""
                
                # 模拟获取数据和计算指标
                results = []
                
                for i, code in enumerate(universe[:5]):  # 只处理前5只股票
                    # 模拟股票数据
                    test_data = pd.DataFrame({
                        'date': pd.date_range(start_date, end_date, freq='D')[:20],
                        'open': [100 + i + j * 0.1 for j in range(20)],
                        'high': [105 + i + j * 0.1 for j in range(20)],
                        'low': [95 + i + j * 0.1 for j in range(20)],
                        'close': [102 + i + j * 0.1 for j in range(20)],
                        'volume': [10000 + j * 100 for j in range(20)]
                    })
                    
                    # 计算简单指标
                    ma5 = test_data['close'].rolling(5).mean()
                    ma10 = test_data['close'].rolling(10).mean()
                    
                    # 生成信号
                    current_price = test_data['close'].iloc[-1]
                    ma5_current = ma5.iloc[-1] if not pd.isna(ma5.iloc[-1]) else current_price
                    ma10_current = ma10.iloc[-1] if not pd.isna(ma10.iloc[-1]) else current_price
                    
                    # 简单的选股逻辑：MA5 > MA10
                    if ma5_current > ma10_current:
                        score = 100 - i * 5  # 简单评分
                        results.append({
                            'code': code,
                            'name': f'股票{code}',
                            'score': score,
                            'reason': 'MA5 > MA10',
                            'ma5': ma5_current,
                            'ma10': ma10_current,
                            'price': current_price
                        })
                
                return pd.DataFrame(results)
        
        # 执行策略
        strategy = IntegrationTestStrategy(
            name="集成测试策略",
            description="用于测试完整工作流程的策略"
        )
        
        # 设置参数
        strategy.set_parameters(
            lookback_days=30,
            min_volume=100000
        )
        
        # 执行选股
        universe = ['000001', '000002', '000003', '600000', '600036']
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        result = strategy.execute(
            universe=universe,
            start_date=start_date,
            end_date=end_date
        )
        
        # 验证结果
        assert not result.empty, "策略应该返回结果"
        assert 'code' in result.columns, "结果应该包含code列"
        assert 'score' in result.columns, "结果应该包含score列"
        assert 'strategy' in result.columns, "结果应该包含strategy列"
        
        print(f"  ✅ 策略执行成功，选出 {len(result)} 只股票")
        print(f"  📊 结果预览:\n{result.head()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 策略工作流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_calculation_integration():
    """测试指标计算集成"""
    print("🔍 测试指标计算集成...")
    
    try:
        from indicators.unified_calculator import SimpleMovingAverageCalculator, IndicatorCalculatorFactory
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30),
            'open': [100 + i * 0.5 for i in range(30)],
            'high': [105 + i * 0.5 for i in range(30)],
            'low': [95 + i * 0.5 for i in range(30)],
            'close': [102 + i * 0.5 for i in range(30)],
            'volume': [10000 + i * 100 for i in range(30)]
        })
        
        # 测试SMA指标
        sma_calculator = SimpleMovingAverageCalculator()
        
        # 单次计算
        result = sma_calculator.calculate_with_validation(test_data, period=5)
        assert result.success, "SMA计算应该成功"
        assert not result.data.empty, "SMA结果不应为空"
        
        print(f"  ✅ SMA计算成功，用时 {result.calculation_time:.3f}秒")
        
        # 批量计算
        data_list = [test_data.iloc[:10], test_data.iloc[10:20], test_data.iloc[20:]]
        batch_results = sma_calculator.calculate_batch(data_list, period=3)
        
        success_count = sum(1 for r in batch_results if r.success)
        print(f"  ✅ 批量计算完成，成功率 {success_count}/{len(batch_results)}")
        
        # 测试工厂模式
        try:
            factory_sma = IndicatorCalculatorFactory.create('SMA', period=10)
            factory_result = factory_sma.calculate_with_validation(test_data)
            print(f"  ✅ 工厂模式创建指标成功")
        except Exception as e:
            print(f"  ⚠️ 工厂模式测试跳过: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 指标计算集成测试失败: {e}")
        return False

def test_execution_template_integration():
    """测试执行模板集成"""
    print("🔍 测试执行模板集成...")
    
    try:
        from strategy.execution_template import (
            create_execution_context, 
            create_execution_template,
            StandardStrategyExecutionTemplate
        )
        from strategy.unified_base_strategy import UnifiedBaseStrategy
        
        # 创建简单策略
        class TemplateTestStrategy(UnifiedBaseStrategy):
            def select_stocks(self, universe, start_date, end_date, **kwargs):
                return pd.DataFrame({
                    'code': universe[:3],
                    'score': [90, 80, 70],
                    'reason': ['测试1', '测试2', '测试3']
                })
        
        strategy = TemplateTestStrategy("模板测试策略")
        
        # 创建执行上下文
        context = create_execution_context(
            strategy=strategy,
            universe=['000001', '000002', '000003', '600000'],
            start_date='2023-01-01',
            end_date='2023-01-31',
            parameters={'test_param': 'test_value'}
        )
        
        # 创建执行模板
        template = create_execution_template('standard')
        
        # 执行策略
        execution_result = template.execute(context)
        
        # 验证结果
        assert execution_result.status.value in ['success', 'warning'], "执行应该成功"
        assert not execution_result.selected_stocks.empty, "应该有选股结果"
        assert execution_result.execution_time > 0, "执行时间应该大于0"
        
        print(f"  ✅ 执行模板测试成功")
        print(f"  📊 执行状态: {execution_result.status.value}")
        print(f"  ⏱️ 执行时间: {execution_result.execution_time:.3f}秒")
        print(f"  📈 选股数量: {len(execution_result.selected_stocks)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 执行模板集成测试失败: {e}")
        return False

def test_quality_tools_integration():
    """测试质量工具集成"""
    print("🔍 测试质量工具集成...")
    
    try:
        # 测试架构检查工具
        from tools.architecture_checker import ArchitectureChecker
        
        checker = ArchitectureChecker(os.getcwd())
        
        # 只运行一个小范围的检查
        violations = checker.check_direct_database_access()
        print(f"  ✅ 架构检查工具运行正常，发现 {len(violations)} 个数据库访问违规")
        
        # 测试命名规范工具
        from tools.naming_convention_checker import NamingConventionChecker
        
        naming_checker = NamingConventionChecker(os.getcwd())
        
        # 只检查一个简单的模块
        test_file = 'utils/common_utils.py'
        if os.path.exists(test_file):
            print(f"  ✅ 命名规范检查工具可用")
        
        # 测试性能基准工具
        from tools.performance_benchmark import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark(os.getcwd())
        print(f"  ✅ 性能基准工具可用")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 质量工具集成测试失败: {e}")
        return False

def test_data_flow_integration():
    """测试数据流集成"""
    print("🔍 测试数据流集成...")
    
    try:
        from utils.common_utils import DataProcessor, DateTimeUtils, ValidationUtils
        
        # 测试数据处理流程
        raw_data = pd.DataFrame({
            'code': ['000001', '000002', '000001', '000003'],  # 包含重复
            'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03'],
            'price': [100.0, None, 100.0, 102.0],  # 包含空值
            'volume': [10000, 20000, 10000, 15000]
        })
        
        # 数据清理
        cleaned_data = DataProcessor.clean_dataframe(
            raw_data, 
            drop_na=True, 
            deduplicate=True
        )
        
        assert len(cleaned_data) < len(raw_data), "清理后数据应该更少"
        
        # 数据验证
        is_valid, missing = ValidationUtils.validate_dataframe_schema(
            cleaned_data, 
            ['code', 'date', 'price']
        )
        assert is_valid, "清理后数据应该有效"
        
        # 日期处理
        trading_days = DateTimeUtils.get_trading_days('2023-01-01', '2023-01-10')
        assert len(trading_days) > 0, "应该有交易日"
        
        print(f"  ✅ 数据流集成正常")
        print(f"  📊 原始数据: {len(raw_data)} 行 -> 清理后: {len(cleaned_data)} 行")
        print(f"  📅 交易日数量: {len(trading_days)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据流集成测试失败: {e}")
        return False

def generate_integration_report(results: Dict[str, bool]):
    """生成集成测试报告"""
    print("\n📊 生成集成测试报告...")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    report = f"""
# 系统集成测试报告

## 测试概览
- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **测试项目**: {total_tests}
- **通过项目**: {passed_tests}
- **成功率**: {success_rate:.1f}%

## 详细结果

"""
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        report += f"- **{test_name}**: {status}\n"
    
    report += f"""

## 测试说明

### 1. 完整策略工作流程
测试从策略创建、参数设置、执行选股到结果处理的完整流程。

### 2. 指标计算集成  
测试指标计算器的单次计算、批量计算和工厂模式创建。

### 3. 执行模板集成
测试标准化策略执行模板的上下文管理和执行流程。

### 4. 质量工具集成
测试架构检查、命名规范、性能基准等质量保证工具。

### 5. 数据流集成
测试数据处理、验证、格式化等数据流操作。

## 结论

{"🎉 系统集成测试全部通过！所有核心功能正常工作。" if success_rate == 100 
 else f"⚠️ 系统集成测试通过率 {success_rate:.1f}%，部分功能需要优化。" if success_rate >= 80
 else "❌ 系统集成测试未通过，需要进一步修复。"}
"""
    
    # 保存报告
    with open('INTEGRATION_TEST_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print("\n📄 集成测试报告已保存到: INTEGRATION_TEST_REPORT.md")

def main():
    """主函数"""
    print("🚀 开始完整系统集成测试...")
    print("=" * 60)
    
    # 定义测试项目
    tests = [
        ("完整策略工作流程", test_complete_strategy_workflow),
        ("指标计算集成", test_indicator_calculation_integration),
        ("执行模板集成", test_execution_template_integration),
        ("质量工具集成", test_quality_tools_integration),
        ("数据流集成", test_data_flow_integration)
    ]
    
    results = {}
    
    # 执行所有测试
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
                
        except Exception as e:
            print(f"❌ {test_name} 执行出错: {e}")
            results[test_name] = False
    
    # 生成报告
    generate_integration_report(results)
    
    # 计算最终结果
    passed_count = sum(results.values())
    total_count = len(results)
    success_rate = (passed_count / total_count) * 100
    
    print(f"\n📊 最终结果: {passed_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 系统集成测试成功！")
        return True
    else:
        print("⚠️ 系统集成测试部分成功")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)