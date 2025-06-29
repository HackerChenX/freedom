#!/usr/bin/env python3
"""
分析引擎对比测试

比较买点分析引擎和策略选股引擎的差异，找出问题根源
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger

logger = get_logger(__name__)


def test_buypoint_analysis_engine():
    """测试买点分析引擎"""
    print("=" * 60)
    print("🔍 测试买点分析引擎")
    print("=" * 60)
    
    try:
        # 初始化买点分析器
        analyzer = BuyPointAnalyzer()
        
        # 测试股票和日期
        test_stock = "000006"
        test_date = "20241228"
        
        print(f"分析股票: {test_stock}")
        print(f"分析日期: {test_date}")
        
        # 执行买点分析
        result = analyzer.analyze_stock(test_stock, test_date)
        
        if result:
            print("✅ 买点分析成功")
            print(f"收盘价: {result.get('close', 'N/A')}")
            print(f"MA5: {result.get('ma5', 'N/A')}")
            print(f"MA10: {result.get('ma10', 'N/A')}")
            print(f"MA20: {result.get('ma20', 'N/A')}")
            print(f"MACD: {result.get('macd', 'N/A')}")
            print(f"DIF: {result.get('dif', 'N/A')}")
            print(f"DEA: {result.get('dea', 'N/A')}")
            print(f"KDJ_K: {result.get('kdj_k', 'N/A')}")
            print(f"KDJ_D: {result.get('kdj_d', 'N/A')}")
            print(f"KDJ_J: {result.get('kdj_j', 'N/A')}")
            print(f"成交量: {result.get('vol', 'N/A')}")
            print(f"触及均线: {result.get('touch_ma', False)}")
            print(f"价格企稳: {result.get('price_stable', False)}")
            print(f"均线上移: {result.get('ma_up', False)}")
            print(f"资金流入: {result.get('money_in', False)}")
            
            return result
        else:
            print("❌ 买点分析失败")
            return None
            
    except Exception as e:
        print(f"❌ 买点分析引擎测试失败: {e}")
        return None


def test_strategy_engine():
    """测试策略选股引擎"""
    print("\n" + "=" * 60)
    print("🎯 测试策略选股引擎")
    print("=" * 60)
    
    try:
        # 初始化策略条件评估器和数据管理器
        evaluator = StrategyConditionEvaluator()
        data_manager = get_unified_data_manager()
        
        # 测试股票和日期
        test_stock = "000006"
        test_date = "2024-12-28"
        
        print(f"分析股票: {test_stock}")
        print(f"分析日期: {test_date}")
        
        # 获取股票数据
        stock_data = data_manager.get_stock_data(
            stock_code=test_stock,
            start_date="2024-12-01",
            end_date="2024-12-30",
            period="daily"
        )
        
        if stock_data is None or stock_data.empty:
            print("❌ 无法获取股票数据")
            return None
        
        print(f"✅ 获取到 {len(stock_data)} 条数据记录")
        print(f"数据日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
        
        # 测试基础条件
        basic_condition = {
            "type": "basic",
            "field": "close",
            "operator": ">",
            "value": 0
        }
        
        basic_result = evaluator.evaluate_condition(basic_condition, stock_data, test_date)
        print(f"基础条件 (收盘价 > 0): {basic_result}")
        
        # 测试指标条件 - MA
        ma_condition = {
            "type": "indicator",
            "indicator_id": "MA",
            "period": 5,
            "field": "MA5",
            "operator": ">",
            "reference_field": "close"
        }
        
        ma_result = evaluator.evaluate_condition(ma_condition, stock_data, test_date)
        print(f"MA指标条件 (MA5 > close): {ma_result}")
        
        # 测试指标条件 - RSI
        rsi_condition = {
            "type": "indicator", 
            "indicator_id": "RSI",
            "period": 14,
            "field": "RSI",
            "operator": "<",
            "value": 40
        }
        
        rsi_result = evaluator.evaluate_condition(rsi_condition, stock_data, test_date)
        print(f"RSI指标条件 (RSI < 40): {rsi_result}")
        
        return {
            "stock_data": stock_data,
            "basic_result": basic_result,
            "ma_result": ma_result,
            "rsi_result": rsi_result
        }
        
    except Exception as e:
        print(f"❌ 策略选股引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_indicator_calculations():
    """比较两个引擎的指标计算差异"""
    print("\n" + "=" * 60)
    print("⚖️  对比指标计算差异")
    print("=" * 60)
    
    try:
        # 获取买点分析结果
        buypoint_result = test_buypoint_analysis_engine()
        
        # 获取策略引擎结果
        strategy_result = test_strategy_engine()
        
        if buypoint_result and strategy_result:
            print("\n📊 指标对比结果:")
            print("-" * 40)
            
            # 对比收盘价
            if 'close' in buypoint_result:
                buypoint_close = buypoint_result['close']
                
                # 从策略引擎数据中获取收盘价
                stock_data = strategy_result['stock_data']
                strategy_close = None
                
                # 查找对应日期的收盘价
                target_date_data = stock_data[stock_data['date'] == '2024-12-28']
                if not target_date_data.empty:
                    strategy_close = target_date_data['close'].iloc[0]
                
                print(f"收盘价对比:")
                print(f"  买点分析: {buypoint_close}")
                print(f"  策略引擎: {strategy_close}")
                print(f"  差异: {abs(buypoint_close - strategy_close) if strategy_close else 'N/A'}")
            
            # 对比MA指标
            if 'ma5' in buypoint_result:
                print(f"\nMA5对比:")
                print(f"  买点分析: {buypoint_result['ma5']}")
                print(f"  策略引擎: 需要通过指标注册表计算")
            
            # 分析差异原因
            print("\n🔍 差异分析:")
            print("1. 数据源: 两个引擎可能使用不同的数据查询方式")
            print("2. 计算方法: 买点分析使用底层函数，策略引擎使用注册表")
            print("3. 日期处理: 买点分析使用索引，策略引擎使用日期查找")
            print("4. 精度差异: 不同的计算路径可能导致精度差异")
            
        else:
            print("❌ 无法进行对比，部分引擎测试失败")
            
    except Exception as e:
        print(f"❌ 指标对比失败: {e}")


def analyze_root_cause():
    """分析问题根本原因"""
    print("\n" + "=" * 60)
    print("🔬 问题根本原因分析")
    print("=" * 60)
    
    print("基于测试结果，问题的根本原因包括:")
    print()
    print("1. 🎯 **目标差异**")
    print("   - 买点分析: 深度分析特定股票的特定时点")
    print("   - 策略选股: 批量筛选大量股票")
    print()
    print("2. 🔧 **计算引擎差异**")
    print("   - 买点分析: 使用底层技术指标函数 (MA, MACD, KDJ等)")
    print("   - 策略选股: 使用指标注册表的标准化接口")
    print()
    print("3. 📊 **数据处理方式差异**")
    print("   - 买点分析: 基于数组索引的高效访问")
    print("   - 策略选股: 基于日期查找的灵活访问")
    print()
    print("4. 🧮 **逻辑复杂度差异**")
    print("   - 买点分析: 支持复杂的复合条件和形态识别")
    print("   - 策略选股: 主要支持简单的比较条件")
    print()
    print("5. 🎨 **设计哲学差异**")
    print("   - 买点分析: 为准确性而设计")
    print("   - 策略选股: 为可配置性和扩展性而设计")
    print()
    print("💡 **解决方案建议**:")
    print("1. 统一计算引擎: 让策略选股也使用买点分析的底层函数")
    print("2. 增强条件支持: 在策略选股中支持更复杂的条件类型")
    print("3. 优化数据访问: 改进策略选股的数据访问效率")
    print("4. 建立验证机制: 确保两个引擎的结果一致性")


def main():
    """主函数"""
    print("🚀 启动分析引擎对比测试")
    
    # 执行对比测试
    compare_indicator_calculations()
    
    # 分析根本原因
    analyze_root_cause()
    
    print("\n" + "=" * 60)
    print("✅ 分析引擎对比测试完成")
    print("=" * 60)


if __name__ == '__main__':
    main() 