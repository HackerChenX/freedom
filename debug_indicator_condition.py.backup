#!/usr/bin/env python3
"""
指标条件评估调试脚本
"""
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from db.unified_data_manager import Unified_data_manager
from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
from indicators.complete_indicator_registry import Complete_indicator_registry
from utils.logger import get_logger

logger = get_logger(__name__)

def debug_single_stock_condition():
    """调试单个股票的条件评估"""
    try:
        # 初始化组件
        data_manager = Unified_data_manager()
        indicator_registry = Complete_indicator_registry()
        evaluator = StrategyConditionEvaluator(data_manager, indicator_registry)
        
        # 测试股票
        test_stock = "000001"
        
        print(f"=== 调试股票 {test_stock} ===")
        
        # 1. 获取股票数据
        print("1. 获取股票数据...")
        stock_data = data_manager.get_stock_info(test_stock)
        if stock_data is None:
            print(f"❌ 无法获取股票 {test_stock} 的数据")
            return
        
        print(f"✅ 获取到股票数据，类型: {type(stock_data)}")
        
        # 转换为DataFrame
        if hasattr(stock_data, 'to_dataframe'):
            df = stock_data.to_dataframe()
            print(f"✅ 转换为DataFrame，形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print(f"最新5行数据:")
            print(df.tail())
        else:
            print(f"❌ 股票数据对象没有to_dataframe方法")
            return
        
        # 2. 测试简单价格条件
        print("\n2. 测试简单价格条件...")
        price_condition = {
            "type": "price",
            "field": "close",
            "operator": ">",
            "value": 0,
            "description": "收盘价大于0"
        }
        
        try:
            result = evaluator.evaluate_condition(price_condition, test_stock)
            print(f"✅ 价格条件评估结果: {result}")
        except Exception as e:
            print(f"❌ 价格条件评估失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. 测试MA指标条件
        print("\n3. 测试MA指标条件...")
        ma_condition = {
            "type": "indicator",
            "indicator_id": "MA",
            "period": "daily",
            "condition": "close > 0",
            "description": "股价大于0(基本有效性)"
        }
        
        try:
            result = evaluator.evaluate_condition(ma_condition, test_stock)
            print(f"✅ MA指标条件评估结果: {result}")
        except Exception as e:
            print(f"❌ MA指标条件评估失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. 测试RSI指标条件
        print("\n4. 测试RSI指标条件...")
        rsi_condition = {
            "type": "indicator",
            "indicator_id": "RSI",
            "period": "daily",
            "condition": "RSI < 70",
            "description": "RSI指标小于70"
        }
        
        try:
            result = evaluator.evaluate_condition(rsi_condition, test_stock)
            print(f"✅ RSI指标条件评估结果: {result}")
        except Exception as e:
            print(f"❌ RSI指标条件评估失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. 尝试直接计算指标
        print("\n5. 尝试直接计算RSI指标...")
        try:
            rsi_indicator = indicator_registry.get_indicator("RSI")
            if rsi_indicator:
                print(f"✅ 获取到RSI指标类: {rsi_indicator}")
                
                # 尝试计算指标
                rsi_result = rsi_indicator.calculate(df)
                print(f"✅ RSI计算结果类型: {type(rsi_result)}")
                if hasattr(rsi_result, 'shape'):
                    print(f"RSI结果形状: {rsi_result.shape}")
                    print(f"最新5个RSI值:")
                    print(rsi_result.tail())
                else:
                    print(f"RSI结果: {rsi_result}")
            else:
                print(f"❌ 无法获取RSI指标类")
        except Exception as e:
            print(f"❌ 直接计算RSI指标失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_stock_condition() 