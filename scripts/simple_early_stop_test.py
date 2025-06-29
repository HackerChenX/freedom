#!/usr/bin/env python3
"""
简单的早停测试脚本

用于演示指标验证的早停功能：
1. 成功选股后立即停止
2. 遇到错误后立即停止
"""

import os
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import (
    IndicatorValidationFramework, 
    IndicatorValidationConfig, 
    ValidationMode
)
from utils.logger import get_logger

logger = get_logger(__name__)


def test_early_stop_success():
    """测试成功后早停功能"""
    print("🧪 测试1: 成功后早停功能")
    print("=" * 50)
    
    # 创建配置，使用更宽松的条件确保能选出股票
    config = IndicatorValidationConfig(
        mode=ValidationMode.QUICK,
        stock_pool_size=50,          # 更小的股票池
        max_selection_ratio=0.5,     # 更宽松的选股比例
        min_selection_count=1,       # 最少1只股票
        parallel_workers=1,          # 单线程，便于观察
        stop_on_success=True,        # 成功后停止
        stop_on_error=False,         # 不在错误时停止
        debug_mode=True,             # 调试模式
        timeout_seconds=60           # 较短的超时时间
    )
    
    framework = IndicatorValidationFramework(config)
    
    # 手动指定几个常见指标
    test_indicators = ['MA', 'EMA', 'RSI']
    
    print(f"📋 测试配置:")
    print(f"  • 测试指标: {test_indicators}")
    print(f"  • 股票池大小: {config.stock_pool_size}")
    print(f"  • 成功后停止: {config.stop_on_success}")
    print(f"  • 错误后停止: {config.stop_on_error}")
    print(f"  • 调试模式: {config.debug_mode}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # 手动验证指标，观察早停行为
        results = []
        stock_pool = framework._prepare_stock_pool()
        
        for i, indicator in enumerate(test_indicators, 1):
            print(f"🔍 验证进度: {i}/{len(test_indicators)} - {indicator}")
            
            try:
                result = framework._validate_indicator(indicator, stock_pool)
                results.append(result)
                
                print(f"  ✅ 指标 {indicator} 验证结果: {result['status']}")
                print(f"  📊 选股数量: {result.get('selected_count', 0)}")
                
                # 检查是否成功选股
                if result['status'] == 'success' and config.stop_on_success:
                    print(f"  🛑 检测到成功选股，触发早停功能！")
                    print(f"  🎯 选中的股票: {result.get('selected_stocks', [])[:5]}...")  # 只显示前5个
                    break
                    
            except Exception as e:
                print(f"  ❌ 指标 {indicator} 验证出错: {str(e)}")
                if config.stop_on_error:
                    print(f"  🛑 检测到验证错误，触发早停功能！")
                    break
                    
        end_time = time.time()
        print(f"\n⏱️ 测试完成，耗时: {end_time - start_time:.2f}秒")
        print(f"📈 验证结果: {len(results)} 个指标完成验证")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断测试")
        return []
    except Exception as e:
        print(f"\n❌ 测试过程出错: {str(e)}")
        return []


def test_early_stop_error():
    """测试错误后早停功能"""
    print("\n🧪 测试2: 错误后早停功能")
    print("=" * 50)
    
    # 创建配置，故意使用会出错的设置
    config = IndicatorValidationConfig(
        mode=ValidationMode.QUICK,
        stock_pool_size=10,          # 很小的股票池
        max_selection_ratio=0.01,    # 很严格的选股比例
        min_selection_count=1,
        parallel_workers=1,          # 单线程
        stop_on_success=False,       # 不在成功时停止
        stop_on_error=True,          # 错误后停止
        debug_mode=True,
        timeout_seconds=30           # 很短的超时时间
    )
    
    framework = IndicatorValidationFramework(config)
    
    # 包含一个可能出错的指标
    test_indicators = ['MA', 'INVALID_INDICATOR', 'RSI']
    
    print(f"📋 测试配置:")
    print(f"  • 测试指标: {test_indicators}")
    print(f"  • 股票池大小: {config.stock_pool_size}")
    print(f"  • 成功后停止: {config.stop_on_success}")
    print(f"  • 错误后停止: {config.stop_on_error}")
    print(f"  • 调试模式: {config.debug_mode}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        results = []
        stock_pool = framework._prepare_stock_pool()
        
        for i, indicator in enumerate(test_indicators, 1):
            print(f"🔍 验证进度: {i}/{len(test_indicators)} - {indicator}")
            
            try:
                result = framework._validate_indicator(indicator, stock_pool)
                results.append(result)
                
                print(f"  ✅ 指标 {indicator} 验证结果: {result['status']}")
                print(f"  📊 选股数量: {result.get('selected_count', 0)}")
                
            except Exception as e:
                print(f"  ❌ 指标 {indicator} 验证出错: {str(e)}")
                if config.stop_on_error:
                    print(f"  🛑 检测到验证错误，触发早停功能！")
                    break
                    
        end_time = time.time()
        print(f"\n⏱️ 测试完成，耗时: {end_time - start_time:.2f}秒")
        print(f"📈 验证结果: {len(results)} 个指标完成验证")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断测试")
        return []
    except Exception as e:
        print(f"\n❌ 测试过程出错: {str(e)}")
        return []


def main():
    """主函数"""
    print("🚀 开始早停功能测试")
    print("=" * 60)
    print(f"⏰ 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 测试1: 成功后早停
        results1 = test_early_stop_success()
        
        # 测试2: 错误后早停
        results2 = test_early_stop_error()
        
        print("\n" + "=" * 60)
        print("📊 测试总结:")
        print(f"  • 测试1 (成功后早停): {len(results1)} 个指标验证完成")
        print(f"  • 测试2 (错误后早停): {len(results2)} 个指标验证完成")
        print()
        print("✅ 早停功能测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        logger.error(f"早停功能测试失败: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main() 