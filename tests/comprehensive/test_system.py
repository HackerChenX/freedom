#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合选股测试系统验证脚本

验证系统各个组件是否正常工作
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.comprehensive.stock_selection_tester import ComprehensiveStockSelectionTester


async def test_system():
    """测试系统功能"""
    print("=== 综合选股测试系统验证 ===")
    
    try:
        # 创建测试系统
        print("1. 初始化测试系统...")
        tester = ComprehensiveStockSelectionTester()
        print("   ✓ 测试系统初始化成功")
        
        # 测试指标发现
        print("\n2. 测试指标发现...")
        indicators = await tester._discover_indicators()
        print(f"   ✓ 发现 {len(indicators)} 个指标")
        
        if indicators:
            print(f"   示例指标: {indicators[:5]}")
        
        # 测试形态获取
        print("\n3. 测试形态获取...")
        if indicators:
            test_indicator = indicators[0]
            patterns = tester._get_indicator_patterns(test_indicator)
            print(f"   ✓ 指标 {test_indicator} 有 {len(patterns)} 个形态")
            if patterns:
                print(f"   示例形态: {patterns[:3]}")
        
        # 运行小规模测试
        print("\n4. 运行小规模测试...")
        
        # 临时修改配置以进行快速测试
        original_timeout = tester.config.timeout_seconds
        original_workers = tester.config.parallel_workers
        
        tester.config.timeout_seconds = 60  # 1分钟测试
        tester.config.parallel_workers = 5   # 减少并行度
        
        try:
            results = await tester.run_comprehensive_test()
            
            print(f"   ✓ 测试完成")
            print(f"   测试指标: {results.total_indicators_tested}")
            print(f"   测试形态: {results.total_patterns_tested}")
            print(f"   选出股票: {results.total_stocks_selected}")
            print(f"   总体成功率: {results.overall_success_rate:.2%}")
            
            execution_time = (results.end_time - results.start_time).total_seconds()
            print(f"   执行时间: {execution_time:.1f}秒")
            
        finally:
            # 恢复原始配置
            tester.config.timeout_seconds = original_timeout
            tester.config.parallel_workers = original_workers
        
        print("\n=== 系统验证完成 ===")
        print("✓ 所有组件工作正常")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 系统验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    success = asyncio.run(test_system())
    
    if success:
        print("\n🎉 系统验证成功！可以开始完整测试。")
        exit(0)
    else:
        print("\n💥 系统验证失败！请检查配置和依赖。")
        exit(1)


if __name__ == "__main__":
    main()