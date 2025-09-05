#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试MACD参数设置问题
"""

import sys
import traceback

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def test_macd_parameters():
    """测试MACD参数设置"""
    print("🔍 调试MACD参数设置问题")
    print("=" * 50)
    
    try:
        # 导入MACD指标
        from indicators.macd import MacdMacd
        print("✅ 成功导入MacdMacd")
        
        # 创建MACD实例
        macd = MacdMacd()
        print("✅ 成功创建MACD实例")
        
        # 测试默认参数
        print("\n📋 测试默认参数:")
        try:
            default_params = macd._get_default_parameters()
            print(f"默认参数: {default_params}")
        except Exception as e:
            print(f"❌ 获取默认参数失败: {e}")
            traceback.print_exc()
        
        # 测试参数设置
        print("\n⚙️ 测试参数设置:")
        try:
            print("调用 set_parameters(fast_period=10, slow_period=20, signal_period=5)")
            macd.set_parameters(fast_period=10, slow_period=20, signal_period=5)
            print("✅ 参数设置成功")
            
            # 检查参数是否正确设置
            print(f"fast_period: {getattr(macd, 'fast_period', 'NOT_SET')}")
            print(f"slow_period: {getattr(macd, 'slow_period', 'NOT_SET')}")
            print(f"signal_period: {getattr(macd, 'signal_period', 'NOT_SET')}")
            print(f"_parameters: {getattr(macd, '_parameters', 'NOT_SET')}")
            
        except Exception as e:
            print(f"❌ 参数设置失败: {e}")
            traceback.print_exc()
        
        # 测试抽象方法
        print("\n🔧 测试抽象方法:")
        try:
            print("调用 set_parameters_Indicator_Base_Indicator(fast_period=15)")
            macd.set_parameters_Indicator_Base_Indicator(fast_period=15)
            print("✅ 抽象方法调用成功")
        except Exception as e:
            print(f"❌ 抽象方法调用失败: {e}")
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"💥 测试过程中发生异常: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_macd_parameters()
    if success:
        print("\n🎉 调试完成")
    else:
        print("\n❌ 调试失败")
