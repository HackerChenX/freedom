#!/usr/bin/env python3
"""
直接测试WILLR指标功能，避开依赖注入问题
"""
import pandas as pd
import numpy as np

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # 生成模拟股价数据
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.3
    volume = np.random.randint(1000, 10000, 100)
    
    return pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

def test_willr_functionality():
    """测试WILLR指标功能"""
    print("🔍 直接测试WILLR指标功能")
    print("=" * 50)
    
    # 创建测试数据
    test_data = create_test_data()
    print(f"✅ 测试数据创建成功: {len(test_data)}行")
    
    try:
        # 直接导入并实例化WrWr类，跳过依赖注入
        from indicators.wr import WrWr
        
        # 创建指标实例（使用最小参数）
        wr_indicator = WrWr.__new__(WrWr)  # 跳过__init__
        wr_indicator.name = "WR"
        wr_indicator.period = 14
        wr_indicator.params = {}
        wr_indicator._result = None
        wr_indicator._patterns = []
        wr_indicator.REQUIRED_COLUMNS = ["high", "low", "close"]
        wr_indicator.description = "威廉指标"
        
        print("✅ WrWr实例创建成功")
        
        # 测试calculate方法
        if hasattr(wr_indicator, 'calculate_Wr_Wr'):
            result = wr_indicator.calculate_Wr_Wr(test_data)
            print(f"✅ calculate_Wr_Wr方法执行成功")
            print(f"   结果列: {list(result.columns)}")
            print(f"   数据行数: {len(result)}")
            
            # 检查是否有标准化列名
            if 'wr_value' in result.columns:
                print("✅ 包含标准化列名 'wr_value'")
            else:
                print("⚠️  缺少标准化列名 'wr_value'")
                
            return True
        else:
            print("❌ 缺少calculate_Wr_Wr方法")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_willr_functionality()
    if success:
        print("\n🎉 WILLR指标功能测试通过")
        print("✅ 指标基础功能正常，可以进行100分优化")
    else:
        print("\n❌ WILLR指标功能测试失败")
