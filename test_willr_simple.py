#!/usr/bin/env python3
"""
简单测试WILLR指标实例化问题
"""

def test_basic_import():
    """测试基础导入"""
    print("1. 测试基础导入...")
    try:
        from indicators.wr import WrWr
        print("✅ WrWr类导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_class_definition():
    """测试类定义"""
    print("2. 测试类定义...")
    try:
        from indicators.wr import WrWr
        print(f"✅ 类名: {WrWr.__name__}")
        print(f"✅ 基类: {[cls.__name__ for cls in WrWr.__bases__]}")
        return True
    except Exception as e:
        print(f"❌ 类定义检查失败: {e}")
        return False

def test_instantiation():
    """测试实例化"""
    print("3. 测试实例化...")
    try:
        from indicators.wr import WrWr
        # 尝试最简单的实例化
        wr = WrWr(period=14)
        print(f"✅ 实例化成功: {wr.name}")
        return True
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 WILLR指标简单测试")
    print("=" * 40)
    
    success = True
    success &= test_basic_import()
    success &= test_class_definition()
    success &= test_instantiation()
    
    if success:
        print("\n✅ 所有测试通过")
    else:
        print("\n❌ 存在问题需要修复")
