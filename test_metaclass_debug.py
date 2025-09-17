#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metaclass冲突调试脚本
用于诊断核心指标的多重继承问题
"""

import sys
import traceback

def test_imports():
    """测试基础导入"""
    try:
        from indicators.base_indicator import BaseIndicator
        print("✅ BaseIndicator导入成功")
        print(f"   BaseIndicator metaclass: {type(BaseIndicator)}")
        
        from indicators.base.pattern_signal_mixin import PatternSignalMixin
        print("✅ PatternSignalMixin导入成功")
        print(f"   PatternSignalMixin metaclass: {type(PatternSignalMixin)}")
        
        from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
        print("✅ MinimumPeriodsMixin导入成功")
        print(f"   MinimumPeriodsMixin metaclass: {type(MinimumPeriodsMixin)}")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_simple_inheritance():
    """测试简单继承"""
    try:
        from indicators.base_indicator import BaseIndicator
        from indicators.base.pattern_signal_mixin import PatternSignalMixin
        from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
        
        # 测试单一继承
        class TestIndicator1(BaseIndicator):
            def calculate(self, data):
                return data
            def get_signal(self, data):
                return {}
            def get_patterns(self, data):
                return []
        
        print("✅ 单一继承BaseIndicator成功")
        
        # 测试双重继承
        class TestIndicator2(BaseIndicator, PatternSignalMixin):
            def calculate(self, data):
                return data
            def get_signal(self, data):
                return {}
            def get_patterns(self, data):
                return []
        
        print("✅ 双重继承BaseIndicator+PatternSignalMixin成功")
        
        # 测试三重继承
        class TestIndicator3(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
            def calculate(self, data):
                return data
            def get_signal(self, data):
                return {}
            def get_patterns(self, data):
                return []
            @property
            def minimum_periods(self):
                return 20
        
        print("✅ 三重继承BaseIndicator+PatternSignalMixin+MinimumPeriodsMixin成功")
        
        return True
    except Exception as e:
        print(f"❌ 继承测试失败: {e}")
        traceback.print_exc()
        return False

def test_ma_import():
    """测试MA指标导入"""
    try:
        from indicators.ma import MaMa
        print("✅ MA指标导入成功")
        return True
    except Exception as e:
        print(f"❌ MA指标导入失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔧 Metaclass冲突调试测试")
    print("=" * 60)
    
    # 测试基础导入
    print("\n🔍 测试基础导入...")
    if not test_imports():
        return
    
    # 测试继承
    print("\n🔍 测试继承模式...")
    if not test_simple_inheritance():
        return
    
    # 测试MA指标
    print("\n🔍 测试MA指标导入...")
    test_ma_import()

if __name__ == "__main__":
    main()
