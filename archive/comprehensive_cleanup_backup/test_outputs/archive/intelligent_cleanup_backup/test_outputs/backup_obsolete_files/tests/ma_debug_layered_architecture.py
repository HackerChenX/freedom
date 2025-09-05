#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标分层架构调试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_layered_architecture():
    """调试分层架构"""
    from indicators.ma import MaMa
    
    print("🔍 调试MA分层架构...")
    
    ma = MaMa()
    
    # 检查继承结构
    ma_class = ma.__class__
    base_classes = ma_class.__bases__
    
    print(f"MA类: {ma_class.__name__}")
    print(f"基类数量: {len(base_classes)}")
    print(f"基类: {[cls.__name__ for cls in base_classes]}")
    
    # 检查是否有适当的基类
    has_base_class = len(base_classes) > 0
    base_class_names = [cls.__name__ for cls in base_classes]
    
    print(f"有基类: {has_base_class}")
    print(f"基类名称: {base_class_names}")
    
    # 检查是否继承自抽象基类或指标基类
    has_indicator_base = any('Indicator' in name or 'Base' in name or 'Mixin' in name for name in base_class_names)
    
    print(f"有指标基类: {has_indicator_base}")
    
    # 检查MRO（方法解析顺序）
    mro = ma_class.__mro__
    mro_length = len(mro)
    
    print(f"MRO长度: {mro_length}")
    print(f"MRO类: {[cls.__name__ for cls in mro]}")
    
    # 评分计算
    score = 0
    if has_base_class:
        score += 40
        print("✅ +40分: 有基类")
    else:
        print("❌ +0分: 无基类")
        
    if has_indicator_base:
        score += 40
        print("✅ +40分: 有指标基类")
    else:
        print("❌ +0分: 无指标基类")
        
    if mro_length >= 3:  # 至少有自己、基类、object
        score += 20
        print("✅ +20分: MRO长度合理")
    else:
        print("❌ +0分: MRO长度不合理")
    
    print(f"\n📊 继承结构评分: {score}/100")
    
    # 检查方法组织
    print(f"\n🔍 检查方法组织...")
    
    # 获取所有方法
    all_methods = []
    for method_name in dir(ma):
        if callable(getattr(ma, method_name, None)):
            all_methods.append(method_name)
    
    # 分类方法
    public_methods = [m for m in all_methods if not m.startswith('_')]
    private_methods = [m for m in all_methods if m.startswith('_') and not m.startswith('__')]
    magic_methods = [m for m in all_methods if m.startswith('__') and m.endswith('__')]
    
    print(f"总方法数: {len(all_methods)}")
    print(f"公共方法 ({len(public_methods)}): {public_methods[:10]}...")  # 只显示前10个
    print(f"私有方法 ({len(private_methods)}): {private_methods[:10]}...")
    print(f"魔术方法 ({len(magic_methods)}): {magic_methods[:10]}...")
    
    # 检查必需的公共方法
    required_public = ['calculate', 'set_parameters', 'get_patterns']
    has_required_public = [m for m in required_public if m in public_methods]
    
    print(f"必需公共方法: {required_public}")
    print(f"已有必需方法: {has_required_public}")
    
    # 检查是否有适当的私有方法（实现细节）
    has_private_implementation = len(private_methods) > 0
    
    print(f"有私有实现: {has_private_implementation}")
    
    # 方法组织评分
    method_score = 0
    if len(has_required_public) >= 2:
        method_score += 50
        print("✅ +50分: 有必需公共方法")
    else:
        print("❌ +0分: 缺少必需公共方法")
        
    if has_private_implementation:
        method_score += 30
        print("✅ +30分: 有私有实现")
    else:
        print("❌ +0分: 无私有实现")
        
    if len(public_methods) >= 3:
        method_score += 20
        print("✅ +20分: 公共方法数量合理")
    else:
        print("❌ +0分: 公共方法数量不足")
    
    print(f"📊 方法组织评分: {method_score}/100")
    
    # 检查抽象合规性
    print(f"\n🔍 检查抽象合规性...")
    
    # 检查是否实现了抽象方法
    required_abstract_methods = [
        'calculate', '_get_default_parameters', 'minimum_periods'
    ]
    
    implemented_methods = []
    missing_methods = []
    
    for method in required_abstract_methods:
        if hasattr(ma, method):
            implemented_methods.append(method)
            print(f"✅ 已实现: {method}")
        else:
            missing_methods.append(method)
            print(f"❌ 缺少: {method}")
    
    # 检查minimum_periods是否为属性
    has_minimum_periods_property = hasattr(ma, 'minimum_periods')
    print(f"有minimum_periods属性: {has_minimum_periods_property}")
    
    # 抽象合规性评分
    implementation_rate = len(implemented_methods) / len(required_abstract_methods)
    abstract_score = implementation_rate * 100
    
    print(f"📊 抽象合规性评分: {abstract_score}/100")
    
    # 总体分层架构评分
    overall_score = (score + method_score + abstract_score) / 3
    print(f"\n📊 总体分层架构评分: {overall_score:.1f}/100")
    
    return overall_score

if __name__ == "__main__":
    debug_layered_architecture()
