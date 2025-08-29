#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标架构合规性详细调试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

def debug_architecture_detailed():
    """详细调试架构合规性"""
    from indicators.ma import MaMa
    
    print("🔍 详细调试MA架构合规性...")
    
    ma = MaMa()
    
    # 测试1: 继承结构测试
    print("\n=== 测试1: 继承结构测试 ===")
    inheritance_result = test_inheritance_structure(ma)
    print(f"继承结构评分: {inheritance_result.get('score', 0)}/100")
    
    # 测试2: 方法组织测试
    print("\n=== 测试2: 方法组织测试 ===")
    method_result = test_method_organization(ma)
    print(f"方法组织评分: {method_result.get('score', 0)}/100")
    
    # 测试3: 抽象合规性测试
    print("\n=== 测试3: 抽象合规性测试 ===")
    abstract_result = test_abstraction_compliance(ma)
    print(f"抽象合规性评分: {abstract_result.get('score', 0)}/100")
    
    # 计算总体评分
    scores = [
        inheritance_result.get('score', 0),
        method_result.get('score', 0),
        abstract_result.get('score', 0)
    ]
    overall_score = sum(scores) / len(scores)
    
    print(f"\n📊 分层架构总体评分: {overall_score:.1f}/100")
    
    return overall_score

def test_inheritance_structure(ma):
    """测试继承结构"""
    print("🔍 测试继承结构...")
    
    try:
        # 检查继承结构
        ma_class = ma.__class__
        base_classes = ma_class.__bases__
        
        # 检查是否有适当的基类
        has_base_class = len(base_classes) > 0
        base_class_names = [cls.__name__ for cls in base_classes]
        
        print(f"基类: {base_class_names}")
        print(f"有基类: {has_base_class}")
        
        # 检查是否继承自抽象基类或指标基类
        has_indicator_base = any('Indicator' in name or 'Base' in name or 'Mixin' in name for name in base_class_names)
        
        print(f"有指标基类: {has_indicator_base}")
        
        # 检查MRO（方法解析顺序）
        mro = ma_class.__mro__
        mro_length = len(mro)
        
        print(f"MRO长度: {mro_length}")
        print(f"MRO: {[cls.__name__ for cls in mro]}")
        
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
        
        return {
            'has_base_class': has_base_class,
            'base_class_names': base_class_names,
            'has_indicator_base': has_indicator_base,
            'mro_length': mro_length,
            'mro_classes': [cls.__name__ for cls in mro],
            'score': score
        }
        
    except Exception as e:
        print(f"❌ 继承结构测试异常: {e}")
        return {'score': 0, 'error': str(e)}

def test_method_organization(ma):
    """测试方法组织"""
    print("🔍 测试方法组织...")
    
    try:
        # 获取所有方法（安全地获取）
        all_methods = []
        for method_name in dir(ma):
            if callable(getattr(ma, method_name, None)):
                all_methods.append(method_name)
        
        # 分类方法
        public_methods = [m for m in all_methods if not m.startswith('_')]
        private_methods = [m for m in all_methods if m.startswith('_') and not m.startswith('__')]
        magic_methods = [m for m in all_methods if m.startswith('__') and m.endswith('__')]
        
        print(f"总方法数: {len(all_methods)}")
        print(f"公共方法数: {len(public_methods)}")
        print(f"私有方法数: {len(private_methods)}")
        print(f"魔术方法数: {len(magic_methods)}")
        
        # 检查必需的公共方法
        required_public = ['calculate', 'set_parameters', 'get_patterns']
        has_required_public = [m for m in required_public if m in public_methods]
        
        print(f"必需公共方法: {required_public}")
        print(f"已有必需方法: {has_required_public}")
        
        # 检查是否有适当的私有方法（实现细节）
        has_private_implementation = len(private_methods) > 0
        
        print(f"有私有实现: {has_private_implementation}")
        
        # 评分
        score = 0
        if len(has_required_public) >= 2:
            score += 50
            print("✅ +50分: 有必需公共方法")
        else:
            print("❌ +0分: 缺少必需公共方法")
            
        if has_private_implementation:
            score += 30
            print("✅ +30分: 有私有实现")
        else:
            print("❌ +0分: 无私有实现")
            
        if len(public_methods) >= 3:
            score += 20
            print("✅ +20分: 公共方法数量合理")
        else:
            print("❌ +0分: 公共方法数量不足")
        
        return {
            'total_methods': len(all_methods),
            'public_methods': len(public_methods),
            'private_methods': len(private_methods),
            'magic_methods': len(magic_methods),
            'required_public_methods': has_required_public,
            'has_private_implementation': has_private_implementation,
            'score': score
        }
        
    except Exception as e:
        print(f"❌ 方法组织测试异常: {e}")
        return {'score': 0, 'error': str(e)}

def test_abstraction_compliance(ma):
    """测试抽象合规性"""
    print("🔍 测试抽象合规性...")
    
    try:
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
        
        # 评分
        implementation_rate = len(implemented_methods) / len(required_abstract_methods)
        score = implementation_rate * 100
        
        print(f"实现率: {implementation_rate:.1%}")
        print(f"评分: {score:.1f}/100")
        
        return {
            'required_abstract_methods': required_abstract_methods,
            'implemented_methods': implemented_methods,
            'missing_methods': missing_methods,
            'has_minimum_periods_property': has_minimum_periods_property,
            'implementation_rate': implementation_rate,
            'score': score
        }
        
    except Exception as e:
        print(f"❌ 抽象合规性测试异常: {e}")
        return {'score': 0, 'error': str(e)}

if __name__ == "__main__":
    debug_architecture_detailed()
