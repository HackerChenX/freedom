#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查缺失标准极性标注的指标
"""

import os
import re
import importlib
import sys
from pathlib import Path

def find_indicator_files():
    """查找所有指标文件"""
    indicator_files = []
    
    # 主indicators目录
    indicators_dir = Path("indicators")
    
    # 扫描所有.py文件
    for py_file in indicators_dir.rglob("*.py"):
        # 排除特殊文件
        if py_file.name in ["__init__.py", "base_indicator.py", "pattern_registry.py", 
                           "indicator_registry.py", "factory.py", "common.py"]:
            continue
        
        # 排除测试文件和工具文件
        if any(x in py_file.name.lower() for x in ["test", "util", "helper", "manager", 
                                                   "adapter", "composite", "score", "framework"]):
            continue
            
        indicator_files.append(str(py_file))
    
    return sorted(indicator_files)

def check_file_for_polarity_annotation(file_path):
    """检查文件是否有标准极性标注"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有register_patterns方法
        has_register_patterns = bool(re.search(r'def register_patterns\s*\(', content))
        
        # 检查是否有register_pattern_to_registry调用
        has_register_calls = bool(re.search(r'register_pattern_to_registry\s*\(', content))
        
        # 检查是否有polarity参数
        has_polarity = bool(re.search(r'polarity\s*=\s*["\'](?:POSITIVE|NEGATIVE|NEUTRAL)["\']', content))
        
        # 检查是否有BaseIndicator继承
        has_base_indicator = bool(re.search(r'class\s+\w+\s*\(\s*BaseIndicator\s*\)', content))
        
        # 检查是否有get_patterns方法（旧方式）
        has_get_patterns = bool(re.search(r'def get_patterns\s*\(', content))
        
        # 检查是否有_calculate方法（表示是真正的指标）
        has_calculate = bool(re.search(r'def _calculate\s*\(', content))
        
        return {
            'has_register_patterns': has_register_patterns,
            'has_register_calls': has_register_calls,
            'has_polarity': has_polarity,
            'has_base_indicator': has_base_indicator,
            'has_get_patterns': has_get_patterns,
            'has_calculate': has_calculate,
            'is_indicator': has_base_indicator and has_calculate
        }
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_class_name(file_path):
    """提取文件中的主要类名"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找继承BaseIndicator的类
        matches = re.findall(r'class\s+(\w+)\s*\(\s*BaseIndicator\s*\)', content)
        if matches:
            return matches[0]
        
        # 查找其他类
        matches = re.findall(r'class\s+(\w+)\s*\([^)]*\)', content)
        if matches:
            return matches[0]
            
        return None
    except:
        return None

def main():
    print("🔍 检查技术指标系统中缺失标准极性标注的指标")
    print("=" * 80)
    
    indicator_files = find_indicator_files()
    print(f"发现 {len(indicator_files)} 个指标文件")
    print()
    
    missing_polarity = []
    has_polarity = []
    not_indicators = []
    
    for file_path in indicator_files:
        print(f"检查: {file_path}")
        
        result = check_file_for_polarity_annotation(file_path)
        if result is None:
            continue
        
        class_name = extract_class_name(file_path)
        
        if not result['is_indicator']:
            not_indicators.append({
                'file': file_path,
                'class': class_name,
                'reason': '不是标准指标类'
            })
            print(f"  ⚪ 不是指标: {class_name}")
            continue
        
        if result['has_polarity']:
            has_polarity.append({
                'file': file_path,
                'class': class_name,
                'has_register_patterns': result['has_register_patterns'],
                'has_register_calls': result['has_register_calls']
            })
            print(f"  ✅ 有极性标注: {class_name}")
        else:
            missing_polarity.append({
                'file': file_path,
                'class': class_name,
                'has_register_patterns': result['has_register_patterns'],
                'has_register_calls': result['has_register_calls'],
                'has_get_patterns': result['has_get_patterns'],
                'has_calculate': result['has_calculate']
            })
            print(f"  ❌ 缺失极性标注: {class_name}")
    
    print()
    print("📊 检查结果汇总")
    print("=" * 80)
    print(f"总文件数: {len(indicator_files)}")
    print(f"标准指标数: {len(has_polarity) + len(missing_polarity)}")
    print(f"有极性标注: {len(has_polarity)}")
    print(f"缺失极性标注: {len(missing_polarity)}")
    print(f"非指标文件: {len(not_indicators)}")
    
    if missing_polarity:
        print()
        print("❌ 缺失标准极性标注的指标:")
        print("-" * 60)
        for item in missing_polarity:
            print(f"📁 {item['file']}")
            print(f"   类名: {item['class']}")
            print(f"   register_patterns: {'✅' if item['has_register_patterns'] else '❌'}")
            print(f"   register_calls: {'✅' if item['has_register_calls'] else '❌'}")
            print(f"   get_patterns: {'✅' if item['has_get_patterns'] else '❌'}")
            print()
    
    print()
    print("🎯 修复建议:")
    print("-" * 60)
    
    priority_1 = []  # 有get_patterns但缺失标准注册的
    priority_2 = []  # 有_calculate但完全缺失形态识别的
    
    for item in missing_polarity:
        if item['has_get_patterns']:
            priority_1.append(item)
        else:
            priority_2.append(item)
    
    if priority_1:
        print("🔥 优先级1 - 需要添加标准极性标注的指标:")
        for item in priority_1:
            print(f"   • {item['class']} ({item['file']})")
            print(f"     - 已有get_patterns方法，需要添加register_patterns方法")
            print(f"     - 需要使用register_pattern_to_registry注册形态")
    
    if priority_2:
        print()
        print("🔶 优先级2 - 需要完整实现形态识别的指标:")
        for item in priority_2:
            print(f"   • {item['class']} ({item['file']})")
            print(f"     - 需要实现get_patterns方法")
            print(f"     - 需要实现register_patterns方法")

if __name__ == "__main__":
    main()
