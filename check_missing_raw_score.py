#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查指标类是否实现了calculate_raw_score方法
"""

import os
import sys
import inspect
import importlib.util
from typing import List, Dict, Any, Tuple, Set

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from indicators.base_indicator import BaseIndicator


def get_all_python_files(directory: str) -> List[str]:
    """获取指定目录下所有Python文件的路径"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                python_files.append(os.path.join(root, file))
    return python_files


def load_module_from_path(file_path: str) -> Any:
    """从文件路径加载模块"""
    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"加载模块 {file_path} 时出错: {e}")
        return None


def get_indicator_classes(module: Any) -> List[Tuple[str, Any]]:
    """获取模块中所有继承自BaseIndicator的类"""
    if module is None:
        return []
    
    indicator_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseIndicator) and obj != BaseIndicator:
            indicator_classes.append((name, obj))
    return indicator_classes


def check_calculate_raw_score_implementation(indicator_class: Any) -> bool:
    """检查指标类是否实现了calculate_raw_score方法"""
    # 获取类的calculate_raw_score方法
    if not hasattr(indicator_class, 'calculate_raw_score'):
        return False
    
    method = getattr(indicator_class, 'calculate_raw_score')
    
    # 检查方法是否来自当前类，而不是从父类继承
    for base in indicator_class.__bases__:
        if hasattr(base, 'calculate_raw_score'):
            base_method = getattr(base, 'calculate_raw_score')
            if method.__code__ is base_method.__code__:
                # 方法来自父类，未在当前类中实现
                return False
    
    return True


def main():
    """主函数"""
    indicators_dir = os.path.join(root_dir, 'indicators')
    python_files = get_all_python_files(indicators_dir)
    
    missing_implementation = []
    implemented = []
    
    print(f"开始检查 {len(python_files)} 个Python文件...")
    
    for file_path in python_files:
        module = load_module_from_path(file_path)
        indicator_classes = get_indicator_classes(module)
        
        for class_name, indicator_class in indicator_classes:
            if not check_calculate_raw_score_implementation(indicator_class):
                missing_implementation.append((file_path, class_name))
            else:
                implemented.append((file_path, class_name))
    
    print("\n=== 未实现 calculate_raw_score 方法的指标类 ===")
    for file_path, class_name in missing_implementation:
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"{class_name} in {rel_path}")
    
    print(f"\n总计：{len(missing_implementation)} 个指标类未实现 calculate_raw_score 方法")
    print(f"已实现：{len(implemented)} 个指标类")
    
    return missing_implementation


if __name__ == "__main__":
    main() 