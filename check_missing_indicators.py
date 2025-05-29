#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查未注册的指标

扫描所有继承自BaseIndicator的类，检查是否都已在IndicatorFactory中注册
"""

import sys
import inspect
import os
import importlib
import pkgutil

sys.path.append('.')

from indicators.base_indicator import BaseIndicator
from indicators.factory import IndicatorFactory

def find_all_indicator_classes():
    """找出所有继承自BaseIndicator的类"""
    indicator_classes = []
    
    # 获取indicators包的路径
    indicators_pkg = sys.modules.get('indicators')
    if not indicators_pkg:
        print("无法找到indicators包")
        return []
        
    indicators_path = os.path.dirname(indicators_pkg.__file__)
    
    # 遍历indicators包及其子包
    for _, module_name, is_pkg in pkgutil.iter_modules([indicators_path]):
        if module_name in ['__pycache__', 'base_indicator', 'factory', 'indicator_registry', 'common']:
            continue
            
        try:
            # 导入模块
            module = importlib.import_module(f'indicators.{module_name}')
            
            # 从模块中找出所有继承自BaseIndicator的类
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseIndicator) and 
                    obj != BaseIndicator):
                    indicator_classes.append(name)
                    
        except Exception as e:
            print(f"处理模块 {module_name} 时出错: {e}")
    
    # 处理子包
    for _, pkg_name, is_pkg in pkgutil.iter_modules([indicators_path]):
        if not is_pkg or pkg_name == '__pycache__':
            continue
            
        try:
            # 导入子包
            pkg = importlib.import_module(f'indicators.{pkg_name}')
            pkg_path = os.path.dirname(pkg.__file__)
            
            # 遍历子包中的模块
            for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
                if module_name in ['__pycache__', '__init__']:
                    continue
                    
                try:
                    # 导入模块
                    module = importlib.import_module(f'indicators.{pkg_name}.{module_name}')
                    
                    # 从模块中找出所有继承自BaseIndicator的类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseIndicator) and 
                            obj != BaseIndicator):
                            indicator_classes.append(name)
                            
                except Exception as e:
                    print(f"处理模块 indicators.{pkg_name}.{module_name} 时出错: {e}")
                    
        except Exception as e:
            print(f"处理子包 {pkg_name} 时出错: {e}")
    
    return indicator_classes

def check_missing_indicators():
    """检查未注册的指标"""
    # 找出所有继承自BaseIndicator的类
    indicator_classes = find_all_indicator_classes()
    print(f"发现继承自BaseIndicator的类: {len(indicator_classes)}个")
    
    # 获取已注册的指标
    registered = IndicatorFactory.get_supported_indicators()
    print(f"已注册指标: {len(registered)}个")
    
    # 检查是否有遗漏的指标
    missing = set([x.upper() for x in indicator_classes]) - set(registered)
    if missing:
        print("未注册的指标类:")
        for m in sorted(missing):
            print(f"  - {m}")
    else:
        print("所有指标类都已成功注册!")
        
    # 检查是否有多余的指标
    extra = set(registered) - set([x.upper() for x in indicator_classes])
    if extra:
        print("\n注册了但未找到对应类的指标:")
        for e in sorted(extra):
            print(f"  - {e}")
            

if __name__ == "__main__":
    check_missing_indicators() 