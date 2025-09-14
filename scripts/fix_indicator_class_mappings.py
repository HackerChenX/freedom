#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复指标类名映射问题

解决注册表中的类名与实际类名不匹配的问题
"""

import os
import re
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

def get_indicator_mappings_from_registry() -> Dict[str, str]:
    """从注册表获取指标映射"""
    try:
        from indicators.complete_indicator_registry import CompleteIndicatorRegistry
        registry = CompleteIndicatorRegistry()
        
        # 获取所有指标映射
        mappings = {}
        
        # 从注册表的各个方法中提取映射
        registry_methods = [
            '_register_core_indicators',
            '_register_trend_indicators', 
            '_register_oscillator_indicators',
            '_register_volume_indicators',
            '_register_volatility_indicators',
            '_register_zxm_indicators',
            '_register_pattern_indicators',
            '_register_enhanced_indicators',
            '_register_other_indicators'
        ]
        
        for method_name in registry_methods:
            if hasattr(registry, method_name):
                try:
                    method = getattr(registry, method_name)
                    # 这里我们需要手动检查每个方法的源码来提取映射
                    pass
                except Exception as e:
                    print(f"⚠️ 无法调用方法 {method_name}: {e}")
        
        return mappings
        
    except Exception as e:
        print(f"❌ 无法从注册表获取映射: {e}")
        return {}

def check_class_exists(module_path: str, class_name: str) -> bool:
    """检查指定模块中是否存在指定类"""
    try:
        module = importlib.import_module(module_path)
        return hasattr(module, class_name)
    except Exception:
        return False

def find_actual_classes_in_module(module_path: str) -> List[str]:
    """查找模块中的实际类名"""
    try:
        module = importlib.import_module(module_path)
        classes = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, '__module__') and attr.__module__ == module_path:
                classes.append(attr_name)
        return classes
    except Exception as e:
        print(f"⚠️ 无法检查模块 {module_path}: {e}")
        return []

def add_class_alias(file_path: str, original_class: str, alias: str) -> bool:
    """在文件末尾添加类别名"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有这个别名
        if f"{alias} = {original_class}" in content:
            return True
        
        # 在文件末尾添加别名
        if not content.endswith('\n'):
            content += '\n'
        
        content += f"\n# 添加类别名供注册系统使用\n{alias} = {original_class}\n"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 添加别名: {file_path} - {alias} = {original_class}")
        return True
        
    except Exception as e:
        print(f"❌ 添加别名失败 {file_path}: {e}")
        return False

def fix_known_mapping_issues():
    """修复已知的映射问题"""
    
    # 已知的映射问题列表
    known_issues = [
        # (注册名, 模块路径, 期望类名, 实际类名)
        ('VR', 'indicators.vr', 'VR', 'VolumeRatio'),
        ('MFI', 'indicators.mfi', 'MFI', 'MoneyFlowIndex'),
        ('KC', 'indicators.kc', 'KC', 'KeltnerChannels'),
        ('VIX', 'indicators.vix', 'VIX', 'VolatilityIndex'),
        ('ENHANCED_KDJ', 'indicators.oscillator.enhanced_kdj', 'ENHANCED_KDJ', 'EnhancedKDJ'),
        ('ENHANCED_STOCHRSI', 'indicators.enhanced_stochrsi', 'ENHANCED_STOCHRSI', 'EnhancedStochasticRSI'),
        ('MTM', 'indicators.mtm', 'MTM', 'Momentum'),
        ('COMPOSITE', 'indicators.composite', 'COMPOSITE', 'CompositeIndicator'),
        ('MACD_SCORE', 'indicators.macd_score', 'MACD_SCORE', 'MacdScore'),
        ('RSI_SCORE', 'indicators.rsi_score', 'RSI_SCORE', 'RsiScore'),
        ('BOLL_SCORE', 'indicators.boll_score', 'BOLL_SCORE', 'BollScore'),
        ('KDJ_SCORE', 'indicators.kdj_score', 'KDJ_SCORE', 'KdjScore'),
    ]
    
    fixed_count = 0
    
    for reg_name, module_path, expected_class, actual_class in known_issues:
        # 转换模块路径为文件路径
        file_path = module_path.replace('.', '/') + '.py'
        
        if os.path.exists(file_path):
            # 检查实际类是否存在
            if check_class_exists(module_path, actual_class):
                # 添加别名
                if add_class_alias(file_path, actual_class, expected_class):
                    fixed_count += 1
            else:
                print(f"⚠️ 模块 {module_path} 中未找到类 {actual_class}")
        else:
            print(f"⚠️ 文件不存在: {file_path}")
    
    return fixed_count

def main():
    """主函数"""
    print("🔧 开始修复指标类名映射问题...")
    
    # 修复已知的映射问题
    print("\n🔍 修复已知的类名映射问题...")
    fixed_count = fix_known_mapping_issues()
    
    print(f"\n🎉 修复完成: 共修复 {fixed_count} 个类名映射问题")
    print("✅ 指标类名映射问题修复完成！")

if __name__ == "__main__":
    main()
