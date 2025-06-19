#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试形态重构系统
验证PatternRegistry是否正确工作，以及buypoint_batch_analyzer是否能正确从PatternRegistry获取形态信息
"""

import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.pattern_registry import PatternRegistry
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from indicators.kdj import KDJ
from indicators.rsi import RSI
from indicators.trix import TRIX
from indicators.roc import ROC
from indicators.cmo import CMO
from utils.logger import get_logger

logger = get_logger(__name__)

def test_pattern_registry():
    """测试PatternRegistry功能"""
    print("=== 测试PatternRegistry功能 ===")
    
    # 创建一些指标实例来注册形态
    kdj = KDJ()
    kdj.initialize()  # 确保调用initialize方法

    rsi = RSI()
    rsi.initialize()  # 确保调用initialize方法

    trix = TRIX()
    trix.initialize()  # 确保调用initialize方法

    roc = ROC()
    roc.initialize()  # 确保调用initialize方法

    cmo = CMO()
    cmo.initialize()  # 确保调用initialize方法
    
    registry = PatternRegistry()
    
    # 测试获取KDJ形态
    kdj_patterns = registry.get_pattern_infos_by_indicator("KDJ")
    print(f"KDJ形态数量: {len(kdj_patterns)}")
    if kdj_patterns:
        print(f"KDJ第一个形态: {kdj_patterns[0].get('display_name', 'N/A')}")
    
    # 测试获取RSI形态
    rsi_patterns = registry.get_pattern_infos_by_indicator("RSI")
    print(f"RSI形态数量: {len(rsi_patterns)}")
    if rsi_patterns:
        print(f"RSI第一个形态: {rsi_patterns[0].get('display_name', 'N/A')}")
    
    # 测试获取TRIX形态
    trix_patterns = registry.get_pattern_infos_by_indicator("TRIX")
    print(f"TRIX形态数量: {len(trix_patterns)}")
    if trix_patterns:
        print(f"TRIX第一个形态: {trix_patterns[0].get('display_name', 'N/A')}")
    
    # 测试获取ROC形态
    roc_patterns = registry.get_pattern_infos_by_indicator("ROC")
    print(f"ROC形态数量: {len(roc_patterns)}")
    if roc_patterns:
        print(f"ROC第一个形态: {roc_patterns[0].get('display_name', 'N/A')}")
    
    # 测试获取CMO形态
    cmo_patterns = registry.get_pattern_infos_by_indicator("CMO")
    print(f"CMO形态数量: {len(cmo_patterns)}")
    if cmo_patterns:
        print(f"CMO第一个形态: {cmo_patterns[0].get('display_name', 'N/A')}")
    
    print()

def test_buypoint_analyzer_pattern_retrieval():
    """测试BuyPointBatchAnalyzer的形态检索功能"""
    print("=== 测试BuyPointBatchAnalyzer形态检索 ===")
    
    analyzer = BuyPointBatchAnalyzer()
    
    # 测试一些常见的形态检索
    test_cases = [
        ("KDJ", "KDJ_GOLDEN_CROSS", "KDJ金叉"),
        ("RSI", "RSI_OVERSOLD", "RSI超卖"),
        ("TRIX", "TRIX_ABOVE_ZERO", "TRIX零轴上方"),
        ("ROC", "ROC_ABOVE_ZERO", "ROC零轴上方"),
        ("CMO", "CMO_RISING", "CMO上升"),
        ("MACD", "MACD_GOLDEN_CROSS", "MACD金叉"),  # 测试已有的P0指标
    ]
    
    for indicator_name, pattern_id, expected_name in test_cases:
        try:
            pattern_info = analyzer.get_precise_pattern_info(indicator_name, pattern_id, "")
            print(f"{indicator_name} - {pattern_id}:")
            print(f"  名称: {pattern_info.get('name', 'N/A')}")
            print(f"  描述: {pattern_info.get('description', 'N/A')[:50]}...")
            print()
        except Exception as e:
            print(f"错误 - {indicator_name} - {pattern_id}: {e}")
            print()

def test_pattern_migration_completeness():
    """测试形态迁移的完整性"""
    print("=== 测试形态迁移完整性 ===")
    
    # 从COMPLETE_INDICATOR_PATTERNS_MAP中获取一些形态，测试是否已经迁移到PatternRegistry
    from analysis.buypoints.buypoint_batch_analyzer import COMPLETE_INDICATOR_PATTERNS_MAP
    
    registry = PatternRegistry()
    analyzer = BuyPointBatchAnalyzer()
    
    # 测试一些已迁移的指标
    migrated_indicators = ["TRIX", "ROC", "CMO"]
    
    for indicator_name in migrated_indicators:
        if indicator_name in COMPLETE_INDICATOR_PATTERNS_MAP:
            centralized_patterns = COMPLETE_INDICATOR_PATTERNS_MAP[indicator_name]
            registry_patterns = registry.get_pattern_infos_by_indicator(indicator_name)
            
            print(f"{indicator_name}:")
            print(f"  中心化映射形态数量: {len(centralized_patterns)}")
            print(f"  PatternRegistry形态数量: {len(registry_patterns)}")
            
            # 测试一些形态是否能正确检索
            for pattern_id in list(centralized_patterns.keys())[:3]:  # 测试前3个
                pattern_info = analyzer.get_precise_pattern_info(indicator_name, pattern_id, "")
                source = "PatternRegistry" if registry.get_pattern(f"{indicator_name}_{pattern_id}".upper()) else "中心化映射"
                print(f"    {pattern_id}: {pattern_info.get('name', 'N/A')} (来源: {source})")
            print()

def main():
    """主测试函数"""
    print("开始测试形态重构系统...")
    print()
    
    try:
        test_pattern_registry()
        test_buypoint_analyzer_pattern_retrieval()
        test_pattern_migration_completeness()
        
        print("✅ 所有测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
