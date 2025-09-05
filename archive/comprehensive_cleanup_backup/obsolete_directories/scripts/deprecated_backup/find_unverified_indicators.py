#!/usr/bin/env python3
"""
查找未验证的指标脚本
分析indicators目录中的所有指标文件，对比进度表中已验证的指标，找出真正未验证的指标
"""

import os
import re
from pathlib import Path

def get_indicator_files():
    """获取indicators目录中的所有指标文件"""
    indicators_dir = Path("indicators")
    indicator_files = []

    # 遍历indicators目录
    for file_path in indicators_dir.rglob("*.py"):
        # 排除特殊文件
        exclude_files = [
            "__init__.py", "base_indicator.py", "common.py", "factory.py",
            "adapter.py", "calculator.py", "manager.py", "registry.py",
            "service_registry.py", "indicator_calculator.py", "indicator_factory.py",
            "indicator_manager.py", "complete_indicator_registry.py",
            "enhanced_factory.py", "formula_indicators.py", "real_technical_indicators.py",
            "technical_indicators.py", "unified_calculator.py", "composite.py",
            "composite_indicator.py", "pattern_detector.py", "pattern_manager.py",
            "pattern_recognition.py", "pattern_registry.py", "scoring_framework.py",
            "score_manager.py", "advanced_vectorized_calculator.py",
            "production_vectorization_optimizer.py", "vectorization_performance_boost.py"
        ]

        if (file_path.name.startswith("__") or
            "backup" in file_path.name or
            file_path.name in exclude_files):
            continue

        # 提取指标名称 - 直接使用文件名
        indicator_name = file_path.stem.upper()

        # 排除一些非指标文件的模式
        exclude_patterns = [
            "MIXIN", "FACTORY", "CALCULATOR", "MANAGER",
            "REGISTRY", "ADAPTER", "COMMON", "UNIFIED", "ENHANCED_FACTORY",
            "VECTORIZED", "OPTIMIZER", "FRAMEWORK", "DETECTOR", "EVALUATOR",
            "COMBINATION", "CONFIRMATION", "QUALITY", "DIAGNOSTICS",
            "BREADTH", "SELECTION", "ORIGINAL", "REAL_TECHNICAL", "FORMULA"
        ]

        if any(pattern in indicator_name for pattern in exclude_patterns):
            continue

        indicator_files.append((indicator_name, str(file_path)))

    return sorted(indicator_files)

def get_verified_indicators():
    """从进度表中获取已验证的指标"""
    progress_file = "docs/finaltesting/技术指标验证进度表.md"
    verified_indicators = set()
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找所有已验证的指标
        pattern = r'\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*[✅🎉]\s*PASSED'
        matches = re.findall(pattern, content)
        
        for match in matches:
            verified_indicators.add(match)
            
    except FileNotFoundError:
        print(f"进度表文件不存在: {progress_file}")
    
    return verified_indicators

def main():
    """主函数"""
    print("🔍 分析indicators目录中的指标文件...")
    
    # 获取所有指标文件
    indicator_files = get_indicator_files()
    print(f"📁 发现 {len(indicator_files)} 个指标文件")
    
    # 获取已验证的指标
    verified_indicators = get_verified_indicators()
    print(f"✅ 进度表中已验证 {len(verified_indicators)} 个指标")
    
    # 找出未验证的指标
    all_indicators = {name for name, _ in indicator_files}
    unverified_indicators = all_indicators - verified_indicators
    
    print(f"\n📊 分析结果:")
    print(f"- 总指标文件: {len(all_indicators)}")
    print(f"- 已验证指标: {len(verified_indicators)}")
    print(f"- 未验证指标: {len(unverified_indicators)}")
    
    if unverified_indicators:
        print(f"\n❌ 未验证的指标 ({len(unverified_indicators)}个):")
        for indicator in sorted(unverified_indicators):
            # 找到对应的文件路径
            file_path = next((path for name, path in indicator_files if name == indicator), "未知路径")
            print(f"   - {indicator} ({file_path})")
    
    # 检查是否有在进度表中但文件不存在的指标
    missing_files = verified_indicators - all_indicators
    if missing_files:
        print(f"\n⚠️ 进度表中存在但文件不存在的指标 ({len(missing_files)}个):")
        for indicator in sorted(missing_files):
            print(f"   - {indicator}")
    
    print(f"\n🎯 建议:")
    if unverified_indicators:
        print(f"1. 将以下 {len(unverified_indicators)} 个未验证指标添加到进度表的待验证列表中")
        print(f"2. 按优先级顺序进行验证")
        print(f"3. 优先验证技术重要性高的指标")
    else:
        print("1. 所有指标文件都已在进度表中记录")
        print("2. 可以继续验证待验证列表中的指标")

if __name__ == "__main__":
    main()
