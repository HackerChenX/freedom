#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查找未验证的指标文件
对比indicators目录中的实际指标文件与验证报告
"""

import sys
import os
from pathlib import Path
from typing import Set

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def get_indicator_files() -> Set[str]:
    """获取indicators目录中的所有指标文件"""
    indicators_dir = Path(root_dir) / "indicators"
    indicator_files = set()
    
    if indicators_dir.exists():
        for file_path in indicators_dir.glob("*.py"):
            if file_path.name not in ["__init__.py", "base_indicator.py"]:
                # 从文件名推断指标名称
                indicator_name = file_path.stem.upper()
                indicator_files.add(indicator_name)
    
    return indicator_files


def get_validated_indicators() -> Set[str]:
    """获取已有验证报告的指标"""
    reports_dir = Path(root_dir) / "docs/finaltesting/indicators"
    validated_indicators = set()
    
    if reports_dir.exists():
        for report_file in reports_dir.glob("*_validation_report.md"):
            # 从文件名提取指标名称
            indicator_name = report_file.stem.replace("_validation_report", "").upper()
            validated_indicators.add(indicator_name)
    
    return validated_indicators


def check_base_indicator_inheritance(indicator_name: str) -> bool:
    """检查指标是否继承BaseIndicator"""
    try:
        indicator_file = Path(root_dir) / "indicators" / f"{indicator_name.lower()}.py"
        if indicator_file.exists():
            with open(indicator_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 检查是否继承BaseIndicator
                if "BaseIndicator" in content and "class" in content:
                    return True
        return False
    except Exception as e:
        logger.error(f"检查指标 {indicator_name} 继承关系失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🔍 查找未验证的指标文件...")
    logger.info("=" * 80)
    
    # 获取指标文件和已验证指标
    indicator_files = get_indicator_files()
    validated_indicators = get_validated_indicators()
    
    # 找出未验证的指标
    unvalidated_indicators = indicator_files - validated_indicators
    
    # 过滤出真正的BaseIndicator实现
    real_indicators = []
    system_files = []
    
    for indicator in sorted(unvalidated_indicators):
        if check_base_indicator_inheritance(indicator):
            real_indicators.append(indicator)
        else:
            system_files.append(indicator)
    
    # 输出结果
    logger.info("📊 指标验证状态分析")
    logger.info("=" * 80)
    logger.info(f"📈 总指标文件数: {len(indicator_files)}")
    logger.info(f"✅ 已验证指标数: {len(validated_indicators)}")
    logger.info(f"⏸️ 未验证指标数: {len(unvalidated_indicators)}")
    logger.info(f"🎯 真正的BaseIndicator实现: {len(real_indicators)}")
    logger.info(f"🔧 系统文件/工厂类: {len(system_files)}")
    
    if real_indicators:
        logger.info("\n🎯 未验证的真正BaseIndicator指标:")
        logger.info("-" * 40)
        for i, indicator in enumerate(real_indicators, 1):
            logger.info(f"  {i:2d}. {indicator}")
            
        # 推荐下一个验证的指标
        logger.info("\n🚀 建议验证的下一个指标:")
        logger.info("-" * 40)
        next_indicator = real_indicators[0]
        logger.info(f"指标名称: {next_indicator}")
        
        # 检查指标文件详情
        indicator_file = Path(root_dir) / "indicators" / f"{next_indicator.lower()}.py"
        if indicator_file.exists():
            logger.info(f"文件路径: {indicator_file}")
            logger.info(f"文件大小: {indicator_file.stat().st_size} bytes")
    else:
        logger.info("\n🎉 所有真正的BaseIndicator指标都已验证完成！")
    
    if system_files:
        logger.info("\n🔧 系统文件/工厂类 (无需验证):")
        logger.info("-" * 40)
        for i, file_name in enumerate(sorted(system_files), 1):
            logger.info(f"  {i:2d}. {file_name}")
    
    logger.info("=" * 80)
    return real_indicators


if __name__ == "__main__":
    main()
