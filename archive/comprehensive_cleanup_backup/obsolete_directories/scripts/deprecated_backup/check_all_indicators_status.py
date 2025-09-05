#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查所有指标的验证状态
对比indicators目录中的实际指标文件与验证进度表中的记录
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Set

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger
from indicators.complete_indicator_registry import complete_registry

logger = get_logger(__name__)


def get_indicators_from_registry() -> Set[str]:
    """从指标注册表获取所有已注册的指标名称"""
    try:
        # 获取所有已注册的指标
        registered_indicators = set(complete_registry.get_all_indicators().keys())
        logger.info(f"📊 从注册表获取到 {len(registered_indicators)} 个指标")
        return registered_indicators
    except Exception as e:
        logger.error(f"❌ 获取注册表指标失败: {e}")
        return set()


def get_indicators_from_files() -> Set[str]:
    """从indicators目录获取所有指标文件"""
    indicators_dir = Path(root_dir) / "indicators"
    indicator_files = set()
    
    if indicators_dir.exists():
        for file_path in indicators_dir.glob("*.py"):
            if file_path.name != "__init__.py" and file_path.name != "base_indicator.py":
                # 从文件名推断指标名称
                indicator_name = file_path.stem.upper()
                indicator_files.add(indicator_name)
    
    logger.info(f"📁 从文件系统获取到 {len(indicator_files)} 个指标文件")
    return indicator_files


def get_validated_indicators() -> Dict[str, str]:
    """从验证进度表获取已验证的指标"""
    progress_file = Path(root_dir) / "docs/finaltesting/技术指标验证进度表.md"
    validated_indicators = {}
    
    if not progress_file.exists():
        logger.warning("⚠️ 验证进度表文件不存在")
        return validated_indicators
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有验证状态的指标
        patterns = [
            r'\|\s*\*\*([^*]+)\*\*\s*\|\s*🎉\s*PASSED_PRODUCTION_READY',
            r'\|\s*\*\*([^*]+)\*\*\s*\|\s*✅\s*PASSED_ARCHITECTURE_COMPLIANT',
            r'\|\s*\*\*([^*]+)\*\*\s*\|\s*⚠️\s*CONDITIONAL_PASS',
            r'\|\s*\*\*([^*]+)\*\*\s*\|\s*❌\s*FAILED',
            r'\|\s*\*\*([^*]+)\*\*\s*\|\s*🔄\s*IN_PROGRESS'
        ]
        
        status_mapping = {
            'PASSED_PRODUCTION_READY': '🎉 PASSED_PRODUCTION_READY',
            'PASSED_ARCHITECTURE_COMPLIANT': '✅ PASSED_ARCHITECTURE_COMPLIANT',
            'CONDITIONAL_PASS': '⚠️ CONDITIONAL_PASS',
            'FAILED': '❌ FAILED',
            'IN_PROGRESS': '🔄 IN_PROGRESS'
        }
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                indicator_name = match.strip()
                # 从pattern推断状态
                if 'PASSED_PRODUCTION_READY' in pattern:
                    status = '🎉 PASSED_PRODUCTION_READY'
                elif 'PASSED_ARCHITECTURE_COMPLIANT' in pattern:
                    status = '✅ PASSED_ARCHITECTURE_COMPLIANT'
                elif 'CONDITIONAL_PASS' in pattern:
                    status = '⚠️ CONDITIONAL_PASS'
                elif 'FAILED' in pattern:
                    status = '❌ FAILED'
                elif 'IN_PROGRESS' in pattern:
                    status = '🔄 IN_PROGRESS'
                else:
                    status = 'UNKNOWN'
                
                validated_indicators[indicator_name] = status
        
        logger.info(f"📋 从验证进度表获取到 {len(validated_indicators)} 个已验证指标")
        return validated_indicators
        
    except Exception as e:
        logger.error(f"❌ 读取验证进度表失败: {e}")
        return validated_indicators


def check_validation_reports() -> Set[str]:
    """检查validation报告目录中的指标"""
    reports_dir = Path(root_dir) / "docs/finaltesting/indicators"
    validated_from_reports = set()
    
    if reports_dir.exists():
        for report_file in reports_dir.glob("*_validation_report.md"):
            # 从文件名提取指标名称
            indicator_name = report_file.stem.replace("_validation_report", "").upper()
            validated_from_reports.add(indicator_name)
    
    logger.info(f"📄 从验证报告获取到 {len(validated_from_reports)} 个已验证指标")
    return validated_from_reports


def main():
    """主函数"""
    logger.info("🔍 开始检查所有指标的验证状态...")
    logger.info("=" * 80)
    
    # 获取各种来源的指标信息
    registry_indicators = get_indicators_from_registry()
    file_indicators = get_indicators_from_files()
    validated_indicators = get_validated_indicators()
    report_indicators = check_validation_reports()
    
    # 合并所有指标
    all_indicators = registry_indicators.union(file_indicators)
    
    # 分析验证状态
    pending_indicators = []
    validated_indicator_names = set(validated_indicators.keys())
    
    for indicator in sorted(all_indicators):
        if indicator not in validated_indicator_names and indicator not in report_indicators:
            pending_indicators.append(indicator)
    
    # 输出结果
    logger.info("📊 指标验证状态统计")
    logger.info("=" * 80)
    logger.info(f"📈 总指标数: {len(all_indicators)}")
    logger.info(f"✅ 已验证指标数: {len(validated_indicator_names)}")
    logger.info(f"📄 有验证报告: {len(report_indicators)}")
    logger.info(f"⏸️ 待验证指标数: {len(pending_indicators)}")
    logger.info(f"📊 验证完成率: {len(validated_indicator_names)/len(all_indicators)*100:.1f}%")
    
    if pending_indicators:
        logger.info("\n⏸️ 待验证指标列表:")
        logger.info("-" * 40)
        for i, indicator in enumerate(pending_indicators, 1):
            logger.info(f"  {i:2d}. {indicator}")
    else:
        logger.info("\n🎉 所有指标都已完成验证！")
    
    # 输出已验证指标的状态分布
    if validated_indicators:
        logger.info("\n✅ 已验证指标状态分布:")
        logger.info("-" * 40)
        status_count = {}
        for indicator, status in validated_indicators.items():
            status_count[status] = status_count.get(status, 0) + 1
        
        for status, count in sorted(status_count.items()):
            logger.info(f"  {status}: {count}个")
    
    # 找出下一个建议验证的指标
    if pending_indicators:
        logger.info("\n🎯 建议验证的下一个指标:")
        logger.info("-" * 40)
        next_indicator = pending_indicators[0]
        logger.info(f"指标名称: {next_indicator}")
        
        # 检查是否在注册表中
        if next_indicator in registry_indicators:
            logger.info("✅ 已在注册表中注册")
        else:
            logger.info("⚠️ 未在注册表中找到")
        
        # 检查是否有对应文件
        indicator_file = Path(root_dir) / "indicators" / f"{next_indicator.lower()}.py"
        if indicator_file.exists():
            logger.info(f"✅ 找到指标文件: {indicator_file}")
        else:
            logger.info("⚠️ 未找到对应的指标文件")
    
    logger.info("=" * 80)
    return pending_indicators


if __name__ == "__main__":
    main()
