#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只统计真正的技术指标
排除系统文件、工厂类、管理器等支持代码
"""

import sys
import os
from pathlib import Path
from typing import Dict, Set, List

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def is_real_indicator(name: str) -> bool:
    """判断是否是真正的技术指标"""
    
    # 系统文件关键词 - 这些不是真正的指标
    system_keywords = [
        'FACTORY', 'REGISTRY', 'MANAGER', 'CALCULATOR', 'ADAPTER',
        'COMMON', 'COMPLETE_INDICATOR_REGISTRY', 'PATTERN_REGISTRY',
        'SERVICE_REGISTRY', 'VECTORIZATION', 'OPTIMIZER', 'BASE_INDICATOR',
        'INDICATOR_FACTORY', 'ENHANCED_FACTORY', 'REAL_TECHNICAL_INDICATORS',
        'ADVANCED_VECTORIZED_CALCULATOR', 'PRODUCTION_VECTORIZATION_OPTIMIZER',
        'VECTORIZATION_PERFORMANCE_BOOST', 'UNIFIED_CALCULATOR',
        'INDICATOR_CALCULATOR', 'CANDLESTICK_PATTERNS', 'SCORING_FRAMEWORK',
        'TECHNICAL_INDICATORS', 'FORMULA_INDICATORS', 'COMPOSITE_INDICATOR',
        'PATTERN_DETECTOR', 'PATTERN_RECOGNITION', 'INDICATOR_MANAGER',
        'PATTERN_MANAGER', 'SCORE_MANAGER'
    ]
    
    # 检查是否包含系统关键词
    for keyword in system_keywords:
        if keyword in name:
            return False
    
    return True


def get_registry_real_indicators() -> Set[str]:
    """从注册表获取真正的技术指标"""
    try:
        from indicators.complete_indicator_registry import complete_registry
        all_indicators = set(complete_registry.get_all_indicators().keys())
        
        # 过滤出真正的指标
        real_indicators = {name for name in all_indicators if is_real_indicator(name)}
        
        logger.info(f"📊 注册表总数: {len(all_indicators)}")
        logger.info(f"📊 真正指标数: {len(real_indicators)}")
        logger.info(f"📊 系统文件数: {len(all_indicators) - len(real_indicators)}")
        
        return real_indicators
    except Exception as e:
        logger.error(f"获取注册表指标失败: {e}")
        return set()


def get_file_real_indicators() -> Set[str]:
    """从文件系统获取真正的技术指标"""
    indicators_dir = Path(root_dir) / "indicators"
    all_files = set()
    
    if indicators_dir.exists():
        for file_path in indicators_dir.glob("*.py"):
            if file_path.name not in ["__init__.py", "base_indicator.py"]:
                indicator_name = file_path.stem.upper()
                all_files.add(indicator_name)
    
    # 过滤出真正的指标
    real_indicators = {name for name in all_files if is_real_indicator(name)}
    
    logger.info(f"📁 文件总数: {len(all_files)}")
    logger.info(f"📁 真正指标数: {len(real_indicators)}")
    logger.info(f"📁 系统文件数: {len(all_files) - len(real_indicators)}")
    
    return real_indicators


def categorize_real_indicators(indicators: Set[str]) -> Dict[str, List[str]]:
    """对真正的指标进行分类"""
    categories = {
        'core_technical': [],      # 核心技术指标 (MA, RSI, MACD等)
        'trend_indicators': [],    # 趋势指标 (SAR, TRIX, AROON等)
        'oscillators': [],         # 振荡器 (KDJ, STOCH, CMO等)
        'volume_indicators': [],   # 成交量指标 (OBV, AD, VR等)
        'volatility_indicators': [], # 波动性指标 (ATR, BOLL, VIX等)
        'pattern_indicators': [],  # 形态指标 (DOJI, HAMMER等)
        'enhanced_versions': [],   # 增强版本 (Enhanced系列)
        'zxm_series': [],         # ZXM专业系列
        'others': []              # 其他指标
    }
    
    # 核心技术指标
    core_indicators = {
        'MA', 'EMA', 'SMA', 'WMA', 'RSI', 'MACD', 'KDJ', 'BOLL', 'CCI', 'DMA'
    }
    
    # 趋势指标
    trend_indicators = {
        'SAR', 'TRIX', 'AROON', 'DMI', 'ADX', 'BIAS', 'MOMENTUM'
    }
    
    # 振荡器
    oscillators = {
        'STOCH', 'STOCHRSI', 'CMO', 'ROC', 'PSY', 'WR', 'WILLIAMS_R'
    }
    
    # 成交量指标
    volume_indicators = {
        'OBV', 'AD', 'VR', 'EMV', 'MFI', 'CHAIKIN', 'PVT', 'VOSC', 'VOLUME_RATIO'
    }
    
    # 波动性指标
    volatility_indicators = {
        'ATR', 'KC', 'VIX', 'STDDEV'
    }
    
    # 形态指标
    pattern_indicators = {
        'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
        'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS',
        'THREE_WHITE_SOLDIERS', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM',
        'TRIANGLE', 'WEDGE', 'FLAG', 'PENNANT', 'V_SHAPED_REVERSAL',
        'FIBONACCI', 'ICHIMOKU', 'PIVOT_POINTS', 'GANN_TOOLS', 'ELLIOTT_WAVE',
        'ISLAND_REVERSAL', 'PLATFORM_BREAKOUT', 'DIVERGENCE'
    }
    
    for indicator in sorted(indicators):
        if indicator.startswith('ENHANCED') or 'Enhanced' in indicator:
            categories['enhanced_versions'].append(indicator)
        elif indicator.startswith('ZXM_'):
            categories['zxm_series'].append(indicator)
        elif indicator in core_indicators:
            categories['core_technical'].append(indicator)
        elif indicator in trend_indicators:
            categories['trend_indicators'].append(indicator)
        elif indicator in oscillators:
            categories['oscillators'].append(indicator)
        elif indicator in volume_indicators:
            categories['volume_indicators'].append(indicator)
        elif indicator in volatility_indicators:
            categories['volatility_indicators'].append(indicator)
        elif indicator in pattern_indicators:
            categories['pattern_indicators'].append(indicator)
        else:
            categories['others'].append(indicator)
    
    return categories


def main():
    """主函数"""
    logger.info("🔍 统计真正的技术指标数量...")
    logger.info("=" * 80)
    
    # 1. 获取注册表中的真正指标
    registry_real_indicators = get_registry_real_indicators()
    
    # 2. 获取文件系统中的真正指标
    file_real_indicators = get_file_real_indicators()
    
    # 3. 合并所有真正的指标
    all_real_indicators = registry_real_indicators.union(file_real_indicators)
    
    # 4. 分类统计
    categories = categorize_real_indicators(all_real_indicators)
    
    # 5. 输出结果
    logger.info("📊 真正的技术指标统计结果:")
    logger.info("=" * 80)
    logger.info(f"📈 注册表真正指标: {len(registry_real_indicators)}")
    logger.info(f"📁 文件系统真正指标: {len(file_real_indicators)}")
    logger.info(f"🎯 合并后真正指标总数: {len(all_real_indicators)}")
    
    logger.info("\n📋 真正指标分类统计:")
    logger.info("-" * 60)
    
    total_count = 0
    for category, indicators in categories.items():
        count = len(indicators)
        total_count += count
        category_name = {
            'core_technical': '核心技术指标',
            'trend_indicators': '趋势指标',
            'oscillators': '振荡器指标',
            'volume_indicators': '成交量指标',
            'volatility_indicators': '波动性指标',
            'pattern_indicators': '形态识别指标',
            'enhanced_versions': '增强版本指标',
            'zxm_series': 'ZXM专业系列',
            'others': '其他指标'
        }.get(category, category)
        
        logger.info(f"  {category_name}: {count}个")
        
        # 显示具体指标名称
        if count > 0:
            if count <= 10:
                for indicator in indicators:
                    logger.info(f"    - {indicator}")
            else:
                # 显示前5个和后5个
                for indicator in indicators[:5]:
                    logger.info(f"    - {indicator}")
                logger.info(f"    ... (省略{count-10}个) ...")
                for indicator in indicators[-5:]:
                    logger.info(f"    - {indicator}")
    
    logger.info(f"\n📊 分类统计总计: {total_count}个")
    
    # 6. 与预期对比
    logger.info("\n🎯 与预期对比:")
    logger.info("-" * 60)
    logger.info(f"预期指标数量: 112个")
    logger.info(f"实际真正指标: {len(all_real_indicators)}个")
    
    if len(all_real_indicators) == 112:
        logger.info("✅ 完全符合预期！")
    elif len(all_real_indicators) > 112:
        extra = len(all_real_indicators) - 112
        logger.info(f"📈 超出预期 {extra}个，可能包含额外的专业指标")
    else:
        missing = 112 - len(all_real_indicators)
        logger.info(f"📉 少于预期 {missing}个，可能有些指标未实现")
    
    # 7. 只在注册表中的指标
    only_in_registry = registry_real_indicators - file_real_indicators
    if only_in_registry:
        logger.info(f"\n📊 只在注册表中的指标 ({len(only_in_registry)}个):")
        for indicator in sorted(only_in_registry):
            logger.info(f"  - {indicator}")
    
    # 8. 只在文件中的指标
    only_in_files = file_real_indicators - registry_real_indicators
    if only_in_files:
        logger.info(f"\n📁 只在文件中的指标 ({len(only_in_files)}个):")
        for indicator in sorted(only_in_files):
            logger.info(f"  - {indicator}")
    
    logger.info("=" * 80)
    logger.info(f"🎯 结论: 系统中共有 {len(all_real_indicators)} 个真正的技术指标")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
