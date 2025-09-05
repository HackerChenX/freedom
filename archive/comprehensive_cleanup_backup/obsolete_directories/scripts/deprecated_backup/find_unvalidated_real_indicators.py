#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查找未验证的真正技术指标
基于135个真正指标，找出还没有验证的指标
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
    
    for keyword in system_keywords:
        if keyword in name:
            return False
    return True


def get_all_real_indicators() -> Set[str]:
    """获取所有真正的技术指标"""
    # 从注册表获取
    registry_indicators = set()
    try:
        from indicators.complete_indicator_registry import complete_registry
        all_registry = set(complete_registry.get_all_indicators().keys())
        registry_indicators = {name for name in all_registry if is_real_indicator(name)}
    except Exception as e:
        logger.error(f"获取注册表指标失败: {e}")
    
    # 从文件系统获取
    file_indicators = set()
    indicators_dir = Path(root_dir) / "indicators"
    if indicators_dir.exists():
        for file_path in indicators_dir.glob("*.py"):
            if file_path.name not in ["__init__.py", "base_indicator.py"]:
                indicator_name = file_path.stem.upper()
                if is_real_indicator(indicator_name):
                    file_indicators.add(indicator_name)
    
    # 合并所有真正的指标
    all_real_indicators = registry_indicators.union(file_indicators)
    logger.info(f"📊 找到 {len(all_real_indicators)} 个真正的技术指标")
    
    return all_real_indicators


def get_validated_indicators() -> Set[str]:
    """获取已验证的指标"""
    validated_indicators = set()
    reports_dir = Path(root_dir) / "docs/finaltesting/indicators"
    
    if reports_dir.exists():
        for report_file in reports_dir.glob("*_validation_report.md"):
            indicator_name = report_file.stem.replace("_validation_report", "").upper()
            validated_indicators.add(indicator_name)
    
    logger.info(f"📋 找到 {len(validated_indicators)} 个已验证指标")
    return validated_indicators


def categorize_unvalidated_indicators(unvalidated: Set[str]) -> Dict[str, List[str]]:
    """对未验证指标进行分类"""
    categories = {
        'core_technical': [],      # 核心技术指标
        'trend_indicators': [],    # 趋势指标
        'oscillators': [],         # 振荡器
        'volume_indicators': [],   # 成交量指标
        'volatility_indicators': [], # 波动性指标
        'pattern_indicators': [],  # 形态指标
        'enhanced_versions': [],   # 增强版本
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
    
    for indicator in sorted(unvalidated):
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


def check_base_indicator_inheritance(indicator_name: str) -> bool:
    """检查指标是否继承BaseIndicator"""
    try:
        indicator_file = Path(root_dir) / "indicators" / f"{indicator_name.lower()}.py"
        if indicator_file.exists():
            with open(indicator_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return "BaseIndicator" in content and "class" in content
        return False
    except Exception:
        return False


def assign_priority(indicator: str, category: str) -> str:
    """分配验证优先级"""
    # P0级别 - 最高优先级（核心基础指标）
    p0_indicators = {'MA', 'EMA', 'RSI', 'MACD', 'KDJ', 'BOLL'}
    
    # P1级别 - 高优先级（重要技术指标）
    p1_indicators = {
        'ADX', 'MFI', 'OBV', 'ROC', 'ATR', 'SAR', 'TRIX', 'AROON',
        'CCI', 'DMI', 'BIAS', 'MOMENTUM'
    }
    
    # P2级别 - 中优先级（常用指标）
    p2_indicators = {
        'KC', 'VIX', 'MTM', 'CMO', 'PSY', 'STOCHRSI', 'WILLIAMS_R',
        'AD', 'VR', 'CHAIKIN', 'PVT', 'VOSC'
    }
    
    # P3级别 - 低优先级（专业指标）
    p3_indicators = {
        'SYNERGY', 'UNIFIED_MA', 'VOLUME_RATIO', 'DIVERGENCE',
        'FIBONACCI_TOOLS', 'GANN_TOOLS', 'PIVOT_POINTS'
    }
    
    if indicator in p0_indicators:
        return 'P0-最高'
    elif indicator in p1_indicators:
        return 'P1-高'
    elif indicator in p2_indicators:
        return 'P2-中'
    elif indicator in p3_indicators:
        return 'P3-低'
    elif indicator.startswith('ZXM_'):
        return 'P4-ZXM专业'
    elif indicator.startswith('ENHANCED') or 'Enhanced' in indicator:
        return 'P5-增强版本'
    elif category == 'pattern_indicators':
        return 'P6-形态识别'
    else:
        return 'P7-其他'


def main():
    """主函数"""
    logger.info("🔍 查找未验证的真正技术指标...")
    logger.info("=" * 80)
    
    # 1. 获取所有真正的指标
    all_real_indicators = get_all_real_indicators()
    
    # 2. 获取已验证的指标
    validated_indicators = get_validated_indicators()
    
    # 3. 找出未验证的指标
    unvalidated_indicators = all_real_indicators - validated_indicators
    
    # 4. 分类未验证指标
    categories = categorize_unvalidated_indicators(unvalidated_indicators)
    
    # 5. 输出统计结果
    logger.info("📊 未验证指标统计结果:")
    logger.info("=" * 80)
    logger.info(f"📈 真正指标总数: {len(all_real_indicators)}")
    logger.info(f"✅ 已验证指标数: {len(validated_indicators)}")
    logger.info(f"⏸️ 未验证指标数: {len(unvalidated_indicators)}")
    logger.info(f"📊 验证完成率: {len(validated_indicators)/len(all_real_indicators)*100:.1f}%")
    
    # 6. 按分类显示未验证指标
    logger.info("\n📋 未验证指标分类详情:")
    logger.info("-" * 80)
    
    total_unvalidated = 0
    priority_groups = {}
    
    for category, indicators in categories.items():
        if indicators:
            count = len(indicators)
            total_unvalidated += count
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
            
            logger.info(f"\n{category_name}: {count}个")
            
            for indicator in indicators:
                priority = assign_priority(indicator, category)
                is_base_indicator = check_base_indicator_inheritance(indicator)
                impl_type = "BaseIndicator" if is_base_indicator else "工厂模式"
                
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append({
                    'name': indicator,
                    'category': category_name,
                    'impl_type': impl_type
                })
                
                logger.info(f"  - {indicator} ({priority}, {impl_type})")
    
    # 7. 按优先级分组显示
    logger.info("\n🎯 按优先级分组的未验证指标:")
    logger.info("-" * 80)
    
    for priority in ['P0-最高', 'P1-高', 'P2-中', 'P3-低', 'P4-ZXM专业', 'P5-增强版本', 'P6-形态识别', 'P7-其他']:
        if priority in priority_groups:
            indicators = priority_groups[priority]
            logger.info(f"\n{priority} ({len(indicators)}个):")
            for item in indicators:
                logger.info(f"  - {item['name']} ({item['category']}, {item['impl_type']})")
    
    # 8. 生成验证建议
    logger.info("\n🚀 验证优先级建议:")
    logger.info("-" * 80)
    
    if 'P0-最高' in priority_groups:
        logger.info(f"🔥 立即验证 (P0): {len(priority_groups['P0-最高'])}个核心基础指标")
    if 'P1-高' in priority_groups:
        logger.info(f"⚡ 优先验证 (P1): {len(priority_groups['P1-高'])}个重要技术指标")
    if 'P2-中' in priority_groups:
        logger.info(f"📊 次要验证 (P2): {len(priority_groups['P2-中'])}个常用指标")
    
    logger.info(f"\n📊 总计未验证: {total_unvalidated}个指标")
    logger.info("=" * 80)
    
    return unvalidated_indicators, categories, priority_groups


if __name__ == "__main__":
    main()
