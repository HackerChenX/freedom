#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终审核：全面检查指标遗漏和验证完整性
"""

import sys
import os
import re
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def get_registered_indicators():
    """获取已注册的指标"""
    logger.info("📦 获取已注册的指标...")
    
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        registry = get_indicator_registry()
        all_indicators = registry.get_all_indicators()
        
        logger.info(f"  ✅ 已注册指标数量: {len(all_indicators)}")
        return set(all_indicators.keys())
    
    except Exception as e:
        logger.error(f"  ❌ 获取注册指标失败: {e}")
        return set()


def parse_progress_table():
    """解析进度表中的验证状态"""
    logger.info("📊 解析进度表中的验证状态...")
    
    progress_file = Path(root_dir) / "docs/finaltesting/技术指标验证进度表.md"
    
    if not progress_file.exists():
        logger.error(f"  ❌ 进度表文件不存在: {progress_file}")
        return set(), {}
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析表格中的指标状态
        verified_indicators = set()
        indicator_details = {}
        
        # 匹配表格行
        table_pattern = r'\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*(✅ PASSED|❌ FAILED|⚠️ PARTIAL)\s*\|\s*([0-9.]+/100|\-)\s*\|'
        matches = re.findall(table_pattern, content)
        
        for indicator_name, status, score in matches:
            verified_indicators.add(indicator_name)
            indicator_details[indicator_name] = {
                'status': status,
                'score': score
            }
        
        logger.info(f"  ✅ 进度表中的指标数量: {len(verified_indicators)}")
        return verified_indicators, indicator_details
    
    except Exception as e:
        logger.error(f"  ❌ 解析进度表失败: {e}")
        return set(), {}


def scan_indicator_files():
    """扫描所有指标实现文件"""
    logger.info("🔍 扫描所有指标实现文件...")
    
    indicator_classes = set()
    indicators_dir = Path(root_dir) / "indicators"
    
    # 扫描所有Python文件
    for py_file in indicators_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找类定义
            class_pattern = r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\):'
            classes = re.findall(class_pattern, content)
            
            for cls in classes:
                # 过滤掉明显不是指标的类
                if any(keyword in cls.lower() for keyword in ['base', 'abstract', 'interface', 'registry', 'manager', 'helper', 'util', 'test']):
                    continue
                indicator_classes.add(cls)
        
        except Exception as e:
            logger.warning(f"  ⚠️ 读取文件失败 {py_file}: {e}")
    
    logger.info(f"  ✅ 发现指标类数量: {len(indicator_classes)}")
    return indicator_classes


def check_standard_indicators():
    """检查标准技术指标覆盖度"""
    logger.info("🔍 检查标准技术指标覆盖度...")
    
    # 核心技术指标列表
    core_indicators = [
        'MA', 'EMA', 'WMA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'CCI', 'DMI', 'SAR',
        'TRIX', 'AROON', 'DMA', 'BIAS', 'CMO', 'STOCHRSI', 'STDDEV', 'VORTEX',
        'MOMENTUM', 'CHAIKIN', 'VOL', 'VR', 'AD', 'EMV', 'PVT', 'VOSC',
        'PSY', 'WR', 'ADX', 'ROC', 'OBV', 'MTM', 'MFI', 'VIX', 'KC', 'ATR'
    ]
    
    # 形态识别指标
    pattern_indicators = [
        'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI',
        'PIERCING_LINE', 'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR',
        'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS', 'V_SHAPED_REVERSAL',
        'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
        'WEDGE', 'FLAG', 'PENNANT', 'ISLAND_REVERSAL'
    ]
    
    # ZXM体系指标
    zxm_indicators = [
        'ZXM_FUND_FLOW', 'ZXM_CHIP_DISTRIBUTION', 'ZXM_INSTITUTION_BEHAVIOR',
        'ZXM_WEEKLY_MACD', 'ZXM_MONTHLY_MACD', 'ZXM_TREND_SCORE',
        'ZXM_HOT_SPOT', 'ZXM_INDUSTRY_ROTATION', 'ZXM_CYCLE_POSITION',
        'ZXM_RISK_CONTROL', 'ZXM_TIMING_SIGNAL', 'ZXM_POSITION_MANAGEMENT',
        'ZXM_PORTFOLIO_OPTIMIZATION', 'ZXM_STRATEGY_COMBINATION',
        'ZXM_PERFORMANCE_ATTRIBUTION', 'ZXM_ALPHA_GENERATION', 'ZXM_BETA_HEDGING',
        'ZXM_CORRELATION_MATRIX', 'ZXM_FACTOR_ANALYSIS', 'ZXM_REGIME_DETECTION',
        'ZXM_SENTIMENT_ANALYSIS', 'ZXM_NEWS_IMPACT', 'ZXM_EVENT_DRIVEN',
        'ZXM_MACRO_FACTOR', 'ZXM_SECTOR_ROTATION', 'ZXM_STYLE_ANALYSIS',
        'ZXM_MOMENTUM_FACTOR', 'ZXM_VALUE_FACTOR', 'ZXM_QUALITY_FACTOR',
        'ZXM_GROWTH_FACTOR', 'ZXM_VOLATILITY_FACTOR', 'ZXM_LIQUIDITY_FACTOR',
        'ZXM_SIZE_FACTOR', 'ZXM_PROFITABILITY_FACTOR', 'ZXM_LEVERAGE_FACTOR',
        'ZXM_EFFICIENCY_FACTOR', 'ZXM_DIVIDEND_FACTOR', 'ZXM_EARNINGS_FACTOR',
        'ZXM_CASH_FLOW_FACTOR'
    ]
    
    # 增强指标
    enhanced_indicators = [
        'EnhancedBOLL', 'EnhancedKDJ', 'EnhancedCCI', 'EnhancedTRIX', 'EnhancedSTOCHRSI'
    ]
    
    # 其他专业指标
    other_indicators = [
        'COMPOSITE', 'SYNERGY', 'MACD_SCORE', 'RSI_SCORE', 'BOLL_SCORE', 'KDJ_SCORE',
        'RSIMA', 'INSTITUTIONAL_BEHAVIOR', 'FUND_FLOW', 'CHIP_DISTRIBUTION'
    ]
    
    all_standard = set(core_indicators + pattern_indicators + zxm_indicators + enhanced_indicators + other_indicators)
    
    return {
        'core': set(core_indicators),
        'pattern': set(pattern_indicators),
        'zxm': set(zxm_indicators),
        'enhanced': set(enhanced_indicators),
        'other': set(other_indicators),
        'all': all_standard
    }


def comprehensive_audit():
    """执行全面审核"""
    logger.info("🚀 开始最终全面审核...")
    logger.info("=" * 80)
    
    # 1. 获取各种数据源
    registered_indicators = get_registered_indicators()
    verified_indicators, verification_details = parse_progress_table()
    file_classes = scan_indicator_files()
    standard_sets = check_standard_indicators()
    
    logger.info("=" * 80)
    logger.info("📊 最终审核结果")
    logger.info("=" * 80)
    
    # 分析1：注册vs验证一致性
    logger.info("🔍 分析1: 注册vs验证一致性检查")
    
    registered_not_verified = registered_indicators - verified_indicators
    verified_not_registered = verified_indicators - registered_indicators
    
    if not registered_not_verified and not verified_not_registered:
        logger.info("  ✅ 完美一致：所有注册指标都已验证，所有验证指标都已注册")
    else:
        if registered_not_verified:
            logger.warning(f"  ⚠️ 已注册但未验证: {len(registered_not_verified)}个")
            for indicator in sorted(registered_not_verified):
                logger.warning(f"    - {indicator}")
        
        if verified_not_registered:
            logger.warning(f"  ⚠️ 已验证但未注册: {len(verified_not_registered)}个")
            for indicator in sorted(verified_not_registered):
                logger.warning(f"    - {indicator}")
    
    # 分析2：验证状态统计
    logger.info("\n🔍 分析2: 验证状态详细统计")
    
    passed_count = sum(1 for details in verification_details.values() if details['status'] == '✅ PASSED')
    failed_count = sum(1 for details in verification_details.values() if details['status'] == '❌ FAILED')
    partial_count = sum(1 for details in verification_details.values() if details['status'] == '⚠️ PARTIAL')
    
    logger.info(f"  📊 验证状态分布:")
    logger.info(f"    - ✅ PASSED: {passed_count}个 ({passed_count/len(verification_details)*100:.1f}%)")
    logger.info(f"    - ❌ FAILED: {failed_count}个 ({failed_count/len(verification_details)*100:.1f}%)")
    logger.info(f"    - ⚠️ PARTIAL: {partial_count}个 ({partial_count/len(verification_details)*100:.1f}%)")
    
    # 分析3：标准指标覆盖度
    logger.info("\n🔍 分析3: 标准指标覆盖度分析")
    
    for category, standard_set in standard_sets.items():
        if category == 'all':
            continue
        covered = registered_indicators & standard_set
        missing = standard_set - registered_indicators
        coverage = len(covered) / len(standard_set) * 100 if standard_set else 0
        
        logger.info(f"  📊 {category.upper()}指标覆盖率: {coverage:.1f}% ({len(covered)}/{len(standard_set)})")
        if missing:
            logger.info(f"    缺失: {sorted(missing)}")
    
    # 分析4：文件vs注册对比
    logger.info("\n🔍 分析4: 文件实现vs注册状态对比")
    
    potential_unregistered = file_classes - registered_indicators
    if potential_unregistered:
        logger.warning(f"  ⚠️ 文件中存在但可能未注册的类: {len(potential_unregistered)}个")
        for cls in sorted(potential_unregistered):
            logger.warning(f"    - {cls}")
    else:
        logger.info("  ✅ 所有文件中的指标类都已正确注册")
    
    # 最终总结
    logger.info("\n" + "=" * 80)
    logger.info("🎯 最终审核总结")
    logger.info("=" * 80)
    
    total_issues = len(registered_not_verified) + len(verified_not_registered) + len(potential_unregistered)
    
    logger.info(f"📊 系统状态统计:")
    logger.info(f"  - 已注册指标: {len(registered_indicators)}个")
    logger.info(f"  - 已验证指标: {len(verified_indicators)}个")
    logger.info(f"  - 验证通过率: {passed_count/len(verification_details)*100:.1f}%")
    logger.info(f"  - 文件中的类: {len(file_classes)}个")
    logger.info(f"  - 标准指标覆盖: {len(registered_indicators & standard_sets['all'])/len(standard_sets['all'])*100:.1f}%")
    
    if total_issues == 0 and passed_count == len(verification_details):
        logger.info("\n🎉 审核结果: 系统完美无缺！")
        logger.info("✅ 所有指标都已正确注册、验证并通过")
        logger.info("✅ 没有发现任何遗漏或不一致问题")
        logger.info("✅ 系统状态: 100%完整性和正确性")
        return True
    else:
        logger.warning(f"\n⚠️ 发现 {total_issues} 个潜在问题")
        if failed_count > 0:
            logger.warning(f"⚠️ 有 {failed_count} 个指标验证失败")
        return False


def main():
    """主函数"""
    try:
        success = comprehensive_audit()
        
        if success:
            logger.info("🎉 最终审核完成：系统完美无缺！")
        else:
            logger.warning("⚠️ 最终审核完成：发现需要关注的问题")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 审核过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
