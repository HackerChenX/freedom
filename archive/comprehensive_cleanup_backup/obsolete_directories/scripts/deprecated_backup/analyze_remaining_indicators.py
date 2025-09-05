#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析剩余未修复的指标
"""

import sys
import os
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def analyze_remaining_indicators():
    """分析剩余未修复的指标"""
    logger.info("🔍 分析剩余未修复的指标...")
    
    try:
        # 1. 获取所有注册的指标
        logger.info("📦 获取所有注册的指标...")
        
        # 导入指标注册系统
        from indicators.complete_indicator_registry import get_indicator_registry

        registry = get_indicator_registry()
        all_indicators = list(registry.get_all_indicators().keys())
        total_indicators = len(all_indicators)
        
        logger.info(f"✅ 总注册指标数: {total_indicators}")
        logger.info(f"  - 指标列表: {sorted(all_indicators)}")
        
        # 2. 已修复的指标列表（基于我们的修复记录）
        fixed_indicators = {
            # 直接修复的指标 (16个)
            'ZXM_VOLATILITY_FORECAST',
            'ZXM_MARKET_SENTIMENT', 
            'ZXM_LIQUIDITY_ANALYSIS',
            'ZXM_CORRELATION_MATRIX',
            'ADX',
            'ROC',
            'OBV',
            'MTM',
            'MFI',
            'VIX',
            'SYNERGY',
            'UNIFIED_MA',
            'THREE_BLACK_CROWS',
            'THREE_WHITE_SOLDIERS',
            'PENNANT',
            'V_SHAPED_REVERSAL',
            
            # 错误标记修正的指标 (15个)
            'DOJI',
            'HAMMER',
            'SHOOTING_STAR',
            'ENGULFING',
            'HARAMI',
            'PIERCING_LINE',
            'DARK_CLOUD_COVER',
            'MORNING_STAR',
            'EVENING_STAR',
            'HEAD_SHOULDERS',
            'DOUBLE_TOP',
            'DOUBLE_BOTTOM',
            'TRIANGLE',
            'WEDGE',
            'FLAG',
            
            # 其他已验证通过的指标
            'MA',
            'EMA',
            'MACD',
            'RSI',
            'KDJ',
            'BOLL',
            'CCI',
            'DMI',
            'SAR',
            'TRIX',
            'AROON',
            'STOCHRSI',
            'STDDEV',
            'KC',
            'AD',
            'EMV',
            'VR',
            'COMPOSITE',
            'MACD_SCORE',
            'RSI_SCORE',
            'BOLL_SCORE',
            'KDJ_SCORE',
            'ATR',
            'ENHANCED_MACD',
            'ENHANCED_CCI',
            'ENHANCED_STOCHRSI',
            'EnhancedBOLL',
            'EnhancedKDJ',
            'EnhancedTRIX',
            'CMO',
            'RSIMA',
            
            # ZXM指标 (38个)
            'ZXM_DAILY_MACD',
            'ZXM_TURNOVER',
            'ZXM_VOLUME_SHRINK',
            'ZXM_MA_CALLBACK',
            'ZXM_BS_ABSORB',
            'ZXM_DAILY_TREND_UP',
            'ZXM_WEEKLY_TREND_UP',
            'ZXM_MONTHLY_KDJ_TREND_UP',
            'ZXM_WEEKLY_MACD',
            'ZXM_MONTHLY_MACD',
            'ZXM_AMPLITUDE_ELASTICITY',
            'ZXM_RISE_ELASTICITY',
            'ZXM_ELASTICITY',
            'ZXM_BOUNCE_DETECTOR',
            'ZXM_BUYPOINT_SCORE',
            'ZXM_TREND_SCORE',
            'ZXM_ELASTIC_SCORE',
            'ZXM_VOLUME_ENERGY',
            'ZXM_PRICE_POSITION',
            'ZXM_TECHNICAL_FORM',
            'ZXM_HOT_SPOT',
            'ZXM_INDUSTRY_ROTATION',
            'ZXM_CYCLE_POSITION',
            'ZXM_RISK_CONTROL',
            'ZXM_TIMING_SIGNAL',
            'ZXM_POSITION_MANAGEMENT',
            'ZXM_PORTFOLIO_OPTIMIZATION',
            'ZXM_STRATEGY_COMBINATION',
            'ZXM_PERFORMANCE_ATTRIBUTION',
            'ZXM_ALPHA_GENERATION',
            'ZXM_BETA_HEDGING',
            'ZXM_FUND_FLOW',
            'ZXM_CHIP_DISTRIBUTION',
            'ZXM_INSTITUTION_BEHAVIOR',
            'ZXM_SENTIMENT_ANALYSIS',
            'ZXM_NEWS_IMPACT',
            'ZXM_SOCIAL_MEDIA_SENTIMENT',
            'ZXM_ANALYST_CONSENSUS'
        }
        
        # 3. 计算剩余未修复的指标
        remaining_indicators = set(all_indicators) - fixed_indicators
        
        logger.info(f"✅ 已修复指标数: {len(fixed_indicators)}")
        logger.info(f"❌ 剩余未修复指标数: {len(remaining_indicators)}")
        
        # 4. 按类别分析剩余指标
        logger.info("📊 剩余未修复指标详细分析:")
        
        # 核心技术指标
        core_indicators = []
        # 增强指标
        enhanced_indicators = []
        # 其他指标
        other_indicators = []
        
        for indicator in sorted(remaining_indicators):
            if indicator.startswith('Enhanced') or indicator.startswith('ENHANCED'):
                enhanced_indicators.append(indicator)
            elif any(keyword in indicator for keyword in ['MA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'CCI', 'DMI', 'SAR', 'TRIX']):
                core_indicators.append(indicator)
            else:
                other_indicators.append(indicator)
        
        logger.info(f"  📈 核心技术指标 ({len(core_indicators)}个):")
        for indicator in core_indicators:
            logger.info(f"    - {indicator}")
        
        logger.info(f"  🔧 增强指标 ({len(enhanced_indicators)}个):")
        for indicator in enhanced_indicators:
            logger.info(f"    - {indicator}")
        
        logger.info(f"  📊 其他指标 ({len(other_indicators)}个):")
        for indicator in other_indicators:
            logger.info(f"    - {indicator}")
        
        # 5. 生成修复优先级建议
        logger.info("🎯 修复优先级建议:")
        
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for indicator in remaining_indicators:
            if any(keyword in indicator for keyword in ['WMA', 'BIAS', 'DMA', 'MOMENTUM', 'PSY', 'WR']):
                high_priority.append(indicator)
            elif indicator.startswith('Enhanced') or indicator.startswith('ENHANCED'):
                medium_priority.append(indicator)
            else:
                low_priority.append(indicator)
        
        logger.info(f"  🔴 高优先级 ({len(high_priority)}个): {sorted(high_priority)}")
        logger.info(f"  🟡 中优先级 ({len(medium_priority)}个): {sorted(medium_priority)}")
        logger.info(f"  🟢 低优先级 ({len(low_priority)}个): {sorted(low_priority)}")
        
        # 6. 生成统计报告
        logger.info("=" * 60)
        logger.info("📊 指标修复统计报告")
        logger.info("=" * 60)
        logger.info(f"总注册指标数: {total_indicators}")
        logger.info(f"已修复指标数: {len(fixed_indicators)}")
        logger.info(f"剩余未修复数: {len(remaining_indicators)}")
        logger.info(f"修复完成率: {len(fixed_indicators)/total_indicators*100:.1f}%")
        logger.info(f"剩余修复率: {len(remaining_indicators)/total_indicators*100:.1f}%")
        logger.info("=" * 60)
        
        # 7. 检查是否有遗漏的已修复指标
        logger.info("🔍 检查可能遗漏的已修复指标...")
        
        # 测试几个可能已经工作的指标
        test_indicators = ['WMA', 'BIAS', 'DMA', 'MOMENTUM', 'PSY', 'WR', 'VORTEX', 'CHAIKIN']
        
        for indicator_name in test_indicators:
            if indicator_name in remaining_indicators:
                try:
                    # 尝试获取指标
                    indicator_class = registry.get_indicator(indicator_name)
                    
                    if indicator_class:
                        logger.info(f"  ✅ {indicator_name}: 已注册，可能已经工作正常")
                    else:
                        logger.info(f"  ❌ {indicator_name}: 未正确注册")
                        
                except Exception as e:
                    logger.warning(f"  ⚠️ {indicator_name}: 测试失败 - {e}")
        
        return {
            'total': total_indicators,
            'fixed': len(fixed_indicators),
            'remaining': len(remaining_indicators),
            'remaining_list': sorted(remaining_indicators),
            'high_priority': sorted(high_priority),
            'medium_priority': sorted(medium_priority),
            'low_priority': sorted(low_priority)
        }
        
    except Exception as e:
        logger.error(f"❌ 分析过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None


def main():
    """主函数"""
    try:
        result = analyze_remaining_indicators()
        
        if result:
            logger.info("✅ 剩余指标分析完成")
            logger.info(f"📊 修复完成率: {result['fixed']}/{result['total']} ({result['fixed']/result['total']*100:.1f}%)")
            logger.info(f"📋 剩余指标数: {result['remaining']}")
        else:
            logger.error("❌ 剩余指标分析失败")
        
        return result is not None
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
