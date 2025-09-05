#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计进度表中标记为PASSED的指标
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


def count_passed_indicators():
    """统计进度表中标记为PASSED的指标"""
    logger.info("📊 统计进度表中标记为PASSED的指标...")
    
    try:
        # 读取进度表文件
        progress_file = Path(root_dir) / "docs/finaltesting/技术指标验证进度表.md"
        
        if not progress_file.exists():
            logger.error(f"❌ 进度表文件不存在: {progress_file}")
            return None
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有标记为PASSED的指标
        passed_pattern = r'\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*✅ PASSED'
        passed_matches = re.findall(passed_pattern, content)
        
        # 去重并排序
        passed_indicators = sorted(list(set(passed_matches)))
        
        logger.info(f"✅ 找到 {len(passed_indicators)} 个标记为PASSED的指标")
        
        # 按类别分组
        core_indicators = []
        zxm_indicators = []
        enhanced_indicators = []
        pattern_indicators = []
        other_indicators = []
        
        for indicator in passed_indicators:
            if indicator.startswith('ZXM_'):
                zxm_indicators.append(indicator)
            elif indicator.startswith('Enhanced') or indicator.startswith('ENHANCED'):
                enhanced_indicators.append(indicator)
            elif indicator in ['DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 
                             'PIERCING_LINE', 'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR',
                             'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS', 'V_SHAPED_REVERSAL',
                             'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE', 
                             'WEDGE', 'FLAG', 'PENNANT']:
                pattern_indicators.append(indicator)
            elif indicator in ['MA', 'EMA', 'WMA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'CCI', 
                             'DMI', 'SAR', 'TRIX', 'AROON', 'DMA', 'BIAS', 'CMO', 'STOCHRSI',
                             'STDDEV', 'VORTEX', 'MOMENTUM', 'CHAIKIN', 'VOL', 'VR', 'AD', 
                             'EMV', 'PVT', 'VOSC', 'PSY', 'WR', 'ADX', 'ROC', 'OBV', 'MTM', 
                             'MFI', 'VIX', 'KC', 'ATR']:
                core_indicators.append(indicator)
            else:
                other_indicators.append(indicator)
        
        # 输出分类统计
        logger.info("📊 按类别统计:")
        logger.info(f"  📈 核心技术指标: {len(core_indicators)}个")
        for indicator in core_indicators:
            logger.info(f"    - {indicator}")
        
        logger.info(f"  🔧 ZXM指标体系: {len(zxm_indicators)}个")
        for indicator in zxm_indicators:
            logger.info(f"    - {indicator}")
        
        logger.info(f"  ⚡ 增强指标: {len(enhanced_indicators)}个")
        for indicator in enhanced_indicators:
            logger.info(f"    - {indicator}")
        
        logger.info(f"  📊 形态识别指标: {len(pattern_indicators)}个")
        for indicator in pattern_indicators:
            logger.info(f"    - {indicator}")
        
        logger.info(f"  🔍 其他指标: {len(other_indicators)}个")
        for indicator in other_indicators:
            logger.info(f"    - {indicator}")
        
        # 获取所有注册的指标进行对比
        logger.info("🔍 对比注册表中的指标...")
        
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            all_indicators = list(registry.get_all_indicators().keys())
            total_indicators = len(all_indicators)
            
            # 计算未标记为PASSED的指标
            not_passed = set(all_indicators) - set(passed_indicators)
            not_passed_list = sorted(list(not_passed))
            
            logger.info(f"📊 总体统计:")
            logger.info(f"  - 总注册指标数: {total_indicators}")
            logger.info(f"  - 标记为PASSED: {len(passed_indicators)} ({len(passed_indicators)/total_indicators*100:.1f}%)")
            logger.info(f"  - 未标记PASSED: {len(not_passed_list)} ({len(not_passed_list)/total_indicators*100:.1f}%)")
            
            logger.info(f"🔍 未标记为PASSED的指标 ({len(not_passed_list)}个):")
            for indicator in not_passed_list:
                logger.info(f"    - {indicator}")
            
            return {
                'total_registered': total_indicators,
                'passed_count': len(passed_indicators),
                'passed_list': passed_indicators,
                'not_passed_count': len(not_passed_list),
                'not_passed_list': not_passed_list,
                'categories': {
                    'core': core_indicators,
                    'zxm': zxm_indicators,
                    'enhanced': enhanced_indicators,
                    'pattern': pattern_indicators,
                    'other': other_indicators
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 获取注册表失败: {e}")
            return {
                'passed_count': len(passed_indicators),
                'passed_list': passed_indicators,
                'categories': {
                    'core': core_indicators,
                    'zxm': zxm_indicators,
                    'enhanced': enhanced_indicators,
                    'pattern': pattern_indicators,
                    'other': other_indicators
                }
            }
        
    except Exception as e:
        logger.error(f"❌ 统计过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None


def main():
    """主函数"""
    try:
        result = count_passed_indicators()
        
        if result:
            logger.info("✅ 进度表PASSED指标统计完成")
            if 'total_registered' in result:
                logger.info(f"📊 完成率: {result['passed_count']}/{result['total_registered']} ({result['passed_count']/result['total_registered']*100:.1f}%)")
            else:
                logger.info(f"📊 PASSED指标数: {result['passed_count']}")
        else:
            logger.error("❌ 进度表PASSED指标统计失败")
        
        return result is not None
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
