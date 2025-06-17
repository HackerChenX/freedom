#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Polarity Based on Technical Analysis
Corrects polarity annotations based on strict technical analysis principles
"""

import os
import re
from typing import Dict, List

def get_technical_correct_polarity(pattern_id: str, display_name: str, description: str, 
                                 pattern_type: str, score_impact: float) -> str:
    """Determine technically correct polarity based on rigorous analysis"""
    
    pattern_lower = pattern_id.lower()
    desc_lower = description.lower() if description else ""
    name_lower = display_name.lower() if display_name else ""
    
    # 1. 明确的技术信号优先级最高
    
    # VIX特殊逻辑 - VIX与股市反向
    if 'vix' in pattern_lower:
        if any(x in pattern_lower for x in ['extreme_panic', 'high_panic', 'top_reversal', 'rapid_rise', 'historical_high']):
            return 'POSITIVE'  # VIX高恐慌对股市是好事
        elif any(x in pattern_lower for x in ['extreme_optimism', 'bottom_reversal', 'historical_low']):
            return 'NEGATIVE'  # VIX低恐慌对股市是坏事
        elif 'uptrend' in pattern_lower:
            return 'NEGATIVE'  # VIX上升趋势对股市不利
        elif 'downtrend' in pattern_lower:
            return 'POSITIVE'  # VIX下降趋势对股市有利
    
    # 交叉信号 - 明确方向性
    if 'golden_cross' in pattern_lower or 'cross_up' in pattern_lower or 'cross_above' in pattern_lower:
        return 'POSITIVE'
    elif 'death_cross' in pattern_lower or 'cross_down' in pattern_lower or 'cross_below' in pattern_lower:
        return 'NEGATIVE'
    elif 'bearish_cross' in pattern_lower:
        return 'NEGATIVE'  # 明确的空头交叉
    elif 'bullish_cross' in pattern_lower:
        return 'POSITIVE'  # 明确的多头交叉
    
    # 突破信号 - 看方向
    if 'breakout' in pattern_lower and 'up' in pattern_lower:
        return 'POSITIVE'
    elif 'breakout' in pattern_lower and 'down' in pattern_lower:
        return 'NEGATIVE'
    elif 'break_up' in pattern_lower or 'break_upper' in pattern_lower:
        return 'POSITIVE'
    elif 'break_down' in pattern_lower or 'break_lower' in pattern_lower or 'breakdown' in pattern_lower:
        return 'NEGATIVE'
    
    # 超买超卖 - 反转逻辑
    if any(x in pattern_lower for x in ['oversold', 'extreme_oversold']):
        return 'POSITIVE'  # 超卖是买入机会
    elif any(x in pattern_lower for x in ['overbought', 'extreme_overbought']):
        return 'NEGATIVE'  # 超买是卖出机会
    
    # 趋势信号
    if any(x in pattern_lower for x in ['uptrend', 'rising', 'bullish_arrangement']):
        return 'POSITIVE'
    elif any(x in pattern_lower for x in ['downtrend', 'falling', 'bearish_arrangement']):
        return 'NEGATIVE'
    
    # 支撑阻力
    if 'support' in pattern_lower:
        return 'POSITIVE'
    elif 'resistance' in pattern_lower:
        return 'NEGATIVE'
    
    # 反转信号 - 看描述确定方向
    if 'reversal' in pattern_lower:
        if 'bullish' in pattern_lower or 'bullish' in name_lower or '看涨' in name_lower:
            return 'POSITIVE'
        elif 'bearish' in pattern_lower or 'bearish' in name_lower or '看跌' in name_lower:
            return 'NEGATIVE'
    
    # 洗盘逻辑 - 根据最终目的
    if 'wash' in pattern_lower:
        if 'completion' in pattern_lower or '完成' in name_lower:
            return 'POSITIVE'  # 洗盘完成是好事
        elif 'support' in pattern_lower or 'breakout' in pattern_lower:
            return 'POSITIVE'  # 有支撑或突破的洗盘
        else:
            return 'NEUTRAL'   # 一般洗盘过程中性
    
    # 机构行为
    if 'absorption' in pattern_lower or 'rally' in pattern_lower or '吸筹' in name_lower or '拉升' in name_lower:
        return 'POSITIVE'
    elif 'distribution' in pattern_lower or 'washout' in pattern_lower or '出货' in name_lower:
        return 'NEGATIVE'
    
    # 成交量确认类 - 通常中性
    if any(x in pattern_lower for x in ['confirmation', 'volume_confirmation', 'trend_alignment']):
        return 'NEUTRAL'
    
    # 时间周期类 - 通常中性
    if any(x in pattern_lower for x in ['time', 'cycle', 'fibonacci_ratio']):
        return 'NEUTRAL'
    
    # 极值状态 - 根据具体情况
    if 'extreme' in pattern_lower:
        if any(x in pattern_lower for x in ['high', 'overbought']):
            return 'NEGATIVE'
        elif any(x in pattern_lower for x in ['low', 'oversold', 'panic']):
            return 'POSITIVE'
        else:
            return 'NEUTRAL'
    
    # 波动性 - ATR/VIX相关
    if 'volatility' in pattern_lower or 'volatile' in pattern_lower:
        if 'high' in pattern_lower:
            return 'NEGATIVE'  # 高波动通常不好
        elif 'low' in pattern_lower:
            return 'POSITIVE'  # 低波动通常好
    
    # 最后根据pattern_type和score_impact
    if pattern_type and score_impact is not None:
        if pattern_type.upper() == 'BULLISH' and score_impact > 0:
            return 'POSITIVE'
        elif pattern_type.upper() == 'BEARISH' and score_impact < 0:
            return 'NEGATIVE'
        elif pattern_type.upper() == 'NEUTRAL':
            return 'NEUTRAL'
    
    # 默认根据评分
    if score_impact is not None:
        if score_impact > 0:
            return 'POSITIVE'
        elif score_impact < 0:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    return 'NEUTRAL'

def fix_critical_polarity_issues():
    """Fix the most critical polarity issues identified in analysis"""
    
    # 关键修正列表 - 基于技术分析的严格判断
    critical_fixes = [
        # VIX相关修正
        ('indicators/vix.py', 'VIX_BOTTOM_REVERSAL', 'NEGATIVE'),  # VIX见底回升对股市不利
        
        # 交叉信号修正
        ('indicators/vortex.py', 'VORTEX_BEARISH_CROSS', 'NEGATIVE'),  # 空头交叉
        
        # 突破方向修正
        ('indicators/boll.py', 'BOLL_PRICE_BREAK_LOWER', 'NEGATIVE'),  # 向下突破
        ('indicators/gann_tools.py', 'GANN_1X1_BREAKOUT_DOWN', 'NEGATIVE'),  # 向下突破
        ('indicators/volume/enhanced_obv.py', 'OBV_BREAKDOWN', 'NEGATIVE'),  # 向下突破
        
        # 超买超卖修正 - 确保反转逻辑正确
        ('indicators/stochrsi.py', 'STOCHRSI_OVERBOUGHT_REVERSAL', 'NEGATIVE'),  # 超买反转向下
        
        # 波动性修正
        ('indicators/atr.py', 'ATR_HIGH_VOLATILITY', 'NEGATIVE'),  # 高波动不利
        ('indicators/atr.py', 'ATR_MARKET_VOLATILE', 'NEGATIVE'),  # 高波动市场
        ('indicators/atr.py', 'ATR_MARKET_QUIET', 'POSITIVE'),  # 低波动有利
        
        # 排列形态修正
        ('indicators/wma.py', 'WMA_BEARISH_ARRANGEMENT', 'NEGATIVE'),  # 空头排列
        ('indicators/ma.py', 'MA_BEARISH_ARRANGEMENT', 'NEGATIVE'),  # 空头排列
        ('indicators/ema.py', 'EMA_BEARISH_ARRANGEMENT', 'NEGATIVE'),  # 空头排列
        
        # 机构行为修正
        ('indicators/institutional_behavior.py', 'INST_DISTRIBUTION_START', 'NEGATIVE'),  # 开始出货
        
        # 筹码分布修正 - 根据技术含义
        ('indicators/chip_distribution.py', 'CHIP_HIGH_CONCENTRATION', 'POSITIVE'),  # 高集中度利于控盘
        ('indicators/chip_distribution.py', 'CHIP_LOW_PROFIT', 'POSITIVE'),  # 低获利盘利于上涨
        
        # 成交量修正
        ('indicators/vol.py', 'VOL_BREAKOUT_DOWN', 'NEGATIVE'),  # 放量下跌
        
        # 价格目标修正为中性
        ('indicators/gann_tools.py', 'GANN_PRICE_TARGET_UP', 'NEUTRAL'),
        ('indicators/gann_tools.py', 'GANN_PRICE_TARGET_DOWN', 'NEUTRAL'),
    ]
    
    print("🔧 修正关键极性问题...")
    
    fixes_made = 0
    for file_path, pattern_id, correct_polarity in critical_fixes:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找并替换极性
                pattern_regex = rf'(register_pattern_to_registry\(\s*pattern_id=["\']' + re.escape(pattern_id) + r'["\'][^)]*polarity=["\'])([^"\']+)(["\'][^)]*\))'
                
                def replace_polarity(match):
                    return match.group(1) + correct_polarity + match.group(3)
                
                new_content = re.sub(pattern_regex, replace_polarity, content, flags=re.DOTALL)
                
                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    fixes_made += 1
                    print(f"  ✅ {pattern_id} -> {correct_polarity}")
                
            except Exception as e:
                print(f"  ❌ 修复 {pattern_id} 失败: {e}")
    
    print(f"\n📊 修复完成: {fixes_made} 个关键问题")
    return fixes_made

if __name__ == "__main__":
    print("🎯 基于技术分析原理修正极性标注")
    print("=" * 60)
    
    fixes = fix_critical_polarity_issues()
    
    if fixes > 0:
        print(f"\n🎉 成功修复 {fixes} 个关键极性问题！")
        print("建议重新运行技术分析验证脚本确认修复效果")
    else:
        print("\n⚠️  没有找到需要修复的问题，请检查文件路径和模式ID")
