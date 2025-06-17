#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Fix Polarity Annotations
Automatically fixes missing polarity annotations in indicator files
"""

import os
import re
from typing import Dict, List, Tuple

def determine_polarity(pattern_id: str, pattern_type: str, score_impact: float) -> str:
    """Determine the correct polarity based on pattern analysis"""
    
    pattern_lower = pattern_id.lower()
    
    # Strong bullish indicators
    bullish_keywords = [
        'golden', 'cross_up', 'cross_above', 'breakout', 'break_up', 'bullish', 'rising', 'uptrend', 
        'above', 'strong_up', 'buy', 'accumulation', 'bottom', 'oversold', 'support', 'bounce',
        'ascending', 'rally', 'absorption', 'completion', 'start', 'positive', 'high_profit',
        'low_trapped', 'near_cost', 'main_wave', 'easy_untrapped', 'low_volatility', 'downtrend_vix'
    ]
    
    # Strong bearish indicators  
    bearish_keywords = [
        'death', 'cross_down', 'cross_below', 'breakdown', 'break_down', 'bearish', 'falling', 
        'downtrend', 'below', 'strong_down', 'sell', 'distribution', 'top', 'overbought', 
        'resistance', 'descending', 'washout', 'concentrated', 'negative', 'high_trapped',
        'far_above_cost', 'hard_untrapped', 'high_volatility', 'uptrend_vix', 'reversal_bottom'
    ]
    
    # Neutral indicators
    neutral_keywords = [
        'neutral', 'consolidation', 'time', 'cycle', 'volume_confirmation', 'trend_alignment',
        'fibonacci_ratio', 'completion', 'confirmation', 'anomaly', 'extreme', 'wide', 'narrow',
        'doji', 'island', 'triangle', 'cup', 'handle', 'convergence', 'shrink', 'range'
    ]
    
    # Check for semantic clues in pattern ID
    for keyword in bullish_keywords:
        if keyword in pattern_lower:
            return 'POSITIVE'
    
    for keyword in bearish_keywords:
        if keyword in pattern_lower:
            return 'NEGATIVE'
    
    for keyword in neutral_keywords:
        if keyword in pattern_lower:
            return 'NEUTRAL'
    
    # Fall back to pattern type and score
    if pattern_type.upper() == 'BULLISH' or score_impact > 0:
        return 'POSITIVE'
    elif pattern_type.upper() == 'BEARISH' or score_impact < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def extract_pattern_info(content: str, pattern_id: str) -> Tuple[str, float]:
    """Extract pattern type and score impact from pattern registration"""
    
    # Find the pattern registration block
    pattern_regex = rf'register_pattern_to_registry\(\s*pattern_id=["\']' + re.escape(pattern_id) + r'["\'][^)]*\)'
    match = re.search(pattern_regex, content, re.DOTALL)
    
    if not match:
        return 'NEUTRAL', 0.0
    
    block = match.group(0)
    
    # Extract pattern_type
    type_match = re.search(r'pattern_type=["\']([^"\']+)["\']', block)
    pattern_type = type_match.group(1) if type_match else 'NEUTRAL'
    
    # Extract score_impact
    score_match = re.search(r'score_impact=([+-]?\d+\.?\d*)', block)
    score_impact = float(score_match.group(1)) if score_match else 0.0
    
    return pattern_type, score_impact

def fix_file_polarity(file_path: str) -> int:
    """Fix missing polarity annotations in a single file"""
    
    if not os.path.exists(file_path):
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all register_pattern_to_registry calls without polarity
        pattern_regex = r'register_pattern_to_registry\(\s*pattern_id=["\']([^"\']+)["\'][^)]*\)'
        matches = re.findall(pattern_regex, content, re.DOTALL)
        
        patterns_without_polarity = []
        for pattern_id in matches:
            # Check if this pattern registration has polarity
            pattern_block_regex = rf'register_pattern_to_registry\(\s*pattern_id=["\']' + re.escape(pattern_id) + r'["\'][^)]*polarity[^)]*\)'
            if not re.search(pattern_block_regex, content, re.DOTALL):
                patterns_without_polarity.append(pattern_id)
        
        if not patterns_without_polarity:
            return 0
        
        print(f"Fixing {len(patterns_without_polarity)} patterns in {file_path}")
        
        # Fix each pattern
        modified_content = content
        fixes_made = 0
        
        for pattern_id in patterns_without_polarity:
            # Extract pattern info
            pattern_type, score_impact = extract_pattern_info(content, pattern_id)
            
            # Determine correct polarity
            polarity = determine_polarity(pattern_id, pattern_type, score_impact)
            
            # Find the pattern registration and add polarity
            pattern_block_regex = rf'(register_pattern_to_registry\(\s*pattern_id=["\']' + re.escape(pattern_id) + r'["\'][^)]*score_impact=[+-]?\d+\.?\d*)\s*\)'
            
            def add_polarity(match):
                return match.group(1) + f',\n            polarity="{polarity}"\n        )'
            
            new_content = re.sub(pattern_block_regex, add_polarity, modified_content, flags=re.DOTALL)
            
            if new_content != modified_content:
                modified_content = new_content
                fixes_made += 1
                print(f"  ✅ {pattern_id} -> {polarity}")
            else:
                print(f"  ❌ Failed to fix {pattern_id}")
        
        # Write back the modified content
        if fixes_made > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
        
        return fixes_made
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def batch_fix_polarity():
    """Batch fix polarity annotations for all indicator files"""
    
    # Files that need fixing based on our detailed check
    files_to_fix = [
        'indicators/vortex.py',
        'indicators/ma.py',
        'indicators/kdj.py',
        'indicators/zxm_washplate.py',
        'indicators/vosc.py',
        'indicators/stochrsi.py',
        'indicators/ema.py',
        'indicators/cmo.py',
        'indicators/roc.py',
        'indicators/dma.py',
        'indicators/gann_tools.py',
        'indicators/institutional_behavior.py',
        'indicators/psy.py',
        'indicators/ichimoku.py',
        'indicators/unified_ma.py',
        'indicators/volume_ratio.py',
        'indicators/wr.py',
        'indicators/mtm.py',
        'indicators/vr.py',
        'indicators/stock_vix.py',
        'indicators/kc.py',
        'indicators/rsi.py',
        'indicators/trix.py',
        'indicators/momentum.py',
        'indicators/trend/enhanced_dmi.py',
        'indicators/trend/enhanced_macd.py',
        'indicators/oscillator/enhanced_kdj.py',
        'indicators/pattern/candlestick_patterns.py',
        'indicators/pattern/zxm_patterns.py',
        'indicators/pattern/advanced_candlestick_patterns.py',
    ]
    
    print("🔧 BATCH FIXING POLARITY ANNOTATIONS")
    print("=" * 60)
    
    total_fixes = 0
    files_fixed = 0
    
    for file_path in files_to_fix:
        fixes = fix_file_polarity(file_path)
        if fixes > 0:
            total_fixes += fixes
            files_fixed += 1
    
    print(f"\n📊 BATCH FIX SUMMARY:")
    print(f"- Files processed: {len(files_to_fix)}")
    print(f"- Files fixed: {files_fixed}")
    print(f"- Total patterns fixed: {total_fixes}")
    
    return total_fixes

if __name__ == "__main__":
    total_fixes = batch_fix_polarity()
    
    if total_fixes > 0:
        print(f"\n🎉 Successfully fixed {total_fixes} missing polarity annotations!")
        print("Running verification...")
        
        # Run verification
        os.system("python detailed_indicator_check.py | grep 'Coverage:'")
    else:
        print(f"\n⚠️  No fixes were made. Check if files exist and have the expected format.")
