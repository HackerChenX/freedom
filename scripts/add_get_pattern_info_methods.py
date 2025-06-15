#!/usr/bin/env python3
"""
批量为技术指标添加get_pattern_info方法的脚本
解决大量指标缺失get_pattern_info方法的ERROR问题
"""

import os
import re
import sys
from typing import Dict, List

def get_pattern_info_template(indicator_name: str) -> str:
    """
    生成get_pattern_info方法的模板代码
    
    Args:
        indicator_name: 指标名称
        
    Returns:
        str: 方法代码模板
    """
    return f'''
    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {{
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{{pattern_id}}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }}
        
        # {indicator_name}指标特定的形态信息映射
        pattern_info_map = {{
            # 基础形态
            "超买区域": {{
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            }},
            "超卖区域": {{
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            }},
            "中性区域": {{
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            }},
            # 趋势形态
            "上升趋势": {{
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            }},
            "下降趋势": {{
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            }},
            # 信号形态
            "买入信号": {{
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            }},
            "卖出信号": {{
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }}
        }}
        
        return pattern_info_map.get(pattern_id, default_pattern)
'''

def add_get_pattern_info_to_file(file_path: str, indicator_name: str) -> bool:
    """
    为指定文件中的指标类添加get_pattern_info方法
    
    Args:
        file_path: 文件路径
        indicator_name: 指标名称
        
    Returns:
        bool: 是否成功添加
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有get_pattern_info方法
        if 'def get_pattern_info(' in content:
            print(f"  ℹ️  {file_path} 已经有get_pattern_info方法")
            return False
        
        # 查找类定义的结束位置（通常在文件末尾或下一个类定义之前）
        # 寻找最后一个方法的结束位置
        method_pattern = r'(\n    def [^(]+\([^)]*\)[^:]*:.*?)(\n\n|\nclass|\n$|\Z)'
        matches = list(re.finditer(method_pattern, content, re.DOTALL))
        
        if not matches:
            print(f"  ❌ 无法找到合适的插入位置: {file_path}")
            return False
        
        # 在最后一个方法后插入新方法
        last_match = matches[-1]
        insert_pos = last_match.end(1)
        
        # 生成方法代码
        method_code = get_pattern_info_template(indicator_name)
        
        # 插入新方法
        new_content = content[:insert_pos] + method_code + content[insert_pos:]
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✅ 成功添加get_pattern_info方法到 {file_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ 处理文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("🔧 开始批量添加get_pattern_info方法...")
    
    # 需要添加方法的指标文件列表
    indicator_files = [
        # ZXM系列指标
        ('indicators/zxm/trend_indicators.py', ['ZXMWeeklyMACD', 'ZXMWeeklyTrendUp', 'ZXMWeeklyKDJDTrendUp', 
                                               'ZXMWeeklyKDJDOrDEATrendUp', 'ZXMMonthlyMACD', 'ZXMMonthlyKDJTrendUp',
                                               'ZXMDailyMACD', 'ZXMDailyTrendUp']),
        ('indicators/zxm/buy_point_indicators.py', ['ZXMBuyPointScore', 'ZXMBSAbsorb']),
        ('indicators/zxm/elasticity_indicators.py', ['ZXMRiseElasticity', 'ZXMElasticityScore']),
        ('indicators/zxm/market_breadth.py', ['ZXMVolumeShrink', 'ZXMTurnover']),
        ('indicators/zxm/selection_model.py', ['SelectionModel']),
        
        # 核心技术指标
        ('indicators/vortex.py', ['Vortex']),
        ('indicators/wr.py', ['WR']),
        ('indicators/vr.py', ['VR']),
        ('indicators/vosc.py', ['VOSC']),
        ('indicators/roc.py', ['ROC']),
        ('indicators/psy.py', ['PSY']),
        ('indicators/mtm.py', ['MTM']),
        ('indicators/mfi.py', ['MFI']),
        ('indicators/kc.py', ['KC']),
        ('indicators/momentum.py', ['Momentum']),
        ('indicators/ema.py', ['EMA']),
        ('indicators/atr.py', ['ATR']),
        ('indicators/aroon.py', ['Aroon']),
        
        # 增强指标
        ('indicators/enhanced_wr.py', ['EnhancedWR']),
        ('indicators/enhanced_mfi.py', ['EnhancedMFI']),
        ('indicators/enhanced_dmi.py', ['EnhancedDMI']),
        ('indicators/unified_ma.py', ['UnifiedMA']),
        ('indicators/trend_detector.py', ['TrendDetector']),
        ('indicators/stock_vix.py', ['StockVIX']),
        ('indicators/stock_score_calculator.py', ['StockScoreCalculator']),
        ('indicators/institutional_behavior.py', ['InstitutionalBehavior']),
        ('indicators/elasticity.py', ['Elasticity']),
        ('indicators/chip_distribution.py', ['ChipDistribution']),
        ('indicators/bounce_detector.py', ['BounceDetector']),
        ('indicators/amplitude_elasticity.py', ['AmplitudeElasticity']),
    ]
    
    total_files = 0
    success_count = 0
    
    for file_path, indicator_names in indicator_files:
        if os.path.exists(file_path):
            total_files += 1
            print(f"\n📁 处理文件: {file_path}")
            
            # 为每个指标名称尝试添加方法
            for indicator_name in indicator_names:
                if add_get_pattern_info_to_file(file_path, indicator_name):
                    success_count += 1
                    break  # 每个文件只添加一次
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print(f"\n📊 批量添加完成:")
    print(f"  总文件数: {total_files}")
    print(f"  成功添加数: {success_count}")
    print(f"  成功率: {(success_count/total_files)*100:.1f}%" if total_files > 0 else "  成功率: 0%")
    
    print(f"\n✅ get_pattern_info方法批量添加完成！")

if __name__ == "__main__":
    main()
