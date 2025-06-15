#!/usr/bin/env python3
"""
为ZXM指标批量添加get_pattern_info方法的脚本
专门处理ZXM系列指标缺失get_pattern_info方法的问题
"""

import os
import re
import sys

def add_get_pattern_info_to_zxm_class(file_path: str, class_name: str) -> bool:
    """
    为ZXM指标类添加get_pattern_info方法
    
    Args:
        file_path: 文件路径
        class_name: 类名
        
    Returns:
        bool: 是否成功添加
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有get_pattern_info方法
        if f'def get_pattern_info(' in content:
            print(f"  ℹ️  {class_name} 已经有get_pattern_info方法")
            return False
        
        # 生成get_pattern_info方法代码
        method_code = f'''
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
        
        # {class_name}指标特定的形态信息映射
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
        
        # 在文件末尾添加方法
        new_content = content + method_code
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✅ 成功添加get_pattern_info方法到 {class_name}")
        return True
        
    except Exception as e:
        print(f"  ❌ 处理类 {class_name} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("🔧 开始为ZXM指标添加get_pattern_info方法...")
    
    # 需要添加方法的ZXM指标类列表
    zxm_classes = [
        ('indicators/zxm/trend_indicators.py', [
            'ZXMMonthlyMACD', 'ZXMWeeklyTrendUp', 'ZXMWeeklyKDJDTrendUp', 
            'ZXMWeeklyKDJDOrDEATrendUp', 'ZXMMonthlyKDJTrendUp', 'TrendDetector'
        ]),
        ('indicators/zxm/buy_point_indicators.py', ['ZXMBuyPointScore', 'ZXMBSAbsorb']),
        ('indicators/zxm/elasticity_indicators.py', ['ZXMRiseElasticity', 'ZXMElasticityScore']),
        ('indicators/zxm/market_breadth.py', ['ZXMVolumeShrink', 'ZXMTurnover']),
        ('indicators/zxm/selection_model.py', ['SelectionModel']),
        ('indicators/zxm/diagnostics.py', ['ZXMDiagnostics']),
        ('indicators/zxm/score_indicators.py', ['ZXMScoreIndicator']),
        ('indicators/zxm/patterns.py', ['ZXMPatternIndicator']),
        ('indicators/enhanced_dmi.py', ['EnhancedDMI']),
        ('indicators/psy.py', ['PSY']),
    ]
    
    total_classes = 0
    success_count = 0
    
    for file_path, class_names in zxm_classes:
        if os.path.exists(file_path):
            print(f"\n📁 处理文件: {file_path}")
            
            for class_name in class_names:
                total_classes += 1
                if add_get_pattern_info_to_zxm_class(file_path, class_name):
                    success_count += 1
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print(f"\n📊 ZXM指标get_pattern_info方法添加完成:")
    print(f"  总类数: {total_classes}")
    print(f"  成功添加数: {success_count}")
    print(f"  成功率: {(success_count/total_classes)*100:.1f}%" if total_classes > 0 else "  成功率: 0%")
    
    print(f"\n✅ ZXM指标get_pattern_info方法批量添加完成！")

if __name__ == "__main__":
    main()
