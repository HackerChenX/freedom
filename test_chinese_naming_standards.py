#!/usr/bin/env python3
"""
中文命名标准验证测试 - 检查所有技术指标的中文命名是否符合标准
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry
from analysis.buypoints.buypoint_batch_analyzer import COMPLETE_INDICATOR_PATTERNS_MAP
import re

# 设置日志级别
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class ChineseNamingValidator:
    """中文命名标准验证器"""
    
    def __init__(self):
        self.registry = PatternRegistry()
        
        # 定义中文命名标准
        self.naming_standards = {
            # 技术指标前缀标准
            'indicator_prefixes': {
                'KDJ': 'KDJ',
                'RSI': 'RSI', 
                'MACD': 'MACD',
                'BOLL': 'BOLL',
                'MA': 'MA',
                'EMA': 'EMA',
                'TRIX': 'TRIX',
                'ROC': 'ROC',
                'CMO': 'CMO',
                'VOL': '成交量',
                'ATR': 'ATR',
                'KC': 'KC',
                'MFI': 'MFI',
                'Vortex': '涡旋指标',
                'OBV': 'OBV',
                'PSY': 'PSY心理线',
                'WR': 'WR',
                'CCI': 'CCI',
                'DMI': 'DMI',
                'ADX': 'ADX',
                'SAR': 'SAR'
            },
            
            # 技术术语标准
            'technical_terms': {
                '金叉': '指标线上穿信号线，看涨信号',
                '死叉': '指标线下穿信号线，看跌信号',
                '超买': '指标值过高，可能面临回调',
                '超卖': '指标值过低，可能出现反弹',
                '零轴上方': '指标位于零轴上方，多头占优',
                '零轴下方': '指标位于零轴下方，空头占优',
                '上升趋势': '指标呈上升趋势',
                '下降趋势': '指标呈下降趋势',
                '多头排列': '短期指标在长期指标上方',
                '空头排列': '短期指标在长期指标下方',
                '背离': '指标与价格走势不一致',
                '突破': '价格或指标突破关键位置',
                '支撑': '价格获得支撑',
                '阻力': '价格遇到阻力'
            },
            
            # 禁用的模糊术语
            'forbidden_terms': [
                '技术形态', '未知形态', '中等股票', '大幅波动区间',
                'AA条件满足', '技术指标分析', '一般形态', '普通信号'
            ]
        }
    
    def is_chinese_text(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def has_forbidden_terms(self, text: str) -> list:
        """检查是否包含禁用的模糊术语"""
        found_terms = []
        for term in self.naming_standards['forbidden_terms']:
            if term in text:
                found_terms.append(term)
        return found_terms
    
    def validate_pattern_name(self, indicator_name: str, pattern_id: str, display_name: str, description: str) -> dict:
        """
        验证单个形态名称的中文命名标准
        
        Returns:
            dict: 验证结果
        """
        result = {
            'indicator_name': indicator_name,
            'pattern_id': pattern_id,
            'display_name': display_name,
            'description': description,
            'issues': [],
            'score': 100  # 满分100分
        }
        
        # 1. 检查是否使用中文
        if not self.is_chinese_text(display_name):
            result['issues'].append('形态名称未使用中文')
            result['score'] -= 30
        
        if not self.is_chinese_text(description):
            result['issues'].append('形态描述未使用中文')
            result['score'] -= 20
        
        # 2. 检查是否包含禁用术语
        forbidden_in_name = self.has_forbidden_terms(display_name)
        if forbidden_in_name:
            result['issues'].append(f'形态名称包含禁用术语: {", ".join(forbidden_in_name)}')
            result['score'] -= 25
        
        forbidden_in_desc = self.has_forbidden_terms(description)
        if forbidden_in_desc:
            result['issues'].append(f'形态描述包含禁用术语: {", ".join(forbidden_in_desc)}')
            result['score'] -= 15
        
        # 3. 检查是否使用了标准技术术语
        has_standard_terms = False
        for term in self.naming_standards['technical_terms'].keys():
            if term in display_name or term in description:
                has_standard_terms = True
                break
        
        if not has_standard_terms:
            result['issues'].append('未使用标准技术术语')
            result['score'] -= 10
        
        # 4. 检查指标前缀是否标准
        expected_prefix = self.naming_standards['indicator_prefixes'].get(indicator_name)
        if expected_prefix and not display_name.startswith(expected_prefix):
            result['issues'].append(f'形态名称缺少标准前缀: {expected_prefix}')
            result['score'] -= 10
        
        # 5. 检查名称长度是否合适（4-12个字符）
        if len(display_name) < 4:
            result['issues'].append('形态名称过短')
            result['score'] -= 5
        elif len(display_name) > 12:
            result['issues'].append('形态名称过长')
            result['score'] -= 5
        
        return result
    
    def validate_pattern_registry(self) -> dict:
        """验证PatternRegistry中的形态命名"""
        print("\n=== 验证PatternRegistry中的形态命名 ===")
        
        # 获取所有已注册的指标
        indicators = ['KDJ', 'RSI', 'TRIX', 'ROC', 'CMO', 'VOL', 'ATR', 'KC', 'MFI', 'Vortex']
        
        all_results = []
        total_patterns = 0
        total_score = 0
        
        for indicator_name in indicators:
            patterns = self.registry.get_patterns_by_indicator(indicator_name)
            
            if isinstance(patterns, list):
                for pattern_info in patterns:
                    if isinstance(pattern_info, dict):
                        pattern_id = pattern_info.get('pattern_id', 'N/A')
                        display_name = pattern_info.get('display_name', '')
                        description = pattern_info.get('description', '')
                        
                        result = self.validate_pattern_name(indicator_name, pattern_id, display_name, description)
                        all_results.append(result)
                        total_patterns += 1
                        total_score += result['score']
                        
                        # 显示验证结果
                        if result['issues']:
                            print(f"⚠️ {indicator_name} - {pattern_id}: {display_name}")
                            for issue in result['issues']:
                                print(f"   - {issue}")
                        else:
                            print(f"✅ {indicator_name} - {pattern_id}: {display_name}")
        
        average_score = total_score / total_patterns if total_patterns > 0 else 0
        
        return {
            'source': 'PatternRegistry',
            'total_patterns': total_patterns,
            'average_score': average_score,
            'results': all_results
        }
    
    def validate_centralized_mapping(self) -> dict:
        """验证集中式映射中的形态命名"""
        print("\n=== 验证集中式映射中的形态命名 ===")
        
        all_results = []
        total_patterns = 0
        total_score = 0
        
        for indicator_name, patterns in COMPLETE_INDICATOR_PATTERNS_MAP.items():
            if isinstance(patterns, dict):
                for pattern_id, pattern_info in patterns.items():
                    if isinstance(pattern_info, dict):
                        display_name = pattern_info.get('name', '')
                        description = pattern_info.get('description', '')
                        
                        result = self.validate_pattern_name(indicator_name, pattern_id, display_name, description)
                        all_results.append(result)
                        total_patterns += 1
                        total_score += result['score']
                        
                        # 显示验证结果
                        if result['issues']:
                            print(f"⚠️ {indicator_name} - {pattern_id}: {display_name}")
                            for issue in result['issues']:
                                print(f"   - {issue}")
                        else:
                            print(f"✅ {indicator_name} - {pattern_id}: {display_name}")
        
        average_score = total_score / total_patterns if total_patterns > 0 else 0
        
        return {
            'source': 'CentralizedMapping',
            'total_patterns': total_patterns,
            'average_score': average_score,
            'results': all_results
        }

def main():
    """主测试函数"""
    print("开始中文命名标准验证测试...")
    
    validator = ChineseNamingValidator()
    
    # 验证PatternRegistry
    registry_results = validator.validate_pattern_registry()
    
    # 验证集中式映射
    mapping_results = validator.validate_centralized_mapping()
    
    # 汇总结果
    print("\n" + "="*60)
    print("中文命名标准验证结果汇总")
    print("="*60)
    
    print(f"\n📊 PatternRegistry验证结果:")
    print(f"   总形态数量: {registry_results['total_patterns']}")
    print(f"   平均得分: {registry_results['average_score']:.1f}/100")
    
    print(f"\n📊 集中式映射验证结果:")
    print(f"   总形态数量: {mapping_results['total_patterns']}")
    print(f"   平均得分: {mapping_results['average_score']:.1f}/100")
    
    # 计算总体得分
    total_patterns = registry_results['total_patterns'] + mapping_results['total_patterns']
    if total_patterns > 0:
        overall_score = (
            registry_results['average_score'] * registry_results['total_patterns'] +
            mapping_results['average_score'] * mapping_results['total_patterns']
        ) / total_patterns
        
        print(f"\n🎯 总体评估:")
        print(f"   总形态数量: {total_patterns}")
        print(f"   总体平均得分: {overall_score:.1f}/100")
        
        if overall_score >= 90:
            print("   评级: 优秀 ✅")
        elif overall_score >= 80:
            print("   评级: 良好 ⚠️")
        elif overall_score >= 70:
            print("   评级: 及格 ⚠️")
        else:
            print("   评级: 需要改进 ❌")
        
        return overall_score >= 80
    else:
        print("⚠️ 未找到任何形态进行验证")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
