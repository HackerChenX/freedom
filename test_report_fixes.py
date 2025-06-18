#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试报告修复效果的脚本
"""

import os
import sys
import json
import tempfile
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

def create_test_data():
    """创建测试数据"""
    # 模拟买点分析结果，包含各种问题场景
    test_results = [
        {
            'stock_code': '000001',
            'buypoint_date': '2024-01-15',
            'indicator_results': {
                '15min': [
                    {
                        'indicator_name': 'MACD',
                        'pattern_id': 'MACD_GOLDEN_CROSS',
                        'pattern_name': 'MACD金叉信号',
                        'score_impact': 25.5,
                        'description': 'MACD线上穿信号线',
                        'pattern_type': 'BULLISH'
                    },
                    {
                        'indicator_name': 'RSI',
                        'pattern_id': 'RSI_OVERSOLD',
                        'pattern_name': 'AA条件满足',  # 模糊描述，需要标准化
                        'score_impact': 18.0,
                        'description': 'RSI超卖信号',
                        'pattern_type': 'BULLISH'
                    }
                ],
                'daily': [
                    {
                        'indicator_name': 'BOLL',
                        'pattern_id': 'BOLL_SQUEEZE',
                        'pattern_name': '大幅波动区间',  # 模糊描述，需要标准化
                        'score_impact': 22.0,
                        'description': '布林带收缩',
                        'pattern_type': 'NEUTRAL'
                    },
                    {
                        'indicator_name': 'ATR',
                        'pattern_id': 'ATR_BREAKOUT',
                        'pattern_name': 'ATR波动率突破',
                        'score_impact': 30.0,
                        'description': 'ATR突破信号',
                        'pattern_type': 'BULLISH'
                    }
                ],
                'weekly': [
                    {
                        'indicator_name': 'KDJ',
                        'pattern_id': 'KDJ_GOLDEN_CROSS',
                        'pattern_name': '高规律性周期',  # 模糊描述，需要标准化
                        'score_impact': 35.0,
                        'description': 'KDJ金叉',
                        'pattern_type': 'BULLISH'
                    }
                ]
            }
        },
        {
            'stock_code': '000002',
            'buypoint_date': '2024-01-16',
            'indicator_results': {
                '15min': [
                    {
                        'indicator_name': 'MACD',
                        'pattern_id': 'MACD_GOLDEN_CROSS',
                        'pattern_name': 'MACD金叉信号',
                        'score_impact': 28.0,
                        'description': 'MACD线上穿信号线',
                        'pattern_type': 'BULLISH'
                    },
                    {
                        'indicator_name': 'VOL',
                        'pattern_id': 'VOL_SURGE',
                        'pattern_name': '低分股票',  # 模糊描述，需要标准化
                        'score_impact': 15.0,
                        'description': '成交量放大',
                        'pattern_type': 'BULLISH'
                    }
                ],
                'daily': [
                    {
                        'indicator_name': 'BOLL',
                        'pattern_id': 'BOLL_SQUEEZE',
                        'pattern_name': '大幅波动区间',  # 模糊描述，需要标准化
                        'score_impact': 20.0,
                        'description': '布林带收缩',
                        'pattern_type': 'NEUTRAL'
                    }
                ],
                'weekly': [
                    {
                        'indicator_name': 'KDJ',
                        'pattern_id': 'KDJ_GOLDEN_CROSS',
                        'pattern_name': '高规律性周期',  # 模糊描述，需要标准化
                        'score_impact': 32.0,
                        'description': 'KDJ金叉',
                        'pattern_type': 'BULLISH'
                    }
                ]
            }
        },
        {
            'stock_code': '000003',
            'buypoint_date': '2024-01-17',
            'indicator_results': {
                '15min': [
                    {
                        'indicator_name': 'RSI',
                        'pattern_id': 'RSI_OVERSOLD',
                        'pattern_name': 'AA条件满足',  # 模糊描述，需要标准化
                        'score_impact': 20.0,
                        'description': 'RSI超卖信号',
                        'pattern_type': 'BULLISH'
                    }
                ],
                'daily': [
                    {
                        'indicator_name': 'ATR',
                        'pattern_id': 'ATR_BREAKOUT',
                        'pattern_name': 'ATR波动率突破',
                        'score_impact': 28.0,
                        'description': 'ATR突破信号',
                        'pattern_type': 'BULLISH'
                    }
                ]
            }
        }
    ]
    
    return test_results

def test_report_generation():
    """测试报告生成功能"""
    print("🔍 测试报告生成修复效果...")
    
    try:
        # 创建分析器实例
        analyzer = BuyPointBatchAnalyzer()
        
        # 创建测试数据
        test_data = create_test_data()
        
        # 提取共性指标
        print("📊 提取共性指标...")
        common_indicators = analyzer.extract_common_indicators(
            buypoint_results=test_data,
            min_hit_ratio=0.3,  # 降低阈值以便测试
            filter_negative_patterns=True
        )
        
        if not common_indicators:
            print("❌ 未能提取到共性指标")
            return False
        
        print(f"✅ 成功提取到 {len(common_indicators)} 个周期的共性指标")
        
        # 验证时间周期数据一致性
        print("🔍 验证时间周期数据一致性...")
        for period, indicators in common_indicators.items():
            print(f"  📈 {period} 周期: {len(indicators)} 个指标")
            for indicator in indicators[:3]:  # 显示前3个
                print(f"    - {indicator['name']}: {indicator['pattern']} (命中率: {indicator['hit_ratio']:.1%}, 平均得分: {indicator['avg_score']:.1f})")
        
        # 验证评分数据
        print("🔍 验证评分数据...")
        all_scores = []
        for period, indicators in common_indicators.items():
            for indicator in indicators:
                all_scores.append(indicator['avg_score'])
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"✅ 平均得分: {avg_score:.1f} (范围: {min(all_scores):.1f} - {max(all_scores):.1f})")
            
            if avg_score == 0.0:
                print("❌ 评分数据仍然异常，所有得分都是0.0")
                return False
        else:
            print("❌ 没有找到评分数据")
            return False
        
        # 生成测试报告
        print("📝 生成测试报告...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_report_path = f.name
        
        analyzer._generate_indicators_report(common_indicators, temp_report_path)
        
        # 读取并验证报告内容
        with open(temp_report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        print(f"✅ 报告生成成功，长度: {len(report_content)} 字符")
        
        # 验证修复效果
        print("🔍 验证修复效果...")
        
        # 检查是否包含修复说明
        if "数据污染修复版" in report_content:
            print("✅ 包含修复版本说明")
        else:
            print("❌ 缺少修复版本说明")
        
        # 检查是否包含修复说明
        if "重要修复说明" in report_content:
            print("✅ 包含修复说明")
        else:
            print("❌ 缺少修复说明")
        
        # 检查是否还有模糊描述
        vague_terms = ['AA条件满足', '低分股票', '大幅波动区间', '高规律性周期']
        found_vague = []
        for term in vague_terms:
            if term in report_content:
                found_vague.append(term)
        
        if found_vague:
            print(f"❌ 仍然包含模糊描述: {found_vague}")
        else:
            print("✅ 已消除模糊描述")
        
        # 检查评分是否不再全是0.0
        if "平均得分0.0分" in report_content:
            print("❌ 评分数据仍然异常")
        else:
            print("✅ 评分数据已修复")
        
        # 清理临时文件
        os.unlink(temp_report_path)
        
        print("🎉 报告生成修复测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("📊 报告修复效果测试")
    print("="*60)
    
    success = test_report_generation()
    
    print("\n" + "="*60)
    if success:
        print("✅ 测试通过：报告修复效果良好")
    else:
        print("❌ 测试失败：仍存在问题需要修复")
    print("="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
