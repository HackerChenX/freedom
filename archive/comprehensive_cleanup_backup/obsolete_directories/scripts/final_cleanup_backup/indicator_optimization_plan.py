#!/usr/bin/env python3
"""
指标优化计划脚本
分析无选股指标的原因并制定优化策略
"""

import json
import os
from typing import Dict, List, Any

class IndicatorOptimizationPlan:
    def __init__(self):
        self.no_selection_indicators = {
            # 第1批-基础指标 (3个)
            '第1批-基础指标': ['macd', 'boll', 'kdj'],
            
            # 第2批-趋势指标 (1个)
            '第2批-趋势指标': ['sar'],
            
            # 第3批-成交量指标 (1个)
            '第3批-成交量指标': ['emv'],
            
            # 第4批-波动率指标 (1个)
            '第4批-波动率指标': ['stock_vix'],
            
            # 第5批-ZXM专业指标 (4个)
            '第5批-ZXM专业指标': ['amplitude_elasticity', 'rise_elasticity', 'elasticity_score', 'market_breadth'],
            
            # 第6批-增强指标 (9个)
            '第6批-增强指标': ['enhanced_rsi', 'enhanced_dmi', 'enhanced_macd_trend', 'enhanced_trix', 
                            'enhanced_kdj_osc', 'enhanced_obv', 'enhanced_stochrsi', 'enhanced_wr', 'enhanced_macd_root'],
            
            # 第7批-复合形态指标 (7个)
            '第7批-复合形态指标': ['composite', 'unified_ma', 'chip_distribution', 'institutional_behavior', 
                                'candlestick_patterns', 'advanced_candlestick', 'patterns'],
            
            # 第8批-工具公式指标 (5个)
            '第8批-工具公式指标': ['fibonacci_tools', 'gann_tools', 'elliott_wave', 'kdj_condition', 'macd_condition'],
            
            # 第9批-多周期指标 (5个)
            '第9批-多周期指标': ['monthly_kdj_trend_up', 'monthly_macd', 'weekly_kdj_d_or_dea_trend_up', 
                               'weekly_kdj_d_trend_up', 'weekly_macd'],
            
            # 第10批-震荡指标 (1个)
            '第10批-震荡指标': ['stochrsi']
        }
        
        self.optimization_strategies = {
            # 优化策略分类
            'parameter_relaxation': {
                'description': '参数放宽策略 - 降低选股条件的严格程度',
                'indicators': ['macd', 'boll', 'kdj', 'sar', 'enhanced_rsi', 'enhanced_dmi', 'enhanced_macd_trend', 
                             'enhanced_trix', 'enhanced_kdj_osc', 'enhanced_stochrsi', 'enhanced_wr', 'enhanced_macd_root',
                             'kdj_condition', 'macd_condition', 'stochrsi']
            },
            'stock_pool_expansion': {
                'description': '股票池扩展策略 - 增加测试股票数量',
                'indicators': ['monthly_kdj_trend_up', 'monthly_macd', 'weekly_kdj_d_or_dea_trend_up', 
                             'weekly_kdj_d_trend_up', 'weekly_macd']
            },
            'logic_adjustment': {
                'description': '逻辑调整策略 - 修改指标计算逻辑',
                'indicators': ['emv', 'stock_vix', 'amplitude_elasticity', 'rise_elasticity', 'elasticity_score', 
                             'market_breadth', 'enhanced_obv']
            },
            'pattern_threshold_adjustment': {
                'description': '形态阈值调整策略 - 降低形态识别阈值',
                'indicators': ['composite', 'unified_ma', 'chip_distribution', 'institutional_behavior', 
                             'candlestick_patterns', 'advanced_candlestick', 'patterns']
            },
            'tool_formula_activation': {
                'description': '工具公式激活策略 - 激活工具类指标的选股功能',
                'indicators': ['fibonacci_tools', 'gann_tools', 'elliott_wave']
            }
        }
    
    def generate_optimization_plan(self):
        """生成优化计划"""
        print("🎯 88个技术指标100%选股率优化计划")
        print("=" * 80)
        
        print(f"\n📊 当前状态分析:")
        print(f"   - 有选股能力: 51个指标 (58.0%)")
        print(f"   - 无选股能力: 37个指标 (42.0%)")
        print(f"   - 目标: 88个指标 (100.0%)")
        
        # 按优先级排序优化任务
        priority_batches = [
            ('第1批-基础指标', '高优先级 - 基础指标必须100%可用'),
            ('第2批-趋势指标', '高优先级 - 趋势分析核心'),
            ('第3批-成交量指标', '高优先级 - 量价分析重要'),
            ('第4批-波动率指标', '中优先级 - 风险控制相关'),
            ('第5批-ZXM专业指标', '高优先级 - 专业选股策略'),
            ('第10批-震荡指标', '中优先级 - 震荡市场重要'),
            ('第6批-增强指标', '中优先级 - 增强版需要调优'),
            ('第8批-工具公式指标', '低优先级 - 工具类指标'),
            ('第9批-多周期指标', '中优先级 - 需要更多数据'),
            ('第7批-复合形态指标', '低优先级 - 复杂形态识别')
        ]
        
        print(f"\n🚀 分阶段优化计划:")
        print("-" * 50)
        
        for i, (batch_name, priority_desc) in enumerate(priority_batches, 1):
            if batch_name in self.no_selection_indicators:
                indicators = self.no_selection_indicators[batch_name]
                if indicators:
                    print(f"\n阶段{i}: {batch_name}")
                    print(f"   优先级: {priority_desc}")
                    print(f"   待优化指标: {len(indicators)}个")
                    print(f"   指标列表: {indicators}")
                    
                    # 匹配优化策略
                    strategies = []
                    for strategy_name, strategy_info in self.optimization_strategies.items():
                        matching_indicators = [ind for ind in indicators if ind in strategy_info['indicators']]
                        if matching_indicators:
                            strategies.append((strategy_name, strategy_info['description'], matching_indicators))
                    
                    if strategies:
                        print(f"   优化策略:")
                        for strategy_name, description, matching_indicators in strategies:
                            print(f"     - {description}")
                            print(f"       应用于: {matching_indicators}")
        
        return self.generate_detailed_optimization_steps()
    
    def generate_detailed_optimization_steps(self):
        """生成详细的优化步骤"""
        print(f"\n🔧 详细优化步骤:")
        print("=" * 80)
        
        steps = [
            {
                'step': 1,
                'title': '扩大股票池测试',
                'description': '将测试股票池从5只扩展到50只，验证是否有更多股票符合条件',
                'indicators': 'all_no_selection',
                'expected_improvement': '30-50%',
                'time_estimate': '1小时'
            },
            {
                'step': 2,
                'title': '基础指标参数优化',
                'description': '调整MACD、BOLL、KDJ等基础指标的参数，降低选股阈值',
                'indicators': ['macd', 'boll', 'kdj'],
                'expected_improvement': '80-100%',
                'time_estimate': '2小时'
            },
            {
                'step': 3,
                'title': '增强指标逻辑修复',
                'description': '检查增强指标的计算逻辑，修复可能的bug',
                'indicators': ['enhanced_rsi', 'enhanced_dmi', 'enhanced_macd_trend', 'enhanced_trix', 
                             'enhanced_kdj_osc', 'enhanced_obv', 'enhanced_stochrsi', 'enhanced_wr', 'enhanced_macd_root'],
                'expected_improvement': '60-80%',
                'time_estimate': '3小时'
            },
            {
                'step': 4,
                'title': '多周期指标数据验证',
                'description': '验证多周期指标的数据获取和计算逻辑',
                'indicators': ['monthly_kdj_trend_up', 'monthly_macd', 'weekly_kdj_d_or_dea_trend_up', 
                             'weekly_kdj_d_trend_up', 'weekly_macd'],
                'expected_improvement': '70-90%',
                'time_estimate': '2小时'
            },
            {
                'step': 5,
                'title': '复合形态指标阈值调整',
                'description': '降低复合形态指标的识别阈值，提高触发概率',
                'indicators': ['composite', 'unified_ma', 'chip_distribution', 'institutional_behavior', 
                             'candlestick_patterns', 'advanced_candlestick', 'patterns'],
                'expected_improvement': '40-70%',
                'time_estimate': '4小时'
            },
            {
                'step': 6,
                'title': '工具公式指标激活',
                'description': '激活工具类指标的选股功能，添加默认选股逻辑',
                'indicators': ['fibonacci_tools', 'gann_tools', 'elliott_wave'],
                'expected_improvement': '90-100%',
                'time_estimate': '2小时'
            }
        ]
        
        total_time = 0
        for step in steps:
            print(f"\n步骤{step['step']}: {step['title']}")
            print(f"   描述: {step['description']}")
            print(f"   涉及指标: {len(step['indicators']) if isinstance(step['indicators'], list) else '全部无选股指标'}")
            print(f"   预期改善: {step['expected_improvement']}")
            print(f"   预估时间: {step['time_estimate']}")
            
            if step['time_estimate'].endswith('小时'):
                total_time += int(step['time_estimate'].replace('小时', ''))
        
        print(f"\n⏱️ 总预估时间: {total_time}小时")
        print(f"🎯 预期最终结果: 88个指标100%选股率")
        
        return steps

if __name__ == "__main__":
    planner = Indicator_optimization_plan()
    planner.generate_optimization_plan() 