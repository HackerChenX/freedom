#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
买点分析与选股策略集成使用示例

展示如何使用BuyPointToStrategyAdapter实现买点分析结果与选股策略的无缝集成
"""

import sys
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from analysis.buypoints.buypoint_strategy_adapter import get_buypoint_strategy_adapter
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from strategy.strategy_executor import StrategyExecutor
from utils.logger import get_logger

logger = get_logger(__name__)


class BuyPointStrategyIntegrationExample:
    """买点分析与选股策略集成使用示例"""
    
    def __init__(self):
        """初始化示例"""
        self.adapter = get_buypoint_strategy_adapter()
        self.buypoint_analyzer = BuyPointBatchAnalyzer()
        self.strategy_executor = StrategyExecutor()
        
        logger.info("买点分析与选股策略集成示例初始化完成")
    
    def example_1_single_stock_integration(self):
        """示例1: 单只股票的买点分析到选股策略集成"""
        print("=" * 60)
        print("示例1: 单只股票买点分析集成")
        print("=" * 60)
        
        stock_code = '000001'
        buypoint_date = '20240601'
        
        try:
            # 步骤1: 执行买点分析
            print(f"步骤1: 分析股票 {stock_code} 的买点...")
            
            # 创建模拟买点分析结果（实际使用中这里会调用真实的买点分析）
            buypoint_result = self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example(stock_code, buypoint_date)
            print(f"✅ 买点分析完成，总体评分: {buypoint_result['summary']['overall_score']}")
            
            # 步骤2: 转换为选股策略格式
            print(f"步骤2: 转换为选股策略格式...")
            strategy_result = self.adapter.convert_buypoint_result(buypoint_result)
            
            if strategy_result:
                print(f"✅ 格式转换成功")
                print(f"   股票名称: {strategy_result['stock_name']}")
                print(f"   行业: {strategy_result['industry']}")
                print(f"   当前价格: {strategy_result['price']:.2f}")
                print(f"   统一评分: {strategy_result['score']:.1f}")
                print(f"   推荐等级: {strategy_result['recommendation']}")
                print(f"   通过指标: {len(strategy_result['match_details']['passing_indicators'])}个")
                
                # 步骤3: 展示选股策略兼容性
                print(f"步骤3: 验证选股策略兼容性...")
                self._validate_strategy_compatibility_Buypoint_Strategy_Integration_Example(strategy_result)
                
            else:
                print("❌ 格式转换失败")
                
        except Exception as e:
            print(f"❌ 示例执行失败: {e}")
    
    def example_2_batch_integration(self):
        """示例2: 批量买点分析结果集成"""
        print("\n" + "=" * 60)
        print("示例2: 批量买点分析集成")
        print("=" * 60)
        
        try:
            # 步骤1: 准备批量买点分析结果
            print("步骤1: 准备批量买点分析结果...")
            
            stock_codes = ['000001', '000002', '600000', '600036', '000858']
            buypoint_results = []
            
            for stock_code in stock_codes:
                result = self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example(stock_code, '20240601')
                buypoint_results.append(result)
            
            print(f"✅ 准备了 {len(buypoint_results)} 个买点分析结果")
            
            # 步骤2: 批量转换为选股策略格式
            print("步骤2: 批量转换为选股策略格式...")
            strategy_df = self.adapter.convert_batch_results(buypoint_results)
            
            if not strategy_df.empty:
                print(f"✅ 批量转换成功，转换了 {len(strategy_df)} 只股票")
                
                # 显示转换结果摘要
                print("\n📊 转换结果摘要:")
                print(f"   平均评分: {strategy_df['score'].mean():.1f}")
                print(f"   最高评分: {strategy_df['score'].max():.1f} ({strategy_df.loc[strategy_df['score'].idxmax(), 'stock_name']})")
                print(f"   推荐买入: {len(strategy_df[strategy_df['recommendation'] == 'buy'])} 只")
                print(f"   推荐持有: {len(strategy_df[strategy_df['recommendation'] == 'hold'])} 只")
                print(f"   推荐卖出: {len(strategy_df[strategy_df['recommendation'] == 'sell'])} 只")
                
                # 显示详细结果
                print("\n📋 详细结果:")
                for _, row in strategy_df.iterrows():
                    print(f"   {row['stock_name']} ({row['stock_code']}): "
                          f"评分 {row['score']:.1f}, 推荐 {row['recommendation']}")
                
            else:
                print("❌ 批量转换失败")
                
        except Exception as e:
            print(f"❌ 示例执行失败: {e}")
    
    def example_3_strategy_workflow_integration(self):
        """示例3: 完整的选股策略工作流集成"""
        print("\n" + "=" * 60)
        print("示例3: 完整选股策略工作流集成")
        print("=" * 60)
        
        try:
            # 步骤1: 模拟买点分析批量处理
            print("步骤1: 执行买点分析批量处理...")
            
            # 模拟从买点分析系统获取结果
            buypoint_results = [
                self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example('000001', '20240601', score=85.0),
                self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example('000002', '20240601', score=72.5),
                self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example('600000', '20240601', score=68.0),
                self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example('600036', '20240601', score=91.2),
                self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example('000858', '20240601', score=45.5)
            ]
            
            print(f"✅ 获取了 {len(buypoint_results)} 个买点分析结果")
            
            # 步骤2: 转换为选股策略格式
            print("步骤2: 转换为选股策略格式...")
            strategy_df = self.adapter.convert_batch_results(buypoint_results)
            
            # 步骤3: 应用选股策略筛选
            print("步骤3: 应用选股策略筛选...")
            
            # 筛选条件：评分 >= 70 且推荐为 buy
            filtered_df = strategy_df[
                (strategy_df['score'] >= 70) & 
                (strategy_df['recommendation'] == 'buy')
            ].copy()
            
            print(f"✅ 筛选出 {len(filtered_df)} 只符合条件的股票")
            
            # 步骤4: 生成最终选股结果
            print("步骤4: 生成最终选股结果...")
            
            if not filtered_df.empty:
                # 按评分排序
                final_selection = filtered_df.sort_values('score', ascending=False)
                
                print("\n🎯 最终选股结果:")
                print("=" * 50)
                
                for i, (_, row) in enumerate(final_selection.iterrows(), 1):
                    print(f"{i}. {row['stock_name']} ({row['stock_code']})")
                    print(f"   评分: {row['score']:.1f}/100")
                    print(f"   推荐: {row['recommendation'].upper()}")
                    print(f"   当前价格: ¥{row['price']:.2f}")
                    print(f"   涨跌幅: {row['change_pct']:+.2f}%")
                    print(f"   通过指标: {len(row['match_details']['passing_indicators'])}个")
                    print(f"   来源: 买点分析系统")
                    print()
                
                # 保存结果
                output_file = f"selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                final_selection.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"📄 选股结果已保存: {output_file}")
                
            else:
                print("⚠️ 没有股票符合筛选条件")
                
        except Exception as e:
            print(f"❌ 示例执行失败: {e}")
    
    def example_4_custom_scoring_integration(self):
        """示例4: 自定义评分权重集成"""
        print("\n" + "=" * 60)
        print("示例4: 自定义评分权重集成")
        print("=" * 60)
        
        try:
            # 创建自定义权重的适配器
            custom_adapter = get_buypoint_strategy_adapter()
            
            # 修改权重配置（更重视吸筹信号）
            custom_adapter.strategy_weights.update({
                "CUSTOM_ABSORPTION": 1.2,  # 提高吸筹信号权重
                "MACD": 0.8,               # 降低MACD权重
                "KDJ": 0.7                 # 降低KDJ权重
            })
            
            print("✅ 自定义权重配置完成（更重视吸筹信号）")
            
            # 使用相同的买点分析结果进行对比
            buypoint_result = self._create_sample_buypoint_result_Buypoint_Strategy_Integration_Example('000001', '20240601')
            
            # 标准权重转换
            standard_result = self.adapter.convert_buypoint_result(buypoint_result)
            
            # 自定义权重转换
            custom_result = custom_adapter.convert_buypoint_result(buypoint_result)
            
            print("\n📊 评分对比:")
            print(f"   标准权重评分: {standard_result['score']:.1f}")
            print(f"   自定义权重评分: {custom_result['score']:.1f}")
            print(f"   评分差异: {custom_result['score'] - standard_result['score']:+.1f}")
            
            print(f"\n推荐等级对比:")
            print(f"   标准权重推荐: {standard_result['recommendation']}")
            print(f"   自定义权重推荐: {custom_result['recommendation']}")
            
        except Exception as e:
            print(f"❌ 示例执行失败: {e}")
    
    def _create_sample_buypoint_result_Buypoint_Strategy_Integration_Example(self, stock_code: str, buypoint_date: str, score: float = None) -> Dict[str, Any]:
        """创建模拟买点分析结果"""
        if score is None:
            import random
            score = random.uniform(40, 95)
        
        return {
            'stock_code': stock_code,
            'buypoint_date': buypoint_date,
            'indicator_results': {
                'period_data': {
                    'daily': pd.DataFrame({'close': [10.0, 10.5, 11.0]}),
                    '60min': pd.DataFrame({'close': [10.2, 10.7, 10.9]})
                },
                'technical_analysis': {
                    'macd_gold': {'value': 0.15, 'signal': 'buy', 'strength': 0.8},
                    'touch_ma': {'value': 1.0, 'signal': 'positive', 'strength': 0.7},
                    'xc': {'value': 1.0, 'signal': 'buy', 'strength': 0.9}
                }
            },
            'pattern_results': {
                'macd_gold': {'detected': True, 'confidence': 0.8, 'description': 'MACD金叉信号'},
                'touch_ma': {'detected': True, 'confidence': 0.7, 'description': '触及均线支撑'},
                'price_stable': {'detected': True, 'confidence': 0.6, 'description': '价格企稳'},
                'xc': {'detected': True, 'confidence': 0.9, 'description': '吸筹信号'}
            },
            'summary': {
                'total_indicators': 8,
                'positive_signals': int(score / 12.5),  # 基于评分计算正面信号数
                'negative_signals': 8 - int(score / 12.5),
                'overall_score': score
            }
        }
    
    def _validate_strategy_compatibility_Buypoint_Strategy_Integration_Example(self, strategy_result: Dict[str, Any]):
        """验证选股策略兼容性"""
        required_fields = [
            'stock_code', 'stock_name', 'industry', 'price',
            'change_pct', 'score', 'match_details', 'selection_date'
        ]
        
        missing_fields = [field for field in required_fields if field not in strategy_result]
        
        if not missing_fields:
            print("✅ 选股策略兼容性验证通过")
            print(f"   包含所有必需字段: {len(required_fields)}个")
            print(f"   评分范围正确: {0 <= strategy_result['score'] <= 100}")
            print(f"   推荐等级有效: {strategy_result['recommendation'] in ['buy', 'hold', 'sell']}")
        else:
            print(f"❌ 选股策略兼容性验证失败，缺少字段: {missing_fields}")


def mainBuypointstrategyintegrationexample():
    """主函数"""
    print("🚀 买点分析与选股策略集成使用示例")
    print("=" * 80)
    print("本示例展示如何使用BuyPointToStrategyAdapter实现无缝集成")
    print("=" * 80)
    
    try:
        # 创建示例实例
        example = BuyPointStrategyIntegrationExample()
        
        # 运行所有示例
        example.example_1_single_stock_integration()
        example.example_2_batch_integration()
        example.example_3_strategy_workflow_integration()
        example.example_4_custom_scoring_integration()
        
        print("\n" + "=" * 80)
        print("🎉 所有示例执行完成！")
        print("=" * 80)
        print("\n💡 使用提示:")
        print("1. 在实际使用中，买点分析结果来自 BuyPointBatchAnalyzer")
        print("2. 适配器支持单个和批量转换")
        print("3. 转换后的结果可直接用于选股策略")
        print("4. 支持自定义评分权重配置")
        print("5. 完全向后兼容，不影响现有系统")
        
        print("\n📚 相关文档:")
        print("- 集成分析报告: analysis/BUYPOINT_STRATEGY_INTEGRATION_ANALYSIS.md")
        print("- 适配器源码: analysis/buypoints/buypoint_strategy_adapter.py")
        print("- 集成测试: tests/integration/buypoint_strategy_integration_test.py")
        
    except Exception as e:
        print(f"❌ 示例执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    mainBuypointstrategyintegrationexample()
