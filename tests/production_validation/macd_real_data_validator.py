#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD真实数据验证器 - 阶段3验证

严格按照三阶段验证框架执行真实数据双向验证：
1. 正向验证：使用系统现有的策略执行器进行MACD形态选股
2. 反向验证：使用买点分析器验证选出股票的MACD形态
3. 完整性验证：每个形态至少选出1支股票，符合系统架构要求
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from strategy.strategy_executor import StrategyExecutor
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from indicators.macd import MacdMacd
from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry
from utils.logger import get_logger

logger = get_logger(__name__)

class MACDRealDataValidator:
    """MACD真实数据验证器 - 阶段3验证"""

    def __init__(self):
        """初始化真实数据验证器"""
        self.macd = MacdMacd()
        self.pattern_registry = get_unified_pattern_registry()

        # 初始化系统组件（按照架构要求）
        try:
            self.strategy_executor = StrategyExecutor()
            self.buypoint_analyzer = BuyPointAnalyzer()
        except Exception as e:
            logger.error(f"初始化系统组件失败: {e}")
            raise RuntimeError(f"阶段3验证失败：无法初始化系统组件 - {e}")

        # 阶段3验证标准
        self.stage3_standards = {
            'min_stocks_per_pattern': 1,                 # 每个形态至少选出1支股票
            'min_total_selected_stocks': 3,              # 总共至少选出3支股票
            'validation_success_rate': 0.8,              # 80%验证成功率
            'max_validation_time': 300                   # 最大验证时间（秒）
        }

        logger.info("🔥 MACD真实数据验证器初始化完成 - 阶段3验证")
    
    def run_stage3_validation(self) -> Dict[str, Any]:
        """运行阶段3真实数据验证"""
        
        print("🔥 开始MACD阶段3真实数据验证")
        print("=" * 80)
        print("📋 验证框架: 三阶段验证 - 阶段3")
        print("📊 数据源: ClickHouse真实股票数据")
        print("📅 历史天数: 60-120天")
        print("🎯 目标: 100%形态识别 + 双向验证")
        print("=" * 80)
        
        validation_result = {
            'stage': 'Stage3_RealData',
            'indicator_name': 'MACD',
            'validation_timestamp': datetime.now().isoformat(),
            'standards': self.stage3_standards,
            'positive_validation': {},      # 正向验证结果
            'negative_validation': {},      # 反向验证结果
            'stock_selection_results': {},  # 股票选择结果
            'overall_assessment': None,
            'stage3_passed': False,
            'issues_found': []
        }
        
        try:
            # 步骤1: 验证数据库连接
            print("\n📊 步骤1: 验证数据库连接和系统组件")
            self._validate_system_components()
            print(f"✅ 系统组件验证通过")
            
            # 步骤2: 使用策略执行器进行MACD形态选股
            print("\n🎯 步骤2: 使用策略执行器进行MACD形态选股")
            strategy_results = self._execute_macd_strategies()
            validation_result['strategy_execution_results'] = strategy_results

            if not strategy_results['success']:
                validation_result['issues_found'].extend(strategy_results['issues'])
                print(f"❌ 策略执行失败: {strategy_results['issues']}")
                return validation_result

            print(f"✅ 策略执行成功: 选出{strategy_results['total_selected_stocks']}支股票")

            # 步骤3: 验证选出的股票确实包含目标形态
            print("\n🛡️ 步骤3: 验证选出股票的形态正确性")
            pattern_validation_results = self._validate_selected_stocks_patterns(strategy_results)
            validation_result['pattern_validation_results'] = pattern_validation_results

            if not pattern_validation_results['success']:
                validation_result['issues_found'].extend(pattern_validation_results['issues'])
                print(f"⚠️ 形态验证需要改进: {pattern_validation_results['issues']}")
            else:
                print(f"✅ 形态验证成功: {pattern_validation_results['validation_rate']:.1%} 验证率")

            # 步骤4: 买点分析验证
            print("\n📈 步骤4: 买点分析验证")
            buypoint_results = self._validate_buypoint_analysis(strategy_results)
            validation_result['buypoint_validation_results'] = buypoint_results

            # 步骤5: 综合评估
            print("\n🏆 步骤5: 阶段3综合评估")
            overall_assessment = self._comprehensive_stage3_assessment(
                strategy_results, pattern_validation_results, buypoint_results
            )
            validation_result['overall_assessment'] = overall_assessment
            validation_result['stage3_passed'] = overall_assessment['passed']
            
            if overall_assessment['passed']:
                print(f"🎉 MACD指标通过阶段3真实数据验证！")
            else:
                validation_result['issues_found'].extend(overall_assessment['issues'])
                print(f"❌ 阶段3验证失败: {overall_assessment['issues']}")
        
        except Exception as e:
            logger.error(f"❌ 阶段3验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _get_real_stock_data(self) -> Dict[str, pd.DataFrame]:
        """获取真实股票数据"""
        
        stock_data = {}
        
        try:
            # 获取股票列表 - 必须使用真实数据
            stock_list_query = """
            SELECT DISTINCT stock_code
            FROM stock_data
            WHERE date >= today() - 150
            AND (stock_code LIKE '00%' OR stock_code LIKE '60%')
            ORDER BY stock_code
            LIMIT 50
            """

            try:
                stock_codes = self.db_manager.query_dataframe(stock_list_query)
                if stock_codes is not None and len(stock_codes) > 0:
                    stock_codes = stock_codes.to_dict('records')
                else:
                    stock_codes = None
            except Exception as e:
                logger.error(f"数据库查询失败: {e}")
                stock_codes = None

            if not stock_codes:
                # 如果无法获取真实数据，验证失败
                logger.error("❌ 无法获取真实股票数据，阶段3验证失败")
                raise RuntimeError("阶段3验证要求使用真实数据，但无法连接到ClickHouse数据库")
            
            # 为每个股票获取历史数据
            for stock_info in stock_codes[:self.stage3_standards['test_stock_count']]:
                stock_code = stock_info['stock_code']
                
                # 获取历史数据 - 必须使用真实数据
                history_query = f"""
                SELECT date, open, high, low, close, volume
                FROM stock_data
                WHERE stock_code = '{stock_code}'
                AND date >= today() - {self.stage3_standards['history_days_max']}
                ORDER BY date ASC
                LIMIT {self.stage3_standards['history_days_max']}
                """

                try:
                    stock_history_df = self.db_manager.query_dataframe(history_query)
                    if stock_history_df is not None and len(stock_history_df) >= self.stage3_standards['history_days_min']:
                        stock_history_df['date'] = pd.to_datetime(stock_history_df['date'])
                        stock_history_df = stock_history_df.sort_values('date').reset_index(drop=True)
                        stock_data[stock_code] = stock_history_df
                        logger.info(f"✅ 获取股票{stock_code}真实数据: {len(stock_history_df)}天")
                    else:
                        logger.warning(f"⚠️ 股票{stock_code}数据不足: {len(stock_history_df) if stock_history_df is not None else 0}天")
                except Exception as e:
                    logger.error(f"❌ 获取股票{stock_code}数据失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"获取真实股票数据失败: {e}")
            # 阶段3验证必须使用真实数据，不允许使用模拟数据
            raise RuntimeError(f"阶段3验证失败：无法获取真实股票数据 - {e}")

        # 验证是否获取到足够的真实数据
        if len(stock_data) < 10:
            raise RuntimeError(f"阶段3验证失败：真实数据不足，仅获取到{len(stock_data)}支股票数据，至少需要10支")
        
        return stock_data
    

    
    def _positive_validation(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """正向验证 - 识别目标形态"""
        
        result = {
            'success': False,
            'recognition_rate': 0.0,
            'pattern_results': {},
            'total_patterns_found': 0,
            'issues': []
        }
        
        try:
            macd_patterns = self.pattern_registry.get_indicator_patterns('MACD')
            total_patterns_found = 0
            pattern_counts = {}
            
            for pattern in macd_patterns:
                pattern_counts[pattern] = 0
                print(f"    🔍 检测形态: {pattern}")
                
                # 在所有股票中检测该形态
                for stock_code, data in stock_data.items():
                    try:
                        patterns_result = self.macd.get_patterns(data)
                        
                        if (patterns_result is not None and 
                            pattern in patterns_result.columns and
                            patterns_result[pattern].sum() > 0):
                            pattern_counts[pattern] += 1
                            total_patterns_found += patterns_result[pattern].sum()
                    
                    except Exception as e:
                        logger.warning(f"股票{stock_code}形态检测失败: {e}")
                        continue
                
                print(f"      📊 {pattern}: 在{pattern_counts[pattern]}支股票中发现")
                result['pattern_results'][pattern] = pattern_counts[pattern]
            
            result['total_patterns_found'] = total_patterns_found
            
            # 计算识别率
            patterns_with_results = sum(1 for count in pattern_counts.values() if count > 0)
            result['recognition_rate'] = patterns_with_results / len(macd_patterns)
            
            # 成功标准：至少80%的形态能被识别
            if result['recognition_rate'] >= 0.8:
                result['success'] = True
            else:
                result['issues'].append(f"形态识别率{result['recognition_rate']:.1%} < 80%")
        
        except Exception as e:
            result['issues'].append(f"正向验证异常: {str(e)}")
        
        return result
    
    def _negative_validation(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """反向验证 - 检测误报"""
        
        result = {
            'success': False,
            'false_positive_rate': 0.0,
            'total_false_positives': 0,
            'total_detections': 0,
            'issues': []
        }
        
        try:
            total_detections = 0
            total_false_positives = 0
            
            # 选择一部分股票进行反向验证
            stock_list = list(stock_data.keys())
            sample_size = min(20, len(stock_data))
            sample_stocks = stock_list[:sample_size]  # 取前20支股票
            
            for stock_code in sample_stocks:
                data = stock_data[stock_code]
                
                try:
                    patterns_result = self.macd.get_patterns(data)
                    
                    if patterns_result is not None:
                        current_detections = patterns_result.sum().sum()
                        total_detections += current_detections
                        
                        # 简单的误报检测：如果检测到的形态过多，可能存在误报
                        if current_detections > len(data) * 0.1:  # 超过10%的数据点有形态
                            total_false_positives += current_detections - int(len(data) * 0.1)
                
                except Exception as e:
                    logger.warning(f"股票{stock_code}反向验证失败: {e}")
                    continue
            
            result['total_detections'] = total_detections
            result['total_false_positives'] = total_false_positives
            
            if total_detections > 0:
                result['false_positive_rate'] = total_false_positives / total_detections
            
            # 成功标准：假阳性率 < 10%
            if result['false_positive_rate'] <= self.stage3_standards['max_false_positive_rate']:
                result['success'] = True
            else:
                result['issues'].append(f"假阳性率{result['false_positive_rate']:.1%} > 10%")
        
        except Exception as e:
            result['issues'].append(f"反向验证异常: {str(e)}")
        
        return result
    
    def _stock_selection_validation(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """股票选择验证"""
        
        result = {
            'success': False,
            'patterns_with_stocks': {},
            'total_selected_stocks': 0,
            'issues': []
        }
        
        try:
            macd_patterns = self.pattern_registry.get_indicator_patterns('MACD')
            patterns_with_stocks = {}
            total_selected = 0
            
            for pattern in macd_patterns:
                selected_stocks = []
                
                for stock_code, data in stock_data.items():
                    try:
                        patterns_result = self.macd.get_patterns(data)
                        
                        if (patterns_result is not None and 
                            pattern in patterns_result.columns and
                            patterns_result[pattern].sum() > 0):
                            selected_stocks.append(stock_code)
                    
                    except Exception as e:
                        continue
                
                patterns_with_stocks[pattern] = len(selected_stocks)
                total_selected += len(selected_stocks)
                
                print(f"    📈 {pattern}: 选出{len(selected_stocks)}支股票")
                
                # 检查是否满足最少股票数要求
                if len(selected_stocks) < self.stage3_standards['min_stocks_per_pattern']:
                    result['issues'].append(f"{pattern}选出股票数{len(selected_stocks)} < {self.stage3_standards['min_stocks_per_pattern']}")
            
            result['patterns_with_stocks'] = patterns_with_stocks
            result['total_selected_stocks'] = total_selected
            
            # 成功标准：每个形态至少选出1支股票
            patterns_meeting_requirement = sum(1 for count in patterns_with_stocks.values() 
                                             if count >= self.stage3_standards['min_stocks_per_pattern'])
            
            if patterns_meeting_requirement == len(macd_patterns):
                result['success'] = True
            else:
                result['issues'].append(f"只有{patterns_meeting_requirement}/{len(macd_patterns)}个形态满足选股要求")
        
        except Exception as e:
            result['issues'].append(f"股票选择验证异常: {str(e)}")
        
        return result
    
    def _comprehensive_stage3_assessment(self, positive_results: Dict, negative_results: Dict, 
                                       selection_results: Dict) -> Dict[str, Any]:
        """阶段3综合评估"""
        
        assessment = {
            'passed': False,
            'overall_score': 0.0,
            'component_scores': {},
            'strengths': [],
            'weaknesses': [],
            'issues': []
        }
        
        try:
            # 计算各组件得分
            positive_score = positive_results['recognition_rate'] * 40  # 40%权重
            negative_score = (1 - negative_results['false_positive_rate']) * 30  # 30%权重
            selection_score = (1 if selection_results['success'] else 0) * 30  # 30%权重
            
            assessment['component_scores'] = {
                'positive_validation': positive_score,
                'negative_validation': negative_score,
                'stock_selection': selection_score
            }
            
            assessment['overall_score'] = positive_score + negative_score + selection_score
            
            # 评估优势和劣势
            if positive_results['recognition_rate'] >= 0.8:
                assessment['strengths'].append("良好的形态识别能力")
            else:
                assessment['weaknesses'].append("形态识别能力需要改进")
            
            if negative_results['false_positive_rate'] <= 0.1:
                assessment['strengths'].append("良好的误报控制")
            else:
                assessment['weaknesses'].append("误报率偏高")
            
            if selection_results['success']:
                assessment['strengths'].append("成功的股票选择能力")
            else:
                assessment['weaknesses'].append("股票选择能力不足")
            
            # 阶段3通过标准：综合得分 >= 80分 且 所有组件都达标
            if (assessment['overall_score'] >= 80 and 
                positive_results['success'] and 
                negative_results['success'] and 
                selection_results['success']):
                assessment['passed'] = True
            else:
                if assessment['overall_score'] < 80:
                    assessment['issues'].append(f"综合得分{assessment['overall_score']:.1f} < 80分")
                if not positive_results['success']:
                    assessment['issues'].append("正向验证未通过")
                if not negative_results['success']:
                    assessment['issues'].append("反向验证未通过")
                if not selection_results['success']:
                    assessment['issues'].append("股票选择验证未通过")
        
        except Exception as e:
            assessment['issues'].append(f"综合评估异常: {str(e)}")
        
        return assessment

def main():
    """主函数"""
    validator = MACDRealDataValidator()
    
    # 运行阶段3验证
    results = validator.run_stage3_validation()
    
    print("\n" + "="*80)
    print("🏆 MACD阶段3真实数据验证结果汇总")
    print("="*80)
    
    print(f"🎯 阶段3通过: {results['stage3_passed']}")
    
    if results['positive_validation']:
        pos_val = results['positive_validation']
        print(f"✅ 正向验证: {pos_val['success']} (识别率: {pos_val['recognition_rate']:.1%})")
    
    if results['negative_validation']:
        neg_val = results['negative_validation']
        print(f"🛡️ 反向验证: {neg_val['success']} (假阳性率: {neg_val['false_positive_rate']:.1%})")
    
    if results['stock_selection_results']:
        sel_res = results['stock_selection_results']
        print(f"📈 股票选择: {sel_res['success']} (总选出: {sel_res['total_selected_stocks']}支)")
    
    if results['overall_assessment']:
        assessment = results['overall_assessment']
        print(f"📊 综合得分: {assessment['overall_score']:.1f}/100")
        
        if assessment['strengths']:
            print(f"💪 优势: {', '.join(assessment['strengths'])}")
        
        if assessment['weaknesses']:
            print(f"⚠️ 劣势: {', '.join(assessment['weaknesses'])}")
    
    if results['stage3_passed']:
        print(f"\n🎉 MACD指标通过阶段3真实数据验证！")
        print(f"✅ 三阶段验证框架全部完成，可以开始下一个P0指标(RSI)验证")
    else:
        print(f"\n🔧 MACD指标阶段3验证需要改进")
        if results['issues_found']:
            print(f"❌ 问题: {', '.join(results['issues_found'])}")

if __name__ == "__main__":
    main()
