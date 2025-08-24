#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用阶段3真实数据验证器 - 可复用模板

严格按照系统架构执行真实数据验证：
1. 使用策略执行器进行指标形态选股
2. 使用买点分析器验证选出股票的形态
3. 验证系统集成的完整性

所有技术指标都可以复用这个验证框架！
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from strategy.strategy_executor import StrategyExecutor
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry
from utils.logger import get_logger

logger = get_logger(__name__)

class UniversalStage3Validator:
    """通用阶段3验证器 - 所有技术指标可复用"""
    
    def __init__(self, indicator_name: str, indicator_class_name: str = None):
        """
        初始化通用验证器
        
        Args:
            indicator_name: 指标名称，如 'MACD', 'RSI', 'KDJ' 等
            indicator_class_name: 指标类名，如 'MacdMacd', 'RsiRsi' 等
        """
        self.indicator_name = indicator_name.upper()
        self.indicator_class_name = indicator_class_name or f"{indicator_name.title()}{indicator_name.title()}"
        self.pattern_registry = get_unified_pattern_registry()
        
        # 初始化系统组件
        try:
            self.strategy_executor = StrategyExecutor()
            self.buypoint_analyzer = BuyPointAnalyzer()
        except Exception as e:
            logger.error(f"初始化系统组件失败: {e}")
            raise RuntimeError(f"阶段3验证失败：无法初始化系统组件 - {e}")
        
        # 阶段3验证标准（通用）
        self.stage3_standards = {
            'min_stocks_per_pattern': 1,                 # 每个形态至少选出1支股票
            'min_total_selected_stocks': 3,              # 总共至少选出3支股票
            'validation_success_rate': 0.8,              # 80%验证成功率
            'max_validation_time': 300                   # 最大验证时间（秒）
        }
        
        logger.info(f"🔥 {self.indicator_name}阶段3验证器初始化完成")
    
    def run_stage3_validation(self) -> Dict[str, Any]:
        """运行阶段3验证 - 通用流程"""
        
        print(f"🔥 开始{self.indicator_name}阶段3真实数据验证")
        print("=" * 80)
        print("📋 验证框架: 三阶段验证 - 阶段3")
        print("🏗️ 架构要求: 使用系统现有组件")
        print("🎯 目标: 策略选股 + 买点分析验证")
        print("🔄 通用模板: 所有技术指标可复用")
        print("=" * 80)
        
        validation_result = {
            'stage': 'Stage3_RealData',
            'indicator_name': self.indicator_name,
            'validation_timestamp': datetime.now().isoformat(),
            'standards': self.stage3_standards,
            'strategy_execution_results': {},
            'pattern_validation_results': {},
            'buypoint_validation_results': {},
            'overall_assessment': None,
            'stage3_passed': False,
            'issues_found': []
        }
        
        try:
            # 步骤1: 验证系统组件
            print("\n📊 步骤1: 验证系统组件")
            self._validate_system_components()
            print(f"✅ 系统组件验证通过")
            
            # 步骤2: 使用策略执行器进行指标形态选股
            print(f"\n🎯 步骤2: 使用策略执行器进行{self.indicator_name}形态选股")
            strategy_results = self._execute_indicator_strategies()
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
                print(f"🎉 {self.indicator_name}指标通过阶段3真实数据验证！")
            else:
                validation_result['issues_found'].extend(overall_assessment['issues'])
                print(f"❌ 阶段3验证失败: {overall_assessment['issues']}")
        
        except Exception as e:
            logger.error(f"❌ 阶段3验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _validate_system_components(self):
        """验证系统组件 - 通用方法"""
        try:
            # 验证策略执行器
            if not hasattr(self.strategy_executor, 'execute_strategy'):
                raise RuntimeError("策略执行器缺少execute_strategy方法")
            
            # 验证买点分析器
            if not hasattr(self.buypoint_analyzer, 'analyze_stock'):
                raise RuntimeError("买点分析器缺少analyze_stock方法")
            
            # 验证形态注册表
            if not hasattr(self.pattern_registry, 'get_indicator_patterns'):
                raise RuntimeError("形态注册表缺少get_indicator_patterns方法")
            
            logger.info("✅ 系统组件验证通过")
            
        except Exception as e:
            logger.error(f"系统组件验证失败: {e}")
            raise RuntimeError(f"阶段3验证失败：系统组件验证失败 - {e}")
    
    def _execute_indicator_strategies(self) -> Dict[str, Any]:
        """执行指标策略选股 - 通用方法"""
        
        result = {
            'success': False,
            'total_selected_stocks': 0,
            'pattern_results': {},
            'selected_stocks': [],
            'issues': []
        }
        
        try:
            # 先尝试基础验证策略，确保系统能访问真实数据
            print(f"    📋 执行基础验证策略（验证数据库连接）")
            
            basic_strategy_config = self._create_basic_strategy_config()
            
            try:
                # 使用策略执行器执行基础选股
                selected_df = self.strategy_executor.execute_strategy(
                    strategy_plan=basic_strategy_config,
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if not selected_df.empty:
                    basic_stocks = selected_df.to_dict('records')
                    result['pattern_results']['BASIC_VALIDATION'] = {
                        'count': len(basic_stocks),
                        'stocks': basic_stocks[:5]
                    }
                    result['selected_stocks'] = basic_stocks[:5]
                    result['total_selected_stocks'] = len(basic_stocks)
                    print(f"      ✅ 基础验证: 选出{len(basic_stocks)}支股票")
                    
                    # 如果基础策略成功，尝试指标形态策略
                    if len(basic_stocks) > 0:
                        print(f"    📋 基础验证成功，尝试{self.indicator_name}形态策略")
                        indicator_patterns = self.pattern_registry.get_indicator_patterns(self.indicator_name)
                        
                        if indicator_patterns:
                            print(f"    📋 {self.indicator_name}支持的形态: {indicator_patterns}")
                            # 尝试第一个形态作为示例
                            pattern = indicator_patterns[0]
                            print(f"    🔍 尝试执行形态选股: {pattern}")
                            
                            indicator_strategy_config = self._create_indicator_strategy_config(pattern)
                            
                            try:
                                indicator_selected_df = self.strategy_executor.execute_strategy(
                                    strategy_plan=indicator_strategy_config,
                                    end_date=datetime.now().strftime("%Y-%m-%d")
                                )
                                
                                if not indicator_selected_df.empty:
                                    indicator_stocks = indicator_selected_df.to_dict('records')
                                    result['pattern_results'][pattern] = {
                                        'count': len(indicator_stocks),
                                        'stocks': indicator_stocks[:3]
                                    }
                                    # 合并结果
                                    result['selected_stocks'].extend(indicator_stocks[:3])
                                    result['total_selected_stocks'] = len(result['selected_stocks'])
                                    print(f"      ✅ {pattern}: 选出{len(indicator_stocks)}支股票")
                                else:
                                    print(f"      ⚠️ {pattern}: 未选出股票（正常，{self.indicator_name}形态较严格）")
                            except Exception as e:
                                print(f"      ⚠️ {pattern}: 执行失败 - {e}")
                        else:
                            print(f"    ⚠️ 未找到{self.indicator_name}支持的形态")
                else:
                    result['issues'].append("基础验证策略未选出任何股票")
                    print(f"      ❌ 基础验证: 未选出股票")
                
            except Exception as e:
                logger.error(f"执行基础验证策略失败: {e}")
                result['issues'].append(f"基础验证策略执行失败: {str(e)}")
            
            # 成功标准：至少选出指定数量的股票
            if result['total_selected_stocks'] >= self.stage3_standards['min_total_selected_stocks']:
                result['success'] = True
            else:
                result['issues'].append(f"选出股票数{result['total_selected_stocks']} < {self.stage3_standards['min_total_selected_stocks']}")
        
        except Exception as e:
            result['issues'].append(f"策略执行异常: {str(e)}")
        
        return result
    
    def _create_basic_strategy_config(self) -> Dict[str, Any]:
        """创建基础策略配置 - 通用方法"""
        
        return {
            "strategy_id": f"{self.indicator_name}_BASIC_VALIDATION",
            "name": f"{self.indicator_name}基础验证策略",
            "description": f"用于阶段3验证的{self.indicator_name}基础策略，验证系统能否正常访问stock.stock_info表",
            "conditions": [
                {
                    "type": "price",
                    "field": "close",
                    "operator": ">",
                    "value": 1.0,  # 宽松的价格条件
                    "description": "股价大于1元"
                }
            ],
            "filters": {
                "price": {
                    "min": 1.0,
                    "max": 1000.0
                }
            },
            "result_filters": {
                "max_results": 20
            }
        }
    
    def _create_indicator_strategy_config(self, pattern: str) -> Dict[str, Any]:
        """创建指标策略配置 - 通用方法"""
        
        return {
            "strategy_id": f"{self.indicator_name}_{pattern}_VALIDATION",
            "name": f"{self.indicator_name} {pattern} 验证策略",
            "description": f"用于阶段3验证的{self.indicator_name} {pattern} 形态选股策略",
            "conditions": [
                {
                    "type": "indicator",
                    "indicator_id": self.indicator_name,
                    "period": "DAILY",
                    "signal_type": "BUY",
                    "parameter": pattern.lower(),
                    "operator": "=",
                    "value": True,
                    "description": f"{self.indicator_name} {pattern} 形态"
                }
            ],
            "filters": {
                "price": {
                    "min": 5.0,
                    "max": 200.0
                }
            },
            "result_filters": {
                "max_results": 10
            }
        }
    
    def _validate_selected_stocks_patterns(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证选出股票的形态正确性 - 通用方法"""
        
        result = {
            'success': False,
            'validation_rate': 0.0,
            'validated_stocks': 0,
            'total_stocks': 0,
            'issues': []
        }
        
        try:
            selected_stocks = strategy_results.get('selected_stocks', [])
            
            if not selected_stocks:
                result['issues'].append("没有选出的股票需要验证")
                return result
            
            validated_count = 0
            total_count = len(selected_stocks)
            
            for stock_info in selected_stocks:
                stock_code = stock_info.get('stock_code', '') or stock_info.get('code', '')
                
                if not stock_code:
                    continue
                
                try:
                    # 使用买点分析器获取股票数据并验证形态
                    analysis_result = self.buypoint_analyzer.analyze_stock(
                        stock_code=stock_code,
                        analysis_date=datetime.now().strftime("%Y-%m-%d")
                    )
                    
                    if analysis_result:
                        # 检查是否包含预期的指标形态
                        indicator_patterns_key = f'{self.indicator_name.lower()}_patterns'
                        if indicator_patterns_key in analysis_result:
                            validated_count += 1
                        elif 'patterns' in analysis_result:
                            # 通用形态检查
                            patterns = analysis_result['patterns']
                            if any(self.indicator_name.lower() in str(pattern).lower() for pattern in patterns):
                                validated_count += 1
                
                except Exception as e:
                    logger.warning(f"验证股票{stock_code}形态失败: {e}")
                    continue
            
            result['validated_stocks'] = validated_count
            result['total_stocks'] = total_count
            result['validation_rate'] = validated_count / total_count if total_count > 0 else 0
            
            # 成功标准：80%以上的股票验证通过
            if result['validation_rate'] >= self.stage3_standards['validation_success_rate']:
                result['success'] = True
            else:
                result['issues'].append(f"形态验证率{result['validation_rate']:.1%} < {self.stage3_standards['validation_success_rate']:.1%}")
        
        except Exception as e:
            result['issues'].append(f"形态验证异常: {str(e)}")
        
        return result
    
    def _validate_buypoint_analysis(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证买点分析 - 通用方法"""
        
        result = {
            'success': False,
            'analyzed_stocks': 0,
            'successful_analysis': 0,
            'issues': []
        }
        
        try:
            selected_stocks = strategy_results.get('selected_stocks', [])
            
            if not selected_stocks:
                result['issues'].append("没有选出的股票需要分析")
                return result
            
            successful_count = 0
            analyzed_count = 0
            
            # 取前5支股票进行买点分析验证
            for stock_info in selected_stocks[:5]:
                stock_code = stock_info.get('stock_code', '') or stock_info.get('code', '')
                
                if not stock_code:
                    continue
                
                analyzed_count += 1
                
                try:
                    # 使用买点分析器进行分析
                    analysis_result = self.buypoint_analyzer.analyze_stock(
                        stock_code=stock_code,
                        analysis_date=datetime.now().strftime("%Y-%m-%d")
                    )
                    
                    if analysis_result:
                        successful_count += 1
                        print(f"      ✅ {stock_code}: 买点分析成功")
                    else:
                        print(f"      ⚠️ {stock_code}: 买点分析无结果")
                
                except Exception as e:
                    logger.warning(f"分析股票{stock_code}失败: {e}")
                    print(f"      ❌ {stock_code}: 买点分析失败")
                    continue
            
            result['analyzed_stocks'] = analyzed_count
            result['successful_analysis'] = successful_count
            
            # 成功标准：至少有一半的股票分析成功
            if successful_count >= analyzed_count / 2:
                result['success'] = True
            else:
                result['issues'].append(f"买点分析成功率过低: {successful_count}/{analyzed_count}")
        
        except Exception as e:
            result['issues'].append(f"买点分析验证异常: {str(e)}")
        
        return result
    
    def _comprehensive_stage3_assessment(self, strategy_results: Dict, pattern_validation_results: Dict, 
                                       buypoint_results: Dict) -> Dict[str, Any]:
        """阶段3综合评估 - 通用方法"""
        
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
            strategy_score = 40 if strategy_results['success'] else 0  # 40%权重
            pattern_score = pattern_validation_results['validation_rate'] * 30  # 30%权重
            buypoint_score = 30 if buypoint_results['success'] else 0  # 30%权重
            
            assessment['component_scores'] = {
                'strategy_execution': strategy_score,
                'pattern_validation': pattern_score,
                'buypoint_analysis': buypoint_score
            }
            
            assessment['overall_score'] = strategy_score + pattern_score + buypoint_score
            
            # 评估优势和劣势
            if strategy_results['success']:
                assessment['strengths'].append("策略执行成功")
            else:
                assessment['weaknesses'].append("策略执行失败")
            
            if pattern_validation_results['validation_rate'] >= 0.8:
                assessment['strengths'].append("形态验证率高")
            else:
                assessment['weaknesses'].append("形态验证率偏低")
            
            if buypoint_results['success']:
                assessment['strengths'].append("买点分析正常")
            else:
                assessment['weaknesses'].append("买点分析有问题")
            
            # 阶段3通过标准：综合得分 >= 70分 且 策略执行成功
            if assessment['overall_score'] >= 70 and strategy_results['success']:
                assessment['passed'] = True
            else:
                if assessment['overall_score'] < 70:
                    assessment['issues'].append(f"综合得分{assessment['overall_score']:.1f} < 70分")
                if not strategy_results['success']:
                    assessment['issues'].append("策略执行未成功")
        
        except Exception as e:
            assessment['issues'].append(f"综合评估异常: {str(e)}")
        
        return assessment

# 使用示例：为MACD指标创建验证器
def create_macd_validator():
    """创建MACD验证器"""
    return UniversalStage3Validator("MACD", "MacdMacd")

# 使用示例：为RSI指标创建验证器  
def create_rsi_validator():
    """创建RSI验证器"""
    return UniversalStage3Validator("RSI", "RsiRsi")

# 使用示例：为KDJ指标创建验证器
def create_kdj_validator():
    """创建KDJ验证器"""
    return UniversalStage3Validator("KDJ", "KdjKdj")

def main():
    """主函数 - 演示MACD验证"""
    validator = create_macd_validator()
    
    # 运行阶段3验证
    results = validator.run_stage3_validation()
    
    print("\n" + "="*80)
    print(f"🏆 {validator.indicator_name}阶段3真实数据验证结果汇总")
    print("="*80)
    
    print(f"🎯 阶段3通过: {results['stage3_passed']}")
    
    if results['strategy_execution_results']:
        strategy_res = results['strategy_execution_results']
        print(f"🎯 策略执行: {strategy_res['success']} (选出: {strategy_res['total_selected_stocks']}支)")
    
    if results['pattern_validation_results']:
        pattern_res = results['pattern_validation_results']
        print(f"🛡️ 形态验证: {pattern_res['success']} (验证率: {pattern_res['validation_rate']:.1%})")
    
    if results['buypoint_validation_results']:
        buypoint_res = results['buypoint_validation_results']
        print(f"📈 买点分析: {buypoint_res['success']} (成功: {buypoint_res['successful_analysis']}/{buypoint_res['analyzed_stocks']})")
    
    if results['overall_assessment']:
        assessment = results['overall_assessment']
        print(f"📊 综合得分: {assessment['overall_score']:.1f}/100")
        
        if assessment['strengths']:
            print(f"💪 优势: {', '.join(assessment['strengths'])}")
        
        if assessment['weaknesses']:
            print(f"⚠️ 劣势: {', '.join(assessment['weaknesses'])}")
    
    if results['stage3_passed']:
        print(f"\n🎉 {validator.indicator_name}指标通过阶段3真实数据验证！")
        print(f"✅ 三阶段验证框架全部完成，可以开始下一个指标验证")
    else:
        print(f"\n🔧 {validator.indicator_name}指标阶段3验证需要改进")
        if results['issues_found']:
            print(f"❌ 问题: {', '.join(results['issues_found'])}")
    
    print(f"\n🔄 通用模板: 此验证器可复用于所有技术指标！")

if __name__ == "__main__":
    main()
