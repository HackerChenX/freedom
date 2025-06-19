#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点策略闭环验证工具

此工具用于验证买点分析生成的选股策略是否能够重新选出原始买点个股，
实现选股系统的闭环验证机制。

使用方法:
    python bin/validate_buypoint_strategy.py --buypoints data/buypoints.csv --strategy data/result/your_buypoints_analysis/generated_strategy.json --output validation_report.json

功能特性:
1. 策略回测验证 - 验证策略是否能选出原始买点股票
2. 匹配率分析 - 计算策略匹配原始买点的准确率
3. 问题诊断 - 分析未匹配股票的原因
4. 改进建议 - 提供策略优化建议
5. 详细报告 - 生成完整的验证报告
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
import logging

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_executor import StrategyExecutor
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger

logger = get_logger(__name__)

class BuyPointStrategyValidator:
    """买点策略验证器"""
    
    def __init__(self):
        self.db_manager = get_clickhouse_db()
        self.strategy_executor = StrategyExecutor(self.db_manager)
        
    def validate_strategy(self, buypoints_file, strategy_file, output_file):
        """
        执行完整的策略验证流程
        
        Args:
            buypoints_file: 买点数据文件路径
            strategy_file: 策略配置文件路径  
            output_file: 验证报告输出路径
        """
        logger.info("开始买点策略闭环验证")
        
        try:
            # 1. 加载数据
            buypoints_df = self._load_buypoints(buypoints_file)
            strategy_config = self._load_strategy(strategy_file)
            
            logger.info(f"加载买点数据: {len(buypoints_df)} 条记录")
            logger.info(f"策略条件数量: {len(strategy_config.get('conditions', []))}")
            
            # 2. 执行验证
            validation_results = self._execute_validation(buypoints_df, strategy_config)
            
            # 3. 生成报告
            report = self._generate_report(validation_results, buypoints_df, strategy_config)
            
            # 4. 保存结果
            self._save_report(report, output_file)
            
            # 5. 输出摘要
            self._print_summary(report)
            
            logger.info("买点策略验证完成")
            
        except Exception as e:
            logger.error(f"验证过程出错: {e}")
            raise
    
    def _load_buypoints(self, file_path):
        """加载买点数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"买点数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        required_columns = ['stock_code', 'buy_date']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"买点数据缺少必需列: {col}")
        
        return df
    
    def _load_strategy(self, file_path):
        """加载策略配置"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"策略文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            strategy = json.load(f)
        
        if 'strategy' not in strategy:
            raise ValueError("策略文件格式错误，缺少strategy字段")
        
        return strategy['strategy']
    
    def _execute_validation(self, buypoints_df, strategy_config):
        """执行验证逻辑"""
        results = {
            'validation_date': datetime.now().isoformat(),
            'original_buypoints': {
                'total_count': len(buypoints_df),
                'unique_stocks': buypoints_df['stock_code'].nunique(),
                'date_range': {
                    'start': buypoints_df['buy_date'].min(),
                    'end': buypoints_df['buy_date'].max()
                }
            },
            'strategy_execution': {},
            'match_analysis': {},
            'detailed_results': []
        }
        
        # 按日期分组验证
        date_groups = buypoints_df.groupby('buy_date')
        
        for buy_date, group in date_groups:
            logger.info(f"验证日期 {buy_date}，买点数量: {len(group)}")
            
            date_result = self._validate_single_date(group, strategy_config, buy_date)
            results['detailed_results'].append(date_result)
        
        # 汇总结果
        results['match_analysis'] = self._aggregate_results(results['detailed_results'])
        
        return results
    
    def _validate_single_date(self, buypoints_group, strategy_config, buy_date):
        """验证单个日期的买点"""
        original_codes = set(buypoints_group['stock_code'].unique())
        
        try:
            # 执行策略
            selected_stocks = self.strategy_executor.execute_strategy(
                strategy_config, 
                stock_pool=list(original_codes),
                end_date=buy_date
            )
            
            if selected_stocks is not None and len(selected_stocks) > 0:
                selected_codes = set(selected_stocks['code'].unique())
            else:
                selected_codes = set()
            
            # 计算匹配结果
            matched_codes = original_codes & selected_codes
            missed_codes = original_codes - selected_codes
            false_positive_codes = selected_codes - original_codes
            
            match_rate = len(matched_codes) / len(original_codes) if original_codes else 0
            
            return {
                'date': buy_date,
                'original_count': len(original_codes),
                'selected_count': len(selected_codes),
                'matched_count': len(matched_codes),
                'missed_count': len(missed_codes),
                'false_positive_count': len(false_positive_codes),
                'match_rate': match_rate,
                'original_stocks': list(original_codes),
                'selected_stocks': list(selected_codes),
                'matched_stocks': list(matched_codes),
                'missed_stocks': list(missed_codes),
                'false_positive_stocks': list(false_positive_codes),
                'execution_success': True
            }
            
        except Exception as e:
            logger.error(f"日期 {buy_date} 策略执行失败: {e}")
            return {
                'date': buy_date,
                'original_count': len(original_codes),
                'selected_count': 0,
                'matched_count': 0,
                'missed_count': len(original_codes),
                'false_positive_count': 0,
                'match_rate': 0.0,
                'original_stocks': list(original_codes),
                'selected_stocks': [],
                'matched_stocks': [],
                'missed_stocks': list(original_codes),
                'false_positive_stocks': [],
                'execution_success': False,
                'error': str(e)
            }
    
    def _aggregate_results(self, detailed_results):
        """汇总验证结果"""
        total_original = sum(r['original_count'] for r in detailed_results)
        total_selected = sum(r['selected_count'] for r in detailed_results)
        total_matched = sum(r['matched_count'] for r in detailed_results)
        total_missed = sum(r['missed_count'] for r in detailed_results)
        total_false_positive = sum(r['false_positive_count'] for r in detailed_results)
        
        overall_match_rate = total_matched / total_original if total_original > 0 else 0
        
        # 计算质量评级
        if overall_match_rate >= 0.8:
            quality_grade = "优秀"
        elif overall_match_rate >= 0.6:
            quality_grade = "良好"
        elif overall_match_rate >= 0.4:
            quality_grade = "一般"
        else:
            quality_grade = "需要改进"
        
        return {
            'overall_match_rate': overall_match_rate,
            'total_original_stocks': total_original,
            'total_selected_stocks': total_selected,
            'total_matched_stocks': total_matched,
            'total_missed_stocks': total_missed,
            'total_false_positive_stocks': total_false_positive,
            'quality_grade': quality_grade,
            'successful_dates': len([r for r in detailed_results if r['execution_success']]),
            'failed_dates': len([r for r in detailed_results if not r['execution_success']])
        }
    
    def _generate_report(self, validation_results, buypoints_df, strategy_config):
        """生成验证报告"""
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'validator_version': '1.0.0',
                'validation_type': 'buypoint_strategy_roundtrip'
            },
            'validation_results': validation_results,
            'strategy_analysis': self._analyze_strategy(strategy_config),
            'recommendations': self._generate_recommendations(validation_results, strategy_config),
            'next_steps': self._suggest_next_steps(validation_results)
        }
        
        return report
    
    def _analyze_strategy(self, strategy_config):
        """分析策略配置"""
        conditions = strategy_config.get('conditions', [])
        
        # 统计条件类型
        condition_types = {}
        indicator_counts = {}
        period_counts = {}
        
        for condition in conditions:
            cond_type = condition.get('type', 'unknown')
            condition_types[cond_type] = condition_types.get(cond_type, 0) + 1
            
            if cond_type == 'indicator':
                indicator = condition.get('indicator', 'unknown')
                period = condition.get('period', 'unknown')
                
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
                period_counts[period] = period_counts.get(period, 0) + 1
        
        return {
            'total_conditions': len(conditions),
            'condition_types': condition_types,
            'indicator_distribution': indicator_counts,
            'period_distribution': period_counts,
            'condition_logic': strategy_config.get('condition_logic', 'unknown')
        }
    
    def _generate_recommendations(self, validation_results, strategy_config):
        """生成改进建议"""
        recommendations = []
        match_rate = validation_results['match_analysis']['overall_match_rate']
        
        if match_rate < 0.3:
            recommendations.append({
                'priority': 'HIGH',
                'category': '匹配率过低',
                'issue': f'策略匹配率仅为 {match_rate:.1%}，远低于可接受水平',
                'suggestion': '建议重新分析买点特征，放宽策略条件或增加OR逻辑分支',
                'action_items': [
                    '检查策略条件是否过于严格',
                    '分析未匹配股票的共同特征',
                    '考虑降低评分阈值',
                    '增加更多指标组合'
                ]
            })
        elif match_rate < 0.6:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': '匹配率偏低', 
                'issue': f'策略匹配率为 {match_rate:.1%}，有改进空间',
                'suggestion': '优化策略条件，重点关注未匹配的股票特征',
                'action_items': [
                    '分析missed_stocks的技术特征',
                    '调整部分指标的权重',
                    '考虑增加辅助条件'
                ]
            })
        
        condition_count = len(strategy_config.get('conditions', []))
        if condition_count > 100:
            recommendations.append({
                'priority': 'HIGH',
                'category': '策略复杂度过高',
                'issue': f'策略包含 {condition_count} 个条件，过于复杂',
                'suggestion': '简化策略，保留最重要的条件',
                'action_items': [
                    '使用特征重要性分析筛选关键条件',
                    '合并相似的条件',
                    '移除冗余或无效的条件'
                ]
            })
        
        return recommendations
    
    def _suggest_next_steps(self, validation_results):
        """建议后续步骤"""
        match_rate = validation_results['match_analysis']['overall_match_rate']
        
        if match_rate >= 0.6:
            return [
                "策略验证通过，可以进行生产环境部署",
                "建议进行更大样本的回测验证",
                "监控策略在实际应用中的表现",
                "定期重新验证策略有效性"
            ]
        else:
            return [
                "策略需要进一步优化",
                "重新分析买点数据，寻找更强的共性特征",
                "考虑使用机器学习方法优化策略",
                "增加更多技术指标或调整现有参数",
                "完成优化后重新进行验证"
            ]
    
    def _save_report(self, report, output_file):
        """保存验证报告"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"验证报告已保存到: {output_file}")
    
    def _print_summary(self, report):
        """打印验证摘要"""
        match_analysis = report['validation_results']['match_analysis']
        
        print("\n" + "="*60)
        print("买点策略验证摘要")
        print("="*60)
        print(f"总体匹配率: {match_analysis['overall_match_rate']:.1%}")
        print(f"质量评级: {match_analysis['quality_grade']}")
        print(f"原始买点数量: {match_analysis['total_original_stocks']}")
        print(f"策略选出数量: {match_analysis['total_selected_stocks']}")
        print(f"成功匹配数量: {match_analysis['total_matched_stocks']}")
        print(f"未匹配数量: {match_analysis['total_missed_stocks']}")
        print(f"误选数量: {match_analysis['total_false_positive_stocks']}")
        
        print(f"\n策略分析:")
        strategy_analysis = report['strategy_analysis']
        print(f"策略条件总数: {strategy_analysis['total_conditions']}")
        print(f"条件逻辑: {strategy_analysis['condition_logic']}")
        
        print(f"\n改进建议数量: {len(report['recommendations'])}")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. [{rec['priority']}] {rec['category']}: {rec['suggestion']}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='买点策略闭环验证工具')
    parser.add_argument('--buypoints', required=True, help='买点数据文件路径')
    parser.add_argument('--strategy', required=True, help='策略配置文件路径')
    parser.add_argument('--output', required=True, help='验证报告输出路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = BuyPointStrategyValidator()
    validator.validate_strategy(args.buypoints, args.strategy, args.output)

if __name__ == '__main__':
    main()
