"""
买点策略闭环验证器

实现策略生成后的闭环验证机制，确保生成的策略能够重新选出原始买点个股
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from utils.logger import get_logger
from strategy.strategy_executor import StrategyExecutor
from db.data_manager import DataManager

logger = get_logger(__name__)


class BuyPointValidator:
    """买点策略闭环验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.strategy_executor = StrategyExecutor()
        self.data_manager = DataManager()
        
    def validate_strategy_roundtrip(self, 
                                  original_buypoints: pd.DataFrame,
                                  generated_strategy: Dict[str, Any],
                                  validation_date: str) -> Dict[str, Any]:
        """
        执行闭环验证：策略是否能重新选出原始买点个股
        
        Args:
            original_buypoints: 原始买点数据
            generated_strategy: 生成的策略配置
            validation_date: 验证日期
            
        Returns:
            Dict: 验证结果
        """
        validation_results = {
            'total_original_stocks': len(original_buypoints),
            'validation_date': validation_date,
            'strategy_summary': self._summarize_strategy(generated_strategy),
            'execution_results': {},
            'match_analysis': {},
            'recommendations': []
        }
        
        try:
            logger.info(f"开始闭环验证，原始买点数量: {len(original_buypoints)}")
            
            # 1. 获取原始买点股票代码
            original_codes = set(original_buypoints['stock_code'].unique())
            
            # 2. 标准化策略格式以兼容策略执行器
            normalized_strategy = self._normalize_strategy_format(generated_strategy)

            # 3. 执行策略获取选股结果
            selected_stocks = self.strategy_executor.execute_strategy(
                strategy_plan=normalized_strategy,
                end_date=validation_date
            )
            
            if selected_stocks is None or len(selected_stocks) == 0:
                validation_results['execution_results'] = {
                    'selected_count': 0,
                    'selected_stocks': [],
                    'execution_error': '策略执行未选出任何股票'
                }
                validation_results['match_analysis']['match_rate'] = 0.0
                return validation_results
            
            # 3. 分析匹配结果
            selected_codes = set(selected_stocks['code'].unique())
            matched_codes = original_codes & selected_codes
            missed_codes = original_codes - selected_codes
            false_positive_codes = selected_codes - original_codes
            
            match_rate = len(matched_codes) / len(original_codes) if original_codes else 0
            
            validation_results['execution_results'] = {
                'selected_count': len(selected_stocks),
                'selected_stocks': list(selected_codes),
                'execution_success': True
            }
            
            validation_results['match_analysis'] = {
                'match_rate': match_rate,
                'matched_count': len(matched_codes),
                'missed_count': len(missed_codes),
                'false_positive_count': len(false_positive_codes),
                'matched_stocks': list(matched_codes),
                'missed_stocks': list(missed_codes),
                'false_positive_stocks': list(false_positive_codes)
            }
            
            # 4. 生成改进建议
            validation_results['recommendations'] = self._generate_recommendations(
                match_rate, missed_codes, false_positive_codes, generated_strategy)
            
            # 5. 质量评级
            validation_results['quality_grade'] = self._assess_quality_grade(match_rate)
            
            logger.info(f"闭环验证完成，匹配率: {match_rate:.2%}")
            
        except Exception as e:
            logger.error(f"策略验证执行失败: {e}")
            validation_results['execution_results'] = {
                'selected_count': 0,
                'selected_stocks': [],
                'execution_error': str(e)
            }
            validation_results['match_analysis']['match_rate'] = 0.0
        
        return validation_results

    def _normalize_strategy_format(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化策略格式以兼容策略执行器

        将indicator字段转换为indicator_id字段，确保策略执行器能正确处理
        """
        normalized_strategy = strategy.copy()

        if 'conditions' in normalized_strategy:
            normalized_conditions = []
            for condition in normalized_strategy['conditions']:
                normalized_condition = condition.copy()

                # 如果有indicator字段但没有indicator_id字段，进行转换
                if 'indicator' in normalized_condition and 'indicator_id' not in normalized_condition:
                    normalized_condition['indicator_id'] = normalized_condition['indicator']

                # 确保必要字段存在
                if normalized_condition.get('type') == 'indicator':
                    # 如果没有period字段，设置默认值
                    if 'period' not in normalized_condition:
                        normalized_condition['period'] = 'daily'

                    # 如果没有pattern字段，设置默认值
                    if 'pattern' not in normalized_condition:
                        normalized_condition['pattern'] = 'BULLISH'

                normalized_conditions.append(normalized_condition)

            normalized_strategy['conditions'] = normalized_conditions

        return normalized_strategy

    def _summarize_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """总结策略信息"""
        conditions = strategy.get('conditions', [])
        return {
            'name': strategy.get('name', '未命名策略'),
            'condition_count': len(conditions),
            'logic': strategy.get('condition_logic', 'OR'),
            'periods': list(set(c.get('period', '') for c in conditions if c.get('period'))),
            'indicators': list(set(c.get('indicator', '') for c in conditions if c.get('indicator')))
        }
    
    def _generate_recommendations(self, match_rate: float, missed_codes: set, 
                                false_positive_codes: set, strategy: Dict) -> List[Dict]:
        """生成改进建议"""
        recommendations = []
        
        if match_rate < 0.3:
            recommendations.append({
                'priority': 'HIGH',
                'issue': '匹配率过低',
                'suggestion': '策略条件过于严格，建议放宽阈值或增加OR条件',
                'action': 'adjust_thresholds'
            })
        elif match_rate < 0.6:
            recommendations.append({
                'priority': 'MEDIUM', 
                'issue': '匹配率偏低',
                'suggestion': '优化策略条件，重点分析未匹配股票的特征',
                'action': 'analyze_missed_stocks'
            })
        
        if len(false_positive_codes) > len(missed_codes) * 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': '误选股票过多',
                'suggestion': '策略条件过于宽松，建议增加筛选条件',
                'action': 'add_filters'
            })
        
        condition_count = len(strategy.get('conditions', []))
        if condition_count > 100:
            recommendations.append({
                'priority': 'HIGH',
                'issue': '策略条件过多',
                'suggestion': '简化策略条件，保留最重要的指标',
                'action': 'simplify_conditions'
            })
        
        return recommendations
    
    def _assess_quality_grade(self, match_rate: float) -> str:
        """评估策略质量等级"""
        if match_rate >= 0.8:
            return "优秀"
        elif match_rate >= 0.6:
            return "良好"
        elif match_rate >= 0.4:
            return "一般"
        else:
            return "需要改进"
    
    def generate_validation_report(self, validation_results: Dict[str, Any], 
                                 output_file: str) -> None:
        """生成验证报告"""
        try:
            # 生成Markdown格式的验证报告
            report_content = self._format_validation_report(validation_results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"验证报告已生成: {output_file}")
            
        except Exception as e:
            logger.error(f"生成验证报告失败: {e}")
    
    def _format_validation_report(self, results: Dict[str, Any]) -> str:
        """格式化验证报告"""
        match_analysis = results.get('match_analysis', {})
        match_rate = match_analysis.get('match_rate', 0)
        
        report = f"""# 策略闭环验证报告

## 验证概览
- **验证时间**: {results.get('validation_date', 'N/A')}
- **原始买点数量**: {results.get('total_original_stocks', 0)}
- **策略名称**: {results.get('strategy_summary', {}).get('name', 'N/A')}
- **质量评级**: {results.get('quality_grade', 'N/A')}

## 匹配分析
- **匹配率**: {match_rate:.2%}
- **匹配股票数**: {match_analysis.get('matched_count', 0)}
- **遗漏股票数**: {match_analysis.get('missed_count', 0)}
- **误选股票数**: {match_analysis.get('false_positive_count', 0)}

## 策略执行结果
- **选出股票总数**: {results.get('execution_results', {}).get('selected_count', 0)}
- **执行状态**: {'成功' if results.get('execution_results', {}).get('execution_success') else '失败'}

## 改进建议
"""
        
        recommendations = results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. **{rec['issue']}** ({rec['priority']}优先级)\n"
                report += f"   - 建议: {rec['suggestion']}\n\n"
        else:
            report += "暂无改进建议\n"
        
        return report
