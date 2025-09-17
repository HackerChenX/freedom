from utils.container import container
"""
双向验证系统

实现策略选股结果与买点回测结果的双向验证机制，
通过交叉验证提高分析结果的可靠性和准确性。

遵循六层架构规范，提供可靠的验证服务。
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class ValidationResult(Enum):
    """验证结果枚举"""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"


class ValidationLevel(Enum):
    """验证级别枚举"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ValidationCriteria:
    """验证标准"""
    min_score_threshold: float = 60.0
    max_score_difference: float = 20.0
    min_confidence_level: float = 0.6
    require_signal_consistency: bool = True
    require_indicator_alignment: bool = True
    weight_strategy_result: float = 0.6
    weight_backtest_result: float = 0.4


@dataclass
class ValidationReport:
    """验证报告"""
    stock_code: str
    validation_date: str
    validation_level: ValidationLevel
    overall_result: ValidationResult
    strategy_validation: Dict[str, Any]
    backtest_validation: Dict[str, Any]
    cross_validation: Dict[str, Any]
    confidence_score: float
    recommendations: List[str]
    warnings: List[str]
    execution_time: float


class BidirectionalValidationSystem:
    """
    双向验证系统
    
    核心功能：
    1. 策略选股结果验证
    2. 买点回测结果验证
    3. 交叉验证分析
    4. 一致性检查
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化双向验证系统
        
        Args:
            validation_level: 验证级别
        """
        self.validation_level = validation_level
        self.logger = logger
        
        # 验证标准配置
        self.criteria = self._get_validation_criteria(validation_level)
        
        # 验证统计
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'partial_validations': 0,
            'average_confidence': 0.0,
            'average_execution_time': 0.0
        }
        
        self.logger.info(f"双向验证系统初始化完成，验证级别: {validation_level.value}")
    
    def _get_validation_criteria(self, level: ValidationLevel) -> ValidationCriteria:
        """根据验证级别获取验证标准"""
        if level == ValidationLevel.BASIC:
            return ValidationCriteria(
                min_score_threshold=50.0,
                max_score_difference=30.0,
                min_confidence_level=0.5,
                require_signal_consistency=False,
                require_indicator_alignment=False
            )
        elif level == ValidationLevel.STANDARD:
            return ValidationCriteria(
                min_score_threshold=60.0,
                max_score_difference=20.0,
                min_confidence_level=0.6,
                require_signal_consistency=True,
                require_indicator_alignment=False
            )
        elif level == ValidationLevel.STRICT:
            return ValidationCriteria(
                min_score_threshold=70.0,
                max_score_difference=15.0,
                min_confidence_level=0.7,
                require_signal_consistency=True,
                require_indicator_alignment=True
            )
        else:  # COMPREHENSIVE
            return ValidationCriteria(
                min_score_threshold=75.0,
                max_score_difference=10.0,
                min_confidence_level=0.8,
                require_signal_consistency=True,
                require_indicator_alignment=True
            )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def validate_integrated_result(self, 
                                  stock_code: str,
                                  strategy_result: Optional[Dict[str, Any]],
                                  backtest_result: Optional[Dict[str, Any]],
                                  validation_date: Optional[str] = None) -> ValidationReport:
        """
        验证集成分析结果
        
        Args:
            stock_code: 股票代码
            strategy_result: 策略选股结果
            backtest_result: 买点回测结果
            validation_date: 验证日期
            
        Returns:
            ValidationReport: 验证报告
        """
        start_time = time.time()
        
        if not validation_date:
            validation_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"开始验证股票 {stock_code} 的集成结果")
        
        # 初始化验证报告
        report = ValidationReport(
            stock_code=stock_code,
            validation_date=validation_date,
            validation_level=self.validation_level,
            overall_result=ValidationResult.INCONCLUSIVE,
            strategy_validation={},
            backtest_validation={},
            cross_validation={},
            confidence_score=0.0,
            recommendations=[],
            warnings=[],
            execution_time=0.0
        )
        
        try:
            # 1. 验证策略选股结果
            if strategy_result:
                report.strategy_validation = self._validate_strategy_result(strategy_result)
            else:
                report.warnings.append("策略选股结果缺失")
            
            # 2. 验证买点回测结果
            if backtest_result:
                report.backtest_validation = self._validate_backtest_result(backtest_result)
            else:
                report.warnings.append("买点回测结果缺失")
            
            # 3. 交叉验证
            if strategy_result and backtest_result:
                report.cross_validation = self._perform_cross_validation(strategy_result, backtest_result)
            else:
                report.warnings.append("无法进行交叉验证，缺少必要数据")
            
            # 4. 计算综合验证结果
            report.overall_result, report.confidence_score = self._calculate_overall_result(report)
            
            # 5. 生成建议
            report.recommendations = self._generate_recommendations(report)
            
            # 6. 更新统计
            report.execution_time = time.time() - start_time
            self._update_validation_stats(report)
            
            self.logger.info(f"验证完成，结果: {report.overall_result.value}, 置信度: {report.confidence_score:.2f}")
            return report
            
        except Exception as e:
            self.logger.error(f"验证过程失败: {e}")
            report.overall_result = ValidationResult.FAIL
            report.warnings.append(f"验证过程异常: {str(e)}")
            report.execution_time = time.time() - start_time
            return report
    
    def _validate_strategy_result(self, strategy_result: Dict[str, Any]) -> Dict[str, Any]:
        """验证策略选股结果"""
        validation = {
            'score_check': False,
            'recommendation_check': False,
            'indicator_check': False,
            'confidence_check': False,
            'issues': []
        }
        
        try:
            # 检查评分
            score = strategy_result.get('score', 0)
            if isinstance(score, (list, tuple)):
                score = score[0] if score else 0
            
            if score >= self.criteria.min_score_threshold:
                validation['score_check'] = True
            else:
                validation['issues'].append(f"策略评分过低: {score}")
            
            # 检查推荐
            recommendation = strategy_result.get('recommendation', '')
            if recommendation in ['买入', 'buy', '强烈买入']:
                validation['recommendation_check'] = True
            elif recommendation in ['持有', 'hold']:
                validation['recommendation_check'] = True
                validation['issues'].append("推荐为持有，需谨慎")
            else:
                validation['issues'].append(f"推荐不明确: {recommendation}")
            
            # 检查指标匹配
            match_details = strategy_result.get('match_details', {})
            passing_indicators = match_details.get('passing_indicators', [])
            failing_indicators = match_details.get('failing_indicators', [])
            
            if len(passing_indicators) > len(failing_indicators):
                validation['indicator_check'] = True
            else:
                validation['issues'].append("通过指标数量不足")
            
            # 计算置信度
            total_indicators = len(passing_indicators) + len(failing_indicators)
            if total_indicators > 0:
                confidence = len(passing_indicators) / total_indicators
                validation['confidence_check'] = confidence >= self.criteria.min_confidence_level
                if not validation['confidence_check']:
                    validation['issues'].append(f"指标置信度过低: {confidence:.2f}")
            
        except Exception as e:
            validation['issues'].append(f"策略结果验证异常: {str(e)}")
        
        return validation
    
    def _validate_backtest_result(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """验证买点回测结果"""
        validation = {
            'signal_quality_check': False,
            'confidence_check': False,
            'pattern_check': False,
            'consistency_check': False,
            'issues': []
        }
        
        try:
            # 检查买点信号质量
            buypoint_signals = backtest_result.get('buypoint_signals', [])
            if buypoint_signals:
                high_quality_signals = [s for s in buypoint_signals if s.confidence > 0.7]
                if high_quality_signals:
                    validation['signal_quality_check'] = True
                else:
                    validation['issues'].append("缺少高质量买点信号")
            else:
                validation['issues'].append("无买点信号")
            
            # 检查整体置信度
            if buypoint_signals:
                avg_confidence = sum(s.confidence for s in buypoint_signals) / len(buypoint_signals)
                validation['confidence_check'] = avg_confidence >= self.criteria.min_confidence_level
                if not validation['confidence_check']:
                    validation['issues'].append(f"买点置信度过低: {avg_confidence:.2f}")
            
            # 检查回测汇总
            backtest_summary = backtest_result.get('backtest_summary', {})
            if isinstance(backtest_summary, dict):
                success_rate = backtest_summary.get('success_rate', 0)
                if success_rate >= 0.6:
                    validation['consistency_check'] = True
                else:
                    validation['issues'].append(f"回测成功率过低: {success_rate:.2f}")
            
        except Exception as e:
            validation['issues'].append(f"回测结果验证异常: {str(e)}")
        
        return validation
    
    def _perform_cross_validation(self, strategy_result: Dict[str, Any], 
                                 backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """执行交叉验证"""
        cross_validation = {
            'score_consistency': False,
            'signal_alignment': False,
            'recommendation_agreement': False,
            'overall_consistency': False,
            'consistency_score': 0.0,
            'issues': []
        }
        
        try:
            # 1. 评分一致性检查
            strategy_score = strategy_result.get('score', 0)
            if isinstance(strategy_score, (list, tuple)):
                strategy_score = strategy_score[0] if strategy_score else 0
            
            # 从买点信号计算回测评分
            buypoint_signals = backtest_result.get('buypoint_signals', [])
            if buypoint_signals:
                backtest_score = sum(s.confidence for s in buypoint_signals) / len(buypoint_signals) * 100
            else:
                backtest_score = 0
            
            score_diff = abs(strategy_score - backtest_score)
            if score_diff <= self.criteria.max_score_difference:
                cross_validation['score_consistency'] = True
            else:
                cross_validation['issues'].append(f"评分差异过大: {score_diff:.1f}")
            
            # 2. 信号一致性检查
            strategy_recommendation = strategy_result.get('recommendation', '')
            if buypoint_signals and strategy_recommendation in ['买入', 'buy', '强烈买入']:
                cross_validation['signal_alignment'] = True
            elif not buypoint_signals and strategy_recommendation in ['卖出', 'sell', '观望']:
                cross_validation['signal_alignment'] = True
            else:
                cross_validation['issues'].append("策略推荐与买点信号不一致")
            
            # 3. 推荐一致性
            if self.criteria.require_signal_consistency:
                if cross_validation['signal_alignment']:
                    cross_validation['recommendation_agreement'] = True
                else:
                    cross_validation['issues'].append("推荐不一致")
            else:
                cross_validation['recommendation_agreement'] = True
            
            # 4. 计算一致性评分
            consistency_factors = [
                cross_validation['score_consistency'],
                cross_validation['signal_alignment'],
                cross_validation['recommendation_agreement']
            ]
            cross_validation['consistency_score'] = sum(consistency_factors) / len(consistency_factors)
            cross_validation['overall_consistency'] = cross_validation['consistency_score'] >= 0.7
            
        except Exception as e:
            cross_validation['issues'].append(f"交叉验证异常: {str(e)}")
        
        return cross_validation
    
    def _calculate_overall_result(self, report: ValidationReport) -> Tuple[ValidationResult, float]:
        """计算综合验证结果"""
        try:
            # 收集所有验证结果
            validation_scores = []
            
            # 策略验证评分
            if report.strategy_validation:
                strategy_checks = [
                    report.strategy_validation.get('score_check', False),
                    report.strategy_validation.get('recommendation_check', False),
                    report.strategy_validation.get('indicator_check', False),
                    report.strategy_validation.get('confidence_check', False)
                ]
                strategy_score = sum(strategy_checks) / len(strategy_checks)
                validation_scores.append(strategy_score * self.criteria.weight_strategy_result)
            
            # 回测验证评分
            if report.backtest_validation:
                backtest_checks = [
                    report.backtest_validation.get('signal_quality_check', False),
                    report.backtest_validation.get('confidence_check', False),
                    report.backtest_validation.get('consistency_check', False)
                ]
                backtest_score = sum(backtest_checks) / len(backtest_checks)
                validation_scores.append(backtest_score * self.criteria.weight_backtest_result)
            
            # 交叉验证评分
            if report.cross_validation:
                cross_score = report.cross_validation.get('consistency_score', 0.0)
                validation_scores.append(cross_score * 0.3)  # 交叉验证权重
            
            # 计算综合置信度
            if validation_scores:
                confidence = sum(validation_scores) / len(validation_scores)
            else:
                confidence = 0.0
            
            # 确定验证结果
            if confidence >= 0.8:
                result = ValidationResult.PASS
            elif confidence >= 0.6:
                result = ValidationResult.PARTIAL
            elif confidence >= 0.3:
                result = ValidationResult.INCONCLUSIVE
            else:
                result = ValidationResult.FAIL
            
            return result, confidence
            
        except Exception as e:
            self.logger.error(f"计算综合结果失败: {e}")
            return ValidationResult.FAIL, 0.0
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """生成验证建议"""
        recommendations = []
        
        try:
            if report.overall_result == ValidationResult.PASS:
                recommendations.append("验证通过，可以考虑投资")
                if report.confidence_score >= 0.9:
                    recommendations.append("高置信度，强烈推荐")
            elif report.overall_result == ValidationResult.PARTIAL:
                recommendations.append("部分验证通过，需要谨慎考虑")
                recommendations.append("建议进一步分析风险因素")
            elif report.overall_result == ValidationResult.INCONCLUSIVE:
                recommendations.append("验证结果不明确，建议观望")
                recommendations.append("需要更多数据支持决策")
            else:
                recommendations.append("验证失败，不建议投资")
                recommendations.append("建议重新评估投资策略")
            
            # 基于具体问题的建议
            all_issues = []
            if report.strategy_validation:
                all_issues.extend(report.strategy_validation.get('issues', []))
            if report.backtest_validation:
                all_issues.extend(report.backtest_validation.get('issues', []))
            if report.cross_validation:
                all_issues.extend(report.cross_validation.get('issues', []))
            
            if "评分过低" in str(all_issues):
                recommendations.append("关注基本面分析")
            if "置信度过低" in str(all_issues):
                recommendations.append("增加技术指标验证")
            if "不一致" in str(all_issues):
                recommendations.append("检查数据质量和分析逻辑")
            
        except Exception as e:
            self.logger.error(f"生成建议失败: {e}")
            recommendations.append("验证过程异常，建议人工复核")
        
        return recommendations
    
    def _update_validation_stats(self, report: ValidationReport):
        """更新验证统计"""
        try:
            self.validation_stats['total_validations'] += 1
            
            if report.overall_result == ValidationResult.PASS:
                self.validation_stats['passed_validations'] += 1
            elif report.overall_result == ValidationResult.PARTIAL:
                self.validation_stats['partial_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
            
            # 更新平均置信度
            total = self.validation_stats['total_validations']
            current_avg = self.validation_stats['average_confidence']
            self.validation_stats['average_confidence'] = (
                (current_avg * (total - 1) + report.confidence_score) / total
            )
            
            # 更新平均执行时间
            current_avg_time = self.validation_stats['average_execution_time']
            self.validation_stats['average_execution_time'] = (
                (current_avg_time * (total - 1) + report.execution_time) / total
            )
            
        except Exception as e:
            self.logger.error(f"更新统计失败: {e}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """获取验证统计"""
        stats = self.validation_stats.copy()
        if stats['total_validations'] > 0:
            stats['pass_rate'] = stats['passed_validations'] / stats['total_validations']
            stats['partial_rate'] = stats['partial_validations'] / stats['total_validations']
            stats['fail_rate'] = stats['failed_validations'] / stats['total_validations']
        else:
            stats['pass_rate'] = 0.0
            stats['partial_rate'] = 0.0
            stats['fail_rate'] = 0.0
        
        return stats
