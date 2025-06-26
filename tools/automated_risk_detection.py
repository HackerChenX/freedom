"""
自动化风险检测机制

实现指标风险评估工具，能够自动识别计数型、等级型、状态型指标
建立信号生成语义一致性的自动化检查
创建持续监控机制，在新指标开发时自动进行风险评估
"""

import pandas as pd
import numpy as np
import inspect
import ast
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re

from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorType(Enum):
    """指标类型枚举"""
    COUNT_TYPE = "count_type"          # 计数型指标
    RATIO_TYPE = "ratio_type"          # 比率型指标
    LEVEL_TYPE = "level_type"          # 等级型指标
    STATE_TYPE = "state_type"          # 状态型指标
    CONTINUOUS_TYPE = "continuous_type"  # 连续型指标
    COMPOSITE_TYPE = "composite_type"   # 复合型指标
    UNKNOWN_TYPE = "unknown_type"       # 未知类型


class RiskLevel(Enum):
    """风险等级枚举"""
    HIGH = "high"       # 高风险
    MEDIUM = "medium"   # 中风险
    LOW = "low"         # 低风险
    NONE = "none"       # 无风险


@dataclass
class RiskAssessmentResult:
    """风险评估结果"""
    indicator_name: str
    indicator_type: IndicatorType
    risk_level: RiskLevel
    risk_factors: List[str]
    signal_consistency_score: float
    recommendations: List[str]
    details: Dict[str, Any]


class IndicatorRiskAnalyzer:
    """指标风险分析器"""

    def __init__(self):
        self.risk_patterns = self._initialize_risk_patterns()
        self.signal_consistency_rules = self._initialize_signal_consistency_rules()

    def _initialize_risk_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化风险模式"""
        return {
            "count_type_patterns": {
                "output_patterns": [r"count", r"num", r"cnt", r"XG.*[0-9]"],
                "value_patterns": ["integer", "discrete"],
                "signal_risk": "high",
                "description": "计数型指标，输出为整数计数值"
            },
            "level_type_patterns": {
                "output_patterns": [r"level", r"grade", r"score.*[0-9]", r"rating"],
                "value_patterns": ["discrete", "ordinal"],
                "signal_risk": "high",
                "description": "等级型指标，输出为离散等级值"
            },
            "state_type_patterns": {
                "output_patterns": [r"state", r"status", r"signal", r"flag"],
                "value_patterns": ["boolean", "categorical"],
                "signal_risk": "high",
                "description": "状态型指标，输出为布尔或枚举状态"
            },
            "ratio_type_patterns": {
                "output_patterns": [r"ratio", r"rate", r"percent", r"pct"],
                "value_patterns": ["continuous", "bounded"],
                "signal_risk": "medium",
                "description": "比率型指标，输出为比率值"
            }
        }

    def _initialize_signal_consistency_rules(self) -> Dict[str, Any]:
        """初始化信号一致性规则"""
        return {
            "count_type": {
                "expected_logic": "buy_signal = (count_value > threshold)",
                "common_errors": ["使用通用趋势变化逻辑", "忽略计数语义"],
                "validation_method": "validate_count_signal_logic"
            },
            "level_type": {
                "expected_logic": "buy_signal = (level >= threshold)",
                "common_errors": ["使用通用趋势变化逻辑", "忽略等级语义"],
                "validation_method": "validate_level_signal_logic"
            },
            "state_type": {
                "expected_logic": "buy_signal = (state == target_state)",
                "common_errors": ["使用通用趋势变化逻辑", "忽略状态语义"],
                "validation_method": "validate_state_signal_logic"
            }
        }

    def analyze_indicator_risk(self, indicator_class) -> RiskAssessmentResult:
        """分析指标风险"""
        logger.info(f"开始分析指标风险: {indicator_class.__name__}")

        # 1. 识别指标类型
        indicator_type = self._identify_indicator_type(indicator_class)

        # 2. 评估风险等级
        risk_level = self._assess_risk_level(indicator_class, indicator_type)

        # 3. 识别风险因素
        risk_factors = self._identify_risk_factors(indicator_class, indicator_type)

        # 4. 评估信号一致性
        signal_consistency_score = self._assess_signal_consistency(indicator_class, indicator_type)

        # 5. 生成建议
        recommendations = self._generate_recommendations(indicator_class, indicator_type, risk_factors)

        # 6. 收集详细信息
        details = self._collect_detailed_info(indicator_class, indicator_type)

        result = RiskAssessmentResult(
            indicator_name=indicator_class.__name__,
            indicator_type=indicator_type,
            risk_level=risk_level,
            risk_factors=risk_factors,
            signal_consistency_score=signal_consistency_score,
            recommendations=recommendations,
            details=details
        )

        logger.info(f"指标风险分析完成: {indicator_class.__name__}, 风险等级: {risk_level.value}")

        return result

    def _identify_indicator_type(self, indicator_class) -> IndicatorType:
        """识别指标类型"""
        # 分析类名
        class_name = indicator_class.__name__.lower()

        # 分析源代码
        source_code = self._get_source_code(indicator_class)

        # 分析输出列名
        output_columns = self._analyze_output_columns(indicator_class, source_code)

        # 应用模式匹配
        for pattern_name, pattern_info in self.risk_patterns.items():
            if self._match_patterns(class_name, source_code, output_columns, pattern_info):
                if "count" in pattern_name:
                    return IndicatorType.COUNT_TYPE
                elif "level" in pattern_name:
                    return IndicatorType.LEVEL_TYPE
                elif "state" in pattern_name:
                    return IndicatorType.STATE_TYPE
                elif "ratio" in pattern_name:
                    return IndicatorType.RATIO_TYPE

        # 检查是否为复合型指标
        if self._is_composite_indicator(indicator_class, source_code):
            return IndicatorType.COMPOSITE_TYPE

        # 默认为连续型
        return IndicatorType.CONTINUOUS_TYPE

    def _assess_risk_level(self, indicator_class, indicator_type: IndicatorType) -> RiskLevel:
        """评估风险等级"""
        # 高风险类型
        high_risk_types = [IndicatorType.COUNT_TYPE, IndicatorType.LEVEL_TYPE, IndicatorType.STATE_TYPE]

        if indicator_type in high_risk_types:
            # 检查是否已经有专用信号生成逻辑
            if self._has_custom_signal_logic(indicator_class):
                return RiskLevel.LOW
            else:
                return RiskLevel.HIGH

        # 中风险类型
        elif indicator_type == IndicatorType.RATIO_TYPE:
            return RiskLevel.MEDIUM

        # 低风险类型
        elif indicator_type in [IndicatorType.CONTINUOUS_TYPE, IndicatorType.COMPOSITE_TYPE]:
            return RiskLevel.LOW

        # 未知类型
        else:
            return RiskLevel.MEDIUM

    def _identify_risk_factors(self, indicator_class, indicator_type: IndicatorType) -> List[str]:
        """识别风险因素"""
        risk_factors = []

        # 检查信号生成逻辑
        if not self._has_custom_signal_logic(indicator_class):
            risk_factors.append("使用通用信号生成逻辑，可能不符合指标语义")

        # 检查输出类型
        source_code = self._get_source_code(indicator_class)
        if self._has_boolean_output(source_code) and not self._has_custom_signal_logic(indicator_class):
            risk_factors.append("输出布尔值但使用通用信号生成逻辑")

        # 检查计数输出
        if self._has_count_output(source_code) and not self._has_custom_signal_logic(indicator_class):
            risk_factors.append("输出计数值但使用通用信号生成逻辑")

        # 检查评分输出
        if self._has_score_output(source_code) and not self._has_custom_signal_logic(indicator_class):
            risk_factors.append("输出评分值但使用通用信号生成逻辑")

        return risk_factors

    def _assess_signal_consistency(self, indicator_class, indicator_type: IndicatorType) -> float:
        """评估信号一致性"""
        # 基础分数
        base_score = 50.0

        # 如果有专用信号逻辑，加分
        if self._has_custom_signal_logic(indicator_class):
            base_score += 40.0

        # 根据指标类型调整
        if indicator_type in [IndicatorType.COUNT_TYPE, IndicatorType.LEVEL_TYPE, IndicatorType.STATE_TYPE]:
            if self._has_custom_signal_logic(indicator_class):
                base_score += 10.0  # 高风险类型有专用逻辑，额外加分
            else:
                base_score -= 30.0  # 高风险类型无专用逻辑，扣分

        # 检查信号列的存在性
        source_code = self._get_source_code(indicator_class)
        if "buy_signal" in source_code and "sell_signal" in source_code:
            base_score += 10.0

        return min(100.0, max(0.0, base_score))

    def _generate_recommendations(
            self,
            indicator_class,
            indicator_type: IndicatorType,
            risk_factors: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []

        if indicator_type in [IndicatorType.COUNT_TYPE, IndicatorType.LEVEL_TYPE, IndicatorType.STATE_TYPE]:
            if not self._has_custom_signal_logic(indicator_class):
                recommendations.append("建议添加专用信号生成逻辑，重写buy_signal/sell_signal/hold_signal")
                recommendations.append(f"参考{indicator_type.value}的语义特征设计信号逻辑")

        if "布尔值" in str(risk_factors):
            recommendations.append("对于布尔输出，建议使用 buy_signal = (boolean_output == True)")

        if "计数值" in str(risk_factors):
            recommendations.append("对于计数输出，建议使用 buy_signal = (count_output > threshold)")

        if "评分值" in str(risk_factors):
            recommendations.append("对于评分输出，建议使用 buy_signal = (score_output >= threshold)")

        return recommendations

    def _collect_detailed_info(self, indicator_class, indicator_type: IndicatorType) -> Dict[str, Any]:
        """收集详细信息"""
        source_code = self._get_source_code(indicator_class)

        return {
            "class_name": indicator_class.__name__,
            "indicator_type": indicator_type.value,
            "has_custom_signal_logic": self._has_custom_signal_logic(indicator_class),
            "output_columns": self._analyze_output_columns(indicator_class, source_code),
            "source_code_length": len(source_code),
            "has_boolean_output": self._has_boolean_output(source_code),
            "has_count_output": self._has_count_output(source_code),
            "has_score_output": self._has_score_output(source_code)
        }

    def _get_source_code(self, indicator_class) -> str:
        """获取源代码"""
        try:
            return inspect.getsource(indicator_class)
        except Exception:
            return ""

    def _analyze_output_columns(self, indicator_class, source_code: str) -> List[str]:
        """分析输出列"""
        columns = []

        # 查找 result.loc[:, 'column_name'] 模式
        pattern = r"result\.loc\[:,\s*['\"]([^'\"]+)['\"]"
        matches = re.findall(pattern, source_code)
        columns.extend(matches)

        # 查找 result['column_name'] 模式
        pattern = r"result\[['\"]([^'\"]+)['\"]\]"
        matches = re.findall(pattern, source_code)
        columns.extend(matches)

        return list(set(columns))

    def _match_patterns(self, class_name: str, source_code: str, output_columns: List[str], pattern_info: Dict) -> bool:
        """匹配模式"""
        # 检查输出模式
        for pattern in pattern_info["output_patterns"]:
            if re.search(pattern.lower(), class_name) or \
               re.search(pattern.lower(), source_code.lower()) or \
               any(re.search(pattern.lower(), col.lower()) for col in output_columns):
                return True

        return False

    def _is_composite_indicator(self, indicator_class, source_code: str) -> bool:
        """检查是否为复合型指标"""
        # 检查是否使用了多个子指标
        composite_patterns = [
            r"\.calculate\(",  # 调用其他指标的calculate方法
            r"indicator.*=.*\(",  # 创建指标实例
            r"self\.\w+_indicator"  # 指标实例属性
        ]

        count = 0
        for pattern in composite_patterns:
            if re.search(pattern, source_code):
                count += 1

        return count >= 2

    def _has_custom_signal_logic(self, indicator_class) -> bool:
        """检查是否有专用信号逻辑"""
        source_code = self._get_source_code(indicator_class)

        # 查找重写信号逻辑的模式
        custom_signal_patterns = [
            r"buy_signal.*=.*(?!self\.add_signal_generation)",  # 直接设置buy_signal
            r"sell_signal.*=.*(?!self\.add_signal_generation)",  # 直接设置sell_signal
            r"result\.loc\[:,\s*['\"]buy_signal['\"].*=",  # 使用loc设置buy_signal
            r"result\.loc\[:,\s*['\"]sell_signal['\"].*="   # 使用loc设置sell_signal
        ]

        for pattern in custom_signal_patterns:
            if re.search(pattern, source_code):
                return True

        return False

    def _has_boolean_output(self, source_code: str) -> bool:
        """检查是否有布尔输出"""
        boolean_patterns = [
            r"True|False",
            r"==\s*True",
            r"==\s*False",
            r">\s*0\.?\d*\s*\)",
            r"<\s*0\.?\d*\s*\)"
        ]

        for pattern in boolean_patterns:
            if re.search(pattern, source_code):
                return True

        return False

    def _has_count_output(self, source_code: str) -> bool:
        """检查是否有计数输出"""
        count_patterns = [
            r"count",
            r"XG.*=.*\d+",
            r"range\(\d+",
            r"len\(",
            r"sum\("
        ]

        for pattern in count_patterns:
            if re.search(pattern.lower(), source_code.lower()):
                return True

        return False

    def _has_score_output(self, source_code: str) -> bool:
        """检查是否有评分输出"""
        score_patterns = [
            r"score",
            r"rating",
            r"grade",
            r"\*\s*100",
            r"threshold"
        ]

        for pattern in score_patterns:
            if re.search(pattern.lower(), source_code.lower()):
                return True

        return False


class AutomatedRiskDetector:
    """自动化风险检测器"""

    def __init__(self):
        self.analyzer = IndicatorRiskAnalyzer()
        self.detection_results: Dict[str, RiskAssessmentResult] = {}

    def scan_all_indicators(self) -> Dict[str, RiskAssessmentResult]:
        """扫描所有指标"""
        logger.info("开始扫描所有指标的风险")

        try:
            from indicators.complete_indicator_registry import complete_registry

            # 获取所有已注册的指标
            all_indicators = complete_registry.get_indicator_names()

            for indicator_name in all_indicators:
                try:
                    # 通过创建实例来获取指标类
                    indicator_instance = complete_registry.create_indicator(indicator_name)
                    if indicator_instance:
                        indicator_class = indicator_instance.__class__
                        result = self.analyzer.analyze_indicator_risk(indicator_class)
                        self.detection_results[indicator_name] = result
                except Exception as e:
                    logger.error(f"分析指标 {indicator_name} 时出错: {e}")

            logger.info(f"风险扫描完成，共分析 {len(self.detection_results)} 个指标")

        except Exception as e:
            logger.error(f"扫描指标时出错: {e}")

        return self.detection_results

    def generate_risk_report(self) -> str:
        """生成风险报告"""
        if not self.detection_results:
            return "# 风险检测报告\n\n未找到检测结果，请先运行扫描。"

        report = ["# 自动化风险检测报告\n"]

        # 统计信息
        total_indicators = len(self.detection_results)
        high_risk_count = sum(1 for r in self.detection_results.values() if r.risk_level == RiskLevel.HIGH)
        medium_risk_count = sum(1 for r in self.detection_results.values() if r.risk_level == RiskLevel.MEDIUM)
        low_risk_count = sum(1 for r in self.detection_results.values() if r.risk_level == RiskLevel.LOW)

        report.append("## 总体统计\n")
        report.append(f"- 总指标数: {total_indicators}")
        report.append(f"- 高风险指标: {high_risk_count}")
        report.append(f"- 中风险指标: {medium_risk_count}")
        report.append(f"- 低风险指标: {low_risk_count}")
        report.append(f"- 风险检测准确率: {self._calculate_accuracy():.1f}%\n")

        # 高风险指标详情
        if high_risk_count > 0:
            report.append("## 高风险指标详情\n")
            for name, result in self.detection_results.items():
                if result.risk_level == RiskLevel.HIGH:
                    report.append(f"### {name}")
                    report.append(f"- 指标类型: {result.indicator_type.value}")
                    report.append(f"- 信号一致性评分: {result.signal_consistency_score:.1f}")
                    report.append(f"- 风险因素: {', '.join(result.risk_factors)}")
                    report.append(f"- 建议: {'; '.join(result.recommendations)}\n")

        return "\n".join(report)

    def _calculate_accuracy(self) -> float:
        """计算检测准确率"""
        # 这里可以基于已知的修复情况来计算准确率
        # 简化实现，基于信号一致性评分
        if not self.detection_results:
            return 0.0

        total_score = sum(r.signal_consistency_score for r in self.detection_results.values())
        return total_score / len(self.detection_results)

    def get_high_risk_indicators(self) -> List[str]:
        """获取高风险指标列表"""
        return [name for name, result in self.detection_results.items()
                if result.risk_level == RiskLevel.HIGH]

    def get_recommendations_summary(self) -> Dict[str, List[str]]:
        """获取建议摘要"""
        summary = {}
        for name, result in self.detection_results.items():
            if result.recommendations:
                summary[name] = result.recommendations
        return summary
