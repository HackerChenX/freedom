#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版买点策略验证工具

用于验证买点分析生成的选股策略的基本结构和逻辑
"""

import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SimpleStrategyValidator:
    """简化版策略验证器"""
    
    def __init__(self):
        """初始化验证器"""
        pass
        
    def validate_strategy(self, strategy_file: str) -> dict:
        """
        验证买点策略的基本结构和逻辑
        
        Args:
            strategy_file: 策略文件路径
            
        Returns:
            dict: 验证结果
        """
        print(f"开始验证策略: {strategy_file}")
        
        try:
            # 1. 加载策略
            strategy = self._load_strategy(strategy_file)
            
            # 2. 验证策略结构
            structure_validation = self._validate_structure(strategy)
            
            # 3. 分析策略条件
            condition_analysis = self._analyze_conditions(strategy)
            
            # 4. 评估策略复杂度
            complexity_assessment = self._assess_complexity(strategy)
            
            # 5. 生成验证报告
            validation_results = {
                "strategy_info": {
                    "name": strategy.get("name", "未知策略"),
                    "description": strategy.get("description", ""),
                    "version": strategy.get("version", "1.0"),
                    "condition_count": len(strategy.get("conditions", [])),
                    "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "structure_validation": structure_validation,
                "condition_analysis": condition_analysis,
                "complexity_assessment": complexity_assessment,
                "validation_status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            # 6. 生成建议
            recommendations = self._generate_recommendations(validation_results)
            validation_results["recommendations"] = recommendations
            
            print(f"策略验证完成，包含 {len(strategy.get('conditions', []))} 个条件")
            return validation_results
            
        except Exception as e:
            print(f"策略验证失败: {e}")
            return {
                "validation_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_strategy(self, strategy_file: str) -> dict:
        """加载策略文件"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            print(f"成功加载策略: {strategy.get('name', '未知')}")
            return strategy
            
        except Exception as e:
            raise Exception(f"加载策略文件失败: {e}")
    
    def _validate_structure(self, strategy: dict) -> dict:
        """验证策略结构"""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # 检查必需字段
        required_fields = ["name", "conditions"]
        for field in required_fields:
            if field not in strategy:
                validation["is_valid"] = False
                validation["issues"].append(f"缺少必需字段: {field}")
        
        # 检查条件格式
        conditions = strategy.get("conditions", [])
        if not isinstance(conditions, list):
            validation["is_valid"] = False
            validation["issues"].append("conditions字段必须是列表")
        elif len(conditions) == 0:
            validation["warnings"].append("策略没有任何条件")
        
        # 检查条件结构
        for i, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                validation["issues"].append(f"条件 {i+1} 格式错误：必须是字典")
                continue
            
            # 检查条件必需字段
            required_condition_fields = ["type", "indicator"]
            for field in required_condition_fields:
                if field not in condition:
                    validation["warnings"].append(f"条件 {i+1} 缺少字段: {field}")
        
        return validation
    
    def _analyze_conditions(self, strategy: dict) -> dict:
        """分析策略条件"""
        conditions = strategy.get("conditions", [])
        
        analysis = {
            "total_conditions": len(conditions),
            "indicator_distribution": {},
            "period_distribution": {},
            "pattern_distribution": {},
            "type_distribution": {}
        }
        
        # 统计指标分布
        indicators = [c.get("indicator", "unknown") for c in conditions]
        analysis["indicator_distribution"] = dict(Counter(indicators))
        
        # 统计周期分布
        periods = [c.get("period", "unknown") for c in conditions]
        analysis["period_distribution"] = dict(Counter(periods))
        
        # 统计形态分布
        patterns = [c.get("pattern", "unknown") for c in conditions]
        analysis["pattern_distribution"] = dict(Counter(patterns))
        
        # 统计类型分布
        types = [c.get("type", "unknown") for c in conditions]
        analysis["type_distribution"] = dict(Counter(types))
        
        # 计算多样性指标
        analysis["indicator_diversity"] = len(set(indicators))
        analysis["period_diversity"] = len(set(periods))
        analysis["pattern_diversity"] = len(set(patterns))
        
        return analysis
    
    def _assess_complexity(self, strategy: dict) -> dict:
        """评估策略复杂度"""
        conditions = strategy.get("conditions", [])
        
        assessment = {
            "complexity_level": "medium",
            "condition_count": len(conditions),
            "logic_type": strategy.get("condition_logic", "OR"),
            "score": 0
        }
        
        # 基于条件数量评估复杂度
        condition_count = len(conditions)
        if condition_count < 50:
            assessment["complexity_level"] = "low"
            assessment["score"] += 1
        elif condition_count < 200:
            assessment["complexity_level"] = "medium"
            assessment["score"] += 2
        elif condition_count < 500:
            assessment["complexity_level"] = "high"
            assessment["score"] += 3
        else:
            assessment["complexity_level"] = "very_high"
            assessment["score"] += 4
        
        # 基于指标多样性评估
        indicators = set(c.get("indicator", "") for c in conditions)
        indicator_diversity = len(indicators)
        
        if indicator_diversity > 20:
            assessment["indicator_diversity"] = "high"
            assessment["score"] += 1
        elif indicator_diversity > 10:
            assessment["indicator_diversity"] = "medium"
        else:
            assessment["indicator_diversity"] = "low"
            assessment["score"] -= 1
        
        # 基于周期多样性评估
        periods = set(c.get("period", "") for c in conditions)
        period_diversity = len(periods)
        
        if period_diversity >= 5:
            assessment["period_diversity"] = "high"
            assessment["score"] += 1
        elif period_diversity >= 3:
            assessment["period_diversity"] = "medium"
        else:
            assessment["period_diversity"] = "low"
        
        # 综合评分
        assessment["overall_score"] = max(0, min(10, assessment["score"]))
        
        return assessment
    
    def _generate_recommendations(self, validation_results: dict) -> list:
        """生成优化建议"""
        recommendations = []
        
        # 结构验证建议
        structure = validation_results.get("structure_validation", {})
        if not structure.get("is_valid", True):
            recommendations.append("修复策略结构问题，确保包含所有必需字段")
        
        if structure.get("warnings"):
            recommendations.append("注意策略结构警告，建议完善条件定义")
        
        # 条件分析建议
        analysis = validation_results.get("condition_analysis", {})
        condition_count = analysis.get("total_conditions", 0)
        
        if condition_count > 1000:
            recommendations.append("条件数量过多，建议精简策略以提高执行效率")
        elif condition_count < 10:
            recommendations.append("条件数量较少，可能导致选股过于宽泛")
        
        # 多样性建议
        indicator_diversity = analysis.get("indicator_diversity", 0)
        if indicator_diversity < 5:
            recommendations.append("指标多样性不足，建议增加不同类型的技术指标")
        
        period_diversity = analysis.get("period_diversity", 0)
        if period_diversity < 3:
            recommendations.append("周期多样性不足，建议增加不同时间周期的分析")
        
        # 复杂度建议
        complexity = validation_results.get("complexity_assessment", {})
        complexity_level = complexity.get("complexity_level", "medium")
        
        if complexity_level == "very_high":
            recommendations.append("策略复杂度过高，建议简化条件以提高可维护性")
        elif complexity_level == "low":
            recommendations.append("策略可能过于简单，建议增加更多筛选条件")
        
        # 逻辑建议
        logic_type = complexity.get("logic_type", "OR")
        if logic_type == "OR" and condition_count > 100:
            recommendations.append("使用OR逻辑且条件较多，可能导致选股过于宽泛")
        
        return recommendations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版买点策略验证工具")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--output", help="输出目录，默认为results/simple_validation")
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else Path("results/simple_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建验证器并执行验证
    validator = SimpleStrategyValidator()
    
    print(f"开始验证策略: {args.strategy}")
    print("-" * 50)
    
    results = validator.validate_strategy(args.strategy)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"validation_result_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成简化报告
    report_file = output_dir / f"validation_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("策略验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        strategy_info = results.get("strategy_info", {})
        f.write(f"策略名称: {strategy_info.get('name', 'N/A')}\n")
        f.write(f"策略描述: {strategy_info.get('description', 'N/A')}\n")
        f.write(f"条件数量: {strategy_info.get('condition_count', 0)}\n")
        f.write(f"验证时间: {strategy_info.get('validation_date', 'N/A')}\n\n")
        
        # 结构验证
        structure = results.get("structure_validation", {})
        f.write("结构验证:\n")
        f.write(f"  状态: {'通过' if structure.get('is_valid', False) else '失败'}\n")
        if structure.get("issues"):
            f.write("  问题:\n")
            for issue in structure["issues"]:
                f.write(f"    - {issue}\n")
        if structure.get("warnings"):
            f.write("  警告:\n")
            for warning in structure["warnings"]:
                f.write(f"    - {warning}\n")
        f.write("\n")
        
        # 条件分析
        analysis = results.get("condition_analysis", {})
        f.write("条件分析:\n")
        f.write(f"  总条件数: {analysis.get('total_conditions', 0)}\n")
        f.write(f"  指标多样性: {analysis.get('indicator_diversity', 0)}\n")
        f.write(f"  周期多样性: {analysis.get('period_diversity', 0)}\n")
        f.write(f"  形态多样性: {analysis.get('pattern_diversity', 0)}\n\n")
        
        # 复杂度评估
        complexity = results.get("complexity_assessment", {})
        f.write("复杂度评估:\n")
        f.write(f"  复杂度级别: {complexity.get('complexity_level', 'N/A')}\n")
        f.write(f"  逻辑类型: {complexity.get('logic_type', 'N/A')}\n")
        f.write(f"  综合评分: {complexity.get('overall_score', 0)}/10\n\n")
        
        # 建议
        recommendations = results.get("recommendations", [])
        if recommendations:
            f.write("优化建议:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"  {i}. {rec}\n")
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📊 验证结果摘要")
    print("=" * 60)
    
    if results["validation_status"] == "success":
        strategy_info = results["strategy_info"]
        structure = results["structure_validation"]
        analysis = results["condition_analysis"]
        complexity = results["complexity_assessment"]
        
        print(f"策略名称: {strategy_info['name']}")
        print(f"条件数量: {strategy_info['condition_count']}")
        print(f"结构验证: {'✅ 通过' if structure.get('is_valid', False) else '❌ 失败'}")
        print(f"指标多样性: {analysis.get('indicator_diversity', 0)} 种")
        print(f"周期多样性: {analysis.get('period_diversity', 0)} 种")
        print(f"复杂度级别: {complexity.get('complexity_level', 'N/A')}")
        print(f"综合评分: {complexity.get('overall_score', 0)}/10")
        
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\n建议 ({len(recommendations)} 条):")
            for i, rec in enumerate(recommendations[:5], 1):  # 只显示前5条
                print(f"  {i}. {rec}")
            if len(recommendations) > 5:
                print(f"  ... 还有 {len(recommendations) - 5} 条建议")
    else:
        print(f"验证失败: {results.get('error', '未知错误')}")
    
    print(f"\n详细结果已保存到:")
    print(f"  JSON: {result_file}")
    print(f"  报告: {report_file}")


if __name__ == "__main__":
    main()
