#!/usr/bin/env python3
"""
真正闭环验证综合报告生成器

汇总所有指标的真正闭环验证结果，生成综合报告
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


def collect_validation_results() -> Dict[str, Any]:
    """收集所有真正闭环验证结果"""
    results_dir = "data/result"
    pattern = os.path.join(results_dir, "true_closed_loop_*_result.json")
    result_files = glob.glob(pattern)
    
    validation_results = {}
    summary_stats = {
        'total_indicators': 0,
        'passed_indicators': 0,
        'failed_indicators': 0,
        'perfect_consistency_indicators': 0,
        'high_quality_indicators': 0
    }
    
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            indicator_name = result['indicator_name']
            validation_results[indicator_name] = result
            
            # 更新统计
            summary_stats['total_indicators'] += 1
            
            if result.get('closed_loop_verified', False):
                summary_stats['passed_indicators'] += 1
            else:
                summary_stats['failed_indicators'] += 1
            
            if result.get('consistency_rate', 0) >= 1.0:
                summary_stats['perfect_consistency_indicators'] += 1
            
            if result.get('quality_score', 0) >= 0.8:
                summary_stats['high_quality_indicators'] += 1
                
        except Exception as e:
            logger.error(f"读取结果文件失败 {file_path}: {e}")
    
    return validation_results, summary_stats


def generate_markdown_report(validation_results: Dict[str, Any], 
                           summary_stats: Dict[str, Any]) -> str:
    """生成Markdown格式的综合报告"""
    
    report_lines = [
        "# 指标真正闭环验证综合报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 概述",
        "",
        "本报告展示了股票分析系统中各个技术指标的真正闭环验证结果。",
        "真正闭环验证是指：使用指标条件进行选股，然后对选出的股票重新计算指标，",
        "验证计算出的指标是否真的满足选股条件，以确保指标逻辑的一致性。",
        "",
        "## 验证统计",
        "",
        f"- **总验证指标数**: {summary_stats['total_indicators']}",
        f"- **通过验证指标数**: {summary_stats['passed_indicators']}",
        f"- **未通过验证指标数**: {summary_stats['failed_indicators']}",
        f"- **完美一致性指标数**: {summary_stats['perfect_consistency_indicators']}",
        f"- **高质量指标数**: {summary_stats['high_quality_indicators']}",
        "",
        f"**验证通过率**: {summary_stats['passed_indicators'] / summary_stats['total_indicators'] * 100:.1f}%" if summary_stats['total_indicators'] > 0 else "**验证通过率**: 0%",
        "",
        "## 详细结果",
        "",
        "| 指标名称 | 验证状态 | 一致性比率 | 质量评分 | 选股数量 | 验证详情 |",
        "|---------|---------|----------|----------|----------|----------|"
    ]
    
    # 按指标名称排序
    sorted_indicators = sorted(validation_results.keys())
    
    for indicator_name in sorted_indicators:
        result = validation_results[indicator_name]
        
        # 验证状态
        status = "✅ 通过" if result.get('closed_loop_verified', False) else "❌ 未通过"
        
        # 一致性比率
        consistency_rate = result.get('consistency_rate', 0)
        consistency_str = f"{consistency_rate:.1%}"
        
        # 质量评分
        quality_score = result.get('quality_score', 0)
        quality_str = f"{quality_score:.2f}"
        
        # 选股数量
        selection_count = result.get('step1_strategy_selection', {}).get('selection_count', 0)
        
        # 验证详情
        verification_result = result.get('step2_indicator_verification', {})
        verified_stocks = verification_result.get('verified_stocks', 0)
        total_stocks = verification_result.get('total_stocks', 0)
        detail_str = f"{verified_stocks}/{total_stocks}"
        
        report_lines.append(
            f"| {indicator_name} | {status} | {consistency_str} | {quality_str} | {selection_count} | {detail_str} |"
        )
    
    # 添加验证逻辑说明
    report_lines.extend([
        "",
        "## 验证逻辑说明",
        "",
        "### 真正闭环验证流程",
        "",
        "1. **策略选股**: 使用指标的真实条件进行选股",
        "   - MA指标: 5日均线高于收盘价（寻找价格在均线下方的反弹机会）",
        "   - RSI指标: RSI小于40（接近超卖区域）",
        "",
        "2. **指标重算**: 对选出的股票重新计算相同的技术指标",
        "",
        "3. **一致性验证**: 检查重新计算的指标是否真的满足选股条件",
        "",
        "4. **结果评估**: 统计一致性比率和质量评分",
        "",
        "### 评估标准",
        "",
        "- **一致性比率**: 重新验证通过的股票数 / 总选出股票数",
        "- **质量评分**: 综合考虑选股成功性(30%) + 一致性(50%) + 闭环验证通过(20%)",
        "- **通过标准**: 一致性比率 ≥ 80%",
        "",
        "### 验证意义",
        "",
        "真正闭环验证确保了：",
        "1. 指标计算逻辑的正确性",
        "2. 选股策略与指标逻辑的一致性", 
        "3. 系统的可靠性和可信度",
        "",
        "这种验证方式比传统的买点分析更加直接和可靠，",
        "直接验证了'选出的股票是否真的符合指标条件'这一核心问题。"
    ])
    
    # 添加详细验证结果
    if validation_results:
        report_lines.extend([
            "",
            "## 详细验证数据",
            ""
        ])
        
        for indicator_name in sorted_indicators:
            result = validation_results[indicator_name]
            report_lines.extend([
                f"### {indicator_name}指标详细结果",
                "",
                f"**验证时间**: {result.get('timestamp', 'N/A')}",
                f"**验证日期**: {result.get('validation_date', 'N/A')}",
                ""
            ])
            
            # 策略选股结果
            step1 = result.get('step1_strategy_selection', {})
            strategy_config = step1.get('strategy_config', {})
            conditions = strategy_config.get('conditions', [])
            
            report_lines.extend([
                "**步骤1: 策略选股**",
                f"- 股票池大小: {step1.get('stock_pool_size', 0)}",
                f"- 选出股票数: {step1.get('selection_count', 0)}",
                f"- 选股比率: {step1.get('selection_ratio', 0):.1%}",
                f"- 选出股票: {', '.join(step1.get('selected_stocks', []))}",
                ""
            ])
            
            if conditions:
                condition = conditions[0]  # 取第一个条件
                report_lines.extend([
                    "**选股条件**:",
                    f"- 指标类型: {condition.get('indicator_id', 'N/A')}",
                    f"- 周期参数: {condition.get('period', 'N/A')}",
                    f"- 判断字段: {condition.get('field', 'N/A')}",
                    f"- 操作符: {condition.get('operator', 'N/A')}",
                    f"- 目标值: {condition.get('value', condition.get('reference_field', 'N/A'))}",
                    f"- 条件描述: {condition.get('description', 'N/A')}",
                    ""
                ])
            
            # 指标验证结果
            step2 = result.get('step2_indicator_verification', {})
            verification_details = step2.get('verification_details', [])
            
            report_lines.extend([
                "**步骤2: 指标验证**",
                f"- 验证股票总数: {step2.get('total_stocks', 0)}",
                f"- 验证通过股票数: {step2.get('verified_stocks', 0)}",
                ""
            ])
            
            if verification_details:
                for detail in verification_details:
                    stock_code = detail.get('stock_code', 'N/A')
                    all_met = detail.get('all_conditions_met', False)
                    conditions_check = detail.get('conditions_check', [])
                    
                    report_lines.extend([
                        f"**股票 {stock_code} 验证结果**: {'✅ 通过' if all_met else '❌ 未通过'}",
                        ""
                    ])
                    
                    for check in conditions_check:
                        condition_met = check.get('condition_met', False)
                        actual_value = check.get('actual_value', 'N/A')
                        expected_value = check.get('expected_value', 'N/A')
                        operator = check.get('operator', 'N/A')
                        
                        report_lines.extend([
                            f"- 条件检查: {'✅ 满足' if condition_met else '❌ 不满足'}",
                            f"- 实际值: {actual_value}",
                            f"- 期望值: {expected_value}",
                            f"- 比较操作: {operator}",
                            ""
                        ])
            
            # 一致性检查结果
            step3 = result.get('step3_consistency_check', {})
            report_lines.extend([
                "**步骤3: 一致性检查**",
                f"- 一致性比率: {step3.get('consistency_rate', 0):.1%}",
                f"- 一致性等级: {step3.get('consistency_level', 'N/A')}",
                f"- 最终验证结果: {'✅ 通过闭环验证' if result.get('closed_loop_verified', False) else '❌ 未通过闭环验证'}",
                f"- 综合质量评分: {result.get('quality_score', 0):.2f}",
                "",
                "---",
                ""
            ])
    
    return "\n".join(report_lines)


def generate_json_summary(validation_results: Dict[str, Any], 
                         summary_stats: Dict[str, Any]) -> Dict[str, Any]:
    """生成JSON格式的摘要"""
    
    summary = {
        'metadata': {
            'report_type': 'true_closed_loop_validation_summary',
            'generation_time': datetime.now().isoformat(),
            'total_indicators_tested': summary_stats['total_indicators']
        },
        'summary_statistics': summary_stats,
        'validation_results': {},
        'quality_ranking': [],
        'consistency_ranking': []
    }
    
    # 整理验证结果
    for indicator_name, result in validation_results.items():
        summary['validation_results'][indicator_name] = {
            'closed_loop_verified': result.get('closed_loop_verified', False),
            'consistency_rate': result.get('consistency_rate', 0),
            'quality_score': result.get('quality_score', 0),
            'selection_count': result.get('step1_strategy_selection', {}).get('selection_count', 0),
            'verification_details': {
                'total_stocks': result.get('step2_indicator_verification', {}).get('total_stocks', 0),
                'verified_stocks': result.get('step2_indicator_verification', {}).get('verified_stocks', 0)
            }
        }
    
    # 质量评分排名
    quality_sorted = sorted(
        validation_results.items(), 
        key=lambda x: x[1].get('quality_score', 0), 
        reverse=True
    )
    summary['quality_ranking'] = [
        {
            'indicator': name,
            'quality_score': result.get('quality_score', 0),
            'consistency_rate': result.get('consistency_rate', 0)
        }
        for name, result in quality_sorted
    ]
    
    # 一致性比率排名
    consistency_sorted = sorted(
        validation_results.items(), 
        key=lambda x: x[1].get('consistency_rate', 0), 
        reverse=True
    )
    summary['consistency_ranking'] = [
        {
            'indicator': name,
            'consistency_rate': result.get('consistency_rate', 0),
            'quality_score': result.get('quality_score', 0)
        }
        for name, result in consistency_sorted
    ]
    
    return summary


def main():
    """主函数"""
    logger.info("🚀 开始生成真正闭环验证综合报告")
    
    try:
        # 收集验证结果
        validation_results, summary_stats = collect_validation_results()
        
        if not validation_results:
            logger.warning("❌ 未找到任何验证结果文件")
            return
        
        logger.info(f"📊 收集到 {len(validation_results)} 个指标的验证结果")
        
        # 生成Markdown报告
        markdown_report = generate_markdown_report(validation_results, summary_stats)
        markdown_file = "data/result/true_closed_loop_validation_report.md"
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        logger.info(f"📝 Markdown报告已保存: {markdown_file}")
        
        # 生成JSON摘要
        json_summary = generate_json_summary(validation_results, summary_stats)
        json_file = "data/result/true_closed_loop_validation_summary.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📊 JSON摘要已保存: {json_file}")
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("🎉 真正闭环验证综合报告")
        print(f"{'='*60}")
        print(f"总验证指标数: {summary_stats['total_indicators']}")
        print(f"通过验证指标数: {summary_stats['passed_indicators']}")
        print(f"验证通过率: {summary_stats['passed_indicators'] / summary_stats['total_indicators'] * 100:.1f}%" if summary_stats['total_indicators'] > 0 else "验证通过率: 0%")
        print(f"完美一致性指标数: {summary_stats['perfect_consistency_indicators']}")
        print(f"高质量指标数: {summary_stats['high_quality_indicators']}")
        print(f"\n详细报告: {markdown_file}")
        print(f"数据摘要: {json_file}")
        print(f"{'='*60}")
        
        logger.info("✅ 综合报告生成完成")
        
    except Exception as e:
        logger.error(f"❌ 生成报告失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 