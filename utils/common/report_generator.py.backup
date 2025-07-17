#!/usr/bin/env python3
"""
通用报告生成器
统一项目中的各种报告生成逻辑
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

class ReportGenerator:
    """通用报告生成器"""
    
    @staticmethod
    def generate_analysis_report(data: Dict[str, Any], title: str = "分析报告") -> str:
        """生成分析报告"""
        report_lines = [
            f"# {title}",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 分析结果",
            f""
        ]
        
        for key, value in data.items():
            if isinstance(value, dict):
                report_lines.append(f"### {key}")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"- **{sub_key}**: {sub_value}")
                report_lines.append("")
            else:
                report_lines.append(f"**{key}**: {value}")
        
        return "\n".join(report_lines)
    
    @staticmethod
    def generate_performance_summary(metrics: Dict[str, float]) -> str:
        """生成性能汇总报告"""
        summary = [
            "## 性能汇总",
            "",
            "| 指标 | 数值 | 单位 |",
            "|------|------|------|"
        ]
        
        for metric, value in metrics.items():
            summary.append(f"| {metric} | {value:.2f} | - |")
        
        return "\n".join(summary)
    
    @staticmethod
    def save_json_report(data: Dict[str, Any], file_path: str) -> None:
        """保存JSON格式报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
