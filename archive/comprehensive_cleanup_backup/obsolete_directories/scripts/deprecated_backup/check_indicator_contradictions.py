#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查指标验证状态矛盾脚本
查找进度表中存在矛盾状态的指标
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class IndicatorContradictionChecker:
    """指标状态矛盾检查器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.progress_file = self.root_dir / "docs/finaltesting/技术指标验证进度表.md"
        self.reports_dir = self.root_dir / "docs/finaltesting/indicators"
        
    def extract_indicator_statuses_from_progress(self) -> Dict[str, List[Dict]]:
        """从进度表中提取所有指标状态"""
        if not self.progress_file.exists():
            logger.error("❌ 进度表文件不存在")
            return {}
        
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        indicator_statuses = {}
        
        for line_num, line in enumerate(lines, 1):
            # 查找表格行中的指标
            if '|' in line and '**' in line:
                # 匹配表格行: | **INDICATOR_NAME** | STATUS | SCORE | DATE | NOTE |
                match = re.search(r'\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|', line)
                if match:
                    indicator_name = match.group(1).strip()
                    status = match.group(2).strip()
                    score = match.group(3).strip()
                    date = match.group(4).strip()
                    note = match.group(5).strip()
                    
                    if indicator_name not in indicator_statuses:
                        indicator_statuses[indicator_name] = []
                    
                    indicator_statuses[indicator_name].append({
                        'line_number': line_num,
                        'status': status,
                        'score': score,
                        'date': date,
                        'note': note,
                        'raw_line': line.strip()
                    })
        
        return indicator_statuses
    
    def extract_actual_validation_results(self) -> Dict[str, Dict]:
        """从验证报告中提取实际验证结果"""
        actual_results = {}
        
        if not self.reports_dir.exists():
            logger.warning("⚠️ 验证报告目录不存在")
            return actual_results
        
        # 检查批量验证报告
        batch_reports = [
            'all_pattern_indicators_95_validation_report.md',
            'final_project_validation_report.md',
            'all_zxm_indicators_95_validation_report.md'
        ]
        
        for report_file in batch_reports:
            report_path = self.reports_dir / report_file
            if report_path.exists():
                results = self._parse_batch_report(report_path)
                actual_results.update(results)
        
        # 检查单独验证报告
        for report_file in self.reports_dir.glob("*validation_report.md"):
            if report_file.name not in batch_reports:
                result = self._parse_individual_report(report_file)
                if result:
                    actual_results.update(result)
        
        return actual_results
    
    def _parse_batch_report(self, report_path: Path) -> Dict[str, Dict]:
        """解析批量验证报告"""
        results = {}
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line in lines:
                # 匹配格式: - **INDICATOR_NAME**: 85.0/100 ❌
                match = re.search(r'-\s*\*\*([A-Z_]+)\*\*:\s*([0-9.]+)/100\s*([✅❌])', line)
                if match:
                    indicator_name = match.group(1).strip()
                    score = float(match.group(2))
                    status_symbol = match.group(3).strip()
                    status = "PASSED" if status_symbol == "✅" else "FAILED"
                    
                    results[indicator_name] = {
                        'score': score,
                        'status': status,
                        'source': report_path.name
                    }
        
        except Exception as e:
            logger.error(f"❌ 解析批量报告失败 {report_path}: {e}")
        
        return results
    
    def _parse_individual_report(self, report_path: Path) -> Dict[str, Dict]:
        """解析单独验证报告"""
        try:
            # 从文件名推断指标名称
            filename = report_path.stem
            indicator_name = filename.replace('_validation_report', '').replace('_fixed_validation_report', '').upper()
            
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找总得分
            score_match = re.search(r'总得分.*?([0-9.]+)/100', content)
            status_match = re.search(r'验证状态.*?(PASSED|FAILED)', content)
            
            if score_match:
                score = float(score_match.group(1))
                status = status_match.group(1) if status_match else ("PASSED" if score >= 95 else "FAILED")
                
                return {
                    indicator_name: {
                        'score': score,
                        'status': status,
                        'source': report_path.name
                    }
                }
        
        except Exception as e:
            logger.error(f"❌ 解析单独报告失败 {report_path}: {e}")
        
        return {}
    
    def find_contradictions(self) -> Dict[str, Dict]:
        """查找状态矛盾"""
        logger.info("🔍 开始检查指标状态矛盾...")
        
        # 获取数据
        progress_statuses = self.extract_indicator_statuses_from_progress()
        actual_results = self.extract_actual_validation_results()
        
        contradictions = {}
        
        for indicator_name, progress_entries in progress_statuses.items():
            # 检查进度表内部矛盾
            if len(progress_entries) > 1:
                statuses = set(entry['status'] for entry in progress_entries)
                if len(statuses) > 1:
                    contradictions[indicator_name] = {
                        'type': 'internal_contradiction',
                        'progress_entries': progress_entries,
                        'actual_result': actual_results.get(indicator_name)
                    }
            
            # 检查进度表与实际验证结果的矛盾
            if indicator_name in actual_results:
                actual_result = actual_results[indicator_name]
                
                for entry in progress_entries:
                    progress_status = entry['status']
                    progress_score = entry['score']
                    
                    # 检查状态矛盾
                    if ('PASSED' in progress_status and actual_result['status'] == 'FAILED') or \
                       ('PENDING' in progress_status and actual_result['status'] == 'PASSED'):
                        
                        if indicator_name not in contradictions:
                            contradictions[indicator_name] = {
                                'type': 'progress_vs_actual',
                                'progress_entries': progress_entries,
                                'actual_result': actual_result
                            }
                        else:
                            contradictions[indicator_name]['type'] = 'multiple_contradictions'
        
        return contradictions
    
    def generate_contradiction_report(self, contradictions: Dict) -> str:
        """生成矛盾报告"""
        report = f"""# 指标验证状态矛盾检查报告

## 检查概览
- **检查时间**: {Path(__file__).stat().st_mtime}
- **发现矛盾指标数**: {len(contradictions)}个

## 矛盾详情

"""
        
        if not contradictions:
            report += "🎉 **未发现状态矛盾！**\n\n所有指标的验证状态都是一致的。\n"
        else:
            for i, (indicator_name, contradiction) in enumerate(contradictions.items(), 1):
                report += f"### {i}. **{indicator_name}** - {contradiction['type']}\n\n"
                
                # 进度表中的状态
                report += "**进度表中的状态**:\n"
                for entry in contradiction['progress_entries']:
                    report += f"- 第{entry['line_number']}行: {entry['status']} | {entry['score']} | {entry['date']}\n"
                
                # 实际验证结果
                if contradiction['actual_result']:
                    actual = contradiction['actual_result']
                    report += f"\n**实际验证结果**:\n"
                    report += f"- 状态: {actual['status']}\n"
                    report += f"- 得分: {actual['score']}/100\n"
                    report += f"- 来源: {actual['source']}\n"
                
                report += "\n---\n\n"
        
        report += f"""
## 修复建议

### 🔧 立即修复
1. **统一状态**: 将进度表中的矛盾状态统一为实际验证结果
2. **删除重复**: 移除进度表中的重复条目
3. **更新评分**: 确保评分与实际验证报告一致

### 📋 验证原则
- **以实际验证报告为准**: 验证报告是权威数据源
- **95分以上为PASSED**: 统一验证标准
- **单一状态**: 每个指标在进度表中只能有一个状态

---
*检查工具: 指标状态矛盾检查器*
*数据源: 技术指标验证进度表 + 验证报告*
"""
        
        return report
    
    def run_check(self) -> Dict:
        """运行矛盾检查"""
        logger.info("🚀 开始指标状态矛盾检查...")
        
        # 查找矛盾
        contradictions = self.find_contradictions()
        
        # 生成报告
        report_content = self.generate_contradiction_report(contradictions)
        
        # 保存报告
        report_file = self.reports_dir / "indicator_contradictions_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 矛盾检查报告已保存: {report_file}")
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("🔍 指标状态矛盾检查结果")
        logger.info("=" * 60)
        logger.info(f"🔍 发现矛盾指标: {len(contradictions)}个")
        
        if contradictions:
            logger.warning("⚠️ 发现以下指标存在状态矛盾:")
            for indicator_name, contradiction in list(contradictions.items())[:10]:
                logger.warning(f"  - {indicator_name}: {contradiction['type']}")
            if len(contradictions) > 10:
                logger.warning(f"  ... 还有 {len(contradictions) - 10} 个")
        else:
            logger.info("🎉 未发现状态矛盾！")
        
        logger.info("=" * 60)
        
        return {
            'contradictions_count': len(contradictions),
            'contradictions': contradictions
        }


def main():
    """主函数"""
    checker = IndicatorContradictionChecker()
    result = checker.run_check()
    
    if result['contradictions_count'] == 0:
        logger.info("🎉 所有指标状态一致！")
        return True
    else:
        logger.warning(f"⚠️ 发现 {result['contradictions_count']} 个指标存在状态矛盾")
        return False


if __name__ == "__main__":
    main()
