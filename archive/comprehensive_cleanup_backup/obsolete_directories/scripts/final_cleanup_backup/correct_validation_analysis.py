#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正确的验证状态分析脚本
以进度表中的状态为准，而不是验证报告文件
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


class CorrectValidationAnalyzer:
    """正确的验证状态分析器 - 以进度表为准"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.progress_file = self.root_dir / "docs/finaltesting/技术指标验证进度表.md"
        
    def get_registered_indicators(self) -> Set[str]:
        """获取所有已注册的指标"""
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            all_indicators = registry.get_all_indicators()
            return set(all_indicators.keys())
        except Exception as e:
            logger.error(f"❌ 获取注册表指标失败: {e}")
            return set()
    
    def parse_progress_table(self) -> Dict[str, Dict]:
        """解析进度表，提取所有指标的验证状态"""
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
                match = re.search(r'\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]*)\s*\|', line)
                if match:
                    indicator_name = match.group(1).strip()
                    status = match.group(2).strip()
                    score = match.group(3).strip()
                    date = match.group(4).strip()
                    note = match.group(5).strip()
                    
                    # 解析状态
                    if '✅' in status and 'PASSED' in status:
                        parsed_status = 'PASSED'
                    elif '❌' in status and 'FAILED' in status:
                        parsed_status = 'FAILED'
                    elif '⏸️' in status and 'PENDING' in status:
                        parsed_status = 'PENDING'
                    else:
                        continue  # 跳过无法识别的状态
                    
                    # 解析分数
                    score_match = re.search(r'([0-9.]+)/100', score)
                    parsed_score = float(score_match.group(1)) if score_match else 0.0
                    
                    # 只保留最新的状态（如果有重复）
                    if indicator_name not in indicator_statuses:
                        indicator_statuses[indicator_name] = {
                            'status': parsed_status,
                            'score': parsed_score,
                            'date': date,
                            'note': note,
                            'line_number': line_num
                        }
        
        return indicator_statuses
    
    def categorize_by_progress_table(self, registered: Set[str], progress_statuses: Dict[str, Dict]) -> Dict[str, List]:
        """根据进度表状态分类指标"""
        
        passed_indicators = []
        failed_indicators = []
        pending_indicators = []
        not_in_progress_table = []
        
        for indicator in registered:
            if indicator in progress_statuses:
                status_info = progress_statuses[indicator]
                
                if status_info['status'] == 'PASSED':
                    passed_indicators.append({
                        'name': indicator,
                        'score': status_info['score'],
                        'date': status_info['date'],
                        'note': status_info['note']
                    })
                elif status_info['status'] == 'FAILED':
                    failed_indicators.append({
                        'name': indicator,
                        'score': status_info['score'],
                        'date': status_info['date'],
                        'note': status_info['note']
                    })
                elif status_info['status'] == 'PENDING':
                    pending_indicators.append({
                        'name': indicator,
                        'note': status_info['note']
                    })
            else:
                not_in_progress_table.append(indicator)
        
        return {
            'passed': passed_indicators,
            'failed': failed_indicators,
            'pending': pending_indicators,
            'not_in_table': not_in_progress_table
        }
    
    def generate_correct_status_report(self, categorized: Dict, registered_count: int) -> str:
        """生成正确的验证状态报告"""
        
        passed_count = len(categorized['passed'])
        failed_count = len(categorized['failed'])
        pending_count = len(categorized['pending'])
        not_in_table_count = len(categorized['not_in_table'])
        
        validation_rate = (passed_count / registered_count) * 100
        
        report = f"""# 正确的指标验证状态报告 (基于进度表)

## 验证概览
- **已注册指标总数**: {registered_count}个
- **验证通过指标**: {passed_count}个 (✅ PASSED)
- **验证失败指标**: {failed_count}个 (❌ FAILED)
- **待验证指标**: {pending_count}个 (⏸️ PENDING)
- **未在进度表中**: {not_in_table_count}个
- **验证完成率**: {validation_rate:.1f}%

## ✅ 验证通过的指标 ({passed_count}个)

"""
        
        if categorized['passed']:
            for i, indicator in enumerate(categorized['passed'], 1):
                report += f"{i}. **{indicator['name']}** - {indicator['score']:.1f}/100 ✅ ({indicator['date']})\n"
        else:
            report += "暂无验证通过的指标\n"
        
        report += f"""
## ❌ 验证失败的指标 ({failed_count}个)

"""
        
        if categorized['failed']:
            for i, indicator in enumerate(categorized['failed'], 1):
                report += f"{i}. **{indicator['name']}** - {indicator['score']:.1f}/100 ❌ ({indicator['date']})\n"
        else:
            report += "暂无验证失败的指标\n"
        
        report += f"""
## ⏸️ 待验证的指标 ({pending_count}个)

"""
        
        if categorized['pending']:
            for i, indicator in enumerate(categorized['pending'], 1):
                report += f"{i}. **{indicator['name']}** - ⏸️ PENDING\n"
        else:
            report += "暂无待验证的指标\n"
        
        report += f"""
## 📋 未在进度表中的指标 ({not_in_table_count}个)

"""
        
        if categorized['not_in_table']:
            for i, indicator in enumerate(categorized['not_in_table'], 1):
                report += f"{i}. **{indicator}** - 未记录\n"
        else:
            report += "所有指标都在进度表中\n"
        
        # 添加验证状态总结
        if validation_rate >= 95:
            status = "🎉 验证基本完成"
        elif validation_rate >= 80:
            status = "✅ 验证进展良好"
        elif validation_rate >= 60:
            status = "👍 验证进展中等"
        else:
            status = "⚠️ 需要加强验证"
        
        report += f"""
## 验证状态总结

### {status}

- **验证完成率**: {validation_rate:.1f}%
- **需要修复的指标**: {failed_count}个
- **需要验证的指标**: {pending_count + not_in_table_count}个

### 📋 下一步行动计划

1. **修复失败指标**: {failed_count}个
2. **验证待验证指标**: {pending_count}个
3. **处理未记录指标**: {not_in_table_count}个

---
*基于技术指标验证进度表的准确数据分析*
*判断标准: 以进度表中的状态标记为准*
"""
        
        return report
    
    def run_analysis(self) -> Dict:
        """运行正确的验证状态分析"""
        logger.info("🚀 开始正确的验证状态分析 (以进度表为准)...")
        
        # 获取已注册指标
        registered_indicators = self.get_registered_indicators()
        logger.info(f"📋 已注册指标: {len(registered_indicators)}个")
        
        # 解析进度表
        progress_statuses = self.parse_progress_table()
        logger.info(f"📊 进度表中找到: {len(progress_statuses)}个指标状态")
        
        # 分类指标
        categorized = self.categorize_by_progress_table(registered_indicators, progress_statuses)
        
        # 生成报告
        report_content = self.generate_correct_status_report(categorized, len(registered_indicators))
        
        # 保存报告
        report_file = self.root_dir / "docs/finaltesting/indicators/correct_validation_status_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 正确验证状态报告已保存: {report_file}")
        
        # 输出关键信息
        logger.info("=" * 60)
        logger.info("📊 正确验证状态分析结果 (基于进度表)")
        logger.info("=" * 60)
        logger.info(f"📈 已注册指标: {len(registered_indicators)}个")
        logger.info(f"✅ 验证通过: {len(categorized['passed'])}个")
        logger.info(f"❌ 验证失败: {len(categorized['failed'])}个")
        logger.info(f"⏸️ 待验证: {len(categorized['pending'])}个")
        logger.info(f"📋 未记录: {len(categorized['not_in_table'])}个")
        logger.info(f"🎯 验证完成率: {(len(categorized['passed']) / len(registered_indicators) * 100):.1f}%")
        
        logger.info("=" * 60)
        
        return {
            'registered_count': len(registered_indicators),
            'passed_count': len(categorized['passed']),
            'failed_count': len(categorized['failed']),
            'pending_count': len(categorized['pending']),
            'not_in_table_count': len(categorized['not_in_table']),
            'validation_rate': (len(categorized['passed']) / len(registered_indicators)) * 100,
            'categorized': categorized
        }


def main():
    """主函数"""
    analyzer = CorrectValidationAnalyzer()
    result = analyzer.run_analysis()
    
    logger.info(f"\n🎯 分析完成！正确验证完成率: {result['validation_rate']:.1f}%")
    
    return result


if __name__ == "__main__":
    main()
