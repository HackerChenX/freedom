#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析真实验证状态脚本
基于实际验证报告，重新整理需要验证的指标
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


class RealValidationAnalyzer:
    """真实验证状态分析器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.reports_dir = self.root_dir / "docs/finaltesting/indicators"
        
        # 验证通过标准
        self.pass_threshold = 95.0
        
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
    
    def analyze_validation_reports(self) -> Dict[str, Dict]:
        """分析所有验证报告，获取真实验证状态"""
        validation_results = {}
        
        if not self.reports_dir.exists():
            logger.error("❌ 验证报告目录不存在")
            return validation_results
        
        # 1. 分析批量验证报告
        batch_reports = [
            'all_pattern_indicators_95_validation_report.md',
            'final_project_validation_report.md', 
            'all_zxm_indicators_95_validation_report.md',
            'enhanced_boll_indicators_validation_report.md'
        ]
        
        for report_name in batch_reports:
            report_path = self.reports_dir / report_name
            if report_path.exists():
                logger.info(f"📄 分析批量报告: {report_name}")
                results = self._parse_batch_report(report_path)
                validation_results.update(results)
        
        # 2. 分析单独验证报告
        individual_reports = []
        for report_file in self.reports_dir.glob("*validation_report.md"):
            if report_file.name not in batch_reports:
                individual_reports.append(report_file)
        
        logger.info(f"📄 发现 {len(individual_reports)} 个单独验证报告")
        
        for report_path in individual_reports:
            logger.info(f"📄 分析单独报告: {report_path.name}")
            result = self._parse_individual_report(report_path)
            if result:
                validation_results.update(result)
        
        return validation_results
    
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
                    status = "PASSED" if score >= self.pass_threshold else "FAILED"
                    
                    results[indicator_name] = {
                        'score': score,
                        'status': status,
                        'source': report_path.name,
                        'type': 'batch_report'
                    }
                
                # 匹配其他格式
                match2 = re.search(r'\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*([0-9.]+)/100\s*\|\s*([✅❌])', line)
                if match2:
                    indicator_name = match2.group(1).strip()
                    score = float(match2.group(2))
                    status = "PASSED" if score >= self.pass_threshold else "FAILED"
                    
                    results[indicator_name] = {
                        'score': score,
                        'status': status,
                        'source': report_path.name,
                        'type': 'batch_report'
                    }
        
        except Exception as e:
            logger.error(f"❌ 解析批量报告失败 {report_path}: {e}")
        
        return results
    
    def _parse_individual_report(self, report_path: Path) -> Dict[str, Dict]:
        """解析单独验证报告"""
        try:
            # 从文件名推断指标名称
            filename = report_path.stem
            
            # 清理文件名，提取指标名称
            indicator_name = filename
            for suffix in ['_validation_report', '_fixed_validation_report', '_strict_validation_report', 
                          '_99_validation_report', '_95_validation_report', '_revalidation_report']:
                indicator_name = indicator_name.replace(suffix, '')
            
            indicator_name = indicator_name.upper()
            
            # 跳过非指标报告
            skip_keywords = ['batch', 'final', 'complete', 'sample', 'current_status', 
                           'all_', 'factory_', 'enhanced_pattern', 'quick_fixed_pattern',
                           'comprehensive', 'precise', 'smart', 'contradiction']
            
            if any(keyword in indicator_name.lower() for keyword in skip_keywords):
                return {}
            
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找总得分
            score_patterns = [
                r'总得分.*?([0-9.]+)/100',
                r'总分.*?([0-9.]+)/100',
                r'得分.*?([0-9.]+)/100',
                r'Score.*?([0-9.]+)/100'
            ]
            
            score = None
            for pattern in score_patterns:
                score_match = re.search(pattern, content, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                    break
            
            # 查找验证状态
            status_patterns = [
                r'验证状态.*?(PASSED|FAILED)',
                r'状态.*?(PASSED|FAILED)',
                r'Status.*?(PASSED|FAILED)'
            ]
            
            status = None
            for pattern in status_patterns:
                status_match = re.search(pattern, content, re.IGNORECASE)
                if status_match:
                    status = status_match.group(1).upper()
                    break
            
            # 如果没有明确状态，根据分数判断
            if score is not None and status is None:
                status = "PASSED" if score >= self.pass_threshold else "FAILED"
            
            if score is not None:
                return {
                    indicator_name: {
                        'score': score,
                        'status': status or "UNKNOWN",
                        'source': report_path.name,
                        'type': 'individual_report'
                    }
                }
        
        except Exception as e:
            logger.error(f"❌ 解析单独报告失败 {report_path}: {e}")
        
        return {}
    
    def categorize_indicators(self, registered: Set[str], validated: Dict[str, Dict]) -> Dict[str, List]:
        """分类指标状态"""
        
        # 已验证且通过的指标
        passed_indicators = []
        # 已验证但失败的指标  
        failed_indicators = []
        # 未验证的指标
        not_validated_indicators = []
        
        for indicator in registered:
            if indicator in validated:
                result = validated[indicator]
                if result['status'] == 'PASSED':
                    passed_indicators.append({
                        'name': indicator,
                        'score': result['score'],
                        'source': result['source']
                    })
                else:
                    failed_indicators.append({
                        'name': indicator,
                        'score': result['score'],
                        'source': result['source']
                    })
            else:
                not_validated_indicators.append(indicator)
        
        return {
            'passed': passed_indicators,
            'failed': failed_indicators,
            'not_validated': not_validated_indicators
        }
    
    def generate_real_status_report(self, categorized: Dict, registered_count: int) -> str:
        """生成真实验证状态报告"""
        
        passed_count = len(categorized['passed'])
        failed_count = len(categorized['failed'])
        not_validated_count = len(categorized['not_validated'])
        
        validation_rate = (passed_count / registered_count) * 100
        
        report = f"""# 真实指标验证状态报告

## 验证概览
- **已注册指标总数**: {registered_count}个
- **验证通过指标**: {passed_count}个 (≥95分)
- **验证失败指标**: {failed_count}个 (<95分)
- **未验证指标**: {not_validated_count}个
- **真实验证完成率**: {validation_rate:.1f}%

## 验证通过的指标 ({passed_count}个)

"""
        
        if categorized['passed']:
            for i, indicator in enumerate(categorized['passed'], 1):
                report += f"{i}. **{indicator['name']}** - {indicator['score']:.1f}/100 ✅ (来源: {indicator['source']})\n"
        else:
            report += "暂无验证通过的指标\n"
        
        report += f"""
## 验证失败的指标 ({failed_count}个)

"""
        
        if categorized['failed']:
            for i, indicator in enumerate(categorized['failed'], 1):
                report += f"{i}. **{indicator['name']}** - {indicator['score']:.1f}/100 ❌ (来源: {indicator['source']})\n"
        else:
            report += "暂无验证失败的指标\n"
        
        report += f"""
## 🔄 需要验证的指标 ({not_validated_count}个)

"""
        
        if categorized['not_validated']:
            for i, indicator in enumerate(categorized['not_validated'], 1):
                report += f"{i}. **{indicator}** - ⏸️ PENDING\n"
        else:
            report += "🎉 所有指标都已验证！\n"
        
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

- **真实验证完成率**: {validation_rate:.1f}%
- **需要验证的指标**: {not_validated_count}个
- **需要修复的指标**: {failed_count}个

### 📋 下一步行动计划

1. **优先验证**: {not_validated_count}个未验证指标
2. **修复失败**: {failed_count}个验证失败指标
3. **质量目标**: 达到95%验证通过率

---
*基于实际验证报告的真实数据分析*
*验证标准: ≥95分为通过*
"""
        
        return report
    
    def run_analysis(self) -> Dict:
        """运行真实验证状态分析"""
        logger.info("🚀 开始真实验证状态分析...")
        
        # 获取已注册指标
        registered_indicators = self.get_registered_indicators()
        logger.info(f"📋 已注册指标: {len(registered_indicators)}个")
        
        # 分析验证报告
        validation_results = self.analyze_validation_reports()
        logger.info(f"📊 找到验证结果: {len(validation_results)}个")
        
        # 分类指标
        categorized = self.categorize_indicators(registered_indicators, validation_results)
        
        # 生成报告
        report_content = self.generate_real_status_report(categorized, len(registered_indicators))
        
        # 保存报告
        report_file = self.reports_dir / "real_validation_status_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 真实验证状态报告已保存: {report_file}")
        
        # 输出关键信息
        logger.info("=" * 60)
        logger.info("📊 真实验证状态分析结果")
        logger.info("=" * 60)
        logger.info(f"📈 已注册指标: {len(registered_indicators)}个")
        logger.info(f"✅ 验证通过: {len(categorized['passed'])}个")
        logger.info(f"❌ 验证失败: {len(categorized['failed'])}个")
        logger.info(f"⏸️ 未验证: {len(categorized['not_validated'])}个")
        logger.info(f"🎯 真实验证完成率: {(len(categorized['passed']) / len(registered_indicators) * 100):.1f}%")
        
        if categorized['not_validated']:
            logger.info(f"\n🔍 需要验证的指标 ({len(categorized['not_validated'])}个):")
            for indicator in categorized['not_validated'][:10]:
                logger.info(f"  - {indicator}")
            if len(categorized['not_validated']) > 10:
                logger.info(f"  ... 还有 {len(categorized['not_validated']) - 10} 个")
        
        if categorized['failed']:
            logger.info(f"\n⚠️ 需要修复的指标 ({len(categorized['failed'])}个):")
            for indicator in categorized['failed'][:5]:
                logger.info(f"  - {indicator['name']}: {indicator['score']:.1f}/100")
            if len(categorized['failed']) > 5:
                logger.info(f"  ... 还有 {len(categorized['failed']) - 5} 个")
        
        logger.info("=" * 60)
        
        return {
            'registered_count': len(registered_indicators),
            'passed_count': len(categorized['passed']),
            'failed_count': len(categorized['failed']),
            'not_validated_count': len(categorized['not_validated']),
            'validation_rate': (len(categorized['passed']) / len(registered_indicators)) * 100,
            'categorized': categorized
        }


def main():
    """主函数"""
    analyzer = RealValidationAnalyzer()
    result = analyzer.run_analysis()
    
    logger.info(f"\n🎯 分析完成！真实验证完成率: {result['validation_rate']:.1f}%")
    
    return result


if __name__ == "__main__":
    main()
