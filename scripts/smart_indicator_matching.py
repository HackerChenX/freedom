#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能指标名称匹配脚本
解决注册表和验证报告之间的命名不一致问题
"""

import sys
import os
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple
from difflib import SequenceMatcher

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class SmartIndicatorMatcher:
    """智能指标名称匹配器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.reports_dir = self.root_dir / "docs/finaltesting/indicators"
        
        # 名称映射规则
        self.name_mappings = {
            # 增强版指标映射
            'EnhancedMACD': ['ENHANCED_MACD', 'ENHANCED_MACD_TREND'],
            'EnhancedRSI': ['ENHANCED_RSI', 'ENHANCEDRSI'],
            'EnhancedBOLL': ['ENHANCED_BOLL', 'ENHANCED_BOLL_INDICATORS'],
            'EnhancedCCI': ['ENHANCED_CCI'],
            'EnhancedKDJ': ['ENHANCED_KDJ'],
            'EnhancedSTOCHRSI': ['ENHANCED_STOCHRSI'],
            'EnhancedTRIX': ['ENHANCED_TRIX'],
            'EnhancedWR': ['ENHANCED_WR'],
            
            # 修复版指标映射
            'ATR': ['ATR_FIXED'],
            'COMPOSITE': ['COMPOSITE_FIXED'],
            
            # 其他特殊映射
            'UNIFIED_MA': ['UNIFIED_MA_STRICT'],
            'ADX': ['ADX_STRICT_99'],
            'VIX': ['VIX_STRICT'],
            'MTM': ['MTM_STRICT'],
            'SYNERGY': ['SYNERGY_STRICT'],
        }
    
    def normalize_name(self, name: str) -> str:
        """标准化指标名称"""
        # 移除常见后缀
        name = re.sub(r'_(validation_report|fixed_validation_report|strict_validation_report|99_validation_report|95_validation_report|revalidation_report)$', '', name)
        
        # 转换为大写
        name = name.upper()
        
        # 处理特殊情况
        if name.startswith('ENHANCED_'):
            # 转换为驼峰命名
            base_name = name.replace('ENHANCED_', '')
            name = f'Enhanced{base_name.title()}'
        
        return name
    
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
    
    def get_validation_reports(self) -> Dict[str, str]:
        """获取所有验证报告及其原始文件名"""
        reports = {}
        
        if not self.reports_dir.exists():
            logger.warning("⚠️ 验证报告目录不存在")
            return reports
        
        # 扫描所有验证报告
        for report_file in self.reports_dir.glob("*validation_report.md"):
            filename = report_file.stem
            normalized_name = self.normalize_name(filename)
            reports[normalized_name] = str(report_file)
        
        return reports
    
    def similarity(self, a: str, b: str) -> float:
        """计算两个字符串的相似度"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def smart_match(self, registered: Set[str], reports: Dict[str, str]) -> Dict[str, any]:
        """智能匹配注册表和验证报告"""
        logger.info("🧠 开始智能匹配...")
        
        matched = {}
        unmatched_registered = set(registered)
        unmatched_reports = set(reports.keys())
        
        # 1. 精确匹配
        for reg_name in list(unmatched_registered):
            if reg_name in unmatched_reports:
                matched[reg_name] = reports[reg_name]
                unmatched_registered.remove(reg_name)
                unmatched_reports.remove(reg_name)
        
        # 2. 映射规则匹配
        for reg_name in list(unmatched_registered):
            if reg_name in self.name_mappings:
                for mapped_name in self.name_mappings[reg_name]:
                    if mapped_name in unmatched_reports:
                        matched[reg_name] = reports[mapped_name]
                        unmatched_registered.remove(reg_name)
                        unmatched_reports.remove(mapped_name)
                        break
        
        # 3. 相似度匹配（阈值0.8）
        for reg_name in list(unmatched_registered):
            best_match = None
            best_score = 0.8  # 最低相似度阈值
            
            for report_name in unmatched_reports:
                score = self.similarity(reg_name, report_name)
                if score > best_score:
                    best_match = report_name
                    best_score = score
            
            if best_match:
                matched[reg_name] = reports[best_match]
                unmatched_registered.remove(reg_name)
                unmatched_reports.remove(best_match)
        
        # 4. 部分匹配（包含关系）
        for reg_name in list(unmatched_registered):
            for report_name in list(unmatched_reports):
                # 检查是否有包含关系
                if (reg_name.lower() in report_name.lower() or 
                    report_name.lower() in reg_name.lower()):
                    matched[reg_name] = reports[report_name]
                    unmatched_registered.remove(reg_name)
                    unmatched_reports.remove(report_name)
                    break
        
        return {
            'matched': matched,
            'unmatched_registered': unmatched_registered,
            'unmatched_reports': unmatched_reports,
            'total_registered': len(registered),
            'total_reports': len(reports),
            'matched_count': len(matched)
        }
    
    def analyze_validation_status(self) -> Dict[str, any]:
        """分析验证状态"""
        logger.info("🔍 开始智能验证状态分析...")
        
        # 获取数据
        registered_indicators = self.get_registered_indicators()
        validation_reports = self.get_validation_reports()
        
        # 智能匹配
        match_result = self.smart_match(registered_indicators, validation_reports)
        
        # 计算验证完成率
        verification_rate = (match_result['matched_count'] / match_result['total_registered']) * 100
        
        result = {
            'total_registered': match_result['total_registered'],
            'total_reports': match_result['total_reports'],
            'matched_count': match_result['matched_count'],
            'verification_rate': verification_rate,
            'matched_indicators': match_result['matched'],
            'unmatched_registered': match_result['unmatched_registered'],
            'unmatched_reports': match_result['unmatched_reports']
        }
        
        return result
    
    def generate_report(self, analysis: Dict) -> str:
        """生成智能匹配报告"""
        report = f"""# 智能指标验证状态报告

## 验证概览
- **已注册指标总数**: {analysis['total_registered']}个
- **验证报告总数**: {analysis['total_reports']}个
- **成功匹配**: {analysis['matched_count']}个
- **验证完成率**: {analysis['verification_rate']:.1f}%

## 匹配结果

### ✅ 已验证指标 ({analysis['matched_count']}个)
"""
        
        for i, (indicator, report_path) in enumerate(sorted(analysis['matched_indicators'].items()), 1):
            report_name = Path(report_path).stem
            report += f"{i}. **{indicator}** → `{report_name}`\n"
        
        report += f"""
### ❌ 未验证指标 ({len(analysis['unmatched_registered'])}个)
"""
        
        if analysis['unmatched_registered']:
            for i, indicator in enumerate(sorted(analysis['unmatched_registered']), 1):
                report += f"{i}. **{indicator}**\n"
        else:
            report += "🎉 所有已注册指标都已验证！\n"
        
        report += f"""
### 📄 未匹配的验证报告 ({len(analysis['unmatched_reports'])}个)
"""
        
        if analysis['unmatched_reports']:
            for i, report_name in enumerate(sorted(analysis['unmatched_reports']), 1):
                report += f"{i}. **{report_name}**\n"
        else:
            report += "✅ 所有验证报告都已匹配\n"
        
        # 添加验证状态总结
        if analysis['verification_rate'] >= 95:
            status = "🎉 验证基本完成"
        elif analysis['verification_rate'] >= 80:
            status = "✅ 验证进展良好"
        elif analysis['verification_rate'] >= 60:
            status = "👍 验证进展中等"
        else:
            status = "⚠️ 需要加强验证"
        
        report += f"""
## 验证状态总结

### {status}

- **验证完成率**: {analysis['verification_rate']:.1f}%
- **剩余工作**: {len(analysis['unmatched_registered'])}个指标需要验证

{'### 🚀 建议立即部署' if analysis['verification_rate'] >= 95 else '### 📋 建议继续验证剩余指标'}

---
*智能匹配报告生成时间: {Path(__file__).stat().st_mtime}*
"""
        
        return report
    
    def run_analysis(self) -> Dict:
        """运行智能分析"""
        logger.info("🚀 开始智能指标验证状态分析...")
        
        # 分析验证状态
        analysis = self.analyze_validation_status()
        
        # 生成报告
        report_content = self.generate_report(analysis)
        
        # 保存报告
        report_file = self.reports_dir / "smart_validation_status_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 智能验证状态报告已保存: {report_file}")
        
        # 输出关键信息
        logger.info("=" * 60)
        logger.info("🧠 智能验证状态分析结果")
        logger.info("=" * 60)
        logger.info(f"📈 已注册指标: {analysis['total_registered']}个")
        logger.info(f"📄 验证报告: {analysis['total_reports']}个")
        logger.info(f"✅ 成功匹配: {analysis['matched_count']}个")
        logger.info(f"🎯 验证完成率: {analysis['verification_rate']:.1f}%")
        logger.info(f"❌ 需要验证: {len(analysis['unmatched_registered'])}个")
        
        if analysis['unmatched_registered']:
            logger.info("\n🔍 需要验证的指标:")
            for indicator in sorted(list(analysis['unmatched_registered']))[:10]:
                logger.info(f"  - {indicator}")
            if len(analysis['unmatched_registered']) > 10:
                logger.info(f"  ... 还有 {len(analysis['unmatched_registered']) - 10} 个")
        else:
            logger.info("🎉 所有已注册指标都已验证！")
        
        logger.info("=" * 60)
        
        return analysis


def main():
    """主函数"""
    matcher = SmartIndicatorMatcher()
    result = matcher.run_analysis()
    
    # 判断验证状态
    if result['verification_rate'] >= 95:
        logger.info("🎉 指标验证基本完成！")
        return True
    elif result['verification_rate'] >= 80:
        logger.info("✅ 指标验证进展良好！")
        return True
    else:
        logger.warning(f"⚠️ 还需要验证更多指标，当前完成率: {result['verification_rate']:.1f}%")
        return False


if __name__ == "__main__":
    main()
