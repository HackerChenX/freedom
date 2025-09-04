#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精确的指标验证状态检查
对比注册表中的指标与验证报告，找出真正未验证的指标
"""

import sys
import os
import re
from pathlib import Path
from typing import Set, Dict, List

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class PreciseIndicatorChecker:
    """精确的指标检查器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.reports_dir = self.root_dir / "docs/finaltesting/indicators"
        
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
    
    def get_validated_indicators(self) -> Set[str]:
        """获取所有已验证的指标（从验证报告文件名推断）"""
        validated = set()
        
        if not self.reports_dir.exists():
            logger.warning("⚠️ 验证报告目录不存在")
            return validated
        
        # 扫描所有验证报告
        for report_file in self.reports_dir.glob("*validation_report.md"):
            # 提取指标名称
            filename = report_file.stem
            
            # 移除后缀
            if filename.endswith("_validation_report"):
                indicator_name = filename.replace("_validation_report", "")
            elif filename.endswith("_fixed_validation_report"):
                indicator_name = filename.replace("_fixed_validation_report", "")
            elif filename.endswith("_strict_validation_report"):
                indicator_name = filename.replace("_strict_validation_report", "")
            elif filename.endswith("_99_validation_report"):
                indicator_name = filename.replace("_99_validation_report", "")
            elif filename.endswith("_95_validation_report"):
                indicator_name = filename.replace("_95_validation_report", "")
            elif filename.endswith("_revalidation_report"):
                indicator_name = filename.replace("_revalidation_report", "")
            else:
                continue
            
            # 跳过批量验证报告
            if any(keyword in indicator_name.lower() for keyword in [
                'batch', 'final', 'complete', 'sample', 'current_status',
                'all_', 'factory_', 'enhanced_pattern', 'enhanced_boll_indicators',
                'quick_fixed_pattern'
            ]):
                continue
            
            # 标准化指标名称
            indicator_name = indicator_name.upper()
            validated.add(indicator_name)
        
        logger.info(f"📋 从验证报告中识别出 {len(validated)} 个已验证指标")
        return validated
    
    def analyze_validation_status(self) -> Dict[str, any]:
        """分析验证状态"""
        logger.info("🔍 开始精确验证状态分析...")
        
        # 获取数据
        registered_indicators = self.get_registered_indicators()
        validated_indicators = self.get_validated_indicators()
        
        # 分析差异
        registered_but_not_validated = registered_indicators - validated_indicators
        validated_but_not_registered = validated_indicators - registered_indicators
        both_registered_and_validated = registered_indicators & validated_indicators
        
        # 统计结果
        result = {
            'total_registered': len(registered_indicators),
            'total_validated': len(validated_indicators),
            'both_registered_and_validated': len(both_registered_and_validated),
            'registered_but_not_validated': len(registered_but_not_validated),
            'validated_but_not_registered': len(validated_but_not_registered),
            'registered_indicators': sorted(list(registered_indicators)),
            'validated_indicators': sorted(list(validated_indicators)),
            'missing_validation': sorted(list(registered_but_not_validated)),
            'extra_validation': sorted(list(validated_but_not_registered))
        }
        
        return result
    
    def generate_report(self, analysis: Dict) -> str:
        """生成详细报告"""
        report = f"""# 精确指标验证状态报告

## 验证概览
- **已注册指标总数**: {analysis['total_registered']}个
- **已验证指标总数**: {analysis['total_validated']}个
- **已注册且已验证**: {analysis['both_registered_and_validated']}个
- **已注册但未验证**: {analysis['registered_but_not_validated']}个
- **已验证但未注册**: {analysis['validated_but_not_registered']}个

## 验证完成率
- **验证完成率**: {(analysis['both_registered_and_validated'] / analysis['total_registered'] * 100):.1f}%

## 需要验证的指标清单

### 已注册但未验证的指标 ({analysis['registered_but_not_validated']}个)
"""
        
        if analysis['missing_validation']:
            for i, indicator in enumerate(analysis['missing_validation'], 1):
                report += f"{i}. **{indicator}**\n"
        else:
            report += "🎉 所有已注册指标都已验证！\n"
        
        report += f"""
### 已验证但未注册的指标 ({analysis['validated_but_not_registered']}个)
"""
        
        if analysis['extra_validation']:
            for i, indicator in enumerate(analysis['extra_validation'], 1):
                report += f"{i}. **{indicator}**\n"
        else:
            report += "✅ 没有多余的验证报告\n"
        
        report += f"""
## 所有已注册指标清单 ({analysis['total_registered']}个)

"""
        for i, indicator in enumerate(analysis['registered_indicators'], 1):
            status = "✅" if indicator in analysis['validated_indicators'] else "❌"
            report += f"{i}. {status} **{indicator}**\n"
        
        report += f"""
---
*报告生成时间: {Path(__file__).stat().st_mtime}*
"""
        
        return report
    
    def run_check(self) -> Dict:
        """运行检查"""
        logger.info("🚀 开始精确指标验证状态检查...")
        
        # 分析验证状态
        analysis = self.analyze_validation_status()
        
        # 生成报告
        report_content = self.generate_report(analysis)
        
        # 保存报告
        report_file = self.reports_dir / "precise_validation_status_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 精确验证状态报告已保存: {report_file}")
        
        # 输出关键信息
        logger.info("=" * 60)
        logger.info("📊 精确验证状态检查结果")
        logger.info("=" * 60)
        logger.info(f"📈 已注册指标: {analysis['total_registered']}个")
        logger.info(f"✅ 已验证指标: {analysis['total_validated']}个")
        logger.info(f"🎯 验证完成率: {(analysis['both_registered_and_validated'] / analysis['total_registered'] * 100):.1f}%")
        logger.info(f"❌ 需要验证: {analysis['registered_but_not_validated']}个")
        
        if analysis['missing_validation']:
            logger.info("\n🔍 需要验证的指标:")
            for indicator in analysis['missing_validation'][:10]:  # 只显示前10个
                logger.info(f"  - {indicator}")
            if len(analysis['missing_validation']) > 10:
                logger.info(f"  ... 还有 {len(analysis['missing_validation']) - 10} 个")
        else:
            logger.info("🎉 所有已注册指标都已验证！")
        
        logger.info("=" * 60)
        
        return analysis


def main():
    """主函数"""
    checker = PreciseIndicatorChecker()
    result = checker.run_check()
    
    # 判断是否完成
    if result['registered_but_not_validated'] == 0:
        logger.info("🎉 所有指标验证完成！")
        return True
    else:
        logger.warning(f"⚠️ 还有 {result['registered_but_not_validated']} 个指标需要验证")
        return False


if __name__ == "__main__":
    main()
