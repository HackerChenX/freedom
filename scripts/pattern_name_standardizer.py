#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形态名称标准化工具

用于检测和修复代码中的形态名称不一致问题，确保所有组件使用统一的形态名称。
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import logging

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry

logger = logging.getLogger(__name__)

class PatternNameStandardizer:
    """形态名称标准化工具"""
    
    def __init__(self, project_root: str = '/Users/hacker/PycharmProjects/freedom'):
        """初始化标准化工具"""
        self.project_root = Path(project_root)
        self.registry = get_unified_pattern_registry()
        
        # 需要检查的文件类型
        self.file_patterns = ['*.py', '*.yaml', '*.yml', '*.json']
        
        # 需要排除的目录
        self.exclude_dirs = {
            '__pycache__', '.git', 'venv', '.venv', 'node_modules',
            '.pytest_cache', '.mypy_cache', 'build', 'dist'
        }
        
        # 形态名称检测模式
        self.pattern_detection_regex = [
            r"['\"]([A-Z_]*(?:GOLDEN|DEATH|BULL|BEAR|DIVERGENCE|BREAKOUT|CROSS|OVERBOUGHT|OVERSOLD)[A-Z_]*)['\"]",
            r"pattern_type\s*=\s*['\"]([^'\"]+)['\"]",
            r"pattern_name\s*=\s*['\"]([^'\"]+)['\"]",
            r"pattern_id\s*=\s*['\"]([^'\"]+)['\"]",
        ]
    
    def scan_project(self) -> Dict[str, List[Tuple[str, int, str]]]:
        """扫描项目中的形态名称使用情况"""
        logger.info("🔍 开始扫描项目中的形态名称使用情况")
        
        results = {}
        total_files = 0
        
        for pattern in self.file_patterns:
            for file_path in self.project_root.rglob(pattern):
                # 跳过排除的目录
                if any(exclude_dir in file_path.parts for exclude_dir in self.exclude_dirs):
                    continue
                
                total_files += 1
                file_results = self._scan_file(file_path)
                if file_results:
                    results[str(file_path)] = file_results
        
        logger.info(f"✅ 扫描完成，检查了 {total_files} 个文件，发现 {len(results)} 个文件包含形态名称")
        return results
    
    def _scan_file(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """扫描单个文件中的形态名称"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            results = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for regex_pattern in self.pattern_detection_regex:
                    matches = re.finditer(regex_pattern, line, re.IGNORECASE)
                    for match in matches:
                        pattern_name = match.group(1)
                        # 过滤掉明显不是形态名称的内容
                        if self._is_likely_pattern_name(pattern_name):
                            results.append((pattern_name, line_num, line.strip()))
            
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 扫描文件失败 {file_path}: {e}")
            return []
    
    def _is_likely_pattern_name(self, name: str) -> bool:
        """判断是否可能是形态名称"""
        if len(name) < 3:
            return False
        
        # 包含形态相关关键词
        pattern_keywords = [
            'GOLDEN', 'DEATH', 'CROSS', 'BULL', 'BEAR', 'DIVERGENCE',
            'BREAKOUT', 'OVERBOUGHT', 'OVERSOLD', 'REVERSAL', 'CONTINUATION',
            'UPPER', 'LOWER', 'SQUEEZE', 'HISTOGRAM'
        ]
        
        return any(keyword in name.upper() for keyword in pattern_keywords)
    
    def analyze_inconsistencies(self, scan_results: Dict[str, List[Tuple[str, int, str]]]) -> Dict[str, Any]:
        """分析形态名称不一致问题"""
        logger.info("📊 分析形态名称不一致问题")
        
        # 收集所有发现的形态名称
        all_patterns = set()
        pattern_locations = {}
        
        for file_path, file_results in scan_results.items():
            for pattern_name, line_num, line_content in file_results:
                all_patterns.add(pattern_name)
                if pattern_name not in pattern_locations:
                    pattern_locations[pattern_name] = []
                pattern_locations[pattern_name].append((file_path, line_num, line_content))
        
        # 分析每个形态名称
        analysis = {
            'total_patterns_found': len(all_patterns),
            'canonical_patterns': set(),
            'non_canonical_patterns': set(),
            'unknown_patterns': set(),
            'mapping_suggestions': {},
            'statistics': {}
        }
        
        for pattern_name in all_patterns:
            canonical_name = self.registry.get_canonical_pattern_name(pattern_name)
            
            if canonical_name == pattern_name and self.registry.get_pattern_info(pattern_name):
                # 这是规范名称
                analysis['canonical_patterns'].add(pattern_name)
            elif canonical_name != pattern_name and self.registry.get_pattern_info(canonical_name):
                # 这是非规范名称，但有对应的规范名称
                analysis['non_canonical_patterns'].add(pattern_name)
                analysis['mapping_suggestions'][pattern_name] = canonical_name
            else:
                # 未知形态名称
                analysis['unknown_patterns'].add(pattern_name)
        
        # 统计信息
        analysis['statistics'] = {
            'canonical_count': len(analysis['canonical_patterns']),
            'non_canonical_count': len(analysis['non_canonical_patterns']),
            'unknown_count': len(analysis['unknown_patterns']),
            'consistency_rate': len(analysis['canonical_patterns']) / len(all_patterns) * 100 if all_patterns else 0
        }
        
        analysis['pattern_locations'] = pattern_locations
        
        return analysis
    
    def generate_standardization_report(self, analysis: Dict[str, Any]) -> str:
        """生成标准化报告"""
        report = []
        report.append("# 形态名称标准化分析报告")
        report.append("")
        report.append(f"**分析时间**: {self._get_current_time()}")
        report.append(f"**项目路径**: {self.project_root}")
        report.append("")
        
        # 统计概览
        stats = analysis['statistics']
        report.append("## 📊 统计概览")
        report.append("")
        report.append(f"- **总形态数量**: {analysis['total_patterns_found']}")
        report.append(f"- **规范形态**: {stats['canonical_count']} ({stats['canonical_count']/analysis['total_patterns_found']*100:.1f}%)")
        report.append(f"- **非规范形态**: {stats['non_canonical_count']} ({stats['non_canonical_count']/analysis['total_patterns_found']*100:.1f}%)")
        report.append(f"- **未知形态**: {stats['unknown_count']} ({stats['unknown_count']/analysis['total_patterns_found']*100:.1f}%)")
        report.append(f"- **一致性率**: {stats['consistency_rate']:.1f}%")
        report.append("")
        
        # 需要修复的形态
        if analysis['non_canonical_patterns']:
            report.append("## 🔧 需要标准化的形态")
            report.append("")
            for pattern in sorted(analysis['non_canonical_patterns']):
                canonical = analysis['mapping_suggestions'][pattern]
                locations = analysis['pattern_locations'][pattern]
                report.append(f"### {pattern} → {canonical}")
                report.append(f"**出现次数**: {len(locations)}")
                report.append("**位置**:")
                for file_path, line_num, line_content in locations[:5]:  # 只显示前5个位置
                    relative_path = str(Path(file_path).relative_to(self.project_root))
                    report.append(f"- `{relative_path}:{line_num}` - {line_content[:80]}...")
                if len(locations) > 5:
                    report.append(f"- ... 还有 {len(locations) - 5} 个位置")
                report.append("")
        
        # 未知形态
        if analysis['unknown_patterns']:
            report.append("## ❓ 未知形态")
            report.append("")
            report.append("以下形态在统一注册表中未找到对应项，需要人工确认：")
            report.append("")
            for pattern in sorted(analysis['unknown_patterns']):
                locations = analysis['pattern_locations'][pattern]
                report.append(f"- **{pattern}** (出现 {len(locations)} 次)")
            report.append("")
        
        # 建议
        report.append("## 💡 标准化建议")
        report.append("")
        report.append("1. **立即修复**: 将所有非规范形态名称替换为对应的规范名称")
        report.append("2. **统一接口**: 修改数据生成器、测试框架等组件，使用统一形态注册表")
        report.append("3. **代码审查**: 建立代码审查规则，确保新代码使用规范形态名称")
        report.append("4. **自动化检查**: 集成形态名称检查到CI/CD流程")
        report.append("")
        
        return "\n".join(report)
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_fix_script(self, analysis: Dict[str, Any]) -> str:
        """生成自动修复脚本"""
        script_lines = []
        script_lines.append("#!/usr/bin/env python3")
        script_lines.append("# -*- coding: utf-8 -*-")
        script_lines.append('"""')
        script_lines.append("自动生成的形态名称标准化修复脚本")
        script_lines.append('"""')
        script_lines.append("")
        script_lines.append("import re")
        script_lines.append("from pathlib import Path")
        script_lines.append("")
        script_lines.append("def fix_pattern_names():")
        script_lines.append('    """修复形态名称"""')
        script_lines.append("    fixes = {")
        
        # 添加修复映射
        for pattern, canonical in analysis['mapping_suggestions'].items():
            script_lines.append(f'        "{pattern}": "{canonical}",')
        
        script_lines.append("    }")
        script_lines.append("")
        script_lines.append("    for old_name, new_name in fixes.items():")
        script_lines.append("        print(f'修复: {old_name} → {new_name}')")
        script_lines.append("        # 在这里添加具体的文件修复逻辑")
        script_lines.append("")
        script_lines.append("if __name__ == '__main__':")
        script_lines.append("    fix_pattern_names()")
        
        return "\n".join(script_lines)

def run_pattern_standardization_analysis():
    """运行形态名称标准化分析"""
    print("🚀 开始形态名称标准化分析")
    print("=" * 80)
    
    # 初始化工具
    standardizer = PatternNameStandardizer()
    
    # 扫描项目
    scan_results = standardizer.scan_project()
    
    # 分析不一致问题
    analysis = standardizer.analyze_inconsistencies(scan_results)
    
    # 生成报告
    report = standardizer.generate_standardization_report(analysis)
    
    # 保存报告
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"pattern_standardization_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 生成修复脚本
    fix_script = standardizer.generate_fix_script(analysis)
    fix_script_file = f"pattern_name_fix_script_{timestamp}.py"
    
    with open(fix_script_file, 'w', encoding='utf-8') as f:
        f.write(fix_script)
    
    # 显示结果
    print("📊 分析结果:")
    print(f"  总形态数量: {analysis['total_patterns_found']}")
    print(f"  规范形态: {analysis['statistics']['canonical_count']}")
    print(f"  非规范形态: {analysis['statistics']['non_canonical_count']}")
    print(f"  未知形态: {analysis['statistics']['unknown_count']}")
    print(f"  一致性率: {analysis['statistics']['consistency_rate']:.1f}%")
    print()
    print(f"📄 详细报告: {report_file}")
    print(f"🔧 修复脚本: {fix_script_file}")
    
    if analysis['statistics']['consistency_rate'] < 80:
        print("⚠️  一致性率较低，建议立即进行标准化修复")
    else:
        print("✅ 一致性率良好")
    
    return analysis

if __name__ == "__main__":
    run_pattern_standardization_analysis()
