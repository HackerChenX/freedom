#!/usr/bin/env python3
"""
指标形态注册和极性标注检查工具

系统性检查代码库中所有指标脚本文件的形态注册方式和极性标注情况
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class PatternInfo:
    """形态信息"""
    pattern_id: str
    file_path: str
    line_number: int
    has_polarity: bool
    polarity_value: str = ""
    registration_method: str = ""  # 注册方式：new/old
    score_impact: float = 0.0

@dataclass
class IndicatorFileInfo:
    """指标文件信息"""
    file_path: str
    class_name: str
    has_register_patterns_method: bool
    registration_methods: List[str]
    patterns: List[PatternInfo]
    issues: List[str]

class IndicatorPatternAuditor:
    """指标形态审计器"""
    
    def __init__(self, indicators_dir: str = "indicators"):
        self.indicators_dir = Path(indicators_dir)
        self.indicator_files: List[IndicatorFileInfo] = []
        self.issues: List[str] = []
        
    def scan_all_indicator_files(self) -> List[str]:
        """扫描所有指标文件"""
        indicator_files = []
        
        # 扫描indicators目录及其子目录
        for root, dirs, files in os.walk(self.indicators_dir):
            # 跳过__pycache__目录
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    indicator_files.append(file_path)
        
        return sorted(indicator_files)
    
    def analyze_file(self, file_path: str) -> IndicatorFileInfo:
        """分析单个指标文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取类名
            class_name = self._extract_class_name(content)
            
            # 检查是否有register_patterns方法
            has_register_patterns = self._has_register_patterns_method(content)
            
            # 查找注册方法
            registration_methods = self._find_registration_methods(content)
            
            # 提取形态信息
            patterns = self._extract_patterns(content, file_path)
            
            # 检查问题
            issues = self._check_issues(content, patterns, file_path)
            
            return IndicatorFileInfo(
                file_path=file_path,
                class_name=class_name,
                has_register_patterns_method=has_register_patterns,
                registration_methods=registration_methods,
                patterns=patterns,
                issues=issues
            )
            
        except Exception as e:
            return IndicatorFileInfo(
                file_path=file_path,
                class_name="",
                has_register_patterns_method=False,
                registration_methods=[],
                patterns=[],
                issues=[f"文件读取错误: {str(e)}"]
            )
    
    def _extract_class_name(self, content: str) -> str:
        """提取主要的指标类名"""
        # 查找继承自BaseIndicator的类
        pattern = r'class\s+(\w+)\s*\([^)]*BaseIndicator[^)]*\):'
        matches = re.findall(pattern, content)
        if matches:
            return matches[0]
        
        # 查找普通类定义
        pattern = r'class\s+(\w+)\s*\([^)]*\):'
        matches = re.findall(pattern, content)
        if matches:
            return matches[0]
        
        return ""
    
    def _has_register_patterns_method(self, content: str) -> bool:
        """检查是否有register_patterns方法"""
        patterns = [
            r'def\s+register_patterns\s*\(',
            r'def\s+_register_patterns\s*\(',
            r'def\s+_register_\w+_patterns\s*\('
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _find_registration_methods(self, content: str) -> List[str]:
        """查找形态注册方法"""
        methods = []
        
        # 查找新版本注册方法
        if 'register_pattern_to_registry' in content:
            methods.append('new_api')
        if 'registry.register(' in content:
            methods.append('new_registry_api')
        
        # 查找旧版本注册方法
        if 'PatternRegistry.register_indicator_pattern' in content:
            methods.append('old_api')
        
        return methods
    
    def _extract_patterns(self, content: str, file_path: str) -> List[PatternInfo]:
        """提取形态信息"""
        patterns = []
        lines = content.split('\n')
        
        # 查找register_pattern_to_registry调用
        for i, line in enumerate(lines):
            if 'register_pattern_to_registry' in line:
                pattern_info = self._parse_pattern_registration(lines, i, file_path)
                if pattern_info:
                    patterns.append(pattern_info)
        
        # 查找registry.register调用
        for i, line in enumerate(lines):
            if 'registry.register(' in line and 'pattern_id' in line:
                pattern_info = self._parse_registry_registration(lines, i, file_path)
                if pattern_info:
                    patterns.append(pattern_info)
        
        return patterns
    
    def _parse_pattern_registration(self, lines: List[str], start_line: int, file_path: str) -> PatternInfo:
        """解析register_pattern_to_registry调用"""
        # 提取多行注册代码
        registration_code = ""
        i = start_line
        paren_count = 0
        
        while i < len(lines):
            line = lines[i]
            registration_code += line + "\n"
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0 and ')' in line:
                break
            i += 1
        
        # 提取pattern_id
        pattern_id_match = re.search(r'pattern_id=["\']([^"\']+)["\']', registration_code)
        if not pattern_id_match:
            return None
        
        pattern_id = pattern_id_match.group(1)
        
        # 检查是否有polarity
        has_polarity = 'polarity=' in registration_code
        polarity_value = ""
        if has_polarity:
            polarity_match = re.search(r'polarity=["\']([^"\']+)["\']', registration_code)
            if polarity_match:
                polarity_value = polarity_match.group(1)
        
        # 提取score_impact
        score_impact = 0.0
        score_match = re.search(r'score_impact=([+-]?\d+\.?\d*)', registration_code)
        if score_match:
            score_impact = float(score_match.group(1))
        
        return PatternInfo(
            pattern_id=pattern_id,
            file_path=file_path,
            line_number=start_line + 1,
            has_polarity=has_polarity,
            polarity_value=polarity_value,
            registration_method="new_api",
            score_impact=score_impact
        )
    
    def _parse_registry_registration(self, lines: List[str], start_line: int, file_path: str) -> PatternInfo:
        """解析registry.register调用"""
        # 提取多行注册代码
        registration_code = ""
        i = start_line
        paren_count = 0
        
        while i < len(lines):
            line = lines[i]
            registration_code += line + "\n"
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0 and ')' in line:
                break
            i += 1
        
        # 提取pattern_id
        pattern_id_match = re.search(r'pattern_id=["\']([^"\']+)["\']', registration_code)
        if not pattern_id_match:
            return None
        
        pattern_id = pattern_id_match.group(1)
        
        # 检查是否有polarity
        has_polarity = 'polarity=' in registration_code
        polarity_value = ""
        if has_polarity:
            polarity_match = re.search(r'polarity=["\']([^"\']+)["\']', registration_code)
            if polarity_match:
                polarity_value = polarity_match.group(1)
        
        # 提取score_impact
        score_impact = 0.0
        score_match = re.search(r'score_impact=([+-]?\d+\.?\d*)', registration_code)
        if score_match:
            score_impact = float(score_match.group(1))
        
        return PatternInfo(
            pattern_id=pattern_id,
            file_path=file_path,
            line_number=start_line + 1,
            has_polarity=has_polarity,
            polarity_value=polarity_value,
            registration_method="new_registry_api",
            score_impact=score_impact
        )
    
    def _check_issues(self, content: str, patterns: List[PatternInfo], file_path: str) -> List[str]:
        """检查问题"""
        issues = []
        
        # 检查是否使用旧版本API
        if 'PatternRegistry.register_indicator_pattern' in content:
            issues.append("使用旧版本形态注册API")
        
        # 检查缺失极性标注的形态
        patterns_without_polarity = [p for p in patterns if not p.has_polarity]
        if patterns_without_polarity:
            pattern_ids = [p.pattern_id for p in patterns_without_polarity]
            issues.append(f"缺失极性标注的形态: {', '.join(pattern_ids)}")
        
        # 检查是否有形态注册但没有register_patterns方法
        if patterns and not self._has_register_patterns_method(content):
            issues.append("有形态注册但缺少register_patterns方法")
        
        return issues
    
    def run_audit(self) -> Dict:
        """运行完整审计"""
        print("🔍 开始扫描指标文件...")
        
        # 扫描所有文件
        all_files = self.scan_all_indicator_files()
        print(f"发现 {len(all_files)} 个指标文件")
        
        # 分析每个文件
        print("\n📊 分析指标文件...")
        for file_path in all_files:
            print(f"  分析: {file_path}")
            file_info = self.analyze_file(file_path)
            self.indicator_files.append(file_info)
        
        # 生成统计信息
        return self._generate_statistics()
    
    def _generate_statistics(self) -> Dict:
        """生成统计信息"""
        stats = {
            'total_files': len(self.indicator_files),
            'files_with_patterns': 0,
            'files_with_register_method': 0,
            'total_patterns': 0,
            'patterns_with_polarity': 0,
            'patterns_without_polarity': 0,
            'files_with_issues': 0,
            'old_api_usage': 0,
            'new_api_usage': 0
        }
        
        for file_info in self.indicator_files:
            if file_info.patterns:
                stats['files_with_patterns'] += 1
            if file_info.has_register_patterns_method:
                stats['files_with_register_method'] += 1
            if file_info.issues:
                stats['files_with_issues'] += 1
            if 'old_api' in file_info.registration_methods:
                stats['old_api_usage'] += 1
            if 'new_api' in file_info.registration_methods or 'new_registry_api' in file_info.registration_methods:
                stats['new_api_usage'] += 1
            
            stats['total_patterns'] += len(file_info.patterns)
            stats['patterns_with_polarity'] += len([p for p in file_info.patterns if p.has_polarity])
            stats['patterns_without_polarity'] += len([p for p in file_info.patterns if not p.has_polarity])
        
        return stats

def main():
    """主函数"""
    auditor = IndicatorPatternAuditor()
    stats = auditor.run_audit()
    
    print("\n" + "="*60)
    print("📋 指标形态注册审计报告")
    print("="*60)
    
    print(f"\n📊 统计信息:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  有形态的文件: {stats['files_with_patterns']}")
    print(f"  有register_patterns方法的文件: {stats['files_with_register_method']}")
    print(f"  总形态数: {stats['total_patterns']}")
    print(f"  有极性标注的形态: {stats['patterns_with_polarity']}")
    print(f"  缺失极性标注的形态: {stats['patterns_without_polarity']}")
    print(f"  有问题的文件: {stats['files_with_issues']}")
    print(f"  使用旧API的文件: {stats['old_api_usage']}")
    print(f"  使用新API的文件: {stats['new_api_usage']}")
    
    # 详细问题报告
    print(f"\n🚨 问题详情:")
    for file_info in auditor.indicator_files:
        if file_info.issues:
            print(f"\n📁 {file_info.file_path}")
            print(f"   类名: {file_info.class_name}")
            for issue in file_info.issues:
                print(f"   ❌ {issue}")
    
    # 缺失极性的形态详情
    print(f"\n🔍 缺失极性标注的形态详情:")
    for file_info in auditor.indicator_files:
        patterns_without_polarity = [p for p in file_info.patterns if not p.has_polarity]
        if patterns_without_polarity:
            print(f"\n📁 {file_info.file_path}")
            for pattern in patterns_without_polarity:
                print(f"   🔸 {pattern.pattern_id} (行 {pattern.line_number}, 影响值: {pattern.score_impact})")

if __name__ == "__main__":
    main()
