#!/usr/bin/env python3
"""
代码重复度检查工具

检测代码重复，包括：
1. 相同函数/方法重复
2. 相同代码块重复
3. 相似逻辑重复
4. 提供重构建议
"""

import os
import sys
import re
import ast
import json
import difflib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import hashlib


@dataclass
class CodeBlock:
    """代码块信息"""
    file_path: str
    start_line: int
    end_line: int
    content: str
    hash_value: str
    type: str  # 'function', 'class', 'block'
    name: Optional[str] = None
    
    def __post_init__(self):
        if not self.hash_value:
            self.hash_value = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class DuplicationResult:
    """重复检测结果"""
    type: str
    severity: str
    blocks: List[CodeBlock]
    similarity_ratio: float
    suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'severity': self.severity,
            'blocks': [
                {
                    'file': block.file_path,
                    'start_line': block.start_line,
                    'end_line': block.end_line,
                    'type': block.type,
                    'name': block.name
                }
                for block in self.blocks
            ],
            'similarity_ratio': self.similarity_ratio,
            'suggestions': self.suggestions
        }


class CodeDuplicationChecker:
    """代码重复度检查器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.duplications = []
        self.min_block_size = 5  # 最小代码块行数
        self.min_similarity_ratio = 0.8  # 最小相似度
        
    def check_all_duplications(self) -> List[DuplicationResult]:
        """检查所有类型的重复"""
        print("🔍 开始代码重复度检查...")
        
        # 1. 收集所有代码块
        all_blocks = self._collect_code_blocks()
        
        # 2. 检查完全重复
        exact_duplicates = self._find_exact_duplicates(all_blocks)
        
        # 3. 检查相似重复
        similar_duplicates = self._find_similar_duplicates(all_blocks)
        
        # 4. 检查函数/方法重复
        function_duplicates = self._find_function_duplicates(all_blocks)
        
        # 5. 合并结果
        results = exact_duplicates + similar_duplicates + function_duplicates
        
        print(f"    发现 {len(results)} 个重复问题")
        return results
    
    def _collect_code_blocks(self) -> List[CodeBlock]:
        """收集所有代码块"""
        print("  📋 收集代码块...")
        blocks = []
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # 解析AST获取函数和类
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        block = self._create_function_block(py_file, node, lines)
                        if block:
                            blocks.append(block)
                    
                    elif isinstance(node, ast.ClassDef):
                        block = self._create_class_block(py_file, node, lines)
                        if block:
                            blocks.append(block)
                
                # 收集普通代码块
                block_blocks = self._create_generic_blocks(py_file, lines)
                blocks.extend(block_blocks)
                
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
                continue
        
        print(f"    收集到 {len(blocks)} 个代码块")
        return blocks
    
    def _create_function_block(self, file_path: Path, node: ast.FunctionDef, lines: List[str]) -> Optional[CodeBlock]:
        """创建函数代码块"""
        if node.end_lineno - node.lineno < self.min_block_size:
            return None
        
        content = '\n'.join(lines[node.lineno-1:node.end_lineno])
        # 标准化内容（去除空行和注释）
        normalized_content = self._normalize_code(content)
        
        return CodeBlock(
            file_path=str(file_path.relative_to(self.root_dir)),
            start_line=node.lineno,
            end_line=node.end_lineno,
            content=normalized_content,
            hash_value="",
            type="function",
            name=node.name
        )
    
    def _create_class_block(self, file_path: Path, node: ast.ClassDef, lines: List[str]) -> Optional[CodeBlock]:
        """创建类代码块"""
        if node.end_lineno - node.lineno < self.min_block_size:
            return None
        
        content = '\n'.join(lines[node.lineno-1:node.end_lineno])
        normalized_content = self._normalize_code(content)
        
        return CodeBlock(
            file_path=str(file_path.relative_to(self.root_dir)),
            start_line=node.lineno,
            end_line=node.end_lineno,
            content=normalized_content,
            hash_value="",
            type="class",
            name=node.name
        )
    
    def _create_generic_blocks(self, file_path: Path, lines: List[str]) -> List[CodeBlock]:
        """创建通用代码块"""
        blocks = []
        
        # 滑动窗口方式创建代码块
        for i in range(len(lines) - self.min_block_size + 1):
            block_lines = []
            for j in range(i, min(i + self.min_block_size * 2, len(lines))):
                line = lines[j].strip()
                if line and not line.startswith('#'):
                    block_lines.append(line)
            
            if len(block_lines) >= self.min_block_size:
                content = '\n'.join(block_lines)
                normalized_content = self._normalize_code(content)
                
                block = CodeBlock(
                    file_path=str(file_path.relative_to(self.root_dir)),
                    start_line=i + 1,
                    end_line=i + len(block_lines),
                    content=normalized_content,
                    hash_value="",
                    type="block"
                )
                blocks.append(block)
        
        return blocks
    
    def _normalize_code(self, code: str) -> str:
        """标准化代码（去除空行、注释、多余空格）"""
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                # 去除多余空格
                line = re.sub(r'\s+', ' ', line)
                lines.append(line)
        return '\n'.join(lines)
    
    def _find_exact_duplicates(self, blocks: List[CodeBlock]) -> List[DuplicationResult]:
        """查找完全重复的代码"""
        print("  📋 检查完全重复...")
        
        duplicates = []
        hash_groups = defaultdict(list)
        
        # 按hash分组
        for block in blocks:
            if len(block.content) > 50:  # 过滤太短的代码
                hash_groups[block.hash_value].append(block)
        
        # 查找重复组
        for hash_value, group in hash_groups.items():
            if len(group) > 1:
                # 排除同一文件内的重复（可能是正常的）
                unique_files = set(block.file_path for block in group)
                if len(unique_files) > 1:
                    suggestions = self._generate_refactoring_suggestions(group)
                    
                    duplicates.append(DuplicationResult(
                        type="exact_duplicate",
                        severity="high",
                        blocks=group,
                        similarity_ratio=1.0,
                        suggestions=suggestions
                    ))
        
        print(f"    发现 {len(duplicates)} 个完全重复")
        return duplicates
    
    def _find_similar_duplicates(self, blocks: List[CodeBlock]) -> List[DuplicationResult]:
        """查找相似重复的代码"""
        print("  📋 检查相似重复...")
        
        duplicates = []
        checked_pairs = set()
        
        for i, block1 in enumerate(blocks):
            for j, block2 in enumerate(blocks[i+1:], i+1):
                if block1.file_path == block2.file_path:
                    continue
                
                pair_key = (min(i, j), max(i, j))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                # 计算相似度
                similarity = difflib.SequenceMatcher(None, block1.content, block2.content).ratio()
                
                if similarity >= self.min_similarity_ratio:
                    suggestions = self._generate_refactoring_suggestions([block1, block2])
                    
                    duplicates.append(DuplicationResult(
                        type="similar_duplicate",
                        severity="medium" if similarity > 0.9 else "low",
                        blocks=[block1, block2],
                        similarity_ratio=similarity,
                        suggestions=suggestions
                    ))
        
        print(f"    发现 {len(duplicates)} 个相似重复")
        return duplicates
    
    def _find_function_duplicates(self, blocks: List[CodeBlock]) -> List[DuplicationResult]:
        """查找重复的函数/方法"""
        print("  📋 检查函数重复...")
        
        duplicates = []
        function_blocks = [b for b in blocks if b.type == "function"]
        
        # 按函数名分组
        name_groups = defaultdict(list)
        for block in function_blocks:
            if block.name:
                name_groups[block.name].append(block)
        
        # 查找同名函数
        for name, group in name_groups.items():
            if len(group) > 1:
                # 检查是否真的重复
                for i, block1 in enumerate(group):
                    for block2 in group[i+1:]:
                        if block1.file_path != block2.file_path:
                            similarity = difflib.SequenceMatcher(None, block1.content, block2.content).ratio()
                            
                            if similarity >= self.min_similarity_ratio:
                                suggestions = [
                                    f"考虑将函数 '{name}' 提取到共同的工具模块中",
                                    f"如果函数功能相同，保留一个实现并统一调用",
                                    f"如果函数功能略有不同，考虑使用参数化重构"
                                ]
                                
                                duplicates.append(DuplicationResult(
                                    type="function_duplicate",
                                    severity="high",
                                    blocks=[block1, block2],
                                    similarity_ratio=similarity,
                                    suggestions=suggestions
                                ))
        
        print(f"    发现 {len(duplicates)} 个函数重复")
        return duplicates
    
    def _generate_refactoring_suggestions(self, blocks: List[CodeBlock]) -> List[str]:
        """生成重构建议"""
        suggestions = []
        
        if len(blocks) == 2:
            block1, block2 = blocks
            
            if block1.type == "function" and block2.type == "function":
                suggestions.extend([
                    f"提取公共函数到 utils/ 或 common/ 模块",
                    f"使用继承或组合模式消除重复",
                    f"考虑使用装饰器模式"
                ])
            
            elif block1.type == "class" and block2.type == "class":
                suggestions.extend([
                    f"提取公共基类",
                    f"使用抽象基类定义接口",
                    f"考虑使用组合模式"
                ])
            
            else:
                suggestions.extend([
                    f"提取公共代码块到独立函数",
                    f"使用模板方法模式",
                    f"考虑配置驱动的方式"
                ])
        
        else:
            suggestions.extend([
                f"创建通用工具类处理重复逻辑",
                f"使用策略模式处理不同情况",
                f"考虑使用配置文件驱动"
            ])
        
        return suggestions
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应跳过文件"""
        skip_patterns = [
            '__pycache__',
            '.pyc',
            'test_',
            '_test.py',
            'tests/',
            'examples/',
            '.git/',
            'node_modules/',
            'venv/',
            '.env',
            'migration'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def generate_report(self, duplications: List[DuplicationResult]) -> Dict[str, Any]:
        """生成检查报告"""
        total_duplications = len(duplications)
        high_severity = sum(1 for d in duplications if d.severity == "high")
        medium_severity = sum(1 for d in duplications if d.severity == "medium")
        low_severity = sum(1 for d in duplications if d.severity == "low")
        
        # 按类型分组
        type_counts = defaultdict(int)
        for d in duplications:
            type_counts[d.type] += 1
        
        # 计算重复度分数
        if total_duplications == 0:
            duplication_score = 100
        else:
            # 严重程度权重计算
            weighted_score = high_severity * 10 + medium_severity * 5 + low_severity * 2
            duplication_score = max(0, 100 - weighted_score)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_duplications': total_duplications,
            'severity_breakdown': {
                'high': high_severity,
                'medium': medium_severity,
                'low': low_severity
            },
            'type_breakdown': dict(type_counts),
            'duplication_score': duplication_score,
            'assessment': self._get_duplication_assessment(duplication_score),
            'duplications': [d.to_dict() for d in duplications]
        }
    
    def _get_duplication_assessment(self, score: int) -> str:
        """获取重复度评估"""
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "中等"
        elif score >= 60:
            return "需要改进"
        else:
            return "严重重复"


def main_code_duplication_checker():
    """主函数"""
    root_dir = os.getcwd()
    checker = CodeDuplicationChecker(root_dir)
    
    print("🚀 代码重复度检查工具")
    print("=" * 50)
    
    # 执行检查
    duplications = checker.check_all_duplications()
    
    # 生成报告
    report = checker.generate_report(duplications)
    
    # 输出结果
    print(f"\n📊 检查结果:")
    print(f"  总重复数: {report['total_duplications']}")
    print(f"  严重重复: {report['severity_breakdown']['high']}")
    print(f"  中等重复: {report['severity_breakdown']['medium']}")
    print(f"  轻微重复: {report['severity_breakdown']['low']}")
    print(f"  重复度分数: {report['duplication_score']}/100")
    print(f"  评估等级: {report['assessment']}")
    
    # 保存详细报告
    report_file = 'code_duplication_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    
    # 输出一些具体的重复示例
    if duplications:
        print(f"\n🔍 重复示例:")
        for i, dup in enumerate(duplications[:3]):  # 只显示前3个
            print(f"\n  {i+1}. {dup.type} (相似度: {dup.similarity_ratio:.2f})")
            for block in dup.blocks:
                print(f"     - {block.file_path}:{block.start_line}-{block.end_line}")
            if dup.suggestions:
                print(f"     建议: {dup.suggestions[0]}")
    
    # 返回状态码
    if report['severity_breakdown']['high'] > 0:
        print(f"\n❌ 发现严重重复，建议立即重构")
        return 1
    elif report['duplication_score'] < 80:
        print(f"\n⚠️  重复度较高，建议优化")
        return 1
    else:
        print(f"\n✅ 代码重复度检查通过")
        return 0


if __name__ == "__main__":
    sys.exit(main())