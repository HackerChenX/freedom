#!/usr/bin/env python3
"""
架构合规性检查工具

自动检测架构违规，包括跨层依赖、命名规范、代码重复等问题
"""

import os
import sys
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from datetime import datetime
import subprocess

class ArchitectureChecker:
    """架构合规性检查器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.violations = []
        
        # 六层架构定义
        self.layers = {
            'L6': {'name': '用户接口层', 'dirs': ['bin/', 'api/']},
            'L5': {'name': '业务应用层', 'dirs': ['strategy/', 'analysis/']},
            'L4': {'name': '核心服务层', 'dirs': ['indicators/', 'formula/']},
            'L3': {'name': '数据服务层', 'dirs': ['db/interfaces/', 'db/managers/']},
            'L2': {'name': '存储访问层', 'dirs': ['db/clickhouse_db.py', 'db/unified_data_manager.py']},
            'L1': {'name': '基础设施层', 'dirs': ['utils/', 'config/', 'enums/']}
        }
        
        # 禁止的跨层依赖
        self.forbidden_dependencies = {
            'L6': ['L4', 'L3', 'L2', 'L1'],  # L6只能调用L5
            'L5': ['L3', 'L2', 'L1'],       # L5只能调用L4
            'L4': ['L2', 'L1'],             # L4只能调用L3
            'L3': ['L1'],                   # L3只能调用L2
            'L2': [],                       # L2只能调用L1
            'L1': []                        # L1不依赖其他层
        }
        
        # 命名规范
        self.naming_patterns = {
            'class': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),  # 大驼峰
            'method': re.compile(r'^[a-z][a-z0-9_]*$'),   # 小写+下划线
            'variable': re.compile(r'^[a-z][a-z0-9_]*$'), # 小写+下划线
            'constant': re.compile(r'^[A-Z][A-Z0-9_]*$'), # 大写+下划线
        }
    
    def check_all(self) -> Dict[str, Any]:
        """执行全面的架构检查"""
        print("🔍 开始架构合规性检查...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'cross_layer_violations': self.check_cross_layer_dependencies(),
            'naming_violations': self.check_naming_conventions(),
            'code_duplication': self.check_code_duplication(),
            'direct_db_access': self.check_direct_database_access(),
            'dependency_injection': self.check_dependency_injection_usage(),
            'summary': {}
        }
        
        # 计算总体合规性分数
        results['summary'] = self._calculate_compliance_score(results)
        
        return results
    
    def check_cross_layer_dependencies(self) -> List[Dict[str, Any]]:
        """检查跨层依赖违规"""
        print("  📋 检查跨层依赖...")
        violations = []
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            file_layer = self._get_file_layer(py_file)
            if not file_layer:
                continue
            
            imports = self._extract_imports(py_file)
            
            for import_stmt in imports:
                import_layer = self._get_import_layer(import_stmt)
                if import_layer and self._is_forbidden_dependency(file_layer, import_layer):
                    violations.append({
                        'type': 'cross_layer_dependency',
                        'file': str(py_file.relative_to(self.root_dir)),
                        'file_layer': file_layer,
                        'import_statement': import_stmt,
                        'import_layer': import_layer,
                        'severity': 'critical',
                        'message': f'{self.layers[file_layer]["name"]} 不应直接依赖 {self.layers[import_layer]["name"]}'
                    })
        
        print(f"    发现 {len(violations)} 个跨层依赖违规")
        return violations
    
    def check_naming_conventions(self) -> List[Dict[str, Any]]:
        """检查命名规范"""
        print("  📋 检查命名规范...")
        violations = []
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if not self.naming_patterns['class'].match(node.name):
                            violations.append({
                                'type': 'naming_convention',
                                'file': str(py_file.relative_to(self.root_dir)),
                                'line': node.lineno,
                                'element_type': 'class',
                                'element_name': node.name,
                                'severity': 'medium',
                                'message': f'类名 "{node.name}" 不符合大驼峰命名规范'
                            })
                    
                    elif isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_') and not self.naming_patterns['method'].match(node.name):
                            violations.append({
                                'type': 'naming_convention',
                                'file': str(py_file.relative_to(self.root_dir)),
                                'line': node.lineno,
                                'element_type': 'method',
                                'element_name': node.name,
                                'severity': 'low',
                                'message': f'方法名 "{node.name}" 不符合小写+下划线命名规范'
                            })
            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        print(f"    发现 {len(violations)} 个命名规范违规")
        return violations
    
    def check_code_duplication(self) -> List[Dict[str, Any]]:
        """检查代码重复"""
        print("  📋 检查代码重复...")
        violations = []
        
        # 使用简单的文本匹配检测重复代码
        code_blocks = {}
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 检查连续的代码块（忽略空行和注释）
                for i in range(len(lines) - 5):  # 至少5行
                    block = []
                    for j in range(i, min(i + 10, len(lines))):  # 最多10行
                        line = lines[j].strip()
                        if line and not line.startswith('#') and not line.startswith('"""'):
                            block.append(line)
                    
                    if len(block) >= 5:  # 至少5行有效代码
                        block_hash = hash('\n'.join(block))
                        if block_hash in code_blocks:
                            violations.append({
                                'type': 'code_duplication',
                                'file1': str(code_blocks[block_hash]['file']),
                                'line1': code_blocks[block_hash]['line'],
                                'file2': str(py_file.relative_to(self.root_dir)),
                                'line2': i + 1,
                                'block_size': len(block),
                                'severity': 'medium',
                                'message': f'发现重复代码块（{len(block)} 行）'
                            })
                        else:
                            code_blocks[block_hash] = {
                                'file': py_file.relative_to(self.root_dir),
                                'line': i + 1
                            }
            
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        print(f"    发现 {len(violations)} 个代码重复问题")
        return violations
    
    def check_direct_database_access(self) -> List[Dict[str, Any]]:
        """检查直接数据库访问"""
        print("  📋 检查直接数据库访问...")
        violations = []
        
        forbidden_patterns = [
            r'get_unified_data_manager\(\)',
            r'from db\.unified_data_manager import',
            r'from db\.clickhouse_db import',
            r'\.ch_db\.',
            r'ClickHouseClient\(\)'
        ]
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file) or 'db/' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern in forbidden_patterns:
                        if re.search(pattern, line):
                            violations.append({
                                'type': 'direct_database_access',
                                'file': str(py_file.relative_to(self.root_dir)),
                                'line': i,
                                'pattern': pattern,
                                'content': line.strip(),
                                'severity': 'critical',
                                'message': '应使用依赖注入的数据访问接口，而不是直接访问数据库'
                            })
            
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        print(f"    发现 {len(violations)} 个直接数据库访问违规")
        return violations
    
    def check_dependency_injection_usage(self) -> List[Dict[str, Any]]:
        """检查依赖注入使用情况"""
        print("  📋 检查依赖注入使用...")
        violations = []
        
        service_locator_patterns = [
            r'get_container\(\)',
            r'container\.resolve\(',
            r'container\.get_'
        ]
        
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern in service_locator_patterns:
                        if re.search(pattern, line) and 'config/' not in str(py_file):
                            violations.append({
                                'type': 'service_locator_antipattern',
                                'file': str(py_file.relative_to(self.root_dir)),
                                'line': i,
                                'pattern': pattern,
                                'content': line.strip(),
                                'severity': 'medium',
                                'message': '建议使用构造函数注入而不是服务定位器模式'
                            })
            
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        print(f"    发现 {len(violations)} 个服务定位器使用")
        return violations
    
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
    
    def _get_file_layer(self, file_path: Path) -> str:
        """获取文件所属的架构层"""
        relative_path = file_path.relative_to(self.root_dir)
        path_str = str(relative_path)
        
        for layer, config in self.layers.items():
            for dir_pattern in config['dirs']:
                if path_str.startswith(dir_pattern.rstrip('/')):
                    return layer
        
        return None
    
    def _get_import_layer(self, import_stmt: str) -> str:
        """获取导入语句对应的架构层"""
        for layer, config in self.layers.items():
            for dir_pattern in config['dirs']:
                dir_name = dir_pattern.rstrip('/').replace('.py', '')
                if import_stmt.startswith(dir_name):
                    return layer
        
        return None
    
    def _is_forbidden_dependency(self, from_layer: str, to_layer: str) -> bool:
        """检查是否为禁止的依赖关系"""
        if from_layer not in self.forbidden_dependencies:
            return False
        
        return to_layer in self.forbidden_dependencies[from_layer]
    
    def _extract_imports(self, file_path: Path) -> List[str]:
        """提取文件中的导入语句"""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 匹配 from ... import 和 import 语句
            import_patterns = [
                r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.extend(matches)
        
        except (UnicodeDecodeError, FileNotFoundError):
            pass
        
        return imports
    
    def _calculate_compliance_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算合规性分数"""
        total_violations = 0
        critical_violations = 0
        
        for category in ['cross_layer_violations', 'naming_violations', 'code_duplication', 
                        'direct_db_access', 'dependency_injection']:
            violations = results[category]
            total_violations += len(violations)
            critical_violations += sum(1 for v in violations if v.get('severity') == 'critical')
        
        # 计算分数 (0-100)
        if total_violations == 0:
            score = 100
        else:
            # 严重违规扣分更多
            penalty = critical_violations * 10 + (total_violations - critical_violations) * 2
            score = max(0, 100 - penalty)
        
        return {
            'total_violations': total_violations,
            'critical_violations': critical_violations,
            'compliance_score': score,
            'assessment': self._get_compliance_assessment(score)
        }
    
    def _get_compliance_assessment(self, score: int) -> str:
        """获取合规性评估"""
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "中等"
        elif score >= 60:
            return "及格"
        else:
            return "不合格"


def main_architecture_checker():
    """主函数"""
    root_dir = os.getcwd()
    checker = ArchitectureChecker(root_dir)
    
    print("🚀 架构合规性检查工具")
    print("=" * 50)
    
    results = checker.check_all()
    
    # 输出结果
    print(f"\n📊 检查结果:")
    print(f"  总违规数: {results['summary']['total_violations']}")
    print(f"  严重违规: {results['summary']['critical_violations']}")
    print(f"  合规分数: {results['summary']['compliance_score']}/100")
    print(f"  评估等级: {results['summary']['assessment']}")
    
    # 保存详细报告
    report_file = 'architecture_compliance_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    
    # 如果有严重违规，返回错误代码
    if results['summary']['critical_violations'] > 0:
        print("\n❌ 发现严重违规，需要立即修复")
        return 1
    elif results['summary']['compliance_score'] < 80:
        print("\n⚠️  合规性分数较低，建议优化")
        return 1
    else:
        print("\n✅ 架构合规性检查通过")
        return 0


if __name__ == "__main__":
    sys.exit(main_architecture_checker())