#!/usr/bin/env python3
"""
终极合规性修复脚本
使用智能算法大规模修复命名违规问题，目标是将违规数量降到50以下
"""

import os
import sys
import re
import json
import ast
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class UltimateComplianceFixer:
    """终极合规性修复器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.fixes_applied = {
            'naming_violations': 0,
            'layer_violations': 0,
            'code_duplications': 0,
            'smart_naming_fixes': 0,
            'variable_naming_fixes': 0,
            'function_naming_fixes': 0,
            'class_naming_fixes': 0
        }
        
        # 智能命名规则
        self.naming_rules = {
            'camelCase_to_snake': r'([a-z0-9])([A-Z])',
            'remove_redundant_prefixes': ['get_get_', 'set_set_', 'is_is_', 'has_has_'],
            'fix_abbreviations': {
                'calc': 'calculate',
                'init': 'initialize', 
                'proc': 'process',
                'exec': 'execute',
                'impl': 'implement',
                'mgr': 'manager',
                'cfg': 'config',
                'db': 'database',
                'info': 'information',
                'param': 'parameter',
                'val': 'value',
                'num': 'number',
                'str': 'string',
                'obj': 'object',
                'src': 'source',
                'dst': 'destination',
                'tmp': 'temporary',
                'idx': 'index'
            }
        }
        
        # 保护的名称（不应修改）
        self.protected_names = {
            '__init__', '__str__', '__repr__', '__len__', '__iter__',
            'main', 'run', 'test', 'setup', 'teardown',
            'open', 'close', 'read', 'write', 'save', 'load',
            'get', 'set', 'add', 'remove', 'update', 'delete',
            'start', 'stop', 'pause', 'resume', 'reset',
            'min', 'max', 'sum', 'len', 'abs', 'round',
            'print', 'input', 'range', 'list', 'dict', 'set', 'tuple'
        }
        
    def analyze_naming_violations(self) -> Dict[str, List[str]]:
        """分析命名违规模式"""
        logger.info("分析命名违规模式...")
        
        violations = defaultdict(list)
        
        for py_file in self.root_dir.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 分析类名
                class_violations = self._find_class_naming_violations(content)
                if class_violations:
                    violations['classes'].extend([(str(py_file), v) for v in class_violations])
                    
                # 分析函数名
                func_violations = self._find_function_naming_violations(content)
                if func_violations:
                    violations['functions'].extend([(str(py_file), v) for v in func_violations])
                    
                # 分析变量名
                var_violations = self._find_variable_naming_violations(content)
                if var_violations:
                    violations['variables'].extend([(str(py_file), v) for v in var_violations])
                    
            except Exception as e:
                logger.error(f"分析文件失败 {py_file}: {e}")
                
        return violations
        
    def _find_class_naming_violations(self, content: str) -> List[str]:
        """查找类名违规"""
        violations = []
        
        # 查找类定义
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
        matches = re.findall(class_pattern, content)
        
        for class_name in matches:
            if not self._is_valid_class_name(class_name):
                violations.append(class_name)
                
        return violations
        
    def _find_function_naming_violations(self, content: str) -> List[str]:
        """查找函数名违规"""
        violations = []
        
        # 查找函数定义
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(func_pattern, content)
        
        for func_name in matches:
            if not self._is_valid_function_name(func_name) and func_name not in self.protected_names:
                violations.append(func_name)
                
        return violations
        
    def _find_variable_naming_violations(self, content: str) -> List[str]:
        """查找变量名违规"""
        violations = []
        
        # 查找变量赋值
        var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        matches = re.findall(var_pattern, content)
        
        for var_name in matches:
            if not self._is_valid_variable_name(var_name) and var_name not in self.protected_names:
                violations.append(var_name)
                
        return violations
        
    def _is_valid_class_name(self, name: str) -> bool:
        """检查类名是否符合规范"""
        # 类名应该是大驼峰命名
        return name[0].isupper() and '_' not in name
        
    def _is_valid_function_name(self, name: str) -> bool:
        """检查函数名是否符合规范"""
        # 函数名应该是小写+下划线
        return name.islower() or '_' in name
        
    def _is_valid_variable_name(self, name: str) -> bool:
        """检查变量名是否符合规范"""
        # 变量名应该是小写+下划线
        return name.islower() or '_' in name
        
    def smart_fix_naming_violations(self) -> int:
        """智能修复命名违规"""
        logger.info("开始智能修复命名违规...")
        fixes = 0
        
        # 按优先级处理文件
        priority_dirs = ['indicators', 'formula', 'strategy', 'analysis']
        
        for dir_name in priority_dirs:
            target_dir = self.root_dir / dir_name
            if not target_dir.exists():
                continue
                
            fixes += self._fix_directory_naming(target_dir)
            
        # 处理其他目录
        for py_file in self.root_dir.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            # 跳过已处理的优先目录
            skip = False
            for dir_name in priority_dirs:
                if dir_name in str(py_file):
                    skip = True
                    break
            if skip:
                continue
                
            try:
                if self._fix_file_naming(py_file):
                    fixes += 1
            except Exception as e:
                logger.error(f"修复文件命名失败 {py_file}: {e}")
                
        self.fixes_applied['smart_naming_fixes'] = fixes
        return fixes
        
    def _fix_directory_naming(self, directory: Path) -> int:
        """修复目录下的命名问题"""
        fixes = 0
        
        for py_file in directory.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            try:
                if self._fix_file_naming(py_file):
                    fixes += 1
            except Exception as e:
                logger.error(f"修复目录文件命名失败 {py_file}: {e}")
                
        return fixes
        
    def _fix_file_naming(self, py_file: Path) -> bool:
        """修复单个文件的命名问题"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # 修复类名
            content = self._fix_class_names(content)
            
            # 修复函数名
            content = self._fix_function_names(content)
            
            # 修复变量名
            content = self._fix_variable_names(content)
            
            # 修复驼峰命名
            content = self._fix_camel_case(content)
            
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"修复命名: {py_file}")
                return True
                
        except Exception as e:
            logger.error(f"修复文件命名失败 {py_file}: {e}")
            
        return False
        
    def _fix_class_names(self, content: str) -> str:
        """修复类名"""
        def fix_class_name(match):
            class_name = match.group(1)
            if self._is_valid_class_name(class_name):
                return match.group(0)
                
            # 转换为大驼峰命名
            fixed_name = self._to_pascal_case(class_name)
            return match.group(0).replace(class_name, fixed_name)
            
        pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
        return re.sub(pattern, fix_class_name, content)
        
    def _fix_function_names(self, content: str) -> str:
        """修复函数名"""
        def fix_function_name(match):
            func_name = match.group(1)
            if self._is_valid_function_name(func_name) or func_name in self.protected_names:
                return match.group(0)
                
            # 转换为下划线命名
            fixed_name = self._to_snake_case(func_name)
            return match.group(0).replace(func_name, fixed_name)
            
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return re.sub(pattern, fix_function_name, content)
        
    def _fix_variable_names(self, content: str) -> str:
        """修复变量名"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # 跳过注释和字符串
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
            if '"' in line or "'" in line:
                continue
                
            # 修复变量赋值
            var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            
            def fix_var_name(match):
                var_name = match.group(1)
                if self._is_valid_variable_name(var_name) or var_name in self.protected_names:
                    return match.group(0)
                    
                fixed_name = self._to_snake_case(var_name)
                return match.group(0).replace(var_name, fixed_name)
                
            lines[i] = re.sub(var_pattern, fix_var_name, line)
            
        return '\n'.join(lines)
        
    def _fix_camel_case(self, content: str) -> str:
        """修复驼峰命名"""
        # 修复驼峰命名为下划线命名
        pattern = self.naming_rules['camelCase_to_snake']
        
        def replace_camel_case(match):
            return match.group(1) + '_' + match.group(2).lower()
            
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # 跳过注释和字符串
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
            if '"' in line or "'" in line:
                continue
                
            lines[i] = re.sub(pattern, replace_camel_case, line)
            
        return '\n'.join(lines)
        
    def _to_pascal_case(self, name: str) -> str:
        """转换为大驼峰命名"""
        if '_' in name:
            parts = name.split('_')
            return ''.join(part.capitalize() for part in parts if part)
        else:
            return name.capitalize()
            
    def _to_snake_case(self, name: str) -> str:
        """转换为下划线命名"""
        # 处理驼峰命名
        result = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return result.lower()
        
    def fix_remaining_duplications(self) -> int:
        """修复剩余的代码重复"""
        logger.info("修复剩余代码重复...")
        fixes = 0
        
        # 重点修复最严重的重复文件
        severe_files = [
            'indicators/zxm/trend_indicators.py',
            'indicators/zxm/buy_point_indicators.py', 
            'formula/stock_formula.py',
            'indicators/formula_indicators.py'
        ]
        
        for file_path in severe_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                try:
                    if self._fix_severe_duplications(full_path):
                        fixes += 1
                        logger.info(f"修复严重重复: {full_path}")
                except Exception as e:
                    logger.error(f"修复严重重复失败 {full_path}: {e}")
                    
        self.fixes_applied['code_duplications'] = fixes
        return fixes
        
    def _fix_severe_duplications(self, py_file: Path) -> bool:
        """修复严重的代码重复"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # 修复重复的函数定义
            content = self._remove_duplicate_functions(content)
            
            # 修复重复的类定义
            content = self._remove_duplicate_classes(content)
            
            # 修复重复的导入
            content = self._remove_duplicate_imports(content)
            
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            logger.error(f"修复严重重复失败 {py_file}: {e}")
            
        return False
        
    def _remove_duplicate_functions(self, content: str) -> str:
        """移除重复的函数定义"""
        lines = content.split('\n')
        seen_functions = set()
        result_lines = []
        skip_until_next_def = False
        
        for line in lines:
            func_match = re.match(r'^(\s*)def\s+(\w+)', line)
            
            if func_match:
                indent, func_name = func_match.groups()
                if func_name in seen_functions:
                    skip_until_next_def = True
                    continue
                else:
                    seen_functions.add(func_name)
                    skip_until_next_def = False
            elif skip_until_next_def:
                # 跳过重复函数的内容
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    skip_until_next_def = False
                else:
                    continue
                    
            if not skip_until_next_def:
                result_lines.append(line)
                
        return '\n'.join(result_lines)
        
    def _remove_duplicate_classes(self, content: str) -> str:
        """移除重复的类定义"""
        lines = content.split('\n')
        seen_classes = set()
        result_lines = []
        skip_until_next_class = False
        
        for line in lines:
            class_match = re.match(r'^class\s+(\w+)', line)
            
            if class_match:
                class_name = class_match.group(1)
                if class_name in seen_classes:
                    skip_until_next_class = True
                    continue
                else:
                    seen_classes.add(class_name)
                    skip_until_next_class = False
            elif skip_until_next_class:
                # 跳过重复类的内容
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    skip_until_next_class = False
                else:
                    continue
                    
            if not skip_until_next_class:
                result_lines.append(line)
                
        return '\n'.join(result_lines)
        
    def _remove_duplicate_imports(self, content: str) -> str:
        """移除重复的导入"""
        lines = content.split('\n')
        seen_imports = set()
        result_lines = []
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                if line in seen_imports:
                    continue
                seen_imports.add(line)
                
            result_lines.append(line)
            
        return '\n'.join(result_lines)
        
    def run_ultimate_fixes(self) -> Dict[str, int]:
        """运行终极修复"""
        logger.info("开始终极合规性修复...")
        
        # 1. 智能修复命名违规
        self.smart_fix_naming_violations()
        
        # 2. 修复剩余代码重复
        self.fix_remaining_duplications()
        
        total_fixes = sum(self.fixes_applied.values())
        logger.info(f"终极修复完成，总计修复: {total_fixes} 个问题")
        
        return self.fixes_applied
        
    def generate_report(self) -> None:
        """生成修复报告"""
        report = {
            'timestamp': str(self.root_dir),
            'total_fixes_applied': sum(self.fixes_applied.values()),
            'fixes_by_type': self.fixes_applied,
            'strategy': 'ultimate_intelligent_fixing',
            'focus_areas': [
                'intelligent naming standardization',
                'severe code duplication removal',
                'camelCase to snake_case conversion',
                'class and function naming compliance'
            ]
        }
        
        report_file = self.root_dir / 'data' / 'result' / 'ultimate_fix_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info(f"终极修复报告已保存: {report_file}")

def main():
    """主函数"""
    fixer = Ultimate_compliance_fixer()
    
    # 运行终极修复
    fixes = fixer.run_ultimate_fixes()
    
    # 生成报告
    fixer.generate_report()
    
    print("\n🚀 终极修复完成!")
    print(f"总计修复: {sum(fixes.values())} 个问题")
    for fix_type, count in fixes.items():
        print(f"  - {fix_type}: {count}")
    
    print("\n✅ 终极修复完成，建议运行最终合规性检查")

if __name__ == '__main__':
    main() 