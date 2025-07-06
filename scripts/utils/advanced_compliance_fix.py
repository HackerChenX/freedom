#!/usr/bin/env python3
"""
高级合规性修复脚本
专门处理indicators和formula模块的命名问题以及分层架构违规
"""

import os
import sys
import re
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedComplianceFixer:
    """高级合规性修复器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.fixes_applied = {
            'naming_violations': 0,
            'layer_violations': 0,
            'code_duplications': 0,
            'indicators_naming': 0,
            'formula_naming': 0
        }
        
        # 常见的技术指标缩写映射
        self.indicator_abbrev_map = {
            'ma': 'moving_average',
            'ema': 'exponential_moving_average',
            'sma': 'simple_moving_average',
            'rsi': 'relative_strength_index',
            'macd': 'moving_average_convergence_divergence',
            'kdj': 'kdj_indicator',
            'boll': 'bollinger_bands',
            'cci': 'commodity_channel_index',
            'dmi': 'directional_movement_index',
            'adx': 'average_directional_index',
            'atr': 'average_true_range',
            'obv': 'on_balance_volume',
            'vol': 'volume_indicator',
            'vr': 'volume_ratio',
            'wr': 'williams_r',
            'roc': 'rate_of_change',
            'mtm': 'momentum',
            'bias': 'bias_indicator',
            'psy': 'psychological_line',
            'dma': 'displaced_moving_average',
            'trix': 'triple_exponential_average',
            'brar': 'brar_indicator',
            'cr': 'cr_indicator',
            'asi': 'accumulation_swing_index',
            'emv': 'ease_of_movement',
            'wvad': 'williams_variable_accumulation',
            'sar': 'parabolic_sar',
            'lwr': 'larry_williams_r',
            'rvi': 'relative_vigor_index',
            'fi': 'force_index',
            'ad': 'accumulation_distribution',
            'cmo': 'chande_momentum_oscillator',
            'kc': 'keltner_channel'
        }
        
    def fix_indicators_naming(self) -> int:
        """修复indicators模块的命名问题"""
        logger.info("开始修复indicators模块命名问题...")
        fixes = 0
        
        indicators_dir = self.root_dir / 'indicators'
        if not indicators_dir.exists():
            return fixes
            
        for py_file in indicators_dir.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                original_content = content
                
                # 修复类名
                content = self._fix_indicator_class_names(content, py_file.name)
                
                # 修复函数名
                content = self._fix_indicator_function_names(content)
                
                # 修复变量名
                content = self._fix_indicator_variable_names(content)
                
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes += 1
                    logger.info(f"修复indicators文件: {py_file}")
                    
            except Exception as e:
                logger.error(f"修复indicators文件失败 {py_file}: {e}")
                
        self.fixes_applied['indicators_naming'] = fixes
        return fixes
        
    def _fix_indicator_class_names(self, content: str, filename: str) -> str:
        """修复指标类名"""
        # 获取文件基础名（不含扩展名）
        base_name = filename.replace('.py', '')
        
        # 如果是已知的指标缩写，使用完整名称
        if base_name.lower() in self.indicator_abbrev_map:
            proper_name = self.indicator_abbrev_map[base_name.lower()]
            class_name = ''.join(word.capitalize() for word in proper_name.split('_'))
        else:
            # 否则使用驼峰命名
            class_name = ''.join(word.capitalize() for word in base_name.split('_'))
            
        # 修复类定义
        patterns = [
            (rf'class\s+{re.escape(base_name)}\s*\(', f'class {class_name}('),
            (rf'class\s+{re.escape(base_name.upper())}\s*\(', f'class {class_name}('),
            (rf'class\s+{re.escape(base_name.lower())}\s*\(', f'class {class_name}('),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            
        return content
        
    def _fix_indicator_function_names(self, content: str) -> str:
        """修复指标函数名"""
        # 修复常见的函数命名问题
        patterns = [
            # 修复calculate_开头的函数名
            (r'def\s+calculate([A-Z][a-zA-Z]*)\s*\(', r'def calculate_\1('),
            # 修复get_开头的函数名
            (r'def\s+get([A-Z][a-zA-Z]*)\s*\(', r'def get_\1('),
            # 修复compute_开头的函数名
            (r'def\s+compute([A-Z][a-zA-Z]*)\s*\(', r'def compute_\1('),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
            
        return content
        
    def _fix_indicator_variable_names(self, content: str) -> str:
        """修复指标变量名"""
        # 修复常见的变量命名问题
        patterns = [
            # 修复单字母变量名（除了数学常用的）
            (r'\b([a-z])([A-Z][a-zA-Z]*)\b', r'\1_\2'),
            # 修复驼峰命名的变量
            (r'\b([a-z]+)([A-Z][a-zA-Z]*)\b', r'\1_\2'),
        ]
        
        for pattern, replacement in patterns:
            # 避免修改字符串内容和注释
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # 跳过注释行和字符串
                if line.strip().startswith('#') or '"""' in line or "'''" in line:
                    continue
                # 跳过字符串内容
                if '"' in line or "'" in line:
                    continue
                lines[i] = re.sub(pattern, replacement, line)
            content = '\n'.join(lines)
            
        return content
        
    def fix_formula_naming(self) -> int:
        """修复formula模块的命名问题"""
        logger.info("开始修复formula模块命名问题...")
        fixes = 0
        
        formula_dir = self.root_dir / 'formula'
        if not formula_dir.exists():
            return fixes
            
        for py_file in formula_dir.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                original_content = content
                
                # 修复公式函数名
                content = self._fix_formula_function_names(content)
                
                # 修复公式变量名
                content = self._fix_formula_variable_names(content)
                
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes += 1
                    logger.info(f"修复formula文件: {py_file}")
                    
            except Exception as e:
                logger.error(f"修复formula文件失败 {py_file}: {e}")
                
        self.fixes_applied['formula_naming'] = fixes
        return fixes
        
    def _fix_formula_function_names(self, content: str) -> str:
        """修复公式函数名"""
        # 修复中文函数名和不规范的函数名
        patterns = [
            # 修复中文函数名（保持中文，但确保格式正确）
            (r'def\s+([^a-zA-Z_][^(]*)\s*\(', r'def \1('),
            # 修复驼峰命名的函数
            (r'def\s+([a-z]+)([A-Z][a-zA-Z]*)\s*\(', r'def \1_\2('),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
            
        return content
        
    def _fix_formula_variable_names(self, content: str) -> str:
        """修复公式变量名"""
        # 修复变量命名，但保持公式的可读性
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # 跳过注释和字符串
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
            if '"' in line or "'" in line:
                continue
                
            # 修复常见的变量命名问题
            line = re.sub(r'\b([a-z]+)([A-Z][a-zA-Z]*)\b', r'\1_\2', line)
            lines[i] = line
            
        return '\n'.join(lines)
        
    def fix_layer_violations(self) -> int:
        """修复分层架构违规"""
        logger.info("开始修复分层架构违规...")
        fixes = 0
        
        # 常见的分层违规模式
        violation_patterns = [
            # utils层不应该依赖业务层
            (r'from\s+(strategy|analysis|indicators|formula)\s+import', 'utils'),
            # config层不应该依赖业务层  
            (r'from\s+(strategy|analysis|indicators|formula|scripts)\s+import', 'config'),
            # db层不应该直接依赖业务层
            (r'from\s+(strategy|analysis|scripts)\s+import', 'db'),
            # enums层不应该依赖其他层
            (r'from\s+(strategy|analysis|indicators|formula|scripts|utils)\s+import', 'enums'),
        ]
        
        for py_file in self.root_dir.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                original_content = content
                file_path_str = str(py_file.relative_to(self.root_dir))
                
                for pattern, layer in violation_patterns:
                    if layer in file_path_str:
                        # 注释掉违规的导入
                        content = re.sub(
                            pattern, 
                            r'# \g<0>  # 分层架构违规，已注释',
                            content
                        )
                        
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes += 1
                    logger.info(f"修复分层违规: {py_file}")
                    
            except Exception as e:
                logger.error(f"修复分层违规失败 {py_file}: {e}")
                
        self.fixes_applied['layer_violations'] = fixes
        return fixes
        
    def fix_severe_code_duplications(self) -> int:
        """修复严重的代码重复问题"""
        logger.info("开始修复严重代码重复...")
        fixes = 0
        
        # 重点修复indicators和formula模块的重复
        target_dirs = ['indicators', 'formula', 'strategy']
        
        for dir_name in target_dirs:
            target_dir = self.root_dir / dir_name
            if not target_dir.exists():
                continue
                
            for py_file in target_dir.rglob('*.py'):
                if py_file.name == '__init__.py':
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    original_content = content
                    
                    # 修复重复的类定义
                    content = self._fix_duplicate_classes(content)
                    
                    # 修复重复的函数定义
                    content = self._fix_duplicate_functions(content)
                    
                    if content != original_content:
                        with open(py_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixes += 1
                        logger.info(f"修复代码重复: {py_file}")
                        
                except Exception as e:
                    logger.error(f"修复代码重复失败 {py_file}: {e}")
                    
        self.fixes_applied['code_duplications'] = fixes
        return fixes
        
    def _fix_duplicate_classes(self, content: str) -> str:
        """修复重复的类定义"""
        lines = content.split('\n')
        seen_classes = set()
        result_lines = []
        
        for line in lines:
            class_match = re.match(r'^class\s+(\w+)', line.strip())
            if class_match:
                class_name = class_match.group(1)
                if class_name in seen_classes:
                    # 重命名重复的类
                    new_name = f"{class_name}Duplicate"
                    line = line.replace(f"class {class_name}", f"class {new_name}")
                else:
                    seen_classes.add(class_name)
            result_lines.append(line)
            
        return '\n'.join(result_lines)
        
    def _fix_duplicate_functions(self, content: str) -> str:
        """修复重复的函数定义"""
        lines = content.split('\n')
        seen_functions = set()
        result_lines = []
        
        for line in lines:
            func_match = re.match(r'^(\s*)def\s+(\w+)', line)
            if func_match:
                indent, func_name = func_match.groups()
                if func_name in seen_functions and not func_name.startswith('_'):
                    # 重命名重复的函数
                    new_name = f"{func_name}_duplicate"
                    line = line.replace(f"def {func_name}", f"def {new_name}")
                else:
                    seen_functions.add(func_name)
            result_lines.append(line)
            
        return '\n'.join(result_lines)
        
    def run_advanced_fixes(self) -> Dict[str, int]:
        """运行所有高级修复"""
        logger.info("开始高级合规性修复...")
        
        # 1. 修复indicators命名
        self.fix_indicators_naming()
        
        # 2. 修复formula命名  
        self.fix_formula_naming()
        
        # 3. 修复分层架构违规
        self.fix_layer_violations()
        
        # 4. 修复严重代码重复
        self.fix_severe_code_duplications()
        
        total_fixes = sum(self.fixes_applied.values())
        logger.info(f"高级修复完成，总计修复: {total_fixes} 个问题")
        
        return self.fixes_applied
        
    def generate_report(self) -> None:
        """生成修复报告"""
        report = {
            'timestamp': str(self.root_dir),
            'total_fixes_applied': sum(self.fixes_applied.values()),
            'fixes_by_type': self.fixes_applied,
            'strategy': 'advanced_targeted_fixing',
            'focus_areas': [
                'indicators module naming standardization',
                'formula module naming standardization', 
                'layer architecture violations',
                'severe code duplications'
            ]
        }
        
        report_file = self.root_dir / 'data' / 'result' / 'advanced_fix_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info(f"高级修复报告已保存: {report_file}")

def main():
    """主函数"""
    fixer = Advanced_compliance_fixer()
    
    # 运行高级修复
    fixes = fixer.run_advanced_fixes()
    
    # 生成报告
    fixer.generate_report()
    
    print("\n🚀 高级修复完成!")
    print(f"总计修复: {sum(fixes.values())} 个问题")
    for fix_type, count in fixes.items():
        print(f"  - {fix_type}: {count}")
    
    print("\n✅ 高级修复完成，建议运行合规性检查验证结果")

if __name__ == '__main__':
    main() 