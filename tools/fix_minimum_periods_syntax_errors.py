#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复最小周期要求添加过程中产生的语法错误

主要修复问题：
1. minimum_periods属性被错误地添加到类外面
2. 缩进错误导致的语法问题
3. 重复的minimum_periods定义
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

class MinimumPeriodsSyntaxFixer:
    """修复最小周期要求语法错误的工具类"""
    
    def __init__(self, indicators_dir: str = "/Users/hacker/PycharmProjects/freedom/indicators"):
        """初始化修复工具"""
        self.indicators_dir = Path(indicators_dir)
        self.fixed_files = []
        self.error_files = []
        
        # 需要修复的文件列表（从错误日志中提取）
        self.error_files_list = [
            'ma.py', 'ema.py', 'rsi.py', 'boll.py', 'psy.py', 'dma.py',
            'adx.py', 'aroon.py', 'cci.py', 'wma.py', 'kdj.py', 'wr.py',
            'stochrsi.py', 'enhanced_rsi.py', 'enhanced_wr.py', 'momentum.py',
            'obv.py', 'vol.py', 'pvt.py', 'chaikin.py', 'chip_distribution.py',
            'institutional_behavior.py', 'enhanced_macd.py', 'enhanced_stochrsi.py',
            'fibonacci.py', 'elliott_wave.py', 'gann_tools.py', 'bias.py', 'mtm.py'
        ]
        
        # 子目录中的文件
        self.subdirectory_files = {
            'pattern/candlestick_patterns.py': 'candlestick_patterns.py',
            'trend/enhanced_dmi.py': 'enhanced_dmi.py',
            'trend/enhanced_macd.py': 'enhanced_macd.py',
            'oscillator/enhanced_kdj.py': 'enhanced_kdj.py',
            'volume/enhanced_obv.py': 'enhanced_obv.py'
        }
    
    def find_syntax_errors(self) -> List[Path]:
        """查找有语法错误的指标文件"""
        error_files = []
        
        # 检查主目录文件
        for filename in self.error_files_list:
            file_path = self.indicators_dir / filename
            if file_path.exists():
                if self._has_syntax_error(file_path):
                    error_files.append(file_path)
        
        # 检查子目录文件
        for subpath, filename in self.subdirectory_files.items():
            file_path = self.indicators_dir / subpath
            if file_path.exists():
                if self._has_syntax_error(file_path):
                    error_files.append(file_path)
        
        return error_files
    
    def _has_syntax_error(self, file_path: Path) -> bool:
        """检查文件是否有语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查常见的语法错误模式
            error_patterns = [
                r'^\s*@property\s*$',  # 类外的@property
                r'^\s*def minimum_periods\(',  # 类外的minimum_periods方法
                r'^\s*return \d+\s*$',  # 类外的return语句
                r'class.*:\s*@property',  # 类定义后直接跟@property
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, content, re.MULTILINE):
                    return True
            
            # 尝试编译检查语法
            try:
                compile(content, str(file_path), 'exec')
                return False
            except SyntaxError:
                return True
                
        except Exception:
            return True
    
    def fix_file_syntax(self, file_path: Path) -> bool:
        """修复单个文件的语法错误"""
        try:
            print(f"🔧 修复文件: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复模式1: 移除类外的minimum_periods相关代码
            content = self._remove_orphaned_minimum_periods(content)
            
            # 修复模式2: 修复缩进错误
            content = self._fix_indentation_errors(content)
            
            # 修复模式3: 移除重复的minimum_periods定义
            content = self._remove_duplicate_minimum_periods(content)
            
            # 如果内容有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  ✅ 修复完成")
                self.fixed_files.append(str(file_path))
                return True
            else:
                print(f"  ⚠️ 无需修复")
                return True
                
        except Exception as e:
            print(f"  ❌ 修复失败: {e}")
            self.error_files.append(str(file_path))
            return False
    
    def _remove_orphaned_minimum_periods(self, content: str) -> str:
        """移除类外的minimum_periods相关代码"""
        lines = content.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # 检查是否是类外的@property或minimum_periods
            if (re.match(r'^\s*@property\s*$', line) or 
                re.match(r'^\s*def minimum_periods\(', line)):
                
                # 检查前面是否有类定义
                has_class_before = False
                for j in range(i-1, max(0, i-20), -1):
                    if re.match(r'^\s*class\s+\w+.*:', lines[j]):
                        has_class_before = True
                        break
                    elif re.match(r'^\s*def\s+\w+\(', lines[j]) and not lines[j].strip().startswith('def minimum_periods'):
                        break
                
                # 如果没有类定义在前面，这是孤立的minimum_periods代码
                if not has_class_before:
                    print(f"    🗑️ 移除孤立的代码: {line.strip()}")
                    
                    # 跳过整个minimum_periods块
                    if re.match(r'^\s*@property\s*$', line):
                        i += 1  # 跳过@property
                        # 跳过def行
                        while i < len(lines) and not re.match(r'^\s*def minimum_periods\(', lines[i]):
                            i += 1
                        if i < len(lines):
                            i += 1  # 跳过def行
                    
                    # 跳过文档字符串和方法体
                    while i < len(lines):
                        current_line = lines[i]
                        if (re.match(r'^\s*"""', current_line) or 
                            re.match(r'^\s*return\s+', current_line) or
                            current_line.strip() == '' or
                            current_line.startswith('        ')):  # 方法体缩进
                            i += 1
                        else:
                            break
                    continue
            
            result_lines.append(line)
            i += 1
        
        return '\n'.join(result_lines)
    
    def _fix_indentation_errors(self, content: str) -> str:
        """修复缩进错误"""
        lines = content.split('\n')
        result_lines = []
        
        for i, line in enumerate(lines):
            # 检查是否是错误缩进的@property
            if re.match(r'^    @property\s*$', line):
                # 检查前一行是否是类定义或方法定义的结束
                if i > 0:
                    prev_line = lines[i-1].strip()
                    if (prev_line.endswith(':') or 
                        prev_line == '' or
                        prev_line.startswith('#')):
                        # 这可能是正确的缩进，保留
                        result_lines.append(line)
                    else:
                        # 这可能是错误的缩进，移除
                        print(f"    🔧 修复缩进错误: {line.strip()}")
                        continue
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_duplicate_minimum_periods(self, content: str) -> str:
        """移除重复的minimum_periods定义"""
        # 查找所有minimum_periods定义
        pattern = r'@property\s*\n\s*def minimum_periods\(.*?\n(?:\s*.*\n)*?\s*return\s+\d+'
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
        
        if len(matches) > 1:
            print(f"    🔍 发现{len(matches)}个minimum_periods定义，保留第一个")
            
            # 保留第一个，移除其他的
            for match in reversed(matches[1:]):
                content = content[:match.start()] + content[match.end():]
        
        return content
    
    def fix_all_syntax_errors(self) -> Dict[str, int]:
        """修复所有语法错误"""
        print("🔍 查找语法错误文件...")
        error_files = self.find_syntax_errors()
        
        print(f"📋 找到{len(error_files)}个有语法错误的文件")
        
        results = {
            'total_files': len(error_files),
            'fixed_files': 0,
            'failed_files': 0
        }
        
        for file_path in error_files:
            if self.fix_file_syntax(file_path):
                results['fixed_files'] += 1
            else:
                results['failed_files'] += 1
        
        return results
    
    def print_summary(self, results: Dict[str, int]):
        """打印修复结果汇总"""
        print("\n" + "="*60)
        print("📊 语法错误修复结果汇总")
        print("="*60)
        
        print(f"📁 总文件数: {results['total_files']}")
        print(f"✅ 修复成功: {results['fixed_files']}")
        print(f"❌ 修复失败: {results['failed_files']}")
        
        if self.fixed_files:
            print(f"\n✅ 修复成功的文件:")
            for file_path in self.fixed_files:
                print(f"  {file_path}")
        
        if self.error_files:
            print(f"\n❌ 修复失败的文件:")
            for file_path in self.error_files:
                print(f"  {file_path}")

    def verify_fixes(self) -> Dict[str, bool]:
        """验证修复效果"""
        print("\n🔍 验证修复效果...")
        verification_results = {}

        for file_path_str in self.fixed_files:
            file_path = Path(file_path_str)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 尝试编译
                compile(content, str(file_path), 'exec')
                verification_results[file_path.name] = True
                print(f"  ✅ {file_path.name}: 语法正确")

            except SyntaxError as e:
                verification_results[file_path.name] = False
                print(f"  ❌ {file_path.name}: 仍有语法错误 - {e}")
            except Exception as e:
                verification_results[file_path.name] = False
                print(f"  ❌ {file_path.name}: 验证失败 - {e}")

        return verification_results

def main():
    """主函数"""
    print("🔧 修复最小周期要求语法错误")
    print("="*60)

    # 创建修复工具
    fixer = MinimumPeriodsSyntaxFixer()

    # 修复所有语法错误
    results = fixer.fix_all_syntax_errors()

    # 打印汇总
    fixer.print_summary(results)

    # 验证修复效果
    if fixer.fixed_files:
        verification_results = fixer.verify_fixes()

        success_count = sum(1 for success in verification_results.values() if success)
        total_count = len(verification_results)

        print(f"\n📊 验证结果: {success_count}/{total_count} 文件语法正确")

        if success_count == total_count:
            print("🎉 所有文件修复成功！")
        else:
            print("⚠️ 部分文件仍需手动修复")

    print(f"\n💡 修复完成后建议:")
    print(f"  1. 运行语法检查确认修复效果")
    print(f"  2. 重新运行指标注册测试")
    print(f"  3. 验证minimum_periods属性正常工作")

if __name__ == "__main__":
    main()
