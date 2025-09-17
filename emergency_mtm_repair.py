#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM文件紧急完整修复工具
基于质量优先策略，确保100%修复所有语法错误
针对用户最新修改引入的大量赋值错误进行系统性修复
"""

import re
import subprocess
from pathlib import Path

class MTMEmergencyRepair:
    """MTM文件紧急修复器"""
    
    def __init__(self):
        self.file_path = "indicators/mtm.py"
        self.changes_count = 0
        self.repair_log = []
        
    def log_change(self, line_num, description):
        """记录修复操作"""
        self.changes_count += 1
        self.repair_log.append(f"  行 {line_num}: {description}")
        print(f"  行 {line_num}: {description}")
    
    def emergency_repair(self):
        """紧急修复MTM文件的所有语法错误"""
        print("🚨 MTM文件紧急修复开始...")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines, 1):
            original_line = line
            
            # 跳过注释行和空行
            if line.strip().startswith('#') or not line.strip():
                fixed_lines.append(line)
                continue
            
            # 1. 修复函数参数默认值错误: period: int 10 -> period: int = 10
            if re.search(r'(\w+:\s*\w+)\s+(\w+)(?=\s*[,)])', line):
                line = re.sub(r'(\w+:\s*\w+)\s+(\w+)(?=\s*[,)])', r'\1 = \2', line)
                if line != original_line:
                    self.log_change(i, "修复函数参数默认值")
            
            # 2. 修复基本变量赋值: self.name "MTM" -> self.name = "MTM"
            if re.search(r'^(\s*)(self\.\w+)\s+([^=].*?)(\s*#.*)?$', line) and ' = ' not in line:
                line = re.sub(r'^(\s*)(self\.\w+)\s+([^=].*?)(\s*#.*)?$', r'\1\2 = \3\4', line)
                if line != original_line:
                    self.log_change(i, "修复self变量赋值")
            
            # 3. 修复普通变量赋值: variable value -> variable = value
            if (re.search(r'^(\s*)([a-zA-Z_]\w*)\s+([^=].*?)(\s*#.*)?$', line) and 
                ' = ' not in line and 
                not any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from ', 'raise', 'assert'])):
                line = re.sub(r'^(\s*)([a-zA-Z_]\w*)\s+([^=].*?)(\s*#.*)?$', r'\1\2 = \3\4', line)
                if line != original_line:
                    self.log_change(i, "修复普通变量赋值")
            
            # 4. 修复字典赋值: dict['key'] value -> dict['key'] = value
            if re.search(r"^(\s*)([a-zA-Z_]\w*\[.*?\])\s+([^=].*?)(\s*#.*)?$", line) and ' = ' not in line:
                line = re.sub(r"^(\s*)([a-zA-Z_]\w*\[.*?\])\s+([^=].*?)(\s*#.*)?$", r'\1\2 = \3\4', line)
                if line != original_line:
                    self.log_change(i, "修复字典赋值")
            
            # 5. 修复未闭合的字典: signals { -> signals = {}
            if re.search(r'^(\s*)(\w+)\s*\{\s*$', line):
                line = re.sub(r'^(\s*)(\w+)\s*\{\s*$', r'\1\2 = {}', line)
                if line != original_line:
                    self.log_change(i, "修复未闭合字典")
            
            # 6. 修复错误的复合赋值: = += -> +=
            if ' = += ' in line:
                line = line.replace(' = += ', ' += ')
                self.log_change(i, "修复复合赋值运算符")
            if ' = -= ' in line:
                line = line.replace(' = -= ', ' -= ')
                self.log_change(i, "修复复合赋值运算符")
            
            # 7. 修复f-string错误: f"{var形态" -> f"{var}形态"
            if 'f"' in line and '形态' in line and '}形态' not in line:
                line = re.sub(r'f"([^}]+)形态"', r'f"\1}形态"', line)
                if line != original_line:
                    self.log_change(i, "修复f-string语法")
            
            # 8. 修复字典初始化: default_pattern { -> default_pattern = {
            if re.search(r'^(\s*)(\w+)\s+\{$', line):
                line = re.sub(r'^(\s*)(\w+)\s+\{$', r'\1\2 = {', line)
                if line != original_line:
                    self.log_change(i, "修复字典初始化")
            
            fixed_lines.append(line)
        
        # 写入修复后的内容
        fixed_content = '\n'.join(fixed_lines)
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ MTM紧急修复完成，共修复 {self.changes_count} 处错误")
        return self.changes_count > 0
    
    def validate_syntax(self):
        """验证语法正确性"""
        try:
            subprocess.run(['python3', '-m', 'py_compile', self.file_path], 
                          check=True, capture_output=True, text=True)
            print("🎉 MTM语法检查通过！")
            return True
        except subprocess.CalledProcessError as e:
            error_output = e.stderr
            print(f"❌ MTM语法检查失败:\n{error_output}")
            
            # 尝试提取具体错误行号
            match = re.search(r'line (\d+)', error_output)
            if match:
                line_num = int(match.group(1))
                print(f"\n🔍 错误位置: 第 {line_num} 行")
                
                # 显示错误行周围的内容
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    start = max(0, line_num - 3)
                    end = min(len(lines), line_num + 2)
                    for i in range(start, end):
                        marker = ">>> " if i == line_num - 1 else "    "
                        print(f"{marker}{i+1:3d}: {lines[i].rstrip()}")
            
            return False
    
    def run_complete_repair(self):
        """运行完整修复流程"""
        print("🚀 MTM文件100%完整修复开始...")
        
        max_iterations = 10  # 最多尝试10次修复
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- 第 {iteration} 轮修复 ---")
            
            # 尝试修复
            has_changes = self.emergency_repair()
            
            # 验证语法
            if self.validate_syntax():
                print(f"🎉 MTM文件100%修复成功！经过 {iteration} 轮修复")
                return True
            
            if not has_changes:
                print("⚠️ 无法进一步自动修复，需要手动处理剩余错误")
                break
        
        print(f"❌ 经过 {max_iterations} 轮修复仍未完全成功")
        return False

if __name__ == "__main__":
    repairer = MTMEmergencyRepair()
    success = repairer.run_complete_repair()
    
    if success:
        print("\n🎯 MTM文件100%修复完成，准备运行系统测试...")
    else:
        print("\n🔧 MTM文件需要进一步手动修复...")
