#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM紧急修复工具
基于质量优先原则，快速修复所有赋值语法错误
确保系统稳定性和指标注册成功率恢复
"""

import re
import subprocess

def emergency_fix_mtm():
    """紧急修复MTM文件中的所有赋值错误"""
    file_path = "indicators/mtm.py"
    
    print("🚨 MTM紧急修复开始...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录修改
    lines = content.split('\n')
    fixed_lines = []
    changes_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 跳过已经正确的行（包含=的赋值、注释、函数定义等）
        if (line.strip().startswith('#') or 
            not line.strip() or 
            ' = ' in line or
            ' == ' in line or
            ' != ' in line or
            ' <= ' in line or
            ' >= ' in line or
            any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from ', 'raise', 'assert', 'yield', 'lambda', 'global', 'nonlocal', '@'])):
            fixed_lines.append(line)
            continue
        
        # 修复各种赋值错误模式
        
        # 1. 参数定义错误: int 10 -> int = 10
        line = re.sub(r'(\w+:\s*int)\s+(\d+)', r'\1 = \2', line)
        line = re.sub(r'(\w+:\s*float)\s+(\d+(?:\.\d+)?)', r'\1 = \2', line)
        line = re.sub(r'(\w+:\s*str)\s+(".*?")', r'\1 = \2', line)
        line = re.sub(r'(\w+:\s*bool)\s+(True|False)', r'\1 = \2', line)
        
        # 2. 基本变量赋值错误: self.name "MTM" -> self.name = "MTM"
        line = re.sub(r'^(\s+)(self\.\w+)\s+(".*?")$', r'\1\2 = \3', line)
        line = re.sub(r'^(\s+)(self\.\w+)\s+([a-zA-Z_]\w*)$', r'\1\2 = \3', line)
        line = re.sub(r'^(\s+)(self\.\w+)\s+(\d+(?:\.\d+)?)$', r'\1\2 = \3', line)
        line = re.sub(r'^(\s+)(self\.\w+)\s+(True|False|None)$', r'\1\2 = \3', line)
        
        # 3. 表达式赋值错误: self.var (expr) -> self.var = (expr)
        line = re.sub(r'^(\s+)(self\.\w+)\s+(\([^)]*\))$', r'\1\2 = \3', line)
        line = re.sub(r'^(\s+)(self\.\w+)\s+(\[.*?\])$', r'\1\2 = \3', line)
        
        # 4. 方法调用赋值错误: df_copy df.copy() -> df_copy = df.copy()
        line = re.sub(r'^(\s+)([a-zA-Z_]\w*)\s+(df\.\w+\([^)]*\))$', r'\1\2 = \3', line)
        line = re.sub(r'^(\s+)([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*\([^)]*\))$', r'\1\2 = \3', line)
        
        # 5. DataFrame操作赋值错误
        line = re.sub(r"^(\s+)(df_copy\['\w+'\])\s+(.+)$", r"\1\2 = \3", line)
        line = re.sub(r"^(\s+)([a-zA-Z_]\w*\['\w+'\])\s+(.+)$", r"\1\2 = \3", line)
        
        # 6. 复杂表达式赋值错误
        line = re.sub(r'^(\s+)([a-zA-Z_]\w*)\s+(.*\w+.*\w+.*)$', r'\1\2 = \3', line)
        
        # 7. 字典初始化错误: signals { -> signals = {
        line = re.sub(r'^(\s+)([a-zA-Z_]\w*)\s+\{$', r'\1\2 = {', line)
        
        # 8. 函数参数错误: period: int None -> period: int = None
        line = re.sub(r'(\w+:\s*\w+)\s+(None)(?=\s*[,)])', r'\1 = \2', line)
        
        if line != original_line:
            changes_count += 1
            print(f"  行 {i+1}: 修复赋值错误")
        
        fixed_lines.append(line)
    
    # 写入修复后的内容
    fixed_content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✅ MTM紧急修复完成，共修复 {changes_count} 行")
    
    # 验证语法
    try:
        subprocess.run(['python3', '-m', 'py_compile', file_path], check=True, capture_output=True)
        print("✅ MTM语法检查通过")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ MTM语法检查失败: {e}")
        return False

if __name__ == "__main__":
    emergency_fix_mtm()
