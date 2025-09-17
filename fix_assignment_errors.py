#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专门修复赋值错误（缺少等号）的精确工具
基于具体错误模式进行精确修复
"""

import os
import re
import subprocess
from pathlib import Path


class AssignmentErrorFixer:
    """赋值错误修复器"""
    
    def __init__(self):
        self.fixed_count = 0
        self.error_patterns = [
            # 常见赋值错误模式
            (r'(\s+)(\w+)\s+(\w+\.\w+\([^)]*\))', r'\1\2 = \3'),  # variable method_call()
            (r'(\s+)(\w+)\s+(\d+)', r'\1\2 = \3'),  # variable number
            (r'(\s+)(\w+)\s+(".*?")', r'\1\2 = \3'),  # variable "string"
            (r'(\s+)(\w+)\s+(\'.*?\')', r'\1\2 = \3'),  # variable 'string'
            (r'(\s+)(\w+)\s+(\[.*?\])', r'\1\2 = \3'),  # variable [list]
            (r'(\s+)(\w+)\s+(\{.*?\})', r'\1\2 = \3'),  # variable {dict}
            (r'(\s+)(\w+)\s+(df\.\w+.*)', r'\1\2 = \3'),  # variable df.something
            (r'(\s+)(\w+)\s+(self\.\w+.*)', r'\1\2 = \3'),  # variable self.something
            (r'(\s+)(self\.\w+)\s+(\w+)', r'\1\2 = \3'),  # self.variable value
            (r'(\s+)(self\.\w+)\s+(".*?")', r'\1\2 = \3'),  # self.variable "string"
            (r'(\s+)(self\.\w+)\s+(\'.*?\')', r'\1\2 = \3'),  # self.variable 'string'
            (r'(\s+)(self\.\w+)\s+(\(.*?\))', r'\1\2 = \3'),  # self.variable (expression)
            (r'(\s+)(df_copy\[\'.*?\'\])\s+(\w+.*)', r'\1\2 = \3'),  # df_copy['col'] value
        ]
    
    def fix_file(self, file_path: str) -> bool:
        """修复单个文件的赋值错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 应用所有赋值错误修复模式
            for pattern, replacement in self.error_patterns:
                content = re.sub(pattern, replacement, content)
            
            # 如果有修改，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # 验证语法
                result = subprocess.run(['python3', '-m', 'py_compile', file_path], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ 成功修复: {file_path}")
                    self.fixed_count += 1
                    return True
                else:
                    print(f"⚠️  修复后仍有语法错误: {file_path}")
                    print(f"   错误: {result.stderr.strip()}")
                    # 如果修复后还有错误，恢复原文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    return False
            else:
                print(f"ℹ️  无需修复: {file_path}")
                return True
                
        except Exception as e:
            print(f"❌ 修复失败: {file_path} - {e}")
            return False
    
    def run_batch_fix(self):
        """批量修复indicators目录下的所有文件"""
        print("🚀 开始批量修复赋值错误...")
        
        indicators_dir = Path("indicators")
        if not indicators_dir.exists():
            print("❌ indicators目录不存在")
            return
        
        python_files = list(indicators_dir.glob("*.py"))
        print(f"📁 找到 {len(python_files)} 个Python文件")
        
        success_count = 0
        for file_path in python_files:
            if self.fix_file(str(file_path)):
                success_count += 1
        
        print(f"\n📊 修复完成:")
        print(f"   ✅ 成功修复: {self.fixed_count} 个文件")
        print(f"   📁 总处理: {len(python_files)} 个文件")
        print(f"   📈 成功率: {success_count/len(python_files)*100:.1f}%")


if __name__ == "__main__":
    fixer = AssignmentErrorFixer()
    fixer.run_batch_fix()
