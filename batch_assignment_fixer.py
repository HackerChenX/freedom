#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量赋值错误修复工具
一次性修复文件中的所有赋值语法错误
"""

import re
import subprocess
from pathlib import Path


class BatchAssignmentFixer:
    """批量赋值错误修复器"""
    
    def __init__(self):
        self.fixed_files = []
        
    def fix_all_assignments_in_file(self, file_path: str) -> bool:
        """修复文件中的所有赋值错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            lines = content.split('\n')
            fixed_lines = []
            changes_count = 0
            
            for i, line in enumerate(lines):
                original_line = line
                
                # 跳过注释行、空行、已有等号的行、函数定义等
                if (line.strip().startswith('#') or 
                    not line.strip() or 
                    '=' in line or
                    any(keyword in line for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return', 'import', 'from '])):
                    fixed_lines.append(line)
                    continue
                
                # 应用赋值修复模式
                patterns = [
                    # 基本变量赋值 - variable value
                    (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*(?:\([^)]*\))?)', r'\1\2 = \3'),
                    # self属性赋值 - self.attr value  
                    (r'^(\s+)(self\.[a-zA-Z_][a-zA-Z0-9_]*)\s+([^=].*)', r'\1\2 = \3'),
                    # 数字赋值 - variable 123
                    (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(\d+(?:\.\d+)?)', r'\1\2 = \3'),
                    # 字符串赋值 - variable "string" 
                    (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(".*?")', r'\1\2 = \3'),
                    # None/True/False赋值
                    (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(None|True|False)', r'\1\2 = \3'),
                    # 数组索引赋值 - df['col'] value
                    (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\])\s+([^=].*)', r'\1\2 = \3'),
                    # 复杂表达式赋值 - variable (expression)
                    (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(\(.*)', r'\1\2 = \3'),
                    # pandas操作赋值 - df_copy df.copy()
                    (r'^(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s+(df\.[a-zA-Z_][a-zA-Z0-9_]*.*)', r'\1\2 = \3'),
                ]
                
                for pattern, replacement in patterns:
                    new_line = re.sub(pattern, replacement, line)
                    if new_line != line:
                        fixed_lines.append(new_line)
                        changes_count += 1
                        print(f"  📝 第{i+1}行: {original_line.strip()} → {new_line.strip()}")
                        break
                else:
                    fixed_lines.append(line)
            
            if changes_count > 0:
                # 写回文件
                new_content = '\n'.join(fixed_lines)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  🔄 总共修复了 {changes_count} 行")
                return True
            else:
                print(f"  ℹ️  没有发现需要修复的赋值错误")
                return True
                
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            return False
    
    def fix_file(self, filename: str) -> bool:
        """修复单个文件"""
        file_path = Path("indicators") / filename
        if not file_path.exists():
            print(f"⚠️  文件不存在: {filename}")
            return False
        
        print(f"\n🔧 批量修复文件: {filename}")
        
        # 批量修复赋值错误
        if self.fix_all_assignments_in_file(str(file_path)):
            # 验证语法
            result = subprocess.run(['python3', '-m', 'py_compile', str(file_path)], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {filename} 修复成功，语法正确！")
                return True
            else:
                print(f"⚠️  {filename} 修复后仍有语法错误:")
                print(f"   {result.stderr.strip()[:200]}...")
                return False
        else:
            return False
    
    def run_batch_fixes(self):
        """运行批量修复"""
        print("🚀 L4层批量赋值错误修复启动...")
        
        # 优先修复最简单的文件
        simple_files = [
            'mtm.py',        # 参数定义和简单赋值
            'vol.py',        # 参数定义和简单赋值  
            'vr.py',         # 简单赋值
            'chaikin.py',    # 方法调用赋值
            'vix.py',        # 方法调用赋值
        ]
        
        for filename in simple_files:
            if self.fix_file(filename):
                self.fixed_files.append(filename)
        
        print(f"\n📊 批量修复完成:")
        print(f"   ✅ 成功修复: {len(self.fixed_files)} 个文件")
        print(f"   📁 目标文件: {len(simple_files)} 个")
        print(f"   📈 成功率: {len(self.fixed_files)/len(simple_files)*100:.1f}%")
        
        if self.fixed_files:
            print(f"\n✅ 成功修复的文件:")
            for filename in self.fixed_files:
                print(f"   - {filename}")


if __name__ == "__main__":
    fixer = BatchAssignmentFixer()
    fixer.run_batch_fixes()
