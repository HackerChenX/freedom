#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终修复剩余赋值错误的专用工具
针对 "invalid syntax. Perhaps you forgot a comma?" 类型错误
"""

import re
import subprocess
from pathlib import Path

class FinalAssignmentFixer:
    """最终赋值错误修复器"""
    
    def __init__(self):
        self.target_files = [
            'mtm.py', 'obv.py', 'vix.py', 'vol.py', 'vr.py', 
            'vosc.py', 'pvt.py', 'chaikin.py', 'psy.py',
            'elliott_wave.py', 'composite.py'
        ]
        self.fixed_count = 0
    
    def fix_specific_line(self, file_path: str, line_num: int, content: str) -> str:
        """修复特定行的赋值错误"""
        lines = content.split('\\n')
        if line_num <= len(lines):
            line = lines[line_num - 1]
            
            # 检测并修复各种赋值错误模式
            patterns = [
                # 基本赋值错误
                (r'^(\s+)(\w+)\s+([A-Za-z_]\w*.*)', r'\\1\\2 = \\3'),  # variable value
                (r'^(\s+)(\w+)\s+(None|True|False|\d+)', r'\\1\\2 = \\3'),  # variable literal
                (r'^(\s+)(\w+)\s+(".*?"|\'.*?\')', r'\\1\\2 = \\3'),  # variable string
                (r'^(\s+)(\w+)\s+(pd\\..*)', r'\\1\\2 = \\3'),  # variable pd.something
                (r'^(\s+)(\w+)\s+(self\\..*)', r'\\1\\2 = \\3'),  # variable self.something
                (r'^(\s+)(self\\.\w+)\s+([A-Za-z_].*)', r'\\1\\2 = \\3'),  # self.variable value
            ]
            
            for pattern, replacement in patterns:
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    lines[line_num - 1] = new_line
                    print(f"  📝 修复第{line_num}行: {line.strip()} → {new_line.strip()}")
                    break
        
        return '\\n'.join(lines)
    
    def fix_file_by_error_line(self, file_path: str) -> bool:
        """基于已知错误行修复文件"""
        try:
            # 先检测语法错误
            result = subprocess.run(['python3', '-m', 'py_compile', file_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {file_path} 已经正确")
                return True
            
            # 解析错误信息获取行号
            error_msg = result.stderr
            if "Perhaps you forgot a comma?" not in error_msg:
                print(f"⚠️  {file_path} 不是赋值错误")
                return False
            
            # 提取行号
            import re
            line_match = re.search(r'line (\d+)', error_msg)
            if not line_match:
                print(f"⚠️  无法解析 {file_path} 的错误行号")
                return False
            
            error_line = int(line_match.group(1))
            print(f"🔧 修复 {file_path} 第{error_line}行...")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修复特定行
            fixed_content = self.fix_specific_line(file_path, error_line, content)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            # 验证修复
            result = subprocess.run(['python3', '-m', 'py_compile', file_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {file_path} 修复成功")
                self.fixed_count += 1
                return True
            else:
                print(f"⚠️  {file_path} 修复后仍有错误: {result.stderr.strip()[:100]}...")
                return False
                
        except Exception as e:
            print(f"❌ 修复 {file_path} 失败: {e}")
            return False
    
    def run_targeted_fix(self):
        """运行针对性修复"""
        print("🎯 开始针对性修复赋值错误...")
        
        indicators_dir = Path("indicators")
        success_count = 0
        
        for filename in self.target_files:
            file_path = indicators_dir / filename
            if file_path.exists():
                print(f"\\n🔧 处理文件: {filename}")
                if self.fix_file_by_error_line(str(file_path)):
                    success_count += 1
            else:
                print(f"⚠️  文件不存在: {filename}")
        
        print(f"\\n📊 针对性修复完成:")
        print(f"   ✅ 成功修复: {self.fixed_count} 个文件")
        print(f"   📁 目标文件: {len(self.target_files)} 个")
        print(f"   📈 成功率: {success_count/len(self.target_files)*100:.1f}%")

if __name__ == "__main__":
    fixer = FinalAssignmentFixer()
    fixer.run_targeted_fix()
