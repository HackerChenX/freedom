#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
针对发现的特定错误模式的精确修复工具
1. 中文字符编码问题
2. Import语句错误模式
3. Return语句错误
4. 括号不匹配问题
"""

import os
import re
import subprocess
from pathlib import Path


class SpecificErrorFixer:
    """特定错误模式修复器"""
    
    def __init__(self):
        self.fixed_count = 0
        
    def fix_chinese_characters(self, content: str) -> str:
        """修复中文标点符号问题"""
        # 中文标点符号替换为英文标点符号
        replacements = {
            '，': ',',
            '。': '.',
            '：': ':',
            '；': ';',
            '？': '?',
            '！': '!',
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '×': '*',
            '、': ',',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'"
        }
        
        for chinese_char, english_char in replacements.items():
            content = content.replace(chinese_char, english_char)
        
        return content
    
    def fix_import_errors(self, content: str) -> str:
        """修复import语句错误"""
        # 修复 "from xxx import yyy = \"\"\"" 模式
        content = re.sub(r'(from\s+\S+\s+import\s+\S+)\s*=\s*""".*?"""', r'\1', content, flags=re.DOTALL)
        return content
    
    def fix_return_errors(self, content: str) -> str:
        """修复return语句错误"""
        # 修复 "return = {...}" 模式
        content = re.sub(r'(\s+)return\s*=\s*(\{.*?\})', r'\1return \2', content)
        content = re.sub(r'(\s+)return\s*=\s*([^\\n]+)', r'\1return \2', content)
        return content
    
    def fix_bracket_mismatches(self, content: str) -> str:
        """修复简单的括号不匹配问题"""
        lines = content.split('\\n')
        fixed_lines = []
        
        for line in lines:
            # 删除明显多余的单独括号行
            if line.strip() in [')', '}', ']']:
                # 检查前面是否有对应的开括号
                if len(fixed_lines) > 0:
                    last_line = fixed_lines[-1]
                    # 如果上一行已经有完整的结构，跳过这个多余的括号
                    if not (last_line.count('(') > last_line.count(')') or 
                           last_line.count('{') > last_line.count('}') or 
                           last_line.count('[') > last_line.count(']')):
                        continue
            
            fixed_lines.append(line)
        
        return '\\n'.join(fixed_lines)
    
    def fix_triple_quote_errors(self, content: str) -> str:
        """修复三引号错误"""
        # 修复未终止的三引号字符串
        content = re.sub(r'= """.*?(?<!""")\s*$', '= ""', content, flags=re.MULTILINE)
        # 修复含有 = """ 的错误模式
        content = re.sub(r'(\w+.*?)\s*=\s*""".*?$', r'\1', content, flags=re.MULTILINE)
        return content
    
    def fix_file(self, file_path: str) -> bool:
        """修复单个文件的特定错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 应用所有修复
            content = self.fix_chinese_characters(content)
            content = self.fix_import_errors(content)
            content = self.fix_return_errors(content)
            content = self.fix_triple_quote_errors(content)
            content = self.fix_bracket_mismatches(content)
            
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
                    print(f"⚠️  修复后仍有语法错误: {os.path.basename(file_path)}")
                    print(f"   错误: {result.stderr.strip()[:100]}...")
                    return False
            else:
                return True
                
        except Exception as e:
            print(f"❌ 修复失败: {file_path} - {e}")
            return False
    
    def run_batch_fix(self):
        """批量修复indicators目录下的所有文件"""
        print("🚀 开始修复特定错误模式...")
        
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
        
        print(f"\\n📊 修复完成:")
        print(f"   ✅ 成功修复: {self.fixed_count} 个文件")
        print(f"   📁 总处理: {len(python_files)} 个文件")
        print(f"   📈 成功率: {success_count/len(python_files)*100:.1f}%")


if __name__ == "__main__":
    fixer = SpecificErrorFixer()
    fixer.run_batch_fix()
