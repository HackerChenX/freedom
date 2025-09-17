#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
L4核心服务层简单语法错误修复工具
专门针对"invalid syntax. Perhaps you forgot a comma?"类型错误
采用"简单优先"策略，确保每个修复都能通过语法检查
"""

import subprocess
import re
from pathlib import Path


class SimpleSyntaxFixer:
    """简单语法错误修复器"""
    
    def __init__(self):
        self.target_files = [
            # 基于测试输出识别的需要修复的文件
            'psy.py',      # line 522
            'obv.py',      # line 102
            'vol.py',      # line 207
            'vr.py',       # line 153
            'vosc.py',     # line 230
            'pvt.py',      # line 83
            'chaikin.py',  # line 62
            'vix.py',      # line 69
            'elliott_wave.py',  # line 105
            'composite.py',     # line 109
            'mtm.py',      # line 36
        ]
        self.fixed_files = []
        self.failed_files = []

    def get_syntax_error_line(self, file_path: str) -> tuple:
        """获取语法错误的具体行号和错误信息"""
        try:
            result = subprocess.run(['python3', '-m', 'py_compile', file_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                return None, None  # 没有语法错误
            
            error_msg = result.stderr
            if "Perhaps you forgot a comma?" not in error_msg:
                return None, error_msg  # 不是目标错误类型
            
            # 提取行号
            line_match = re.search(r'line (\d+)', error_msg)
            if line_match:
                return int(line_match.group(1)), error_msg
            
            return None, error_msg
            
        except Exception as e:
            return None, str(e)

    def fix_specific_assignment_error(self, file_path: str, error_line: int) -> bool:
        """修复特定行的赋值错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if error_line > len(lines):
                return False
            
            # 获取错误行
            line = lines[error_line - 1]
            original_line = line
            
            # 常见的赋值错误模式修复
            patterns = [
                # 参数定义错误：period: int 10 -> period: int = 10
                (r'(\w+:\s*\w+)\s+(\w+)', r'\1 = \2'),
                # 变量赋值错误：signals pd.DataFrame -> signals = pd.DataFrame
                (r'^(\s*)(\w+)\s+([a-zA-Z_]\w*.*)', r'\1\2 = \3'),
                # self属性赋值：self.name "value" -> self.name = "value"
                (r'^(\s*)(self\.\w+)\s+([^=].*)', r'\1\2 = \3'),
                # 方法调用赋值：result func() -> result = func()
                (r'^(\s*)(\w+)\s+(\w+\(.*\))', r'\1\2 = \3'),
            ]
            
            for pattern, replacement in patterns:
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    lines[error_line - 1] = new_line
                    print(f"  📝 修复第{error_line}行: {original_line.strip()} → {new_line.strip()}")
                    
                    # 写回文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ 修复失败: {e}")
            return False

    def fix_file(self, filename: str) -> bool:
        """修复单个文件"""
        file_path = Path("indicators") / filename
        if not file_path.exists():
            print(f"⚠️  文件不存在: {filename}")
            return False
        
        print(f"\n🔧 修复文件: {filename}")
        
        # 检查语法错误
        error_line, error_msg = self.get_syntax_error_line(str(file_path))
        
        if error_line is None:
            if error_msg is None:
                print(f"✅ {filename} 语法已正确")
                return True
            else:
                print(f"⚠️  {filename} 错误类型不匹配: {error_msg[:100]}...")
                return False
        
        # 尝试修复
        if self.fix_specific_assignment_error(str(file_path), error_line):
            # 验证修复结果
            new_error_line, new_error_msg = self.get_syntax_error_line(str(file_path))
            
            if new_error_line is None and new_error_msg is None:
                print(f"✅ {filename} 修复成功！")
                return True
            else:
                print(f"⚠️  {filename} 修复后仍有错误: {new_error_msg[:100]}...")
                return False
        else:
            print(f"❌ {filename} 修复失败")
            return False

    def run_simple_fixes(self):
        """运行简单修复策略"""
        print("🎯 L4层简单优先修复策略启动...")
        print(f"📁 目标文件: {len(self.target_files)} 个")
        
        for filename in self.target_files:
            if self.fix_file(filename):
                self.fixed_files.append(filename)
            else:
                self.failed_files.append(filename)
        
        print(f"\n📊 修复完成:")
        print(f"   ✅ 成功修复: {len(self.fixed_files)} 个文件")
        print(f"   ❌ 修复失败: {len(self.failed_files)} 个文件")
        print(f"   📈 成功率: {len(self.fixed_files)/len(self.target_files)*100:.1f}%")
        
        if self.fixed_files:
            print(f"\n✅ 成功修复的文件:")
            for filename in self.fixed_files:
                print(f"   - {filename}")


if __name__ == "__main__":
    fixer = SimpleSyntaxFixer()
    fixer.run_simple_fixes()
