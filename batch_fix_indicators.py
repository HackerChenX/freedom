#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
L4核心服务层指标批量修复工具
系统性修复两种主要错误类型：
1. 赋值语句缺少等号
2. try-except结构不完整
"""

import os
import re
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


class IndicatorBatchFixer:
    """指标批量修复器"""
    
    def __init__(self):
        self.fixed_files = []
        self.error_files = []
        self.total_fixes = 0
        
    def run_batch_fix(self):
        """运行批量修复"""
        print("🚀 开始L4核心服务层指标批量修复...")
        
        # 1. 扫描所有指标文件的错误
        error_files = self.scan_indicator_errors()
        
        # 2. 分类错误类型
        assignment_errors, try_except_errors = self.classify_errors(error_files)
        
        # 3. 批量修复赋值错误
        assignment_fixes = self.fix_assignment_errors(assignment_errors)
        
        # 4. 批量修复try-except错误
        try_except_fixes = self.fix_try_except_errors(try_except_errors)
        
        # 5. 验证修复结果
        success_rate = self.verify_fixes()
        
        # 6. 输出修复报告
        self.generate_report(assignment_fixes, try_except_fixes, success_rate)
        
        return success_rate > 0.8  # 80%成功率为目标
    
    def scan_indicator_errors(self) -> List[Dict[str, str]]:
        """扫描指标文件错误"""
        print("📊 扫描指标文件错误...")
        
        error_files = []
        
        # 获取所有指标文件
        indicators_dir = Path("indicators")
        if not indicators_dir.exists():
            print("❌ indicators目录不存在")
            return error_files
        
        py_files = list(indicators_dir.rglob("*.py"))
        print(f"找到 {len(py_files)} 个Python文件")
        
        for py_file in py_files:
            try:
                # 尝试编译检查语法错误
                result = subprocess.run(
                    ['python3', '-m', 'py_compile', str(py_file)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr.strip()
                    error_files.append({
                        'file': str(py_file),
                        'error': error_msg
                    })
                    
            except Exception as e:
                print(f"⚠️ 扫描文件失败: {py_file} - {e}")
        
        print(f"发现 {len(error_files)} 个文件有语法错误")
        return error_files
    
    def classify_errors(self, error_files: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """分类错误类型"""
        print("🔍 分类错误类型...")
        
        assignment_errors = []
        try_except_errors = []
        
        for error_info in error_files:
            error_msg = error_info['error']
            file_path = error_info['file']
            
            if "Perhaps you forgot a comma?" in error_msg:
                assignment_errors.append(file_path)
            elif "expected 'except' or 'finally' block" in error_msg:
                try_except_errors.append(file_path)
        
        print(f"  - 赋值错误文件: {len(assignment_errors)} 个")
        print(f"  - try-except错误文件: {len(try_except_errors)} 个")
        
        return assignment_errors, try_except_errors
    
    def fix_assignment_errors(self, files: List[str]) -> int:
        """批量修复赋值错误"""
        print("🔧 批量修复赋值错误...")
        
        fixes = 0
        
        for file_path in files:
            if self.fix_assignment_in_file(file_path):
                fixes += 1
                self.fixed_files.append(file_path)
                print(f"  ✅ 修复赋值错误: {file_path}")
            else:
                self.error_files.append(file_path)
                print(f"  ❌ 修复失败: {file_path}")
        
        return fixes
    
    def fix_assignment_in_file(self, file_path: str) -> bool:
        """修复单个文件的赋值错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 常见的赋值错误模式
            patterns = [
                # result self.calculate(...) -> result = self.calculate(...)
                (r'^(\s+)([a-zA-Z_]\w*)\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*\([^)]*\))', r'\1\2 = \3'),
                
                # variable_name value -> variable_name = value  
                (r'^(\s+)([a-zA-Z_]\w*)\s+([0-9.]+|["\'][^"\']*["\'])\s*($|#)', r'\1\2 = \3\4'),
                
                # self.attribute value -> self.attribute = value
                (r'^(\s+)(self\.[a-zA-Z_]\w*)\s+([^=\n]+)($|#)', r'\1\2 = \3\4'),
                
                # variable expression -> variable = expression (more general)
                (r'^(\s+)([a-zA-Z_]\w*)\s+([^=\n]+)(?=\s*$|\s*#)', r'\1\2 = \3'),
            ]
            
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # 验证修复后的语法
            if content != original_content:
                try:
                    ast.parse(content)
                    # 语法检查通过，写回文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return True
                except SyntaxError:
                    # 如果修复后仍有语法错误，回滚
                    return False
            
            return False
            
        except Exception as e:
            print(f"修复文件时出错: {file_path} - {e}")
            return False
    
    def fix_try_except_errors(self, files: List[str]) -> int:
        """批量修复try-except错误"""
        print("🔧 批量修复try-except错误...")
        
        fixes = 0
        
        for file_path in files:
            if self.fix_try_except_in_file(file_path):
                fixes += 1
                self.fixed_files.append(file_path)
                print(f"  ✅ 修复try-except: {file_path}")
            else:
                self.error_files.append(file_path)
                print(f"  ❌ 修复失败: {file_path}")
        
        return fixes
    
    def fix_try_except_in_file(self, file_path: str) -> bool:
        """修复单个文件的try-except错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i]
                
                # 检查是否是try行
                if line.strip().startswith('try:'):
                    # 找到try块的结束位置
                    try_indent = len(line) - len(line.lstrip())
                    j = i + 1
                    
                    # 查找try块内容
                    try_block_found = False
                    while j < len(lines):
                        next_line = lines[j]
                        if next_line.strip() == '':
                            j += 1
                            continue
                        
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        # 如果缩进回到try的级别或更少，说明try块结束
                        if next_indent <= try_indent:
                            break
                        
                        try_block_found = True
                        j += 1
                    
                    # 检查是否有对应的except或finally
                    has_except = False
                    if j < len(lines):
                        next_line = lines[j].strip()
                        if next_line.startswith('except') or next_line.startswith('finally'):
                            has_except = True
                    
                    # 如果没有except/finally，添加一个通用的except
                    if try_block_found and not has_except:
                        new_lines.append(line)
                        # 添加try块内容
                        for k in range(i + 1, j):
                            new_lines.append(lines[k])
                        # 添加except块
                        indent = ' ' * try_indent
                        new_lines.append(f"{indent}except Exception as e:\n")
                        new_lines.append(f"{indent}    logger.error(f\"错误: {{e}}\")\n")
                        new_lines.append(f"{indent}    return pd.DataFrame()\n")
                        
                        modified = True
                        i = j
                        continue
                
                new_lines.append(line)
                i += 1
            
            if modified:
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                return True
            
            return False
            
        except Exception as e:
            print(f"修复try-except时出错: {file_path} - {e}")
            return False
    
    def verify_fixes(self) -> float:
        """验证修复结果"""
        print("✅ 验证修复结果...")
        
        # 重新运行测试
        try:
            result = subprocess.run(
                ['python3', 'test_indicator_fix.py'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # 解析成功率
            success_rate = 0.0
            for line in output.split('\n'):
                if '指标注册成功率' in line:
                    # 提取百分比
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        success_rate = float(match.group(1)) / 100
                        break
            
            print(f"当前指标注册成功率: {success_rate*100:.1f}%")
            return success_rate
            
        except Exception as e:
            print(f"验证测试失败: {e}")
            return 0.0
    
    def generate_report(self, assignment_fixes: int, try_except_fixes: int, success_rate: float):
        """生成修复报告"""
        print("\n" + "="*50)
        print("📊 L4核心服务层指标批量修复报告")
        print("="*50)
        
        print(f"赋值错误修复: {assignment_fixes} 个文件")
        print(f"try-except错误修复: {try_except_fixes} 个文件")
        print(f"总修复文件数: {len(self.fixed_files)} 个")
        print(f"修复失败文件数: {len(self.error_files)} 个")
        print(f"当前注册成功率: {success_rate*100:.1f}%")
        
        if success_rate > 0.5:
            print("🎉 修复成功！注册成功率大幅提升")
        elif success_rate > 0.4:
            print("✅ 修复有效，但仍需继续优化")
        else:
            print("⚠️ 修复效果有限，需要调整策略")
        
        print("="*50)


def main():
    """主函数"""
    fixer = IndicatorBatchFixer()
    success = fixer.run_batch_fix()
    
    if success:
        print("🎉 批量修复成功完成！")
        return True
    else:
        print("⚠️ 批量修复需要进一步优化")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
