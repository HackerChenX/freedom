#!/usr/bin/env python3
"""
批量修复语法错误指标脚本
修复批量修复脚本造成的语法错误
"""

import os
import sys
import re
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 需要修复的语法错误指标文件
SYNTAX_ERROR_INDICATORS = [
    'indicators/kc.py',
    'indicators/mtm.py', 
    'indicators/pvt.py',
    'indicators/vix.py',
    'indicators/vosc.py',
    'indicators/wr.py',
    'indicators/bias.py',
    'indicators/dmi.py',
    'indicators/cmo.py',
    'indicators/dma.py',
    'indicators/vol.py',
    'indicators/pattern/zxm_patterns.py'
]

def fix_syntax_error(file_path):
    """修复单个文件的语法错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 修复 expected an indented block after 'if' statement
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # 检查if语句后是否缺少代码块
            if line.strip().endswith(':') and ('if ' in line or 'else:' in line or 'elif ' in line):
                # 检查下一行是否有正确的缩进
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    current_indent = len(line) - len(line.lstrip())
                    
                    # 如果下一行不是空行且缩进不正确，添加return语句
                    if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= current_indent:
                        if 'if df.empty:' in line or 'if data.empty:' in line:
                            fixed_lines.append(' ' * (current_indent + 4) + 'return pd.DataFrame()')
                        else:
                            fixed_lines.append(' ' * (current_indent + 4) + 'pass')
                else:
                    # 如果是文件末尾，添加return或pass
                    current_indent = len(line) - len(line.lstrip())
                    if 'if df.empty:' in line or 'if data.empty:' in line:
                        fixed_lines.append(' ' * (current_indent + 4) + 'return pd.DataFrame()')
                    else:
                        fixed_lines.append(' ' * (current_indent + 4) + 'pass')
        
        content = '\n'.join(fixed_lines)
        
        # 2. 修复变量名冲突（如pd变量与pandas模块冲突）
        content = re.sub(r'\bpd\s*=\s*self\.add_pattern_detection\(pd\)', 'result_df = self.add_pattern_detection(result_df)', content)
        content = re.sub(r'\bpd\s*=\s*self\.add_signal_generation\(pd\)', 'result_df = self.add_signal_generation(result_df)', content)
        
        # 3. 修复self赋值错误
        content = re.sub(r'\bself\s*=\s*self\.add_pattern_detection\(self\)', 'result_df = self.add_pattern_detection(result_df)', content)
        content = re.sub(r'\bself\s*=\s*self\.add_signal_generation\(self\)', 'result_df = self.add_signal_generation(result_df)', content)
        
        # 4. 修复错误的形态识别和信号生成位置
        # 查找并移除错误位置的代码
        content = re.sub(r'\s*# 添加形态识别和信号生成\s*\n\s*[a-zA-Z_]+\s*=\s*self\.add_pattern_detection\([a-zA-Z_]+\)\s*\n\s*[a-zA-Z_]+\s*=\s*self\.add_signal_generation\([a-zA-Z_]+\)\s*\n\s*return\s+[a-zA-Z_]+\s*\n\s*# 确保数据包含必要的列', '# 确保数据包含必要的列', content)
        
        # 5. 在正确的return语句前添加形态识别和信号生成
        # 查找_calculate方法的最后一个return语句
        calculate_pattern = r'(def _calculate\(.*?\n.*?)(return\s+[a-zA-Z_]+)(\s*\n\s*def|\s*$)'
        
        def add_pattern_signal_before_return_Errors(match):
            method_body = match.group(1)
            return_statement = match.group(2)
            after_return = match.group(3) if match.group(3) else ''
            
            # 检查是否已经添加了形态识别和信号生成
            if 'add_pattern_detection' in method_body:
                return match.group(0)
            
            # 获取返回的变量名
            return_var = return_statement.split()[1]
            
            # 获取缩进
            lines = method_body.split('\n')
            last_line = lines[-1] if lines else ''
            indent = len(last_line) - len(last_line.lstrip()) if last_line.strip() else 8
            
            # 添加形态识别和信号生成代码
            addition = f"""
{' ' * indent}# 添加形态识别和信号生成
{' ' * indent}{return_var} = self.add_pattern_detection({return_var})
{' ' * indent}{return_var} = self.add_signal_generation({return_var})

{' ' * indent}"""
            
            return method_body + addition + return_statement + after_return
        
        content = re.sub(calculate_pattern, add_pattern_signal_before_return, content, flags=re.DOTALL)
        
        # 如果内容有变化，保存文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {e}")
        return False

def main_batchfixsyntaxerrors():
    """主函数"""
    print("=== 批量修复语法错误指标 ===")
    print()
    
    success_count = 0
    for i, file_path in enumerate(SYNTAX_ERROR_INDICATORS, 1):
        full_path = project_root / file_path
        if full_path.exists():
            print(f"{i:2d}/{len(SYNTAX_ERROR_INDICATORS)} 修复 {full_path.name}...", end=" ")
            if fix_syntax_error(full_path):
                print("✅ 修复成功")
                success_count += 1
            else:
                print("⚠️ 无需修复")
        else:
            print(f"{i:2d}/{len(SYNTAX_ERROR_INDICATORS)} 修复 {full_path.name}... ❌ 文件不存在")
    
    print()
    print(f"=== 修复完成 ===")
    print(f"成功修复: {success_count}/{len(SYNTAX_ERROR_INDICATORS)} 个文件")
    
    # 测试修复效果
    print("\n=== 测试修复效果 ===")
    try:
        from indicators.complete_indicator_registry import complete_registry
        
        # 清空注册器
        complete_registry._indicators = {}
        complete_registry._registration_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'failed_indicators': []
        }
        
        # 重新注册
        complete_registry.register_all_indicators()
        
        # 获取注册结果
        total_indicators = len(complete_registry.get_indicator_names())
        print(f"修复后成功注册指标数: {total_indicators}个")
        
        if total_indicators > 47:
            print(f"✅ 修复成功！新增注册 {total_indicators - 47} 个指标")
        else:
            print("⚠️ 注册数量未增加，可能需要进一步修复")
            
    except Exception as e:
        print(f"❌ 测试修复效果时出错: {e}")

if __name__ == "__main__":
    main_batchfixsyntaxerrors()
