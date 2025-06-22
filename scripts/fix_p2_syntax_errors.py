#!/usr/bin/env python3
"""
P2语法错误指标批量修复脚本
修复剩余的语法错误指标
"""

import os
import sys
import re
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# P2语法错误指标列表（除了已修复的WMA, KC, MTM）
P2_SYNTAX_ERROR_INDICATORS = [
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

def fix_syntax_error_precise(file_path):
    """精确修复单个文件的语法错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 修复 expected an indented block after 'if' statement
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            fixed_lines.append(line)
            
            # 检查if语句后是否缺少代码块
            if line.strip().endswith(':') and ('if ' in line or 'else:' in line or 'elif ' in line):
                # 检查下一行是否有正确的缩进
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    current_indent = len(line) - len(line.lstrip())
                    
                    # 如果下一行不是空行且缩进不正确，添加适当的语句
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
            i += 1
        
        content = '\n'.join(fixed_lines)
        
        # 2. 修复错误的形态识别和信号生成位置
        # 移除错误位置的代码
        error_patterns = [
            r'\s*# 添加形态识别和信号生成\s*\n\s*[a-zA-Z_]+\s*=\s*self\.add_pattern_detection\([a-zA-Z_]+\)\s*\n\s*[a-zA-Z_]+\s*=\s*self\.add_signal_generation\([a-zA-Z_]+\)\s*\n\s*return\s+[a-zA-Z_]+\s*\n',
            r'\s*# 添加形态识别和信号生成\s*\n\s*df\s*=\s*self\.add_pattern_detection\(df\)\s*\n\s*df\s*=\s*self\.add_signal_generation\(df\)\s*\n\s*return\s+df\s*\n'
        ]
        
        for pattern in error_patterns:
            content = re.sub(pattern, '\n', content, flags=re.MULTILINE)
        
        # 3. 在正确的return语句前添加形态识别和信号生成
        # 查找_calculate方法的最后一个return语句
        calculate_pattern = r'(def _calculate\(.*?\n.*?)(return\s+[a-zA-Z_]+)(\s*\n\s*def|\s*$)'
        
        def add_pattern_signal_before_return(match):
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
        
        # 4. 修复导入错误
        content = re.sub(r', PatternResult', '', content)
        content = re.sub(r'from indicators\.base\.pattern_signal_mixin import.*PatternResult.*\n', '', content)
        
        # 5. 确保PatternSignalMixin正确导入
        if 'PatternSignalMixin' in content and 'from indicators.base.pattern_signal_mixin import PatternSignalMixin' not in content:
            # 在BaseIndicator导入后添加PatternSignalMixin导入
            content = re.sub(
                r'from indicators\.base_indicator import BaseIndicator',
                'from indicators.base_indicator import BaseIndicator\nfrom indicators.base.pattern_signal_mixin import PatternSignalMixin',
                content
            )
        
        # 如果内容有变化，保存文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {e}")
        return False

def test_indicator_fix(indicator_file):
    """测试指标修复效果"""
    try:
        # 从文件路径推断模块名和类名
        module_path = indicator_file.replace('/', '.').replace('.py', '')
        class_name = Path(indicator_file).stem.upper()
        
        # 动态导入模块
        module = __import__(module_path, fromlist=[''])
        indicator_class = getattr(module, class_name)
        
        # 创建实例
        indicator = indicator_class()
        
        # 测试计算
        import pandas as pd
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'turnover_rate': [0.01, 0.02, 0.015, 0.025, 0.018]
        })
        
        result = indicator.calculate(test_data)
        
        # 检查形态识别和信号生成列
        pattern_columns = [col for col in result.columns if 'pattern' in col.lower()]
        signal_columns = [col for col in result.columns if 'signal' in col.lower()]
        
        return len(pattern_columns) > 0 and len(signal_columns) > 0
        
    except Exception as e:
        return False

def main():
    """主函数"""
    print("=== P2语法错误指标批量修复 ===")
    print()
    
    success_count = 0
    test_success_count = 0
    
    for i, file_path in enumerate(P2_SYNTAX_ERROR_INDICATORS, 1):
        full_path = project_root / file_path
        if full_path.exists():
            print(f"{i:2d}/{len(P2_SYNTAX_ERROR_INDICATORS)} 修复 {full_path.name}...", end=" ")
            
            if fix_syntax_error_precise(full_path):
                print("✅ 修复", end=" ")
                success_count += 1
                
                # 测试修复效果
                if test_indicator_fix(file_path):
                    print("🎉 测试通过")
                    test_success_count += 1
                else:
                    print("⚠️ 测试失败")
            else:
                print("⚠️ 无需修复")
        else:
            print(f"{i:2d}/{len(P2_SYNTAX_ERROR_INDICATORS)} 修复 {full_path.name}... ❌ 文件不存在")
    
    print()
    print(f"=== P2修复完成 ===")
    print(f"文件修复: {success_count}/{len(P2_SYNTAX_ERROR_INDICATORS)}")
    print(f"测试通过: {test_success_count}/{len(P2_SYNTAX_ERROR_INDICATORS)}")
    
    # 测试整体修复效果
    print("\n=== 测试整体修复效果 ===")
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
        
        if total_indicators > 49:
            print(f"✅ P2修复成功！新增注册 {total_indicators - 49} 个指标")
        else:
            print("⚠️ 注册数量未显著增加，可能需要进一步修复")
            
    except Exception as e:
        print(f"❌ 测试修复效果时出错: {e}")

if __name__ == "__main__":
    main()
