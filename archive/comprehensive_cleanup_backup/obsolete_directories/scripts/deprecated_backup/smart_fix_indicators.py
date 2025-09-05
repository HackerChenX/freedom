#!/usr/bin/env python3
"""
智能修复技术指标脚本
只修复真正需要修复的核心指标文件
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 需要修复的核心指标文件
CORE_INDICATORS = [
    'indicators/kdj.py',
    'indicators/boll.py', 
    'indicators/ma.py',
    'indicators/ema.py',
    'indicators/cci.py',
    'indicators/dmi.py',
    'indicators/stochrsi.py',
    'indicators/wr.py'
]

def fix_indicator_file(file_path):
    """修复单个指标文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经修复
        if "PatternSignalMixin" in content:
            print(f"     ✅ 已修复")
            return True
        
        # 1. 添加PatternSignalMixin导入
        if "from indicators.base_indicator import BaseIndicator" in content:
            content = content.replace(
                "from indicators.base_indicator import BaseIndicator",
                "from indicators.base_indicator import BaseIndicator\nfrom indicators.base.pattern_signal_mixin import PatternSignalMixin"
            )
        
        # 2. 修改类定义，添加PatternSignalMixin继承
        import re
        class_pattern = r"class\s+(\w+)\(BaseIndicator\):"
        match = re.search(class_pattern, content)
        if match:
            class_name = match.group(1)
            content = re.sub(
                class_pattern,
                f"class {class_name}(BaseIndicator, PatternSignalMixin):",
                content
            )
        
        # 3. 在_calculate方法的return之前添加形态识别和信号生成
        # 查找最后一个return语句
        lines = content.split('\n')
        modified_lines = []
        in_calculate_method = False
        method_indent = 0
        
        for i, line in enumerate(lines):
            # 检测_calculate方法开始
            if 'def _calculate(' in line or 'def calculate_Smart_Fix_Indicators(' in line:
                in_calculate_method = True
                method_indent = len(line) - len(line.lstrip())
                modified_lines.append(line)
                continue
            
            # 检测方法结束
            if in_calculate_method and line.strip() and not line.startswith(' ' * (method_indent + 1)):
                in_calculate_method = False
            
            # 在_calculate方法中查找return语句
            if in_calculate_method and line.strip().startswith('return ') and 'result' in line:
                # 检查是否已经添加了形态识别和信号生成
                if i > 0 and 'add_pattern_detection' not in lines[i-1] and 'add_pattern_detection' not in lines[i-2]:
                    # 获取返回的变量名
                    return_var = line.strip().split()[1]
                    
                    # 添加形态识别和信号生成代码
                    indent = ' ' * (method_indent + 4)
                    modified_lines.append(f'{indent}# 添加形态识别和信号生成')
                    modified_lines.append(f'{indent}{return_var} = self.add_pattern_detection({return_var})')
                    modified_lines.append(f'{indent}{return_var} = self.add_signal_generation({return_var})')
                    modified_lines.append('')
            
            modified_lines.append(line)
        
        # 4. 保存修复后的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines))
        
        print(f"     ✅ 修复成功")
        return True
    
    except Exception as e:
        print(f"     ❌ 修复失败: {e}")
        return False

def main_smartfixindicators():
    """主函数"""
    print("=== 智能修复核心技术指标脚本 ===")
    print()
    
    success_count = 0
    for i, file_path in enumerate(CORE_INDICATORS, 1):
        full_path = project_root / file_path
        if full_path.exists():
            print(f"{i:2d}/{len(CORE_INDICATORS)} 修复 {full_path.name}...", end=" ")
            if fix_indicator_file(full_path):
                success_count += 1
        else:
            print(f"{i:2d}/{len(CORE_INDICATORS)} 修复 {full_path.name}... ❌ 文件不存在")
    
    print()
    print(f"=== 修复完成 ===")
    print(f"成功修复: {success_count}/{len(CORE_INDICATORS)} 个文件")
    print(f"成功率: {success_count/len(CORE_INDICATORS):.1%}")

if __name__ == "__main__":
    mainSmartfixindicators()
