#!/usr/bin/env python3
"""
批量修复技术指标脚本
为所有88个技术指标添加形态识别和信号生成功能
"""

import os
import sys
import re
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def find_indicator_files():
    """查找所有指标文件"""
    indicator_files = []
    indicators_dir = project_root / "indicators"
    
    # 查找所有Python文件
    for file_path in indicators_dir.rglob("*.py"):
        if file_path.name != "__init__.py" and "base" not in str(file_path):
            indicator_files.append(file_path)
    
    return indicator_files

def check_if_needs_fix(file_path):
    """检查文件是否需要修复"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经继承了PatternSignalMixin
        if "PatternSignalMixin" in content:
            return False
        
        # 检查是否是BaseIndicator的子类
        if "class " in content and "BaseIndicator" in content:
            return True
        
        return False
    except Exception as e:
        print(f"检查文件 {file_path} 时出错: {e}")
        return False

def fix_indicator_file(file_path):
    """修复单个指标文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. 添加PatternSignalMixin导入
        if "from indicators.base.pattern_signal_mixin import PatternSignalMixin" not in content:
            # 查找BaseIndicator导入行
            base_import_pattern = r"from indicators\.base_indicator import BaseIndicator"
            if re.search(base_import_pattern, content):
                content = re.sub(
                    base_import_pattern,
                    "from indicators.base_indicator import BaseIndicator\nfrom indicators.base.pattern_signal_mixin import PatternSignalMixin",
                    content
                )
        
        # 2. 修改类定义，添加PatternSignalMixin继承
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
        # 查找_calculate方法的return语句
        calculate_pattern = r"(def _calculate\(self[^}]*?\n.*?)(return\s+\w+)"
        
        def add_pattern_signal(match):
            method_body = match.group(1)
            return_statement = match.group(2)
            
            # 检查是否已经添加了形态识别和信号生成
            if "add_pattern_detection" in method_body:
                return match.group(0)
            
            # 获取返回的变量名
            return_var = return_statement.split()[1]
            
            # 添加形态识别和信号生成代码
            addition = f"""
        # 添加形态识别和信号生成
        {return_var} = self.add_pattern_detection({return_var})
        {return_var} = self.add_signal_generation({return_var})

        """
            
            return method_body + addition + return_statement
        
        content = re.sub(calculate_pattern, add_pattern_signal, content, flags=re.DOTALL)
        
        # 4. 保存修复后的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("=== 批量修复技术指标脚本 ===")
    print()
    
    # 查找所有指标文件
    indicator_files = find_indicator_files()
    print(f"找到 {len(indicator_files)} 个指标文件")
    
    # 检查需要修复的文件
    files_to_fix = []
    for file_path in indicator_files:
        if check_if_needs_fix(file_path):
            files_to_fix.append(file_path)
    
    print(f"需要修复的文件: {len(files_to_fix)} 个")
    
    if not files_to_fix:
        print("✅ 所有指标文件都已修复！")
        return
    
    # 修复文件
    success_count = 0
    for i, file_path in enumerate(files_to_fix, 1):
        print(f"{i:2d}/{len(files_to_fix)} 修复 {file_path.name}...", end=" ")
        
        if fix_indicator_file(file_path):
            print("✅ 成功")
            success_count += 1
        else:
            print("❌ 失败")
    
    print()
    print(f"=== 修复完成 ===")
    print(f"成功修复: {success_count}/{len(files_to_fix)} 个文件")
    print(f"成功率: {success_count/len(files_to_fix):.1%}")

if __name__ == "__main__":
    main()
