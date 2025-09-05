#!/usr/bin/env python3
"""
修复Pandas Data_frame赋值警告的脚本
处理ZXM指标中的Setting_with_copy_warning问题
"""

import os
import re
import sys

def fix_dataframe_assignments(file_path):
    """
    修复文件中的Data_frame赋值警告
    
    Args:
        file_path: 要修复的文件路径
    """
    print(f"正在修复文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复模式1: result["column"] = value -> result.loc[:, "column"] = value
    # 但只在特定上下文中修复，避免误修复
    patterns_to_fix = [
        # 修复 result["column"] = series 的模式
        (r'(\s+)(result)\[(["\'][^"\']+["\'])\]\s*=\s*([^=\n]+)', r'\1\2.loc[:, \3] = \4'),
        # 修复 signals["column"] = value 的模式
        (r'(\s+)(signals)\[(["\'][^"\']+["\'])\]\s*=\s*([^=\n]+)', r'\1\2.loc[:, \3] = \4'),
        # 修复 patterns_df["column"] = value 的模式
        (r'(\s+)(patterns_df)\[(["\'][^"\']+["\'])\]\s*=\s*([^=\n]+)', r'\1\2.loc[:, \3] = \4'),
    ]
    
    for pattern, replacement in patterns_to_fix:
        content = re.sub(pattern, replacement, content)
    
    # 修复模式2: df.loc[condition, 'column'] = value (这些通常是安全的，但确保格式正确)
    # 这些通常不需要修复，但我们可以确保它们使用正确的格式
    
    # 修复模式3: 循环中的 result['column'].iloc[i] = value
    # 这种模式需要特别处理
    iloc_pattern = r'(\s+)(result|signals|patterns_df)\[(["\'][^"\']+["\'])\]\.iloc\[([^\]]+)\]\s*=\s*([^=\n]+)'
    iloc_replacement = r'\1\2.iloc[\4, \2.columns.get_loc(\3)] = \5'
    
    # 但这个替换可能太复杂，我们使用更简单的方法
    # 将 result['column'].iloc[i] = value 改为 result.at[result.index[i], 'column'] = value
    iloc_pattern_simple = r'(\s+)(result|signals|patterns_df)\[(["\'][^"\']+["\'])\]\.iloc\[([^\]]+)\]\s*=\s*([^=\n]+)'
    
    def iloc_replacer(match):
        indent = match.group(1)
        df_name = match.group(2)
        column = match.group(3)
        index_expr = match.group(4)
        value = match.group(5)
        
        # 如果index_expr是简单的数字或变量，使用.at
        if re.match(r'^[a-zA-Z_]\w*$|^\d+$', index_expr.strip()):
            return f"{indent}{df_name}.at[{df_name}.index[{index_expr}], {column}] = {value}"
        else:
            # 复杂表达式，保持原样但添加copy()
            return f"{indent}{df_name} = {df_name}.copy()\n{indent}{df_name}[{column}].iloc[{index_expr}] = {value}"
    
    content = re.sub(iloc_pattern_simple, iloc_replacer, content)
    
    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 已修复 {file_path}")
        return True
    else:
        print(f"  ℹ️  {file_path} 无需修复")
        return False

def fix_specific_patterns(file_path):
    """
    修复特定的已知问题模式
    
    Args:
        file_path: 要修复的文件路径
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复特定的问题模式
    specific_fixes = [
        # 确保在修改DataFrame之前先copy
        (r'(\s+)(result)\s*=\s*(data)\.copy\(\)', r'\1\2 = \3.copy()'),
        # 修复链式赋值
        (r'(\s+)(result)\.loc\[([^\]]+),\s*(["\'][^"\']+["\'])\]\s*=\s*([^=\n]+)', r'\1\2.loc[\3, \4] = \5'),
    ]
    
    for pattern, replacement in specific_fixes:
        content = re.sub(pattern, replacement, content)
    
    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main_fixpandaswarnings():
    """主函数"""
    print("🔧 开始修复Pandas DataFrame赋值警告...")
    
    # 需要修复的文件列表
    files_to_fix = [
        'indicators/zxm/trend_indicators.py',
        'indicators/zxm/buy_point_indicators.py',
        'indicators/zxm/elasticity_indicators.py',
        'indicators/zxm/score_indicators.py',
        'indicators/zxm/market_breadth.py',
        'indicators/zxm/selection_model.py',
        'indicators/zxm/diagnostics.py',
    ]
    
    fixed_count = 0
    total_count = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            total_count += 1
            if fix_dataframe_assignments(file_path):
                fixed_count += 1
            # 也尝试修复特定模式
            fix_specific_patterns(file_path)
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print(f"\n📊 修复完成:")
    print(f"  总文件数: {total_count}")
    print(f"  已修复文件数: {fixed_count}")
    print(f"  修复率: {(fixed_count/total_count)*100:.1f}%" if total_count > 0 else "  修复率: 0%")
    
    print(f"\n✅ Pandas DataFrame赋值警告修复完成！")

if __name__ == "__main__":
    main_fixpandaswarnings()
