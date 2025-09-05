#!/usr/bin/env python3
"""
修复技术指标中数据列映射问题的脚本
统一处理不同的列名格式（如close vs Close, volume vs Volume等）
"""

import os
import re
import glob
from typing import List, Tuple

def create_column_mapping_function() -> str:
    """
    创建通用的列名映射函数代码
    
    Returns:
        str: 函数代码字符串
    """
    return '''
def _get_column_name(self, data: pd.DataFrame, column_type: str) -> str:
    """
    获取指定类型的列名，支持多种格式
    
    Args:
        data: 数据Data_frame
        column_type: 列类型 ('open', 'high', 'low', 'close', 'volume')
        
    Returns:
        str: 实际的列名
        
    Raises:
        ValueError: 如果找不到对应的列
    """
    column_mappings = {
        'open': ['open', 'Open', 'OPEN', 'o', 'O'],
        'high': ['high', 'High', 'HIGH', 'h', 'H'],
        'low': ['low', 'Low', 'LOW', 'l', 'L'],
        'close': ['close', 'Close', 'CLOSE', 'c', 'C'],
        'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'v', 'V']
    }
    
    if column_type not in column_mappings:
        raise ValueError(f"不支持的列类型: {column_type}")
    
    possible_names = column_mappings[column_type]
    
    for name in possible_names:
        if name in data.columns:
            return name
    
    raise ValueError(f"无法找到{column_type}列，支持的列名: {possible_names}")
'''

def find_column_access_patterns(file_path: str) -> List[Tuple[int, str, str]]:
    """
    查找文件中直接访问列的模式
    
    Args:
        file_path: 文件路径
        
    Returns:
        List[Tuple[int, str, str]]: (行号, 原始代码, 列名) 的列表
    """
    patterns = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 匹配模式：data['column_name'] 或 df['column_name']
        column_pattern = re.compile(r"(data|df)\[(['\"])([a-zA-Z_]+)\2\]")
        
        for i, line in enumerate(lines, 1):
            matches = column_pattern.findall(line)
            for match in matches:
                var_name, quote, column_name = match
                if column_name.lower() in ['open', 'high', 'low', 'close', 'volume']:
                    patterns.append((i, line.strip(), column_name))
                    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
    
    return patterns

def main_fixcolumnmapping():
    """主函数"""
    print("🔧 开始修复技术指标数据列映射问题...")
    
    # 获取所有指标文件
    indicator_patterns = [
        'indicators/*.py',
        'indicators/zxm/*.py',
        'indicators/enhanced/*.py'
    ]
    
    all_files = []
    for pattern in indicator_patterns:
        all_files.extend(glob.glob(pattern))
    
    # 过滤掉__init__.py和一些特殊文件
    indicator_files = [f for f in all_files if not f.endswith('__init__.py') and os.path.isfile(f)]
    
    print(f"📁 找到 {len(indicator_files)} 个指标文件")
    
    total_issues = 0
    files_with_issues = 0
    
    for file_path in indicator_files:
        patterns = find_column_access_patterns(file_path)
        
        if patterns:
            files_with_issues += 1
            total_issues += len(patterns)
            print(f"\n📄 {file_path}:")
            
            for line_num, code, column_name in patterns:
                print(f"  第{line_num}行: {code}")
                print(f"    -> 建议使用: self._get_column_name(data, '{column_name.lower()}')")
    
    print(f"\n📊 数据列映射问题统计:")
    print(f"  有问题的文件数: {files_with_issues}")
    print(f"  总问题数: {total_issues}")
    
    if total_issues > 0:
        print(f"\n💡 修复建议:")
        print(f"  1. 在指标类中添加 _get_column_name 方法")
        print(f"  2. 将直接列访问替换为 self._get_column_name(data, 'column_type')")
        print(f"  3. 使用返回的列名进行数据访问")
    else:
        print(f"\n✅ 未发现数据列映射问题！")

if __name__ == "__main__":
    main_fixcolumnmapping()
