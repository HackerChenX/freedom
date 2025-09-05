#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复导入顺序问题

确保所有指标文件都有正确的 get_logger 导入
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_import_order_in_file(file_path: Path) -> bool:
    """
    修复单个文件的导入顺序
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否修复成功
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # 检查是否需要修复
        has_get_logger_call = 'get_logger(' in content
        has_correct_import = 'from utils.dependency_injection import get_logger' in content
        
        if not has_get_logger_call:
            return True  # 不需要修复
        
        if has_correct_import:
            # 检查导入是否在正确位置
            import_line_idx = -1
            docstring_end = -1
            
            in_docstring = False
            docstring_quotes = None
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # 检测文档字符串
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = True
                        docstring_quotes = stripped[:3]
                        if stripped.count(docstring_quotes) >= 2:
                            # 单行文档字符串
                            in_docstring = False
                            docstring_end = i
                elif in_docstring:
                    if docstring_quotes in stripped:
                        in_docstring = False
                        docstring_end = i
                
                # 查找导入行
                if 'from utils.dependency_injection import get_logger' in line:
                    import_line_idx = i
                    break
            
            # 如果导入在文档字符串内，需要移动
            if import_line_idx != -1 and import_line_idx <= docstring_end:
                # 移除错误位置的导入
                lines.pop(import_line_idx)
                
                # 在文档字符串后添加导入
                insert_pos = docstring_end  # 调整索引
                
                # 找到合适的插入位置（在其他导入之前）
                for i in range(insert_pos + 1, len(lines)):
                    if lines[i].strip().startswith('from ') or lines[i].strip().startswith('import '):
                        insert_pos = i
                        break
                    elif lines[i].strip() and not lines[i].strip().startswith('#'):
                        insert_pos = i
                        break
                
                lines.insert(insert_pos, 'from utils.dependency_injection import get_logger')
                
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                return True
        else:
            # 添加缺失的导入
            insert_pos = 0
            
            # 跳过文档字符串
            in_docstring = False
            docstring_quotes = None
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = True
                        docstring_quotes = stripped[:3]
                        if stripped.count(docstring_quotes) >= 2:
                            in_docstring = False
                            insert_pos = i + 1
                elif in_docstring:
                    if docstring_quotes in stripped:
                        in_docstring = False
                        insert_pos = i + 1
                        break
            
            # 在合适位置插入导入
            if insert_pos < len(lines):
                # 找到第一个导入语句的位置
                for i in range(insert_pos, len(lines)):
                    if lines[i].strip().startswith('from ') or lines[i].strip().startswith('import '):
                        insert_pos = i
                        break
                    elif lines[i].strip() and not lines[i].strip().startswith('#'):
                        insert_pos = i
                        break
            
            lines.insert(insert_pos, 'from utils.dependency_injection import get_logger')
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            return True
        
        return True
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def fix_all_import_orders():
    """修复所有文件的导入顺序"""
    print("🔧 修复所有指标文件的导入顺序...")
    
    indicators_dir = project_root / "indicators"
    
    if not indicators_dir.exists():
        print("❌ indicators目录不存在")
        return 0
    
    fixed_count = 0
    total_count = 0
    
    for py_file in indicators_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        total_count += 1
        
        if fix_import_order_in_file(py_file):
            fixed_count += 1
            print(f"  ✓ {py_file.relative_to(project_root)}")
        else:
            print(f"  ✗ {py_file.relative_to(project_root)}")
    
    print(f"✅ 处理了 {fixed_count}/{total_count} 个文件")
    return fixed_count

def fix_reverse_validation_framework():
    """修复反向验证框架的导入问题"""
    print("🔧 修复反向验证框架...")
    
    framework_file = project_root / "tests/reverse_validation/reverse_validation_framework.py"
    
    if not framework_file.exists():
        print("❌ 反向验证框架文件不存在")
        return False
    
    try:
        with open(framework_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有导入问题
        if 'get_logger(' in content and 'from utils.dependency_injection import get_logger' not in content:
            # 在导入部分添加
            lines = content.split('\n')
            
            # 找到合适的插入位置
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('from utils.dependency_injection import'):
                    insert_pos = i + 1
                    break
                elif line.strip().startswith('from ') or line.strip().startswith('import '):
                    continue
                elif line.strip() and not line.strip().startswith('#'):
                    insert_pos = i
                    break
            
            # 检查是否已经有导入
            if 'from utils.dependency_injection import get_logger' not in content:
                lines.insert(insert_pos, 'from utils.dependency_injection import get_logger')
                
                with open(framework_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                print("  ✓ 添加了 get_logger 导入")
        
        print("  ✅ 反向验证框架修复完成")
        return True
        
    except Exception as e:
        print(f"  ❌ 修复反向验证框架失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("修复导入顺序问题")
    print("=" * 80)
    
    # 修复指标文件
    fixed_indicators = fix_all_import_orders()
    
    # 修复反向验证框架
    framework_fixed = fix_reverse_validation_framework()
    
    print("\n" + "=" * 80)
    print("修复完成总结:")
    print(f"修复指标文件数: {fixed_indicators}")
    print(f"反向验证框架: {'✅ 修复成功' if framework_fixed else '❌ 修复失败'}")
    print("=" * 80)
    
    return 0 if (fixed_indicators > 0 and framework_fixed) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
