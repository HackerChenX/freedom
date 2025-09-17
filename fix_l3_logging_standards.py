#!/usr/bin/env python3
"""
L3层日志和异常处理标准修复脚本
批量修复L3层文件以符合L1标准
"""

import os
import re
import sys

def fix_logging_imports(file_path):
    """修复日志导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 替换logging导入
        if 'import logging' in content and 'from utils.logger import get_logger' not in content:
            # 添加标准日志导入
            if 'from utils.logger import get_logger' not in content:
                # 找到合适的位置插入导入
                lines = content.split('\n')
                import_section_end = 0
                for i, line in enumerate(lines):
                    if line.startswith('from ') or line.startswith('import '):
                        import_section_end = i
                
                # 在导入区域末尾添加标准日志导入
                lines.insert(import_section_end + 1, 'from utils.logger import get_logger')
                content = '\n'.join(lines)
        
        # 替换logger初始化
        content = re.sub(r'logger = logging\.getLogger\(__name__\)', 'logger = get_logger(__name__)', content)
        
        # 如果有修改，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 修复日志标准: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复日志标准失败 {file_path}: {e}")
        return False

def fix_decorator_imports(file_path):
    """修复装饰器导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 替换异常处理装饰器导入
        if 'from utils.decorators import exception_handler' in content:
            content = content.replace(
                'from utils.decorators import exception_handler',
                'from utils.enhanced_exception_handler import exception_handler'
            )
        
        if 'from utils.decorators import performance_monitor' in content:
            content = content.replace(
                'from utils.decorators import performance_monitor',
                'from utils.enhanced_performance_monitor import performance_monitor'
            )
        
        # 处理组合导入
        if 'from utils.decorators import exception_handler, performance_monitor' in content:
            content = content.replace(
                'from utils.decorators import exception_handler, performance_monitor',
                'from utils.enhanced_exception_handler import exception_handler\nfrom utils.enhanced_performance_monitor import performance_monitor'
            )
        
        # 如果有修改，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 修复装饰器标准: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复装饰器标准失败 {file_path}: {e}")
        return False

def get_l3_python_files():
    """获取L3层所有Python文件"""
    l3_files = []
    l3_directories = ['db/services', 'db/managers', 'db/interfaces']
    
    for directory in l3_directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        l3_files.append(os.path.join(root, file))
    
    # 添加其他L3相关文件
    additional_files = [
        'db/service_registry.py',
        'db/parallel_processor.py'
    ]
    
    for file_path in additional_files:
        if os.path.exists(file_path):
            l3_files.append(file_path)
    
    return l3_files

def main():
    """主函数"""
    print("🔧 开始批量修复L3层日志和异常处理标准")
    
    l3_files = get_l3_python_files()
    
    logging_fixes = 0
    decorator_fixes = 0
    
    for file_path in l3_files:
        print(f"\n检查文件: {file_path}")
        
        # 修复日志标准
        if fix_logging_imports(file_path):
            logging_fixes += 1
        
        # 修复装饰器标准
        if fix_decorator_imports(file_path):
            decorator_fixes += 1
    
    print(f"\n📊 修复完成:")
    print(f"  日志标准修复: {logging_fixes} 个文件")
    print(f"  装饰器标准修复: {decorator_fixes} 个文件")
    print(f"  总计检查文件: {len(l3_files)} 个")
    
    if logging_fixes > 0 or decorator_fixes > 0:
        print("\n✅ L3层标准修复完成，建议重新运行一致性验证")
    else:
        print("\n✅ 所有文件已符合标准")

if __name__ == "__main__":
    main()
