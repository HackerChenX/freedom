#!/usr/bin/env python3
"""
批量修复日志导入语句
将 from utils.logger import get_logger 
替换为 from utils.logger import get_logger
"""

import os
import re
import sys
from pathlib import Path

def fix_logger_imports(file_path: str) -> bool:
    """
    修复单个文件的日志导入
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否修改了文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 替换导入语句
        patterns = [
            (r'from utils\.dependency_injection import get_logger', 'from utils.logger import get_logger'),
            (r'from utils\.dependency_injection import get_logger,', 'from utils.logger import get_logger,'),
            (r'from utils\.dependency_injection import ([^,\n]*,\s*)?get_logger', r'from utils.logger import get_logger\nfrom utils.dependency_injection import \1'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # 清理多余的空导入
        content = re.sub(r'from utils\.dependency_injection import\s*\n', '', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"修复文件失败 {file_path}: {e}")
        return False

def find_python_files() -> list:
    """查找所有需要修复的Python文件"""
    files = []
    
    for root, dirs, filenames in os.walk('.'):
        # 跳过不需要的目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'backup', 'archive', 'venv', '.venv']]
        
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(root, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'from utils.logger import get_logger' in content:
                            files.append(file_path)
                except:
                    continue
    
    return files

def main():
    """主函数"""
    print("🚀 开始批量修复日志导入语句...")
    
    # 查找需要修复的文件
    files_to_fix = find_python_files()
    print(f"📊 找到 {len(files_to_fix)} 个需要修复的文件")
    
    if not files_to_fix:
        print("✅ 没有需要修复的文件")
        return
    
    # 批量修复
    fixed_count = 0
    failed_count = 0
    
    for file_path in files_to_fix:
        try:
            if fix_logger_imports(file_path):
                fixed_count += 1
                print(f"✅ 修复: {file_path}")
            else:
                print(f"⚠️  跳过: {file_path}")
        except Exception as e:
            failed_count += 1
            print(f"❌ 失败: {file_path} - {e}")
    
    print(f"\n📊 修复结果:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")
    print(f"⚠️  跳过文件: {len(files_to_fix) - fixed_count - failed_count} 个文件")
    
    if fixed_count > 0:
        print("🎉 日志导入语句批量修复完成！")
    else:
        print("🚫 没有文件被修复")

if __name__ == "__main__":
    main()
