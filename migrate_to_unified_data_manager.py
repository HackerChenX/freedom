#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
数据管理器迁移脚本

将系统中所有使用data_manager的模块迁移到统一的unified_data_manager
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def find_python_files(root_dir: str) -> List[str]:
    """查找所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过一些目录
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def analyze_data_manager_imports(file_path: str) -> List[Dict[str, str]]:
    """分析文件中的data_manager导入"""
    imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找各种导入模式
        import_patterns = [
            # from db.unified_data_manager import get_unified_data_manager
            (r'from\s+db\.data_manager\s+import\s+(\w+)', 'db.data_manager'),
            # from db.unified_data_manager import get_unified_data_manager
            (r'from\s+db\.enhanced_data_manager\s+import\s+(\w+)', 'db.enhanced_data_manager'),
            # from db.unified_data_manager import get_unified_data_manager
            (r'from\s+db\.data_manager_adapter\s+import\s+(\w+)', 'db.data_manager_adapter'),
            # import db.data_manager
            (r'import\s+(db\.data_manager)', 'db.data_manager'),
            (r'import\s+(db\.enhanced_data_manager)', 'db.enhanced_data_manager'),
            (r'import\s+(db\.data_manager_adapter)', 'db.data_manager_adapter'),
        ]
        
        for pattern, module in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                imports.append({
                    'file': file_path,
                    'module': module,
                    'imported_name': match,
                    'pattern': pattern
                })
    
    except Exception as e:
        print(f"分析文件 {file_path} 时出错: {e}")
    
    return imports

def create_migration_plan(imports: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """创建迁移计划"""
    migration_plan = {}
    
    for imp in imports:
        file_path = imp['file']
        if file_path not in migration_plan:
            migration_plan[file_path] = []
        
        # 根据导入的内容确定替换策略
        module = imp['module']
        imported_name = imp['imported_name']
        
        if module == 'db.data_manager':
            if imported_name == 'DataManager':
                migration_plan[file_path].append({
                    'old': f"from db.unified_data_manager import get_unified_data_manager",
                    'new': f"from db.unified_data_manager import get_unified_data_manager"
                })
                migration_plan[file_path].append({
                    'old': f"get_unified_data_manager()",
                    'new': f"get_unified_data_manager()"
                })
        
        elif module == 'db.enhanced_data_manager':
            if imported_name == 'get_enhanced_data_manager':
                migration_plan[file_path].append({
                    'old': f"from db.unified_data_manager import get_unified_data_manager",
                    'new': f"from db.unified_data_manager import get_unified_data_manager"
                })
                migration_plan[file_path].append({
                    'old': f"get_unified_data_manager()",
                    'new': f"get_unified_data_manager()"
                })
        
        elif module == 'db.data_manager_adapter':
            if imported_name == 'get_data_manager_adapter':
                migration_plan[file_path].append({
                    'old': f"from db.unified_data_manager import get_unified_data_manager",
                    'new': f"from db.unified_data_manager import get_unified_data_manager"
                })
                migration_plan[file_path].append({
                    'old': f"get_unified_data_manager()",
                    'new': f"get_unified_data_manager()"
                })
    
    return migration_plan

def apply_migration(file_path: str, replacements: List[Dict[str, str]]) -> bool:
    """应用迁移到指定文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 应用所有替换
        for replacement in replacements:
            old = replacement['old']
            new = replacement['new']
            content = content.replace(old, new)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已更新: {file_path}")
            return True
        else:
            print(f"⏭️  无需更新: {file_path}")
            return False
    
    except Exception as e:
        print(f"❌ 更新文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("🔄 开始数据管理器迁移...")
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找所有Python文件
    print("📁 扫描Python文件...")
    python_files = find_python_files(root_dir)
    print(f"找到 {len(python_files)} 个Python文件")
    
    # 分析导入
    print("🔍 分析data_manager导入...")
    all_imports = []
    for file_path in python_files:
        imports = analyze_data_manager_imports(file_path)
        all_imports.extend(imports)
    
    print(f"找到 {len(all_imports)} 个data_manager导入")
    
    # 按文件分组显示
    files_with_imports = {}
    for imp in all_imports:
        file_path = imp['file']
        if file_path not in files_with_imports:
            files_with_imports[file_path] = []
        files_with_imports[file_path].append(imp)
    
    print(f"涉及 {len(files_with_imports)} 个文件:")
    for file_path, imports in files_with_imports.items():
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"  📄 {rel_path}: {len(imports)} 个导入")
    
    # 创建迁移计划
    print("\n📋 创建迁移计划...")
    migration_plan = create_migration_plan(all_imports)
    
    # 显示迁移计划
    print("迁移计划:")
    for file_path, replacements in migration_plan.items():
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"  📄 {rel_path}:")
        for replacement in replacements:
            print(f"    🔄 '{replacement['old']}' → '{replacement['new']}'")
    
    # 确认执行
    response = input("\n是否执行迁移? (y/N): ")
    if response.lower() != 'y':
        print("❌ 迁移已取消")
        return
    
    # 执行迁移
    print("\n🚀 执行迁移...")
    updated_files = 0
    for file_path, replacements in migration_plan.items():
        if apply_migration(file_path, replacements):
            updated_files += 1
    
    print(f"\n✅ 迁移完成! 更新了 {updated_files} 个文件")
    
    # 创建备份旧文件的建议
    print("\n📝 迁移后建议:")
    print("1. 运行测试确保所有功能正常")
    print("2. 检查导入是否正确")
    print("3. 验证603359股票的选股测试")
    print("4. 如果一切正常，可以删除旧的data_manager文件:")
    print("   - db/data_manager.py")
    print("   - db/enhanced_data_manager.py") 
    print("   - db/data_manager_adapter.py")

if __name__ == "__main__":
    main()
