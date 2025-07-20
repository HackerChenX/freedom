#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量修复测试文件中的setUp方法问题

将非标准的setUp方法名（如set_up_XXX）修改为标准的setUp方法
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

def fix_setup_method_in_file(file_path):
    """修复单个文件中的setUp方法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 查找非标准的setUp方法模式
        setup_pattern = re.compile(r'def (set_up_\w+)\(self\):')
        teardown_pattern = re.compile(r'def (tear_down_\w+)\(self\):')
        
        # 查找所有匹配的方法
        setup_matches = setup_pattern.findall(content)
        teardown_matches = teardown_pattern.findall(content)
        
        if not setup_matches:
            return False, "没有找到需要修复的setUp方法"
        
        # 修复setUp方法
        for old_method_name in setup_matches:
            # 替换方法定义
            content = content.replace(
                f'def {old_method_name}(self):',
                'def setUp(self):'
            )
            
            # 替换方法调用（如果有的话）
            content = content.replace(
                f'{old_method_name}(self)',
                'setUp(self)'
            )
            
            # 替换super()调用
            content = content.replace(
                f'super().{old_method_name}()',
                'super().setUp()'
            )
        
        # 修复tearDown方法
        for old_method_name in teardown_matches:
            # 替换方法定义
            content = content.replace(
                f'def {old_method_name}(self):',
                'def tearDown(self):'
            )
            
            # 替换方法调用（如果有的话）
            content = content.replace(
                f'{old_method_name}(self)',
                'tearDown(self)'
            )
            
            # 替换super()调用
            content = content.replace(
                f'super().{old_method_name}()',
                'super().tearDown()'
            )
        
        # 修复Log_capture_mixin的调用
        # 将 Log_capture_mixin.set_up_XXX(self) 改为 Log_capture_mixin.setUp(self)
        log_capture_pattern = re.compile(r'Log_capture_mixin\.set_up_\w+\(self\)')
        content = log_capture_pattern.sub('Log_capture_mixin.setUp(self)', content)
        
        log_capture_teardown_pattern = re.compile(r'Log_capture_mixin\.tear_down_\w+\(self\)')
        content = log_capture_teardown_pattern.sub('Log_capture_mixin.tearDown(self)', content)
        
        # 修复Test_data_generator调用
        content = content.replace(
            'Test_data_generator.generate_price_sequence(',
            'Test_data_generator.generate_price_sequence_Generator('
        )
        
        if content != original_content:
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, f"修复了setUp方法: {', '.join(setup_matches)}"
        else:
            return False, "没有需要修复的内容"
            
    except Exception as e:
        return False, f"修复失败: {e}"

def find_test_files_with_setup_issues():
    """查找所有有setUp方法问题的测试文件"""
    test_files = []
    tests_dir = Path(root_dir) / "tests"
    
    for file_path in tests_dir.rglob("*.py"):
        if file_path.name.startswith("test_") or "test" in file_path.name:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有测试类和非标准setUp方法
                has_test_class = 'class Test' in content
                has_nonstandard_setup = re.search(r'def set_up_\w+\(self\):', content)
                
                if has_test_class and has_nonstandard_setup:
                    test_files.append(str(file_path))
                    
            except Exception as e:
                print(f"❌ 无法检查文件 {file_path}: {e}")
    
    return test_files

def main():
    """主函数"""
    print("🚀 开始批量修复setUp方法问题...")
    
    # 查找需要修复的文件
    test_files = find_test_files_with_setup_issues()
    print(f"📊 找到 {len(test_files)} 个需要修复的文件")
    
    if not test_files:
        print("✅ 没有找到需要修复的文件")
        return
    
    # 修复每个文件
    fixed_count = 0
    failed_count = 0
    
    for file_path in test_files:
        print(f"\n🔧 修复文件: {file_path}")
        success, message = fix_setup_method_in_file(file_path)
        
        if success:
            print(f"✅ {message}")
            fixed_count += 1
        else:
            print(f"❌ {message}")
            failed_count += 1
    
    print(f"\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")
    
    if fixed_count > 0:
        print("\n🔍 建议运行以下命令验证修复效果:")
        print("python3 tools/test_framework_fixer.py")

if __name__ == "__main__":
    main()
