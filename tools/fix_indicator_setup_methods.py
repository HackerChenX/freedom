#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量修复测试文件中的indicator属性初始化问题

为缺少indicator属性的测试类补充正确的setUp方法初始化
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

def fix_indicator_setup_in_file(file_path):
    """修复单个文件中的indicator setUp问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 查找需要修复的模式
        # 1. 修复非标准的setUp方法名
        setup_patterns = [
            (r'def setUp_\w+\(self\):', 'def setUp(self):'),
        ]
        
        for pattern, replacement in setup_patterns:
            content = re.sub(pattern, replacement, content)
        
        # 2. 确保setUp方法调用super().setUp()
        if 'def setUp(self):' in content and 'super().setUp()' not in content:
            content = re.sub(
                r'def setUp\(self\):\s*\n',
                'def setUp(self):\n        super().setUp()\n',
                content
            )
        
        # 3. 修复TestDataGenerator方法调用
        content = content.replace(
            'TestDataGenerator.generate_price_sequence(',
            'TestDataGenerator.generate_price_sequence_Generator('
        )
        
        # 4. 修复指标创建方式 - 使用complete_registry
        # 查找直接实例化指标的模式，如 self.indicator = SomeIndicator(...)
        indicator_patterns = [
            # 匹配 self.indicator = IndicatorName(...) 的模式
            (r'self\.indicator = (\w+)\((.*?)\)', 
             lambda m: f"self.indicator = complete_registry.create_indicator('{m.group(1).upper()}', {m.group(2)})"),
        ]
        
        for pattern, replacement_func in indicator_patterns:
            matches = re.finditer(pattern, content)
            for match in reversed(list(matches)):  # 从后往前替换，避免位置偏移
                if callable(replacement_func):
                    replacement = replacement_func(match)
                else:
                    replacement = replacement_func
                content = content[:match.start()] + replacement + content[match.end():]
        
        # 5. 确保导入complete_registry
        if 'complete_registry.create_indicator' in content:
            if 'from indicators.complete_indicator_registry import complete_registry' not in content:
                # 在import语句后添加导入
                lines = content.split('\n')
                import_added = False
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        # 在最后一个import语句后添加
                        if i + 1 < len(lines) and not (lines[i + 1].startswith('import ') or lines[i + 1].startswith('from ')):
                            lines.insert(i + 1, 'from indicators.complete_indicator_registry import complete_registry')
                            import_added = True
                            break
                
                if not import_added:
                    # 如果没有找到合适的位置，在文件开头添加
                    for i, line in enumerate(lines):
                        if not line.startswith('#') and line.strip():
                            lines.insert(i, 'from indicators.complete_indicator_registry import complete_registry')
                            break
                
                content = '\n'.join(lines)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def find_files_with_indicator_setup_issues():
    """查找所有有indicator setUp问题的测试文件"""
    test_files = []
    tests_dir = Path(root_dir) / "tests" / "unit"
    
    for file_path in tests_dir.rglob("*.py"):
        if file_path.name.startswith("test_") and file_path.name != "__init__.py":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有测试类和可能的setUp问题
                has_test_class = 'class Test' in content and 'IndicatorTestMixin' in content
                has_setup_issue = (
                    'def setUp_' in content or  # 非标准setUp方法名
                    ('self.indicator =' in content and 'complete_registry.create_indicator' not in content)  # 直接实例化指标
                )
                
                if has_test_class and has_setup_issue:
                    test_files.append(str(file_path))
                    
            except Exception as e:
                print(f"❌ 无法检查文件 {file_path}: {e}")
    
    return test_files

def main():
    """主函数"""
    print("🚀 开始批量修复indicator setUp方法问题...")
    
    # 查找需要修复的文件
    test_files = find_files_with_indicator_setup_issues()
    print(f"📊 找到 {len(test_files)} 个需要修复的文件")
    
    if not test_files:
        print("✅ 没有找到需要修复的文件")
        return
    
    # 修复每个文件
    fixed_count = 0
    failed_count = 0
    
    for file_path in test_files:
        print(f"\n🔧 修复文件: {file_path}")
        if fix_indicator_setup_in_file(file_path):
            print(f"✅ 修复成功")
            fixed_count += 1
        else:
            print(f"❌ 修复失败")
            failed_count += 1
    
    print(f"\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")
    
    if fixed_count > 0:
        print("\n🔍 建议运行以下命令验证修复效果:")
        print("python3 -m pytest tests/unit/ -k \"test_calculation_runs_without_error_Mixin\" --tb=line -q")

if __name__ == "__main__":
    main()
