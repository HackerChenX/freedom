#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复单个测试文件
"""

import re

def fix_trend_indicators_file():
    """修复test_trend_indicators.py文件"""
    file_path = "tests/unit/test_trend_indicators.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复所有的setUp方法
    content = re.sub(
        r'def setUp_IndicatorsTesttrendindicators\(self\):',
        'def setUp(self):',
        content
    )
    
    # 修复所有的tearDown方法
    content = re.sub(
        r'def tearDown_IndicatorsTesttrendindicators\(self\):',
        'def tearDown(self):',
        content
    )
    
    # 删除LogCaptureMixin调用
    content = re.sub(
        r'\s*LogCaptureMixin\.setUp_IndicatorsTesttrendindicators\(self\)\s*\n',
        '',
        content
    )
    
    content = re.sub(
        r'\s*LogCaptureMixin\.tearDown_IndicatorsTesttrendindicators\(self\)\s*\n',
        '',
        content
    )
    
    # 在setUp方法开头添加super().setUp()
    content = re.sub(
        r'def setUp\(self\):\s*\n(\s*)(self\.indicator)',
        r'def setUp(self):\n\1super().setUp()\n\1\2',
        content
    )
    
    # 修复指标创建
    indicator_mappings = {
        'MA(': 'complete_registry.create_indicator(\'MA\',',
        'EMA(': 'complete_registry.create_indicator(\'EMA\',',
        'WMA(': 'complete_registry.create_indicator(\'WMA\',',
        'DMI(': 'complete_registry.create_indicator(\'DMI\',',
        'ATR(': 'complete_registry.create_indicator(\'ATR\',',
    }
    
    for old, new in indicator_mappings.items():
        content = content.replace(old, new)
    
    # 修复TestDataGenerator调用
    content = content.replace(
        'TestDataGenerator.generate_price_sequence(',
        'TestDataGenerator.generate_price_sequence_Generator('
    )
    
    # 确保导入complete_registry
    if 'complete_registry.create_indicator' in content:
        if 'from indicators.complete_indicator_registry import complete_registry' not in content:
            # 在import语句后添加导入
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from tests.unit.indicator_test_mixin'):
                    lines.insert(i + 1, 'from indicators.complete_indicator_registry import complete_registry')
                    break
            content = '\n'.join(lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 修复完成")

if __name__ == "__main__":
    fix_trend_indicators_file()
