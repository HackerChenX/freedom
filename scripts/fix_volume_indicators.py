#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复test_volume_related_indicators.py中剩余的类
"""

def fix_volume_indicators():
    """修复剩余的类"""
    file_path = "tests/unit/test_volume_related_indicators.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复所有的setUp方法名
    content = content.replace(
        'def setUp_IndicatorsTestvolumerelatedindicators(self):',
        'def setUp(self):'
    )
    
    # 删除LogCaptureMixin调用
    content = content.replace(
        '        LogCaptureMixin.setUp_IndicatorsTestvolumerelatedindicators(self)\n',
        '        super().setUp()\n'
    )
    
    content = content.replace(
        '    def tearDown_IndicatorsTestvolumerelatedindicators(self):\n        LogCaptureMixin.tearDown_IndicatorsTestvolumerelatedindicators(self)\n',
        ''
    )
    
    # 修复TestDataGenerator调用
    content = content.replace(
        'TestDataGenerator.generate_price_sequence(',
        'TestDataGenerator.generate_price_sequence_Generator('
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 修复完成")

if __name__ == "__main__":
    fix_volume_indicators()
