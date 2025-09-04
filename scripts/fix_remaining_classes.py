#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复test_trend_indicators.py中剩余的类
"""

import re

def fix_remaining_classes():
    """修复剩余的类"""
    file_path = "tests/unit/test_trend_indicators.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义需要修复的类和对应的指标名称
    class_fixes = [
        {
            'class_name': 'TestEMA_Indicators',
            'indicator_name': 'EMA',
            'old_indicator': 'EMA(periods=[12, 26])',
            'new_indicator': "complete_registry.create_indicator('EMA', periods=[12, 26])",
            'expected_columns': "['EMA12', 'EMA26']"
        },
        {
            'class_name': 'TestWMA_Indicators', 
            'indicator_name': 'WMA',
            'old_indicator': 'WMA(periods=[5, 10])',
            'new_indicator': "complete_registry.create_indicator('WMA', periods=[5, 10])",
            'expected_columns': "['WMA5', 'WMA10']"
        },
        {
            'class_name': 'TestDMI',
            'indicator_name': 'DMI', 
            'old_indicator': 'DMI(period=14)',
            'new_indicator': "complete_registry.create_indicator('DMI', period=14)",
            'expected_columns': "['dmi_plus', 'dmi_minus', 'adx']"
        },
        {
            'class_name': 'TestATR',
            'indicator_name': 'ATR',
            'old_indicator': 'ATR(period=14)',
            'new_indicator': "complete_registry.create_indicator('ATR', period=14)",
            'expected_columns': "['atr']"
        }
    ]
    
    for fix in class_fixes:
        # 构建要替换的模式
        old_pattern = f"""class {fix['class_name']}(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp_IndicatorsTesttrendindicators(self):
        LogCaptureMixin.setUp_IndicatorsTesttrendindicators(self)
        self.indicator = {fix['old_indicator']}
        self.data = TestDataGenerator.generate_price_sequence([
            {{'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}}
        ])
        self.expected_columns = {fix['expected_columns']}

    def tearDown_IndicatorsTesttrendindicators(self):
        LogCaptureMixin.tearDown_IndicatorsTesttrendindicators(self)"""
        
        new_pattern = f"""class {fix['class_name']}(unittest.TestCase, IndicatorTestMixin, LogCaptureMixin):
    def setUp(self):
        super().setUp()
        self.indicator = {fix['new_indicator']}
        self.data = TestDataGenerator.generate_price_sequence_Generator([
            {{'type': 'trend', 'start_price': 100, 'end_price': 110, 'periods': 50}}
        ])
        self.expected_columns = {fix['expected_columns']}"""
        
        content = content.replace(old_pattern, new_pattern)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 修复完成")

if __name__ == "__main__":
    fix_remaining_classes()
