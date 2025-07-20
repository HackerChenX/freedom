#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量修复业务逻辑测试

系统性地修复测试文件中的业务逻辑问题
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 指标名称映射表 - 将测试文件中的类名映射到注册表中的指标名
INDICATOR_MAPPINGS = {
    'Chaikin': 'CHAIKIN',
    'DMI': 'DMI', 
    'Enhanced_dMI': 'ENHANCED_DMI',
    'Enhanced_kDJ': 'ENHANCED_KDJ',
    'Enhanced_mACD': 'ENHANCED_MACD',
    'Enhanced_mFI': 'ENHANCED_MFI',
    'Enhanced_oBV': 'ENHANCED_OBV',
    'Enhanced_tRIX': 'ENHANCED_TRIX',
    'Fibonacci_tools': 'FIBONACCI_TOOLS',
    'Gann_tools': 'GANN_TOOLS',
    'Ichimoku': 'ICHIMOKU',
    'KC': 'KC',
    'MTM': 'MTM',
    'PSY': 'PSY',
    'TRIX': 'TRIX',
    'UnifiedMA': 'UNIFIED_MA',
    'VOL': 'VOL',
    'Volume_ratio': 'VOLUME_RATIO',
    'VORTEX': 'VORTEX',
    'VOSC': 'VOSC',
    'VR': 'VR',
    'WR': 'WR',
    'ZXMPattern_indicator': 'ZXM_PATTERN',
    'ZXMAbsorb': 'ZXM_ABSORB',
    'Advanced_candlestick_patterns': 'ADVANCED_CANDLESTICK',
    'Pattern_combination': 'PATTERN_COMBINATION',
    'Pattern_confirmation': 'PATTERN_CONFIRMATION',
    'Pattern_quality_evaluator': 'PATTERN_QUALITY_EVALUATOR',
}

def fix_business_logic_in_file(file_path):
    """修复单个文件中的业务逻辑问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 修复导入语句
        content = re.sub(
            r'from tests\.unit\.indicator_test_mixin import Indicator_test_mixin',
            'from tests.unit.indicator_test_mixin import IndicatorTestMixin',
            content
        )
        
        content = re.sub(
            r'from tests\.helper\.data_generator import Test_data_generator',
            'from tests.helper.data_generator import TestDataGenerator',
            content
        )
        
        content = re.sub(
            r'from tests\.helper\.log_capture import Log_capture_mixin, LogCaptureMixin',
            'from tests.helper.log_capture import LogCaptureMixin',
            content
        )
        
        content = re.sub(
            r'from tests\.helper\.log_capture import Log_capture_mixin',
            'from tests.helper.log_capture import LogCaptureMixin',
            content
        )
        
        # 2. 修复类继承
        content = re.sub(
            r'unittest\.TestCase, Indicator_test_mixin, Log_capture_mixin',
            'unittest.TestCase, IndicatorTestMixin, LogCaptureMixin',
            content
        )
        
        content = re.sub(
            r'unittest\.TestCase, Indicator_test_mixin',
            'unittest.TestCase, IndicatorTestMixin',
            content
        )
        
        # 3. 修复setUp方法调用
        content = re.sub(
            r'Log_capture_mixin\.setUp\(self\)',
            'super().setUp()',
            content
        )
        
        # 4. 删除tearDown方法
        content = re.sub(
            r'\s*def tearDown\(self\):\s*\n\s*""".*?"""\s*\n\s*Log_capture_mixin\.tearDown\(self\)\s*\n',
            '',
            content,
            flags=re.DOTALL
        )
        
        # 5. 修复数据生成器调用
        content = re.sub(
            r'Test_data_generator\.generate_price_sequence',
            'TestDataGenerator.generate_price_sequence',
            content
        )
        
        # 6. 修复指标实例化 - 使用complete_registry
        for class_name, indicator_name in INDICATOR_MAPPINGS.items():
            # 匹配 self.indicator = ClassName(...) 的模式
            pattern = rf'self\.indicator = {re.escape(class_name)}\((.*?)\)'
            replacement = f"self.indicator = complete_registry.create_indicator('{indicator_name}', \\1)"
            content = re.sub(pattern, replacement, content)
        
        # 7. 确保导入complete_registry
        if 'complete_registry.create_indicator' in content:
            if 'from indicators.complete_indicator_registry import complete_registry' not in content:
                # 在import语句后添加导入
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('import unittest'):
                        lines.insert(i + 1, 'from indicators.complete_indicator_registry import complete_registry')
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

def find_files_with_business_logic_issues():
    """查找所有有业务逻辑问题的测试文件"""
    test_files = []
    tests_dir = Path(root_dir) / "tests" / "unit"
    
    # 重点关注的测试文件模式
    target_patterns = [
        'test_*candlestick*.py',
        'test_*pattern*.py', 
        'test_chaikin.py',
        'test_dmi.py',
        'test_enhanced_*.py',
        'test_fibonacci*.py',
        'test_gann*.py',
        'test_ichimoku.py',
        'test_kc.py',
        'test_mtm.py',
        'test_psy.py',
        'test_trix.py',
        'test_unified*.py',
        'test_vol*.py',
        'test_vr.py',
        'test_wr.py',
        'test_zxm*.py',
        'test_composite*.py',
        'test_institutional*.py',
        'test_chip*.py',
        'test_stock*.py',
    ]
    
    for pattern in target_patterns:
        for file_path in tests_dir.glob(pattern):
            if file_path.name != "__init__.py":
                test_files.append(str(file_path))
    
    return list(set(test_files))  # 去重

def main():
    """主函数"""
    print("🚀 开始批量修复业务逻辑测试问题...")
    
    # 查找需要修复的文件
    test_files = find_files_with_business_logic_issues()
    print(f"📊 找到 {len(test_files)} 个需要修复的文件")
    
    if not test_files:
        print("✅ 没有找到需要修复的文件")
        return
    
    # 修复每个文件
    fixed_count = 0
    failed_count = 0
    
    for file_path in test_files:
        print(f"\n🔧 修复文件: {file_path}")
        if fix_business_logic_in_file(file_path):
            print(f"✅ 修复成功")
            fixed_count += 1
        else:
            print(f"❌ 修复失败或无需修复")
            failed_count += 1
    
    print(f"\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")
    
    if fixed_count > 0:
        print("\n🔍 建议运行以下命令验证修复效果:")
        print("python3 -m pytest tests/unit/ -k \"test_calculation_runs_without_error_Mixin or test_returns_dataframe_Mixin\" --tb=line -q")

if __name__ == "__main__":
    main()
