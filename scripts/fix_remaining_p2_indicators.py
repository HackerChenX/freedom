#!/usr/bin/env python3
"""
快速修复剩余P2语法错误指标脚本
"""

import os
import sys
import re
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 剩余P2语法错误指标列表
REMAINING_P2_INDICATORS = [
    'indicators/bias.py',
    'indicators/dmi.py', 
    'indicators/cmo.py',
    'indicators/dma.py',
    'indicators/vol.py',
    'indicators/pattern/zxm_patterns.py'
]

def fix_syntax_error_fast(file_path):
    """快速修复单个文件的语法错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 修复 expected an indented block after 'if' statement
        # 查找并修复 "if df.empty:# 确保数据包含必要的列" 模式
        content = re.sub(
            r'(\s+)if\s+(df|data)\.empty:\s*#\s*确保数据包含必要的列\s*\n\s*required_columns',
            r'\1if \2.empty:\n\1    return pd.DataFrame()\n\1    \n\1# 确保数据包含必要的列\n\1required_columns',
            content
        )
        
        # 2. 修复其他常见的语法错误模式
        content = re.sub(
            r'(\s+)if\s+(df|data)\.empty:\s*\n(\s+)([^#\n])',
            r'\1if \2.empty:\n\1    return pd.DataFrame()\n\1    \n\3\4',
            content
        )
        
        # 3. 在_calculate方法的return语句前添加形态识别和信号生成
        # 查找存储结果的模式
        storage_patterns = [
            r'(\s+)# 存储结果\s*\n(\s+)self\._result\s*=\s*([a-zA-Z_]+)\s*\n(\s+)def\s+',
            r'(\s+)# 保存结果\s*\n(\s+)self\._result\s*=\s*([a-zA-Z_]+)\s*\n(\s+)def\s+',
            r'(\s+)self\._result\s*=\s*([a-zA-Z_]+)\s*\n(\s+)def\s+'
        ]
        
        for pattern in storage_patterns:
            def add_pattern_signal_before_method(match):
                indent = match.group(1)
                result_var = match.group(3) if len(match.groups()) >= 3 else match.group(2)
                next_method = match.group(4) if len(match.groups()) >= 4 else match.group(3)
                
                # 添加形态识别和信号生成代码
                addition = f"""{indent}# 添加形态识别和信号生成
{indent}{result_var} = self.add_pattern_detection({result_var})
{indent}{result_var} = self.add_signal_generation({result_var})

{indent}# 存储结果
{indent}self._result = {result_var}

{indent}return {result_var}

{indent}def """
                
                return addition + next_method
            
            content = re.sub(pattern, add_pattern_signal_before_method, content)
        
        # 4. 确保PatternSignalMixin正确导入
        if 'PatternSignalMixin' in content and 'from indicators.base.pattern_signal_mixin import PatternSignalMixin' not in content:
            # 在BaseIndicator导入后添加PatternSignalMixin导入
            content = re.sub(
                r'from indicators\.base_indicator import BaseIndicator',
                'from indicators.base_indicator import BaseIndicator\nfrom indicators.base.pattern_signal_mixin import PatternSignalMixin',
                content
            )
        
        # 5. 修复导入错误
        content = re.sub(r', PatternResult', '', content)
        content = re.sub(r', MarketEnvironment', '', content)
        
        # 如果内容有变化，保存文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {e}")
        return False

def test_indicator_fix(indicator_file):
    """测试指标修复效果"""
    try:
        # 从文件路径推断模块名和类名
        if 'pattern/' in indicator_file:
            module_path = indicator_file.replace('/', '.').replace('.py', '')
            class_name = 'ZXMPatterns'  # ZXM_PATTERNS的类名
        else:
            module_path = indicator_file.replace('/', '.').replace('.py', '')
            class_name = Path(indicator_file).stem.upper()
        
        # 动态导入模块
        module = __import__(module_path, fromlist=[''])
        indicator_class = getattr(module, class_name)
        
        # 创建实例
        indicator = indicator_class()
        
        # 测试计算
        import pandas as pd
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'turnover_rate': [0.01, 0.02, 0.015, 0.025, 0.018]
        })
        
        result = indicator.calculate(test_data)
        
        # 检查形态识别和信号生成列
        pattern_columns = [col for col in result.columns if 'pattern' in col.lower()]
        signal_columns = [col for col in result.columns if 'signal' in col.lower()]
        
        return len(pattern_columns) > 0 and len(signal_columns) > 0
        
    except Exception as e:
        return False

def main():
    """主函数"""
    print("=== 快速修复剩余P2语法错误指标 ===")
    print()
    
    success_count = 0
    test_success_count = 0
    
    for i, file_path in enumerate(REMAINING_P2_INDICATORS, 1):
        full_path = project_root / file_path
        if full_path.exists():
            print(f"{i:2d}/{len(REMAINING_P2_INDICATORS)} 修复 {full_path.name}...", end=" ")
            
            if fix_syntax_error_fast(full_path):
                print("✅ 修复", end=" ")
                success_count += 1
                
                # 测试修复效果
                if test_indicator_fix(file_path):
                    print("🎉 测试通过")
                    test_success_count += 1
                else:
                    print("⚠️ 测试失败")
            else:
                print("⚠️ 无需修复")
        else:
            print(f"{i:2d}/{len(REMAINING_P2_INDICATORS)} 修复 {full_path.name}... ❌ 文件不存在")
    
    print()
    print(f"=== P2剩余指标修复完成 ===")
    print(f"文件修复: {success_count}/{len(REMAINING_P2_INDICATORS)}")
    print(f"测试通过: {test_success_count}/{len(REMAINING_P2_INDICATORS)}")
    
    # 测试整体修复效果
    print("\n=== 测试整体修复效果 ===")
    try:
        from indicators.complete_indicator_registry import complete_registry
        
        # 清空注册器
        complete_registry._indicators = {}
        complete_registry._registration_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'failed_indicators': []
        }
        
        # 重新注册
        complete_registry.register_all_indicators()
        
        # 获取注册结果
        total_indicators = len(complete_registry.get_indicator_names())
        print(f"修复后成功注册指标数: {total_indicators}个")
        
        if total_indicators > 52:
            print(f"✅ P2剩余指标修复成功！新增注册 {total_indicators - 52} 个指标")
        else:
            print("⚠️ 注册数量未显著增加，可能需要进一步修复")
            
    except Exception as e:
        print(f"❌ 测试修复效果时出错: {e}")

if __name__ == "__main__":
    main()
