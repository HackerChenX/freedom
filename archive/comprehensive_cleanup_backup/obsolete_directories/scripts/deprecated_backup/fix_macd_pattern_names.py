#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复MACD指标形态名称

将MACD指标的形态名称修改为使用统一注册表的规范名称
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry

logger = logging.getLogger(__name__)

def fix_macd_pattern_names():
    """修复MACD指标形态名称"""
    
    print("🔧 开始修复MACD指标形态名称")
    print("=" * 60)
    
    # 获取统一注册表
    registry = get_unified_pattern_registry()
    
    # 查看MACD当前支持的规范形态
    macd_patterns = registry.get_indicator_patterns('MACD')
    print(f"📋 MACD支持的规范形态: {macd_patterns}")
    
    # 读取当前MACD指标文件
    macd_file = '/Users/hacker/PycharmProjects/freedom/indicators/macd.py'
    
    if not os.path.exists(macd_file):
        print(f"❌ MACD文件不存在: {macd_file}")
        return False
    
    with open(macd_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📖 读取MACD指标文件成功")
    
    # 定义形态名称映射
    pattern_mappings = {
        'MACD_GOLDEN_CROSS': 'GOLDEN_CROSS',
        'MACD_DEATH_CROSS': 'DEATH_CROSS',
        'MACD_HISTOGRAM_DIVERGENCE': 'BEARISH_DIVERGENCE',
        'MACD_ABOVE_ZERO_GOLDEN': 'MACD_ABOVE_ZERO_GOLDEN'  # 保持不变，这是MACD特有形态
    }
    
    print("🔄 开始替换形态名称...")
    
    # 执行替换
    modified_content = content
    changes_made = 0
    
    for old_name, new_name in pattern_mappings.items():
        if old_name in modified_content:
            # 替换字符串字面量
            modified_content = modified_content.replace(f"'{old_name}'", f"'{new_name}'")
            modified_content = modified_content.replace(f'"{old_name}"', f'"{new_name}"')
            
            # 替换字典键
            modified_content = modified_content.replace(f"['{old_name}']", f"['{new_name}']")
            modified_content = modified_content.replace(f'["{old_name}"]', f'["{new_name}"]')
            
            changes_made += 1
            print(f"  ✅ {old_name} → {new_name}")
    
    if changes_made == 0:
        print("ℹ️  没有发现需要修改的形态名称")
        return True
    
    # 备份原文件
    backup_file = f"{macd_file}.backup_{int(os.path.getmtime(macd_file))}"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 原文件已备份: {backup_file}")
    
    # 写入修改后的内容
    with open(macd_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ MACD指标文件修复完成，共修改 {changes_made} 个形态名称")
    
    # 验证修复结果
    print("\n🔍 验证修复结果...")
    try:
        # 重新导入MACD模块
        import importlib
        if 'indicators.macd' in sys.modules:
            importlib.reload(sys.modules['indicators.macd'])
        
        from indicators.macd import MacdMacd
        from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator
        
        # 测试修复后的MACD
        macd = MacdMacd()
        generator = StockInfoCompatibleDataGenerator()
        
        test_data = generator.generate_stockinfo_compatible_data(
            indicator_name='MACD',
            pattern_type='GOLDEN_CROSS',
            stock_code='TEST_FIX_VERIFICATION',
            history_days=60
        )
        
        if test_data is not None:
            patterns_result = macd.get_patterns(test_data)
            if patterns_result is not None:
                actual_patterns = list(patterns_result.columns)
                print(f"📊 修复后的MACD形态: {actual_patterns}")
                
                # 检查是否包含规范形态名称
                expected_patterns = ['GOLDEN_CROSS', 'DEATH_CROSS', 'BEARISH_DIVERGENCE']
                found_patterns = [p for p in expected_patterns if p in actual_patterns]
                
                if found_patterns:
                    print(f"✅ 发现规范形态: {found_patterns}")
                    return True
                else:
                    print(f"⚠️  未发现预期的规范形态")
                    return False
            else:
                print("❌ get_patterns返回None")
                return False
        else:
            print("❌ 测试数据生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        return False

def main():
    """主函数"""
    success = fix_macd_pattern_names()
    
    if success:
        print("\n🎉 MACD指标形态名称修复成功！")
        print("现在可以重新运行双向验证测试")
    else:
        print("\n❌ MACD指标形态名称修复失败")
        print("请检查错误信息并手动修复")

if __name__ == "__main__":
    main()
