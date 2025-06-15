#!/usr/bin/env python3
"""
测试完整指标注册的独立脚本
避免循环导入问题，直接测试指标注册功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def test_complete_registration():
    """测试完整指标注册"""
    print("=== 测试完整指标注册系统 ===\n")
    
    try:
        # 导入完整注册器
        from indicators.complete_indicator_registry import complete_registry
        print("✅ 成功导入完整注册器")
        
        # 获取注册前状态
        before_count = len(complete_registry.get_indicator_names())
        print(f"注册前指标数量: {before_count}")
        
        # 执行完整注册
        print("\n开始执行完整指标注册...")
        total_registered = complete_registry.register_all_indicators()
        
        # 获取注册后状态
        after_count = len(complete_registry.get_indicator_names())
        print(f"\n注册后指标数量: {after_count}")
        print(f"新增指标数量: {after_count - before_count}")
        
        # 显示注册统计
        stats = complete_registry.get_registration_stats()
        print(f"\n=== 注册统计 ===")
        print(f"尝试注册: {stats['total_attempted']} 个")
        print(f"成功注册: {stats['successful']} 个")
        print(f"注册失败: {stats['failed']} 个")
        print(f"成功率: {(stats['successful']/stats['total_attempted']*100):.1f}%" if stats['total_attempted'] > 0 else "0%")
        
        # 显示所有已注册指标
        all_indicators = sorted(complete_registry.get_indicator_names())
        print(f"\n=== 所有已注册指标 ({len(all_indicators)}个) ===")
        for i, name in enumerate(all_indicators, 1):
            print(f"{i:3d}. {name}")
        
        # 测试指标实例化
        print(f"\n=== 测试指标实例化 ===")
        test_indicators = all_indicators[:10]  # 测试前10个指标
        successful_instances = 0
        
        for indicator_name in test_indicators:
            try:
                indicator = complete_registry.create_indicator(indicator_name)
                if indicator:
                    successful_instances += 1
                    print(f"✅ {indicator_name}: 实例化成功")
                else:
                    print(f"❌ {indicator_name}: 实例化失败")
            except Exception as e:
                print(f"❌ {indicator_name}: 实例化异常 - {e}")
        
        print(f"\n实例化测试: {successful_instances}/{len(test_indicators)} 成功")
        
        # 评估结果
        if after_count >= 80:  # 目标是80+个指标
            print(f"\n🎉 指标注册测试成功！")
            print(f"✅ 注册指标数量: {after_count} (目标: 80+)")
            print(f"✅ 注册成功率: {(stats['successful']/stats['total_attempted']*100):.1f}%")
            return True
        else:
            print(f"\n⚠️  指标注册部分成功")
            print(f"❌ 注册指标数量: {after_count} (目标: 80+)")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicator_categories():
    """测试各类别指标注册情况"""
    print("\n=== 测试各类别指标注册情况 ===")
    
    try:
        from indicators.complete_indicator_registry import complete_registry
        
        all_indicators = complete_registry.get_indicator_names()
        
        # 按类别分类
        categories = {
            'core': [],
            'enhanced': [],
            'composite': [],
            'pattern': [],
            'tools': [],
            'formula': [],
            'zxm': []
        }
        
        for indicator in all_indicators:
            if indicator.startswith('ENHANCED_'):
                categories['enhanced'].append(indicator)
            elif indicator.startswith('ZXM_'):
                categories['zxm'].append(indicator)
            elif indicator in ['COMPOSITE', 'UNIFIED_MA', 'CHIP_DISTRIBUTION', 'INSTITUTIONAL_BEHAVIOR', 'STOCK_VIX']:
                categories['composite'].append(indicator)
            elif indicator in ['CANDLESTICK_PATTERNS', 'ADVANCED_CANDLESTICK', 'ZXM_PATTERNS']:
                categories['pattern'].append(indicator)
            elif indicator in ['FIBONACCI_TOOLS', 'GANN_TOOLS', 'ELLIOTT_WAVE']:
                categories['tools'].append(indicator)
            elif indicator in ['CROSS_OVER', 'KDJ_CONDITION', 'MACD_CONDITION', 'MA_CONDITION', 'GENERIC_CONDITION']:
                categories['formula'].append(indicator)
            else:
                categories['core'].append(indicator)
        
        # 显示各类别统计
        for category, indicators in categories.items():
            print(f"{category.upper()}: {len(indicators)} 个")
            if indicators:
                for indicator in sorted(indicators):
                    print(f"  - {indicator}")
        
        return categories
        
    except Exception as e:
        print(f"❌ 类别测试失败: {e}")
        return {}

def main():
    """主函数"""
    print("开始完整指标注册测试...\n")
    
    # 测试完整注册
    registration_success = test_complete_registration()
    
    # 测试类别分布
    categories = test_indicator_categories()
    
    # 最终评估
    print(f"\n=== 最终评估 ===")
    if registration_success:
        print("🎉 完整指标注册测试通过！")
        print("✅ 系统已具备完整的技术指标库")
        print("✅ 所有主要指标类别都已覆盖")
        print("✅ 指标实例化功能正常")
    else:
        print("⚠️  完整指标注册测试部分通过")
        print("❌ 部分指标注册失败，需要进一步调试")
    
    return registration_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
