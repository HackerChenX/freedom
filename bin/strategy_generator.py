#!/usr/bin/env python3
"""
⚠️ 此文件已废弃 ⚠️

原文件: bin/strategy_generator.py
文件类型: strategy_execution
废弃日期: 2025-09-14 16:28:42

此入口已被废弃，请使用新的统一入口：

from bin.unified_analysis_controller import unified_controller
from db.sql_manager import SQLManager, QueryType

# 买点分析
result = unified_controller.analyze_buypoint(stock_code, buypoint_date)

# 策略选股  
result = unified_controller.execute_stock_selection(strategy_config)

# 技术指标
result = unified_controller.get_technical_indicators(stock_code, date, indicators)

详细文档: docs/optimization/basic_usability_architecture_plan.md
"""

import warnings
import sys
import os
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def deprecated_entry_warning():
    """废弃入口警告"""
    warnings.warn(
        f"此入口文件已废弃: /Users/hacker/PycharmProjects/freedom/bin/entry_deprecation_manager.py\n"
        f"请使用统一入口: bin.unified_analysis_controller\n"
        f"详情请查看文档: docs/optimization/basic_usability_architecture_plan.md",
        DeprecationWarning,
        stacklevel=2
    )

def redirect_to_unified_controller(*args, **kwargs):
    """重定向到统一控制器"""
    deprecated_entry_warning()
    
    try:
        from bin.unified_analysis_controller import unified_controller
from db.sql_manager import SQLManager, QueryType
    except ImportError as e:
        print(f"❌ 无法导入统一控制器: {e}")
        print("请确保 bin/unified_analysis_controller.py 文件存在")
        return {'success': False, 'error': '统一控制器不可用'}
    
    # 根据文件类型判断重定向目标
    file_type = "strategy_execution"
    
    if file_type == "buypoint_analysis":
        if len(args) >= 2:
            return unified_controller.analyze_buypoint(args[0], args[1], kwargs.get('analysis_config'))
        else:
            print("❌ 买点分析需要 stock_code 和 buypoint_date 参数")
            print("使用方法: python {__file__} <stock_code> <buypoint_date>")
            return {'success': False, 'error': '参数不足'}
    
    elif file_type == "stock_selection":
        if len(args) >= 1 and isinstance(args[0], dict):
            return unified_controller.execute_stock_selection(args[0], kwargs.get('selection_params'))
        else:
            print("❌ 策略选股需要 strategy_config 参数")
            print("使用方法: 请参考统一入口文档")
            return {'success': False, 'error': '参数不足'}
    
    elif file_type == "strategy_execution":
        # 策略执行重定向到选股
        strategy_config = {'name': kwargs.get('strategy_name', 'unknown')}
        return unified_controller.execute_stock_selection(strategy_config, kwargs)
    
    else:
        print("❌ 无法确定重定向目标，请直接使用统一入口")
        print("统一入口: bin/unified_analysis_controller.py")
        return {'success': False, 'error': '无法确定重定向目标'}

# 为向后兼容提供的函数别名
main = redirect_to_unified_controller
analyze = redirect_to_unified_controller
execute = redirect_to_unified_controller
run = redirect_to_unified_controller
select = redirect_to_unified_controller
detect = redirect_to_unified_controller

if __name__ == "__main__":
    deprecated_entry_warning()
    
    print(f"⚠️  此文件已废弃: entry_deprecation_manager.py")
    print(f"📁 原文件类型: strategy_execution")
    print(f"🔄 请使用统一入口: bin/unified_analysis_controller.py")
    print()
    print("📖 使用示例:")
    print("   python bin/unified_analysis_controller.py buypoint --stock 000001 --date 2024-01-15")
    print("   python bin/unified_analysis_controller.py selection --strategy momentum")
    print("   python bin/unified_analysis_controller.py indicators --stock 000001 --date 2024-01-15 --indicators RSI,MACD")
    print()
    print("📚 详细文档: docs/optimization/basic_usability_architecture_plan.md")
    
    # 尝试自动重定向
    if len(sys.argv) > 1:
        print("🔄 尝试自动重定向...")
        result = redirect_to_unified_controller(*sys.argv[1:])
        if result.get('success'):
            print("✅ 重定向成功")
        else:
            print(f"❌ 重定向失败: {result.get('error')}")
