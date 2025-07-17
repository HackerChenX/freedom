#!/usr/bin/env python3
"""
指标闭环验证执行脚本

便捷执行指标闭环验证的命令行工具

使用方法:
    python bin/validate_indicators_closed_loop.py --mode quick
    python bin/validate_indicators_closed_loop.py --indicator MACD
    python bin/validate_indicators_closed_loop.py --mode priority
    python bin/validate_indicators_closed_loop.py --mode full

Author: AI Assistant
Date: 2024-12-28
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.indicator_closed_loop_validator import IndicatorClosedLoopValidator
from utils.logger import get_logger

logger = get_logger(__name__)


def print_banner():
    """打印程序横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                        指标闭环验证系统                                  ║
║                                                                       ║
║  🔄 完整的指标验证闭环流程：                                              ║
║     1. 逐个指标生成选股策略                                              ║
║     2. 使用ClickHouse真实数据进行选股                                    ║
║     3. 对选出的股票进行指标验证分析                                       ║
║     4. 验证指标的有效性，形成闭环                                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
验证模式说明:
  quick      快速验证 (验证前5个优先级指标)
  priority   优先级验证 (验证所有优先级指标)
  full       完整验证 (验证所有已注册指标)

使用示例:
  # 快速验证
  python bin/validate_indicators_closed_loop.py --mode quick
  
  # 验证单个指标
  python bin/validate_indicators_closed_loop.py --indicator MACD
  
  # 验证所有优先级指标
  python bin/validate_indicators_closed_loop.py --mode priority
  
  # 完整验证所有指标
  python bin/validate_indicators_closed_loop.py --mode full
  
  # 使用自定义配置文件
  python bin/validate_indicators_closed_loop.py --mode quick --config config/my_config.json
    """
    print(help_text)


def format_validation_summary(report: dict) -> str:
    """格式化验证摘要"""
    summary = report.get('summary', {})
    
    summary_text = f"""
📊 验证摘要:
   总指标数量: {summary.get('total_indicators', 0)}
   成功验证: {summary.get('successful_validations', 0)}
   失败验证: {summary.get('failed_validations', 0)}
   有选股结果: {summary.get('indicators_with_selections', 0)}
   闭环验证通过: {summary.get('closed_loop_success', 0)}
   
📈 成功率: {report.get('success_rate', 0):.2%}
🔄 闭环验证通过率: {report.get('closed_loop_rate', 0):.2%}
    """
    
    return summary_text


def format_indicator_result(indicator_name: str, result: dict) -> str:
    """格式化单个指标验证结果"""
    status_emoji = {
        'success': '✅',
        'no_selection': '⚠️',
        'over_selection': '📈',
        'failed': '❌',
        'strategy_generation_failed': '🔧'
    }
    
    status = result.get('status', 'unknown')
    emoji = status_emoji.get(status, '❓')
    
    result_text = f"""
{emoji} 指标: {indicator_name}
   状态: {status}
   选股数量: {result.get('selection_count', 0)}
   选股比例: {result.get('selection_ratio', 0):.2%}
   质量评分: {result.get('quality_score', 0):.2f}
   闭环验证: {'通过' if result.get('closed_loop_verified', False) else '未通过'}
   执行时间: {result.get('execution_time', 0):.2f}秒
    """
    
    if result.get('error'):
        result_text += f"   错误信息: {result['error']}\n"
    
    return result_text


def display_recommendations(recommendations: list):
    """显示改进建议"""
    if not recommendations:
        return
    
    print("\n💡 改进建议:")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"   {i}. {recommendation}")


def main_31():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='指标闭环验证系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
验证模式说明:
  quick      快速验证 (验证前5个优先级指标)
  priority   优先级验证 (验证所有优先级指标)  
  full       完整验证 (验证所有已注册指标)

示例:
  %(prog)s --mode quick
  %(prog)s --indicator MACD
  %(prog)s --mode priority --config config/my_config.json
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['quick', 'priority', 'full'], 
        default='quick',
        help='验证模式 (默认: quick)'
    )
    
    parser.add_argument(
        '--indicator', 
        type=str,
        help='验证单个指标 (指标名称)'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='配置文件路径 (可选)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    
    parser.add_argument(
        '--help-modes',
        action='store_true',
        help='显示验证模式详细说明'
    )
    
    args = parser.parse_args()
    
    # 显示帮助信息
    if args.help_modes:
        print_help()
        return
    
    # 打印横幅
    print_banner()
    
    try:
        # 创建验证器
        config_file = args.config or 'config/indicator_closed_loop_config.json'
        logger.info(f"使用配置文件: {config_file}")
        
        validator = IndicatorClosedLoopValidator(config_file=config_file)
        
        start_time = datetime.now()
        
        if args.indicator:
            # 验证单个指标
            print(f"🔍 开始验证单个指标: {args.indicator}")
            result = validator.validate_single_indicator(args.indicator)
            
            print(format_indicator_result(args.indicator, result))
            
        else:
            # 批量验证
            print(f"🚀 开始批量验证，模式: {args.mode}")
            report = validator.validate_all_indicators(mode=args.mode)
            
            # 显示验证摘要
            print(format_validation_summary(report))
            
            # 显示详细结果（如果启用详细模式）
            if args.verbose:
                print("\n📋 详细验证结果:")
                for indicator_name, result in report['results'].items():
                    print(format_indicator_result(indicator_name, result))
            
            # 显示改进建议
            display_recommendations(report.get('recommendations', []))
            
            # 显示输出文件位置
            output_config = validator.config.get('output', {})
            print(f"\n📁 结果文件:")
            print(f"   详细结果: {output_config.get('results_file', 'N/A')}")
            print(f"   验证报告: {output_config.get('report_file', 'N/A')}")
            print(f"   CSV文件: {output_config.get('csv_file', 'N/A')}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n⏱️  总执行时间: {duration:.2f}秒")
        print("🎉 验证完成!")
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断验证过程")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
        print(f"\n❌ 验证失败: {e}")
        if args.verbose:
            import traceback
            print("\n🔍 详细错误信息:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_31() 