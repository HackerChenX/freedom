#!/usr/bin/env python3
"""
通用验证接口 - 符合六层架构的生产级实现
L6: 用户接口层 - 统一操作入口
"""

import sys
import os
import argparse
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from analysis.universal_bidirectional_validator import UniversalBidirectionalValidator
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class UniversalValidationInterface:
    """
    通用验证接口
    
    符合六层架构设计：
    - L6: 用户接口层 - 统一操作入口
    - 提供命令行和编程接口
    - 优化用户体验
    """
    
    def __init__(self):
        """初始化接口"""
        self.validator = UniversalBidirectionalValidator()
        logger.info("通用验证接口初始化完成")
    
    def run_single_validation(self, stock_code: str, target_date: str,
                             timeframes: Optional[List[str]] = None,
                             test_stocks: Optional[List[str]] = None,
                             save_report: bool = True) -> Dict[str, Any]:
        """
        运行单个股票的双向验证
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期 (YYYYMMDD)
            timeframes: 时间周期列表
            test_stocks: 测试股票列表
            save_report: 是否保存报告
            
        Returns:
            Dict: 验证结果
        """
        print(f"\n🚀 开始验证: {stock_code} ({target_date})")
        print("=" * 60)
        
        try:
            # 执行验证
            result = self.validator.validate_bidirectional(
                stock_code=stock_code,
                target_date=target_date,
                timeframes=timeframes,
                test_stocks=test_stocks
            )
            
            # 显示结果
            self._display_single_result(result)
            
            # 保存报告
            if save_report:
                report_file = f"validation_report_{stock_code}_{target_date}.json"
                self.validator.save_validation_report(result, report_file)
                print(f"\n📄 报告已保存: {report_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            print(f"❌ 验证失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_batch_validation(self, stock_list: List[Dict[str, str]],
                            timeframes: Optional[List[str]] = None,
                            test_stocks: Optional[List[str]] = None,
                            save_report: bool = True) -> Dict[str, Any]:
        """
        运行批量双向验证
        
        Args:
            stock_list: 股票列表
            timeframes: 时间周期列表
            test_stocks: 测试股票列表
            save_report: 是否保存报告
            
        Returns:
            Dict: 批量验证结果
        """
        print(f"\n🚀 开始批量验证: {len(stock_list)} 只股票")
        print("=" * 60)
        
        try:
            # 执行批量验证
            results = self.validator.batch_validate(
                stock_list=stock_list,
                timeframes=timeframes,
                test_stocks=test_stocks
            )
            
            # 显示结果
            self._display_batch_results(results)
            
            # 保存报告
            if save_report:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_file = f"batch_validation_report_{timestamp}.json"
                self.validator.save_validation_report(results, report_file)
                print(f"\n📄 批量报告已保存: {report_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"批量验证失败: {e}")
            print(f"❌ 批量验证失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _display_single_result(self, result: Dict[str, Any]):
        """显示单个验证结果"""
        stock_code = result.get('stock_code', 'N/A')
        target_date = result.get('target_date', 'N/A')
        success = result.get('validation_success', False)
        
        print(f"\n📊 验证结果: {stock_code} ({target_date})")
        print("-" * 40)
        
        # 买点分析结果
        buypoint = result.get('buypoint_analysis', {})
        print(f"🔍 买点分析:")
        print(f"   ├─ 技术指标数量: {buypoint.get('total_indicators', 0)}")
        print(f"   ├─ 形态匹配数量: {buypoint.get('total_patterns', 0)}")
        print(f"   ├─ 买点评分: {buypoint.get('buypoint_score', 0):.4f}")
        print(f"   └─ 分析状态: {'✅ 成功' if buypoint.get('success') else '❌ 失败'}")
        
        # 策略生成结果
        strategy = result.get('strategy_generation', {})
        print(f"\n⚙️  策略生成:")
        print(f"   ├─ 策略ID: {strategy.get('strategy_id', 'N/A')}")
        print(f"   ├─ 配置文件: {strategy.get('config_file', 'N/A')}")
        print(f"   └─ 生成状态: {'✅ 成功' if strategy.get('success') else '❌ 失败'}")
        
        # 策略验证结果
        validation = result.get('strategy_validation', {})
        print(f"\n🎯 策略验证:")
        print(f"   ├─ 目标选中: {'✅ 是' if validation.get('target_selected') else '❌ 否'}")
        print(f"   ├─ 目标评分: {validation.get('target_score', 0):.4f}")
        print(f"   ├─ 选中数量: {validation.get('total_selected', 0)}")
        print(f"   ├─ 选中股票: {', '.join(validation.get('selected_stocks', []))}")
        print(f"   └─ 验证状态: {'✅ 成功' if validation.get('success') else '❌ 失败'}")
        
        # 最终结果
        print(f"\n🏆 最终结果: {'✅ 双向验证成功' if success else '❌ 双向验证失败'}")
    
    def _display_batch_results(self, results: Dict[str, Any]):
        """显示批量验证结果"""
        summary = results.get('summary', {})
        total = summary.get('total_stocks', 0)
        success = summary.get('success_count', 0)
        failed = summary.get('failed_count', 0)
        success_rate = summary.get('success_rate', 0)
        
        print(f"\n📊 批量验证汇总")
        print("-" * 40)
        print(f"📈 总计股票: {total}")
        print(f"✅ 验证成功: {success}")
        print(f"❌ 验证失败: {failed}")
        print(f"📊 成功率: {success_rate:.1f}%")
        
        # 失败原因分析
        failure_reasons = summary.get('failure_reasons', {})
        if failure_reasons:
            print(f"\n🔍 失败原因分析:")
            for reason, count in failure_reasons.items():
                print(f"   ├─ {reason}: {count} 次")
        
        # 详细结果
        print(f"\n📋 详细结果:")
        for i, result in enumerate(results.get('validation_results', []), 1):
            stock_code = result.get('stock_code', 'N/A')
            target_date = result.get('target_date', 'N/A')
            success = result.get('validation_success', False)
            status = '✅' if success else '❌'
            
            print(f"   {i:2d}. {status} {stock_code} ({target_date})")
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🎯 通用买点分析与策略验证系统")
        print("=" * 50)
        print("符合六层架构的生产级实现")
        print("支持动态指标获取和配置文件驱动策略")
        print("=" * 50)
        
        while True:
            print("\n📋 请选择操作:")
            print("1. 单个股票验证")
            print("2. 批量股票验证")
            print("3. 退出")
            
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == '1':
                self._interactive_single_validation()
            elif choice == '2':
                self._interactive_batch_validation()
            elif choice == '3':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def _interactive_single_validation(self):
        """交互式单个验证"""
        print("\n📝 单个股票验证")
        print("-" * 30)
        
        try:
            stock_code = input("股票代码: ").strip()
            target_date = input("目标日期 (YYYYMMDD): ").strip()
            
            if not stock_code or not target_date:
                print("❌ 股票代码和日期不能为空")
                return
            
            # 可选参数
            timeframes_input = input("时间周期 (默认: 日线, 多个用逗号分隔): ").strip()
            timeframes = [tf.strip() for tf in timeframes_input.split(',')] if timeframes_input else None
            
            test_stocks_input = input("测试股票列表 (默认: 系统自动选择, 多个用逗号分隔): ").strip()
            test_stocks = [ts.strip() for ts in test_stocks_input.split(',')] if test_stocks_input else None
            
            # 执行验证
            self.run_single_validation(
                stock_code=stock_code,
                target_date=target_date,
                timeframes=timeframes,
                test_stocks=test_stocks
            )
            
        except KeyboardInterrupt:
            print("\n⏹️  操作已取消")
        except Exception as e:
            print(f"❌ 操作失败: {e}")
    
    def _interactive_batch_validation(self):
        """交互式批量验证"""
        print("\n📝 批量股票验证")
        print("-" * 30)
        
        try:
            print("请输入股票列表 (格式: 股票代码,日期):")
            print("示例: 603359,20250512")
            print("输入空行结束:")
            
            stock_list = []
            while True:
                line = input().strip()
                if not line:
                    break
                
                parts = line.split(',')
                if len(parts) == 2:
                    stock_list.append({
                        'code': parts[0].strip(),
                        'date': parts[1].strip()
                    })
                else:
                    print(f"❌ 格式错误，跳过: {line}")
            
            if not stock_list:
                print("❌ 没有有效的股票数据")
                return
            
            # 可选参数
            timeframes_input = input("时间周期 (默认: 日线, 多个用逗号分隔): ").strip()
            timeframes = [tf.strip() for tf in timeframes_input.split(',')] if timeframes_input else None
            
            test_stocks_input = input("测试股票列表 (默认: 系统自动选择, 多个用逗号分隔): ").strip()
            test_stocks = [ts.strip() for ts in test_stocks_input.split(',')] if test_stocks_input else None
            
            # 执行批量验证
            self.run_batch_validation(
                stock_list=stock_list,
                timeframes=timeframes,
                test_stocks=test_stocks
            )
            
        except KeyboardInterrupt:
            print("\n⏹️  操作已取消")
        except Exception as e:
            print(f"❌ 操作失败: {e}")


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description="通用买点分析与策略验证系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互模式
  python universal_validation_interface.py
  
  # 单个股票验证
  python universal_validation_interface.py --stock 603359 --date 20250512
  
  # 批量验证
  python universal_validation_interface.py --batch stocks.json
  
  # 指定时间周期
  python universal_validation_interface.py --stock 603359 --date 20250512 --timeframes 日线,30分钟
        """
    )
    
    parser.add_argument('--stock', help='股票代码')
    parser.add_argument('--date', help='目标日期 (YYYYMMDD)')
    parser.add_argument('--batch', help='批量验证文件 (JSON格式)')
    parser.add_argument('--timeframes', help='时间周期 (逗号分隔)')
    parser.add_argument('--test-stocks', help='测试股票列表 (逗号分隔)')
    parser.add_argument('--no-report', action='store_true', help='不保存报告')
    
    args = parser.parse_args()
    
    interface = UniversalValidationInterface()
    
    try:
        if args.stock and args.date:
            # 单个股票验证
            timeframes = args.timeframes.split(',') if args.timeframes else None
            test_stocks = args.test_stocks.split(',') if args.test_stocks else None
            
            interface.run_single_validation(
                stock_code=args.stock,
                target_date=args.date,
                timeframes=timeframes,
                test_stocks=test_stocks,
                save_report=not args.no_report
            )
            
        elif args.batch:
            # 批量验证
            with open(args.batch, 'r', encoding='utf-8') as f:
                stock_list = json.load(f)
            
            timeframes = args.timeframes.split(',') if args.timeframes else None
            test_stocks = args.test_stocks.split(',') if args.test_stocks else None
            
            interface.run_batch_validation(
                stock_list=stock_list,
                timeframes=timeframes,
                test_stocks=test_stocks,
                save_report=not args.no_report
            )
            
        else:
            # 交互模式
            interface.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 程序执行失败: {e}")


if __name__ == "__main__":
    main()
