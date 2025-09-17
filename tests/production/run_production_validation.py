#!/usr/bin/env python3
"""
生产环境指标验证系统运行脚本
快速启动和测试验证功能
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from scripts.production_indicator_validator import Production_indicator_validator
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


def test_single_indicator():
    """测试单个指标验证"""
    print("=" * 60)
    print("测试单个指标验证（ZXM成交量缩量）")
    print("=" * 60)
    
    validator = Production_indicator_validator(max_stocks=20)
    
    # 获取最新交易日期
    test_date = validator.get_latest_trading_date()
    print(f"使用测试日期: {test_date}")
    
    # 验证单个指标
    results = validator.run_validation(['volume_shrink'])
    
    if results:
        print(f"\n验证结果:")
        print(f"- 测试日期: {results['test_date']}")
        print(f"- 股票池大小: {results['stock_pool_size']}")
        
        volume_shrink_result = results['indicators']['volume_shrink']
        print(f"- 成交量缩量指标:")
        print(f"  * 处理成功率: {volume_shrink_result['success_rate']:.2%}")
        print(f"  * 选股数量: {len(volume_shrink_result['selected_stocks'])}")
        print(f"  * 选股比例: {volume_shrink_result.get('selection_rate', 0.0):.2%}")
        
        if volume_shrink_result['selected_stocks']:
            print(f"  * 选中股票前5只:")
            for i, stock in enumerate(volume_shrink_result['selected_stocks'][:5]):
                print(f"    {i+1}. {stock['code']} - 信号强度: {stock['signal_strength']:.2f}")
    else:
        print("验证失败！")


def test_multiple_indicators():
    """测试多指标验证"""
    print("=" * 60)
    print("测试多指标验证（ZXM系列指标）")
    print("=" * 60)
    
    validator = Production_indicator_validator(max_stocks=30)
    
    # 验证多个ZXM指标
    zxm_indicators = ['volume_shrink', 'bs_absorb', 'turnover_rate']
    results = validator.run_validation(zxm_indicators)
    
    if results:
        print(f"\n验证结果:")
        print(f"- 测试日期: {results['test_date']}")
        print(f"- 股票池大小: {results['stock_pool_size']}")
        print(f"- 验证指标数: {len(results['indicators'])}")
        
        summary = results['summary']
        print(f"- 总选股数: {summary['total_selections']}")
        print(f"- 平均选股率: {summary['average_selection_rate']:.2%}")
        print(f"- 最佳指标: {summary['best_indicator']}")
        
        print(f"\n各指标详情:")
        for name, result in results['indicators'].items():
            print(f"  {name}:")
            print(f"    - 成功率: {result['success_rate']:.2%}")
            print(f"    - 选股数: {len(result['selected_stocks'])}")
            print(f"    - 选股率: {result.get('selection_rate', 0.0):.2%}")
        
        # 显示重叠分析
        if results['overlap_analysis']:
            print(f"\n指标重叠分析:")
            for key, overlap in results['overlap_analysis'].items():
                print(f"  {key}: 重叠率={overlap['overlap_rate']:.2%}, 重叠数={overlap['overlap_count']}")
    else:
        print("验证失败！")


def test_database_connection_Validation():
    """测试数据库连接"""
    print("=" * 60)
    print("测试ClickHouse数据库连接")
    print("=" * 60)
    
    try:
        validator = Production_indicator_validator()
        
        # 测试获取最新交易日期
        latest_date = validator.get_latest_trading_date()
        print(f"✓ 最新交易日期: {latest_date}")
        
        # 测试获取股票池
        stock_codes = validator.get_stock_pool(latest_date)
        print(f"✓ 股票池大小: {len(stock_codes)}")
        
        if stock_codes:
            print(f"✓ 股票池前10只: {stock_codes[:10]}")
            
            # 测试获取单只股票数据
            test_code = stock_codes[0]
            df = validator.get_stock_data(test_code, latest_date, days=30)
            print(f"✓ 股票 {test_code} 数据: {len(df)} 条记录")
        
        print("✓ 数据库连接测试通过")
        
    except Exception as e:
        print(f"✗ 数据库连接测试失败: {e}")


def list_available_indicators():
    """列出所有可用指标"""
    print("=" * 60)
    print("可用指标列表")
    print("=" * 60)
    
    validator = Production_indicator_validator()
    
    print("ZXM系列指标:")
    zxm_indicators = ['volume_shrink', 'bs_absorb', 'turnover_rate', 'price_volume_trend', 'breakthrough']
    for indicator in zxm_indicators:
        if indicator in validator.indicators:
            print(f"  ✓ {indicator}")
    
    print("\n传统技术指标:")
    traditional_indicators = ['macd', 'rsi', 'kdj', 'boll', 'ma']
    for indicator in traditional_indicators:
        if indicator in validator.indicators:
            print(f"  ✓ {indicator}")
    
    print(f"\n总计: {len(validator.indicators)} 个指标")


def main_runproductionvalidation():
    """主函数"""
    parser = argparse.ArgumentParser(description='生产环境指标验证系统测试')
    parser.add_argument('--test', choices=['db', 'single', 'multiple', 'list'], 
                       default='single', help='测试类型')
    parser.add_argument('--indicators', nargs='+', help='指定要验证的指标')
    parser.add_argument('--date', type=str, help='指定测试日期')
    parser.add_argument('--max-stocks', type=int, default=50, help='最大股票数量')
    
    args = parser.parse_args()
    
    print(f"生产环境指标验证系统测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.test == 'db':
        test_database_connection_Validation()
    elif args.test == 'single':
        test_single_indicator()
    elif args.test == 'multiple':
        test_multiple_indicators()
    elif args.test == 'list':
        list_available_indicators()
    else:
        # 自定义验证
        if args.indicators:
            print("=" * 60)
            print(f"自定义指标验证: {args.indicators}")
            print("=" * 60)
            
            validator = Production_indicator_validator(
                test_date=args.date,
                max_stocks=args.max_stocks
            )
            
            results = validator.run_validation(args.indicators)
            
            if results:
                print(f"\n验证完成!")
                print(f"详细结果已保存到 results/ 目录")
            else:
                print("验证失败!")


if __name__ == '__main__':
    main_runproductionvalidation() 