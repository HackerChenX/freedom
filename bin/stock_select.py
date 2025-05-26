#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
选股命令行工具

提供基于配置文件执行选股功能的命令行接口
"""

import os
import sys
import argparse
import json
import yaml
from datetime import datetime
import pandas as pd

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_parser import StrategyParser
from strategy.strategy_manager import StrategyManager
from strategy.strategy_executor import StrategyExecutor
from utils.logger import get_logger
from utils.path_utils import get_strategy_dir, get_result_dir

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于配置文件执行选股')
    
    # 策略相关参数
    strategy_group = parser.add_argument_group('策略参数')
    strategy_group.add_argument('-s', '--strategy', help='策略ID')
    strategy_group.add_argument('-f', '--file', help='策略配置文件路径')
    
    # 执行参数
    exec_group = parser.add_argument_group('执行参数')
    exec_group.add_argument('-d', '--date', help='执行日期，格式：YYYY-MM-DD，默认为当前日期')
    exec_group.add_argument('-p', '--pool', help='股票池文件路径，每行一个股票代码')
    
    # 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('-o', '--output', help='输出文件路径，支持.csv和.xlsx格式')
    output_group.add_argument('--save-db', action='store_true', help='是否保存结果到数据库')
    output_group.add_argument('--pretty', action='store_true', help='是否美化输出')
    
    # 管理参数
    manage_group = parser.add_argument_group('管理参数')
    manage_group.add_argument('--list', action='store_true', help='列出所有可用的策略')
    manage_group.add_argument('--show', help='显示指定策略的配置')
    manage_group.add_argument('--import', dest='import_file', help='导入策略配置文件')
    manage_group.add_argument('--export', help='导出策略配置到文件')
    manage_group.add_argument('--delete', help='删除指定策略')
    
    return parser.parse_args()


def load_strategy(strategy_id=None, file_path=None):
    """
    加载策略配置
    
    Args:
        strategy_id: 策略ID
        file_path: 策略配置文件路径
        
    Returns:
        策略配置字典
    """
    strategy_manager = StrategyManager()
    
    if strategy_id:
        # 从数据库或配置文件加载策略
        strategy_config = strategy_manager.get_strategy(strategy_id)
        if not strategy_config:
            logger.error(f"策略 {strategy_id} 不存在")
            sys.exit(1)
            
        return strategy_config
    elif file_path:
        # 从指定文件加载策略
        if not os.path.exists(file_path):
            logger.error(f"策略配置文件 {file_path} 不存在")
            sys.exit(1)
            
        # 根据文件扩展名确定解析方式
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_ext in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.error(f"不支持的配置文件格式: {file_ext}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"解析策略配置文件失败: {e}")
            sys.exit(1)
    else:
        logger.error("必须指定策略ID或配置文件路径")
        sys.exit(1)


def load_stock_pool(pool_file=None):
    """
    加载股票池
    
    Args:
        pool_file: 股票池文件路径
        
    Returns:
        股票代码列表
    """
    if not pool_file:
        return None
        
    if not os.path.exists(pool_file):
        logger.error(f"股票池文件 {pool_file} 不存在")
        sys.exit(1)
        
    try:
        with open(pool_file, 'r', encoding='utf-8') as f:
            stock_codes = [line.strip() for line in f if line.strip()]
            
        if not stock_codes:
            logger.warning("股票池为空")
            
        return stock_codes
    except Exception as e:
        logger.error(f"加载股票池失败: {e}")
        sys.exit(1)


def execute_strategy(strategy_config, date=None, stock_pool=None):
    """
    执行选股策略
    
    Args:
        strategy_config: 策略配置
        date: 执行日期
        stock_pool: 股票池
        
    Returns:
        选股结果DataFrame
    """
    executor = StrategyExecutor()
    
    try:
        result = executor.execute(
            strategy_config=strategy_config,
            date=date,
            stock_pool=stock_pool
        )
        
        return result
    except Exception as e:
        logger.error(f"执行选股策略失败: {e}")
        sys.exit(1)


def save_result(result, output_file=None, save_db=False, pretty=False):
    """
    保存选股结果
    
    Args:
        result: 选股结果DataFrame
        output_file: 输出文件路径
        save_db: 是否保存到数据库
        pretty: 是否美化输出
        
    Returns:
        无
    """
    if result.empty:
        logger.warning("选股结果为空")
        return
        
    # 打印结果
    if pretty:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        
    print("\n选股结果:")
    print(result)
    print(f"\n共找到 {len(result)} 只符合条件的股票")
    
    # 保存到文件
    if output_file:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # 根据文件扩展名保存
            file_ext = os.path.splitext(output_file)[1].lower()
            
            if file_ext == '.csv':
                result.to_csv(output_file, index=False, encoding='utf-8-sig')
            elif file_ext == '.xlsx':
                result.to_excel(output_file, index=False, engine='openpyxl')
            else:
                # 默认保存为CSV
                output_file = output_file + '.csv'
                result.to_csv(output_file, index=False, encoding='utf-8-sig')
                
            logger.info(f"已保存选股结果到文件: {output_file}")
        except Exception as e:
            logger.error(f"保存选股结果到文件失败: {e}")
    
    # 保存到数据库
    if save_db:
        from db.data_manager import DataManager
        data_manager = DataManager()
        
        if data_manager.save_selection_result(result):
            logger.info("已保存选股结果到数据库")


def list_strategies():
    """列出所有可用的策略"""
    strategy_manager = StrategyManager()
    strategies = strategy_manager.list_strategies()
    
    if not strategies:
        print("没有可用的策略")
        return
        
    # 格式化输出
    print("\n可用的策略列表:")
    print(f"{'ID':<20} {'名称':<30} {'版本':<10} {'作者':<15} {'更新时间':<20}")
    print("-" * 95)
    
    for strategy in strategies:
        strategy_info = strategy["strategy"]
        print(f"{strategy_info['id']:<20} {strategy_info['name']:<30} "
              f"{strategy_info.get('version', '1.0'):<10} "
              f"{strategy_info.get('author', 'system'):<15} "
              f"{strategy_info.get('update_time', ''):<20}")
    
    print(f"\n共 {len(strategies)} 个可用策略")


def show_strategy(strategy_id):
    """
    显示指定策略的配置
    
    Args:
        strategy_id: 策略ID
    """
    strategy_manager = StrategyManager()
    strategy_config = strategy_manager.get_strategy(strategy_id)
    
    if not strategy_config:
        logger.error(f"策略 {strategy_id} 不存在")
        return
        
    # 格式化输出
    print(f"\n策略 {strategy_id} 的配置:")
    print(json.dumps(strategy_config, ensure_ascii=False, indent=2))


def import_strategy(file_path):
    """
    导入策略配置
    
    Args:
        file_path: 策略配置文件路径
    """
    if not os.path.exists(file_path):
        logger.error(f"策略配置文件 {file_path} 不存在")
        return
        
    strategy_manager = StrategyManager()
    
    try:
        strategy_id = strategy_manager.import_strategy(file_path)
        logger.info(f"已成功导入策略: {strategy_id}")
    except Exception as e:
        logger.error(f"导入策略失败: {e}")


def export_strategy(strategy_id, file_path=None):
    """
    导出策略配置
    
    Args:
        strategy_id: 策略ID
        file_path: 导出文件路径，如果为None则使用默认路径
    """
    strategy_manager = StrategyManager()
    
    if not file_path:
        # 使用默认路径
        file_path = os.path.join(get_result_dir(), f"{strategy_id}.json")
        
    # 确定导出格式
    file_ext = os.path.splitext(file_path)[1].lower()
    format_type = 'json'
    if file_ext in ['.yml', '.yaml']:
        format_type = 'yaml'
        
    try:
        success = strategy_manager.export_strategy(
            strategy_id=strategy_id,
            file_path=file_path,
            format_type=format_type
        )
        
        if success:
            logger.info(f"已成功导出策略 {strategy_id} 到文件: {file_path}")
    except Exception as e:
        logger.error(f"导出策略失败: {e}")


def delete_strategy(strategy_id):
    """
    删除指定策略
    
    Args:
        strategy_id: 策略ID
    """
    strategy_manager = StrategyManager()
    
    # 先显示策略信息
    strategy_config = strategy_manager.get_strategy(strategy_id)
    
    if not strategy_config:
        logger.error(f"策略 {strategy_id} 不存在")
        return
        
    # 确认删除
    strategy_name = strategy_config["strategy"]["name"]
    confirm = input(f"确定要删除策略 {strategy_id} ({strategy_name}) 吗? [y/N] ")
    
    if confirm.lower() != 'y':
        logger.info("已取消删除")
        return
        
    # 执行删除
    success = strategy_manager.delete_strategy(strategy_id)
    
    if success:
        logger.info(f"已成功删除策略: {strategy_id}")
    else:
        logger.error(f"删除策略 {strategy_id} 失败")


def main():
    """主函数"""
    args = parse_args()
    
    # 管理命令
    if args.list:
        list_strategies()
        return
        
    if args.show:
        show_strategy(args.show)
        return
        
    if args.import_file:
        import_strategy(args.import_file)
        return
        
    if args.export:
        export_strategy(args.export)
        return
        
    if args.delete:
        delete_strategy(args.delete)
        return
    
    # 执行选股
    if not args.strategy and not args.file:
        logger.error("必须指定策略ID或配置文件路径")
        sys.exit(1)
        
    # 加载策略配置
    strategy_config = load_strategy(args.strategy, args.file)
    
    # 加载股票池
    stock_pool = load_stock_pool(args.pool)
    
    # 执行选股
    result = execute_strategy(
        strategy_config=strategy_config,
        date=args.date,
        stock_pool=stock_pool
    )
    
    # 保存结果
    save_result(
        result=result,
        output_file=args.output,
        save_db=args.save_db,
        pretty=args.pretty
    )


if __name__ == '__main__':
    main() 