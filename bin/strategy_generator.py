#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import pandas as pd
import json
import datetime
from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir
from scripts.backtest.indicator_analysis import analyze_buypoint_indicators
from strategy.strategy_factory import StrategyFactory

logger = get_logger(__name__)


def load_stock_list(input_file: str) -> List[Dict[str, str]]:
    """
    从文件加载股票列表和买点日期
    
    Args:
        input_file: 输入文件路径，支持CSV或TXT格式
        
    Returns:
        List[Dict]: 股票数据列表，每项包含code、buy_date和可选的pattern_type
    """
    if input_file.endswith('.csv'):
        # CSV格式，要求至少包含code和buy_date列
        df = pd.read_csv(input_file)
        if 'code' not in df.columns or 'buy_date' not in df.columns:
            raise ValueError("CSV文件必须包含code和buy_date列")
            
        # 确保code是字符串类型
        df['code'] = df['code'].astype(str)
        
        # 添加pattern_type列（如果不存在）
        if 'pattern_type' not in df.columns:
            df['pattern_type'] = ""
            
        return df[['code', 'buy_date', 'pattern_type']].to_dict('records')
    else:
        # 默认为TXT格式，每行格式为: code buy_date [pattern_type]
        result = []
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    item = {
                        'code': parts[0],
                        'buy_date': parts[1],
                        'pattern_type': parts[2] if len(parts) > 2 else ""
                    }
                    result.append(item)
        return result


def run_analysis(input_file: str, output_file: str = None, source_type: str = 'csv') -> Dict:
    """
    运行买点指标分析
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则使用默认路径
        source_type: 输入源类型
        
    Returns:
        Dict: 分析结果
    """
    logger.info(f"开始分析买点指标，输入文件: {input_file}")
    result = analyze_buypoint_indicators(input_file, source_type, output_file)
    
    if result and result.get('stocks_count', 0) > 0:
        logger.info(f"分析完成，共分析 {result['stocks_count']} 只股票")
        common_patterns = result.get('common_patterns', [])
        if common_patterns:
            logger.info("发现的共性特征:")
            for pattern in common_patterns:
                logger.info(f"  - {pattern['pattern']}: {pattern['count']}只股票 ({pattern['ratio']*100:.1f}%)")
    else:
        logger.warning("分析结果为空")
    
    return result


def generate_strategy(analysis_result: Dict, strategy_type: str = None, output_file: str = None) -> str:
    """
    根据分析结果生成选股策略
    
    Args:
        analysis_result: 分析结果
        strategy_type: 策略类型，如果为None则自动判断
        output_file: 输出文件路径，如果为None则使用默认路径
        
    Returns:
        str: 策略文件路径
    """
    try:
        # 创建策略实例
        strategy = StrategyFactory.create_strategy_from_analysis(analysis_result, strategy_type)
        
        # 获取策略参数
        strategy_params = {}
        
        # 根据共性特征优化策略参数
        common_patterns = analysis_result.get('common_patterns', [])
        
        # 策略参数优化逻辑
        if isinstance(strategy, StrategyFactory.get_strategy_class('回踩反弹')):
            # 回踩反弹策略参数优化
            ma_period = 5  # 默认值
            
            # 检查是否有MA5相关的共性特征
            for pattern in common_patterns:
                if pattern['pattern'] == '回踩5日均线反弹' and pattern['ratio'] > 0.5:
                    ma_period = 5
                    break
                elif pattern['pattern'] == 'MA5上穿MA10' and pattern['ratio'] > 0.5:
                    ma_period = 5
                    break
            
            strategy_params['ma_period'] = ma_period
            
            # KDJ相关优化
            kdj_up = False
            for pattern in common_patterns:
                if pattern['pattern'] == 'KDJ金叉' and pattern['ratio'] > 0.3:
                    kdj_up = True
                    break
            
            strategy_params['kdj_up'] = kdj_up
            
        elif isinstance(strategy, StrategyFactory.get_strategy_class('横盘突破')):
            # 横盘突破策略参数优化
            volume_ratio = 1.5  # 默认值
            
            # 检查是否有成交量相关的共性特征
            for pattern in common_patterns:
                if pattern['pattern'] == '成交量放大' and pattern['ratio'] > 0.5:
                    volume_ratio = 2.0  # 提高成交量要求
                    break
            
            strategy_params['volume_ratio'] = volume_ratio
            
            # BOLL相关优化
            boll_use = False
            for pattern in common_patterns:
                if pattern['pattern'] == 'BOLL中轨上穿' and pattern['ratio'] > 0.3:
                    boll_use = True
                    break
            
            strategy_params['boll_use'] = boll_use
        
        # 应用优化后的参数
        if strategy_params:
            strategy.set_parameters(strategy_params)
        
        # 生成策略配置文件
        if output_file is None:
            today = datetime.datetime.now().strftime("%Y%m%d")
            strategy_name = strategy.name.replace(" ", "_")
            output_file = f"{get_backtest_result_dir()}/{today}_{strategy_name}_策略.json"
        
        # 创建目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存策略配置
        strategy_config = {
            'name': strategy.name,
            'description': strategy.description,
            'type': strategy.__class__.__name__,
            'parameters': strategy.parameters,
            'creation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_summary': {
                'stocks_count': analysis_result.get('stocks_count', 0),
                'common_patterns': analysis_result.get('common_patterns', [])
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(strategy_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"策略已生成并保存到 {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"生成策略时出错: {e}")
        raise


def create_buypoint_table(input_data: List[Dict[str, str]], output_file: str = None) -> str:
    """
    在ClickHouse中创建买点数据表
    
    Args:
        input_data: 买点数据
        output_file: 输出表名，如果为None则使用默认名称
        
    Returns:
        str: 表名
    """
    from db.db_manager import DBManager
    
    # 使用数据库管理器单例获取数据库连接
    db_manager = DBManager.get_instance()
    client = db_manager.db.client
    
    # 表名
    if output_file is None:
        today = datetime.datetime.now().strftime("%Y%m%d")
        table_name = f"buypoint_{today}"
    else:
        table_name = output_file
    
    # 创建表
    try:
        client.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            code String,
            buy_date Date,
            pattern_type String DEFAULT '',
            datetime DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (code, buy_date)
        ''')
        
        # 插入数据
        data = []
        for item in input_data:
            code = item['code']
            buy_date = datetime.datetime.strptime(item['buy_date'], "%Y%m%d").date()
            pattern_type = item.get('pattern_type', '')
            data.append((code, buy_date, pattern_type))
        
        client.execute(f"INSERT INTO {table_name} (code, buy_date, pattern_type) VALUES", data)
        
        logger.info(f"买点数据已保存到表 {table_name}，共 {len(data)} 条记录")
        return table_name
        
    except Exception as e:
        logger.error(f"创建买点数据表时出错: {e}")
        raise


def run(args):
    """
    运行策略生成器
    
    Args:
        args: 命令行参数
    """
    try:
        # 加载股票列表
        stock_list = load_stock_list(args.input)
        logger.info(f"加载了 {len(stock_list)} 只股票")
        
        # 保存到数据库（如果需要）
        if args.save_to_db:
            table_name = create_buypoint_table(stock_list, args.table_name)
            
            # 如果指定了使用数据库，则修改输入源为数据库表
            if args.use_db:
                args.input = table_name
                args.type = 'db'
        
        # 运行分析
        analysis_result = run_analysis(args.input, args.output, args.type)
        
        # 生成策略
        if analysis_result and analysis_result.get('stocks_count', 0) > 0:
            strategy_file = generate_strategy(analysis_result, args.strategy_type)
            logger.info(f"策略生成完成: {strategy_file}")
        else:
            logger.warning("由于分析结果为空，未生成策略")
        
    except Exception as e:
        logger.error(f"运行策略生成器时出错: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='买点策略生成工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径，支持CSV或TXT格式')
    parser.add_argument('--type', type=str, default="csv", choices=['csv', 'db'], help='输入源类型: csv-CSV文件, db-数据库表')
    parser.add_argument('--output', type=str, help='分析结果输出文件路径')
    parser.add_argument('--strategy-type', type=str, help='策略类型，如"回踩反弹"、"横盘突破"等，不指定则自动判断')
    parser.add_argument('--save-to-db', action='store_true', help='是否将买点数据保存到数据库')
    parser.add_argument('--use-db', action='store_true', help='是否使用数据库中的买点数据进行分析')
    parser.add_argument('--table-name', type=str, help='数据库表名')
    
    args = parser.parse_args()
    
    run(args) 