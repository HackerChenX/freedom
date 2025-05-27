#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
完整策略验证工具

用于验证包含所有重要指标（特别是ZXM指标）的完整策略在买点日期的表现
"""

import os
import sys
import pandas as pd
from datetime import datetime
import traceback

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_parser import StrategyParser
from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
from utils.logger import get_logger, init_logging
from utils.path_utils import get_result_dir
from db.data_manager import DataManager
from enums.period import Period

logger = get_logger(__name__)

def load_buypoints(file_path):
    """加载买点数据"""
    try:
        buypoints_df = pd.read_csv(file_path)
        
        # 处理日期列，将 buy_date 转换为 date 列
        if 'buy_date' in buypoints_df.columns and 'date' not in buypoints_df.columns:
            buypoints_df['date'] = buypoints_df['buy_date'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}" if isinstance(x, str) and len(x) >= 8 else x)
        
        # 处理股票代码列，将 code 转换为 stock_code 列
        if 'code' in buypoints_df.columns and 'stock_code' not in buypoints_df.columns:
            buypoints_df['stock_code'] = buypoints_df['code']
            
        # 添加股票名称列（如果没有）
        if 'stock_name' not in buypoints_df.columns:
            buypoints_df['stock_name'] = ''
            
        logger.info(f"成功加载买点数据，共 {len(buypoints_df)} 条记录")
        logger.info(f"买点数据列: {buypoints_df.columns.tolist()}")
        return buypoints_df
    except Exception as e:
        logger.error(f"加载买点数据失败: {e}")
        return None

def verify_strategy():
    """验证策略"""
    try:
        # 初始化日志
        init_logging(level="INFO")
        
        # 加载买点数据
        buypoints_file = os.path.join(root_dir, "data", "buypoints.csv")
        buypoints_df = load_buypoints(buypoints_file)
        if buypoints_df is None or len(buypoints_df) == 0:
            logger.error("买点数据为空，无法进行验证")
            return
            
        # 解析策略配置
        logger.info("开始解析完整策略配置")
        strategy_parser = StrategyParser()
        strategy_file = os.path.join(root_dir, "config", "strategies", "complete_backtest_strategy.yaml")
        strategy = strategy_parser.parse_from_file(strategy_file)
        logger.info(f"成功解析策略: {strategy['name']}")
        
        # 初始化数据管理器
        data_manager = DataManager()
        
        # 初始化条件评估器
        evaluator = StrategyConditionEvaluator()
        
        # 评估每个买点股票是否满足策略条件
        verification_results = []
        success_count = 0
        total_count = len(buypoints_df)
        
        logger.info(f"开始验证 {total_count} 个买点")
        
        # 对每个买点日期分组处理
        grouped_buypoints = buypoints_df.groupby('date')
        
        for date, group in grouped_buypoints:
            logger.info(f"验证日期 {date} 的 {len(group)} 个股票")
            date_success_count = 0
            
            for _, row in group.iterrows():
                stock_code = row['stock_code']
                stock_name = row['stock_name']
                
                # 确保股票代码是字符串
                if isinstance(stock_code, (int, float)):
                    stock_code = str(int(stock_code)).zfill(6)
                
                logger.info(f"验证股票 {stock_code} - {stock_name}")
                
                try:
                    # 获取K线数据
                    period = Period.DAILY
                    
                    # 格式化日期字符串为YYYY-MM-DD格式
                    formatted_date = date
                    if isinstance(formatted_date, (int, str)) and not isinstance(formatted_date, bool):
                        formatted_date = str(formatted_date)
                        if len(formatted_date) == 8:
                            formatted_date = f"{formatted_date[:4]}-{formatted_date[4:6]}-{formatted_date[6:8]}"
                    
                    logger.info(f"查询K线数据，股票代码: {stock_code}，日期: {date} (格式化后: {formatted_date})")
                    
                    k_data = data_manager.get_kline_data(
                        stock_code=stock_code,
                        period=period,
                        end_date=formatted_date,
                        fields=None  # 获取所有字段
                    )
                    
                    if k_data is None or len(k_data) == 0:
                        logger.error(f"获取股票 {stock_code} 的K线数据失败")
                        selected = False
                        verification_results.append({
                            'date': date,
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'selected': selected,
                            'reason': '获取K线数据失败'
                        })
                    else:
                        # 获取列名映射
                        column_mapping = None
                        if hasattr(k_data, 'dtypes'):
                            # 检查是否有以col_开头的列，如果有，需要进行映射
                            cols = k_data.columns.tolist()
                            if any(col.startswith('col_') for col in cols):
                                # 映射规则
                                column_mapping = {
                                    'col_0': 'date',
                                    'col_1': 'open',
                                    'col_2': 'high',
                                    'col_3': 'low',
                                    'col_4': 'close',
                                    'col_5': 'volume'
                                }
                        
                        # 应用列名映射
                        if column_mapping:
                            logger.info(f"应用列名映射: {column_mapping}")
                            k_data = k_data.rename(columns=column_mapping)
                        
                        # 确保所有必需的列都存在
                        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                        missing_columns = [col for col in required_columns if col not in k_data.columns]
                        
                        if missing_columns:
                            logger.error(f"股票 {stock_code} 的K线数据缺少必需的列: {missing_columns}")
                            selected = False
                            verification_results.append({
                                'date': date,
                                'stock_code': stock_code,
                                'stock_name': stock_name,
                                'selected': selected,
                                'reason': f'K线数据缺少必需的列: {missing_columns}'
                            })
                        else:
                            logger.info(f"K线数据列: {k_data.columns.tolist()}")
                            
                            # 确保数据类型正确
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                if col in k_data.columns:
                                    k_data[col] = pd.to_numeric(k_data[col], errors='coerce')
                            
                            # 输出数据类型
                            logger.info(f"K线数据类型:\n{k_data.dtypes}")
                            
                            # 确保数据有索引并且索引是日期类型
                            if not isinstance(k_data.index, pd.DatetimeIndex) and 'date' in k_data.columns:
                                k_data = k_data.set_index('date')
                                k_data.index = pd.to_datetime(k_data.index)
                                
                            # 输出数据示例
                            logger.info(f"K线数据示例 (最后5行):\n{k_data.tail()}")
                            
                            # 输出策略条件详情
                            logger.info(f"策略条件详情:\n{strategy['conditions']}")
                            
                            # 评估股票是否满足策略条件
                            try:
                                evaluation_result = evaluator.evaluate_conditions(strategy['conditions'], k_data, stock_code)
                                
                                # 处理评估结果
                                if isinstance(evaluation_result, dict) and 'result' in evaluation_result:
                                    selected = evaluation_result['result']
                                    details = evaluation_result.get('details', {})
                                    
                                    # 输出通过和失败的指标
                                    passing_indicators = details.get('passing_indicators', [])
                                    failing_indicators = details.get('failing_indicators', [])
                                    
                                    if passing_indicators:
                                        logger.info(f"通过的指标 ({len(passing_indicators)}): {', '.join(passing_indicators)}")
                                    
                                    if failing_indicators:
                                        logger.info(f"未通过的指标 ({len(failing_indicators)}): {', '.join(failing_indicators)}")
                                    
                                    # 准备保存到验证结果中的详情
                                    detail_reason = "满足策略条件" if selected else "不满足策略条件"
                                    if passing_indicators:
                                        detail_reason += f"，满足指标: {', '.join(passing_indicators)}"
                                    if failing_indicators:
                                        detail_reason += f"，不满足指标: {', '.join(failing_indicators)}"
                                else:
                                    # 兼容旧的返回方式
                                    selected = bool(evaluation_result)
                                    detail_reason = "满足策略条件" if selected else "不满足策略条件"
                                
                                # 记录详细的评估过程和结果
                                for i, condition in enumerate(strategy['conditions']):
                                    if isinstance(condition, dict):
                                        if condition.get('type') == 'logic':
                                            logger.info(f"条件 {i+1}: 逻辑操作符 {condition.get('value')}")
                                        elif condition.get('type') == 'indicator':
                                            logger.info(f"条件 {i+1}: 指标 {condition.get('indicator_id')} - 参数: {condition.get('parameters', {})}")
                                            # 单独评估这个条件
                                            try:
                                                single_result = evaluator._evaluate_indicator_condition(condition, k_data, stock_code)
                                                logger.info(f"条件 {i+1} 评估结果: {single_result}")
                                            except Exception as e:
                                                logger.error(f"评估条件 {i+1} 时出错: {e}")
                            except Exception as e:
                                logger.error(f"评估股票 {stock_code} 是否满足策略条件时出错: {e}")
                                logger.error(traceback.format_exc())
                                selected = False
                                detail_reason = f"评估出错: {str(e)}"
                                
                            logger.info(f"股票 {stock_code} 是否满足策略条件: {selected}")
                            
                            if selected:
                                success_count += 1
                                date_success_count += 1
                                
                            verification_results.append({
                                'date': date,
                                'stock_code': stock_code,
                                'stock_name': stock_name,
                                'selected': selected,
                                'reason': detail_reason
                            })
                except Exception as e:
                    logger.error(f"验证股票 {stock_code} 时出错: {e}")
                    logger.error(traceback.format_exc())
                    selected = False
                    verification_results.append({
                        'date': date,
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'selected': selected,
                        'reason': f'处理异常: {str(e)}'
                    })
            
            # 输出当前日期的验证结果
            date_success_rate = date_success_count / len(group) * 100 if len(group) > 0 else 0
            logger.info(f"日期 {date} 验证结果: {date_success_count}/{len(group)} 成功，成功率 {date_success_rate:.2f}%")
                
        # 保存验证结果
        results_df = pd.DataFrame(verification_results)
        result_file = os.path.join(get_result_dir(), "complete_strategy_verification_report.csv")
        results_df.to_csv(result_file, index=False)
        logger.info(f"验证结果已保存到: {result_file}")
        
        # 统计成功率
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        logger.info(f"验证完成: 共 {total_count} 个买点，成功 {success_count} 个，成功率 {success_rate:.2f}%")
        
    except Exception as e:
        logger.error(f"验证过程中出错: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    verify_strategy() 