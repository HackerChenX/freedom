#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
优化版回测系统

基于增强版回测系统，优化性能和精度，支持并行计算和更复杂的交易策略评估
"""

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import argparse
import datetime
import pandas as pd
import numpy as np
import json
import time
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Callable
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from enums.period import Period
from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_stock_result_file, get_strategies_dir
from utils.period_manager import PeriodManager
from utils.period_data_structure import (
    IndicatorPeriodResult, 
    PeriodAnalysisResult, 
    MultiPeriodAnalysisResult
)
from utils.decorators import performance_monitor, time_it
from db.db_manager import DBManager
from indicators.factory import IndicatorFactory
from indicators.pattern_registry import PatternRegistry
from strategy.strategy_condition_evaluator import StrategyConditionEvaluator

# 获取日志记录器
logger = get_logger(__name__)


class OptimizedBacktest:
    """
    优化版回测系统
    
    特点:
    1. 基于增强版回测系统的全部功能
    2. 使用并行计算加速批量回测
    3. 支持更复杂的交易策略评估和优化
    4. 提供更全面的回测指标和可视化输出
    5. 支持动态调整参数和交易规则
    6. 引入资金管理和风险控制模型
    """
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None,
                cpu_cores: int = None):
        """
        初始化优化版回测系统
        
        Args:
            config: 回测配置
            cpu_cores: 并行计算的CPU核心数，默认为系统可用核心数的75%
        """
        self.db_manager = DBManager.get_instance()
        self.period_manager = PeriodManager.get_instance()
        self.result_dir = get_backtest_result_dir()
        self.strategies_dir = get_strategies_dir()
        
        # 确保目录存在
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.strategies_dir, exist_ok=True)
        
        # 存储分析结果
        self.analysis_results = []
        self.pattern_stats = defaultdict(int)
        
        # 指标工厂
        self.indicator_factory = IndicatorFactory()
        
        # 策略条件评估器
        self.condition_evaluator = StrategyConditionEvaluator()
        
        # 回测性能统计
        self.performance_stats = {
            "analysis_time": 0,
            "total_stocks": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "start_time": None,
            "end_time": None
        }
        
        # CPU核心数设置
        available_cores = multiprocessing.cpu_count()
        if cpu_cores is None:
            # 默认使用75%的可用核心
            self.cpu_cores = max(1, int(available_cores * 0.75))
        else:
            self.cpu_cores = min(cpu_cores, available_cores)
        
        logger.info(f"并行计算将使用 {self.cpu_cores} 个CPU核心（系统共有 {available_cores} 个）")
        
        # 合并配置
        self.default_config = {
            'days_before': 60,              # 分析买点前的天数
            'days_after': 20,               # 分析买点后的天数
            'periods': [                    # 分析周期
                Period.DAILY, 
                Period.MIN_60, 
                Period.MIN_30, 
                Period.MIN_15,
                Period.WEEKLY,
                Period.MONTHLY
            ],
            'indicators': {                 # 分析指标
                'MACD': {'enabled': True},
                'KDJ': {'enabled': True},
                'RSI': {'enabled': True},
                'BOLL': {'enabled': True},
                'MA': {'enabled': True},
                'VOL': {'enabled': True},
                'DMI': {'enabled': True},
                'ATR': {'enabled': True},
                'SAR': {'enabled': True},
                'CCI': {'enabled': True},
                'OBV': {'enabled': True},
                'WR': {'enabled': True}
            },
            'trading_rules': {              # 交易规则
                'entry': {
                    'min_score': 70,        # 入场最低得分
                    'max_spread': 0.02,     # 最大买入价差
                    'prefer_pattern': True  # 优先考虑形态信号
                },
                'exit': {
                    'take_profit': 0.15,    # 止盈比例
                    'stop_loss': 0.07,      # 止损比例
                    'trailing_stop': 0.05,  # 跟踪止损比例
                    'max_hold_days': 30     # 最大持有天数
                },
                'position_sizing': {
                    'method': 'fixed',      # 仓位管理方法
                    'size': 0.1,            # 每笔交易使用资金比例
                    'max_positions': 5      # 最大同时持有标的数
                }
            },
            'parallel': {
                'enabled': True,            # 是否启用并行计算
                'chunk_size': 10            # 每个进程处理的股票数量
            },
            'output': {
                'save_interim': False,      # 是否保存中间结果
                'detail_level': 'medium',   # 详细程度：low, medium, high
                'formats': ['json', 'excel'] # 输出格式
            }
        }
        
        if config is not None:
            self.config = self._merge_config(self.default_config, config)
        else:
            self.config = self.default_config
        
        logger.info("优化版回测系统初始化完成")
    
    def _merge_config(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归合并配置字典
        
        Args:
            default: 默认配置
            override: 要覆盖的配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        result = default.copy()
        
        for key, value in override.items():
            # 如果两个都是字典，递归合并
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                # 否则直接覆盖
                result[key] = value
                
        return result 

    @time_it
    @performance_monitor()
    def analyze_stock(self, code: str, buy_date: str, 
                     custom_config: Optional[Dict[str, Any]] = None) -> MultiPeriodAnalysisResult:
        """
        分析单个股票的买点
        
        Args:
            code: 股票代码
            buy_date: 买点日期，格式为YYYYMMDD
            custom_config: 自定义配置，会与基础配置合并
            
        Returns:
            MultiPeriodAnalysisResult: 多周期分析结果
        """
        try:
            # 合并配置
            if custom_config is None:
                config = self.config
            else:
                config = self._merge_config(self.config, custom_config)
            
            # 转换日期格式
            buy_date_obj = datetime.datetime.strptime(buy_date, "%Y%m%d")
            start_date = (buy_date_obj - datetime.timedelta(days=config['days_before'])).strftime("%Y%m%d")
            end_date = (buy_date_obj + datetime.timedelta(days=config['days_after'])).strftime("%Y%m%d")
            
            # 创建多周期分析结果对象
            result = MultiPeriodAnalysisResult(code, buy_date)
            
            # 记录配置
            result.metadata['config'] = config
            result.metadata['analysis_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 分析各个周期
            for period in config['periods']:
                period_result = self._analyze_period(code, buy_date, period, start_date, end_date, config)
                if period_result is not None:
                    result.add_period_result(period_result)
            
            # 添加到分析结果列表
            self.analysis_results.append(result)
            
            # 统计模式出现次数
            for period_result in result.periods.values():
                for pattern in period_result.get_all_patterns():
                    if 'pattern_id' in pattern:
                        self.pattern_stats[pattern['pattern_id']] += 1
            
            # 更新性能统计
            self.performance_stats["total_stocks"] += 1
            self.performance_stats["successful_analyses"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {code} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 更新性能统计
            self.performance_stats["total_stocks"] += 1
            self.performance_stats["failed_analyses"] += 1
            
            # 返回空结果
            empty_result = MultiPeriodAnalysisResult(code, buy_date)
            empty_result.metadata['error'] = str(e)
            return empty_result
    
    @time_it
    def _analyze_period(self, code: str, buy_date: str, period: Period, 
                       start_date: str, end_date: str, 
                       config: Dict[str, Any]) -> Optional[PeriodAnalysisResult]:
        """
        分析指定周期的技术指标
        
        Args:
            code: 股票代码
            buy_date: 买点日期
            period: 周期类型
            start_date: 开始日期
            end_date: 结束日期
            config: 分析配置
            
        Returns:
            Optional[PeriodAnalysisResult]: 周期分析结果
        """
        try:
            # 从周期管理器获取数据
            data = self.period_manager.get_data(code, period, start_date, end_date)
            
            if data is None or data.empty:
                logger.warning(f"未获取到股票 {code} 周期 {period.value} 的数据")
                return None
            
            # 找到买点日期对应的索引
            buy_date_obj = datetime.datetime.strptime(buy_date, "%Y%m%d").date()
            buy_index = None
            
            for i, date in enumerate(data['date']):
                if isinstance(date, str):
                    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                elif isinstance(date, pd.Timestamp):
                    date = date.date()
                
                if date >= buy_date_obj:
                    buy_index = i
                    break
            
            if buy_index is None:
                logger.warning(f"未找到股票 {code} 周期 {period.value} 买点日期 {buy_date} 的数据")
                return None
            
            # 创建周期分析结果对象
            period_result = PeriodAnalysisResult(period)
            
            # 记录买点信息
            period_result.metadata['buy_date'] = buy_date
            period_result.metadata['buy_index'] = buy_index
            period_result.metadata['buy_price'] = float(data.iloc[buy_index]['close'])
            
            # 计算买点前后的数据
            period_result.metadata['data_before_buy'] = len(data.iloc[:buy_index])
            period_result.metadata['data_after_buy'] = len(data.iloc[buy_index:])
            
            # 分析所有启用的指标
            for indicator_name, indicator_config in config['indicators'].items():
                if indicator_config.get('enabled', True):
                    # 使用工厂创建指标实例，并传递特定参数
                    indicator_params = indicator_config.get('parameters', {})
                    
                    indicator = self.indicator_factory.create(indicator_name, **indicator_params)
                    if indicator is None:
                        logger.warning(f"无法创建指标: {indicator_name}")
                        continue
                    
                    # 计算指标
                    try:
                        # 计算和分析指标
                        indicator_data = indicator.calculate(data)
                        
                        # 获取形态和评分
                        patterns = indicator.get_patterns(indicator_data)
                        score = indicator.calculate_score(indicator_data)
                        signals = indicator.generate_signals(indicator_data)
                        
                        # 创建指标周期结果
                        indicator_result = IndicatorPeriodResult(indicator_name, period)
                        indicator_result.set_data(indicator_data)
                        indicator_result.set_patterns(patterns)
                        indicator_result.set_score(score.get('final_score') if isinstance(score, dict) else score)
                        indicator_result.set_signals(signals)
                        
                        # 记录买点前后的信号
                        if buy_index is not None and 'buy_signal' in signals.columns:
                            signals_before = signals.iloc[:buy_index]
                            signals_after = signals.iloc[buy_index:]
                            
                            indicator_result.metadata['signals_before_buy'] = signals_before['buy_signal'].sum()
                            indicator_result.metadata['signals_after_buy'] = signals_after['buy_signal'].sum()
                        
                        # 添加到周期结果
                        period_result.add_indicator_result(indicator_result)
                        
                    except Exception as e:
                        logger.error(f"计算指标 {indicator_name} 时出错: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            # 计算周期综合得分
            period_result.calculate_composite_score()
            
            return period_result
            
        except Exception as e:
            logger.error(f"分析周期 {period.value} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @time_it
    @performance_monitor()
    def batch_analyze(self, input_file: str, output_file: str, 
                     custom_config: Optional[Dict[str, Any]] = None) -> List[MultiPeriodAnalysisResult]:
        """
        批量分析多个股票的买点
        
        Args:
            input_file: 输入文件路径，包含股票代码和买点日期
            output_file: 输出文件路径
            custom_config: 自定义配置，会与基础配置合并
            
        Returns:
            List[MultiPeriodAnalysisResult]: 分析结果列表
        """
        try:
            # 开始计时
            self.performance_stats["start_time"] = time.time()
            
            # 合并配置
            if custom_config is None:
                config = self.config
            else:
                config = self._merge_config(self.config, custom_config)
            
            # 读取输入文件
            buy_points = pd.read_csv(input_file)
            required_columns = ['code', 'buy_date']
            
            for col in required_columns:
                if col not in buy_points.columns:
                    raise ValueError(f"输入文件缺少必要的列: {col}")
            
            total = len(buy_points)
            logger.info(f"开始批量分析 {total} 个买点")
            
            # 整理输入数据
            stock_data = []
            for _, row in buy_points.iterrows():
                # 确保日期格式正确
                buy_date = str(row['buy_date'])
                if len(buy_date) == 8:
                    pass  # 已经是YYYYMMDD格式
                elif '-' in buy_date:
                    buy_date = buy_date.replace('-', '')
                
                stock_data.append((row['code'], buy_date))
            
            # 判断是否使用并行计算
            results = []
            if config['parallel']['enabled'] and total > 1:
                # 使用并行计算
                chunk_size = config['parallel']['chunk_size']
                logger.info(f"使用并行计算，CPU核心数: {self.cpu_cores}, 分块大小: {chunk_size}")
                
                # 创建进程池
                with multiprocessing.Pool(processes=self.cpu_cores) as pool:
                    # 将任务分块提交给进程池
                    tasks = [(code, buy_date, config) for code, buy_date in stock_data]
                    parallel_results = pool.starmap(self._parallel_analyze_stock, tasks, chunksize=chunk_size)
                    
                    # 收集结果
                    results = [r for r in parallel_results if r is not None]
            else:
                # 使用串行计算
                logger.info(f"使用串行计算")
                
                for i, (code, buy_date) in enumerate(stock_data):
                    logger.info(f"分析进度: {i+1}/{total} - 股票: {code}, 买点日期: {buy_date}")
                    
                    # 调用分析函数
                    result = self.analyze_stock(code, buy_date, config)
                    results.append(result)
                    
                    # 保存中间结果
                    if config['output']['save_interim'] and (i+1) % 10 == 0:
                        interim_file = f"{output_file}.interim_{i+1}"
                        self.save_results(results, interim_file)
                        logger.info(f"已保存中间结果: {interim_file}")
            
            # 保存最终结果
            self.save_results(results, output_file)
            
            # 打印形态统计
            self._print_pattern_stats()
            
            # 结束计时
            self.performance_stats["end_time"] = time.time()
            self.performance_stats["analysis_time"] = self.performance_stats["end_time"] - self.performance_stats["start_time"]
            
            # 打印性能统计
            logger.info(f"批量分析完成，用时: {self.performance_stats['analysis_time']:.2f} 秒")
            logger.info(f"分析成功: {self.performance_stats['successful_analyses']}/{total}, "
                      f"失败: {self.performance_stats['failed_analyses']}/{total}")
            
            return results
            
        except Exception as e:
            logger.error(f"批量分析出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 结束计时
            self.performance_stats["end_time"] = time.time()
            self.performance_stats["analysis_time"] = self.performance_stats["end_time"] - self.performance_stats["start_time"]
            
            return []
    
    def _parallel_analyze_stock(self, code: str, buy_date: str, 
                              config: Dict[str, Any]) -> Optional[MultiPeriodAnalysisResult]:
        """
        并行分析单个股票的包装函数
        
        Args:
            code: 股票代码
            buy_date: 买点日期
            config: 分析配置
            
        Returns:
            Optional[MultiPeriodAnalysisResult]: 分析结果
        """
        try:
            # 创建独立的回测实例以避免多进程共享问题
            backtest = OptimizedBacktest(config)
            result = backtest.analyze_stock(code, buy_date)
            
            # 更新全局统计信息（这部分在主进程中汇总）
            if result:
                logger.info(f"并行分析完成: {code}, {buy_date}")
                return result
            return None
        except Exception as e:
            logger.error(f"并行分析股票 {code} 时出错: {e}")
            return None
    
    @time_it
    def save_results(self, results: List[MultiPeriodAnalysisResult], output_file: str) -> None:
        """
        保存分析结果到文件
        
        Args:
            results: 分析结果列表
            output_file: 输出文件路径
        """
        try:
            # 转换为可序列化的字典
            results_dict = {
                'stocks': [result.to_dict() for result in results],
                'pattern_stats': dict(self.pattern_stats),
                'performance_stats': self.performance_stats,
                'config': self.config,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 获取文件扩展名
            base_name, ext = os.path.splitext(output_file)
            if not ext:
                ext = '.json'  # 默认使用json格式
            
            # 根据配置选择输出格式
            formats = self.config['output'].get('formats', ['json'])
            
            # 保存JSON格式
            if 'json' in formats:
                json_file = f"{base_name}.json" if ext != '.json' else output_file
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"分析结果已保存到JSON文件: {json_file}")
            
            # 保存Excel格式
            if 'excel' in formats:
                excel_file = f"{base_name}.xlsx" if ext != '.xlsx' else output_file
                self._save_to_excel(results, excel_file)
                logger.info(f"分析结果已保存到Excel文件: {excel_file}")
            
            # 保存CSV格式
            if 'csv' in formats:
                csv_file = f"{base_name}.csv" if ext != '.csv' else output_file
                self._save_to_csv(results, csv_file)
                logger.info(f"分析结果已保存到CSV文件: {csv_file}")
            
        except Exception as e:
            logger.error(f"保存分析结果时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _save_to_excel(self, results: List[MultiPeriodAnalysisResult], output_file: str) -> None:
        """
        将结果保存为Excel格式
        
        Args:
            results: 分析结果列表
            output_file: 输出文件路径
        """
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 摘要表
                summary_data = []
                for result in results:
                    summary = {
                        'code': result.stock_code,
                        'buy_date': result.buy_date,
                        'periods_analyzed': len(result.periods)
                    }
                    
                    # 添加各周期综合得分
                    for period, period_result in result.periods.items():
                        summary[f'{period.value}_score'] = period_result.get_composite_score()
                    
                    # 添加整体评价
                    summary['overall_score'] = result.get_overall_score()
                    
                    summary_data.append(summary)
                
                # 创建摘要DataFrame并保存
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 形态统计表
                pattern_df = pd.DataFrame([
                    {'pattern_id': pattern_id, 'count': count}
                    for pattern_id, count in self.pattern_stats.items()
                ])
                if not pattern_df.empty:
                    pattern_df = pattern_df.sort_values('count', ascending=False)
                    pattern_df.to_excel(writer, sheet_name='Patterns', index=False)
                
                # 性能统计表
                perf_df = pd.DataFrame([self.performance_stats])
                perf_df.to_excel(writer, sheet_name='Performance', index=False)
                
                # 详细指标表
                if self.config['output']['detail_level'] in ['medium', 'high']:
                    indicator_data = []
                    for result in results:
                        for period, period_result in result.periods.items():
                            for indicator_name, indicator_result in period_result.indicators.items():
                                indicator_info = {
                                    'code': result.stock_code,
                                    'buy_date': result.buy_date,
                                    'period': period.value,
                                    'indicator': indicator_name,
                                    'score': indicator_result.score
                                }
                                
                                # 添加形态信息
                                patterns = indicator_result.get_patterns()
                                if patterns:
                                    indicator_info['patterns'] = ', '.join([p.get('pattern_id', '') for p in patterns])
                                
                                indicator_data.append(indicator_info)
                    
                    if indicator_data:
                        indicators_df = pd.DataFrame(indicator_data)
                        indicators_df.to_excel(writer, sheet_name='Indicators', index=False)
                
                # 超详细信息
                if self.config['output']['detail_level'] == 'high':
                    # 为每个股票创建单独的工作表
                    for result in results[:20]:  # 限制为前20个，避免文件过大
                        sheet_name = f"{result.stock_code}_{result.buy_date}"
                        # Excel工作表名称不能超过31个字符
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        
                        # 创建股票详情数据
                        detail_data = []
                        for period, period_result in result.periods.items():
                            for indicator_name, indicator_result in period_result.indicators.items():
                                detail = {
                                    'period': period.value,
                                    'indicator': indicator_name,
                                    'score': indicator_result.score
                                }
                                
                                # 添加形态信息
                                patterns = indicator_result.get_patterns()
                                for i, pattern in enumerate(patterns):
                                    detail[f'pattern_{i+1}'] = pattern.get('pattern_id', '')
                                    detail[f'pattern_{i+1}_confidence'] = pattern.get('confidence', 0)
                                
                                detail_data.append(detail)
                        
                        if detail_data:
                            detail_df = pd.DataFrame(detail_data)
                            detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Excel文件已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"保存Excel文件时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _save_to_csv(self, results: List[MultiPeriodAnalysisResult], output_file: str) -> None:
        """
        将结果保存为CSV格式
        
        Args:
            results: 分析结果列表
            output_file: 输出文件路径
        """
        try:
            # 创建摘要数据
            summary_data = []
            for result in results:
                summary = {
                    'code': result.stock_code,
                    'buy_date': result.buy_date,
                    'periods_analyzed': len(result.periods)
                }
                
                # 添加各周期综合得分
                for period, period_result in result.periods.items():
                    summary[f'{period.value}_score'] = period_result.get_composite_score()
                
                # 添加整体评价
                summary['overall_score'] = result.get_overall_score()
                
                summary_data.append(summary)
            
            # 创建DataFrame并保存
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(output_file, index=False)
                logger.info(f"CSV文件已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"保存CSV文件时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    @time_it
    def _print_pattern_stats(self) -> None:
        """打印形态统计信息"""
        logger.info("形态统计结果:")
        
        # 按照出现次数排序
        sorted_patterns = sorted(self.pattern_stats.items(), key=lambda x: x[1], reverse=True)
        
        # 获取形态显示名称
        pattern_registry = PatternRegistry()
        
        for pattern_id, count in sorted_patterns[:20]:  # 只显示前20个
            display_name = pattern_registry.get_display_name(pattern_id)
            logger.info(f"{display_name} ({pattern_id}): {count}次")
        
        total_patterns = len(self.pattern_stats)
        if total_patterns > 20:
            logger.info(f"... 还有 {total_patterns - 20} 个形态未显示 ...")
    
    @time_it
    @performance_monitor()
    def generate_strategy(self, results: List[MultiPeriodAnalysisResult], 
                         output_file: str, threshold: int = 2) -> Dict[str, Any]:
        """
        根据分析结果生成选股策略
        
        Args:
            results: 分析结果列表
            output_file: 输出策略文件路径
            threshold: 形态出现次数阈值，默认为2
            
        Returns:
            Dict[str, Any]: 生成的策略配置
        """
        try:
            # 统计形态出现情况
            pattern_stats = defaultdict(int)
            period_patterns = defaultdict(lambda: defaultdict(int))
            indicator_patterns = defaultdict(lambda: defaultdict(int))
            
            # 收集所有周期、所有指标的形态
            for result in results:
                for period, period_result in result.periods.items():
                    for indicator_name, indicator_result in period_result.indicators.items():
                        for pattern in indicator_result.get_patterns():
                            if 'pattern_id' in pattern:
                                pattern_id = pattern['pattern_id']
                                pattern_stats[pattern_id] += 1
                                period_patterns[period.value][pattern_id] += 1
                                indicator_patterns[indicator_name][pattern_id] += 1
            
            # 提取高频形态
            frequent_patterns = {pattern_id: count for pattern_id, count in pattern_stats.items() 
                               if count >= threshold}
            
            # 获取形态元数据
            pattern_registry = PatternRegistry()
            
            # 构建策略对象
            strategy = {
                'name': f"优化策略_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'description': f"基于{len(results)}个买点样本自动生成的优化策略",
                'conditions': [],
                'period_conditions': {},
                'period_weights': {},
                'indicator_weights': {},
                'metadata': {
                    'generated_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sample_count': len(results),
                    'pattern_threshold': threshold,
                    'top_patterns': {k: v for k, v in sorted(frequent_patterns.items(), 
                                                          key=lambda item: item[1], reverse=True)[:10]}
                }
            }
            
            # 为每个周期生成条件
            for period_value, patterns in period_patterns.items():
                # 筛选高频形态
                period_frequent_patterns = {pattern_id: count for pattern_id, count in patterns.items() 
                                         if pattern_id in frequent_patterns}
                
                if not period_frequent_patterns:
                    continue
                
                # 按照指标分组形态
                period_indicator_patterns = defaultdict(list)
                for pattern_id in period_frequent_patterns:
                    # 从形态ID提取指标名称
                    parts = pattern_id.split('_')
                    if len(parts) > 0:
                        indicator_name = parts[0]
                        period_indicator_patterns[indicator_name].append(pattern_id)
                
                # 生成周期条件
                conditions = []
                for indicator_name, pattern_ids in period_indicator_patterns.items():
                    # 提取形态信息
                    pattern_infos = []
                    for pattern_id in pattern_ids:
                        pattern_info = pattern_registry.get_pattern_info(pattern_id)
                        if pattern_info:
                            pattern_infos.append({
                                'id': pattern_id,
                                'display_name': pattern_info.get('display_name', pattern_id),
                                'score_impact': pattern_info.get('score_impact', 0),
                                'count': period_frequent_patterns[pattern_id],
                                'confidence': min(1.0, period_frequent_patterns[pattern_id] / len(results))
                            })
                    
                    # 生成指标条件
                    if pattern_infos:
                        condition = {
                            'type': 'indicator',
                            'indicator_id': indicator_name,
                            'indicator': {'id': indicator_name},
                            'period': period_value,
                            'patterns': pattern_infos,
                            'logic': 'OR',  # 默认使用OR逻辑
                            'weight': sum(info['count'] for info in pattern_infos) / sum(period_frequent_patterns.values())
                        }
                        conditions.append(condition)
                
                # 如果有条件，添加到策略中
                if conditions:
                    strategy['period_conditions'][period_value] = conditions
                    # 设置周期权重
                    strategy['period_weights'][period_value] = sum(period_frequent_patterns.values()) / sum(frequent_patterns.values())
            
            # 生成全局条件
            # 先创建逻辑组
            logic_groups = []
            for period_value, conditions in strategy['period_conditions'].items():
                if conditions:
                    # 创建周期组，默认使用OR逻辑连接该周期内的所有条件
                    period_group = {
                        'type': 'logic_group',
                        'conditions': conditions,
                        'logic': 'OR',
                        'period': period_value,
                        'weight': strategy['period_weights'].get(period_value, 1.0)
                    }
                    logic_groups.append(period_group)
            
            # 将不同周期的条件组合起来，默认使用AND逻辑
            if len(logic_groups) > 1:
                combined_condition = {
                    'type': 'combined_condition',
                    'groups': logic_groups,
                    'logic': 'AND',  # 不同周期之间使用AND逻辑
                    'weight': 1.0
                }
                strategy['conditions'].append(combined_condition)
            elif len(logic_groups) == 1:
                # 只有一个周期，直接使用该周期的条件
                strategy['conditions'] = logic_groups[0]['conditions']
            
            # 计算指标权重
            total_pattern_count = sum(count for count in pattern_stats.values())
            for indicator_name, patterns in indicator_patterns.items():
                indicator_pattern_count = sum(count for count in patterns.values())
                if total_pattern_count > 0:
                    strategy['indicator_weights'][indicator_name] = indicator_pattern_count / total_pattern_count
            
            # 保存策略
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(strategy, f, ensure_ascii=False, indent=2)
            
            logger.info(f"选股策略已生成并保存到: {output_file}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"生成策略时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    @time_it
    @performance_monitor()
    def backtest_strategy(self, strategy: Dict[str, Any], 
                         stock_pool: List[str], 
                         start_date: str, 
                         end_date: str,
                         output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        回测策略性能
        
        Args:
            strategy: 策略配置
            stock_pool: 股票池
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            output_file: 可选的输出文件路径
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        try:
            logger.info(f"开始回测策略: {strategy.get('name', 'unnamed')}")
            logger.info(f"股票池大小: {len(stock_pool)}, 时间范围: {start_date} - {end_date}")
            
            # 初始化回测结果
            backtest_result = {
                'strategy_name': strategy.get('name', 'unnamed'),
                'start_date': start_date,
                'end_date': end_date,
                'stock_pool_size': len(stock_pool),
                'selected_stocks': [],
                'trades': [],
                'performance': {
                    'win_rate': 0.0,
                    'profit_loss_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'annual_return': 0.0,
                    'total_return': 0.0,
                    'trade_count': 0
                },
                'dates': []
            }
            
            # 交易规则
            trading_rules = self.config.get('trading_rules', {})
            
            # 获取日期列表
            start_date_obj = datetime.datetime.strptime(start_date, "%Y%m%d")
            end_date_obj = datetime.datetime.strptime(end_date, "%Y%m%d")
            date_range = []
            
            current_date = start_date_obj
            while current_date <= end_date_obj:
                date_range.append(current_date.strftime("%Y%m%d"))
                current_date += datetime.timedelta(days=1)
            
            # 回测每一天
            portfolio = {}  # 当前持仓
            cash = 1000000.0  # 初始资金
            daily_value = []  # 每日资产价值
            
            for date in date_range:
                # 获取当天选出的股票
                selected_stocks = self._select_stocks_by_strategy(strategy, stock_pool, date)
                
                # 更新回测结果
                backtest_result['dates'].append(date)
                
                # 更新持仓并模拟交易
                if selected_stocks:
                    # 处理买入信号
                    for stock_code in selected_stocks:
                        # 如果不在持仓中，考虑买入
                        if stock_code not in portfolio and cash > 0:
                            # 获取股票价格
                            price = self._get_stock_price(stock_code, date)
                            if price > 0:
                                # 计算买入数量
                                position_size = min(trading_rules.get('position_sizing', {}).get('size', 0.1), 1.0)
                                amount = cash * position_size
                                shares = int(amount / price)
                                
                                if shares > 0:
                                    # 记录交易
                                    trade = {
                                        'type': 'buy',
                                        'stock_code': stock_code,
                                        'date': date,
                                        'price': price,
                                        'shares': shares,
                                        'amount': shares * price
                                    }
                                    backtest_result['trades'].append(trade)
                                    
                                    # 更新持仓和现金
                                    portfolio[stock_code] = {
                                        'shares': shares,
                                        'buy_price': price,
                                        'buy_date': date,
                                        'current_price': price,
                                        'current_value': shares * price
                                    }
                                    cash -= shares * price
                    
                # 更新现有持仓价格和检查卖出条件
                stocks_to_sell = []
                for stock_code, position in portfolio.items():
                    # 更新价格
                    current_price = self._get_stock_price(stock_code, date)
                    if current_price > 0:
                        position['current_price'] = current_price
                        position['current_value'] = position['shares'] * current_price
                        
                        # 检查止盈止损条件
                        buy_price = position['buy_price']
                        profit_pct = (current_price - buy_price) / buy_price
                        
                        take_profit = trading_rules.get('exit', {}).get('take_profit', 0.15)
                        stop_loss = trading_rules.get('exit', {}).get('stop_loss', 0.07)
                        
                        if profit_pct >= take_profit or profit_pct <= -stop_loss:
                            stocks_to_sell.append(stock_code)
                        
                        # 检查最大持有时间
                        buy_date_obj = datetime.datetime.strptime(position['buy_date'], "%Y%m%d")
                        current_date_obj = datetime.datetime.strptime(date, "%Y%m%d")
                        hold_days = (current_date_obj - buy_date_obj).days
                        
                        max_hold_days = trading_rules.get('exit', {}).get('max_hold_days', 30)
                        if hold_days >= max_hold_days:
                            stocks_to_sell.append(stock_code)
                
                # 执行卖出
                for stock_code in stocks_to_sell:
                    position = portfolio[stock_code]
                    price = position['current_price']
                    shares = position['shares']
                    
                    # 记录交易
                    trade = {
                        'type': 'sell',
                        'stock_code': stock_code,
                        'date': date,
                        'price': price,
                        'shares': shares,
                        'amount': shares * price,
                        'profit': shares * (price - position['buy_price'])
                    }
                    backtest_result['trades'].append(trade)
                    
                    # 更新现金
                    cash += shares * price
                    
                    # 从持仓中移除
                    del portfolio[stock_code]
                
                # 计算当日总资产价值
                portfolio_value = sum(p['current_value'] for p in portfolio.values())
                total_value = cash + portfolio_value
                daily_value.append(total_value)
            
            # 计算回测性能指标
            if backtest_result['trades']:
                # 计算胜率
                winning_trades = [t for t in backtest_result['trades'] 
                                if t['type'] == 'sell' and t['profit'] > 0]
                losing_trades = [t for t in backtest_result['trades'] 
                               if t['type'] == 'sell' and t['profit'] <= 0]
                
                trade_count = len(winning_trades) + len(losing_trades)
                win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0
                
                # 计算盈亏比
                avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
                avg_loss = sum(abs(t['profit']) for t in losing_trades) / len(losing_trades) if losing_trades else 1
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                
                # 计算最大回撤
                max_drawdown = 0
                peak = daily_value[0]
                for value in daily_value:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                # 计算夏普比率和年化收益率
                returns = []
                for i in range(1, len(daily_value)):
                    returns.append((daily_value[i] - daily_value[i-1]) / daily_value[i-1])
                
                if returns:
                    avg_return = sum(returns) / len(returns)
                    std_return = np.std(returns) if len(returns) > 1 else 0.01
                    sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
                    
                    # 总收益和年化收益
                    total_days = len(daily_value)
                    total_return = (daily_value[-1] - daily_value[0]) / daily_value[0]
                    annual_return = (1 + total_return) ** (252 / total_days) - 1 if total_days > 0 else 0
                else:
                    sharpe_ratio = 0
                    total_return = 0
                    annual_return = 0
                
                # 更新性能指标
                backtest_result['performance'] = {
                    'win_rate': win_rate,
                    'profit_loss_ratio': profit_loss_ratio,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'annual_return': annual_return,
                    'total_return': total_return,
                    'trade_count': trade_count,
                    'final_value': daily_value[-1] if daily_value else 0,
                    'daily_returns': returns
                }
            
            # 保存回测结果
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(backtest_result, f, ensure_ascii=False, indent=2)
                logger.info(f"回测结果已保存到: {output_file}")
            
            # 打印回测摘要
            self._print_backtest_summary(backtest_result)
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"回测策略时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'strategy_name': strategy.get('name', 'unnamed'),
                'start_date': start_date,
                'end_date': end_date
            }
    
    def _select_stocks_by_strategy(self, strategy: Dict[str, Any], 
                                 stock_pool: List[str], 
                                 date: str) -> List[str]:
        """
        根据策略选择股票
        
        Args:
            strategy: 策略配置
            stock_pool: 股票池
            date: 日期
            
        Returns:
            List[str]: 选中的股票列表
        """
        try:
            selected_stocks = []
            
            # 获取周期条件
            period_conditions = strategy.get('period_conditions', {})
            
            # 如果策略中直接包含条件，使用这些条件
            if not period_conditions and 'conditions' in strategy:
                conditions = strategy['conditions']
                
                # 遍历股票池
                for stock_code in stock_pool:
                    # 获取股票数据
                    data = self._get_stock_data(stock_code, date, Period.DAILY)
                    if data is None or data.empty:
                        continue
                    
                    # 评估条件
                    result = self.condition_evaluator.evaluate_conditions(conditions, data, stock_code)
                    
                    # 如果满足条件，添加到选中列表
                    if isinstance(result, dict) and result.get('result', False):
                        selected_stocks.append(stock_code)
            else:
                # 使用周期条件
                for period_value, conditions in period_conditions.items():
                    # 确定对应的周期类型
                    try:
                        period = Period(period_value)
                    except ValueError:
                        logger.warning(f"无效的周期值: {period_value}")
                        continue
                    
                    # 遍历股票池
                    for stock_code in stock_pool:
                        if stock_code in selected_stocks:
                            continue  # 已经选中，跳过
                        
                        # 获取股票数据
                        data = self._get_stock_data(stock_code, date, period)
                        if data is None or data.empty:
                            continue
                        
                        # 评估条件
                        result = self.condition_evaluator.evaluate_conditions(conditions, data, stock_code)
                        
                        # 如果满足条件，添加到选中列表
                        if isinstance(result, dict) and result.get('result', False):
                            selected_stocks.append(stock_code)
            
            return selected_stocks
            
        except Exception as e:
            logger.error(f"选择股票时出错: {e}")
            return []
    
    def _get_stock_data(self, stock_code: str, date: str, period: Period) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            date: 日期
            period: 周期类型
            
        Returns:
            Optional[pd.DataFrame]: 股票数据
        """
        try:
            # 计算开始日期（回溯足够的天数以获取足够的数据）
            date_obj = datetime.datetime.strptime(date, "%Y%m%d")
            start_date = (date_obj - datetime.timedelta(days=60)).strftime("%Y%m%d")
            
            # 从周期管理器获取数据
            data = self.period_manager.get_data(stock_code, period, start_date, date)
            return data
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据时出错: {e}")
            return None
    
    def _get_stock_price(self, stock_code: str, date: str) -> float:
        """
        获取股票价格
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            float: 股票价格，如果获取失败则返回0
        """
        try:
            # 获取日K数据
            data = self._get_stock_data(stock_code, date, Period.DAILY)
            if data is None or data.empty:
                return 0
            
            # 查找最近的收盘价
            date_obj = datetime.datetime.strptime(date, "%Y%m%d").date()
            closest_date = None
            min_days_diff = float('inf')
            
            for i, row_date in enumerate(data['date']):
                if isinstance(row_date, str):
                    row_date = datetime.datetime.strptime(row_date, "%Y-%m-%d").date()
                elif isinstance(row_date, pd.Timestamp):
                    row_date = row_date.date()
                
                days_diff = abs((row_date - date_obj).days)
                if days_diff < min_days_diff:
                    min_days_diff = days_diff
                    closest_date = i
            
            if closest_date is not None:
                return float(data.iloc[closest_date]['close'])
            
            return 0
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 价格时出错: {e}")
            return 0
    
    def _print_backtest_summary(self, result: Dict[str, Any]) -> None:
        """
        打印回测摘要
        
        Args:
            result: 回测结果
        """
        logger.info("=" * 50)
        logger.info(f"策略: {result.get('strategy_name', 'unnamed')}")
        logger.info(f"回测期间: {result.get('start_date', '')} - {result.get('end_date', '')}")
        logger.info(f"股票池大小: {result.get('stock_pool_size', 0)}")
        
        performance = result.get('performance', {})
        logger.info("-" * 50)
        logger.info(f"总收益率: {performance.get('total_return', 0) * 100:.2f}%")
        logger.info(f"年化收益率: {performance.get('annual_return', 0) * 100:.2f}%")
        logger.info(f"最大回撤: {performance.get('max_drawdown', 0) * 100:.2f}%")
        logger.info(f"夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
        logger.info(f"胜率: {performance.get('win_rate', 0) * 100:.2f}%")
        logger.info(f"盈亏比: {performance.get('profit_loss_ratio', 0):.2f}")
        logger.info(f"交易次数: {performance.get('trade_count', 0)}")
        logger.info("=" * 50)
    
    @staticmethod
    def get_instance() -> 'OptimizedBacktest':
        """获取单例实例"""
        if not hasattr(OptimizedBacktest, '_instance'):
            OptimizedBacktest._instance = OptimizedBacktest()
        return OptimizedBacktest._instance 