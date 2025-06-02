#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
增强版回测系统

基于统一回测系统，使用周期管理器和周期分离数据结构
解决周期与指标混淆问题，提供更精细的配置和结果管理
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
from typing import List, Dict, Any, Optional, Tuple, Union, Set
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
from db.db_manager import DBManager
from indicators.factory import IndicatorFactory
from indicators.pattern_registry import PatternRegistry

# 获取日志记录器
logger = get_logger(__name__)


class EnhancedBacktest:
    """
    增强版回测系统
    
    特点:
    1. 使用周期管理器获取和管理不同周期的数据
    2. 使用周期分离数据结构存储分析结果
    3. 严格区分不同周期的同名指标
    4. 提供更精细的配置选项和结果管理
    """
    
    def __init__(self):
        """初始化增强版回测系统"""
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
        
        # 默认配置
        self.default_config = {
            'days_before': 20,
            'days_after': 10,
            'periods': [
                Period.DAILY, 
                Period.MIN_60, 
                Period.MIN_30, 
                Period.MIN_15,
                Period.WEEKLY,
                Period.MONTHLY
            ],
            'indicators': {
                'MACD': {'enabled': True},
                'KDJ': {'enabled': True},
                'RSI': {'enabled': True},
                'BOLL': {'enabled': True},
                'MA': {'enabled': True},
                'VOL': {'enabled': True},
                'DMI': {'enabled': True},
                'ATR': {'enabled': True},
                'SAR': {'enabled': True}
            }
        }
        
        logger.info("增强版回测系统初始化完成")
    
    def analyze_stock(self, code: str, buy_date: str, 
                     config: Optional[Dict[str, Any]] = None) -> MultiPeriodAnalysisResult:
        """
        分析单个股票的买点
        
        Args:
            code: 股票代码
            buy_date: 买点日期，格式为YYYYMMDD
            config: 分析配置，如不提供则使用默认配置
            
        Returns:
            MultiPeriodAnalysisResult: 多周期分析结果
        """
        try:
            # 合并配置
            if config is None:
                config = self.default_config
            else:
                # 确保配置完整
                for key, value in self.default_config.items():
                    if key not in config:
                        config[key] = value
            
            # 转换日期格式
            buy_date_obj = datetime.datetime.strptime(buy_date, "%Y%m%d")
            end_date = (buy_date_obj + datetime.timedelta(days=config['days_after'])).strftime("%Y%m%d")
            
            # 创建多周期分析结果对象
            result = MultiPeriodAnalysisResult(code, buy_date)
            
            # 分析各个周期
            for period in config['periods']:
                period_result = self._analyze_period(code, buy_date, period, end_date, config)
                if period_result is not None:
                    result.add_period_result(period_result)
            
            # 添加到分析结果列表
            self.analysis_results.append(result)
            
            # 统计模式出现次数
            for period_result in result.periods.values():
                for pattern in period_result.get_all_patterns():
                    if 'pattern_id' in pattern:
                        self.pattern_stats[pattern['pattern_id']] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {code} 时出错: {e}")
            return MultiPeriodAnalysisResult(code, buy_date)
    
    def _analyze_period(self, code: str, buy_date: str, period: Period, 
                       end_date: str, config: Dict[str, Any]) -> Optional[PeriodAnalysisResult]:
        """
        分析指定周期的技术指标
        
        Args:
            code: 股票代码
            buy_date: 买点日期
            period: 周期类型
            end_date: 结束日期
            config: 分析配置
            
        Returns:
            Optional[PeriodAnalysisResult]: 周期分析结果
        """
        try:
            # 从周期管理器获取数据
            data = self.period_manager.get_data(code, period, end_date)
            
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
            
            # 分析所有启用的指标
            for indicator_name, indicator_config in config['indicators'].items():
                if indicator_config.get('enabled', True):
                    # 使用工厂创建指标实例
                    indicator = self.indicator_factory.create_indicator(indicator_name)
                    if indicator is None:
                        logger.warning(f"无法创建指标: {indicator_name}")
                        continue
                    
                    # 计算指标
                    try:
                        indicator_data = indicator.calculate(data)
                        
                        # 获取形态和评分
                        patterns = indicator.get_patterns(indicator_data)
                        score = indicator.calculate_score(indicator_data)
                        signals = indicator.generate_trading_signals(indicator_data)
                        
                        # 创建指标周期结果
                        indicator_result = IndicatorPeriodResult(indicator_name, period)
                        indicator_result.set_data(indicator_data)
                        indicator_result.set_patterns(patterns)
                        indicator_result.set_score(score.get('final_score') if isinstance(score, dict) else score)
                        indicator_result.set_signals(signals)
                        
                        # 添加到周期结果
                        period_result.add_indicator_result(indicator_result)
                        
                    except Exception as e:
                        logger.error(f"计算指标 {indicator_name} 时出错: {e}")
            
            return period_result
            
        except Exception as e:
            logger.error(f"分析周期 {period.value} 时出错: {e}")
            return None
    
    def batch_analyze(self, input_file: str, output_file: str, 
                     config: Optional[Dict[str, Any]] = None) -> List[MultiPeriodAnalysisResult]:
        """
        批量分析多个股票的买点
        
        Args:
            input_file: 输入文件路径，包含股票代码和买点日期
            output_file: 输出文件路径
            config: 分析配置，如不提供则使用默认配置
            
        Returns:
            List[MultiPeriodAnalysisResult]: 分析结果列表
        """
        try:
            # 读取输入文件
            buy_points = pd.read_csv(input_file)
            required_columns = ['code', 'buy_date']
            
            for col in required_columns:
                if col not in buy_points.columns:
                    raise ValueError(f"输入文件缺少必要的列: {col}")
            
            results = []
            total = len(buy_points)
            
            # 批量分析
            for i, row in buy_points.iterrows():
                logger.info(f"分析进度: {i+1}/{total} - 股票: {row['code']}, 买点日期: {row['buy_date']}")
                
                # 确保日期格式正确
                buy_date = str(row['buy_date'])
                if len(buy_date) == 8:
                    pass  # 已经是YYYYMMDD格式
                elif '-' in buy_date:
                    buy_date = buy_date.replace('-', '')
                
                # 调用分析函数
                result = self.analyze_stock(row['code'], buy_date, config)
                results.append(result)
            
            # 保存结果
            self.save_results(results, output_file)
            
            # 打印形态统计
            self._print_pattern_stats()
            
            return results
            
        except Exception as e:
            logger.error(f"批量分析出错: {e}")
            return []
    
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
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 将结果保存为JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存分析结果时出错: {e}")
    
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
            
            for result in results:
                for period, period_result in result.periods.items():
                    for pattern in period_result.get_all_patterns():
                        if 'pattern_id' in pattern:
                            pattern_id = pattern['pattern_id']
                            pattern_stats[pattern_id] += 1
                            period_patterns[period.value][pattern_id] += 1
            
            # 提取高频形态
            frequent_patterns = {pattern_id: count for pattern_id, count in pattern_stats.items() 
                               if count >= threshold}
            
            # 按照周期组织策略条件
            strategy = {
                'name': f"回测生成策略_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'description': f"基于{len(results)}个买点样本自动生成的策略",
                'period_conditions': {},
                'period_weights': {},
                'metadata': {
                    'generated_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sample_count': len(results),
                    'pattern_threshold': threshold,
                    'top_patterns': {k: v for k, v in sorted(frequent_patterns.items(), 
                                                          key=lambda item: item[1], reverse=True)[:10]}
                }
            }
            
            # 获取形态元数据
            pattern_registry = PatternRegistry()
            
            # 为每个周期生成条件
            for period_value, patterns in period_patterns.items():
                # 筛选高频形态
                period_frequent_patterns = {pattern_id: count for pattern_id, count in patterns.items() 
                                         if pattern_id in frequent_patterns}
                
                if not period_frequent_patterns:
                    continue
                
                # 按照指标分组形态
                indicator_patterns = defaultdict(list)
                for pattern_id in period_frequent_patterns:
                    # 从形态ID提取指标名称
                    parts = pattern_id.split('_')
                    if len(parts) > 0:
                        indicator_name = parts[0]
                        indicator_patterns[indicator_name].append(pattern_id)
                
                # 生成周期条件
                conditions = []
                for indicator_name, pattern_ids in indicator_patterns.items():
                    # 提取形态信息
                    pattern_infos = []
                    for pattern_id in pattern_ids:
                        pattern_info = pattern_registry.get_pattern_info(pattern_id)
                        if pattern_info:
                            pattern_infos.append({
                                'id': pattern_id,
                                'display_name': pattern_info.get('display_name', pattern_id),
                                'score_impact': pattern_info.get('score_impact', 0),
                                'count': period_frequent_patterns[pattern_id]
                            })
                    
                    # 生成指标条件
                    if pattern_infos:
                        condition = {
                            'indicator': indicator_name,
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
            
            # 保存策略
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(strategy, f, ensure_ascii=False, indent=2)
            
            logger.info(f"选股策略已生成并保存到: {output_file}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"生成策略时出错: {e}")
            return {}
    
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
    
    @staticmethod
    def get_instance() -> 'EnhancedBacktest':
        """获取单例实例"""
        if not hasattr(EnhancedBacktest, '_instance'):
            EnhancedBacktest._instance = EnhancedBacktest()
        return EnhancedBacktest._instance 