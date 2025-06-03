#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
买点批量分析器

分析多个股票买点的共性指标特征，提取共性指标并生成选股策略
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from collections import Counter, defaultdict

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import ensure_dir_exists
from analysis.buypoints.period_data_processor import PeriodDataProcessor
from analysis.buypoints.auto_indicator_analyzer import AutoIndicatorAnalyzer
from strategy.strategy_generator import StrategyGenerator

logger = get_logger(__name__)

class BuyPointBatchAnalyzer:
    """买点批量分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.data_processor = PeriodDataProcessor()
        self.indicator_analyzer = AutoIndicatorAnalyzer()
        self.strategy_generator = StrategyGenerator()
        
    def load_buypoints_from_csv(self, csv_file: str) -> pd.DataFrame:
        """
        从CSV文件加载买点数据
        
        Args:
            csv_file: CSV文件路径
            
        Returns:
            pd.DataFrame: 买点数据
        """
        try:
            # 读取CSV文件
            buypoints_df = pd.read_csv(csv_file)
            
            # 验证必要的列
            required_columns = ['stock_code', 'buypoint_date']
            for col in required_columns:
                if col not in buypoints_df.columns:
                    raise ValueError(f"CSV文件缺少必要的列: {col}")
            
            # 确保日期格式正确
            buypoints_df['buypoint_date'] = pd.to_datetime(buypoints_df['buypoint_date'])
            buypoints_df['buypoint_date'] = buypoints_df['buypoint_date'].dt.strftime('%Y%m%d')
            
            logger.info(f"已加载 {len(buypoints_df)} 个买点")
            return buypoints_df
            
        except Exception as e:
            logger.error(f"加载买点CSV文件时出错: {e}")
            return pd.DataFrame()
    
    def analyze_single_buypoint(self, 
                             stock_code: str, 
                             buypoint_date: str) -> Dict[str, Any]:
        """
        分析单个买点
        
        Args:
            stock_code: 股票代码
            buypoint_date: 买点日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            logger.info(f"开始分析买点: {stock_code} {buypoint_date}")
            
            # 获取多周期数据
            stock_data = self.data_processor.get_multi_period_data(
                stock_code=stock_code,
                end_date=buypoint_date
            )
            
            # 如果没有获取到数据，返回空结果
            if not stock_data:
                logger.warning(f"未能获取 {stock_code} 的数据")
                return {}
                
            # 分析所有指标
            indicator_results = self.indicator_analyzer.analyze_all_indicators(
                stock_data=stock_data,
                date=buypoint_date
            )
            
            # 构建结果
            result = {
                'stock_code': stock_code,
                'buypoint_date': buypoint_date,
                'indicators': indicator_results
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分析买点 {stock_code} {buypoint_date} 时出错: {e}")
            return {}
    
    def analyze_batch_buypoints(self, 
                             buypoints_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        批量分析买点
        
        Args:
            buypoints_df: 买点数据DataFrame
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        results = []
        
        # 遍历所有买点
        for idx, row in buypoints_df.iterrows():
            stock_code = row['stock_code']
            buypoint_date = row['buypoint_date']
            
            # 分析单个买点
            buypoint_result = self.analyze_single_buypoint(
                stock_code=stock_code,
                buypoint_date=buypoint_date
            )
            
            # 如果有结果，添加到列表
            if buypoint_result:
                results.append(buypoint_result)
                
        logger.info(f"已完成 {len(results)}/{len(buypoints_df)} 个买点的分析")
        return results
    
    def extract_common_indicators(self, 
                              buypoint_results: List[Dict[str, Any]],
                              min_hit_ratio: float = 0.6) -> Dict[str, List[Dict[str, Any]]]:
        """
        提取共性指标
        
        Args:
            buypoint_results: 买点分析结果列表
            min_hit_ratio: 最小命中比例，默认0.6（60%）
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 按周期分组的共性指标列表
        """
        try:
            # 如果结果为空，返回空字典
            if not buypoint_results:
                return {}
                
            # 按周期分组的指标统计
            period_indicators = defaultdict(lambda: defaultdict(list))
            
            # 遍历所有买点结果
            for result in buypoint_results:
                # 遍历每个周期
                for period, indicators in result.get('indicators', {}).items():
                    # 遍历该周期下的所有指标
                    for indicator in indicators:
                        # 构建指标标识（指标类型_指标名_形态）
                        if indicator['type'] == 'indicator':
                            indicator_id = f"{indicator['type']}_{indicator['name']}_{indicator['pattern']}"
                        else:  # pattern类型
                            indicator_id = f"{indicator['type']}_{indicator['name']}"
                            
                        # 添加到对应周期的指标列表
                        period_indicators[period][indicator_id].append({
                            'stock_code': result['stock_code'],
                            'buypoint_date': result['buypoint_date'],
                            'score': indicator['score'],
                            'details': indicator.get('details', {})
                        })
            
            # 计算每个周期下各指标的命中率和平均得分
            common_indicators = {}
            total_buypoints = len(buypoint_results)
            
            for period, indicators in period_indicators.items():
                period_common = []
                
                for indicator_id, hits in indicators.items():
                    # 计算命中率
                    hit_ratio = len(hits) / total_buypoints
                    
                    # 如果命中率达到阈值，认为是共性指标
                    if hit_ratio >= min_hit_ratio:
                        # 计算平均得分
                        avg_score = sum(hit['score'] for hit in hits) / len(hits)
                        
                        # 拆分指标ID
                        parts = indicator_id.split('_', 2)
                        indicator_type = parts[0]
                        
                        if indicator_type == 'indicator':
                            indicator_name = parts[1]
                            pattern_name = parts[2] if len(parts) > 2 else ''
                            
                            period_common.append({
                                'type': indicator_type,
                                'name': indicator_name,
                                'pattern': pattern_name,
                                'hit_ratio': hit_ratio,
                                'hit_count': len(hits),
                                'avg_score': avg_score,
                                'hits': hits
                            })
                        else:  # pattern类型
                            indicator_name = parts[1]
                            
                            period_common.append({
                                'type': indicator_type,
                                'name': indicator_name,
                                'hit_ratio': hit_ratio,
                                'hit_count': len(hits),
                                'avg_score': avg_score,
                                'hits': hits
                            })
                
                # 按平均得分排序
                period_common.sort(key=lambda x: x['avg_score'], reverse=True)
                
                # 存储到结果字典
                if period_common:
                    common_indicators[period] = period_common
            
            return common_indicators
            
        except Exception as e:
            logger.error(f"提取共性指标时出错: {e}")
            return {}
    
    def generate_strategy(self, 
                       common_indicators: Dict[str, List[Dict[str, Any]]],
                       strategy_name: str = "BuyPointCommonStrategy") -> Dict[str, Any]:
        """
        生成选股策略
        
        Args:
            common_indicators: 共性指标
            strategy_name: 策略名称
            
        Returns:
            Dict[str, Any]: 生成的策略
        """
        try:
            # 如果没有共性指标，返回空字典
            if not common_indicators:
                return {}
                
            # 构建策略条件
            strategy_conditions = []
            
            # 遍历各周期的共性指标
            for period, indicators in common_indicators.items():
                # 遍历该周期下的共性指标
                for indicator in indicators:
                    # 根据指标类型构建条件
                    if indicator['type'] == 'indicator':
                        # 技术指标形态
                        condition = {
                            'type': 'indicator',
                            'period': period,
                            'indicator': indicator['name'],
                            'pattern': indicator['pattern'],
                            'score_threshold': indicator['avg_score'] * 0.8  # 设置分数阈值为平均分的80%
                        }
                    else:  # pattern类型
                        # K线形态
                        condition = {
                            'type': 'pattern',
                            'period': period,
                            'pattern': indicator['name'],
                            'score_threshold': indicator['avg_score'] * 0.8  # 设置分数阈值为平均分的80%
                        }
                        
                    strategy_conditions.append(condition)
            
            # 生成策略
            strategy = self.strategy_generator.generate_strategy(
                strategy_name=strategy_name,
                conditions=strategy_conditions,
                condition_logic="OR"  # 使用OR逻辑，满足任一条件即可
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"生成选股策略时出错: {e}")
            return {}
    
    def save_results(self, 
                   buypoint_results: List[Dict[str, Any]],
                   common_indicators: Dict[str, List[Dict[str, Any]]],
                   strategy: Dict[str, Any],
                   output_dir: str):
        """
        保存分析结果
        
        Args:
            buypoint_results: 买点分析结果
            common_indicators: 共性指标
            strategy: 生成的策略
            output_dir: 输出目录
        """
        try:
            # 确保输出目录存在
            ensure_dir_exists(output_dir)
            
            # 保存买点分析结果
            with open(os.path.join(output_dir, 'buypoint_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(buypoint_results, f, ensure_ascii=False, indent=2)
                
            # 保存共性指标
            with open(os.path.join(output_dir, 'common_indicators.json'), 'w', encoding='utf-8') as f:
                json.dump(common_indicators, f, ensure_ascii=False, indent=2)
                
            # 保存生成的策略
            with open(os.path.join(output_dir, 'generated_strategy.json'), 'w', encoding='utf-8') as f:
                json.dump(strategy, f, ensure_ascii=False, indent=2)
                
            # 生成共性指标报告
            self._generate_common_indicators_report(common_indicators, output_dir)
                
            logger.info(f"分析结果已保存到 {output_dir}")
            
        except Exception as e:
            logger.error(f"保存分析结果时出错: {e}")
    
    def _generate_common_indicators_report(self, 
                                       common_indicators: Dict[str, List[Dict[str, Any]]],
                                       output_dir: str):
        """
        生成共性指标报告
        
        Args:
            common_indicators: 共性指标
            output_dir: 输出目录
        """
        try:
            # 构建报告内容
            report = ["# 买点共性指标分析报告\n\n"]
            report.append(f"## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 添加各周期的共性指标
            for period, indicators in common_indicators.items():
                report.append(f"## {period} 周期共性指标\n\n")
                
                # 添加表格头
                report.append("| 指标类型 | 指标名称 | 形态 | 命中率 | 命中数量 | 平均得分 |\n")
                report.append("|---------|----------|------|--------|----------|----------|\n")
                
                # 添加各指标信息
                for indicator in indicators:
                    indicator_type = indicator['type']
                    indicator_name = indicator['name']
                    pattern = indicator.get('pattern', '-')
                    hit_ratio = f"{indicator['hit_ratio']:.2%}"
                    hit_count = indicator['hit_count']
                    avg_score = f"{indicator['avg_score']:.2f}"
                    
                    report.append(f"| {indicator_type} | {indicator_name} | {pattern} | {hit_ratio} | {hit_count} | {avg_score} |\n")
                
                report.append("\n")
                
            # 写入报告文件
            with open(os.path.join(output_dir, 'common_indicators_report.md'), 'w', encoding='utf-8') as f:
                f.writelines(report)
                
        except Exception as e:
            logger.error(f"生成共性指标报告时出错: {e}")
    
    def run_analysis(self, 
                  input_csv: str, 
                  output_dir: str,
                  min_hit_ratio: float = 0.6,
                  strategy_name: str = "BuyPointCommonStrategy"):
        """
        运行买点批量分析
        
        Args:
            input_csv: 输入CSV文件路径
            output_dir: 输出目录
            min_hit_ratio: 最小命中比例
            strategy_name: 生成的策略名称
        """
        try:
            # 加载买点数据
            buypoints_df = self.load_buypoints_from_csv(input_csv)
            if buypoints_df.empty:
                logger.error(f"未能加载买点数据，分析终止")
                return
                
            # 批量分析买点
            buypoint_results = self.analyze_batch_buypoints(buypoints_df)
            if not buypoint_results:
                logger.error(f"买点分析未产生结果，分析终止")
                return
                
            # 提取共性指标
            common_indicators = self.extract_common_indicators(
                buypoint_results=buypoint_results,
                min_hit_ratio=min_hit_ratio
            )
            if not common_indicators:
                logger.warning(f"未能提取到共性指标")
                
            # 生成选股策略
            strategy = self.generate_strategy(
                common_indicators=common_indicators,
                strategy_name=strategy_name
            )
            
            # 保存结果
            self.save_results(
                buypoint_results=buypoint_results,
                common_indicators=common_indicators,
                strategy=strategy,
                output_dir=output_dir
            )
            
            logger.info(f"买点批量分析完成")
            
        except Exception as e:
            logger.error(f"运行买点批量分析时出错: {e}") 