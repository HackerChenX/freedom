#!/usr/bin/env python3
"""
指标评分系统

本模块实现对技术指标的评分和排序功能，用于选股策略优化。
"""

import os
import sys
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 使用依赖注入架构
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.decorators import exception_handler, performance_monitor
from utils.logger import get_logger
from indicators.base.indicator_factory import Indicator_factory
from strategy.base_strategy import BaseStrategy

logger = get_logger(__name__)

class IndicatorScoringSystem:
    """指标评分系统"""
    
    def __init___16(self):
        """初始化指标评分系统"""
        self.container = get_container()
        self.data_access = self.get_service(Data_access_interface)
        self.indicator_factory = Indicator_factory()
        self.scoring_results = {}
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def load_stock_data_Scoring(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载股票数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            logger.info(f"加载股票数据: {code}, {start_date} - {end_date}")
            
            # 使用数据访问接口获取数据
            stock_data = self.data_access.get_stock_data(
                code=code,
                start_date=start_date,
                end_date=end_date
            )
            
            if stock_data.empty:
                logger.warning(f"未找到股票 {code} 的数据")
                return pd.DataFrame()
                
            logger.info(f"成功加载 {len(stock_data)} 条数据")
            return stock_data
            
        except Exception as e:
            logger.error(f"加载股票数据失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def calculate_indicator_scores(self, data: pd.DataFrame, 
                                 indicator_configs: Dict[str, Any]) -> Dict[str, float]:
        """
        计算指标评分
        
        Args:
            data: 股票数据
            indicator_configs: 指标配置
            
        Returns:
            Dict[str, float]: 指标评分结果
        """
        scores = {}
        
        try:
            for indicator_name, config in indicator_configs.items():
                logger.info(f"计算指标评分: {indicator_name}")
                
                # 创建指标实例
                indicator = self.indicator_factory.create_indicator(
                    indicator_name, 
                    **config.get('params', {})
                )
                
                # 计算指标值
                indicator_values = indicator.calculate(data)
                
                # 计算评分（这里使用简单的评分逻辑，实际可以更复杂）
                score = self._calculate_score(indicator_values, config)
                scores[indicator_name] = score
                
                logger.info(f"{indicator_name} 评分: {score:.4f}")
                
        except Exception as e:
            logger.error(f"计算指标评分失败: {e}")
            raise
            
        return scores
    
    def _calculate_score(self, indicator_values: pd.Series, config: Dict[str, Any]) -> float:
        """
        计算单个指标的评分
        
        Args:
            indicator_values: 指标值序列
            config: 指标配置
            
        Returns:
            float: 评分
        """
        if indicator_values.empty:
            return 0.0
            
        # 获取最新值
        latest_value = indicator_values.iloc[-1]
        
        # 根据配置的评分规则计算分数
        scoring_rules = config.get('scoring_rules', {})
        
        if 'thresholds' in scoring_rules:
            # 基于阈值的评分
            thresholds = scoring_rules['thresholds']
            if latest_value >= thresholds.get('excellent', 0.8):
                return 1.0
            elif latest_value >= thresholds.get('good', 0.6):
                return 0.8
            elif latest_value >= thresholds.get('fair', 0.4):
                return 0.6
            else:
                return 0.4
        else:
            # 默认评分：归一化到0-1范围
            return min(max(latest_value, 0.0), 1.0)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def run_scoring_analysis(self, stock_codes: List[str], 
                           start_date: str, end_date: str,
                           indicator_configs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        运行评分分析
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            indicator_configs: 指标配置
            
        Returns:
            Dict[str, Dict[str, float]]: 评分结果
        """
        results = {}
        
        logger.info(f"开始评分分析，股票数量: {len(stock_codes)}")
        
        for i, code in enumerate(stock_codes, 1):
            try:
                logger.info(f"处理股票 {i}/{len(stock_codes)}: {code}")
                
                # 加载股票数据
                data = self.load_stock_data_Scoring(code, start_date, end_date)
                
                if data.empty:
                    logger.warning(f"跳过股票 {code}，无数据")
                    continue
                
                # 计算指标评分
                scores = self.calculate_indicator_scores(data, indicator_configs)
                results[code] = scores
                
            except Exception as e:
                logger.error(f"处理股票 {code} 失败: {e}")
                continue
                
        logger.info(f"评分分析完成，处理了 {len(results)} 只股票")
        return results
    
    @exception_handler(reraise=True)
    def save_results_Scoring(self, results: Dict[str, Dict[str, float]], 
                    output_file: str) -> None:
        """
        保存评分结果
        
        Args:
            results: 评分结果
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存为JSON格式
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            logger.info(f"评分结果已保存到: {output_file}")
            
            # 生成汇总报告
            summary_file = output_file.replace('.json', '_summary.txt')
            self._generate_summary_report(results, summary_file)
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def _generate_summary_report(self, results: Dict[str, Dict[str, float]], 
                               summary_file: str) -> None:
        """生成汇总报告"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("指标评分汇总报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"股票数量: {len(results)}\n\n")
                
                if results:
                    # 计算各指标的平均分
                    indicator_names = list(next(iter(results.values())).keys())
                    f.write("指标平均分:\n")
                    f.write("-" * 30 + "\n")
                    
                    for indicator in indicator_names:
                        scores = [results[code].get(indicator, 0) for code in results]
                        avg_score = sum(scores) / len(scores) if scores else 0
                        f.write(f"{indicator}: {avg_score:.4f}\n")
                    
                    f.write("\n")
                    
                    # 前10名股票
                    f.write("综合评分前10名:\n")
                    f.write("-" * 30 + "\n")
                    
                    stock_total_scores = {}
                    for code, scores in results.items():
                        total_score = sum(scores.values())
                        stock_total_scores[code] = total_score
                    
                    top_stocks = sorted(stock_total_scores.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]
                    
                    for i, (code, score) in enumerate(top_stocks, 1):
                        f.write(f"{i:2d}. {code}: {score:.4f}\n")
                
            logger.info(f"汇总报告已生成: {summary_file}")
            
        except Exception as e:
            logger.error(f"生成汇总报告失败: {e}")

def load_default_indicator_configs() -> Dict[str, Any]:
    """加载默认指标配置"""
    return {
        'RSI': {
            'params': {'period': 14},
            'scoring_rules': {
                'thresholds': {'excellent': 0.8, 'good': 0.6, 'fair': 0.4}
            }
        },
        'MACD': {
            'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'scoring_rules': {
                'thresholds': {'excellent': 0.7, 'good': 0.5, 'fair': 0.3}
            }
        },
        'KDJ': {
            'params': {'k_period': 9, 'd_period': 3, 'j_period': 3},
            'scoring_rules': {
                'thresholds': {'excellent': 0.75, 'good': 0.55, 'fair': 0.35}
            }
        }
    }

@exception_handler(reraise=True)
@performance_monitor(threshold=60.0)
def main_30():
    """主函数"""
    parser = argparse.ArgumentParser(description='指标评分系统')
    parser.add_argument('--codes', type=str, nargs='+', 
                       help='股票代码列表')
    parser.add_argument('--start-date', type=str, 
                       default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, 
                       default='data/result/indicator_scoring_results.json',
                       help='输出文件路径')
    parser.add_argument('--config', type=str, 
                       help='指标配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 初始化评分系统
        scoring_system = Indicator_scoring_system()
        
        # 获取股票代码列表
        if args.codes:
            stock_codes = args.codes
        else:
            # 默认使用一些示例股票
            stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        
        # 加载指标配置
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                indicator_configs = json.load(f)
        else:
            indicator_configs = load_default_indicator_configs()
        
        logger.info(f"开始指标评分分析")
        logger.info(f"股票代码: {stock_codes}")
        logger.info(f"时间范围: {args.start_date} - {args.end_date}")
        logger.info(f"指标配置: {list(indicator_configs.keys())}")
        
        # 运行评分分析
        results = scoring_system.run_scoring_analysis(
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date,
            indicator_configs=indicator_configs
        )
        
        # 保存结果
        scoring_system.save_results_Scoring(results, args.output)
        
        logger.info("指标评分分析完成")
        
    except Exception as e:
        logger.error(f"指标评分分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_30() 