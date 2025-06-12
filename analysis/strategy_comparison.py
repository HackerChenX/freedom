#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_result_dir
from utils.decorators import safe_run, performance_monitor
from strategy.strategy_factory import StrategyFactory
from strategy.strategy_manager import StrategyManager
from strategy.strategy_executor import StrategyExecutor
from db.db_manager import DBManager
from analysis.strategy_validator import StrategyValidator

# 获取日志记录器
logger = get_logger(__name__)

class StrategyComparison:
    """
    策略比较功能增强模块 - 提供高级策略比较和分析能力
    
    主要功能：
    1. 多维度策略对比 - 从多个维度比较不同策略的表现
    2. 策略相似度分析 - 分析不同策略之间的相似程度
    3. 策略组合分析 - 分析多个策略组合的效果
    4. 最优策略推荐 - 根据历史表现推荐最优策略组合
    5. 策略可视化比较 - 可视化展示策略比较结果
    """
    
    def __init__(self):
        """初始化策略比较器"""
        logger.info("初始化策略比较器")
        
        # 初始化结果目录
        self.result_dir = get_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 初始化策略管理器
        self.strategy_manager = StrategyManager()
        
        # 初始化策略执行器
        self.strategy_executor = StrategyExecutor()
        
        # 初始化策略验证器
        self.strategy_validator = StrategyValidator()
        
        # 初始化数据库管理器
        self.db_manager = DBManager.get_instance()
        
        logger.info("策略比较器初始化完成") 

    @safe_run
    @performance_monitor
    def compare_strategies_multi_dimension(self, strategy_ids: List[str], 
                                       time_periods: List[Dict[str, str]] = None,
                                       stock_pool: List[str] = None,
                                       dimensions: List[str] = None) -> Dict[str, Any]:
        """
        多维度策略对比 - 从多个维度比较不同策略的表现
        
        Args:
            strategy_ids: 策略ID列表
            time_periods: 时间周期列表，每个元素为包含start_date和end_date的字典
                例如: [{"start_date": "20220101", "end_date": "20220131", "name": "1月"}]
            stock_pool: 股票池，默认为None表示使用全市场样本
            dimensions: 比较维度列表，可选值包括:
                - "selection_ratio": 选股比例
                - "return_performance": 收益表现
                - "stability": 稳定性
                - "risk": 风险指标
                - "style": 风格特征
                - "industry": 行业分布
                - "overlap": 重叠度
                如果为None，则使用全部维度
                
        Returns:
            Dict: 多维度比较结果
        """
        logger.info(f"开始多维度比较策略: {strategy_ids}")
        
        # 设置默认比较维度
        if dimensions is None:
            dimensions = [
                "selection_ratio", "return_performance", "stability", 
                "risk", "style", "industry", "overlap"
            ]
            
        # 设置默认时间周期
        if time_periods is None:
            today = datetime.now()
            
            # 最近一个月
            month_end = today
            month_start = month_end - timedelta(days=30)
            
            # 最近三个月
            quarter_end = today
            quarter_start = quarter_end - timedelta(days=90)
            
            # 最近六个月
            half_year_end = today
            half_year_start = half_year_end - timedelta(days=180)
            
            time_periods = [
                {"start_date": month_start.strftime("%Y%m%d"), 
                 "end_date": month_end.strftime("%Y%m%d"), 
                 "name": "近1月"},
                {"start_date": quarter_start.strftime("%Y%m%d"), 
                 "end_date": quarter_end.strftime("%Y%m%d"), 
                 "name": "近3月"},
                {"start_date": half_year_start.strftime("%Y%m%d"), 
                 "end_date": half_year_end.strftime("%Y%m%d"), 
                 "name": "近6月"},
            ]
        
        # 获取策略配置
        strategy_configs = {}
        for strategy_id in strategy_ids:
            config = self.strategy_manager.get_strategy(strategy_id)
            if not config:
                logger.warning(f"未找到策略: {strategy_id}")
                continue
            strategy_configs[strategy_id] = config
            
        if not strategy_configs:
            logger.error("未找到任何有效策略")
            return {"error": "未找到任何有效策略"}
            
        # 如果没有指定股票池，获取默认股票池
        if stock_pool is None:
            # 获取全市场股票或样本股票
            try:
                # 尝试使用get_all_stock_codes方法
                stock_pool = self.db_manager.get_all_stock_codes()
            except AttributeError:
                # 如果方法不存在，使用替代方法
                try:
                    sql = "SELECT DISTINCT stock_code FROM stock_info"
                    result = self.db_manager.execute_query(sql)
                    stock_pool = [row['stock_code'] for row in result] if result else []
                except Exception as e:
                    logger.error(f"获取股票池时出错: {e}")
                    stock_pool = []
            
            # 限制比较股票数量，避免过长时间
            if len(stock_pool) > 300:
                import random
                stock_pool = random.sample(stock_pool, 300)
        
        # 在各个时间周期上执行策略并收集结果
        all_period_results = []
        
        for period in time_periods:
            start_date = period["start_date"]
            end_date = period["end_date"]
            period_name = period.get("name", f"{start_date}-{end_date}")
            
            period_results = []
            
            for strategy_id, strategy_config in strategy_configs.items():
                # 执行策略
                result = self.strategy_executor.execute_strategy(
                    strategy_config["strategy"], stock_pool, start_date, end_date)
                
                # 计算策略选股后续表现
                performance = None
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    performance = self._calculate_extended_performance(
                        result['code'].tolist(), end_date)
                    
                # 准备策略结果
                strategy_result = {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_config["strategy"]["name"],
                    "selected_stocks": len(result) if isinstance(result, pd.DataFrame) else 0,
                    "selection_ratio": len(result) / len(stock_pool) if isinstance(result, pd.DataFrame) and len(stock_pool) > 0 else 0,
                    "stocks": result.to_dict(orient='records') if isinstance(result, pd.DataFrame) else [],
                    "performance": performance
                }
                
                # 添加行业分布分析
                if "industry" in dimensions and isinstance(result, pd.DataFrame) and len(result) > 0:
                    industry_distribution = self._analyze_industry_distribution(result['code'].tolist())
                    strategy_result["industry_distribution"] = industry_distribution
                
                # 添加风格特征分析
                if "style" in dimensions and isinstance(result, pd.DataFrame) and len(result) > 0:
                    style_features = self._analyze_style_features(result['code'].tolist())
                    strategy_result["style_features"] = style_features
                
                period_results.append(strategy_result)
                
            # 计算策略之间的重叠度
            if "overlap" in dimensions and len(period_results) > 1:
                overlap_matrix = self._calculate_overlap_matrix(period_results)
                
                # 添加重叠度到各策略结果
                for i, strategy_result in enumerate(period_results):
                    strategy_result["overlap"] = {
                        other_result["strategy_id"]: overlap_matrix[i][j]
                        for j, other_result in enumerate(period_results)
                        if i != j
                    }
            
            # 添加时间周期信息
            period_data = {
                "period_name": period_name,
                "start_date": start_date,
                "end_date": end_date,
                "strategy_results": period_results
            }
            
            all_period_results.append(period_data)
        
        # 汇总多维度比较结果
        comparison_result = {
            "strategies": [{"id": s_id, "name": strategy_configs[s_id]["strategy"]["name"]} 
                         for s_id in strategy_ids if s_id in strategy_configs],
            "dimensions": dimensions,
            "time_periods": time_periods,
            "period_results": all_period_results,
            "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加维度排名
        comparison_result["dimension_rankings"] = self._calculate_dimension_rankings(
            all_period_results, dimensions)
        
        # 添加综合评分
        comparison_result["overall_scores"] = self._calculate_overall_scores(
            comparison_result["dimension_rankings"])
        
        logger.info(f"多维度策略比较完成，比较了 {len(strategy_configs)} 个策略在 {len(time_periods)} 个时间周期")
        return comparison_result
    
    def _calculate_extended_performance(self, stock_codes: List[str], 
                                      start_date: str, days_list: List[int] = None) -> Dict[str, Any]:
        """
        计算股票在多个时间段的后续表现
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            days_list: 后续跟踪天数列表，默认为[5, 10, 20, 60]
            
        Returns:
            Dict: 表现统计
        """
        if days_list is None:
            days_list = [5, 10, 20, 60]
            
        results = {}
        
        for days in days_list:
            perf = self.strategy_validator._calculate_performance(stock_codes, start_date, days)
            results[f"d{days}"] = perf
            
        # 计算综合评分
        avg_returns = [results[f"d{days}"]["avg_return"] for days in days_list]
        positive_ratios = [results[f"d{days}"]["positive_ratio"] for days in days_list]
        
        # 使用加权平均，较长期限权重更高
        weights = np.array(days_list) / sum(days_list)
        weighted_return = np.sum(np.array(avg_returns) * weights)
        weighted_positive = np.sum(np.array(positive_ratios) * weights)
        
        results["weighted_return"] = float(weighted_return)
        results["weighted_positive_ratio"] = float(weighted_positive)
        
        return results
    
    def _analyze_industry_distribution(self, stock_codes: List[str]) -> Dict[str, Any]:
        """
        分析股票的行业分布
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict: 行业分布
        """
        try:
            if not stock_codes:
                return {}
                
            # 使用参数化查询替代字符串拼接，避免SQL注入
            query_params = []
            placeholders = []
            
            for code in stock_codes:
                placeholders.append("?")
                query_params.append(code)
                
            placeholders_str = ", ".join(placeholders)
            
            sql = f"""
            SELECT industry, COUNT(*) as count
            FROM stock_info
            WHERE stock_code IN ({placeholders_str})
            GROUP BY industry
            ORDER BY count DESC
            """
            
            # 执行查询时传入参数
            try:
                result = self.db_manager.execute_query(sql, query_params)
            except TypeError:
                # 如果不支持参数化查询，使用安全的字符串方式
                # 确保股票代码只包含安全字符
                safe_codes = []
                for code in stock_codes:
                    if isinstance(code, str) and code.replace(".", "").isalnum():
                        safe_codes.append(code)
                
                if not safe_codes:
                    return {}
                    
                stock_codes_str = "', '".join(safe_codes)
                sql = f"""
                SELECT industry, COUNT(*) as count
                FROM stock_info
                WHERE stock_code IN ('{stock_codes_str}')
                GROUP BY industry
                ORDER BY count DESC
                """
                result = self.db_manager.execute_query(sql)
            
            if not result:
                return {}
            
            # 构建行业分布
            distribution = {}
            for row in result:
                industry = row['industry'] if row['industry'] else "其他"
                count = int(row['count'])
                distribution[industry] = count
            
            # 计算行业分布百分比
            total = sum(distribution.values())
            percentages = {k: v / total for k, v in distribution.items()}
            
            # 计算行业集中度 (HHI指数)
            hhi = sum([p * p for p in percentages.values()])
            
            return {
                "counts": distribution,
                "percentages": percentages,
                "concentration": hhi
            }
            
        except Exception as e:
            logger.error(f"分析行业分布时出错: {e}")
            return {}
    
    def _analyze_style_features(self, stock_codes: List[str]) -> Dict[str, Any]:
        """
        分析股票的风格特征
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict: 风格特征
        """
        try:
            if not stock_codes:
                return {}
                
            # 使用参数化查询替代字符串拼接，避免SQL注入
            query_params = []
            placeholders = []
            
            for code in stock_codes:
                placeholders.append("?")
                query_params.append(code)
                
            placeholders_str = ", ".join(placeholders)
            
            sql = f"""
            SELECT stock_code, market_cap, pb_ratio, pe_ratio
            FROM stock_info
            WHERE stock_code IN ({placeholders_str})
            """
            
            # 执行查询时传入参数
            try:
                result = self.db_manager.execute_query(sql, query_params)
            except TypeError:
                # 如果不支持参数化查询，使用安全的字符串方式
                # 确保股票代码只包含安全字符
                safe_codes = []
                for code in stock_codes:
                    if isinstance(code, str) and code.replace(".", "").isalnum():
                        safe_codes.append(code)
                
                if not safe_codes:
                    return {}
                    
                stock_codes_str = "', '".join(safe_codes)
                sql = f"""
                SELECT stock_code, market_cap, pb_ratio, pe_ratio
                FROM stock_info
                WHERE stock_code IN ('{stock_codes_str}')
                """
                result = self.db_manager.execute_query(sql)
            
            if not result:
                return {}
            
            # 转换为DataFrame
            df = pd.DataFrame(result)
            
            # 数据清洗和类型转换
            for col in ['market_cap', 'pb_ratio', 'pe_ratio']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算风格指标
            style_features = {
                "avg_market_cap": float(df['market_cap'].mean()) if 'market_cap' in df.columns else 0,
                "median_market_cap": float(df['market_cap'].median()) if 'market_cap' in df.columns else 0,
                "avg_pb": float(df['pb_ratio'].mean()) if 'pb_ratio' in df.columns else 0,
                "median_pb": float(df['pb_ratio'].median()) if 'pb_ratio' in df.columns else 0,
                "avg_pe": float(df['pe_ratio'].mean()) if 'pe_ratio' in df.columns else 0,
                "median_pe": float(df['pe_ratio'].median()) if 'pe_ratio' in df.columns else 0,
            }
            
            # 计算市值分布
            market_cap_ranges = {
                "small": (0, 50),   # 小市值: 0-50亿
                "medium": (50, 200), # 中市值: 50-200亿
                "large": (200, 1000), # 大市值: 200-1000亿
                "mega": (1000, float('inf'))  # 超大市值: >1000亿
            }
            
            market_cap_distribution = {}
            if 'market_cap' in df.columns:
                for name, (min_cap, max_cap) in market_cap_ranges.items():
                    count = len(df[(df['market_cap'] >= min_cap) & (df['market_cap'] < max_cap)])
                    market_cap_distribution[name] = count
            
            # 计算市值分布百分比
            total = sum(market_cap_distribution.values())
            if total > 0:
                market_cap_percentages = {k: v / total for k, v in market_cap_distribution.items()}
            else:
                market_cap_percentages = {k: 0 for k in market_cap_distribution}
            
            style_features["market_cap_distribution"] = market_cap_distribution
            style_features["market_cap_percentages"] = market_cap_percentages
            
            # 判断风格类型
            if style_features["median_pe"] < 15:
                value_bias = "value"  # 价值风格
            elif style_features["median_pe"] > 30:
                value_bias = "growth"  # 成长风格
            else:
                value_bias = "blend"  # 混合风格
                
            if style_features["median_market_cap"] < 100:
                size_bias = "small"  # 小盘风格
            elif style_features["median_market_cap"] > 500:
                size_bias = "large"  # 大盘风格
            else:
                size_bias = "mid"  # 中盘风格
                
            style_features["style_type"] = f"{size_bias}-{value_bias}"
            
            return style_features
            
        except Exception as e:
            logger.error(f"分析风格特征时出错: {e}")
            return {}
    
    def _calculate_overlap_matrix(self, strategy_results: List[Dict[str, Any]]) -> List[List[float]]:
        """
        计算策略之间的重叠度矩阵
        
        Args:
            strategy_results: 策略结果列表
            
        Returns:
            List[List[float]]: 重叠度矩阵
        """
        n = len(strategy_results)
        
        # 检查策略结果是否为空
        if n == 0:
            logger.warning("策略结果为空，无法计算重叠度矩阵")
            return []
            
        overlap_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # 获取每个策略选出的股票集合
        stock_sets = []
        for result in strategy_results:
            stocks = set()
            for stock in result.get("stocks", []):
                # 处理不同的数据类型
                if isinstance(stock, dict):
                    stock_code = stock.get("code", "")
                else:
                    stock_code = stock
                    
                if stock_code:  # 确保股票代码不为空
                    stocks.add(stock_code)
            stock_sets.append(stocks)
        
        # 检查是否所有策略都没有选出股票
        if all(len(s) == 0 for s in stock_sets):
            logger.warning("所有策略都没有选出股票，重叠度矩阵将全为0")
            return overlap_matrix
        
        # 计算每对策略之间的重叠度
        for i in range(n):
            for j in range(n):
                if i == j:
                    overlap_matrix[i][j] = 1.0  # 自身重叠度为1
                    continue
                    
                set_i = stock_sets[i]
                set_j = stock_sets[j]
                
                if not set_i or not set_j:
                    overlap_matrix[i][j] = 0.0
                    continue
                
                # 计算Jaccard相似度
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                
                if union > 0:
                    overlap_matrix[i][j] = intersection / union
                else:
                    overlap_matrix[i][j] = 0.0
        
        return overlap_matrix

    def _calculate_dimension_rankings(self, period_results: List[Dict[str, Any]], 
                                   dimensions: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        计算各维度的策略排名
        
        Args:
            period_results: 各时间周期的结果
            dimensions: 比较维度列表
            
        Returns:
            Dict: 各维度排名
        """
        rankings = {dim: [] for dim in dimensions}
        
        # 获取所有策略ID
        all_strategy_ids = set()
        for period in period_results:
            for result in period["strategy_results"]:
                all_strategy_ids.add(result["strategy_id"])
        
        # 对每个维度计算排名
        for dimension in dimensions:
            dimension_scores = {s_id: 0.0 for s_id in all_strategy_ids}
            dimension_counts = {s_id: 0 for s_id in all_strategy_ids}
            
            for period in period_results:
                # 根据不同维度获取得分
                if dimension == "selection_ratio":
                    # 选股比例适中(10%-30%)得分高
                    for result in period["strategy_results"]:
                        ratio = result.get("selection_ratio", 0)
                        if 0.1 <= ratio <= 0.3:
                            score = 1.0 - abs(0.2 - ratio) * 5  # 20%是最优的
                        else:
                            score = max(0, 0.5 - abs(0.2 - ratio) * 2)
                        dimension_scores[result["strategy_id"]] += score
                        dimension_counts[result["strategy_id"]] += 1
                
                elif dimension == "return_performance":
                    # 根据加权收益率排名
                    results = []
                    for result in period["strategy_results"]:
                        perf = result.get("performance", {})
                        if perf:
                            weighted_return = perf.get("weighted_return", 0)
                            results.append((result["strategy_id"], weighted_return))
                    
                    if results:
                        # 按收益率排序
                        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                        max_score = len(sorted_results)
                        
                        for rank, (s_id, _) in enumerate(sorted_results):
                            score = (max_score - rank) / max_score  # 标准化为0-1
                            dimension_scores[s_id] += score
                            dimension_counts[s_id] += 1
                
                elif dimension == "stability":
                    # 根据收益稳定性排名
                    results = []
                    for result in period["strategy_results"]:
                        perf = result.get("performance", {})
                        if perf and "d5" in perf and "d20" in perf:
                            # 收益标准差越小越稳定
                            returns = [
                                perf["d5"].get("avg_return", 0),
                                perf.get("d10", {}).get("avg_return", 0),
                                perf["d20"].get("avg_return", 0)
                            ]
                            std_dev = np.std(returns) if returns else 1
                            stability = 1 / (1 + std_dev * 10)  # 转换为0-1之间的分数
                            results.append((result["strategy_id"], stability))
                    
                    if results:
                        # 按稳定性排序
                        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                        max_score = len(sorted_results)
                        
                        for rank, (s_id, _) in enumerate(sorted_results):
                            score = (max_score - rank) / max_score
                            dimension_scores[s_id] += score
                            dimension_counts[s_id] += 1
                
                elif dimension == "risk":
                    # 根据风险指标排名
                    results = []
                    for result in period["strategy_results"]:
                        perf = result.get("performance", {})
                        if perf:
                            # 计算风险调整后收益
                            weighted_return = perf.get("weighted_return", 0)
                            max_drawdown = abs(perf.get("d20", {}).get("min_return", 0))
                            
                            if max_drawdown > 0:
                                risk_adjusted = weighted_return / max_drawdown
                            else:
                                risk_adjusted = weighted_return if weighted_return > 0 else 0
                                
                            results.append((result["strategy_id"], risk_adjusted))
                    
                    if results:
                        # 按风险调整后收益排序
                        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                        max_score = len(sorted_results)
                        
                        for rank, (s_id, _) in enumerate(sorted_results):
                            score = (max_score - rank) / max_score
                            dimension_scores[s_id] += score
                            dimension_counts[s_id] += 1
                
                elif dimension == "industry":
                    # 根据行业分散度排名（低集中度更好）
                    results = []
                    for result in period["strategy_results"]:
                        industry_dist = result.get("industry_distribution", {})
                        if industry_dist:
                            concentration = industry_dist.get("concentration", 1)
                            # 集中度越低越好
                            diversity = 1 - concentration
                            results.append((result["strategy_id"], diversity))
                    
                    if results:
                        # 按多样性排序
                        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                        max_score = len(sorted_results)
                        
                        for rank, (s_id, _) in enumerate(sorted_results):
                            score = (max_score - rank) / max_score
                            dimension_scores[s_id] += score
                            dimension_counts[s_id] += 1
                
                elif dimension == "style":
                    # 此处可以根据特定风格偏好评分
                    # 这里以均衡风格（中等市值、适中PE）为优
                    for result in period["strategy_results"]:
                        style = result.get("style_features", {})
                        if style:
                            pe = style.get("median_pe", 0)
                            market_cap = style.get("median_market_cap", 0)
                            
                            # PE得分：15-25为最优
                            if 15 <= pe <= 25:
                                pe_score = 1.0
                            elif pe < 15:
                                pe_score = pe / 15
                            else:  # pe > 25
                                pe_score = max(0, 1 - (pe - 25) / 25)
                            
                            # 市值得分：100-300亿为最优
                            if 100 <= market_cap <= 300:
                                cap_score = 1.0
                            elif market_cap < 100:
                                cap_score = market_cap / 100
                            else:  # market_cap > 300
                                cap_score = max(0, 1 - (market_cap - 300) / 700)
                            
                            # 综合得分
                            score = (pe_score + cap_score) / 2
                            dimension_scores[result["strategy_id"]] += score
                            dimension_counts[result["strategy_id"]] += 1
                
                elif dimension == "overlap":
                    # 重叠度低的策略得分高
                    for result in period["strategy_results"]:
                        overlap_dict = result.get("overlap", {})
                        if overlap_dict:
                            # 计算平均重叠度
                            avg_overlap = sum(overlap_dict.values()) / len(overlap_dict)
                            # 重叠度越低越好
                            score = 1 - avg_overlap
                            dimension_scores[result["strategy_id"]] += score
                            dimension_counts[result["strategy_id"]] += 1
            
            # 计算平均得分
            avg_scores = []
            for s_id in all_strategy_ids:
                if dimension_counts[s_id] > 0:
                    avg_score = dimension_scores[s_id] / dimension_counts[s_id]
                else:
                    avg_score = 0
                avg_scores.append((s_id, avg_score))
            
            # 排序并创建排名列表
            sorted_scores = sorted(avg_scores, key=lambda x: x[1], reverse=True)
            for rank, (s_id, score) in enumerate(sorted_scores):
                rankings[dimension].append({
                    "strategy_id": s_id,
                    "rank": rank + 1,
                    "score": score
                })
        
        return rankings
    
    def _calculate_overall_scores(self, dimension_rankings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        计算策略的综合评分
        
        Args:
            dimension_rankings: 各维度排名
            
        Returns:
            List: 综合评分排名
        """
        # 为每个维度设置权重
        dimension_weights = {
            "return_performance": 0.35,
            "risk": 0.25,
            "stability": 0.15,
            "selection_ratio": 0.1,
            "industry": 0.05,
            "style": 0.05,
            "overlap": 0.05
        }
        
        # 确保权重存在
        for dim in dimension_rankings:
            if dim not in dimension_weights:
                dimension_weights[dim] = 0.1
        
        # 正规化权重
        total_weight = sum(dimension_weights.values())
        normalized_weights = {k: v / total_weight for k, v in dimension_weights.items()}
        
        # 获取所有策略ID
        all_strategy_ids = set()
        for dim, rankings in dimension_rankings.items():
            for rank_info in rankings:
                all_strategy_ids.add(rank_info["strategy_id"])
        
        # 计算加权得分
        weighted_scores = {}
        for s_id in all_strategy_ids:
            weighted_scores[s_id] = 0
            
            for dim, rankings in dimension_rankings.items():
                # 查找该策略在当前维度的得分
                for rank_info in rankings:
                    if rank_info["strategy_id"] == s_id:
                        dim_score = rank_info["score"]
                        weight = normalized_weights.get(dim, 0)
                        weighted_scores[s_id] += dim_score * weight
                        break
        
        # 创建综合排名
        overall_rankings = []
        for s_id, score in weighted_scores.items():
            overall_rankings.append({
                "strategy_id": s_id,
                "overall_score": score
            })
        
        # 排序并添加排名
        overall_rankings.sort(key=lambda x: x["overall_score"], reverse=True)
        for rank, rank_info in enumerate(overall_rankings):
            rank_info["rank"] = rank + 1
        
        return overall_rankings
    
    @safe_run
    @performance_monitor
    def analyze_strategy_combination(self, strategy_ids: List[str],
                                 weights: Optional[List[float]] = None,
                                 start_date: str = None,
                                 end_date: str = None,
                                 stock_pool: List[str] = None) -> Dict[str, Any]:
        """
        策略组合分析 - 分析多个策略组合的效果
        
        Args:
            strategy_ids: 策略ID列表
            weights: 各策略权重，默认为等权重
            start_date: 开始日期，默认为当前日期前30天
            end_date: 结束日期，默认为当前日期
            stock_pool: 股票池，默认为None表示使用全市场样本
            
        Returns:
            Dict: 策略组合分析结果
        """
        logger.info(f"开始分析策略组合: {strategy_ids}")
        
        # 设置默认日期
        if not start_date or not end_date:
            today = datetime.now()
            end_date = today.strftime("%Y%m%d")
            start_date = (today - timedelta(days=30)).strftime("%Y%m%d")
        
        # 设置默认权重
        if weights is None:
            weights = [1.0 / len(strategy_ids) for _ in strategy_ids]
        elif len(weights) != len(strategy_ids):
            logger.warning(f"权重数量 {len(weights)} 与策略数量 {len(strategy_ids)} 不匹配，使用等权重")
            weights = [1.0 / len(strategy_ids) for _ in strategy_ids]
        else:
            # 标准化权重，确保和为1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(strategy_ids) for _ in strategy_ids]
        
        # 获取策略配置
        strategy_configs = {}
        for strategy_id in strategy_ids:
            config = self.strategy_manager.get_strategy(strategy_id)
            if not config:
                logger.warning(f"未找到策略: {strategy_id}")
                continue
            strategy_configs[strategy_id] = config
            
        if not strategy_configs:
            logger.error("未找到任何有效策略")
            return {"error": "未找到任何有效策略"}
            
        # 如果没有指定股票池，获取默认股票池
        if stock_pool is None:
            # 获取全市场股票或样本股票
            stock_pool = self.db_manager.get_all_stock_codes()
            
            # 限制比较股票数量，避免过长时间
            if len(stock_pool) > 300:
                import random
                stock_pool = random.sample(stock_pool, 300)
        
        # 执行各个策略并收集结果
        strategy_results = []
        
        for strategy_id, strategy_config in strategy_configs.items():
            # 执行策略
            result = self.strategy_executor.execute_strategy(
                strategy_config["strategy"], stock_pool, start_date, end_date)
            
            # 准备策略结果
            strategy_result = {
                "strategy_id": strategy_id,
                "strategy_name": strategy_config["strategy"]["name"],
                "selected_stocks": len(result) if isinstance(result, pd.DataFrame) else 0,
                "selection_ratio": len(result) / len(stock_pool) if isinstance(result, pd.DataFrame) and len(stock_pool) > 0 else 0,
                "stocks": result['code'].tolist() if isinstance(result, pd.DataFrame) else []
            }
            
            strategy_results.append(strategy_result)
        
        # 计算组合选股结果
        combined_stocks = self._calculate_combined_stocks(strategy_results, weights)
        
        # 计算组合性能
        combined_performance = None
        if combined_stocks:
            combined_performance = self._calculate_extended_performance(combined_stocks, end_date)
        
        # 构建组合分析结果
        combination_result = {
            "strategies": [{"id": result["strategy_id"], "name": result["strategy_name"], "weight": weight} 
                         for result, weight in zip(strategy_results, weights)],
            "start_date": start_date,
            "end_date": end_date,
            "total_stock_pool": len(stock_pool),
            "combined_stocks": combined_stocks,
            "combined_stock_count": len(combined_stocks),
            "selection_ratio": len(combined_stocks) / len(stock_pool) if stock_pool else 0,
            "performance": combined_performance,
            "individual_results": strategy_results
        }
        
        # 计算组合与各个单一策略的对比
        comparison = self._compare_combination_with_individuals(combination_result)
        combination_result["comparison"] = comparison
        
        logger.info(f"策略组合分析完成，组合了 {len(strategy_configs)} 个策略，选出 {len(combined_stocks)} 只股票")
        return combination_result
    
    def _calculate_combined_stocks(self, strategy_results: List[Dict[str, Any]], 
                               weights: List[float]) -> List[str]:
        """
        计算组合策略的选股结果
        
        Args:
            strategy_results: 各策略的结果
            weights: 各策略权重
            
        Returns:
            List: 组合选出的股票代码列表
        """
        # 计算每只股票的加权得分
        stock_scores = {}
        
        for result, weight in zip(strategy_results, weights):
            stocks = result.get("stocks", [])
            
            for stock in stocks:
                # 处理不同的数据类型
                stock_code = stock
                if isinstance(stock, dict):
                    stock_code = stock.get("code", "")
                
                if stock_code not in stock_scores:
                    stock_scores[stock_code] = 0
                stock_scores[stock_code] += weight
        
        # 设置选股阈值（权重和的一半）
        threshold = sum(weights) / 2
        
        # 筛选得分超过阈值的股票
        combined_stocks = [stock for stock, score in stock_scores.items() if score >= threshold]
        
        return combined_stocks
    
    def _compare_combination_with_individuals(self, combination_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较组合策略与各个单一策略的表现
        
        Args:
            combination_result: 组合策略结果
            
        Returns:
            Dict: 比较结果
        """
        combined_performance = combination_result.get("performance", {})
        if not combined_performance:
            return {}
            
        combined_weighted_return = combined_performance.get("weighted_return", 0)
        combined_weighted_positive = combined_performance.get("weighted_positive_ratio", 0)
        
        individual_results = combination_result.get("individual_results", [])
        
        comparisons = []
        for result in individual_results:
            strategy_id = result.get("strategy_id", "")
            strategy_name = result.get("strategy_name", "")
            
            # 获取该策略的性能指标
            performance = self._calculate_extended_performance(
                result.get("stocks", []), combination_result.get("end_date", ""))
                
            if not performance:
                continue
                
            strategy_weighted_return = performance.get("weighted_return", 0)
            strategy_weighted_positive = performance.get("weighted_positive_ratio", 0)
            
            # 计算相对表现
            return_diff = combined_weighted_return - strategy_weighted_return
            positive_diff = combined_weighted_positive - strategy_weighted_positive
            
            comparisons.append({
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "return_comparison": {
                    "combined": combined_weighted_return,
                    "individual": strategy_weighted_return,
                    "difference": return_diff,
                    "improvement": return_diff / abs(strategy_weighted_return) if strategy_weighted_return != 0 else 0
                },
                "positive_ratio_comparison": {
                    "combined": combined_weighted_positive,
                    "individual": strategy_weighted_positive,
                    "difference": positive_diff,
                    "improvement": positive_diff / strategy_weighted_positive if strategy_weighted_positive != 0 else 0
                },
                "overall_improvement": (
                    (return_diff / abs(strategy_weighted_return) if strategy_weighted_return != 0 else 0) + 
                    (positive_diff / strategy_weighted_positive if strategy_weighted_positive != 0 else 0)
                ) / 2
            })
        
        # 按整体改进幅度排序
        comparisons.sort(key=lambda x: x.get("overall_improvement", 0), reverse=True)
        
        return {
            "individual_comparisons": comparisons,
            "average_return_improvement": sum(c["return_comparison"]["improvement"] for c in comparisons) / len(comparisons) if comparisons else 0,
            "average_positive_improvement": sum(c["positive_ratio_comparison"]["improvement"] for c in comparisons) / len(comparisons) if comparisons else 0
        }
    
    def save_comparison_result(self, comparison_result: Dict[str, Any], 
                            output_file: str = None, 
                            format_type: str = "json") -> str:
        """
        保存策略比较结果
        
        Args:
            comparison_result: 策略比较结果
            output_file: 输出文件路径，默认为None自动生成
            format_type: 输出格式类型，支持json、markdown和excel
            
        Returns:
            str: 输出文件路径
        """
        if not comparison_result:
            logger.error("比较结果为空，无法保存")
            return ""
            
        # 如果未指定输出文件，生成默认路径
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.result_dir, f"strategy_comparison_{timestamp}.{format_type}")
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format_type.lower() == "json":
                # 保存为JSON
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            
            elif format_type.lower() == "markdown":
                # 生成Markdown内容
                markdown = self._generate_markdown_report(comparison_result)
                
                # 保存为Markdown
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown)
            
            elif format_type.lower() == "excel":
                # 保存为Excel
                self._save_to_excel(comparison_result, output_file)
            
            else:
                logger.error(f"不支持的输出格式: {format_type}")
                return ""
                
            logger.info(f"策略比较结果已保存到: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"保存策略比较结果时出错: {e}")
            return ""
    
    def _generate_markdown_report(self, comparison_result: Dict[str, Any]) -> str:
        """
        生成Markdown格式的比较报告
        
        Args:
            comparison_result: 策略比较结果
            
        Returns:
            str: Markdown内容
        """
        markdown = "# 策略比较分析报告\n\n"
        markdown += f"生成时间: {comparison_result.get('comparison_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n\n"
        
        # 添加策略列表
        strategies = comparison_result.get("strategies", [])
        if strategies:
            markdown += "## 比较策略\n\n"
            markdown += "| ID | 策略名称 |\n"
            markdown += "| --- | --- |\n"
            
            for s in strategies:
                markdown += f"| {s.get('id', '')} | {s.get('name', '')} |\n"
            
            markdown += "\n"
        
        # 添加综合评分
        overall_scores = comparison_result.get("overall_scores", [])
        if overall_scores:
            markdown += "## 综合评分\n\n"
            markdown += "| 排名 | 策略ID | 综合评分 |\n"
            markdown += "| --- | --- | --- |\n"
            
            for score_info in overall_scores:
                markdown += f"| {score_info.get('rank', '')} | {score_info.get('strategy_id', '')} | {score_info.get('overall_score', 0):.4f} |\n"
            
            markdown += "\n"
        
        # 添加各维度排名
        dimension_rankings = comparison_result.get("dimension_rankings", {})
        if dimension_rankings:
            markdown += "## 各维度排名\n\n"
            
            for dim, rankings in dimension_rankings.items():
                markdown += f"### {dim}\n\n"
                markdown += "| 排名 | 策略ID | 得分 |\n"
                markdown += "| --- | --- | --- |\n"
                
                for rank_info in rankings:
                    markdown += f"| {rank_info.get('rank', '')} | {rank_info.get('strategy_id', '')} | {rank_info.get('score', 0):.4f} |\n"
                
                markdown += "\n"
        
        # 添加时间周期表现
        period_results = comparison_result.get("period_results", [])
        if period_results:
            markdown += "## 时间周期表现\n\n"
            
            for period in period_results:
                period_name = period.get("period_name", "")
                start_date = period.get("start_date", "")
                end_date = period.get("end_date", "")
                
                markdown += f"### {period_name} ({start_date} - {end_date})\n\n"
                markdown += "| 策略ID | 选股数量 | 选股比例 | 加权收益 | 胜率 |\n"
                markdown += "| --- | --- | --- | --- | --- |\n"
                
                strategy_results = period.get("strategy_results", [])
                for result in strategy_results:
                    s_id = result.get("strategy_id", "")
                    selected = result.get("selected_stocks", 0)
                    ratio = result.get("selection_ratio", 0)
                    
                    perf = result.get("performance", {})
                    weighted_return = perf.get("weighted_return", 0) if perf else 0
                    weighted_positive = perf.get("weighted_positive_ratio", 0) if perf else 0
                    
                    markdown += f"| {s_id} | {selected} | {ratio:.2%} | {weighted_return:.2%} | {weighted_positive:.2%} |\n"
                
                markdown += "\n"
        
        return markdown
    
    def _save_to_excel(self, comparison_result: Dict[str, Any], output_file: str) -> None:
        """
        保存比较结果到Excel
        
        Args:
            comparison_result: 策略比较结果
            output_file: Excel文件路径
        """
        # 创建Excel写入器
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 保存综合评分
            overall_scores = comparison_result.get("overall_scores", [])
            if overall_scores:
                df_overall = pd.DataFrame(overall_scores)
                df_overall.to_excel(writer, sheet_name="综合评分", index=False)
            
            # 保存各维度排名
            dimension_rankings = comparison_result.get("dimension_rankings", {})
            for dim, rankings in dimension_rankings.items():
                if rankings:
                    df_dim = pd.DataFrame(rankings)
                    df_dim.to_excel(writer, sheet_name=f"维度_{dim}", index=False)
            
            # 保存时间周期表现
            period_results = comparison_result.get("period_results", [])
            for period in period_results:
                period_name = period.get("period_name", "").replace("/", "_")
                strategy_results = period.get("strategy_results", [])
                
                if strategy_results:
                    # 提取基本信息和性能指标
                    data = []
                    for result in strategy_results:
                        row = {
                            "strategy_id": result.get("strategy_id", ""),
                            "strategy_name": result.get("strategy_name", ""),
                            "selected_stocks": result.get("selected_stocks", 0),
                            "selection_ratio": result.get("selection_ratio", 0)
                        }
                        
                        # 添加性能指标
                        perf = result.get("performance", {})
                        if perf:
                            for period_key, period_perf in perf.items():
                                if isinstance(period_perf, dict):
                                    for k, v in period_perf.items():
                                        row[f"{period_key}_{k}"] = v
                                else:
                                    row[period_key] = period_perf
                        
                        data.append(row)
                    
                    df_period = pd.DataFrame(data)
                    df_period.to_excel(writer, sheet_name=f"周期_{period_name[:28]}", index=False)


def main():
    """命令行入口函数"""
    pass

if __name__ == "__main__":
    main() 