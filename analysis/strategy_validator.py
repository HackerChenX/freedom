#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_strategies_dir
from strategy.strategy_factory import StrategyFactory
from strategy.strategy_manager import StrategyManager
from strategy.strategy_executor import StrategyExecutor
from db.db_manager import DBManager

# 获取日志记录器
logger = get_logger(__name__)

class StrategyValidator:
    """
    策略验证器 - 验证选股策略有效性
    
    主要功能：
    1. 多周期验证 - 在不同时间周期上验证策略有效性
    2. 参数敏感性分析 - 分析策略参数变化对结果的影响
    3. 策略对比 - 比较不同策略在相同条件下的表现
    4. 业绩跟踪 - 跟踪策略选出股票的后续表现
    5. 失效预警 - 监控策略表现随时间的变化趋势
    """
    
    def __init__(self):
        """初始化策略验证器"""
        logger.info("初始化策略验证器")
        
        # 获取路径
        self.backtest_dir = get_backtest_result_dir()
        self.strategy_dir = get_strategies_dir()
        
        # 确保目录存在
        os.makedirs(self.backtest_dir, exist_ok=True)
        os.makedirs(self.strategy_dir, exist_ok=True)
        
        # 初始化策略管理器
        self.strategy_manager = StrategyManager()
        
        # 初始化策略执行器
        self.strategy_executor = StrategyExecutor()
        
        # 初始化数据库管理器
        self.db_manager = DBManager.get_instance()
        
        logger.info("策略验证器初始化完成")
        
    def validate_multi_period(self, strategy_id: str, periods: List[Tuple[str, str]] = None, 
                           stock_pool: List[str] = None) -> Dict[str, Any]:
        """
        在多个时间周期上验证策略有效性
        
        Args:
            strategy_id: 策略ID
            periods: 验证周期列表，每个元素为(开始日期, 结束日期)的元组，默认为过去一周、一个月、三个月
            stock_pool: 股票池，默认为None表示使用全市场
            
        Returns:
            Dict: 验证结果
        """
        logger.info(f"开始多周期验证策略: {strategy_id}")
        
        try:
            # 获取策略配置
            strategy_config = self.strategy_manager.get_strategy(strategy_id)
            
            if not strategy_config:
                logger.error(f"未找到策略: {strategy_id}")
                return {"error": f"未找到策略: {strategy_id}"}
                
            # 设置默认验证周期
            if periods is None:
                today = datetime.now()
                
                # 过去一周
                week_end = today
                week_start = week_end - timedelta(days=7)
                
                # 过去一个月
                month_end = today
                month_start = month_end - timedelta(days=30)
                
                # 过去三个月
                quarter_end = today
                quarter_start = quarter_end - timedelta(days=90)
                
                periods = [
                    (week_start.strftime("%Y%m%d"), week_end.strftime("%Y%m%d")),
                    (month_start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")),
                    (quarter_start.strftime("%Y%m%d"), quarter_end.strftime("%Y%m%d"))
                ]
            
            # 如果没有指定股票池，获取默认股票池
            if stock_pool is None:
                # 获取全市场股票
                stock_pool = self.db_manager.get_all_stock_codes()
                
                # 限制验证股票数量，避免过长时间
                if len(stock_pool) > 300:
                    import random
                    stock_pool = random.sample(stock_pool, 300)
            
            # 在各个周期上执行验证
            period_results = []
            
            for start_date, end_date in periods:
                # 执行策略
                result = self.strategy_executor.execute_strategy(
                    strategy_config["strategy"], stock_pool, start_date, end_date)
                
                # 准备验证结果
                period_result = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_stocks": len(stock_pool),
                    "selected_stocks": len(result) if isinstance(result, pd.DataFrame) else 0,
                    "selection_ratio": len(result) / len(stock_pool) if isinstance(result, pd.DataFrame) and len(stock_pool) > 0 else 0,
                    "stocks": result.to_dict(orient='records') if isinstance(result, pd.DataFrame) else []
                }
                
                # 计算选出股票的后续表现
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    performance = self._calculate_performance(result['code'].tolist(), end_date)
                    period_result.update(performance)
                
                period_results.append(period_result)
                logger.info(f"周期 {start_date} - {end_date} 选出 {period_result['selected_stocks']} 只股票，"
                          f"选股比例 {period_result['selection_ratio']:.2%}")
            
            # 汇总结果
            validation_result = {
                "strategy_id": strategy_id,
                "strategy_name": strategy_config["strategy"]["name"],
                "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "periods": period_results,
                "summary": self._calculate_multi_period_summary(period_results)
            }
            
            logger.info(f"多周期验证完成，共验证 {len(periods)} 个时间周期")
            return validation_result
            
        except Exception as e:
            logger.error(f"多周期验证时出错: {e}")
            return {"error": str(e)}
            
    def _calculate_performance(self, stock_codes: List[str], 
                             start_date: str, days: int = 10) -> Dict[str, Any]:
        """
        计算股票后续表现
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            days: 后续跟踪天数
            
        Returns:
            Dict: 表现统计
        """
        try:
            # 解析开始日期
            start_date_obj = datetime.strptime(start_date, "%Y%m%d")
            
            # 计算结束日期
            end_date_obj = start_date_obj + timedelta(days=days)
            end_date = end_date_obj.strftime("%Y%m%d")
            
            # 如果结束日期超过今天，使用今天
            today = datetime.now()
            if end_date_obj > today:
                end_date = today.strftime("%Y%m%d")
                
            # 获取股票数据
            returns = []
            for code in stock_codes:
                # 获取股票在指定日期范围内的收盘价
                data = self.db_manager.get_stock_daily_prices(
                    code, start_date, end_date)
                
                if data is None or len(data) < 2:
                    continue
                    
                # 计算收益率
                start_price = data[0]['close']
                end_price = data[-1]['close']
                
                if start_price > 0:
                    return_rate = (end_price - start_price) / start_price
                    returns.append(return_rate)
            
            # 计算统计指标
            if returns:
                avg_return = np.mean(returns)
                max_return = np.max(returns)
                min_return = np.min(returns)
                positive_ratio = np.sum(np.array(returns) > 0) / len(returns)
                
                return {
                    "avg_return": float(avg_return),
                    "max_return": float(max_return),
                    "min_return": float(min_return),
                    "positive_ratio": float(positive_ratio),
                    "valid_stock_count": len(returns)
                }
            
            return {
                "avg_return": 0,
                "max_return": 0,
                "min_return": 0,
                "positive_ratio": 0,
                "valid_stock_count": 0
            }
            
        except Exception as e:
            logger.error(f"计算股票表现时出错: {e}")
            return {
                "avg_return": 0,
                "max_return": 0,
                "min_return": 0,
                "positive_ratio": 0,
                "valid_stock_count": 0,
                "error": str(e)
            }
            
    def _calculate_multi_period_summary(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算多周期验证的汇总统计
        
        Args:
            period_results: 各周期验证结果
            
        Returns:
            Dict: 汇总统计
        """
        if not period_results:
            return {}
            
        # 提取统计数据
        selection_ratios = [p['selection_ratio'] for p in period_results]
        avg_returns = [p.get('avg_return', 0) for p in period_results]
        positive_ratios = [p.get('positive_ratio', 0) for p in period_results]
        
        # 计算汇总统计
        summary = {
            "avg_selection_ratio": float(np.mean(selection_ratios)),
            "selection_ratio_stability": float(np.std(selection_ratios)),
            "avg_return": float(np.mean(avg_returns)),
            "avg_positive_ratio": float(np.mean(positive_ratios)),
            "period_count": len(period_results)
        }
        
        # 评估策略稳定性
        if summary["selection_ratio_stability"] < 0.02 and summary["avg_selection_ratio"] > 0.01:
            summary["stability_assessment"] = "高度稳定"
        elif summary["selection_ratio_stability"] < 0.05:
            summary["stability_assessment"] = "较为稳定"
        else:
            summary["stability_assessment"] = "不稳定"
            
        # 评估策略有效性
        if summary["avg_return"] > 0.05 and summary["avg_positive_ratio"] > 0.6:
            summary["effectiveness_assessment"] = "高度有效"
        elif summary["avg_return"] > 0 and summary["avg_positive_ratio"] > 0.5:
            summary["effectiveness_assessment"] = "有效"
        else:
            summary["effectiveness_assessment"] = "效果不佳"
            
        return summary

    def analyze_parameter_sensitivity(self, strategy_id: str, 
                                   parameters: Dict[str, List[Any]],
                                   start_date: str = None, 
                                   end_date: str = None,
                                   stock_pool: List[str] = None) -> Dict[str, Any]:
        """
        分析策略参数敏感性
        
        Args:
            strategy_id: 策略ID
            parameters: 参数变化范围，格式为 {参数名: [参数值列表]}
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            
        Returns:
            Dict: 分析结果
        """
        logger.info(f"开始分析策略参数敏感性: {strategy_id}")
        
        try:
            # 获取策略配置
            strategy_config = self.strategy_manager.get_strategy(strategy_id)
            
            if not strategy_config:
                logger.error(f"未找到策略: {strategy_id}")
                return {"error": f"未找到策略: {strategy_id}"}
                
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
                
            if start_date is None:
                # 默认分析最近30天
                start_date = (datetime.strptime(end_date, "%Y%m%d") - 
                            timedelta(days=30)).strftime("%Y%m%d")
            
            # 如果没有指定股票池，获取默认股票池
            if stock_pool is None:
                # 获取全市场股票
                stock_pool = self.db_manager.get_all_stock_codes()
                
                # 限制验证股票数量
                if len(stock_pool) > 200:
                    import random
                    stock_pool = random.sample(stock_pool, 200)
            
            # 进行参数敏感性分析
            results = []
            
            # 生成参数组合
            import itertools
            param_names = list(parameters.keys())
            param_values = list(parameters.values())
            
            # 限制参数组合数量，避免过多计算
            max_combinations = 20
            combinations = list(itertools.product(*param_values))
            
            if len(combinations) > max_combinations:
                import random
                combinations = random.sample(combinations, max_combinations)
                
            # 执行各参数组合
            for combo in combinations:
                # 创建参数映射
                param_mapping = dict(zip(param_names, combo))
                
                # 复制策略配置并应用参数
                config_copy = self._apply_parameters(strategy_config, param_mapping)
                
                # 执行策略
                result = self.strategy_executor.execute_strategy(
                    config_copy["strategy"], stock_pool, start_date, end_date)
                
                # 准备结果
                combo_result = {
                    "parameters": param_mapping,
                    "total_stocks": len(stock_pool),
                    "selected_stocks": len(result) if isinstance(result, pd.DataFrame) else 0,
                    "selection_ratio": len(result) / len(stock_pool) if isinstance(result, pd.DataFrame) and len(stock_pool) > 0 else 0
                }
                
                # 计算选出股票的后续表现
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    performance = self._calculate_performance(result['code'].tolist(), end_date)
                    combo_result.update(performance)
                
                results.append(combo_result)
                logger.info(f"参数组合 {param_mapping} 选出 {combo_result['selected_stocks']} 只股票，"
                          f"选股比例 {combo_result['selection_ratio']:.2%}")
            
            # 分析参数敏感性
            sensitivity = self._analyze_sensitivity(results, param_names)
            
            # 汇总结果
            analysis_result = {
                "strategy_id": strategy_id,
                "strategy_name": strategy_config["strategy"]["name"],
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "start_date": start_date,
                "end_date": end_date,
                "parameters_analyzed": param_names,
                "combination_count": len(combinations),
                "combinations": results,
                "sensitivity": sensitivity
            }
            
            logger.info(f"参数敏感性分析完成，共分析 {len(combinations)} 种参数组合")
            return analysis_result
            
        except Exception as e:
            logger.error(f"参数敏感性分析时出错: {e}")
            return {"error": str(e)}
            
    def _apply_parameters(self, strategy_config: Dict[str, Any], 
                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        将参数应用到策略配置中
        
        Args:
            strategy_config: 策略配置
            parameters: 参数映射
            
        Returns:
            Dict: 应用参数后的策略配置
        """
        import copy
        config = copy.deepcopy(strategy_config)
        
        # 遍历条件
        for condition in config["strategy"]["conditions"]:
            if "indicator_id" in condition and "parameters" in condition:
                # 查找需要替换的参数
                for param_name, param_value in parameters.items():
                    # 参数名格式：indicator_id.parameter_name
                    if "." in param_name:
                        indicator_id, param_key = param_name.split(".", 1)
                        
                        # 如果是当前指标，替换参数
                        if condition["indicator_id"] == indicator_id and param_key in condition["parameters"]:
                            condition["parameters"][param_key] = param_value
                    elif param_name in condition["parameters"]:
                        # 直接参数名匹配
                        condition["parameters"][param_name] = param_value
        
        return config
        
    def _analyze_sensitivity(self, results: List[Dict[str, Any]], 
                           param_names: List[str]) -> Dict[str, Any]:
        """
        分析参数敏感性
        
        Args:
            results: 各参数组合的结果
            param_names: 参数名列表
            
        Returns:
            Dict: 敏感性分析结果
        """
        sensitivity = {}
        
        # 对每个参数进行敏感性分析
        for param_name in param_names:
            # 按参数值分组
            param_groups = {}
            
            for result in results:
                if param_name in result["parameters"]:
                    param_value = result["parameters"][param_name]
                    
                    if param_value not in param_groups:
                        param_groups[param_value] = []
                        
                    param_groups[param_value].append(result)
            
            # 计算各参数值组的统计指标
            value_stats = {}
            
            for value, group in param_groups.items():
                selection_ratios = [r["selection_ratio"] for r in group]
                avg_returns = [r.get("avg_return", 0) for r in group]
                
                value_stats[str(value)] = {
                    "avg_selection_ratio": float(np.mean(selection_ratios)),
                    "avg_return": float(np.mean(avg_returns)),
                    "sample_count": len(group)
                }
            
            # 计算参数敏感度
            if len(value_stats) > 1:
                selection_variance = np.var([s["avg_selection_ratio"] for s in value_stats.values()])
                return_variance = np.var([s["avg_return"] for s in value_stats.values()])
                
                sensitivity[param_name] = {
                    "selection_sensitivity": float(selection_variance),
                    "return_sensitivity": float(return_variance),
                    "value_stats": value_stats
                }
        
        # 参数敏感度排序
        sensitivity_ranking = []
        
        for param_name, stats in sensitivity.items():
            sensitivity_ranking.append({
                "parameter": param_name,
                "selection_sensitivity": stats["selection_sensitivity"],
                "return_sensitivity": stats["return_sensitivity"]
            })
            
        # 按选股敏感度排序
        sensitivity_ranking.sort(key=lambda x: x["selection_sensitivity"], reverse=True)
        
        return {
            "parameter_sensitivity": sensitivity,
            "ranking": sensitivity_ranking
        }
        
    def compare_strategies(self, strategy_ids: List[str], 
                         start_date: str = None, 
                         end_date: str = None,
                         stock_pool: List[str] = None) -> Dict[str, Any]:
        """
        比较多个策略在相同条件下的表现
        
        Args:
            strategy_ids: 策略ID列表
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            
        Returns:
            Dict: 比较结果
        """
        logger.info(f"开始比较策略: {strategy_ids}")
        
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
                
            if start_date is None:
                # 默认分析最近30天
                start_date = (datetime.strptime(end_date, "%Y%m%d") - 
                            timedelta(days=30)).strftime("%Y%m%d")
            
            # 如果没有指定股票池，获取默认股票池
            if stock_pool is None:
                # 获取全市场股票
                stock_pool = self.db_manager.get_all_stock_codes()
                
                # 限制验证股票数量
                if len(stock_pool) > 300:
                    import random
                    stock_pool = random.sample(stock_pool, 300)
            
            # 执行各策略并收集结果
            strategy_results = []
            all_selected_stocks = set()
            
            for strategy_id in strategy_ids:
                # 获取策略配置
                strategy_config = self.strategy_manager.get_strategy(strategy_id)
                
                if not strategy_config:
                    logger.warning(f"未找到策略: {strategy_id}")
                    continue
                
                # 执行策略
                result = self.strategy_executor.execute_strategy(
                    strategy_config["strategy"], stock_pool, start_date, end_date)
                
                # 准备结果
                strategy_result = {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_config["strategy"]["name"],
                    "total_stocks": len(stock_pool),
                    "selected_stocks": len(result) if isinstance(result, pd.DataFrame) else 0,
                    "selection_ratio": len(result) / len(stock_pool) if isinstance(result, pd.DataFrame) and len(stock_pool) > 0 else 0,
                    "stock_codes": result['code'].tolist() if isinstance(result, pd.DataFrame) else []
                }
                
                # 更新所有选出的股票集合
                if isinstance(result, pd.DataFrame):
                    all_selected_stocks.update(result['code'].tolist())
                
                # 计算选出股票的后续表现
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    performance = self._calculate_performance(result['code'].tolist(), end_date)
                    strategy_result.update(performance)
                
                strategy_results.append(strategy_result)
                logger.info(f"策略 {strategy_id} 选出 {strategy_result['selected_stocks']} 只股票，"
                          f"选股比例 {strategy_result['selection_ratio']:.2%}")
            
            # 计算策略重叠度
            overlap_matrix = self._calculate_overlap_matrix(strategy_results)
            
            # 比较策略表现
            performance_comparison = self._compare_strategy_performance(strategy_results)
            
            # 汇总结果
            comparison_result = {
                "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "start_date": start_date,
                "end_date": end_date,
                "total_strategies": len(strategy_ids),
                "valid_strategies": len(strategy_results),
                "total_stock_pool": len(stock_pool),
                "total_selected_stocks": len(all_selected_stocks),
                "strategy_results": strategy_results,
                "overlap_matrix": overlap_matrix,
                "performance_comparison": performance_comparison
            }
            
            logger.info(f"策略比较完成，共比较 {len(strategy_results)} 个策略")
            return comparison_result
            
        except Exception as e:
            logger.error(f"比较策略时出错: {e}")
            return {"error": str(e)}
            
    def _calculate_overlap_matrix(self, strategy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算策略重叠度矩阵
        
        Args:
            strategy_results: 各策略的结果
            
        Returns:
            Dict: 重叠度矩阵
        """
        # 创建策略索引映射
        strategy_indices = {r["strategy_id"]: i for i, r in enumerate(strategy_results)}
        
        # 创建重叠度矩阵
        n = len(strategy_results)
        overlap_matrix = np.zeros((n, n))
        
        # 计算两两策略的重叠度
        for i in range(n):
            for j in range(i, n):
                stocks_i = set(strategy_results[i]["stock_codes"])
                stocks_j = set(strategy_results[j]["stock_codes"])
                
                # 避免除零错误
                if not stocks_i or not stocks_j:
                    overlap_matrix[i, j] = overlap_matrix[j, i] = 0
                    continue
                
                # 计算Jaccard相似度
                intersection = len(stocks_i.intersection(stocks_j))
                union = len(stocks_i.union(stocks_j))
                
                overlap = intersection / union if union > 0 else 0
                overlap_matrix[i, j] = overlap_matrix[j, i] = overlap
        
        # 转换为字典格式
        matrix_dict = {
            "strategies": [r["strategy_id"] for r in strategy_results],
            "strategy_names": [r["strategy_name"] for r in strategy_results],
            "overlap_values": overlap_matrix.tolist()
        }
        
        return matrix_dict
        
    def _compare_strategy_performance(self, strategy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        比较策略表现
        
        Args:
            strategy_results: 各策略的结果
            
        Returns:
            Dict: 表现比较结果
        """
        # 提取关键指标
        performance_metrics = []
        
        for result in strategy_results:
            metrics = {
                "strategy_id": result["strategy_id"],
                "strategy_name": result["strategy_name"],
                "selection_ratio": result["selection_ratio"],
                "avg_return": result.get("avg_return", 0),
                "positive_ratio": result.get("positive_ratio", 0)
            }
            performance_metrics.append(metrics)
        
        # 按平均收益率排序
        ranked_by_return = sorted(
            performance_metrics, 
            key=lambda x: x["avg_return"], 
            reverse=True
        )
        
        # 按正收益率排序
        ranked_by_positive = sorted(
            performance_metrics, 
            key=lambda x: x["positive_ratio"], 
            reverse=True
        )
        
        # 综合评分
        for metrics in performance_metrics:
            # 归一化处理
            max_return = max(r["avg_return"] for r in performance_metrics) if performance_metrics else 0
            max_positive = max(r["positive_ratio"] for r in performance_metrics) if performance_metrics else 0
            
            # 避免除零错误
            norm_return = metrics["avg_return"] / max_return if max_return > 0 else 0
            norm_positive = metrics["positive_ratio"] / max_positive if max_positive > 0 else 0
            
            # 综合评分 (70% 收益率 + 30% 正收益率)
            metrics["composite_score"] = 0.7 * norm_return + 0.3 * norm_positive
        
        # 按综合评分排序
        ranked_by_score = sorted(
            performance_metrics, 
            key=lambda x: x["composite_score"], 
            reverse=True
        )
        
        return {
            "ranked_by_return": ranked_by_return,
            "ranked_by_positive_ratio": ranked_by_positive,
            "ranked_by_composite_score": ranked_by_score
        }

    def monitor_strategy_effectiveness(self, strategy_id: str, 
                                    monitoring_periods: int = 6, 
                                    period_length: int = 30) -> Dict[str, Any]:
        """
        监测策略有效性随时间的变化趋势，检测策略是否失效
        
        Args:
            strategy_id: 策略ID
            monitoring_periods: 监测周期数量
            period_length: 每个周期的天数
            
        Returns:
            Dict: 监测结果
        """
        logger.info(f"开始监测策略有效性: {strategy_id}")
        
        try:
            # 获取策略配置
            strategy_config = self.strategy_manager.get_strategy(strategy_id)
            
            if not strategy_config:
                logger.error(f"未找到策略: {strategy_id}")
                return {"error": f"未找到策略: {strategy_id}"}
            
            # 创建监测周期
            today = datetime.now()
            periods = []
            
            for i in range(monitoring_periods):
                end_date = today - timedelta(days=i * period_length)
                start_date = end_date - timedelta(days=period_length)
                
                periods.append((
                    start_date.strftime("%Y%m%d"),
                    end_date.strftime("%Y%m%d")
                ))
            
            # 获取部分股票作为样本
            stock_pool = self.db_manager.get_all_stock_codes()
            
            # 限制股票数量，提高效率
            if len(stock_pool) > 200:
                import random
                stock_pool = random.sample(stock_pool, 200)
            
            # 在各个周期上执行策略
            period_results = []
            
            for start_date, end_date in periods:
                # 执行策略
                result = self.strategy_executor.execute_strategy(
                    strategy_config["strategy"], stock_pool, start_date, end_date)
                
                # 准备结果
                period_result = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_stocks": len(stock_pool),
                    "selected_stocks": len(result) if isinstance(result, pd.DataFrame) else 0,
                    "selection_ratio": len(result) / len(stock_pool) if isinstance(result, pd.DataFrame) and len(stock_pool) > 0 else 0,
                }
                
                # 计算选出股票的后续表现
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    performance = self._calculate_performance(result['code'].tolist(), end_date)
                    period_result.update(performance)
                
                period_results.append(period_result)
                logger.info(f"周期 {start_date} - {end_date} 选出 {period_result['selected_stocks']} 只股票，"
                          f"选股比例 {period_result['selection_ratio']:.2%}")
            
            # 分析有效性趋势
            effectiveness_trend = self._analyze_effectiveness_trend(period_results)
            
            # 汇总结果
            monitoring_result = {
                "strategy_id": strategy_id,
                "strategy_name": strategy_config["strategy"]["name"],
                "monitoring_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "monitoring_periods": monitoring_periods,
                "period_length": period_length,
                "period_results": period_results,
                "effectiveness_trend": effectiveness_trend
            }
            
            logger.info(f"策略有效性监测完成，共监测 {len(period_results)} 个周期")
            return monitoring_result
            
        except Exception as e:
            logger.error(f"监测策略有效性时出错: {e}")
            return {"error": str(e)}
            
    def _analyze_effectiveness_trend(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析策略有效性趋势
        
        Args:
            period_results: 各周期的结果
            
        Returns:
            Dict: 趋势分析结果
        """
        # 提取关键指标
        selection_ratios = [r["selection_ratio"] for r in period_results]
        avg_returns = [r.get("avg_return", 0) for r in period_results]
        positive_ratios = [r.get("positive_ratio", 0) for r in period_results]
        
        # 计算趋势
        selection_trend = self._calculate_trend(selection_ratios)
        return_trend = self._calculate_trend(avg_returns)
        positive_trend = self._calculate_trend(positive_ratios)
        
        # 评估策略是否失效
        is_declining = (return_trend["slope"] < -0.01 and 
                      positive_trend["slope"] < -0.05)
        
        # 失效风险评估
        if is_declining and return_trend["p_value"] < 0.1:
            effectiveness_status = "高风险"
            recommendation = "建议重新优化策略参数或暂停使用"
        elif is_declining:
            effectiveness_status = "中风险"
            recommendation = "建议密切监控策略表现"
        elif return_trend["slope"] < 0:
            effectiveness_status = "低风险"
            recommendation = "策略表现有轻微下降趋势，但仍在可接受范围内"
        else:
            effectiveness_status = "稳定"
            recommendation = "策略表现稳定，可以继续使用"
        
        return {
            "selection_ratio_trend": selection_trend,
            "return_trend": return_trend,
            "positive_ratio_trend": positive_trend,
            "is_declining": is_declining,
            "effectiveness_status": effectiveness_status,
            "recommendation": recommendation
        }
        
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """
        计算数据趋势
        
        Args:
            values: 数据列表
            
        Returns:
            Dict: 趋势分析结果
        """
        try:
            # 确保有足够的数据点
            if len(values) < 2:
                return {
                    "slope": 0,
                    "p_value": 1.0,
                    "r_squared": 0,
                    "trend_description": "数据点不足"
                }
            
            # 创建X值（时间点）
            x = np.arange(len(values))
            
            # 线性回归
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # 计算R方值
            r_squared = r_value ** 2
            
            # 趋势描述
            if p_value < 0.05:
                if slope > 0.01:
                    trend_description = "显著上升"
                elif slope < -0.01:
                    trend_description = "显著下降"
                else:
                    trend_description = "基本稳定"
            else:
                trend_description = "无明显趋势"
            
            return {
                "slope": float(slope),
                "p_value": float(p_value),
                "r_squared": float(r_squared),
                "trend_description": trend_description
            }
            
        except Exception as e:
            logger.error(f"计算趋势时出错: {e}")
            return {
                "slope": 0,
                "p_value": 1.0,
                "r_squared": 0,
                "trend_description": "计算出错",
                "error": str(e)
            }

def main():
    """命令行入口函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="策略验证工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 多周期验证子命令
    multi_period_parser = subparsers.add_parser('multi_period', help='多周期验证')
    multi_period_parser.add_argument('-s', '--strategy', required=True, help='策略ID')
    multi_period_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    multi_period_parser.add_argument('-p', '--pool', help='股票池文件路径')
    
    # 参数敏感性分析子命令
    sensitivity_parser = subparsers.add_parser('sensitivity', help='参数敏感性分析')
    sensitivity_parser.add_argument('-s', '--strategy', required=True, help='策略ID')
    sensitivity_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    sensitivity_parser.add_argument('-p', '--parameters', required=True, 
                                  help='参数配置文件路径，JSON格式')
    
    # 策略对比子命令
    compare_parser = subparsers.add_parser('compare', help='策略对比')
    compare_parser.add_argument('-s', '--strategies', required=True, 
                              help='策略ID列表，用逗号分隔')
    compare_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    
    # 有效性监测子命令
    monitor_parser = subparsers.add_parser('monitor', help='策略有效性监测')
    monitor_parser.add_argument('-s', '--strategy', required=True, help='策略ID')
    monitor_parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    monitor_parser.add_argument('-p', '--periods', type=int, default=6, help='监测周期数量')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = StrategyValidator()
    
    if args.command == 'multi_period':
        # 读取股票池
        stock_pool = None
        if args.pool:
            try:
                with open(args.pool, 'r', encoding='utf-8') as f:
                    stock_pool = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"读取股票池文件时出错: {e}")
        
        # 执行多周期验证
        result = validator.validate_multi_period(args.strategy, stock_pool=stock_pool)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"多周期验证结果已保存到: {args.output}")
        
    elif args.command == 'sensitivity':
        # 读取参数配置
        try:
            with open(args.parameters, 'r', encoding='utf-8') as f:
                parameters = json.load(f)
        except Exception as e:
            logger.error(f"读取参数配置文件时出错: {e}")
            return
        
        # 执行参数敏感性分析
        result = validator.analyze_parameter_sensitivity(args.strategy, parameters)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"参数敏感性分析结果已保存到: {args.output}")
        
    elif args.command == 'compare':
        # 解析策略ID列表
        strategy_ids = [s.strip() for s in args.strategies.split(',') if s.strip()]
        
        # 执行策略对比
        result = validator.compare_strategies(strategy_ids)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"策略对比结果已保存到: {args.output}")
        
    elif args.command == 'monitor':
        # 执行策略有效性监测
        result = validator.monitor_strategy_effectiveness(args.strategy, args.periods)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"策略有效性监测结果已保存到: {args.output}")
        
    else:
        parser.print_help()
    
if __name__ == "__main__":
    main() 