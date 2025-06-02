"""
策略组合管理器模块

负责多策略组合执行和结果合并
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import json
import concurrent.futures
from datetime import datetime, timedelta

from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from db.data_manager import DataManager
from utils.logger import get_logger
from utils.decorators import performance_monitor, log_calls, safe_run

logger = get_logger(__name__)


class StrategyCombiner:
    """
    策略组合管理器
    
    负责执行多个策略并合并结果，支持并行执行和加权评分
    """
    
    def __init__(self, strategy_manager: Optional[StrategyManager] = None,
                 strategy_executor: Optional[StrategyExecutor] = None,
                 data_manager: Optional[DataManager] = None):
        """
        初始化策略组合管理器
        
        Args:
            strategy_manager: 策略管理器实例
            strategy_executor: 策略执行器实例
            data_manager: 数据管理器实例
        """
        self.strategy_manager = strategy_manager or StrategyManager()
        self.strategy_executor = strategy_executor or StrategyExecutor()
        self.data_manager = data_manager or DataManager()
        
    @performance_monitor(threshold=10.0)
    @log_calls(level="info")
    def execute_strategies(self, strategy_ids: List[str], 
                          weights: Optional[Dict[str, float]] = None,
                          stock_pool: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          parallel: bool = True,
                          max_workers: int = 5) -> pd.DataFrame:
        """
        执行多个策略并合并结果
        
        Args:
            strategy_ids: 策略ID列表
            weights: 策略权重字典，键为策略ID，值为权重（默认每个策略权重相等）
            stock_pool: 股票池，默认为None表示使用策略过滤器选择
            start_date: 开始日期，默认为近30天
            end_date: 结束日期，默认为当前日期
            parallel: 是否并行执行策略
            max_workers: 最大并行工作线程数
            
        Returns:
            合并后的选股结果DataFrame
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            # 默认取近30天数据
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - 
                        timedelta(days=30)).strftime("%Y-%m-%d")
        
        # 获取策略配置
        strategy_plans = self._get_strategy_plans(strategy_ids)
        if not strategy_plans:
            logger.warning("未找到有效的策略")
            return pd.DataFrame()
            
        # 设置默认权重
        if weights is None:
            weights = {strategy_id: 1.0 for strategy_id in strategy_ids}
        
        # 执行所有策略
        strategy_results = {}
        
        if parallel and len(strategy_plans) > 1:
            # 并行执行策略
            strategy_results = self._execute_parallel(
                strategy_plans, weights, stock_pool, start_date, end_date, max_workers
            )
        else:
            # 串行执行策略
            strategy_results = self._execute_serial(
                strategy_plans, weights, stock_pool, start_date, end_date
            )
        
        # 合并结果
        combined_result = self._combine_results(strategy_results, weights)
        
        # 设置合并策略ID和日期
        combined_id = "_".join(strategy_ids)
        if len(combined_id) > 50:  # 避免ID过长
            combined_id = f"COMBINED_{len(strategy_ids)}_STRATEGIES"
            
        # 添加组合策略信息
        if not combined_result.empty:
            combined_result["strategy_id"] = combined_id
            combined_result["selection_date"] = end_date
            
            # 更新排名
            combined_result = combined_result.sort_values(by="combined_score", ascending=False)
            combined_result["rank"] = range(1, len(combined_result) + 1)
        
        logger.info(f"策略组合执行完成，共找到 {len(combined_result)} 只满足条件的股票")
        return combined_result
    
    @safe_run(default_return={})
    def _get_strategy_plans(self, strategy_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        获取策略执行计划
        
        Args:
            strategy_ids: 策略ID列表
            
        Returns:
            策略执行计划字典，键为策略ID，值为执行计划
        """
        strategy_plans = {}
        
        for strategy_id in strategy_ids:
            strategy_config = self.strategy_manager.get_strategy(strategy_id)
            if strategy_config:
                strategy_plans[strategy_id] = strategy_config
            else:
                logger.warning(f"策略 {strategy_id} 不存在")
                
        return strategy_plans
    
    def _execute_serial(self, strategy_plans: Dict[str, Dict[str, Any]], 
                       weights: Dict[str, float],
                       stock_pool: Optional[List[str]],
                       start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        串行执行策略
        
        Args:
            strategy_plans: 策略执行计划字典
            weights: 策略权重字典
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            策略执行结果字典，键为策略ID，值为结果DataFrame
        """
        results = {}
        
        for strategy_id, strategy_plan in strategy_plans.items():
            logger.info(f"执行策略: {strategy_id}")
            
            result = self.strategy_executor.execute_strategy(
                strategy_plan=strategy_plan,
                stock_pool=stock_pool,
                start_date=start_date,
                end_date=end_date
            )
            
            if not result.empty:
                # 添加权重分数列
                weight = weights.get(strategy_id, 1.0)
                result["weighted_score"] = result["score"] * weight
                
                results[strategy_id] = result
                logger.info(f"策略 {strategy_id} 选出 {len(result)} 只股票")
            else:
                logger.warning(f"策略 {strategy_id} 未选出股票")
                
        return results
    
    def _execute_parallel(self, strategy_plans: Dict[str, Dict[str, Any]], 
                         weights: Dict[str, float],
                         stock_pool: Optional[List[str]],
                         start_date: str, end_date: str,
                         max_workers: int) -> Dict[str, pd.DataFrame]:
        """
        并行执行策略
        
        Args:
            strategy_plans: 策略执行计划字典
            weights: 策略权重字典
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期
            max_workers: 最大并行工作线程数
            
        Returns:
            策略执行结果字典，键为策略ID，值为结果DataFrame
        """
        results = {}
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_strategy = {
                executor.submit(
                    self._execute_single_strategy,
                    strategy_id,
                    strategy_plan,
                    weights.get(strategy_id, 1.0),
                    stock_pool,
                    start_date,
                    end_date
                ): strategy_id
                for strategy_id, strategy_plan in strategy_plans.items()
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy_id = future_to_strategy[future]
                try:
                    strategy_result = future.result()
                    if strategy_result is not None and not strategy_result.empty:
                        results[strategy_id] = strategy_result
                        logger.info(f"策略 {strategy_id} 选出 {len(strategy_result)} 只股票")
                    else:
                        logger.warning(f"策略 {strategy_id} 未选出股票")
                except Exception as e:
                    logger.error(f"执行策略 {strategy_id} 时出错: {e}")
                    
        return results
    
    def _execute_single_strategy(self, strategy_id: str, strategy_plan: Dict[str, Any],
                               weight: float, stock_pool: Optional[List[str]],
                               start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        执行单个策略（用于并行执行）
        
        Args:
            strategy_id: 策略ID
            strategy_plan: 策略执行计划
            weight: 策略权重
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            策略执行结果DataFrame或None
        """
        logger.info(f"执行策略: {strategy_id}")
        
        try:
            # 创建新的执行器实例，避免共享状态
            executor = StrategyExecutor()
            
            result = executor.execute_strategy(
                strategy_plan=strategy_plan,
                stock_pool=stock_pool,
                start_date=start_date,
                end_date=end_date
            )
            
            if not result.empty:
                # 添加权重分数列
                result["weighted_score"] = result["score"] * weight
                return result
                
        except Exception as e:
            logger.error(f"执行策略 {strategy_id} 时出错: {e}")
            
        return None
    
    def _combine_results(self, strategy_results: Dict[str, pd.DataFrame], 
                        weights: Dict[str, float]) -> pd.DataFrame:
        """
        合并多个策略的结果
        
        Args:
            strategy_results: 策略执行结果字典
            weights: 策略权重字典
            
        Returns:
            合并后的结果DataFrame
        """
        if not strategy_results:
            return pd.DataFrame()
            
        # 提取所有选中的股票代码
        all_stocks = set()
        for result in strategy_results.values():
            all_stocks.update(result["stock_code"].tolist())
            
        # 如果没有选中任何股票，返回空结果
        if not all_stocks:
            return pd.DataFrame()
            
        # 获取所有股票的基本信息
        stock_info = self.data_manager.get_stock_info_info(list(all_stocks))
        stock_info_dict = {row["stock_code"]: row for _, row in stock_info.iterrows()}
        
        # 准备合并结果
        combined_data = []
        
        for stock_code in all_stocks:
            stock_data = {
                "stock_code": stock_code,
                "stock_name": stock_info_dict.get(stock_code, {}).get("stock_name", ""),
                "combined_score": 0.0,
                "strategy_count": 0,
                "strategy_details": {}
            }
            
            # 收集该股票在各策略中的表现
            for strategy_id, result in strategy_results.items():
                stock_rows = result[result["stock_code"] == stock_code]
                
                if not stock_rows.empty:
                    stock_row = stock_rows.iloc[0]
                    
                    # 记录策略结果详情
                    stock_data["strategy_details"][strategy_id] = {
                        "rank": int(stock_row["rank"]),
                        "score": float(stock_row["score"]),
                        "weighted_score": float(stock_row["weighted_score"])
                    }
                    
                    # 累加加权分数
                    stock_data["combined_score"] += float(stock_row["weighted_score"])
                    stock_data["strategy_count"] += 1
            
            # 将策略详情转为JSON字符串
            stock_data["strategy_details"] = json.dumps(stock_data["strategy_details"])
            
            # 添加到结果列表
            combined_data.append(stock_data)
        
        # 转换为DataFrame
        combined_df = pd.DataFrame(combined_data)
        
        # 根据组合分数降序排序
        combined_df = combined_df.sort_values(by="combined_score", ascending=False)
        
        return combined_df
    
    @log_calls(level="info")
    def get_strategy_analysis(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """
        获取策略组合分析信息
        
        Args:
            strategy_ids: 策略ID列表
            
        Returns:
            分析信息字典
        """
        analysis = {
            "strategy_count": len(strategy_ids),
            "strategies": {},
            "common_filters": {},
            "indicator_usage": {},
            "period_usage": {}
        }
        
        # 获取策略配置
        strategy_plans = self._get_strategy_plans(strategy_ids)
        
        # 收集策略信息
        common_filters = None
        
        for strategy_id, strategy_plan in strategy_plans.items():
            # 记录策略基本信息
            analysis["strategies"][strategy_id] = {
                "name": strategy_plan.get("name", ""),
                "description": strategy_plan.get("description", ""),
                "condition_count": len(strategy_plan.get("conditions", [])),
                "has_filters": bool(strategy_plan.get("filters", {}))
            }
            
            # 收集指标和周期使用情况
            for condition in strategy_plan.get("conditions", []):
                if condition.get("type") != "logic":
                    indicator_id = condition.get("indicator_id", "")
                    period = condition.get("period", "")
                    
                    if indicator_id:
                        analysis["indicator_usage"][indicator_id] = analysis["indicator_usage"].get(indicator_id, 0) + 1
                        
                    if period:
                        analysis["period_usage"][period] = analysis["period_usage"].get(period, 0) + 1
            
            # 处理过滤器，寻找共同的过滤条件
            filters = strategy_plan.get("filters", {})
            if filters:
                if common_filters is None:
                    common_filters = filters.copy()
                else:
                    # 保留共同的过滤条件
                    for key in list(common_filters.keys()):
                        if key not in filters:
                            common_filters.pop(key, None)
                        elif isinstance(common_filters[key], list) and isinstance(filters[key], list):
                            # 取交集
                            common_filters[key] = list(set(common_filters[key]) & set(filters[key]))
                        elif isinstance(common_filters[key], dict) and isinstance(filters[key], dict):
                            # 取范围交集
                            for range_key in ["min", "max"]:
                                if range_key in common_filters[key] and range_key in filters[key]:
                                    if range_key == "min":
                                        common_filters[key][range_key] = max(common_filters[key][range_key], 
                                                                          filters[key][range_key])
                                    else:
                                        common_filters[key][range_key] = min(common_filters[key][range_key], 
                                                                          filters[key][range_key])
                                elif range_key in common_filters[key]:
                                    common_filters[key].pop(range_key, None)
                        else:
                            # 不同类型，移除该过滤条件
                            common_filters.pop(key, None)
        
        # 记录共同的过滤条件
        if common_filters:
            analysis["common_filters"] = common_filters
            
        # 排序使用情况
        analysis["indicator_usage"] = dict(sorted(analysis["indicator_usage"].items(), 
                                             key=lambda x: x[1], reverse=True))
        analysis["period_usage"] = dict(sorted(analysis["period_usage"].items(), 
                                          key=lambda x: x[1], reverse=True))
        
        return analysis 