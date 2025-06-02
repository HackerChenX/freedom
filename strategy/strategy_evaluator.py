"""
策略评分系统 - 多维度评估指标

提供对选股策略的多维度评分和分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from utils.logger import get_logger
from db.data_manager import DataManager
from utils.decorators import performance_monitor

logger = get_logger(__name__)

class StrategyEvaluator:
    """策略评估器，提供多维度评分和评估功能"""
    
    def __init__(self):
        """初始化策略评估器"""
        self.data_manager = DataManager()
        self.evaluation_cache = {}
    
    @performance_monitor()
    def evaluate_strategy(self, strategy_id: str, 
                        evaluation_period: Optional[Tuple[str, str]] = None,
                        metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        评估选股策略的多维度表现
        
        Args:
            strategy_id: 策略ID
            evaluation_period: 评估周期，格式为 (start_date, end_date)
            metrics: 要评估的指标列表，如果为None则评估所有指标
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 默认评估指标
        default_metrics = [
            "win_rate", "profit_factor", "average_gain", "max_drawdown", 
            "sharpe_ratio", "stability", "consistency", "diversity"
        ]
        
        metrics = metrics or default_metrics
        
        # 检查缓存
        cache_key = f"{strategy_id}_{evaluation_period}_{'-'.join(metrics)}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # 获取策略执行历史
        history = self._get_strategy_history(strategy_id, evaluation_period)
        
        if not history:
            logger.warning(f"未找到策略 {strategy_id} 的历史数据")
            return {"error": "未找到策略历史数据"}
        
        # 计算评估指标
        results = {
            "strategy_id": strategy_id,
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "period": evaluation_period,
            "metrics": {}
        }
        
        # 计算各项指标
        for metric in metrics:
            try:
                metric_value = self._calculate_metric(metric, history)
                results["metrics"][metric] = metric_value
            except Exception as e:
                logger.error(f"计算指标 {metric} 时出错: {e}")
                results["metrics"][metric] = None
        
        # 计算综合评分
        results["total_score"] = self._calculate_total_score(results["metrics"])
        
        # 缓存结果
        self.evaluation_cache[cache_key] = results
        
        return results
    
    def _get_strategy_history(self, strategy_id: str, 
                            period: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        获取策略的历史执行数据
        
        Args:
            strategy_id: 策略ID
            period: 时间周期，格式为 (start_date, end_date)
            
        Returns:
            pd.DataFrame: 历史数据
        """
        # 构建查询参数
        query_params = {"strategy_id": strategy_id}
        
        if period:
            query_params["start_date"] = period[0]
            query_params["end_date"] = period[1]
        
        # 从数据库获取历史数据
        try:
            history_data = self.data_manager.get_strategy_execution_history(**query_params)
            return history_data
        except Exception as e:
            logger.error(f"获取策略 {strategy_id} 历史数据时出错: {e}")
            
            # 返回模拟数据用于测试
            logger.warning("返回模拟数据用于测试")
            return self._generate_mock_history_data(strategy_id, period)
    
    def _generate_mock_history_data(self, strategy_id: str, 
                                  period: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        生成模拟历史数据用于测试
        
        Args:
            strategy_id: 策略ID
            period: 时间周期
            
        Returns:
            pd.DataFrame: 模拟历史数据
        """
        # 生成日期范围
        if period:
            start_date = datetime.strptime(period[0], "%Y-%m-%d")
            end_date = datetime.strptime(period[1], "%Y-%m-%d")
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
        
        # 生成交易日列表（简化处理，实际应该使用真实交易日历）
        dates = []
        current_date = start_date
        while current_date <= end_date:
            # 跳过周末
            if current_date.weekday() < 5:  # 0-4表示周一至周五
                dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        # 每月选股一次
        selection_dates = []
        for date in dates:
            if date[8:10] == "01":  # 每月1日
                selection_dates.append(date)
        
        # 生成模拟数据
        data = []
        for date in selection_dates:
            # 模拟选出的股票数量
            stocks_count = np.random.randint(5, 20)
            
            # 模拟收益率
            for _ in range(stocks_count):
                stock_code = f"60{np.random.randint(1000, 9999)}"
                stock_name = f"模拟股票{np.random.randint(1, 100)}"
                selection_date = date
                
                # 模拟后续表现
                future_return_5d = np.random.normal(0.02, 0.05)  # 均值2%，标准差5%
                future_return_10d = np.random.normal(0.03, 0.08)
                future_return_20d = np.random.normal(0.05, 0.12)
                
                data.append({
                    "strategy_id": strategy_id,
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "selection_date": selection_date,
                    "future_return_5d": future_return_5d,
                    "future_return_10d": future_return_10d,
                    "future_return_20d": future_return_20d,
                    "max_profit": max(future_return_5d, future_return_10d, future_return_20d),
                    "max_loss": min(future_return_5d, future_return_10d, future_return_20d)
                })
        
        return pd.DataFrame(data)
    
    def _calculate_metric(self, metric: str, history: pd.DataFrame) -> float:
        """
        计算指定评估指标
        
        Args:
            metric: 指标名称
            history: 历史数据
            
        Returns:
            float: 指标值
        """
        if metric == "win_rate":
            return self._calculate_win_rate(history)
        elif metric == "profit_factor":
            return self._calculate_profit_factor(history)
        elif metric == "average_gain":
            return self._calculate_average_gain(history)
        elif metric == "max_drawdown":
            return self._calculate_max_drawdown(history)
        elif metric == "sharpe_ratio":
            return self._calculate_sharpe_ratio(history)
        elif metric == "stability":
            return self._calculate_stability(history)
        elif metric == "consistency":
            return self._calculate_consistency(history)
        elif metric == "diversity":
            return self._calculate_diversity(history)
        else:
            logger.warning(f"未知指标: {metric}")
            return 0.0
    
    def _calculate_win_rate(self, history: pd.DataFrame) -> float:
        """
        计算胜率
        
        Args:
            history: 历史数据
            
        Returns:
            float: 胜率 (0-100)
        """
        if history.empty:
            return 0.0
        
        # 使用10日收益率作为胜负判断标准
        if "future_return_10d" in history.columns:
            wins = (history["future_return_10d"] > 0).sum()
            total = len(history)
            return (wins / total) * 100 if total > 0 else 0.0
        else:
            logger.warning("历史数据中缺少future_return_10d列")
            return 0.0
    
    def _calculate_profit_factor(self, history: pd.DataFrame) -> float:
        """
        计算盈亏比
        
        Args:
            history: 历史数据
            
        Returns:
            float: 盈亏比
        """
        if history.empty:
            return 0.0
        
        if "future_return_10d" in history.columns:
            # 分离盈利和亏损
            profits = history[history["future_return_10d"] > 0]["future_return_10d"].sum()
            losses = abs(history[history["future_return_10d"] < 0]["future_return_10d"].sum())
            
            return profits / losses if losses > 0 else profits if profits > 0 else 0.0
        else:
            logger.warning("历史数据中缺少future_return_10d列")
            return 0.0
    
    def _calculate_average_gain(self, history: pd.DataFrame) -> float:
        """
        计算平均收益率
        
        Args:
            history: 历史数据
            
        Returns:
            float: 平均收益率 (%)
        """
        if history.empty:
            return 0.0
        
        if "future_return_10d" in history.columns:
            return history["future_return_10d"].mean() * 100
        else:
            logger.warning("历史数据中缺少future_return_10d列")
            return 0.0
    
    def _calculate_max_drawdown(self, history: pd.DataFrame) -> float:
        """
        计算最大回撤
        
        Args:
            history: 历史数据
            
        Returns:
            float: 最大回撤 (%)
        """
        if history.empty:
            return 0.0
        
        # 按日期分组计算每个选股日期的平均收益
        if "selection_date" in history.columns and "future_return_10d" in history.columns:
            daily_returns = history.groupby("selection_date")["future_return_10d"].mean()
            
            # 计算累积收益
            cumulative_returns = (1 + daily_returns).cumprod()
            
            # 计算回撤
            rolling_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / rolling_max - 1) * 100
            
            return abs(drawdowns.min()) if not drawdowns.empty else 0.0
        else:
            logger.warning("历史数据中缺少selection_date或future_return_10d列")
            return 0.0
    
    def _calculate_sharpe_ratio(self, history: pd.DataFrame) -> float:
        """
        计算夏普比率
        
        Args:
            history: 历史数据
            
        Returns:
            float: 夏普比率
        """
        if history.empty:
            return 0.0
        
        if "future_return_10d" in history.columns:
            # 计算每个选股日期的平均收益
            daily_returns = history.groupby("selection_date")["future_return_10d"].mean()
            
            # 计算年化收益率和标准差
            mean_return = daily_returns.mean() * 252  # 假设一年252个交易日
            std_return = daily_returns.std() * np.sqrt(252)
            
            # 无风险利率假设为3%
            risk_free_rate = 0.03
            
            return (mean_return - risk_free_rate) / std_return if std_return > 0 else 0.0
        else:
            logger.warning("历史数据中缺少future_return_10d列")
            return 0.0
    
    def _calculate_stability(self, history: pd.DataFrame) -> float:
        """
        计算策略稳定性
        
        Args:
            history: 历史数据
            
        Returns:
            float: 稳定性评分 (0-100)
        """
        if history.empty:
            return 0.0
        
        if "future_return_10d" in history.columns:
            # 按选股日期分组计算每个批次的平均收益
            batch_returns = history.groupby("selection_date")["future_return_10d"].mean()
            
            if len(batch_returns) < 2:
                return 50.0  # 数据不足，给予中等评分
            
            # 计算批次间收益的标准差
            std_returns = batch_returns.std()
            
            # 标准差越小，稳定性越高
            # 将标准差映射到0-100的评分，使用逆向映射
            stability_score = 100 * np.exp(-5 * std_returns)
            
            return min(100, max(0, stability_score))
        else:
            logger.warning("历史数据中缺少future_return_10d列")
            return 0.0
    
    def _calculate_consistency(self, history: pd.DataFrame) -> float:
        """
        计算策略一致性
        
        Args:
            history: 历史数据
            
        Returns:
            float: 一致性评分 (0-100)
        """
        if history.empty:
            return 0.0
        
        if "selection_date" in history.columns and "future_return_10d" in history.columns:
            # 按选股日期分组计算每个批次的胜率
            batch_win_rates = history.groupby("selection_date").apply(
                lambda x: (x["future_return_10d"] > 0).mean() * 100
            )
            
            if len(batch_win_rates) < 2:
                return 50.0  # 数据不足，给予中等评分
            
            # 计算批次间胜率的标准差
            std_win_rates = batch_win_rates.std()
            
            # 标准差越小，一致性越高
            # 将标准差映射到0-100的评分，使用逆向映射
            consistency_score = 100 * np.exp(-0.05 * std_win_rates)
            
            return min(100, max(0, consistency_score))
        else:
            logger.warning("历史数据中缺少selection_date或future_return_10d列")
            return 0.0
    
    def _calculate_diversity(self, history: pd.DataFrame) -> float:
        """
        计算策略多样性
        
        Args:
            history: 历史数据
            
        Returns:
            float: 多样性评分 (0-100)
        """
        if history.empty:
            return 0.0
        
        if "stock_code" in history.columns and "industry" in history.columns:
            # 计算行业分布
            industry_counts = history["industry"].value_counts()
            industry_ratios = industry_counts / industry_counts.sum()
            
            # 使用熵来衡量多样性
            entropy = -np.sum(industry_ratios * np.log(industry_ratios)) if len(industry_ratios) > 0 else 0
            
            # 将熵映射到0-100的评分
            max_entropy = np.log(len(industry_ratios)) if len(industry_ratios) > 0 else 1
            diversity_score = 100 * (entropy / max_entropy) if max_entropy > 0 else 0
            
            return diversity_score
        elif "stock_code" in history.columns:
            # 如果没有行业信息，使用股票代码的多样性
            stock_counts = history["stock_code"].value_counts()
            stock_ratios = stock_counts / stock_counts.sum()
            
            # 使用熵来衡量多样性
            entropy = -np.sum(stock_ratios * np.log(stock_ratios)) if len(stock_ratios) > 0 else 0
            
            # 将熵映射到0-100的评分
            max_entropy = np.log(len(stock_ratios)) if len(stock_ratios) > 0 else 1
            diversity_score = 100 * (entropy / max_entropy) if max_entropy > 0 else 0
            
            return diversity_score
        else:
            logger.warning("历史数据中缺少stock_code列")
            return 0.0
    
    def _calculate_total_score(self, metrics: Dict[str, float]) -> float:
        """
        计算综合评分
        
        Args:
            metrics: 各项指标的评分
            
        Returns:
            float: 综合评分 (0-100)
        """
        # 设置各指标权重
        weights = {
            "win_rate": 0.25,
            "profit_factor": 0.20,
            "average_gain": 0.15,
            "max_drawdown": 0.10,
            "sharpe_ratio": 0.10,
            "stability": 0.10,
            "consistency": 0.05,
            "diversity": 0.05
        }
        
        # 标准化指标
        normalized_metrics = {}
        
        for metric, value in metrics.items():
            if value is None:
                continue
                
            if metric == "win_rate":
                # 胜率范围: 0-100
                normalized_metrics[metric] = min(100, max(0, value))
            elif metric == "profit_factor":
                # 盈亏比范围: 0-5，映射到0-100
                normalized_metrics[metric] = min(100, max(0, value * 20))
            elif metric == "average_gain":
                # 平均收益率范围: -100% - 100%，映射到0-100
                normalized_metrics[metric] = min(100, max(0, value + 50))
            elif metric == "max_drawdown":
                # 最大回撤范围: 0-100%，逆向映射到0-100
                normalized_metrics[metric] = min(100, max(0, 100 - value))
            elif metric == "sharpe_ratio":
                # 夏普比率范围: -3 - 3，映射到0-100
                normalized_metrics[metric] = min(100, max(0, (value + 3) * 16.67))
            elif metric in ["stability", "consistency", "diversity"]:
                # 这些指标已经在0-100范围内
                normalized_metrics[metric] = value
            else:
                # 未知指标
                normalized_metrics[metric] = 50.0
        
        # 计算加权平均分
        total_weight = sum(weights.get(m, 0) for m in normalized_metrics.keys())
        
        if total_weight == 0:
            return 50.0  # 默认中等评分
            
        weighted_sum = sum(normalized_metrics[m] * weights.get(m, 0) for m in normalized_metrics.keys())
        
        return weighted_sum / total_weight
    
    def get_strategy_performance_history(self, strategy_id: str, 
                                       period: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        获取策略历史表现数据
        
        Args:
            strategy_id: 策略ID
            period: 时间周期
            
        Returns:
            Dict[str, Any]: 历史表现数据
        """
        # 获取历史数据
        history = self._get_strategy_history(strategy_id, period)
        
        if history.empty:
            logger.warning(f"未找到策略 {strategy_id} 的历史数据")
            return {"error": "未找到策略历史数据"}
        
        # 按时间排序
        if "selection_date" in history.columns:
            history = history.sort_values("selection_date")
        
        # 按选股日期分组计算表现
        if "selection_date" in history.columns and "future_return_10d" in history.columns:
            performance_by_date = history.groupby("selection_date").agg({
                "future_return_10d": ["mean", "std", "min", "max"],
                "stock_code": "count"
            })
            
            # 重命名列
            performance_by_date.columns = [
                "avg_return", "std_return", "min_return", "max_return", "stock_count"
            ]
            
            # 计算胜率
            win_rates = history.groupby("selection_date").apply(
                lambda x: (x["future_return_10d"] > 0).mean() * 100
            ).rename("win_rate")
            
            performance_by_date = performance_by_date.join(win_rates)
            
            # 转换为列表
            performance_history = []
            for date, row in performance_by_date.iterrows():
                performance_history.append({
                    "date": date,
                    "avg_return": row["avg_return"] * 100,  # 转换为百分比
                    "std_return": row["std_return"] * 100,
                    "min_return": row["min_return"] * 100,
                    "max_return": row["max_return"] * 100,
                    "win_rate": row["win_rate"],
                    "stock_count": row["stock_count"]
                })
            
            return {
                "strategy_id": strategy_id,
                "performance_history": performance_history
            }
        else:
            logger.warning("历史数据中缺少selection_date或future_return_10d列")
            return {"error": "历史数据格式不正确"}
    
    def clear_cache(self):
        """清除评估缓存"""
        self.evaluation_cache.clear()
        logger.info("已清除策略评估缓存") 