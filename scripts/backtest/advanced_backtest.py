#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from analysis.pattern_recognition_analyzer import PatternRecognitionAnalyzer
from scripts.backtest.pattern_backtest import PatternBacktester
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.path_utils import get_result_path, ensure_dir

logger = get_logger("advanced_backtest")

class AdvancedBacktester(PatternBacktester):
    """
    高级形态回测器
    
    扩展基础形态回测器，增加更多高级功能:
    1. 多周期组合信号回测
    2. 形态组合条件回测
    3. 指标交叉验证
    4. 回测性能分析
    """
    
    def __init__(self, indicators: List[str] = None, periods: List[str] = None):
        """初始化高级回测器"""
        super().__init__(indicators=indicators, periods=periods)
        
        # 回测配置
        self.config = {
            "min_pattern_strength": 60,       # 最小形态强度
            "min_pattern_count": 1,           # 最小形态数量
            "combination_mode": "any",        # 组合模式：any-任一符合, all-全部符合
            "cross_validation": True,         # 是否启用指标交叉验证
            "require_multi_period": False,    # 是否要求多周期确认
            "score_threshold": 60,            # 评分阈值
            "profit_taking": 0.05,            # 止盈比例
            "stop_loss": -0.03                # 止损比例
        }
        
        # 回测结果
        self.advanced_results = {
            "stocks": {},
            "patterns": {},
            "combinations": {},
            "performance": {},
            "config": self.config.copy()
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        设置回测配置
        
        Args:
            config: 配置字典
        """
        self.config.update(config)
        self.advanced_results["config"] = self.config.copy()
        logger.info(f"已更新回测配置: {self.config}")
    
    def backtest_with_combination(self, stock_codes: List[str], 
                                start_date: str, 
                                end_date: str,
                                pattern_combinations: List[Dict[str, Any]],
                                forward_days: int = 5,
                                threshold: float = 0.02) -> Dict[str, Any]:
        """
        使用形态组合进行回测
        
        Args:
            stock_codes: 股票代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            pattern_combinations: 形态组合列表，每个组合是一个字典，包含indicator_id, pattern_id, period
            forward_days: 向前看多少天
            threshold: 涨幅阈值
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        logger.info(f"开始组合形态回测，共 {len(stock_codes)} 只股票，{len(pattern_combinations)} 个形态组合")
        
        # 设置基础回测结果的配置
        self.backtest_results["config"].update({
            "start_date": start_date,
            "end_date": end_date,
            "forward_days": forward_days,
            "threshold": threshold,
            "pattern_combinations": pattern_combinations
        })
        
        # 设置高级回测结果的配置
        self.advanced_results["config"].update({
            "start_date": start_date,
            "end_date": end_date,
            "forward_days": forward_days,
            "threshold": threshold,
            "pattern_combinations": pattern_combinations
        })
        
        # 组合回测结果
        combination_results = {}
        
        # 遍历股票列表进行回测
        for stock_code in stock_codes:
            try:
                logger.info(f"回测股票 {stock_code}")
                
                # 获取买点日期列表
                buy_dates = self._get_buy_dates(stock_code, start_date, end_date)
                
                if not buy_dates:
                    logger.warning(f"股票 {stock_code} 在指定日期范围内没有买点")
                    continue
                
                # 分析每个买点
                for buy_date in buy_dates:
                    # 获取买点前后的数据
                    pre_buy_date = (datetime.strptime(buy_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
                    post_buy_date = (datetime.strptime(buy_date, "%Y-%m-%d") + timedelta(days=forward_days + 10)).strftime("%Y-%m-%d")
                    
                    # 获取不同周期的数据
                    period_data = {}
                    for period in self.periods:
                        try:
                            period_data[period] = self.db.get_kline_data(
                                stock_code=stock_code,
                                start_date=pre_buy_date,
                                end_date=post_buy_date,
                                period=period
                            )
                        except Exception as e:
                            logger.error(f"获取 {stock_code} {period} 周期数据时出错: {e}")
                    
                    # 对齐数据到买点日期
                    aligned_data = self._align_data_to_date(period_data, buy_date)
                    
                    if not aligned_data:
                        logger.warning(f"股票 {stock_code} 在买点 {buy_date} 没有对齐的数据，跳过此买点")
                        continue
                    
                    # 获取股票名称
                    stock_name = self.db.get_stock_name(stock_code)
                    
                    # 使用形态识别分析器分析
                    analysis_result = self.analyzer.analyze(
                        data=aligned_data,
                        stock_code=stock_code,
                        stock_name=stock_name
                    )
                    
                    # 检测是否符合形态组合条件
                    is_match, matched_patterns = self._match_pattern_combinations(
                        analysis_result, 
                        pattern_combinations,
                        min_strength=self.config["min_pattern_strength"]
                    )
                    
                    if not is_match:
                        continue
                    
                    # 获取买点后的实际涨跌幅
                    future_change, price_series = self._calculate_detailed_future_change(
                        stock_code=stock_code,
                        buy_date=buy_date,
                        forward_days=forward_days
                    )
                    
                    # 计算最大收益和最大回撤
                    max_profit, max_drawdown, hold_days = self._calculate_price_metrics(
                        price_series, 
                        profit_taking=self.config["profit_taking"],
                        stop_loss=self.config["stop_loss"]
                    )
                    
                    # 判断买点是否成功
                    is_success = future_change >= threshold
                    
                    # 记录组合结果
                    combination_key = self._generate_combination_key(pattern_combinations)
                    
                    if combination_key not in combination_results:
                        combination_results[combination_key] = {
                            "patterns": pattern_combinations,
                            "count": 0,
                            "success_count": 0,
                            "total_gain": 0,
                            "instances": [],
                            "max_profit_total": 0,
                            "max_drawdown_total": 0,
                            "avg_hold_days": 0
                        }
                    
                    # 更新组合统计
                    combination_results[combination_key]["count"] += 1
                    if is_success:
                        combination_results[combination_key]["success_count"] += 1
                    combination_results[combination_key]["total_gain"] += future_change
                    combination_results[combination_key]["max_profit_total"] += max_profit
                    combination_results[combination_key]["max_drawdown_total"] += max_drawdown
                    combination_results[combination_key]["avg_hold_days"] += hold_days
                    
                    # 记录实例
                    combination_results[combination_key]["instances"].append({
                        "stock_code": stock_code,
                        "stock_name": stock_name,
                        "date": buy_date,
                        "future_change": future_change,
                        "is_success": is_success,
                        "max_profit": max_profit,
                        "max_drawdown": max_drawdown,
                        "hold_days": hold_days,
                        "matched_patterns": matched_patterns
                    })
                    
            except Exception as e:
                logger.error(f"回测股票 {stock_code} 时出错: {e}")
        
        # 计算组合的成功率和平均收益
        for key, result in combination_results.items():
            if result["count"] > 0:
                result["success_rate"] = result["success_count"] / result["count"]
                result["avg_gain"] = result["total_gain"] / result["count"]
                result["avg_max_profit"] = result["max_profit_total"] / result["count"]
                result["avg_max_drawdown"] = result["max_drawdown_total"] / result["count"]
                result["avg_hold_days"] = result["avg_hold_days"] / result["count"]
        
        # 保存到高级回测结果
        self.advanced_results["combinations"] = combination_results
        
        # 计算整体性能指标
        self._calculate_overall_performance(combination_results)
        
        return self.advanced_results
    
    def _align_data_to_date(self, period_data: Dict[str, pd.DataFrame], 
                          target_date: str) -> Dict[str, pd.DataFrame]:
        """
        将各周期数据对齐到目标日期
        
        Args:
            period_data: 各周期数据
            target_date: 目标日期
            
        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据
        """
        aligned_data = {}
        target_date_dt = pd.to_datetime(target_date)
        
        for period, data in period_data.items():
            if data is None or data.empty:
                continue
                
            # 找到不晚于目标日期的最后一个日期
            if isinstance(data.index, pd.DatetimeIndex):
                mask = data.index <= target_date_dt
                if not mask.any():
                    logger.warning(f"目标日期 {target_date} 在 {period} 周期数据中不存在")
                    continue
                
                buy_idx = mask.sum() - 1
                aligned_data[period] = data.iloc[:buy_idx+1]
            else:
                # 非日期索引的处理
                aligned_data[period] = data
        
        return aligned_data
    
    def _match_pattern_combinations(self, analysis_result: Dict[str, Any], 
                                  pattern_combinations: List[Dict[str, Any]],
                                  min_strength: float = 60) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        检测是否符合形态组合条件
        
        Args:
            analysis_result: 分析结果
            pattern_combinations: 形态组合列表
            min_strength: 最小形态强度
            
        Returns:
            Tuple[bool, List[Dict[str, Any]]]: (是否匹配, 匹配的形态列表)
        """
        # 如果没有指定形态组合，则使用评分门槛
        if not pattern_combinations:
            score = analysis_result.get("scores", {}).get("total_score", 0)
            return score >= self.config["score_threshold"], []
        
        matched_patterns = []
        
        for combination in pattern_combinations:
            indicator_id = combination.get("indicator_id")
            pattern_id = combination.get("pattern_id")
            period = combination.get("period")
            
            # 检查指定周期的形态
            if period and period in analysis_result.get("periods", {}):
                period_patterns = analysis_result["periods"][period].get("patterns", [])
                
                for pattern in period_patterns:
                    # 匹配指标和形态ID
                    if (pattern.get("indicator_id") == indicator_id and 
                        pattern.get("pattern_id") == pattern_id and 
                        pattern.get("strength", 0) >= min_strength):
                        
                        matched_patterns.append({
                            "indicator_id": indicator_id,
                            "pattern_id": pattern_id,
                            "period": period,
                            "strength": pattern.get("strength", 0),
                            "display_name": pattern.get("display_name", "")
                        })
                        break
        
        # 根据组合模式判断是否匹配
        if self.config["combination_mode"] == "all":
            # 必须所有形态都匹配
            return len(matched_patterns) == len(pattern_combinations), matched_patterns
        else:
            # 至少一个形态匹配
            return len(matched_patterns) >= self.config["min_pattern_count"], matched_patterns
    
    def _calculate_detailed_future_change(self, stock_code: str, buy_date: str, 
                                        forward_days: int) -> Tuple[float, pd.Series]:
        """
        计算未来涨跌幅和价格序列
        
        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            forward_days: 向前看的天数
            
        Returns:
            Tuple[float, pd.Series]: (涨跌幅, 价格序列)
        """
        try:
            # 获取买点日期的收盘价
            buy_price_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=buy_date,
                end_date=buy_date,
                period="DAILY"
            )
            
            if buy_price_data is None or buy_price_data.empty:
                logger.warning(f"无法获取 {stock_code} 在 {buy_date} 的价格数据")
                return 0, pd.Series()
            
            buy_price = buy_price_data['close'].iloc[0]
            
            # 获取未来日期的价格序列
            future_date = (datetime.strptime(buy_date, "%Y-%m-%d") + timedelta(days=forward_days)).strftime("%Y-%m-%d")
            future_end_date = (datetime.strptime(future_date, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
            
            future_price_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=buy_date,
                end_date=future_end_date,
                period="DAILY"
            )
            
            if future_price_data is None or future_price_data.empty:
                logger.warning(f"无法获取 {stock_code} 在 {future_date} 的价格数据")
                return 0, pd.Series()
            
            # 计算价格涨跌幅序列
            price_series = future_price_data['close'].copy()
            
            # 找到不早于future_date的第一个日期
            future_date_dt = pd.to_datetime(future_date)
            future_price = None
            
            if isinstance(future_price_data.index, pd.DatetimeIndex):
                # 找到距离目标日期最近的行
                closest_idx = (future_price_data.index - future_date_dt).abs().argmin()
                future_price = future_price_data['close'].iloc[closest_idx]
            else:
                # 取最后一个价格作为未来价格
                future_price = future_price_data['close'].iloc[-1]
            
            # 计算涨跌幅
            future_change = 0
            if buy_price > 0:
                future_change = (future_price - buy_price) / buy_price
            
            return future_change, price_series
            
        except Exception as e:
            logger.error(f"计算未来涨跌幅时出错: {e}")
            return 0, pd.Series()
    
    def _calculate_price_metrics(self, price_series: pd.Series, 
                               profit_taking: float = 0.05, 
                               stop_loss: float = -0.03) -> Tuple[float, float, int]:
        """
        计算价格序列的关键指标
        
        Args:
            price_series: 价格序列
            profit_taking: 止盈比例
            stop_loss: 止损比例
            
        Returns:
            Tuple[float, float, int]: (最大收益, 最大回撤, 持有天数)
        """
        if price_series.empty or len(price_series) < 2:
            return 0, 0, 0
        
        # 初始价格
        initial_price = price_series.iloc[0]
        
        # 计算涨跌幅序列
        returns = (price_series / initial_price) - 1
        
        # 最大收益
        max_profit = returns.max()
        
        # 最大回撤
        peak = returns.cummax()
        drawdown = returns - peak
        max_drawdown = drawdown.min()
        
        # 计算持有天数 (考虑止盈止损)
        hold_days = len(returns)
        
        for i, ret in enumerate(returns):
            if ret >= profit_taking or ret <= stop_loss:
                hold_days = i + 1
                break
        
        return max_profit, max_drawdown, hold_days
    
    def _generate_combination_key(self, pattern_combinations: List[Dict[str, Any]]) -> str:
        """
        生成形态组合的唯一键
        
        Args:
            pattern_combinations: 形态组合列表
            
        Returns:
            str: 组合键
        """
        parts = []
        
        for combination in sorted(pattern_combinations, key=lambda x: (x.get("indicator_id", ""), x.get("pattern_id", ""))):
            indicator_id = combination.get("indicator_id", "")
            pattern_id = combination.get("pattern_id", "")
            period = combination.get("period", "")
            parts.append(f"{indicator_id}_{pattern_id}_{period}")
        
        return "|".join(parts)
    
    def _calculate_overall_performance(self, combination_results: Dict[str, Dict[str, Any]]) -> None:
        """
        计算整体性能指标
        
        Args:
            combination_results: 组合回测结果
        """
        total_trades = 0
        total_success = 0
        total_gain = 0
        total_max_profit = 0
        total_max_drawdown = 0
        total_hold_days = 0
        
        # 汇总所有交易
        for key, result in combination_results.items():
            total_trades += result["count"]
            total_success += result["success_count"]
            total_gain += result["total_gain"]
            total_max_profit += result["max_profit_total"]
            total_max_drawdown += result["max_drawdown_total"]
            total_hold_days += result["avg_hold_days"] * result["count"]
        
        # 计算平均值
        avg_gain = total_gain / total_trades if total_trades > 0 else 0
        success_rate = total_success / total_trades if total_trades > 0 else 0
        avg_max_profit = total_max_profit / total_trades if total_trades > 0 else 0
        avg_max_drawdown = total_max_drawdown / total_trades if total_trades > 0 else 0
        avg_hold_days = total_hold_days / total_trades if total_trades > 0 else 0
        
        # 计算收益风险比
        risk_reward_ratio = abs(avg_max_profit / avg_max_drawdown) if avg_max_drawdown != 0 else 0
        
        # 计算年化收益率 (假设一年250个交易日)
        yearly_return = ((1 + avg_gain) ** (250 / avg_hold_days)) - 1 if avg_hold_days > 0 else 0
        
        # 计算夏普比率 (简化版，假设无风险利率为0)
        returns_std = np.std([inst["future_change"] for result in combination_results.values() for inst in result["instances"]])
        sharpe_ratio = avg_gain / returns_std if returns_std > 0 else 0
        
        # 保存性能指标
        self.advanced_results["performance"] = {
            "total_trades": total_trades,
            "success_rate": success_rate,
            "avg_gain": avg_gain,
            "avg_max_profit": avg_max_profit,
            "avg_max_drawdown": avg_max_drawdown,
            "avg_hold_days": avg_hold_days,
            "risk_reward_ratio": risk_reward_ratio,
            "yearly_return": yearly_return,
            "sharpe_ratio": sharpe_ratio
        }

    def backtest_with_pattern_combination(self, stock_codes: List[str], 
                                         start_date: str, 
                                         end_date: str,
                                         forward_days: int = 5,
                                         min_pattern_strength: float = 60.0,
                                         require_multiple_indicators: bool = True) -> Dict[str, Any]:
        """
        使用形态组合进行回测，自动识别和验证形态组合
        
        Args:
            stock_codes: 股票代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            forward_days: 向前看多少天
            min_pattern_strength: 最小形态强度
            require_multiple_indicators: 是否要求多指标确认
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        logger.info(f"开始形态组合回测，共 {len(stock_codes)} 只股票")
        
        # 设置配置
        self.config["min_pattern_strength"] = min_pattern_strength
        
        # 存储回测结果
        results = {
            "config": {
                "start_date": start_date,
                "end_date": end_date,
                "forward_days": forward_days,
                "min_pattern_strength": min_pattern_strength,
                "require_multiple_indicators": require_multiple_indicators
            },
            "stock_results": {},
            "pattern_combinations": {},
            "performance": {
                "success_rate": 0.0,
                "avg_gain": 0.0,
                "max_gain": 0.0,
                "avg_loss": 0.0,
                "max_loss": 0.0,
                "profit_factor": 0.0
            }
        }
        
        total_combinations = 0
        successful_combinations = 0
        total_gain = 0.0
        total_loss = 0.0
        max_gain = 0.0
        max_loss = 0.0
        
        # 遍历股票
        for stock_code in stock_codes:
            try:
                logger.info(f"回测股票 {stock_code}")
                
                # 获取买点日期列表
                buy_dates = self._get_buy_dates(stock_code, start_date, end_date)
                
                if not buy_dates:
                    logger.warning(f"股票 {stock_code} 在指定日期范围内没有买点")
                    continue
                
                # 分析每个买点
                stock_results = []
                
                for buy_date in buy_dates:
                    # 获取买点前后的数据
                    pre_buy_date = (datetime.strptime(buy_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
                    post_buy_date = (datetime.strptime(buy_date, "%Y-%m-%d") + timedelta(days=forward_days + 10)).strftime("%Y-%m-%d")
                    
                    # 获取不同周期的数据
                    period_data = {}
                    for period in self.periods:
                        try:
                            period_data[period] = self.db.get_kline_data(
                                stock_code=stock_code,
                                start_date=pre_buy_date,
                                end_date=post_buy_date,
                                period=period
                            )
                        except Exception as e:
                            logger.error(f"获取 {stock_code} {period} 周期数据时出错: {e}")
                    
                    # 对齐数据到买点日期
                    aligned_data = self._align_data_to_date(period_data, buy_date)
                    
                    if not aligned_data:
                        logger.warning(f"股票 {stock_code} 在买点 {buy_date} 没有对齐的数据，跳过此买点")
                        continue
                    
                    # 获取股票名称
                    stock_name = self.db.get_stock_name(stock_code)
                    
                    # 使用形态识别分析器分析
                    analysis_result = self.analyzer.analyze(
                        data=aligned_data,
                        stock_code=stock_code,
                        stock_name=stock_name
                    )
                    
                    # 获取形态组合
                    pattern_combinations = self.analyzer.get_pattern_combinations(
                        min_occurrence=1,
                        min_strength=min_pattern_strength
                    )
                    
                    if not pattern_combinations and require_multiple_indicators:
                        logger.info(f"股票 {stock_code} 在买点 {buy_date} 没有满足条件的形态组合")
                        continue
                    
                    # 获取买点后的实际涨跌幅
                    future_change, price_series = self._calculate_detailed_future_change(
                        stock_code=stock_code,
                        buy_date=buy_date,
                        forward_days=forward_days
                    )
                    
                    # 计算最大收益和最大回撤
                    max_profit, max_drawdown, hold_days = self._calculate_price_metrics(
                        price_series, 
                        profit_taking=self.config["profit_taking"],
                        stop_loss=self.config["stop_loss"]
                    )
                    
                    # 记录每个形态组合的表现
                    for combo in pattern_combinations:
                        combo_id = combo["combo_id"]
                        combo_score = self.analyzer.calculate_combination_score(combo)
                        
                        # 更新组合统计
                        if combo_id not in results["pattern_combinations"]:
                            results["pattern_combinations"][combo_id] = {
                                "patterns": combo["patterns"],
                                "score": combo_score,
                                "instances": [],
                                "success_count": 0,
                                "total_count": 0,
                                "total_gain": 0.0,
                                "avg_gain": 0.0,
                                "max_gain": 0.0
                            }
                        
                        # 添加实例
                        instance = {
                            "stock_code": stock_code,
                            "stock_name": stock_name,
                            "buy_date": buy_date,
                            "future_change": future_change,
                            "max_profit": max_profit,
                            "max_drawdown": max_drawdown,
                            "hold_days": hold_days
                        }
                        
                        results["pattern_combinations"][combo_id]["instances"].append(instance)
                        results["pattern_combinations"][combo_id]["total_count"] += 1
                        
                        # 如果上涨超过2%，视为成功
                        if future_change >= 2.0:
                            results["pattern_combinations"][combo_id]["success_count"] += 1
                            results["pattern_combinations"][combo_id]["total_gain"] += future_change
                            
                            # 更新最大收益
                            if future_change > results["pattern_combinations"][combo_id]["max_gain"]:
                                results["pattern_combinations"][combo_id]["max_gain"] = future_change
                        
                        # 更新全局统计
                        total_combinations += 1
                        
                        if future_change >= 2.0:
                            successful_combinations += 1
                            total_gain += future_change
                            
                            if future_change > max_gain:
                                max_gain = future_change
                        else:
                            total_loss += abs(future_change) if future_change < 0 else 0
                            
                            if future_change < max_loss:
                                max_loss = future_change
                    
                    # 记录买点结果
                    stock_results.append({
                        "buy_date": buy_date,
                        "future_change": future_change,
                        "max_profit": max_profit,
                        "max_drawdown": max_drawdown,
                        "hold_days": hold_days,
                        "pattern_combinations": [
                            {
                                "combo_id": combo["combo_id"],
                                "patterns": combo["patterns"],
                                "score": self.analyzer.calculate_combination_score(combo)
                            }
                            for combo in pattern_combinations
                        ]
                    })
                
                # 记录股票结果
                results["stock_results"][stock_code] = {
                    "stock_name": self.db.get_stock_name(stock_code),
                    "buy_points": stock_results
                }
                
            except Exception as e:
                logger.error(f"回测股票 {stock_code} 时出错: {e}")
        
        # 计算每个组合的平均收益
        for combo_id, combo_data in results["pattern_combinations"].items():
            if combo_data["success_count"] > 0:
                combo_data["avg_gain"] = combo_data["total_gain"] / combo_data["success_count"]
        
        # 计算总体性能指标
        if total_combinations > 0:
            results["performance"]["success_rate"] = successful_combinations / total_combinations * 100
        
        if successful_combinations > 0:
            results["performance"]["avg_gain"] = total_gain / successful_combinations
        
        results["performance"]["max_gain"] = max_gain
        
        if total_loss != 0:
            results["performance"]["avg_loss"] = total_loss / (total_combinations - successful_combinations) if (total_combinations - successful_combinations) > 0 else 0
            results["performance"]["profit_factor"] = total_gain / total_loss if total_loss > 0 else float('inf')
        
        results["performance"]["max_loss"] = max_loss
        
        return results

    def generate_strategy_from_backtest(self, backtest_results: Dict[str, Any], 
                                       min_success_rate: float = 60.0,
                                       min_profit_factor: float = 2.0,
                                       max_combinations: int = 5) -> Dict[str, Any]:
        """
        从回测结果生成选股策略
        
        Args:
            backtest_results: 回测结果
            min_success_rate: 最小成功率（百分比）
            min_profit_factor: 最小盈亏比
            max_combinations: 最多使用的组合数量
            
        Returns:
            Dict[str, Any]: 选股策略配置
        """
        logger.info("开始从回测结果生成选股策略")
        
        # 获取所有形态组合
        pattern_combinations = backtest_results.get("pattern_combinations", {})
        
        if not pattern_combinations:
            logger.warning("回测结果中没有形态组合数据")
            return {"strategy": {"conditions": []}}
        
        # 过滤和排序形态组合
        filtered_combinations = []
        
        for combo_id, combo_data in pattern_combinations.items():
            # 计算成功率
            success_rate = combo_data["success_count"] / combo_data["total_count"] * 100 if combo_data["total_count"] > 0 else 0
            
            # 计算盈亏比
            avg_gain = combo_data["avg_gain"]
            avg_loss = backtest_results["performance"]["avg_loss"]
            profit_factor = avg_gain / abs(avg_loss) if avg_loss != 0 else float('inf')
            
            # 过滤不满足条件的组合
            if success_rate >= min_success_rate and profit_factor >= min_profit_factor and combo_data["total_count"] >= 3:
                filtered_combinations.append({
                    "combo_id": combo_id,
                    "patterns": combo_data["patterns"],
                    "score": combo_data["score"],
                    "success_rate": success_rate,
                    "profit_factor": profit_factor,
                    "avg_gain": avg_gain,
                    "total_count": combo_data["total_count"]
                })
        
        # 按成功率和盈亏比排序
        sorted_combinations = sorted(
            filtered_combinations,
            key=lambda x: (x["success_rate"], x["profit_factor"], x["avg_gain"]),
            reverse=True
        )
        
        # 选择前N个组合
        selected_combinations = sorted_combinations[:max_combinations]
        
        if not selected_combinations:
            logger.warning("没有满足条件的形态组合")
            return {"strategy": {"conditions": []}}
        
        # 生成策略条件
        strategy_conditions = []
        
        for combo in selected_combinations:
            patterns = combo["patterns"]
            
            # 将每个形态转换为条件
            pattern_conditions = []
            
            for pattern in patterns:
                pattern_id = pattern["pattern_id"]
                indicator_id = pattern["indicator_id"]
                period = pattern["period"]
                
                # 创建条件
                condition = {
                    "type": "indicator",
                    "indicator_id": indicator_id,
                    "period": period,
                    "parameter": "patterns",
                    "operator": "contains",
                    "value": pattern_id
                }
                
                pattern_conditions.append(condition)
            
            # 如果有多个条件，添加AND逻辑操作符
            if len(pattern_conditions) > 1:
                for i in range(len(pattern_conditions) - 1):
                    pattern_conditions.insert(2 * i + 1, {"type": "logic", "value": "AND"})
            
            # 将该组合条件添加到策略条件列表
            strategy_conditions.extend(pattern_conditions)
            
            # 如果不是最后一个组合，添加OR逻辑操作符
            if combo != selected_combinations[-1]:
                strategy_conditions.append({"type": "logic", "value": "OR"})
        
        # 生成完整策略
        strategy = {
            "strategy": {
                "id": f"pattern_combo_strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "name": "形态组合选股策略",
                "description": f"基于{len(selected_combinations)}个高成功率形态组合的选股策略",
                "conditions": strategy_conditions,
                "filters": {
                    "exchange": ["SSE", "SZSE"],
                    "exclude_st": True
                },
                "source": "backtest",
                "backtest_summary": {
                    "success_rate": backtest_results["performance"]["success_rate"],
                    "avg_gain": backtest_results["performance"]["avg_gain"],
                    "profit_factor": backtest_results["performance"]["profit_factor"]
                },
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        return strategy

    def validate_strategy(self, strategy: Dict[str, Any], stock_codes: List[str], 
                         date: str) -> Dict[str, Any]:
        """
        验证策略在给定日期对给定股票的表现
        
        Args:
            strategy: 策略配置
            stock_codes: 股票代码列表
            date: 验证日期
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info(f"开始验证策略，共 {len(stock_codes)} 只股票")
        
        # 从策略配置中提取条件
        conditions = strategy.get("strategy", {}).get("conditions", [])
        
        if not conditions:
            logger.warning("策略中没有条件")
            return {"matched_stocks": [], "success_rate": 0.0}
        
        # 验证结果
        results = {
            "matched_stocks": [],
            "total_stocks": len(stock_codes),
            "match_count": 0,
            "conditions": conditions,
            "validation_date": date
        }
        
        # 遍历股票
        for stock_code in stock_codes:
            try:
                # 获取股票数据
                start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
                
                # 获取不同周期的数据
                period_data = {}
                for period in self.periods:
                    try:
                        period_data[period] = self.db.get_kline_data(
                            stock_code=stock_code,
                            start_date=start_date,
                            end_date=date,
                            period=period
                        )
                    except Exception as e:
                        logger.error(f"获取 {stock_code} {period} 周期数据时出错: {e}")
                
                # 对齐数据到指定日期
                aligned_data = self._align_data_to_date(period_data, date)
                
                if not aligned_data:
                    logger.warning(f"股票 {stock_code} 在日期 {date} 没有对齐的数据，跳过")
                    continue
                
                # 获取股票名称
                stock_name = self.db.get_stock_name(stock_code)
                
                # 使用形态识别分析器分析
                analysis_result = self.analyzer.analyze(
                    data=aligned_data,
                    stock_code=stock_code,
                    stock_name=stock_name
                )
                
                # 检查是否满足策略条件
                from strategy.strategy_condition_evaluator import StrategyConditionEvaluator
                evaluator = StrategyConditionEvaluator()
                
                # 获取主时间周期的数据用于条件评估
                main_period = "DAILY"  # 默认使用日线
                if main_period in aligned_data:
                    eval_data = aligned_data[main_period]
                else:
                    # 使用可用的第一个周期
                    eval_data = aligned_data[list(aligned_data.keys())[0]]
                
                # 评估条件
                evaluation_result = evaluator.evaluate_conditions(conditions, eval_data, stock_code)
                
                is_match = False
                if isinstance(evaluation_result, dict):
                    is_match = evaluation_result.get("result", False)
                else:
                    is_match = evaluation_result
                
                if is_match:
                    # 添加到匹配列表
                    results["matched_stocks"].append({
                        "stock_code": stock_code,
                        "stock_name": stock_name,
                        "match_details": evaluation_result.get("details", {}) if isinstance(evaluation_result, dict) else {}
                    })
                    
                    results["match_count"] += 1
                
            except Exception as e:
                logger.error(f"验证股票 {stock_code} 时出错: {e}")
        
        # 计算匹配率
        results["match_rate"] = results["match_count"] / results["total_stocks"] * 100 if results["total_stocks"] > 0 else 0
        
        return results

    def identify_cross_period_patterns(self, stock_code: str, date: str) -> Dict[str, Any]:
        """
        识别跨周期形态组合
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            Dict[str, Any]: 跨周期形态结果
        """
        logger.info(f"开始识别股票 {stock_code} 在 {date} 的跨周期形态")
        
        # 获取股票数据
        start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
        
        # 获取不同周期的数据
        period_data = {}
        for period in self.periods:
            try:
                period_data[period] = self.db.get_kline_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=date,
                    period=period
                )
            except Exception as e:
                logger.error(f"获取 {stock_code} {period} 周期数据时出错: {e}")
        
        # 对齐数据到指定日期
        aligned_data = self._align_data_to_date(period_data, date)
        
        if not aligned_data:
            logger.warning(f"股票 {stock_code} 在日期 {date} 没有对齐的数据")
            return {"cross_period_patterns": []}
        
        # 获取股票名称
        stock_name = self.db.get_stock_name(stock_code)
        
        # 使用形态识别分析器分析
        analysis_result = self.analyzer.analyze(
            data=aligned_data,
            stock_code=stock_code,
            stock_name=stock_name
        )
        
        # 查找跨周期形态
        cross_period_patterns = self.analyzer.find_cross_period_patterns()
        
        # 添加股票信息
        result = {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "date": date,
            "cross_period_patterns": cross_period_patterns
        }
        
        return result

# 示例用法
if __name__ == "__main__":
    # 初始化高级回测器
    backtester = AdvancedBacktester(
        indicators=["MACD", "KDJ", "RSI"],
        periods=["DAILY", "WEEKLY"]
    )
    
    # 设置回测配置
    backtester.set_config({
        "min_pattern_strength": 70,
        "profit_taking": 0.08,
        "stop_loss": -0.05
    })
    
    # 定义形态组合
    pattern_combinations = [
        {
            "indicator_id": "MACD",
            "pattern_id": "golden_cross",
            "period": "DAILY"
        },
        {
            "indicator_id": "RSI",
            "pattern_id": "oversold",
            "period": "DAILY"
        }
    ]
    
    # 运行回测
    results = backtester.backtest_with_combination(
        stock_codes=["000001.SZ", "600000.SH"],
        start_date="2023-01-01",
        end_date="2023-06-30",
        pattern_combinations=pattern_combinations,
        forward_days=10,
        threshold=0.03
    )
    
    # 输出结果
    print(json.dumps(results["performance"], indent=2)) 