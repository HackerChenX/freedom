"""
策略执行引擎模块

负责根据配置执行选股逻辑
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from strategy.strategy_parser import StrategyParser
from indicators.factory import IndicatorFactory
from db.data_manager import DataManager
from enums.period_types import PeriodType
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategyExecutor:
    """
    策略执行引擎，负责根据配置执行选股逻辑
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        初始化策略执行引擎
        
        Args:
            data_manager: 数据管理器实例，如果为None则创建新实例
        """
        self.data_manager = data_manager or DataManager()
        self.parser = StrategyParser()
        self.indicator_factory = IndicatorFactory
        
    def execute(self, strategy_config: Dict[str, Any], 
                date: Optional[str] = None,
                stock_pool: Optional[List[str]] = None) -> pd.DataFrame:
        """
        执行策略
        
        Args:
            strategy_config: 策略配置
            date: 执行日期，如果为None则使用当前日期
            stock_pool: 股票池，如果为None则使用全市场
            
        Returns:
            pd.DataFrame: 选股结果
        """
        # 解析策略配置
        execution_plan = self.parser.parse_strategy(strategy_config)
        
        # 准备执行日期
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        # 准备股票池
        if stock_pool is None:
            # 获取全市场股票列表
            stock_pool_df = self.data_manager.get_stock_list(execution_plan.get("filters", {}))
            stock_pool = stock_pool_df["stock_code"].tolist()
            
        # 执行策略条件
        selected_stocks = self._execute_conditions(
            stock_pool=stock_pool,
            conditions=execution_plan["conditions"],
            date=date
        )
        
        # 处理排序
        sorted_stocks = self._sort_results(selected_stocks, execution_plan.get("sort", []))
        
        # 组装最终结果
        result = self._build_result(sorted_stocks, execution_plan, date)
        
        return result
    
    def _execute_conditions(self, stock_pool: List[str], conditions: List[Dict[str, Any]], 
                           date: str) -> pd.DataFrame:
        """
        执行策略条件
        
        Args:
            stock_pool: 股票池
            conditions: 条件列表
            date: 执行日期
            
        Returns:
            pd.DataFrame: 满足条件的股票
        """
        # 初始化结果DataFrame
        result = pd.DataFrame({"stock_code": stock_pool})
        result["selected"] = True
        
        # 用于记录中间结果和条件栈
        condition_results = {}
        condition_stack = []
        condition_group_stack = [result["selected"].copy()]
        
        # 遍历条件
        for i, condition in enumerate(conditions):
            condition_type = condition["type"]
            
            if condition_type == "logic":
                # 逻辑运算符
                logic_op = condition["value"]
                
                if logic_op == "NOT":
                    # 一元运算符，对栈顶结果取反
                    if len(condition_stack) < 1:
                        raise ValueError("无效的NOT运算：没有足够的操作数")
                        
                    operand = condition_stack.pop()
                    condition_stack.append(~operand)
                else:
                    # 二元运算符，对栈顶的两个结果进行运算
                    if len(condition_stack) < 2:
                        raise ValueError(f"无效的{logic_op}运算：没有足够的操作数")
                        
                    right = condition_stack.pop()
                    left = condition_stack.pop()
                    
                    if logic_op == "AND":
                        condition_stack.append(left & right)
                    elif logic_op == "OR":
                        condition_stack.append(left | right)
            elif condition_type == "group_start":
                # 条件分组开始
                condition_group_stack.append([])
            elif condition_type == "group_end":
                # 条件分组结束
                if len(condition_group_stack) <= 1:
                    raise ValueError("条件分组不匹配：多余的结束分组")
                    
                group_results = condition_group_stack.pop()
                
                # 根据分组的逻辑运算符合并结果
                if len(group_results) > 0:
                    group_result = group_results[0]
                    for j in range(1, len(group_results)):
                        group_result = group_result & group_results[j]
                        
                    condition_stack.append(group_result)
            elif condition_type == "indicator":
                # 指标条件
                indicator_result = self._execute_indicator_condition(
                    stock_pool=stock_pool,
                    condition=condition,
                    date=date
                )
                
                # 保存中间结果
                condition_key = f"condition_{i}"
                condition_results[condition_key] = indicator_result
                
                # 将结果压入栈
                condition_stack.append(indicator_result)
                
                # 如果在分组中，也添加到当前分组
                if len(condition_group_stack) > 0:
                    condition_group_stack[-1].append(indicator_result)
        
        # 最终结果应该只有一个元素在栈中
        if len(condition_stack) != 1:
            raise ValueError(f"条件处理错误：最终栈中有{len(condition_stack)}个元素，应该只有1个")
            
        final_result = condition_stack[0]
        
        # 应用最终结果
        result["selected"] = final_result
        
        # 只保留选中的股票
        result = result[result["selected"]].copy()
        
        # 添加中间结果列
        for key, value in condition_results.items():
            result[key] = value
            
        return result
    
    def _execute_indicator_condition(self, stock_pool: List[str], condition: Dict[str, Any], 
                                    date: str) -> pd.Series:
        """
        执行指标条件
        
        Args:
            stock_pool: 股票池
            condition: 指标条件
            date: 执行日期
            
        Returns:
            pd.Series: 条件结果，索引为股票代码，值为布尔值表示是否满足条件
        """
        indicator_id = condition["indicator_id"]
        period_str = condition["period"]
        signal_type = condition["signal_type"]
        parameters = condition.get("parameters", {})
        
        # 将周期字符串转换为枚举
        period = getattr(PeriodType, period_str)
        
        # 计算数据查询起止日期
        end_date = date
        
        # 根据周期和指标计算所需的历史数据长度
        lookback_days = self._calculate_lookback_days(period_str, indicator_id, parameters)
        start_date = self._get_start_date(end_date, lookback_days)
        
        # 初始化结果Series
        result = pd.Series(False, index=stock_pool)
        
        # 获取指标实例
        indicator = self.indicator_factory.create(indicator_id, **parameters)
        if indicator is None:
            logger.error(f"创建指标 {indicator_id} 失败")
            return result
        
        # 遍历股票池
        for stock_code in stock_pool:
            try:
                # 获取K线数据
                kline_data = self.data_manager.get_kline_data(
                    stock_code=stock_code,
                    period=period_str,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if kline_data.empty:
                    continue
                
                # 计算指标
                indicator_values = indicator.calculate(kline_data)
                
                # 生成信号
                signals = indicator.generate_signals(indicator_values)
                
                # 检查信号类型
                if signal_type in signals.columns:
                    # 获取最后一条记录的信号值
                    last_signal = signals[signal_type].iloc[-1]
                    result[stock_code] = last_signal
            except Exception as e:
                logger.error(f"执行指标 {indicator_id} 失败，股票代码：{stock_code}，错误：{e}")
                
        return result
    
    def _calculate_lookback_days(self, period_str: str, indicator_id: str, 
                                parameters: Dict[str, Any]) -> int:
        """
        计算所需的历史数据长度
        
        Args:
            period_str: 周期字符串
            indicator_id: 指标ID
            parameters: 指标参数
            
        Returns:
            int: 历史数据长度（天数）
        """
        # 基础查询天数
        base_days = {
            "MIN5": 3,
            "MIN15": 3,
            "MIN30": 5,
            "MIN60": 5,
            "DAILY": 60,
            "WEEKLY": 52,
            "MONTHLY": 24
        }
        
        # 特定指标的历史数据需求
        indicator_days = {
            "MACD": 60,  # MACD通常需要更长的历史数据
            "KDJ": 30,
            "BOLL": 30,
            "RSI": 30,
            "DMI": 45,
            "MULTI_PERIOD_RESONANCE": 120,
            "ELLIOTT_WAVE": 200,
            "DIVERGENCE": 90
        }
        
        # 周期转换系数
        period_multipliers = {
            "MIN5": 1/288,  # 每天约288个5分钟
            "MIN15": 1/96,  # 每天约96个15分钟
            "MIN30": 1/48,  # 每天约48个30分钟
            "MIN60": 1/24,  # 每天约24个60分钟
            "DAILY": 1,
            "WEEKLY": 7,
            "MONTHLY": 30
        }
        
        # 获取指标特定的天数要求
        days = indicator_days.get(indicator_id, base_days.get(period_str, 60))
        
        # 考虑参数中的周期
        for key, value in parameters.items():
            if "period" in key.lower() and isinstance(value, (int, float)):
                # 周期参数会影响所需的历史数据长度
                days = max(days, value * 3)  # 一般需要至少3倍周期的数据
                
        # 应用周期系数
        days = days * period_multipliers.get(period_str, 1)
        
        # 确保至少有5天的数据
        return max(int(days), 5)
    
    def _get_start_date(self, end_date: str, lookback_days: int) -> str:
        """
        根据结束日期和回溯天数计算开始日期
        
        Args:
            end_date: 结束日期
            lookback_days: 回溯天数
            
        Returns:
            str: 开始日期
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days)
        return start_dt.strftime("%Y-%m-%d")
    
    def _sort_results(self, selected_stocks: pd.DataFrame, 
                     sort_config: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        对选股结果进行排序
        
        Args:
            selected_stocks: 选中的股票
            sort_config: 排序配置
            
        Returns:
            pd.DataFrame: 排序后的结果
        """
        if selected_stocks.empty or not sort_config:
            return selected_stocks
            
        # 构建排序条件
        sort_cols = []
        ascending = []
        
        for sort_item in sort_config:
            field = sort_item["field"]
            direction = sort_item["direction"]
            
            # 检查排序字段是否存在
            if field in selected_stocks.columns:
                sort_cols.append(field)
                ascending.append(direction == "ASC")
                
        # 如果有排序字段，执行排序
        if sort_cols:
            return selected_stocks.sort_values(
                by=sort_cols,
                ascending=ascending
            ).reset_index(drop=True)
            
        return selected_stocks
    
    def _build_result(self, sorted_stocks: pd.DataFrame, execution_plan: Dict[str, Any], 
                     date: str) -> pd.DataFrame:
        """
        构建最终结果
        
        Args:
            sorted_stocks: 排序后的股票
            execution_plan: 执行计划
            date: 执行日期
            
        Returns:
            pd.DataFrame: 最终结果
        """
        if sorted_stocks.empty:
            # 返回空结果
            return pd.DataFrame({
                "stock_code": [],
                "stock_name": [],
                "strategy_id": [],
                "strategy_name": [],
                "selection_date": [],
                "rank": []
            })
            
        # 添加基本信息
        result = sorted_stocks.copy()
        
        # 添加股票名称
        stock_codes = result["stock_code"].tolist()
        stock_infos = self.data_manager.get_stock_basic_info(stock_codes)
        
        if not stock_infos.empty:
            result = pd.merge(
                result,
                stock_infos[["stock_code", "stock_name"]],
                on="stock_code",
                how="left"
            )
            
        # 添加策略信息
        result["strategy_id"] = execution_plan["strategy_id"]
        result["strategy_name"] = execution_plan["name"]
        result["selection_date"] = date
        
        # 添加排名
        result["rank"] = range(1, len(result) + 1)
        
        # 选择需要的列
        base_columns = [
            "stock_code", "stock_name", "strategy_id", "strategy_name",
            "selection_date", "rank"
        ]
        
        # 保留中间条件结果列
        condition_columns = [col for col in result.columns if col.startswith("condition_")]
        
        # 组合最终输出列
        output_columns = base_columns + condition_columns
        
        # 只保留存在的列
        output_columns = [col for col in output_columns if col in result.columns]
        
        return result[output_columns] 