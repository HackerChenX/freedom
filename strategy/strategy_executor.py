"""
策略执行器模块

负责执行策略，对股票列表进行筛选和评分
"""

import concurrent.futures
import pandas as pd
import numpy as np
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime

from db.data_manager import DataManager
from strategy.strategy_manager import StrategyManager
from indicators.factory import IndicatorFactory
from utils.logger import get_logger
from utils.decorators import performance_monitor, safe_run, cache_result
from utils.exceptions import (
    StrategyExecutionError, 
    StrategyValidationError, 
    DataAccessError, 
    IndicatorExecutionError
)

logger = get_logger(__name__)


class StrategyExecutor:
    """
    策略执行器，负责执行策略，对股票列表进行筛选和评分
    """
    
    def __init__(self, max_workers: int = None, cache_enabled: bool = True):
        """
        初始化策略执行器
        
        Args:
            max_workers: 最大线程数，None表示使用默认值（CPU核心数 * 5）
            cache_enabled: 是否启用结果缓存
        """
        self.data_manager = DataManager()
        self.max_workers = max_workers or min(32, os.cpu_count() * 5)
        self.cache_enabled = cache_enabled
        self.cache = {}
        
        logger.info(f"策略执行器初始化完成，最大线程数: {self.max_workers}, 缓存{'启用' if cache_enabled else '禁用'}")
    
    @performance_monitor(threshold=1.0)
    def execute_strategy_by_id(
        self, 
        strategy_id: str, 
        strategy_manager: StrategyManager,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        通过策略ID执行选股策略
        
        Args:
            strategy_id: 策略ID
            strategy_manager: 策略管理器实例
            start_date: 开始日期，默认为None
            end_date: 结束日期，默认为当前日期
            progress_callback: 进度回调函数，参数为进度百分比和状态消息
            
        Returns:
            选股结果DataFrame
            
        Raises:
            StrategyExecutionError: 策略执行错误
        """
        try:
            # 检查缓存
            cache_key = f"strategy_result_{strategy_id}_{start_date}_{end_date}"
            if self.cache_enabled and cache_key in self.cache:
                logger.info(f"使用缓存的策略执行结果: {strategy_id}")
                return self.cache[cache_key]
            
            # 获取策略配置
            if progress_callback:
                progress_callback(0.1, f"正在加载策略: {strategy_id}")
                
            strategy_config = strategy_manager.get_strategy(strategy_id)
            if not strategy_config:
                raise StrategyValidationError(f"未找到策略: {strategy_id}")
                
            # 解析策略
            if progress_callback:
                progress_callback(0.2, "正在解析策略配置")
                
            from strategy.strategy_parser import StrategyParser
            parser = StrategyParser()
            strategy_plan = parser.parse_strategy(strategy_config)
            
            # 执行策略
            if progress_callback:
                progress_callback(0.3, "开始执行策略")
                
            result = self.execute_strategy(
                strategy_plan=strategy_plan,
                start_date=start_date,
                end_date=end_date,
                progress_callback=lambda p, m: progress_callback(0.3 + p * 0.7, m) if progress_callback else None
            )
            
            # 缓存结果
            if self.cache_enabled:
                self.cache[cache_key] = result
                
            return result
        except Exception as e:
            logger.error(f"执行策略 {strategy_id} 失败: {e}")
            raise StrategyExecutionError(f"执行策略失败: {str(e)}")
    
    @performance_monitor(threshold=1.0)
    def execute_strategy(
        self, 
        strategy_plan: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        执行选股策略
        
        Args:
            strategy_plan: 策略执行计划
            start_date: 开始日期，默认为None
            end_date: 结束日期，默认为当前日期
            progress_callback: 进度回调函数，参数为进度百分比和状态消息
            
        Returns:
            选股结果DataFrame
            
        Raises:
            StrategyExecutionError: 策略执行错误
        """
        try:
            # 1. 验证策略计划
            self._validate_strategy_plan(strategy_plan)
            
            # 2. 处理日期参数
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # 3. 获取股票列表
            if progress_callback:
                progress_callback(0.1, "正在获取股票列表")
                
            stock_list = self._get_filtered_stock_list(strategy_plan.get('filters', {}))
            
            if stock_list.empty:
                logger.warning("过滤后的股票列表为空")
                return pd.DataFrame()
                
            # 4. 获取条件配置
            conditions = strategy_plan.get('conditions', [])
            
            # 5. 并行处理每支股票
            if progress_callback:
                progress_callback(0.2, "正在处理股票数据")
                
            # 计算总股票数量，用于进度更新
            total_stocks = len(stock_list)
            processed_stocks = 0
            
            # 结果列表
            results = []
            
            # 使用线程池并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_stock = {
                    executor.submit(
                        self._process_stock, 
                        stock_code=row['stock_code'],
                        stock_name=row['stock_name'],
                        conditions=conditions,
                        end_date=end_date
                    ): row['stock_code'] for _, row in stock_list.iterrows()
                }
                
                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_to_stock):
                    stock_code = future_to_stock[future]
                    
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"处理股票 {stock_code} 时出错: {e}")
                    
                    # 更新进度
                    processed_stocks += 1
                    if progress_callback:
                        progress_callback(
                            0.2 + (processed_stocks / total_stocks) * 0.7,
                            f"已处理: {processed_stocks}/{total_stocks} 支股票"
                        )
            
            # 6. 创建结果DataFrame
            if not results:
                logger.warning("没有股票满足策略条件")
                return pd.DataFrame()
                
            if progress_callback:
                progress_callback(0.9, "正在处理选股结果")
                
            result_df = pd.DataFrame(results)
            
            # 7. 按策略中的排序规则排序
            if 'sort' in strategy_plan and strategy_plan['sort']:
                for sort_rule in strategy_plan['sort']:
                    field = sort_rule.get('field')
                    direction = sort_rule.get('direction', 'DESC')
                    
                    if field in result_df.columns:
                        ascending = (direction.upper() != 'DESC')
                        result_df = result_df.sort_values(by=field, ascending=ascending)
            
            # 8. 保存结果到数据库
            if progress_callback:
                progress_callback(0.95, "正在保存选股结果")
                
            try:
                self.data_manager.save_selection_result(
                    result=result_df,
                    strategy_id=strategy_plan.get('strategy_id'),
                    selection_date=end_date
                )
            except Exception as e:
                logger.error(f"保存选股结果失败: {e}")
            
            if progress_callback:
                progress_callback(1.0, "选股完成")
                
            return result_df
        except Exception as e:
            logger.error(f"执行策略失败: {e}")
            raise StrategyExecutionError(f"执行策略失败: {str(e)}")
    
    @safe_run(error_logger=logger)
    def _process_stock(
        self, 
        stock_code: str, 
        stock_name: str,
        conditions: List[Dict[str, Any]],
        end_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        处理单只股票
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            conditions: 条件列表
            end_date: 结束日期
            
        Returns:
            股票处理结果，如果不满足条件则返回None
        """
        try:
            logger.debug(f"处理股票: {stock_code} ({stock_name})")
            
            # 1. 创建条件执行栈
            execution_stack = []
            
            # 2. 满足的条件列表
            satisfied_conditions = []
            
            # 3. 信号强度列表
            signal_strengths = []
            
            # 4. 处理每个条件
            for condition in conditions:
                # 处理逻辑运算符
                if 'logic' in condition:
                    logic_op = condition['logic'].upper()
                    
                    # 至少需要两个操作数
                    if len(execution_stack) < 2:
                        logger.warning(f"逻辑运算符 {logic_op} 缺少足够的操作数")
                        continue
                    
                    # 弹出两个操作数
                    operand2 = execution_stack.pop()
                    operand1 = execution_stack.pop()
                    
                    # 执行逻辑运算
                    if logic_op == 'AND':
                        result = operand1 and operand2
                    elif logic_op == 'OR':
                        result = operand1 or operand2
                    else:
                        logger.warning(f"不支持的逻辑运算符: {logic_op}")
                        result = False
                    
                    # 结果入栈
                    execution_stack.append(result)
                else:
                    # 处理指标条件
                    indicator_id = condition.get('indicator_id')
                    period = condition.get('period')
                    parameters = condition.get('parameters', {})
                    
                    # 创建指标
                    indicator = IndicatorFactory.create(indicator_id, **parameters)
                    
                    # 获取K线数据
                    kline_data = self.data_manager.get_kline_data(
                        stock_code=stock_code,
                        period=period,
                        end_date=end_date
                    )
                    
                    if kline_data.empty:
                        # K线数据为空，条件不满足
                        execution_stack.append(False)
                        continue
                    
                    # 生成信号
                    signals = indicator.generate_signals(kline_data)
                    
                    if signals.empty:
                        # 信号为空，条件不满足
                        execution_stack.append(False)
                        continue
                    
                    # 获取最新的信号
                    latest_signal = signals.iloc[-1]
                    
                    # 检查是否有信号触发
                    signal_triggered = False
                    for column in latest_signal.index:
                        if column == 'signal_strength':
                            continue
                            
                        if latest_signal[column]:
                            signal_triggered = True
                            break
                    
                    # 获取信号强度
                    signal_strength = latest_signal.get('signal_strength', 0)
                    
                    # 更新结果
                    execution_stack.append(signal_triggered)
                    
                    if signal_triggered:
                        satisfied_conditions.append(indicator_id)
                        signal_strengths.append(signal_strength)
            
            # 5. 检查最终结果
            if not execution_stack:
                logger.warning(f"股票 {stock_code} 的条件执行栈为空")
                return None
                
            final_result = execution_stack[-1]
            
            if not final_result:
                return None
                
            # 6. 计算平均信号强度
            avg_signal_strength = sum(signal_strengths) / len(signal_strengths) if signal_strengths else 0
            
            # 7. 返回结果
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'satisfied_conditions': satisfied_conditions,
                'signal_strength': avg_signal_strength
            }
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 失败: {e}")
            return None
    
    def _get_filtered_stock_list(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        获取经过过滤的股票列表
        
        Args:
            filters: 过滤条件
            
        Returns:
            股票列表DataFrame
        """
        return self.data_manager.get_stock_list(filters=filters)
    
    def _validate_strategy_plan(self, strategy_plan: Dict[str, Any]) -> bool:
        """
        验证策略执行计划
        
        Args:
            strategy_plan: 策略执行计划
            
        Returns:
            验证通过返回True
            
        Raises:
            StrategyValidationError: 策略验证错误
        """
        # 检查必要字段
        required_fields = ['strategy_id', 'name', 'conditions']
        for field in required_fields:
            if field not in strategy_plan:
                raise StrategyValidationError(f"策略执行计划缺少必要字段: {field}")
        
        # 检查条件列表
        conditions = strategy_plan.get('conditions', [])
        if not conditions:
            raise StrategyValidationError("策略执行计划缺少条件")
        
        # 检查条件列表的有效性
        for condition in conditions:
            if 'logic' in condition:
                # 逻辑运算符
                if condition['logic'].upper() not in ['AND', 'OR']:
                    raise StrategyValidationError(f"不支持的逻辑运算符: {condition['logic']}")
            elif 'type' in condition and condition['type'] == 'logic':
                # 新格式的逻辑运算符
                if condition.get('value', '').upper() not in ['AND', 'OR']:
                    raise StrategyValidationError(f"不支持的逻辑运算符: {condition.get('value')}")
            elif 'type' in condition and condition['type'] == 'indicator':
                # 新格式的指标条件
                if 'indicator_id' not in condition:
                    raise StrategyValidationError(f"条件缺少必要字段: indicator_id")
                if 'period' not in condition:
                    raise StrategyValidationError(f"条件缺少必要字段: period")
            else:
                # 旧格式的指标条件
                required_condition_fields = ['indicator_id', 'period']
                for field in required_condition_fields:
                    if field not in condition:
                        raise StrategyValidationError(f"条件缺少必要字段: {field}")
        
        return True
    
    def clear_cache(self):
        """清除结果缓存"""
        old_size = len(self.cache)
        self.cache.clear()
        logger.info(f"已清除策略执行器缓存，共 {old_size} 项")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        return {
            'enabled': self.cache_enabled,
            'size': len(self.cache),
            'keys': list(self.cache.keys())
        } 