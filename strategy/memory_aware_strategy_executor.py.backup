"""
内存感知的策略执行器

集成大规模内存优化到策略执行流程中，专门优化4000+股票选股场景。
提供智能内存管理、动态批次调整和性能监控。

Author: System  
Date: 2025-01-15
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from strategy.strategy_executor import StrategyExecutor
from strategy.large_scale_memory_optimizer import LargeScaleMemoryOptimizer, LargeScaleMemoryConfig
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)


class MemoryAwareStrategyExecutor(StrategyExecutor):
    """
    内存感知的策略执行器
    
    专门为大规模股票选股场景优化：
    1. 集成内存优化器 - 智能管理内存使用
    2. 流式处理支持 - 避免内存溢出
    3. 动态批次调整 - 根据内存压力调整处理规模
    4. 性能监控集成 - 实时监控内存和性能指标
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 enable_memory_optimization: bool = True,
                 memory_config: Optional[LargeScaleMemoryConfig] = None):
        super().__init__(max_workers)
        
        # 内存优化配置
        self.enable_memory_optimization = enable_memory_optimization
        if self.enable_memory_optimization:
            self.memory_optimizer = LargeScaleMemoryOptimizer(
                memory_config or LargeScaleMemoryConfig()
            )
        else:
            self.memory_optimizer = None
        
        # 性能统计
        self.memory_stats = {
            'total_memory_optimized_executions': 0,
            'memory_peak_usage': 0.0,
            'total_memory_saved_mb': 0.0,
            'gc_triggered_count': 0,
            'batch_adjustments_count': 0
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def execute_strategy_memory_optimized(
        self,
        stock_codes: List[str],
        strategy_plan: Dict[str, Any],
        enable_early_stop: bool = False,
        enable_streaming: bool = True
    ) -> pd.DataFrame:
        """
        执行内存优化的策略选股
        
        Args:
            stock_codes: 股票代码列表  
            strategy_plan: 策略计划
            enable_early_stop: 是否启用早停
            enable_streaming: 是否启用流式处理
            
        Returns:
            pd.DataFrame: 选股结果
        """
        total_stocks = len(stock_codes)
        logger.info(f"🚀 开始内存优化策略执行: {total_stocks}只股票")
        
        if not self.enable_memory_optimization or total_stocks < 1000:
            # 小规模选股，使用常规方法
            logger.info("使用常规策略执行方法")
            return super().execute_strategy(stock_codes, strategy_plan, enable_early_stop)
        
        # 大规模选股，使用内存优化方法
        return self._execute_large_scale_strategy(
            stock_codes, strategy_plan, enable_early_stop, enable_streaming
        )
    
    def _execute_large_scale_strategy(
        self,
        stock_codes: List[str], 
        strategy_plan: Dict[str, Any],
        enable_early_stop: bool,
        enable_streaming: bool
    ) -> pd.DataFrame:
        """
        执行大规模内存优化策略
        
        Args:
            stock_codes: 股票代码列表
            strategy_plan: 策略计划  
            enable_early_stop: 是否启用早停
            enable_streaming: 是否启用流式处理
            
        Returns:
            pd.DataFrame: 选股结果
        """
        start_time = time.time()
        results = []
        successful_count = 0
        
        # 创建处理函数
        def process_single_stock(stock_code: str) -> Optional[Dict[str, Any]]:
            """处理单只股票"""
            try:
                # 获取股票数据
                stock_data = self._get_stock_data_optimized(stock_code, strategy_plan)
                if not stock_data:
                    return None
                
                # 执行策略条件评估
                result = self._evaluate_strategy_conditions_optimized(
                    stock_code, stock_data, strategy_plan
                )
                
                if result and result.get('match', False):
                    return {
                        'stock_code': stock_code,
                        'score': result.get('score', 0),
                        'matched_conditions': result.get('matched_conditions', []),
                        'details': result.get('details', {}),
                        'timestamp': time.time()
                    }
                
                return None
                
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 失败: {e}")
                return None
        
        # 使用内存优化器进行流式处理
        logger.info("🔧 启动内存优化流式处理...")
        
        for result in self.memory_optimizer.process_large_stock_selection(
            stock_codes, 
            process_single_stock
        ):
            if result:
                results.append(result)
                successful_count += 1
                
                # 早停检查
                if enable_early_stop and successful_count >= 100:
                    logger.info(f"早停触发: 已找到 {successful_count} 只匹配股票")
                    break
        
        # 处理结果
        total_time = time.time() - start_time
        
        # 更新内存统计
        self._update_memory_stats()
        
        # 创建结果DataFrame
        result_df = self._process_results_memory_optimized(results, strategy_plan)
        
        # 记录执行统计
        self._log_execution_stats(
            len(stock_codes), successful_count, len(result_df), total_time
        )
        
        return result_df
    
    def _get_stock_data_optimized(
        self, 
        stock_code: str, 
        strategy_plan: Dict[str, Any]
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        获取优化的股票数据
        
        Args:
            stock_code: 股票代码
            strategy_plan: 策略计划
            
        Returns:
            Optional[Dict[str, pd.DataFrame]]: 股票数据
        """
        try:
            # 使用基类方法，但添加内存优化
            stock_data = super()._get_stock_data(stock_code, strategy_plan)
            
            if stock_data and self.enable_memory_optimization:
                # 优化DataFrame内存使用
                for period, df in stock_data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        stock_data[period] = self._optimize_dataframe_memory(df)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"获取股票数据失败 {stock_code}: {e}")
            return None
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化DataFrame内存使用
        
        Args:
            df: 原始DataFrame
            
        Returns:
            pd.DataFrame: 优化后的DataFrame
        """
        if df.empty:
            return df
        
        optimized_df = df.copy()
        
        # 优化数值列的数据类型
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            col_data = optimized_df[col]
            
            # 检查是否可以使用更小的数据类型
            if col_data.dtype == 'float64':
                if col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                    optimized_df[col] = col_data.astype(np.float32)
            
            elif col_data.dtype == 'int64':
                if col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                    optimized_df[col] = col_data.astype(np.int32)
        
        # 优化字符串列
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].dtype == 'object':
                try:
                    optimized_df[col] = optimized_df[col].astype('category')
                except:
                    pass  # 如果转换失败，保持原始类型
        
        return optimized_df
    
    def _evaluate_strategy_conditions_optimized(
        self,
        stock_code: str,
        stock_data: Dict[str, pd.DataFrame], 
        strategy_plan: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        优化的策略条件评估
        
        Args:
            stock_code: 股票代码
            stock_data: 股票数据
            strategy_plan: 策略计划
            
        Returns:
            Optional[Dict[str, Any]]: 评估结果
        """
        try:
            # 使用基类方法进行条件评估
            result = super()._evaluate_strategy_conditions(
                stock_code, stock_data, strategy_plan
            )
            
            # 清理临时数据以节省内存
            if hasattr(self, '_temp_calculations'):
                delattr(self, '_temp_calculations')
            
            return result
            
        except Exception as e:
            logger.error(f"策略条件评估失败 {stock_code}: {e}")
            return None
    
    def _process_results_memory_optimized(
        self,
        results: List[Dict[str, Any]],
        strategy_plan: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        内存优化的结果处理
        
        Args:
            results: 结果列表
            strategy_plan: 策略计划
            
        Returns:
            pd.DataFrame: 处理后的结果
        """
        if not results:
            return pd.DataFrame()
        
        # 使用基类方法处理结果
        result_df = super()._process_results(results, strategy_plan)
        
        # 优化结果DataFrame的内存使用
        if not result_df.empty and self.enable_memory_optimization:
            result_df = self._optimize_dataframe_memory(result_df)
        
        return result_df
    
    def _update_memory_stats(self):
        """更新内存统计"""
        if not self.memory_optimizer:
            return
        
        optimization_report = self.memory_optimizer.get_memory_optimization_report()
        
        self.memory_stats['total_memory_optimized_executions'] += 1
        self.memory_stats['memory_peak_usage'] = max(
            self.memory_stats['memory_peak_usage'],
            optimization_report['performance_stats']['memory_peak']
        )
        self.memory_stats['total_memory_saved_mb'] += optimization_report['performance_stats'].get('memory_saved_mb', 0)
        self.memory_stats['gc_triggered_count'] += optimization_report['performance_stats']['gc_triggered']
        self.memory_stats['batch_adjustments_count'] += optimization_report['performance_stats']['batch_adjustments']
    
    def _log_execution_stats(
        self, 
        total_stocks: int, 
        successful_count: int, 
        result_count: int, 
        total_time: float
    ):
        """
        记录执行统计
        
        Args:
            total_stocks: 总股票数
            successful_count: 成功处理数
            result_count: 结果数量  
            total_time: 总耗时
        """
        logger.info("📊 内存优化策略执行完成统计:")
        logger.info(f"   总股票数: {total_stocks}")
        logger.info(f"   成功处理: {successful_count}")
        logger.info(f"   匹配结果: {result_count}")
        logger.info(f"   处理耗时: {total_time:.2f}秒")
        logger.info(f"   处理速度: {total_stocks/total_time:.1f}股/秒")
        
        if self.memory_optimizer:
            memory_report = self.memory_optimizer.get_memory_optimization_report()
            logger.info(f"   内存峰值: {memory_report['performance_stats']['memory_peak']:.1f}%")
            logger.info(f"   GC触发: {memory_report['performance_stats']['gc_triggered']}次")
            logger.info(f"   批次调整: {memory_report['performance_stats']['batch_adjustments']}次")
    
    def get_memory_optimization_summary(self) -> Dict[str, Any]:
        """
        获取内存优化总结
        
        Returns:
            Dict[str, Any]: 内存优化总结
        """
        summary = {
            'memory_stats': self.memory_stats.copy(),
            'optimization_enabled': self.enable_memory_optimization
        }
        
        if self.memory_optimizer:
            summary['optimizer_report'] = self.memory_optimizer.get_memory_optimization_report()
        
        return summary


# 便捷函数
@exception_handler(reraise=True)
def execute_memory_optimized_selection(
    stock_codes: List[str],
    strategy_config_path: str,
    enable_early_stop: bool = False,
    max_memory_usage: float = 75.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    执行内存优化的股票选股
    
    Args:
        stock_codes: 股票代码列表
        strategy_config_path: 策略配置文件路径
        enable_early_stop: 是否启用早停
        max_memory_usage: 最大内存使用率
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: 选股结果和优化报告
    """
    # 加载策略配置
    with open(strategy_config_path, 'r', encoding='utf-8') as f:
        strategy_plan = json.load(f)
    
    # 创建内存配置
    memory_config = LargeScaleMemoryConfig(
        max_memory_usage_percent=max_memory_usage,
        auto_batch_adjustment=True,
        enable_streaming=True
    )
    
    # 创建执行器
    executor = MemoryAwareStrategyExecutor(
        enable_memory_optimization=True,
        memory_config=memory_config
    )
    
    # 执行选股
    results = executor.execute_strategy_memory_optimized(
        stock_codes, 
        strategy_plan,
        enable_early_stop=enable_early_stop
    )
    
    # 获取优化报告
    optimization_summary = executor.get_memory_optimization_summary()
    
    return results, optimization_summary


if __name__ == "__main__":
    # 测试示例
    import os
    
    # 创建测试股票列表
    test_stocks = [f"{i:06d}.SH" for i in range(1, 4001)]  # 4000只股票
    
    # 测试配置文件路径
    config_path = "config/strategies/test_simple_strategy.yaml"
    
    if os.path.exists(config_path):
        try:
            results, summary = execute_memory_optimized_selection(
                test_stocks,
                config_path, 
                enable_early_stop=True,
                max_memory_usage=75.0
            )
            
            print(f"内存优化选股完成!")
            print(f"处理股票数: {len(test_stocks)}")
            print(f"选股结果数: {len(results)}")
            print(f"内存峰值: {summary.get('optimizer_report', {}).get('performance_stats', {}).get('memory_peak', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"内存优化选股测试失败: {e}")
    else:
        logger.warning(f"测试配置文件不存在: {config_path}") 