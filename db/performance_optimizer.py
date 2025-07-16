"""
性能优化主控制器

整合批量数据优化器、并行处理器和内存优化器，
实现完整的性能优化方案，目标是将4000只股票选股时间从30分钟降到5分钟。

Author: System
Date: 2025-01-15
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from db.batch_data_optimizer import Batch_data_optimizer, Batch_config
from db.parallel_processor import Parallel_processor, Processing_config, Processing_mode
from db.memory_optimizer import Memory_optimizer, Memory_config
from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.cache_interface import ICacheService
from utils.logger import getLogger

logger = getLogger(__name__)


@dataclass
class OptimizationConfig:
    """性能优化配置"""
    # 批量处理配置
    batch_size: int = 150
    max_workers: int = 24
    
    # 内存管理配置
    max_memory_usage_percent: float = 75.0
    chunk_size_mb: int = 128
    
    # 处理模式配置
    processing_mode: processing_mode = Processing_mode.THREAD
    enable_cache: bool = True
    
    # 性能目标
    target_time_minutes: int = 5
    target_stocks_per_second: float = 13.3  # 4000股票/5分钟


@dataclass
class PerformanceReport:
    """性能报告"""
    total_stocks: int
    processing_time_seconds: float
    stocks_per_second: float
    cache_hit_rate: float
    memory_usage_mb: float
    batch_count: int
    parallel_efficiency: float
    target_achieved: bool
    
    def to_dict_Optimizer_Performance_Optimizer(self) -> Dict[str, Any]:
        return {
            'total_stocks': self.total_stocks,
            'processing_time_seconds': self.processing_time_seconds,
            'processing_time_minutes': self.processing_time_seconds / 60,
            'stocks_per_second': self.stocks_per_second,
            'cache_hit_rate': self.cache_hit_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'batch_count': self.batch_count,
            'parallel_efficiency': self.parallel_efficiency,
            'target_achieved': self.target_achieved,
            'performance_improvement': f"{self.stocks_per_second / 2.22:.1f}x"  # 相比原来30分钟的提升
        }


class PerformanceOptimizer:
    """
    性能优化主控制器
    
    整合所有优化组件，提供统一的高性能股票分析接口：
    - 批量数据获取优化
    - 并行指标计算
    - 智能内存管理
    - 性能监控和调优
    """
    
    def __init___32(self, data_access: DataAccessInterface, cache_service: ICacheService,
                 config: Optional[Optimization_config] = None):
        self.data_access = data_access
        self.cache_service = cache_service
        self.config = config or Optimization_config()
        
        # 初始化优化组件
        self.batch_optimizer = Batch_data_optimizer(
            data_access=data_access,
            cache_service=cache_service,
            config=Batch_config(
                batch_size=self.config.batch_size,
                max_workers=self.config.max_workers,
                enable_cache=self.config.enable_cache
            )
        )
        
        self.parallel_processor = Parallel_processor(
            data_access=data_access,
            cache_service=cache_service,
            config=Processing_config(
                mode=self.config.processing_mode,
                max_workers=self.config.max_workers,
                chunk_size=self.config.batch_size // 2
            )
        )
        
        self.memory_optimizer = Memory_optimizer(
            config=Memory_config(
                max_memory_usage_percent=self.config.max_memory_usage_percent,
                chunk_size_mb=self.config.chunk_size_mb,
                auto_optimize=True
            )
        )
        
        self.performance_history: List[Performance_report] = []
    
    def optimize_stock_selection(self, stock_codes: List[str],
                                start_date: str, end_date: str,
                                indicators: List[str],
                                strategy_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        优化股票选股过程
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            indicators: 指标列表
            strategy_params: 策略参数
            
        Returns:
            Dict[str, Any]: 选股结果和性能报告
        """
        start_time = time.time()
        strategy_params = strategy_params or {}
        
        logger.info(f"开始优化股票选股: {len(stock_codes)}只股票, "
                   f"{len(indicators)}个指标, {start_date} 到 {end_date}")
        
        with self.memory_optimizer.memory_monitor("股票选股优化"):
            # 第一步：批量获取股票数据
            logger.info("第一步：批量获取股票数据")
            stock_data = self._get_optimized_stock_data(stock_codes, start_date, end_date)
            
            # 第二步：并行计算指标
            logger.info("第二步：并行计算指标")
            indicator_results = self._calculate_indicators_parallel(stock_data, indicators, strategy_params)
            
            # 第三步：执行选股策略
            logger.info("第三步：执行选股策略")
            selection_results = self._execute_selection_strategy(indicator_results, strategy_params)
            
            # 第四步：生成性能报告
            total_time = time.time() - start_time
            performance_report = self._generate_performance_report_Performance_Optimizer(
                len(stock_codes), total_time, stock_data, indicator_results
            )
            
            self.performance_history.append(performance_report)
            
            logger.info(f"股票选股优化完成: 耗时 {total_time:.2f}秒 "
                       f"({total_time/60:.1f}分钟), "
                       f"处理速度 {performance_report.stocks_per_second:.1f}股/秒")
            
            return {
                'selection_results': selection_results,
                'performance_report': performance_report.to_dict_Optimizer_Performance_Optimizer(),
                'stock_count': len(stock_codes),
                'indicator_count': len(indicators),
                'processing_time': total_time
            }
    
    def _get_optimized_stock_data(self, stock_codes: List[str],
                                 start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """获取优化的股票数据"""
        # 使用批量优化器获取数据
        stock_data = self.batch_optimizer.get_stocks_data_batch(
            stock_codes, start_date, end_date
        )
        
        # 内存优化
        optimized_data = self.memory_optimizer.optimize_stock_data_dict(stock_data)
        
        logger.info(f"股票数据获取完成: {len(optimized_data)}只股票")
        
        return optimized_data
    
    def _calculate_indicators_parallel(self, stock_data: Dict[str, pd.DataFrame],
                                     indicators: List[str],
                                     params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """并行计算指标"""
        # 使用并行处理器计算指标
        indicator_results = self.parallel_processor.process_indicators_parallel(
            stock_data, indicators, params
        )
        
        logger.info(f"指标计算完成: {len(indicator_results)}只股票")
        
        return indicator_results
    
    def _execute_selection_strategy(self, indicator_results: Dict[str, Dict[str, Any]],
                                   strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行选股策略"""
        selected_stocks = []
        selection_details = {}
        
        for stock_code, indicators in indicator_results.items():
            # 简化的选股逻辑（实际应该根据具体策略实现）
            score = self._calculate_stock_score(indicators, strategy_params)
            
            selection_details[stock_code] = {
                'score': score,
                'indicators': indicators,
                'selected': score > strategy_params.get('min_score', 0.6)
            }
            
            if selection_details[stock_code]['selected']:
                selected_stocks.append(stock_code)
        
        logger.info(f"选股策略执行完成: 选中 {len(selected_stocks)} 只股票")
        
        return {
            'selected_stocks': selected_stocks,
            'selection_details': selection_details,
            'total_analyzed': len(indicator_results),
            'selection_rate': len(selected_stocks) / len(indicator_results) if indicator_results else 0
        }
    
    def _calculate_stock_score(self, indicators: Dict[str, Any],
                              strategy_params: Dict[str, Any]) -> float:
        """计算股票评分"""
        # 简化的评分逻辑
        score = 0.5  # 基础分数
        
        # 根据指标调整分数（这里需要根据实际指标实现）
        for indicator_name, indicator_result in indicators.items():
            if hasattr(indicator_result, 'success') and indicator_result.success:
                # 简单的评分逻辑
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _generate_performance_report_Performance_Optimizer(self, total_stocks: int, processing_time: float,
                                   stock_data: Dict[str, pd.DataFrame],
                                   indicator_results: Dict[str, Dict[str, Any]]) -> Performance_report:
        """生成性能报告"""
        stocks_per_second = total_stocks / processing_time if processing_time > 0 else 0
        
        # 获取缓存命中率
        batch_performance = self.batch_optimizer.get_performance_summary()
        cache_hit_rate = 0.0
        if 'recent_performance' in batch_performance:
            cache_hit_rate = batch_performance['recent_performance'].get('avg_cache_hit_rate', 0.0)
        
        # 获取内存使用情况
        memory_stats = self.memory_optimizer.get_memory_stats()
        
        # 计算并行效率
        processing_stats = self.parallel_processor.get_processing_stats()
        parallel_efficiency = min(1.0, stocks_per_second / (self.config.max_workers * 2))
        
        # 检查是否达到目标
        target_achieved = stocks_per_second >= self.config.target_stocks_per_second
        
        return Performance_report(
            total_stocks=total_stocks,
            processing_time_seconds=processing_time,
            stocks_per_second=stocks_per_second,
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=memory_stats.process_memory_mb,
            batch_count=len(stock_data) // self.config.batch_size + (1 if len(stock_data) % self.config.batch_size else 0),
            parallel_efficiency=parallel_efficiency,
            target_achieved=target_achieved
        )
    
    def benchmark_performance(self, test_stock_counts: List[int] = None) -> Dict[str, Any]:
        """
        性能基准测试
        
        Args:
            test_stock_counts: 测试股票数量列表
            
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        test_stock_counts = test_stock_counts or [100, 500, 1000, 2000, 4000]
        benchmark_results = {}
        
        logger.info("开始性能基准测试")
        
        for stock_count in test_stock_counts:
            logger.info(f"测试 {stock_count} 只股票的性能")
            
            # 生成测试股票代码
            test_codes = [f"00000{i:04d}" for i in range(stock_count)]
            
            # 执行测试
            start_time = time.time()
            try:
                result = self.optimize_stock_selection(
                    stock_codes=test_codes,
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    indicators=["MA", "KDJ", "MACD"],
                    strategy_params={'min_score': 0.6}
                )
                
                processing_time = time.time() - start_time
                benchmark_results[f"{stock_count}_stocks"] = {
                    'processing_time_seconds': processing_time,
                    'processing_time_minutes': processing_time / 60,
                    'stocks_per_second': stock_count / processing_time,
                    'performance_report': result.get('performance_report', {}),
                    'success': True
                }
                
            except Exception as e:
                benchmark_results[f"{stock_count}_stocks"] = {
                    'error': str(e),
                    'success': False
                }
                logger.error(f"测试 {stock_count} 只股票失败: {e}")
        
        # 生成基准测试摘要
        successful_tests = {k: v for k, v in benchmark_results.items() if v.get('success', False)}
        
        if successful_tests:
            avg_stocks_per_second = sum(
                v['stocks_per_second'] for v in successful_tests.values()
            ) / len(successful_tests)
            
            # 预测4000只股票的处理时间
            predicted_4000_time = 4000 / avg_stocks_per_second
            
            benchmark_results['summary'] = {
                'successful_tests': len(successful_tests),
                'total_tests': len(test_stock_counts),
                'avg_stocks_per_second': avg_stocks_per_second,
                'predicted_4000_stocks_time_seconds': predicted_4000_time,
                'predicted_4000_stocks_time_minutes': predicted_4000_time / 60,
                'target_achieved': predicted_4000_time <= 300  # 5分钟
            }
        
        logger.info("性能基准测试完成")
        return benchmark_results
    
    def auto_tune_configuration(self, target_stock_count: int = 4000) -> Optimization_config:
        """
        自动调优配置
        
        Args:
            target_stock_count: 目标股票数量
            
        Returns:
            Optimization_config: 优化后的配置
        """
        logger.info(f"开始自动调优配置，目标股票数量: {target_stock_count}")
        
        # 获取系统资源信息
        memory_stats = self.memory_optimizer.get_memory_stats()
        available_memory_gb = memory_stats.available_memory_gb
        
        # 基于可用内存调整配置
        optimized_config = Optimization_config()
        
        # 调整批次大小
        if available_memory_gb > 8:
            optimized_config.batch_size = 200
            optimized_config.max_workers = 32
            optimized_config.chunk_size_mb = 256
        elif available_memory_gb > 4:
            optimized_config.batch_size = 150
            optimized_config.max_workers = 24
            optimized_config.chunk_size_mb = 128
        else:
            optimized_config.batch_size = 100
            optimized_config.max_workers = 16
            optimized_config.chunk_size_mb = 64
        
        # 调整内存使用限制
        optimized_config.max_memory_usage_percent = min(80.0, 60.0 + available_memory_gb * 2)
        
        # 根据目标股票数量调整处理模式
        if target_stock_count > 2000:
            optimized_config.processing_mode = Processing_mode.PROCESS
        else:
            optimized_config.processing_mode = Processing_mode.THREAD
        
        logger.info(f"配置自动调优完成: batch_size={optimized_config.batch_size}, "
                   f"max_workers={optimized_config.max_workers}, "
                   f"processing_mode={optimized_config.processing_mode}")
        
        return optimized_config
    
    def get_optimization_summary_Optimizer(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self.performance_history:
            return {"message": "暂无性能数据"}
        
        recent_reports = self.performance_history[-5:]  # 最近5次
        
        avg_stocks_per_second = sum(r.stocks_per_second for r in recent_reports) / len(recent_reports)
        avg_cache_hit_rate = sum(r.cache_hit_rate for r in recent_reports) / len(recent_reports)
        target_achievement_rate = sum(1 for r in recent_reports if r.target_achieved) / len(recent_reports)
        
        return {
            "performance_summary": {
                "recent_avg_stocks_per_second": avg_stocks_per_second,
                "recent_avg_cache_hit_rate": avg_cache_hit_rate,
                "target_achievement_rate": target_achievement_rate,
                "total_optimizations": len(self.performance_history)
            },
            "component_status": {
                "batch_optimizer": self.batch_optimizer.get_performance_summary(),
                "parallel_processor": self.parallel_processor.get_processing_stats(),
                "memory_optimizer": self.memory_optimizer.get_optimization_report()
            },
            "latest_performance": recent_reports[-1].to_dict_Optimizer_Performance_Optimizer() if recent_reports else None
        } 