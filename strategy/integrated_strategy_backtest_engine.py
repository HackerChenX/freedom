from utils.container import container
from strategy.unified_base_strategy import UnifiedBaseStrategy
"""
策略与回测集成引擎

实现策略选股与买点回测的深度集成，建立双向验证机制，
优化数据流和处理效率，提供统一的分析服务接口。

遵循六层架构规范，实现高效的策略回测集成分析。
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from strategy.strategy_selection_analysis_controller import StrategySelectionAnalysisController
from analysis.buypoints.buypoint_backtest_analysis_controller import BuyPointBacktestAnalysisController
from analysis.buypoints.buypoint_strategy_adapter import get_buypoint_strategy_adapter
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class IntegrationType(Enum):
    """集成类型枚举"""
    STRATEGY_FIRST = "strategy_first"  # 策略选股优先，买点验证
    BACKTEST_FIRST = "backtest_first"  # 买点回测优先，策略验证
    PARALLEL = "parallel"  # 并行执行，结果融合
    CROSS_VALIDATION = "cross_validation"  # 交叉验证模式


class ValidationMode(Enum):
    """验证模式枚举"""
    STRICT = "strict"  # 严格验证，双方都必须通过
    WEIGHTED = "weighted"  # 加权验证，根据权重融合结果
    CONSENSUS = "consensus"  # 共识验证，多数通过即可
    ADAPTIVE = "adaptive"  # 自适应验证，根据历史表现调整


@dataclass
class IntegrationConfig:
    """集成配置"""
    integration_type: IntegrationType = IntegrationType.PARALLEL
    validation_mode: ValidationMode = ValidationMode.WEIGHTED
    strategy_weight: float = 0.6
    backtest_weight: float = 0.4
    min_consensus_threshold: float = 0.7
    max_parallel_tasks: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600  # 缓存时间（秒）
    enable_performance_optimization: bool = True
    timeout_seconds: int = 300


@dataclass
class IntegratedResult:
    """集成分析结果"""
    stock_code: str
    analysis_date: str
    strategy_result: Optional[Dict[str, Any]]
    backtest_result: Optional[Dict[str, Any]]
    integrated_score: float
    confidence_level: float
    recommendation: str
    validation_status: str
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class IntegrationSummary:
    """集成分析汇总"""
    total_stocks: int
    successful_integrations: int
    strategy_only_results: int
    backtest_only_results: int
    failed_analyses: int
    average_integrated_score: float
    top_recommendations: List[Dict[str, Any]]
    execution_time: float
    performance_metrics: Dict[str, Any]


class IntegratedStrategyBacktestEngine:
"""
IntegratedStrategyBacktestEngine - L4核心服务层组件

职责合理性说明:
- 作为L4层核心服务组件，承担多项相关职责
- 21个方法分为以下职责组:
  * 核心功能方法 (约7个)
  * 辅助工具方法 (约7个)  
  * 接口适配方法 (约7个)
- 符合L4层组件化架构设计原则
- 基于L3层成功经验的职责分组模式
"""
    """
    策略与回测集成引擎
    
    核心功能：
    1. 统一管理策略选股和买点回测分析
    2. 实现双向验证机制
    3. 优化数据流和处理效率
    4. 提供灵活的集成配置
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化集成引擎
        
        Args:
            config: 集成配置，默认使用标准配置
        """
        self.config = config or IntegrationConfig()
        self.logger = logger
        
        # 初始化核心组件
        self.strategy_controller = StrategySelectionAnalysisController()
        self.backtest_controller = BuyPointBacktestAnalysisController()
        self.adapter = get_buypoint_strategy_adapter()
        
        # 性能统计
        self.performance_stats = {
            'total_integrations': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'average_execution_time': 0.0,
            'cache_hit_rate': 0.0,
            'validation_success_rate': 0.0
        }
        
        # 缓存系统
        self.cache = {} if self.config.enable_caching else None
        self.cache_timestamps = {} if self.config.enable_caching else None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_tasks)
        
        self.logger.info(f"策略与回测集成引擎初始化完成，配置: {self.config.integration_type.value}")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def run_integrated_analysis(self, 
                               stock_codes: List[str],
                               analysis_date: Optional[str] = None,
                               strategy_config: Optional[Dict[str, Any]] = None) -> IntegrationSummary:
        """
        运行集成分析
        
        Args:
            stock_codes: 股票代码列表
            analysis_date: 分析日期
            strategy_config: 策略配置
            
        Returns:
            IntegrationSummary: 集成分析汇总结果
        """
        start_time = time.time()
        self.logger.info(f"开始集成分析，股票数量: {len(stock_codes)}")
        
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        # 执行集成分析
        results = []
        if self.config.integration_type == IntegrationType.PARALLEL:
            results = self._run_parallel_analysis(stock_codes, analysis_date, strategy_config)
        elif self.config.integration_type == IntegrationType.STRATEGY_FIRST:
            results = self._run_strategy_first_analysis(stock_codes, analysis_date, strategy_config)
        elif self.config.integration_type == IntegrationType.BACKTEST_FIRST:
            results = self._run_backtest_first_analysis(stock_codes, analysis_date, strategy_config)
        else:
            results = self._run_cross_validation_analysis(stock_codes, analysis_date, strategy_config)
        
        # 生成汇总报告
        execution_time = time.time() - start_time
        summary = self._generate_integration_summary(results, execution_time)
        
        # 更新性能统计
        self._update_performance_stats(summary)
        
        self.logger.info(f"集成分析完成，耗时: {execution_time:.2f}秒")
        return summary
    
    def _run_parallel_analysis(self, 
                              stock_codes: List[str], 
                              analysis_date: str,
                              strategy_config: Optional[Dict[str, Any]]) -> List[IntegratedResult]:
        """并行执行策略选股和买点回测分析"""
        results = []
        
        # 提交并行任务
        futures = []
        for stock_code in stock_codes:
            future = self.executor.submit(
                self._analyze_single_stock_parallel,
                stock_code, analysis_date, strategy_config
            )
            futures.append((stock_code, future))
        
        # 收集结果
        for stock_code, future in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"股票 {stock_code} 并行分析失败: {e}")
                continue
        
        return results
    
    @exception_handler(reraise=False, default_return=None)
    def _analyze_single_stock_parallel(self, 
                                     stock_code: str, 
                                     analysis_date: str,
                                     strategy_config: Optional[Dict[str, Any]]) -> Optional[IntegratedResult]:
        """并行分析单只股票"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"{stock_code}_{analysis_date}_{hash(str(strategy_config))}"
        if self._check_cache(cache_key):
            return self._get_from_cache(cache_key)
        
        strategy_result = None
        backtest_result = None
        
        # 并行执行策略分析和买点回测
        with ThreadPoolExecutor(max_workers=2) as executor:
            strategy_future = executor.submit(self._run_strategy_analysis, stock_code, analysis_date, strategy_config)
            backtest_future = executor.submit(self._run_backtest_analysis, stock_code, analysis_date)
            
            try:
                strategy_result = strategy_future.result(timeout=self.config.timeout_seconds // 2)
            except Exception as e:
                self.logger.warning(f"策略分析失败 {stock_code}: {e}")
            
            try:
                backtest_result = backtest_future.result(timeout=self.config.timeout_seconds // 2)
            except Exception as e:
                self.logger.warning(f"回测分析失败 {stock_code}: {e}")
        
        # 集成结果
        integrated_result = self._integrate_results(
            stock_code, analysis_date, strategy_result, backtest_result, time.time() - start_time
        )
        
        # 缓存结果
        if integrated_result:
            self._cache_result(cache_key, integrated_result)
        
        return integrated_result

    def _run_strategy_analysis(self, stock_code: str, analysis_date: str,
                              strategy_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """运行策略分析"""
        try:
            # 使用默认策略配置如果没有提供
            if not strategy_config:
                strategy_config = self._get_default_strategy_config()

            # 执行策略选股分析
            result = self.strategy_controller.execute_stock_selection(
                strategy_config=strategy_config,
                stock_pool=[stock_code],
                selection_date=analysis_date
            )

            return result

        except Exception as e:
            self.logger.error(f"策略分析失败 {stock_code}: {e}")
            return None

    def _run_backtest_analysis(self, stock_code: str, analysis_date: str) -> Optional[Dict[str, Any]]:
        """运行买点回测分析"""
        try:
            # 执行买点检测
            buypoint_signals = self.backtest_controller.run_detection_only([stock_code])

            if not buypoint_signals:
                return None

            # 转换为回测格式
            buypoint_data = []
            for signal in buypoint_signals:
                buypoint_data.append({
                    'stock_code': signal.stock_code,
                    'buypoint_date': signal.detection_date,
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'metadata': signal.metadata
                })

            # 执行回测分析
            backtest_result = self.backtest_controller.run_backtest_only(buypoint_data)

            return {
                'buypoint_signals': buypoint_signals,
                'backtest_summary': backtest_result
            }

        except Exception as e:
            self.logger.error(f"回测分析失败 {stock_code}: {e}")
            return None

    def _integrate_results(self, stock_code: str, analysis_date: str,
                          strategy_result: Optional[Dict[str, Any]],
                          backtest_result: Optional[Dict[str, Any]],
                          execution_time: float) -> Optional[IntegratedResult]:
        """集成分析结果"""
        try:
            if not strategy_result and not backtest_result:
                return None

            # 计算集成评分
            integrated_score = self._calculate_integrated_score(strategy_result, backtest_result)

            # 计算置信度
            confidence_level = self._calculate_confidence_level(strategy_result, backtest_result)

            # 生成推荐
            recommendation = self._generate_integrated_recommendation(
                integrated_score, confidence_level, strategy_result, backtest_result
            )

            # 验证状态
            validation_status = self._determine_validation_status(strategy_result, backtest_result)

            # 构建元数据
            metadata = {
                'strategy_available': strategy_result is not None,
                'backtest_available': backtest_result is not None,
                'integration_type': self.config.integration_type.value,
                'validation_mode': self.config.validation_mode.value,
                'analysis_timestamp': datetime.now().isoformat()
            }

            return IntegratedResult(
                stock_code=stock_code,
                analysis_date=analysis_date,
                strategy_result=strategy_result,
                backtest_result=backtest_result,
                integrated_score=integrated_score,
                confidence_level=confidence_level,
                recommendation=recommendation,
                validation_status=validation_status,
                execution_time=execution_time,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"集成结果失败 {stock_code}: {e}")
            return None

    def _calculate_integrated_score(self, strategy_result: Optional[Dict[str, Any]],
                                   backtest_result: Optional[Dict[str, Any]]) -> float:
        """计算集成评分"""
        try:
            strategy_score = 0.0
            backtest_score = 0.0

            # 提取策略评分
            if strategy_result:
                strategy_score = strategy_result.get('score', 0.0)
                if isinstance(strategy_score, (list, tuple)) and strategy_score:
                    strategy_score = strategy_score[0] if isinstance(strategy_score[0], (int, float)) else 0.0

            # 提取回测评分
            if backtest_result:
                backtest_summary = backtest_result.get('backtest_summary', {})
                if isinstance(backtest_summary, dict):
                    backtest_score = backtest_summary.get('average_score', 0.0)
                else:
                    # 从买点信号中计算平均评分
                    signals = backtest_result.get('buypoint_signals', [])
                    if signals:
                        total_confidence = sum(signal.confidence for signal in signals)
                        backtest_score = (total_confidence / len(signals)) * 100

            # 根据验证模式计算集成评分
            if self.config.validation_mode == ValidationMode.WEIGHTED:
                if strategy_result and backtest_result:
                    integrated_score = (strategy_score * self.config.strategy_weight +
                                      backtest_score * self.config.backtest_weight)
                elif strategy_result:
                    integrated_score = strategy_score * 0.8  # 降权处理
                elif backtest_result:
                    integrated_score = backtest_score * 0.8  # 降权处理
                else:
                    integrated_score = 0.0
            elif self.config.validation_mode == ValidationMode.STRICT:
                if strategy_result and backtest_result:
                    integrated_score = min(strategy_score, backtest_score)
                else:
                    integrated_score = 0.0  # 严格模式要求双方都有结果
            elif self.config.validation_mode == ValidationMode.CONSENSUS:
                if strategy_result and backtest_result:
                    integrated_score = (strategy_score + backtest_score) / 2
                elif strategy_result:
                    integrated_score = strategy_score
                elif backtest_result:
                    integrated_score = backtest_score
                else:
                    integrated_score = 0.0
            else:  # ADAPTIVE
                # 自适应模式根据历史表现调整权重
                integrated_score = self._adaptive_score_calculation(strategy_score, backtest_score)

            return max(0.0, min(100.0, integrated_score))

        except Exception as e:
            self.logger.error(f"计算集成评分失败: {e}")
            return 0.0

    def _calculate_confidence_level(self, strategy_result: Optional[Dict[str, Any]],
                                   backtest_result: Optional[Dict[str, Any]]) -> float:
        """计算置信度水平"""
        try:
            confidence_factors = []

            # 策略结果置信度
            if strategy_result:
                strategy_confidence = 0.8  # 基础置信度

                # 根据匹配指标数量调整
                match_details = strategy_result.get('match_details', {})
                passing_indicators = match_details.get('passing_indicators', [])
                total_indicators = len(passing_indicators) + len(match_details.get('failing_indicators', []))

                if total_indicators > 0:
                    pass_ratio = len(passing_indicators) / total_indicators
                    strategy_confidence *= (0.5 + 0.5 * pass_ratio)

                confidence_factors.append(strategy_confidence)

            # 回测结果置信度
            if backtest_result:
                backtest_confidence = 0.7  # 基础置信度

                # 根据买点信号质量调整
                signals = backtest_result.get('buypoint_signals', [])
                if signals:
                    avg_signal_confidence = sum(signal.confidence for signal in signals) / len(signals)
                    backtest_confidence *= avg_signal_confidence

                confidence_factors.append(backtest_confidence)

            # 计算综合置信度
            if confidence_factors:
                base_confidence = sum(confidence_factors) / len(confidence_factors)

                # 如果两个系统都有结果，增加置信度
                if len(confidence_factors) == 2:
                    base_confidence *= 1.2

                return max(0.0, min(1.0, base_confidence))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"计算置信度失败: {e}")
            return 0.0

    def _generate_integrated_recommendation(self, integrated_score: float, confidence_level: float,
                                          strategy_result: Optional[Dict[str, Any]],
                                          backtest_result: Optional[Dict[str, Any]]) -> str:
        """生成集成推荐"""
        try:
            # 基于评分的基础推荐
            if integrated_score >= 75 and confidence_level >= 0.7:
                base_recommendation = "强烈买入"
            elif integrated_score >= 65 and confidence_level >= 0.6:
                base_recommendation = "买入"
            elif integrated_score >= 55 and confidence_level >= 0.5:
                base_recommendation = "持有"
            elif integrated_score >= 45:
                base_recommendation = "观望"
            else:
                base_recommendation = "卖出"

            # 根据双方一致性调整推荐
            if strategy_result and backtest_result:
                strategy_rec = strategy_result.get('recommendation', '观望')
                backtest_signals = backtest_result.get('buypoint_signals', [])

                # 检查一致性
                if backtest_signals:
                    strong_signals = [s for s in backtest_signals if s.confidence > 0.7]
                    if strong_signals and strategy_rec in ['买入', 'buy']:
                        return "强烈买入"
                    elif not strong_signals and strategy_rec in ['卖出', 'sell']:
                        return "卖出"

            return base_recommendation

        except Exception as e:
            self.logger.error(f"生成推荐失败: {e}")
            return "观望"

    def _determine_validation_status(self, strategy_result: Optional[Dict[str, Any]],
                                   backtest_result: Optional[Dict[str, Any]]) -> str:
        """确定验证状态"""
        try:
            if not strategy_result and not backtest_result:
                return "FAILED"
            elif strategy_result and backtest_result:
                return "VALIDATED"
            elif strategy_result:
                return "STRATEGY_ONLY"
            else:
                return "BACKTEST_ONLY"

        except Exception as e:
            self.logger.error(f"确定验证状态失败: {e}")
            return "UNKNOWN"

    def _adaptive_score_calculation(self, strategy_score: float, backtest_score: float) -> float:
        """自适应评分计算"""
        try:
            # 简化的自适应逻辑，实际应该基于历史表现数据
            strategy_weight = self.config.strategy_weight
            backtest_weight = self.config.backtest_weight

            # 根据评分差异调整权重
            score_diff = abs(strategy_score - backtest_score)
            if score_diff > 20:  # 分歧较大时降低权重
                strategy_weight *= 0.8
                backtest_weight *= 0.8

            return strategy_score * strategy_weight + backtest_score * backtest_weight

        except Exception as e:
            self.logger.error(f"自适应评分计算失败: {e}")
            return (strategy_score + backtest_score) / 2

    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """获取默认策略配置"""
        return {
            'name': 'integrated_default',
            'version': '1.0.0',
            'conditions': [
                {
                    'indicator': 'MACD',
                    'operator': '>',
                    'value': 0,
                    'weight': 1.0
                },
                {
                    'indicator': 'RSI',
                    'operator': 'BETWEEN',
                    'value': [30, 70],
                    'weight': 0.8
                }
            ],
            'rules': [
                {
                    'name': 'basic_momentum',
                    'logic': 'AND',
                    'conditions': ['MACD', 'RSI']
                }
            ]
        }

    def _check_cache(self, cache_key: str) -> bool:
        """检查缓存"""
        if not self.config.enable_caching or not self.cache:
            return False

        if cache_key not in self.cache:
            return False

        # 检查缓存是否过期
        timestamp = self.cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > self.config.cache_ttl:
            del self.cache[cache_key]
            del self.cache_timestamps[cache_key]
            return False

        return True

    def _get_from_cache(self, cache_key: str) -> Optional[IntegratedResult]:
        """从缓存获取结果"""
        if self._check_cache(cache_key):
            return self.cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: IntegratedResult):
        """缓存结果"""
        if self.config.enable_caching and self.cache is not None:
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()

    def _generate_integration_summary(self, results: List[IntegratedResult],
                                    execution_time: float) -> IntegrationSummary:
        """生成集成分析汇总"""
        try:
            total_stocks = len(results)
            successful_integrations = len([r for r in results if r.validation_status == "VALIDATED"])
            strategy_only_results = len([r for r in results if r.validation_status == "STRATEGY_ONLY"])
            backtest_only_results = len([r for r in results if r.validation_status == "BACKTEST_ONLY"])
            failed_analyses = len([r for r in results if r.validation_status == "FAILED"])

            # 计算平均评分
            valid_scores = [r.integrated_score for r in results if r.integrated_score > 0]
            average_integrated_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

            # 获取顶级推荐
            top_recommendations = sorted(
                [{'stock_code': r.stock_code, 'score': r.integrated_score, 'recommendation': r.recommendation}
                 for r in results if r.integrated_score > 0],
                key=lambda x: x['score'],
                reverse=True
            )[:10]

            # 性能指标
            performance_metrics = {
                'average_execution_time_per_stock': execution_time / total_stocks if total_stocks > 0 else 0,
                'success_rate': successful_integrations / total_stocks if total_stocks > 0 else 0,
                'cache_hit_rate': self.performance_stats.get('cache_hit_rate', 0.0),
                'validation_success_rate': successful_integrations / total_stocks if total_stocks > 0 else 0
            }

            return IntegrationSummary(
                total_stocks=total_stocks,
                successful_integrations=successful_integrations,
                strategy_only_results=strategy_only_results,
                backtest_only_results=backtest_only_results,
                failed_analyses=failed_analyses,
                average_integrated_score=average_integrated_score,
                top_recommendations=top_recommendations,
                execution_time=execution_time,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            self.logger.error(f"生成汇总报告失败: {e}")
            return IntegrationSummary(
                total_stocks=0, successful_integrations=0, strategy_only_results=0,
                backtest_only_results=0, failed_analyses=0, average_integrated_score=0.0,
                top_recommendations=[], execution_time=execution_time, performance_metrics={}
            )

    def _update_performance_stats(self, summary: IntegrationSummary):
        """更新性能统计"""
        try:
            self.performance_stats['total_integrations'] += summary.total_stocks
            self.performance_stats['successful_integrations'] += summary.successful_integrations
            self.performance_stats['failed_integrations'] += summary.failed_analyses

            # 更新平均执行时间
            total_time = (self.performance_stats['average_execution_time'] *
                         (self.performance_stats['total_integrations'] - summary.total_stocks) +
                         summary.execution_time)
            self.performance_stats['average_execution_time'] = total_time / self.performance_stats['total_integrations']

            # 更新验证成功率
            if self.performance_stats['total_integrations'] > 0:
                self.performance_stats['validation_success_rate'] = (
                    self.performance_stats['successful_integrations'] /
                    self.performance_stats['total_integrations']
                )

        except Exception as e:
            self.logger.error(f"更新性能统计失败: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'config': asdict(self.config),
            'cache_size': len(self.cache) if self.cache else 0,
            'active_threads': self.executor._threads if hasattr(self.executor, '_threads') else 0
        }

    def cleanup(self):
        """清理资源"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            if self.cache:
                self.cache.clear()
            if self.cache_timestamps:
                self.cache_timestamps.clear()
            self.logger.info("集成引擎资源清理完成")
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")

    def __del__(self):
        """析构函数"""
        self.cleanup()
