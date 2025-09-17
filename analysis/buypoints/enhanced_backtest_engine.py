#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强买点回测引擎

提供强大的买点回测功能，包括：
- 多周期数据分析
- 指标计算和形态识别
- 回测结果评估
- 策略生成和验证
遵循六层架构规范
"""

import os
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from enums.signal_types import SignalType
from enums.pattern_types import Candle_pattern_type as PatternType
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

@dataclass
class BacktestConfig:
    """回测配置"""
    periods: List[str] = None  # 分析周期
    indicators: List[str] = None  # 指标列表
    lookback_days: int = 250  # 回看天数
    min_data_points: int = 50  # 最小数据点数
    parallel_workers: int = 4  # 并行工作线程数
    cache_enabled: bool = True  # 是否启用缓存
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = ['日线', '30分钟', '60分钟']
        if self.indicators is None:
            self.indicators = ['MACD', 'KDJ', 'RSI', 'MA', 'VOL']

@dataclass
class BuyPointData:
    """买点数据"""
    stock_code: str
    stock_name: str = ""
    buypoint_date: str = ""
    expected_return: float = 0.0
    holding_period: int = 0
    note: str = ""
    
@dataclass
class BacktestResult:
    """回测结果"""
    stock_code: str
    buypoint_date: str
    analysis_date: str
    periods_analyzed: List[str]
    indicators_calculated: Dict[str, Any]
    patterns_detected: Dict[str, List[str]]
    comprehensive_score: float
    signal_strength: str
    recommendations: List[str]
    execution_time: float
    
@dataclass
class BacktestSummary:
    """回测汇总"""
    total_buypoints: int
    successful_analyses: int
    success_rate: float
    average_score: float
    top_patterns: List[Tuple[str, int]]
    execution_time: float
    performance_metrics: Dict[str, Any]

class EnhancedBacktestEngine:
    """
    增强买点回测引擎
    
    提供全面的买点回测分析功能，包括多周期分析、
    指标计算、形态识别、结果评估等
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置，如果为None则使用默认配置
        """
        self.config = config or BacktestConfig()
        self.logger = get_logger(__name__)
        
        # 从容器获取服务
        container = get_container()
        try:
            self.data_access = container.resolve("DataAccessInterface")
        except:
            self.data_access = self._create_mock_data_access()
        try:
            self.indicator_registry = container.resolve("IndicatorRegistry")
        except:
            self.indicator_registry = self._create_mock_indicator_registry()
        
        # 缓存和性能统计
        self.cache = {} if self.config.cache_enabled else None
        self.performance_stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'average_execution_time': 0.0,
            'indicator_performance': {}
        }
        
        self.logger.info(f"增强回测引擎初始化完成，配置: {asdict(self.config)}")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def run_backtest(self, buypoints: List[BuyPointData], 
                    output_file: Optional[str] = None) -> BacktestSummary:
        """
        运行买点回测分析
        
        Args:
            buypoints: 买点数据列表
            output_file: 输出文件路径
            
        Returns:
            BacktestSummary: 回测汇总结果
        """
        start_time = time.time()
        self.logger.info(f"开始回测分析，买点数量: {len(buypoints)}")
        
        # 初始化结果
        results = []
        successful_analyses = 0
        total_score = 0.0
        pattern_counter = {}
        
        # 并行处理买点分析
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # 提交任务
            future_to_buypoint = {
                executor.submit(self._analyze_single_buypoint, buypoint): buypoint
                for buypoint in buypoints
            }
            
            # 收集结果
            for future in as_completed(future_to_buypoint):
                buypoint = future_to_buypoint[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        successful_analyses += 1
                        total_score += result.comprehensive_score
                        
                        # 统计形态
                        for period, patterns in result.patterns_detected.items():
                            for pattern in patterns:
                                pattern_key = f"{period}_{pattern}"
                                pattern_counter[pattern_key] = pattern_counter.get(pattern_key, 0) + 1
                        
                        self.logger.debug(f"✅ {buypoint.stock_code} 分析完成，评分: {result.comprehensive_score:.1f}")
                    
                except Exception as e:
                    self.logger.warning(f"❌ {buypoint.stock_code} 分析失败: {e}")
        
        # 计算汇总统计
        execution_time = time.time() - start_time
        success_rate = successful_analyses / len(buypoints) if buypoints else 0.0
        average_score = total_score / successful_analyses if successful_analyses > 0 else 0.0
        
        # 获取最常见的形态
        top_patterns = sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 创建汇总结果
        summary = BacktestSummary(
            total_buypoints=len(buypoints),
            successful_analyses=successful_analyses,
            success_rate=success_rate,
            average_score=average_score,
            top_patterns=top_patterns,
            execution_time=execution_time,
            performance_metrics=self._calculate_performance_metrics(results)
        )
        
        # 保存结果
        if output_file:
            self._save_results(results, summary, output_file)
        
        self.logger.info(f"回测分析完成: {successful_analyses}/{len(buypoints)} 成功，"
                        f"平均评分: {average_score:.1f}，耗时: {execution_time:.2f}秒")
        
        return summary
    
    @exception_handler(reraise=False, default_return=None)
    @performance_monitor(threshold=10.0)
    def _analyze_single_buypoint(self, buypoint: BuyPointData) -> Optional[BacktestResult]:
        """
        分析单个买点
        
        Args:
            buypoint: 买点数据
            
        Returns:
            Optional[BacktestResult]: 分析结果，失败时返回None
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._get_cache_key(buypoint)
        if self.cache and cache_key in self.cache:
            self.performance_stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            # 1. 获取多周期数据
            multi_period_data = self._get_multi_period_data(
                buypoint.stock_code, buypoint.buypoint_date
            )
            
            if not multi_period_data:
                self.logger.warning(f"无法获取 {buypoint.stock_code} 的数据")
                return None
            
            # 2. 计算指标
            indicators_calculated = self._calculate_indicators(multi_period_data)
            
            # 3. 检测形态
            patterns_detected = self._detect_patterns(indicators_calculated, buypoint.buypoint_date)
            
            # 4. 计算综合评分
            comprehensive_score = self._calculate_comprehensive_score(
                patterns_detected, indicators_calculated
            )
            
            # 5. 生成信号强度和建议
            signal_strength = self._determine_signal_strength(comprehensive_score)
            recommendations = self._generate_recommendations(
                patterns_detected, comprehensive_score
            )
            
            # 创建结果
            result = BacktestResult(
                stock_code=buypoint.stock_code,
                buypoint_date=buypoint.buypoint_date,
                analysis_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                periods_analyzed=list(multi_period_data.keys()),
                indicators_calculated=indicators_calculated,
                patterns_detected=patterns_detected,
                comprehensive_score=comprehensive_score,
                signal_strength=signal_strength,
                recommendations=recommendations,
                execution_time=time.time() - start_time
            )
            
            # 缓存结果
            if self.cache:
                self.cache[cache_key] = result
            
            # 更新性能统计
            self.performance_stats['total_analyses'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析买点 {buypoint.stock_code} 失败: {e}")
            return None

    def _get_cache_key(self, buypoint: BuyPointData) -> str:
        """生成缓存键"""
        content = f"{buypoint.stock_code}_{buypoint.buypoint_date}_{'-'.join(self.config.periods)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_multi_period_data(self, stock_code: str, buypoint_date: str) -> Dict[str, pd.DataFrame]:
        """
        获取多周期数据

        Args:
            stock_code: 股票代码
            buypoint_date: 买点日期

        Returns:
            Dict[str, pd.DataFrame]: 多周期数据
        """
        multi_period_data = {}

        # 计算数据范围
        buypoint_dt = datetime.strptime(buypoint_date, '%Y%m%d')
        start_date = (buypoint_dt - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        end_date = buypoint_dt.strftime('%Y-%m-%d')

        for period in self.config.periods:
            try:
                # 构建标准查询
                query = f"""
                SELECT code, name, date, open, high, low, close, volume FROM stock_info WHERE level = %(level)s AND code = '{stock_code}'
                AND level = '{period}'
                AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date ASC
                """

                data = self.data_access.query_dataframe(query)

                if len(data) >= self.config.min_data_points:
                    multi_period_data[period] = data
                    self.logger.debug(f"获取 {stock_code} {period} 数据: {len(data)} 条")
                else:
                    self.logger.warning(f"{stock_code} {period} 数据不足: {len(data)} < {self.config.min_data_points}")

            except Exception as e:
                self.logger.warning(f"获取 {stock_code} {period} 数据失败: {e}")

        return multi_period_data

    def _calculate_indicators(self, multi_period_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        计算技术指标

        Args:
            multi_period_data: 多周期数据

        Returns:
            Dict[str, Any]: 指标计算结果
        """
        indicators_calculated = {}

        for period, data in multi_period_data.items():
            period_indicators = {}

            for indicator_name in self.config.indicators:
                try:
                    # 从指标注册表获取指标
                    indicator_class = self.indicator_registry.get_indicator(indicator_name)
                    if indicator_class:
                        indicator = indicator_class()

                        # 计算指标
                        start_time = time.time()
                        result = indicator.calculate(data)
                        execution_time = time.time() - start_time

                        period_indicators[indicator_name] = {
                            'data': result,
                            'execution_time': execution_time,
                            'data_points': len(result) if result is not None else 0
                        }

                        # 更新性能统计
                        perf_key = f"{period}_{indicator_name}"
                        if perf_key not in self.performance_stats['indicator_performance']:
                            self.performance_stats['indicator_performance'][perf_key] = {
                                'total_time': 0.0,
                                'call_count': 0,
                                'average_time': 0.0
                            }

                        perf_stats = self.performance_stats['indicator_performance'][perf_key]
                        perf_stats['total_time'] += execution_time
                        perf_stats['call_count'] += 1
                        perf_stats['average_time'] = perf_stats['total_time'] / perf_stats['call_count']

                    else:
                        self.logger.warning(f"指标 {indicator_name} 未找到")

                except Exception as e:
                    self.logger.warning(f"计算指标 {indicator_name} 失败: {e}")

            indicators_calculated[period] = period_indicators

        return indicators_calculated

    def _detect_patterns(self, indicators_calculated: Dict[str, Any],
                        buypoint_date: str) -> Dict[str, List[str]]:
        """
        检测技术形态

        Args:
            indicators_calculated: 指标计算结果
            buypoint_date: 买点日期

        Returns:
            Dict[str, List[str]]: 检测到的形态
        """
        patterns_detected = {}

        for period, period_indicators in indicators_calculated.items():
            period_patterns = []

            for indicator_name, indicator_result in period_indicators.items():
                try:
                    if 'data' in indicator_result and indicator_result['data'] is not None:
                        data = indicator_result['data']

                        # 获取指标实例
                        indicator_class = self.indicator_registry.get_indicator(indicator_name)
                        if indicator_class:
                            indicator = indicator_class()

                            # 检测形态（如果指标支持）
                            if hasattr(indicator, 'get_patterns'):
                                patterns = indicator.get_patterns(data)
                                if patterns:
                                    period_patterns.extend([f"{indicator_name}_{p}" for p in patterns])

                            # 检测信号（如果指标支持）
                            if hasattr(indicator, 'get_signal'):
                                signal = indicator.get_signal(data)
                                if signal and signal.get('type') != SignalType.NEUTRAL:
                                    signal_desc = f"{indicator_name}_{signal.get('type', 'unknown')}_signal"
                                    period_patterns.append(signal_desc)

                except Exception as e:
                    self.logger.warning(f"检测 {indicator_name} 形态失败: {e}")

            patterns_detected[period] = period_patterns

        return patterns_detected

    def _calculate_comprehensive_score(self, patterns_detected: Dict[str, List[str]],
                                     indicators_calculated: Dict[str, Any]) -> float:
        """
        计算综合评分

        Args:
            patterns_detected: 检测到的形态
            indicators_calculated: 指标计算结果

        Returns:
            float: 综合评分 (0-100)
        """
        total_score = 0.0
        total_weight = 0.0

        # 周期权重
        period_weights = {
            '日线': 0.5,
            '60分钟': 0.3,
            '30分钟': 0.2
        }

        for period, patterns in patterns_detected.items():
            period_weight = period_weights.get(period, 0.1)
            period_score = 0.0

            # 基础分数
            base_score = 50.0

            # 形态加分
            pattern_bonus = min(len(patterns) * 5, 30)  # 每个形态5分，最多30分

            # 强势形态额外加分
            strong_patterns = [p for p in patterns if any(keyword in p.lower()
                             for keyword in ['golden', 'bullish', 'breakout', 'strong'])]
            strong_bonus = len(strong_patterns) * 10

            # 弱势形态扣分
            weak_patterns = [p for p in patterns if any(keyword in p.lower()
                           for keyword in ['bearish', 'weak', 'decline', 'sell'])]
            weak_penalty = len(weak_patterns) * 8

            period_score = base_score + pattern_bonus + strong_bonus - weak_penalty
            period_score = max(0.0, min(100.0, period_score))  # 限制在0-100范围

            total_score += period_score * period_weight
            total_weight += period_weight

        # 计算加权平均分
        final_score = total_score / total_weight if total_weight > 0 else 50.0
        return round(final_score, 1)

    def _determine_signal_strength(self, score: float) -> str:
        """确定信号强度"""
        if score >= 80:
            return "强烈买入"
        elif score >= 70:
            return "买入"
        elif score >= 60:
            return "弱买入"
        elif score >= 40:
            return "中性"
        elif score >= 30:
            return "弱卖出"
        else:
            return "卖出"

    def _generate_recommendations(self, patterns_detected: Dict[str, List[str]],
                                score: float) -> List[str]:
        """生成投资建议"""
        recommendations = []

        # 基于评分的建议
        if score >= 75:
            recommendations.append("技术形态良好，建议关注")
        elif score >= 60:
            recommendations.append("技术形态一般，谨慎操作")
        else:
            recommendations.append("技术形态较弱，建议观望")

        # 基于形态的建议
        all_patterns = []
        for patterns in patterns_detected.values():
            all_patterns.extend(patterns)

        if any('MACD' in p for p in all_patterns):
            recommendations.append("MACD指标活跃，关注趋势变化")

        if any('volume' in p.lower() for p in all_patterns):
            recommendations.append("成交量配合良好，关注资金流向")

        if len(all_patterns) >= 5:
            recommendations.append("多指标共振，信号较强")

        return recommendations

    def _calculate_performance_metrics(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """计算性能指标"""
        if not results:
            return {}

        scores = [r.comprehensive_score for r in results]
        execution_times = [r.execution_time for r in results]

        return {
            'score_statistics': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'execution_statistics': {
                'mean_time': np.mean(execution_times),
                'total_time': np.sum(execution_times),
                'min_time': np.min(execution_times),
                'max_time': np.max(execution_times)
            },
            'signal_distribution': self._calculate_signal_distribution(results)
        }

    def _calculate_signal_distribution(self, results: List[BacktestResult]) -> Dict[str, int]:
        """计算信号分布"""
        distribution = {}
        for result in results:
            signal = result.signal_strength
            distribution[signal] = distribution.get(signal, 0) + 1
        return distribution

    def _save_results(self, results: List[BacktestResult],
                     summary: BacktestSummary, output_file: str):
        """保存结果到文件"""
        try:
            # 准备输出数据
            output_data = {
                'summary': asdict(summary),
                'results': [asdict(result) for result in results],
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'config': asdict(self.config),
                    'performance_stats': self.performance_stats
                }
            }

            # 保存为JSON文件
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"结果已保存到: {output_file}")

        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")

    @performance_monitor(threshold=5.0)
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'engine_stats': self.performance_stats,
            'config': asdict(self.config),
            'cache_stats': {
                'enabled': self.config.cache_enabled,
                'size': len(self.cache) if self.cache else 0,
                'hit_rate': (self.performance_stats['cache_hits'] /
                           max(self.performance_stats['total_analyses'], 1))
            }
        }

    def _create_mock_data_access(self):
        """创建模拟数据访问对象"""
        class MockDataAccess:
            def query_dataframe(self, query: str) -> pd.DataFrame:
                # 生成模拟股票数据
                dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
                data = []
                for i, date in enumerate(dates):
                    data.append({
                        'code': '000001',
                        'name': '平安银行',
                        'date': date.strftime('%Y-%m-%d'),
                        'open': 10.0 + np.random.normal(0, 0.5),
                        'high': 10.5 + np.random.normal(0, 0.5),
                        'low': 9.5 + np.random.normal(0, 0.5),
                        'close': 10.0 + np.random.normal(0, 0.5),
                        'volume': 1000000 + np.random.randint(0, 500000),
                        'turnover_rate': np.random.uniform(0.5, 5.0)
                    })
                return pd.DataFrame(data)

        return MockDataAccess()

    def _create_mock_indicator_registry(self):
        """创建模拟指标注册表"""
        class MockIndicatorRegistry:
            def get_indicator(self, name: str):
                class MockIndicator:
                    def __init__(self):
                        self.name = name

                    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                        # 返回简单的模拟指标数据
                        result = data.copy()
                        if name == 'MA':
                            result['MA'] = data['close'].rolling(20).mean()
                        elif name == 'MACD':
                            result['MACD'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
                        elif name == 'RSI':
                            result['RSI'] = 50 + np.random.normal(0, 10, len(data))
                        return result

                    def get_patterns(self, data: pd.DataFrame) -> List[str]:
                        return ['pattern1', 'pattern2']

                    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
                        return {'type': SignalType.BUY, 'strength': 0.7}

                return MockIndicator

        return MockIndicatorRegistry()
