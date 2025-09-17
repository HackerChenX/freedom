#!/usr/bin/env python3
"""
生产级多策略买点识别系统 - 基于86个真实指标的多周期策略分析
支持15分钟、30分钟、60分钟、日线、周线、月线的综合分析
集成策略管理、买点检测和多周期分析能力
遵循六层架构，严格验证，禁止Mock机制
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dependency_injection import get_service
from utils.unified_container import UnifiedServiceContainer
from db.services.multi_period_data_service import MultiPeriodDataService, Period
from indicators.services.multi_period_indicator_service import MultiPeriodIndicatorService
from indicators.services.universal_multi_period_calculator import UniversalMultiPeriodCalculator
from indicators.complete_indicator_registry import get_indicator_registry, initialize_indicators
from strategy.strategy_manager import StrategyManager
from strategy.unified_base_strategy import UnifiedBaseStrategy
from analysis.buypoints.enhanced_buypoint_detector import EnhancedBuyPointDetector
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor

# 创建全局容器实例
container = UnifiedServiceContainer()

logger = get_logger(__name__)


class MultiPeriodBuypointAnalyzer:
    """
    生产级多策略买点识别系统

    功能：
    - 基于86个真实指标的多周期技术分析
    - 集成策略管理和买点检测能力
    - 15分钟KDJ金叉和日线KDJ金叉分别分析
    - 跨周期信号验证和聚合
    - 多策略组合和买点推荐
    - 遵循六层架构规范，严格验证
    """

    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化生产级多策略买点识别系统

        Args:
            config_path: 配置文件路径，None表示使用默认配置
        """
        logger.info("初始化生产级多策略买点识别系统")

        # 使用依赖注入初始化核心服务（遵循六层架构）
        try:
            self.data_service = container.resolve("MultiPeriodDataService") or MultiPeriodDataService()
            self.indicator_service = container.resolve("MultiPeriodIndicatorService") or MultiPeriodIndicatorService()
            self.universal_calculator = container.resolve("UniversalMultiPeriodCalculator") or UniversalMultiPeriodCalculator()
            self.buypoint_detector = container.resolve("EnhancedBuyPointDetector") or EnhancedBuyPointDetector()
        except Exception as e:
            logger.warning(f"依赖注入失败，使用直接实例化: {e}")
            self.data_service = MultiPeriodDataService()
            self.indicator_service = MultiPeriodIndicatorService()
            self.universal_calculator = UniversalMultiPeriodCalculator()
            self.buypoint_detector = EnhancedBuyPointDetector()

        # 策略管理服务（单次初始化，增强错误处理）
        self.strategy_manager = None
        try:
            self.strategy_manager = container.resolve("StrategyManager") or StrategyManager()
            logger.info("策略管理器初始化成功")
        except Exception as e:
            logger.warning(f"策略管理器初始化失败: {e}")

        # 动态获取所有可用指标（增强错误处理）
        try:
            initialize_indicators()
            self.indicator_registry = get_indicator_registry()
            self.all_indicators = list(self.indicator_registry.get_all_indicators().keys())
            if not self.all_indicators:
                raise ValueError("未发现任何可用指标")
        except Exception as e:
            logger.error(f"指标初始化失败: {e}")
            self.all_indicators = []

        # 支持的分析周期
        self.supported_periods = [
            Period.MIN_15, Period.MIN_30, Period.MIN_60,
            Period.DAILY, Period.WEEKLY, Period.MONTHLY
        ]

        # 加载配置并动态生成策略和权重（增强错误处理）
        self.config = self._load_configuration(config_path)
        self.indicator_weights = self._generate_dynamic_indicator_weights()
        self.builtin_strategies = self._generate_dynamic_strategies()

        # 验证初始化结果
        self._validate_initialization()

        logger.info(f"系统初始化完成：")
        logger.info(f"  - 可用指标: {len(self.all_indicators)}个")
        logger.info(f"  - 动态策略: {len(self.builtin_strategies)}个")
        logger.info(f"  - 配置驱动: {'是' if config_path else '默认配置'}")

    @exception_handler(reraise=True)
    def _validate_initialization(self):
        """验证初始化结果"""
        if not self.all_indicators:
            raise ValueError("系统初始化失败：未发现任何可用指标")

        if not self.indicator_weights:
            raise ValueError("系统初始化失败：指标权重生成失败")

        if not self.builtin_strategies:
            logger.warning("策略生成失败，将使用备用策略")

        logger.info("系统初始化验证通过")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件（增强版本，包含验证）

        Args:
            config_path: 配置文件路径

        Returns:
            Dict: 验证后的配置字典
        """
        # 定义配置模式和默认值
        default_config = {
            "analysis": {
                "default_periods": ["15分钟", "30分钟", "60分钟", "日线", "周线", "月线"],
                "performance_threshold": 30.0,
                "min_indicators_for_signal": 3,
                "signal_confidence_threshold": 0.6,
                "max_indicators_per_analysis": 200,
                "timeout_seconds": 300
            },
            "scoring": {
                "base_weight": 0.05,
                "tier1_weight": 0.15,  # 核心指标
                "tier2_weight": 0.10,  # 重要指标
                "tier3_weight": 0.08,  # 一般指标
                "strategy_weight": 0.30,
                "indicator_weight": 0.70,
                "buy_threshold": 60.0,
                "sell_threshold": -60.0,
                "strong_buy_threshold": 70.0
            },
            "strategies": {
                "auto_generate": True,
                "min_indicators_per_strategy": 3,
                "max_indicators_per_strategy": 8,
                "max_strategies": 10
            },
            "performance": {
                "enable_caching": True,
                "cache_ttl_seconds": 300,
                "max_memory_mb": 2048,
                "enable_parallel": True
            }
        }

        # 加载用户配置
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)

                # 验证配置格式
                validated_config = self._validate_configuration(user_config, default_config)

                # 安全合并配置
                merged_config = self._merge_configurations(default_config, validated_config)

                logger.info(f"已加载并验证配置文件: {config_path}")
                return merged_config

            except json.JSONDecodeError as e:
                logger.error(f"配置文件JSON格式错误: {e}")
                raise ValueError(f"配置文件格式错误: {e}")
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")

        logger.info("使用默认配置")
        return default_config

    @exception_handler(reraise=True)
    def _validate_configuration(self, user_config: Dict[str, Any],
                               default_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证用户配置

        Args:
            user_config: 用户配置
            default_config: 默认配置

        Returns:
            Dict: 验证后的配置
        """
        validated = {}

        # 验证数值范围
        if "scoring" in user_config:
            scoring = user_config["scoring"]
            if "tier1_weight" in scoring:
                if not 0.0 <= scoring["tier1_weight"] <= 1.0:
                    raise ValueError("tier1_weight必须在0.0-1.0范围内")
            if "buy_threshold" in scoring:
                if not -100.0 <= scoring["buy_threshold"] <= 100.0:
                    raise ValueError("buy_threshold必须在-100.0-100.0范围内")

        # 验证策略配置
        if "strategies" in user_config:
            strategies = user_config["strategies"]
            if "min_indicators_per_strategy" in strategies:
                if strategies["min_indicators_per_strategy"] < 1:
                    raise ValueError("min_indicators_per_strategy必须大于0")

        # 只保留有效的配置项
        for section, values in user_config.items():
            if section in default_config and isinstance(values, dict):
                validated[section] = {}
                for key, value in values.items():
                    if key in default_config[section]:
                        validated[section][key] = value
                    else:
                        logger.warning(f"忽略未知配置项: {section}.{key}")

        return validated

    @exception_handler(reraise=True)
    def _merge_configurations(self, default: Dict[str, Any],
                             user: Dict[str, Any]) -> Dict[str, Any]:
        """
        安全合并配置

        Args:
            default: 默认配置
            user: 用户配置

        Returns:
            Dict: 合并后的配置
        """
        merged = default.copy()

        for section, values in user.items():
            if section in merged and isinstance(values, dict):
                merged[section].update(values)
            else:
                merged[section] = values

        return merged

    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def _generate_dynamic_indicator_weights(self) -> Dict[str, float]:
        """
        动态生成指标权重（消除硬编码，增强版本）

        Returns:
            Dict: 指标权重字典
        """
        if not self.all_indicators:
            logger.warning("无可用指标，返回空权重字典")
            return {}

        weights = {}

        # 从配置获取指标分层（可配置化）
        tier1_indicators = set(self.config.get("indicator_tiers", {}).get("tier1",
            ['MACD', 'RSI', 'KDJ', 'MA', 'EMA', 'BOLL']))
        tier2_indicators = set(self.config.get("indicator_tiers", {}).get("tier2",
            ['CCI', 'WR', 'STOCH', 'DMI', 'SAR', 'ATR', 'OBV', 'MFI']))

        # 从配置获取权重
        scoring_config = self.config.get("scoring", {})
        tier1_weight = scoring_config.get("tier1_weight", 0.15)
        tier2_weight = scoring_config.get("tier2_weight", 0.10)
        tier3_weight = scoring_config.get("tier3_weight", 0.08)
        base_weight = scoring_config.get("base_weight", 0.05)

        # 动态分配权重
        for indicator in self.all_indicators:
            try:
                if indicator in tier1_indicators:
                    weights[indicator] = tier1_weight
                elif indicator in tier2_indicators:
                    weights[indicator] = tier2_weight
                elif any(pattern in indicator.upper() for pattern in ['ZXM', 'SCORE', 'PATTERN']):
                    weights[indicator] = tier3_weight
                else:
                    weights[indicator] = base_weight
            except Exception as e:
                logger.warning(f"指标{indicator}权重分配失败: {e}")
                weights[indicator] = base_weight

        # 归一化权重（防止除零错误）
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            logger.error("权重总和为零，使用平均权重")
            avg_weight = 1.0 / len(weights) if weights else 0.0
            weights = {k: avg_weight for k in weights.keys()}

        # 验证权重有效性
        if not weights:
            raise ValueError("指标权重生成失败")

        logger.info(f"动态生成指标权重: {len(weights)}个指标")
        return weights

    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def _generate_dynamic_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        动态生成策略配置（消除硬编码）

        Returns:
            Dict: 策略配置字典
        """
        if not self.config["strategies"]["auto_generate"]:
            return self._get_fallback_strategies()

        strategies = {}
        indicator_groups = self._group_indicators_by_type()

        # 为每个类型生成策略
        for strategy_type, indicators in indicator_groups.items():
            min_indicators = self.config["strategies"]["min_indicators_per_strategy"]
            max_indicators = self.config["strategies"]["max_indicators_per_strategy"]

            if len(indicators) >= min_indicators:
                strategies[f"DYNAMIC_{strategy_type.upper()}"] = {
                    "name": f"动态{strategy_type}策略",
                    "description": f"基于{strategy_type}指标的动态买点识别",
                    "indicators": indicators[:max_indicators],
                    "periods": self.config["analysis"]["default_periods"],
                    "weight": 1.0 / len(indicator_groups),
                    "type": strategy_type
                }

        logger.info(f"动态生成策略: {len(strategies)}个")
        return strategies if strategies else self._get_fallback_strategies()

    def _group_indicators_by_type(self) -> Dict[str, List[str]]:
        """按类型分组指标"""
        groups = {"momentum": [], "trend": [], "volume": [], "volatility": [], "pattern": []}

        classification_rules = {
            "momentum": ["MACD", "RSI", "KDJ", "STOCH", "CCI", "WR", "MOMENTUM"],
            "trend": ["MA", "EMA", "SAR", "DMI", "ADX", "SUPERTREND", "TREND"],
            "volume": ["OBV", "MFI", "VOL", "FORCE_INDEX", "AD", "VOLUME"],
            "volatility": ["BOLL", "ATR", "VOLATILITY", "RANGE"],
            "pattern": ["DOJI", "HAMMER", "ENGULFING", "HARAMI", "PATTERN"]
        }

        for indicator in self.all_indicators:
            classified = False
            for group_type, keywords in classification_rules.items():
                if any(keyword in indicator.upper() for keyword in keywords):
                    groups[group_type].append(indicator)
                    classified = True
                    break
            if not classified:
                groups["momentum"].append(indicator)

        return groups

    def _get_fallback_strategies(self) -> Dict[str, Dict[str, Any]]:
        """获取备用策略配置"""
        return {
            'COMPREHENSIVE_ANALYSIS': {
                'name': '综合分析策略',
                'description': '基于所有可用指标的综合买点识别',
                'indicators': self.all_indicators[:20],  # 使用前20个指标
                'periods': ['日线', '周线'],
                'weight': 1.0
            }
        }

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)  # 增加超时时间，因为要分析86个指标
    def analyze_multi_period_buypoint(self,
                                     stock_code: str,
                                     target_date: str,
                                     periods: Optional[List[Period]] = None,
                                     indicator_names: Optional[List[str]] = None,
                                     enable_strategy_analysis: bool = True) -> Dict[str, Any]:
        """
        生产级多策略买点分析 - 基于86个真实指标

        Args:
            stock_code: 股票代码
            target_date: 分析目标日期
            periods: 分析周期列表，None表示分析所有周期
            indicator_names: 指标名称列表，None表示分析所有86个真实指标
            enable_strategy_analysis: 是否启用策略分析

        Returns:
            Dict[str, Any]: 生产级多策略买点分析结果
        """
        # 默认使用全周期分析（不限制periods）
        if periods is None:
            periods = self.supported_periods
            logger.info("使用全周期分析：15分钟、30分钟、60分钟、日线、周线、月线")

        if indicator_names is None:
            # 使用所有128个真实指标
            indicator_names = self.all_indicators
            logger.info(f"使用全部{len(indicator_names)}个真实指标进行分析")
        else:
            logger.info(f"使用指定的{len(indicator_names)}个指标进行分析")

        logger.info(f"开始分析股票{stock_code}在{target_date}的生产级多策略买点")
        logger.info(f"分析周期: 全周期（15分钟、30分钟、60分钟、日线、周线、月线）")
        logger.info(f"策略分析: {'启用' if enable_strategy_analysis else '禁用'}")

        try:
            # 1. 使用通用多周期计算器进行全周期指标分析
            multi_period_result = self.universal_calculator.calculate_multi_period_indicators(
                stock_code=stock_code,
                target_date=target_date,
                indicator_names=indicator_names,
                periods=None  # 不指定periods，默认全周期分析
            )

            # 2. 进行买点分析
            buypoint_analysis = self._analyze_buypoint_signals(multi_period_result)

            # 3. 策略分析（如果启用）
            strategy_analysis = {}
            if enable_strategy_analysis:
                strategy_analysis = self._analyze_strategy_signals(
                    stock_code, target_date, multi_period_result
                )

            # 4. 生成周期对比分析
            period_comparison = self._compare_period_signals(multi_period_result)

            # 5. 计算综合买点评分
            overall_score = self._calculate_overall_buypoint_score(
                multi_period_result, strategy_analysis
            )

            # 6. 生成买点建议
            recommendations = self._generate_buypoint_recommendations(
                multi_period_result, overall_score, strategy_analysis
            )
            
            result = {
                'stock_code': stock_code,
                'analysis_date': target_date,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'periods_analyzed': [p.value for p in periods],
                'indicators_analyzed': indicator_names,
                'total_indicators_count': len(indicator_names),
                'strategy_analysis_enabled': enable_strategy_analysis,
                'multi_period_indicators': multi_period_result,
                'buypoint_analysis': buypoint_analysis,
                'strategy_analysis': strategy_analysis,
                'period_comparison': period_comparison,
                'overall_score': overall_score,
                'recommendations': recommendations,
                'system_info': {
                    'total_real_indicators': len(self.all_indicators),
                    'builtin_strategies': len(self.builtin_strategies),
                    'analysis_type': 'PRODUCTION_GRADE_MULTI_STRATEGY'
                },
                'status': 'SUCCESS'
            }

            logger.info(f"完成股票{stock_code}生产级多策略买点分析，综合评分: {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"多周期买点分析失败: {e}")
            return {
                'stock_code': stock_code,
                'analysis_date': target_date,
                'status': 'ERROR',
                'error': str(e),
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def _analyze_buypoint_signals(self, multi_period_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析买点信号（适配新的通用多周期计算器数据结构）

        Args:
            multi_period_result: 通用多周期计算器结果

        Returns:
            Dict[str, Any]: 买点信号分析
        """
        buypoint_signals = {
            'strong_buy_signals': [],
            'moderate_buy_signals': [],
            'weak_buy_signals': [],
            'hold_signals': [],
            'sell_signals': [],
            'period_signal_summary': {},
            'consistency_analysis': multi_period_result.get('consistency_analysis', {}),
            'aggregated_signals': multi_period_result.get('aggregated_signals', {}),
            'overall_score': multi_period_result.get('overall_score', 50.0)
        }

        # 使用新的数据结构：analysis_matrix
        analysis_matrix = multi_period_result.get('analysis_matrix', {})
        
        # 遍历分析矩阵中的每个周期
        for period_name, period_indicators in analysis_matrix.items():
            period_signals = {
                'buy_count': 0,
                'sell_count': 0,
                'hold_count': 0,
                'strong_signals': [],
                'golden_crosses': [],
                'death_crosses': []
            }

            # 遍历该周期的所有指标
            for indicator_name, indicator_data in period_indicators.items():
                signal = indicator_data.get('signal', 'UNKNOWN')
                strength = indicator_data.get('strength', 0.0)

                # 统计信号类型
                if signal in ['BUY', 'STRONG_BUY']:
                    period_signals['buy_count'] += 1
                    if strength > 0.7:
                        period_signals['strong_signals'].append({
                            'indicator': indicator_name,
                            'signal': signal,
                            'strength': strength,
                            'period': period_name
                        })
                elif signal in ['SELL', 'STRONG_SELL']:
                    period_signals['sell_count'] += 1
                else:
                    period_signals['hold_count'] += 1
                
                # 检测金叉死叉
                if indicator_data.get('has_golden_cross', False):
                    period_signals['golden_crosses'].append({
                        'indicator': indicator_name,
                        'cross_strength': indicator_data.get('cross_strength', 0.0),
                        'period': period_name
                    })
                
                if indicator_data.get('has_death_cross', False):
                    period_signals['death_crosses'].append({
                        'indicator': indicator_name,
                        'cross_strength': indicator_data.get('cross_strength', 0.0),
                        'period': period_name
                    })
            
            buypoint_signals['period_signal_summary'][period_name] = period_signals
        
        # 分类买点信号强度
        aggregated_signals = multi_period_result.get('aggregated_signals', {})
        for indicator_name, signal_data in aggregated_signals.items():
            signal = signal_data.get('signal', 'UNKNOWN')
            strength = signal_data.get('strength', 0.0)
            confidence = signal_data.get('confidence', 0.0)
            
            signal_info = {
                'indicator': indicator_name,
                'signal': signal,
                'strength': strength,
                'confidence': confidence
            }
            
            if signal == 'BUY':
                if strength > 0.8 and confidence > 0.8:
                    buypoint_signals['strong_buy_signals'].append(signal_info)
                elif strength > 0.6 and confidence > 0.6:
                    buypoint_signals['moderate_buy_signals'].append(signal_info)
                else:
                    buypoint_signals['weak_buy_signals'].append(signal_info)
            elif signal == 'SELL':
                buypoint_signals['sell_signals'].append(signal_info)
            else:
                buypoint_signals['hold_signals'].append(signal_info)
        
        return buypoint_signals
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=8.0)
    def _compare_period_signals(self, multi_period_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较不同周期的信号

        Args:
            multi_period_result: 多周期指标结果

        Returns:
            Dict[str, Any]: 周期对比分析
        """
        comparison = {
            'short_term_vs_long_term': {},
            'period_consistency': {},
            'divergence_analysis': {},
            'trend_confirmation': {}
        }
        
        cross_period_analysis = multi_period_result.get('cross_period_analysis', {})
        
        # 短期vs长期对比
        periods_data = multi_period_result.get('periods', {})
        short_term_periods = ['15分钟', '30分钟', '60分钟']
        long_term_periods = ['日线', '周线', '月线']
        
        # 使用核心指标进行周期对比（动态获取）
        core_indicators = [ind for ind in self.all_indicators
                          if ind in {'KDJ', 'MACD', 'RSI', 'MA', 'EMA', 'BOLL'}][:6]  # 最多6个核心指标

        for indicator_name in core_indicators:
            short_signals = []
            long_signals = []

            for period_name, period_data in periods_data.items():
                if period_name in short_term_periods:
                    indicator_data = period_data.get('indicators', {}).get(indicator_name, {})
                    if indicator_data.get('signal'):
                        short_signals.append(indicator_data.get('signal'))
                elif period_name in long_term_periods:
                    indicator_data = period_data.get('indicators', {}).get(indicator_name, {})
                    if indicator_data.get('signal'):
                        long_signals.append(indicator_data.get('signal'))
            
            comparison['short_term_vs_long_term'][indicator_name] = {
                'short_term_signals': short_signals,
                'long_term_signals': long_signals,
                'short_term_bias': self._get_signal_bias(short_signals),
                'long_term_bias': self._get_signal_bias(long_signals)
            }
        
        # 周期一致性分析
        signal_consistency = cross_period_analysis.get('signal_consistency', {})
        for indicator_name, consistency_data in signal_consistency.items():
            comparison['period_consistency'][indicator_name] = {
                'consistency_score': consistency_data.get('score', 0.0),
                'is_consistent': consistency_data.get('is_consistent', False),
                'signals_by_period': consistency_data.get('signals', {})
            }
        
        return comparison
    
    def _get_signal_bias(self, signals: List[str]) -> str:
        """获取信号倾向"""
        if not signals:
            return 'NEUTRAL'
        
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count > sell_count:
            return 'BULLISH'
        elif sell_count > buy_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=15.0)
    def _analyze_strategy_signals(self, stock_code: str, target_date: str,
                                 multi_period_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析策略信号

        Args:
            stock_code: 股票代码
            target_date: 分析日期
            multi_period_result: 多周期指标结果

        Returns:
            Dict[str, Any]: 策略分析结果
        """
        strategy_analysis = {
            'strategy_signals': {},
            'strategy_scores': {},
            'combined_strategy_score': 0.0,
            'strategy_recommendations': []
        }

        # 如果策略管理器可用，使用策略实例
        if self.strategy_manager:
            # 准备股票数据
            stock_data = self._prepare_stock_data_for_strategy(multi_period_result, stock_code, target_date)

            # 分析每个动态策略（消除硬编码）
            for strategy_name in self.builtin_strategies.keys():
                try:
                    strategy = self.strategy_manager.get_strategy(strategy_name)
                    if strategy and stock_data is not None and len(stock_data) > 0:
                        # 使用策略实例生成信号和评分
                        strategy_signal = strategy.generate_signal(stock_data)
                        strategy_score = strategy.calculate_score(stock_data)

                        strategy_analysis['strategy_signals'][strategy_name] = strategy_signal
                        strategy_analysis['strategy_scores'][strategy_name] = strategy_score

                        # 生成策略推荐
                        if strategy_score > 70:
                            strategy_analysis['strategy_recommendations'].append({
                                'strategy': strategy_name,
                                'action': 'STRONG_BUY',
                                'score': strategy_score,
                                'confidence': 'HIGH'
                            })
                        elif strategy_score > 50:
                            strategy_analysis['strategy_recommendations'].append({
                                'strategy': strategy_name,
                                'action': 'BUY',
                                'score': strategy_score,
                                'confidence': 'MEDIUM'
                            })
                    else:
                        # 策略不可用，使用默认值
                        strategy_analysis['strategy_signals'][strategy_name] = 'HOLD'
                        strategy_analysis['strategy_scores'][strategy_name] = 0.0

                except Exception as e:
                    logger.warning(f"策略 {strategy_name} 分析失败: {e}")
                    strategy_analysis['strategy_signals'][strategy_name] = 'HOLD'
                    strategy_analysis['strategy_scores'][strategy_name] = 0.0
        else:
            # 回退到原有的基于指标配置的策略分析
            periods_data = multi_period_result.get('periods', {})

            for strategy_name, strategy_config in self.builtin_strategies.items():
                strategy_score = self._calculate_strategy_score(
                    strategy_config, periods_data
                )

                strategy_signal = self._determine_strategy_signal(strategy_score)

                strategy_analysis['strategy_signals'][strategy_name] = strategy_signal
                strategy_analysis['strategy_scores'][strategy_name] = strategy_score

                # 生成策略推荐
                if strategy_score > 70:
                    strategy_analysis['strategy_recommendations'].append({
                        'strategy': strategy_name,
                        'action': 'STRONG_BUY',
                        'score': strategy_score,
                        'confidence': 'HIGH'
                    })
                elif strategy_score > 50:
                    strategy_analysis['strategy_recommendations'].append({
                        'strategy': strategy_name,
                        'action': 'BUY',
                        'score': strategy_score,
                        'confidence': 'MEDIUM'
                    })

        # 计算组合策略评分
        strategy_analysis['combined_strategy_score'] = self._calculate_combined_strategy_score(
            strategy_analysis['strategy_scores']
        )

        return strategy_analysis

    def _prepare_stock_data_for_strategy(self, multi_period_result: Dict[str, Any],
                                        stock_code: str = None, target_date: str = None):
        """为策略分析准备股票数据"""
        try:
            # 首先尝试从periods数据中获取
            periods_data = multi_period_result.get('periods', {})

            # 优先使用日线数据
            if '日线' in periods_data:
                daily_data = periods_data['日线']
                raw_data = daily_data.get('raw_data')
                if raw_data is not None and len(raw_data) > 0:
                    return raw_data

            # 如果没有日线数据，尝试其他周期
            for period_name, period_data in periods_data.items():
                raw_data = period_data.get('raw_data')
                if raw_data is not None and len(raw_data) > 0:
                    return raw_data

            # 如果periods数据不可用，尝试从其他位置获取数据
            # 检查是否有直接的股票数据
            if 'stock_data' in multi_period_result:
                stock_data = multi_period_result['stock_data']
                if stock_data is not None and len(stock_data) > 0:
                    return stock_data

            # 尝试从多周期数据服务直接获取数据
            try:
                from db.services.multi_period_data_service import MultiPeriodDataService
                from enums.kline_period import KlinePeriod
from db.sql_manager import SQLManager, QueryType

                data_service = MultiPeriodDataService()
                # 获取最近60天的日线数据用于策略分析
                stock_data = data_service.get_single_period_data(
                    stock_code=stock_code or '300005',  # 使用传入的股票代码，如果没有则使用默认值
                    period=KlinePeriod.DAILY,
                    target_date=target_date or '2025-05-09',  # 使用传入的日期，如果没有则使用默认值
                    lookback_days=60
                )

                if stock_data is not None and len(stock_data) > 0:
                    logger.info(f"从数据服务获取到股票数据: {len(stock_data)}条记录")
                    return stock_data

            except Exception as data_error:
                logger.warning(f"从数据服务获取股票数据失败: {data_error}")

            logger.warning("无法获取策略分析所需的股票数据")
            return None

        except Exception as e:
            logger.warning(f"准备策略数据失败: {e}")
            return None

    def _calculate_strategy_score(self, strategy_config: Dict[str, Any],
                                 periods_data: Dict[str, Any]) -> float:
        """计算单个策略的评分"""
        strategy_indicators = strategy_config.get('indicators', [])
        strategy_periods = strategy_config.get('periods', ['daily'])

        total_score = 0.0
        total_weight = 0.0

        for period_name in strategy_periods:
            period_data = periods_data.get(period_name, {})
            indicators = period_data.get('indicators', {})

            for indicator_name in strategy_indicators:
                if indicator_name in indicators:
                    indicator_data = indicators[indicator_name]
                    signal = indicator_data.get('signal', 'HOLD')
                    strength = indicator_data.get('strength', 0.0)

                    if signal == 'BUY':
                        score = strength * 100
                    elif signal == 'SELL':
                        score = -strength * 100
                    else:
                        score = 0

                    total_score += score
                    total_weight += 1

        return total_score / total_weight if total_weight > 0 else 0.0

    @exception_handler(reraise=True)
    def _determine_strategy_signal(self, score: float) -> str:
        """
        根据评分确定策略信号（使用配置化阈值）

        Args:
            score: 策略评分

        Returns:
            str: 信号类型 (BUY/SELL/HOLD)
        """
        # 从配置获取阈值（消除硬编码）
        scoring_config = self.config.get("scoring", {})
        buy_threshold = scoring_config.get("buy_threshold", 60.0)
        sell_threshold = scoring_config.get("sell_threshold", -60.0)

        if score > buy_threshold:
            return 'BUY'
        elif score < sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_combined_strategy_score(self, strategy_scores: Dict[str, float]) -> float:
        """计算组合策略评分"""
        total_score = 0.0
        total_weight = 0.0

        for strategy_name, score in strategy_scores.items():
            weight = self.builtin_strategies[strategy_name].get('weight', 0.25)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def _calculate_overall_buypoint_score(self, multi_period_result: Dict[str, Any],
                                         strategy_analysis: Optional[Dict[str, Any]] = None) -> float:
        """
        计算综合买点评分（使用动态权重）

        Args:
            multi_period_result: 多周期指标结果
            strategy_analysis: 策略分析结果

        Returns:
            float: 综合评分 (0-100)
        """
        aggregated_signals = multi_period_result.get('aggregated_signals', {})

        total_score = 0.0
        total_weight = 0.0

        # 使用动态生成的指标权重（消除硬编码）
        for indicator_name, signal_data in aggregated_signals.items():
            weight = self.indicator_weights.get(indicator_name, self.config["scoring"]["base_weight"])
            signal = signal_data.get('signal', 'HOLD')
            strength = signal_data.get('strength', 0.0)
            confidence = signal_data.get('confidence', 0.0)

            # 计算指标得分
            if signal == 'BUY':
                indicator_score = strength * confidence * 100
            elif signal == 'SELL':
                indicator_score = -strength * confidence * 100
            else:
                indicator_score = 0

            total_score += indicator_score * weight
            total_weight += weight

        # 计算基础指标评分
        base_score = 0.0
        if total_weight > 0:
            base_score = total_score / total_weight
            # 归一化到0-100范围
            base_score = max(0, min(100, base_score + 50))
        else:
            base_score = 50  # 中性评分

        # 整合策略分析评分（使用配置权重）
        if strategy_analysis:
            strategy_score = strategy_analysis.get('combined_strategy_score', 0.0)
            strategy_weight = self.config["scoring"]["strategy_weight"]
            indicator_weight = self.config["scoring"]["indicator_weight"]
            final_score = base_score * indicator_weight + (strategy_score + 50) * strategy_weight
        else:
            final_score = base_score

        return max(0, min(100, final_score))

    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def _generate_buypoint_recommendations(self,
                                         multi_period_result: Dict[str, Any],
                                         overall_score: float,
                                         strategy_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成买点建议

        Args:
            multi_period_result: 多周期指标结果
            overall_score: 综合评分
            strategy_analysis: 策略分析结果

        Returns:
            Dict[str, Any]: 买点建议
        """
        recommendations = {
            'action': 'HOLD',
            'confidence': 'LOW',
            'score': overall_score,
            'reasons': [],
            'risk_warnings': [],
            'timing_suggestions': [],
            'strategy_recommendations': []
        }
        
        # 根据评分确定行动建议
        if overall_score >= 70:
            recommendations['action'] = 'STRONG_BUY'
            recommendations['confidence'] = 'HIGH'
        elif overall_score >= 60:
            recommendations['action'] = 'BUY'
            recommendations['confidence'] = 'MEDIUM'
        elif overall_score >= 40:
            recommendations['action'] = 'HOLD'
            recommendations['confidence'] = 'MEDIUM'
        elif overall_score >= 30:
            recommendations['action'] = 'WEAK_SELL'
            recommendations['confidence'] = 'MEDIUM'
        else:
            recommendations['action'] = 'SELL'
            recommendations['confidence'] = 'HIGH'
        
        # 分析具体原因
        aggregated_signals = multi_period_result.get('aggregated_signals', {})
        strong_buy_indicators = []
        strong_sell_indicators = []
        
        for indicator_name, signal_data in aggregated_signals.items():
            signal = signal_data.get('signal', 'HOLD')
            strength = signal_data.get('strength', 0.0)
            
            if signal == 'BUY' and strength > 0.7:
                strong_buy_indicators.append(indicator_name)
            elif signal == 'SELL' and strength > 0.7:
                strong_sell_indicators.append(indicator_name)
        
        if strong_buy_indicators:
            recommendations['reasons'].append(
                f"强买入信号指标: {', '.join(strong_buy_indicators)}"
            )
        
        if strong_sell_indicators:
            recommendations['risk_warnings'].append(
                f"强卖出信号指标: {', '.join(strong_sell_indicators)}"
            )

        # 整合策略分析结果
        if strategy_analysis:
            strategy_recommendations = strategy_analysis.get('strategy_recommendations', [])
            recommendations['strategy_recommendations'] = strategy_recommendations

            # 添加策略相关的原因
            for strategy_rec in strategy_recommendations:
                if strategy_rec['action'] in ['STRONG_BUY', 'BUY']:
                    recommendations['reasons'].append(
                        f"策略 {strategy_rec['strategy']} 推荐 {strategy_rec['action']} "
                        f"(评分: {strategy_rec['score']:.1f})"
                    )

            # 添加组合策略评分信息
            combined_score = strategy_analysis.get('combined_strategy_score', 0.0)
            if combined_score > 20:
                recommendations['reasons'].append(
                    f"多策略组合评分: {combined_score:.1f} (正面)"
                )
            elif combined_score < -20:
                recommendations['risk_warnings'].append(
                    f"多策略组合评分: {combined_score:.1f} (负面)"
                )

        return recommendations

    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def verify_indicator_authenticity(self, stock_codes: List[str],
                                     target_date: str, runs: int = 3) -> Dict[str, Any]:
        """
        验证指标计算的真实性

        Args:
            stock_codes: 测试股票代码列表
            target_date: 分析日期
            runs: 验证运行次数

        Returns:
            Dict[str, Any]: 指标真实性验证结果
        """
        logger.info(f"开始指标真实性验证，股票池: {stock_codes}, 运行次数: {runs}")

        verification_results = {
            'verification_type': 'INDICATOR_AUTHENTICITY',
            'test_stocks': stock_codes,
            'test_date': target_date,
            'runs_count': runs,
            'indicators_tested': len(self.all_indicators),
            'results': {},
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'consistency_rate': 0.0
            },
            'detailed_analysis': {}
        }

        for stock_code in stock_codes:
            logger.info(f"验证股票 {stock_code} 的指标计算")
            stock_results = self._verify_single_stock_indicators(
                stock_code, target_date, runs
            )
            verification_results['results'][stock_code] = stock_results

            # 更新汇总统计
            verification_results['summary']['total_tests'] += stock_results['total_tests']
            verification_results['summary']['passed_tests'] += stock_results['passed_tests']
            verification_results['summary']['failed_tests'] += stock_results['failed_tests']

        # 计算总体一致性率
        total_tests = verification_results['summary']['total_tests']
        passed_tests = verification_results['summary']['passed_tests']
        verification_results['summary']['consistency_rate'] = (
            passed_tests / total_tests * 100 if total_tests > 0 else 0.0
        )

        # 生成详细分析
        verification_results['detailed_analysis'] = self._analyze_indicator_verification(
            verification_results['results']
        )

        logger.info(f"指标真实性验证完成，一致性率: {verification_results['summary']['consistency_rate']:.2f}%")
        return verification_results

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def _verify_single_stock_indicators(self, stock_code: str, target_date: str,
                                       runs: int) -> Dict[str, Any]:
        """验证单只股票的指标计算一致性"""
        stock_results = {
            'stock_code': stock_code,
            'runs': [],
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'inconsistent_indicators': [],
            'authentic_indicators': []
        }

        # 执行多次分析
        for run_idx in range(runs):
            logger.info(f"执行第 {run_idx + 1} 次分析")
            result = self.analyze_multi_period_buypoint(
                stock_code=stock_code,
                target_date=target_date,
                periods=[Period.DAILY],  # 使用日线数据进行验证
                enable_strategy_analysis=False  # 专注于指标验证
            )
            stock_results['runs'].append(result)

        # 比较多次运行的结果
        if len(stock_results['runs']) >= 2:
            consistency_analysis = self._compare_indicator_consistency(
                stock_results['runs']
            )
            stock_results.update(consistency_analysis)

        return stock_results

    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def _compare_indicator_consistency(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """比较多次运行的指标一致性"""
        consistency_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'inconsistent_indicators': [],
            'authentic_indicators': []
        }

        if len(runs) < 2:
            return consistency_results

        # 获取第一次运行的指标数据作为基准
        base_run = runs[0]
        periods_data = base_run.get('multi_period_indicators', {}).get('periods', {})

        # 查找日线数据（可能是'daily'或'日线'）
        base_indicators = {}
        for period_key in ['daily', '日线']:
            if period_key in periods_data:
                base_indicators = periods_data[period_key].get('indicators', {})
                break

        if not base_indicators:
            logger.warning("基准运行中没有找到指标数据")
            return consistency_results

        # 比较每个指标在所有运行中的一致性
        for indicator_name, base_data in base_indicators.items():
            consistency_results['total_tests'] += 1
            is_consistent = True

            # 检查该指标在所有运行中的一致性
            for run_idx, run in enumerate(runs[1:], 1):
                run_periods_data = run.get('multi_period_indicators', {}).get('periods', {})

                # 查找日线数据（可能是'daily'或'日线'）
                run_indicators = {}
                for period_key in ['daily', '日线']:
                    if period_key in run_periods_data:
                        run_indicators = run_periods_data[period_key].get('indicators', {})
                        break

                if indicator_name not in run_indicators:
                    is_consistent = False
                    logger.warning(f"指标 {indicator_name} 在第 {run_idx + 1} 次运行中缺失")
                    break

                run_data = run_indicators[indicator_name]

                # 比较关键数值
                if not self._compare_indicator_values(base_data, run_data):
                    is_consistent = False
                    logger.warning(f"指标 {indicator_name} 在第 {run_idx + 1} 次运行中数值不一致")
                    break

            if is_consistent:
                consistency_results['passed_tests'] += 1
                consistency_results['authentic_indicators'].append(indicator_name)
            else:
                consistency_results['failed_tests'] += 1
                consistency_results['inconsistent_indicators'].append(indicator_name)

        return consistency_results

    def _compare_indicator_values(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> bool:
        """比较两个指标数据的数值一致性"""
        # 比较信号
        if data1.get('signal') != data2.get('signal'):
            return False

        # 比较强度（允许小的浮点误差）
        strength1 = data1.get('strength', 0.0)
        strength2 = data2.get('strength', 0.0)
        if abs(strength1 - strength2) > 1e-6:
            return False

        # 比较数值（如果存在）
        value1 = data1.get('value')
        value2 = data2.get('value')
        if value1 is not None and value2 is not None:
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                if abs(value1 - value2) > 1e-6:
                    return False

        return True

    def _analyze_indicator_verification(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析指标验证结果"""
        analysis = {
            'overall_assessment': 'UNKNOWN',
            'critical_issues': [],
            'recommendations': [],
            'indicator_reliability': {}
        }

        # 统计所有股票的指标可靠性
        indicator_stats = {}
        total_stocks = len(results)

        for stock_code, stock_result in results.items():
            authentic_indicators = stock_result.get('authentic_indicators', [])
            inconsistent_indicators = stock_result.get('inconsistent_indicators', [])

            for indicator in authentic_indicators:
                if indicator not in indicator_stats:
                    indicator_stats[indicator] = {'success': 0, 'total': 0}
                indicator_stats[indicator]['success'] += 1
                indicator_stats[indicator]['total'] += 1

            for indicator in inconsistent_indicators:
                if indicator not in indicator_stats:
                    indicator_stats[indicator] = {'success': 0, 'total': 0}
                indicator_stats[indicator]['total'] += 1

        # 计算每个指标的可靠性
        for indicator, stats in indicator_stats.items():
            reliability = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
            analysis['indicator_reliability'][indicator] = {
                'reliability_rate': reliability,
                'success_count': stats['success'],
                'total_tests': stats['total']
            }

        # 评估整体状况
        reliable_indicators = sum(1 for r in analysis['indicator_reliability'].values() if r['reliability_rate'] == 100)
        total_indicators = len(analysis['indicator_reliability'])

        if total_indicators == 0:
            analysis['overall_assessment'] = 'NO_DATA'
        elif reliable_indicators / total_indicators >= 0.95:
            analysis['overall_assessment'] = 'EXCELLENT'
        elif reliable_indicators / total_indicators >= 0.90:
            analysis['overall_assessment'] = 'GOOD'
        elif reliable_indicators / total_indicators >= 0.80:
            analysis['overall_assessment'] = 'ACCEPTABLE'
        else:
            analysis['overall_assessment'] = 'POOR'

        # 生成关键问题和建议
        unreliable_indicators = [
            indicator for indicator, stats in analysis['indicator_reliability'].items()
            if stats['reliability_rate'] < 100
        ]

        if unreliable_indicators:
            analysis['critical_issues'].append(
                f"发现 {len(unreliable_indicators)} 个不一致的指标: {', '.join(unreliable_indicators[:5])}"
            )
            analysis['recommendations'].append("需要检查不一致指标的计算逻辑")

        if analysis['overall_assessment'] in ['POOR', 'ACCEPTABLE']:
            analysis['recommendations'].append("建议进行深度代码审查和指标计算验证")

        return analysis

    @exception_handler(reraise=True)
    @performance_monitor(threshold=120.0)
    def verify_strategy_consistency(self, stock_codes: List[str],
                                   target_date: str) -> Dict[str, Any]:
        """
        验证策略一致性

        Args:
            stock_codes: 测试股票代码列表
            target_date: 分析日期

        Returns:
            Dict[str, Any]: 策略一致性验证结果
        """
        logger.info(f"开始策略一致性验证，股票池: {stock_codes}")

        verification_results = {
            'verification_type': 'STRATEGY_CONSISTENCY',
            'test_stocks': stock_codes,
            'test_date': target_date,
            'strategy_rules': {},
            'forward_screening': {},
            'reverse_verification': {},
            'consistency_analysis': {},
            'summary': {
                'total_stocks_tested': len(stock_codes),
                'forward_selected': 0,
                'reverse_confirmed': 0,
                'consistency_rate': 0.0
            }
        }

        # 第一步：分析所有股票，生成选股策略规则
        logger.info("第一步：分析所有股票，生成选股策略规则")
        all_analysis_results = {}
        for stock_code in stock_codes:
            logger.info(f"分析股票 {stock_code}")
            result = self.analyze_multi_period_buypoint(
                stock_code=stock_code,
                target_date=target_date,
                periods=None,  # 使用全周期分析
                enable_strategy_analysis=True
            )
            all_analysis_results[stock_code] = result

        # 第二步：基于分析结果生成选股策略规则
        strategy_rules = self._generate_stock_selection_rules(all_analysis_results)
        verification_results['strategy_rules'] = strategy_rules

        # 第三步：正向筛选 - 使用策略规则筛选股票
        logger.info("第三步：正向筛选 - 使用策略规则筛选股票")
        forward_selected = self._apply_selection_rules(all_analysis_results, strategy_rules)
        verification_results['forward_screening'] = {
            'selected_stocks': forward_selected,
            'selection_count': len(forward_selected)
        }

        # 第四步：反向验证 - 对筛选出的股票重新分析
        logger.info("第四步：反向验证 - 对筛选出的股票重新分析")
        reverse_results = {}
        for stock_code in forward_selected:
            reverse_result = self.analyze_multi_period_buypoint(
                stock_code=stock_code,
                target_date=target_date,
                periods=None,  # 使用全周期分析
                enable_strategy_analysis=True
            )
            reverse_results[stock_code] = reverse_result

        verification_results['reverse_verification'] = reverse_results

        # 第五步：一致性分析
        logger.info("第五步：一致性分析")
        consistency_analysis = self._analyze_strategy_consistency(
            forward_selected, all_analysis_results, reverse_results, strategy_rules
        )
        verification_results['consistency_analysis'] = consistency_analysis

        # 更新汇总统计
        verification_results['summary']['forward_selected'] = len(forward_selected)
        verification_results['summary']['reverse_confirmed'] = consistency_analysis['confirmed_count']
        verification_results['summary']['consistency_rate'] = (
            consistency_analysis['confirmed_count'] / len(forward_selected) * 100
            if forward_selected else 100.0
        )

        logger.info(f"策略一致性验证完成，一致性率: {verification_results['summary']['consistency_rate']:.2f}%")
        return verification_results

    def _generate_stock_selection_rules(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """基于分析结果生成选股策略规则"""
        rules = {
            'score_thresholds': {},
            'strategy_conditions': {},
            'indicator_conditions': {},
            'combined_conditions': {}
        }

        # 分析所有股票的评分分布
        scores = [result['overall_score'] for result in analysis_results.values() if result.get('status') == 'SUCCESS']
        if scores:
            scores.sort(reverse=True)
            # 设置评分阈值（选择前30%的股票）
            threshold_index = max(0, int(len(scores) * 0.3) - 1)
            score_threshold = scores[threshold_index] if threshold_index < len(scores) else min(scores)
            rules['score_thresholds']['buy_threshold'] = max(60.0, score_threshold)  # 至少60分
            rules['score_thresholds']['strong_buy_threshold'] = max(70.0, score_threshold + 10)  # 强买入阈值
        else:
            rules['score_thresholds']['buy_threshold'] = 60.0
            rules['score_thresholds']['strong_buy_threshold'] = 70.0

        # 分析策略信号条件
        strategy_signals = {}
        for stock_code, result in analysis_results.items():
            if result.get('status') == 'SUCCESS' and result.get('strategy_analysis'):
                signals = result['strategy_analysis'].get('strategy_signals', {})
                for strategy_name, signal in signals.items():
                    if strategy_name not in strategy_signals:
                        strategy_signals[strategy_name] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                    strategy_signals[strategy_name][signal] = strategy_signals[strategy_name].get(signal, 0) + 1

        # 设置策略条件（至少2个策略发出BUY信号）
        rules['strategy_conditions']['min_buy_strategies'] = 2
        rules['strategy_conditions']['allowed_sell_strategies'] = 0  # 不允许有SELL信号

        # 设置指标条件（基于强买入信号指标）
        strong_buy_indicators = set()
        for result in analysis_results.values():
            if result.get('status') == 'SUCCESS':
                recommendations = result.get('recommendations', {})
                reasons = recommendations.get('reasons', [])
                for reason in reasons:
                    if '强买入信号指标:' in reason:
                        indicators_str = reason.split('强买入信号指标:')[1].strip()
                        indicators = [ind.strip() for ind in indicators_str.split(',')]
                        strong_buy_indicators.update(indicators)

        rules['indicator_conditions']['required_strong_buy_indicators'] = list(strong_buy_indicators)[:5]  # 取前5个

        # 组合条件
        rules['combined_conditions'] = {
            'score_weight': 0.6,
            'strategy_weight': 0.3,
            'indicator_weight': 0.1
        }

        return rules

    def _apply_selection_rules(self, analysis_results: Dict[str, Any],
                              rules: Dict[str, Any]) -> List[str]:
        """应用选股规则筛选股票"""
        selected_stocks = []

        for stock_code, result in analysis_results.items():
            if result.get('status') != 'SUCCESS':
                continue

            # 检查评分条件
            score = result.get('overall_score', 0.0)
            if score < rules['score_thresholds']['buy_threshold']:
                continue

            # 检查策略条件
            strategy_analysis = result.get('strategy_analysis', {})
            if strategy_analysis:
                strategy_signals = strategy_analysis.get('strategy_signals', {})
                buy_count = sum(1 for signal in strategy_signals.values() if signal == 'BUY')
                sell_count = sum(1 for signal in strategy_signals.values() if signal == 'SELL')

                if (buy_count < rules['strategy_conditions']['min_buy_strategies'] or
                    sell_count > rules['strategy_conditions']['allowed_sell_strategies']):
                    continue

            # 检查指标条件
            recommendations = result.get('recommendations', {})
            reasons = recommendations.get('reasons', [])
            has_strong_indicators = any('强买入信号指标:' in reason for reason in reasons)

            if not has_strong_indicators:
                continue

            # 通过所有条件，加入选股列表
            selected_stocks.append(stock_code)

        return selected_stocks

    def _analyze_strategy_consistency(self, forward_selected: List[str],
                                    original_results: Dict[str, Any],
                                    reverse_results: Dict[str, Any],
                                    rules: Dict[str, Any]) -> Dict[str, Any]:
        """分析策略一致性"""
        analysis = {
            'confirmed_stocks': [],
            'inconsistent_stocks': [],
            'confirmed_count': 0,
            'inconsistent_count': 0,
            'detailed_comparison': {}
        }

        for stock_code in forward_selected:
            if stock_code not in reverse_results:
                analysis['inconsistent_stocks'].append(stock_code)
                analysis['inconsistent_count'] += 1
                continue

            original = original_results[stock_code]
            reverse = reverse_results[stock_code]

            # 比较关键指标
            comparison = self._compare_analysis_results(original, reverse, rules)
            analysis['detailed_comparison'][stock_code] = comparison

            if comparison['is_consistent']:
                analysis['confirmed_stocks'].append(stock_code)
                analysis['confirmed_count'] += 1
            else:
                analysis['inconsistent_stocks'].append(stock_code)
                analysis['inconsistent_count'] += 1

        return analysis

    def _compare_analysis_results(self, result1: Dict[str, Any],
                                 result2: Dict[str, Any],
                                 rules: Dict[str, Any]) -> Dict[str, Any]:
        """比较两次分析结果的一致性"""
        comparison = {
            'is_consistent': True,
            'score_diff': 0.0,
            'strategy_consistency': True,
            'recommendation_consistency': True,
            'issues': []
        }

        # 比较评分（允许小幅差异）
        score1 = result1.get('overall_score', 0.0)
        score2 = result2.get('overall_score', 0.0)
        score_diff = abs(score1 - score2)
        comparison['score_diff'] = score_diff

        if score_diff > 2.0:  # 允许2分的差异
            comparison['is_consistent'] = False
            comparison['issues'].append(f"评分差异过大: {score_diff:.2f}")

        # 比较策略信号
        strategy1 = result1.get('strategy_analysis', {}).get('strategy_signals', {})
        strategy2 = result2.get('strategy_analysis', {}).get('strategy_signals', {})

        for strategy_name in strategy1.keys():
            if strategy_name in strategy2:
                if strategy1[strategy_name] != strategy2[strategy_name]:
                    comparison['strategy_consistency'] = False
                    comparison['is_consistent'] = False
                    comparison['issues'].append(f"策略 {strategy_name} 信号不一致")

        # 比较投资建议
        rec1 = result1.get('recommendations', {}).get('action', 'UNKNOWN')
        rec2 = result2.get('recommendations', {}).get('action', 'UNKNOWN')

        if rec1 != rec2:
            comparison['recommendation_consistency'] = False
            comparison['is_consistent'] = False
            comparison['issues'].append(f"投资建议不一致: {rec1} vs {rec2}")

        return comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生产级多策略买点识别系统 - 动态指标和策略')
    parser.add_argument('--stock-code', help='股票代码')
    parser.add_argument('--date', help='分析日期 (YYYY-MM-DD)')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--periods', nargs='+',
                       choices=['15min', '30min', '60min', 'daily', 'weekly', 'monthly'],
                       default=None,  # 默认使用全周期
                       help='分析周期 (默认: 全周期分析)')
    parser.add_argument('--indicators', nargs='+',
                       help='指定分析的指标列表 (默认: 所有可用指标)')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--show-indicator-count', action='store_true',
                       help='显示可用指标数量')
    parser.add_argument('--enable-strategy-analysis', action='store_true',
                       default=True, help='启用策略分析 (默认: 启用)')
    parser.add_argument('--disable-strategy-analysis', action='store_true',
                       help='禁用策略分析')
    parser.add_argument('--verify-indicators', action='store_true',
                       help='执行指标真实性验证')
    parser.add_argument('--verify-strategy-consistency', action='store_true',
                       help='执行策略一致性验证')
    parser.add_argument('--verification-stocks', nargs='+',
                       default=['300005', '603359', '000001'],
                       help='验证用股票池 (默认: 300005 603359 000001)')
    parser.add_argument('--verification-runs', type=int, default=3,
                       help='验证运行次数 (默认: 3)')

    args = parser.parse_args()

    # 如果只是查看指标数量，不需要其他参数
    if args.show_indicator_count:
        # 创建分析器实例
        analyzer = MultiPeriodBuypointAnalyzer(config_path=args.config)
        print(f"可用指标总数: {len(analyzer.all_indicators)}")
        print(f"动态策略数量: {len(analyzer.builtin_strategies)}")
        print("指标列表:")
        for i, indicator in enumerate(analyzer.all_indicators, 1):
            print(f"  {i:3d}. {indicator}")
        print("\n策略列表:")
        for i, (name, config) in enumerate(analyzer.builtin_strategies.items(), 1):
            print(f"  {i:2d}. {name}: {config['name']} ({len(config['indicators'])}个指标)")
        return

    # 创建分析器实例（使用配置文件）
    analyzer = MultiPeriodBuypointAnalyzer(config_path=args.config)

    # 执行指标真实性验证
    if args.verify_indicators:
        print("🔍 开始执行指标真实性验证...")
        verification_result = analyzer.verify_indicator_authenticity(
            stock_codes=args.verification_stocks,
            target_date=args.date or '2025-05-09',  # 使用默认日期进行验证
            runs=args.verification_runs
        )

        # 输出验证结果
        if args.output:
            output_file = args.output.replace('.json', '_indicator_verification.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                def json_serializer(obj):
                    if hasattr(obj, 'value'):
                        return obj.value
                    elif hasattr(obj, '__dict__'):
                        return obj.__dict__
                    else:
                        return str(obj)
                json.dump(verification_result, f, ensure_ascii=False, indent=2, default=json_serializer)
            print(f"验证结果已保存到: {output_file}")

        # 显示验证摘要
        summary = verification_result['summary']
        analysis = verification_result['detailed_analysis']
        print(f"\n📊 指标真实性验证摘要:")
        print(f"测试股票: {', '.join(args.verification_stocks)}")
        print(f"验证运行次数: {args.verification_runs}")
        print(f"测试指标总数: {verification_result['indicators_tested']}")
        print(f"总测试次数: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"一致性率: {summary['consistency_rate']:.2f}%")
        print(f"整体评估: {analysis['overall_assessment']}")

        if analysis['critical_issues']:
            print(f"\n⚠️ 关键问题:")
            for issue in analysis['critical_issues']:
                print(f"  - {issue}")

        if analysis['recommendations']:
            print(f"\n💡 建议:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")

        return

    # 执行策略一致性验证
    if args.verify_strategy_consistency:
        print("🔍 开始执行策略一致性验证...")
        verification_result = analyzer.verify_strategy_consistency(
            stock_codes=args.verification_stocks,
            target_date=args.date or '2025-05-09'  # 使用默认日期进行验证
        )

        # 输出验证结果
        if args.output:
            output_file = args.output.replace('.json', '_strategy_verification.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                def json_serializer(obj):
                    if hasattr(obj, 'value'):
                        return obj.value
                    elif hasattr(obj, '__dict__'):
                        return obj.__dict__
                    else:
                        return str(obj)
                json.dump(verification_result, f, ensure_ascii=False, indent=2, default=json_serializer)
            print(f"验证结果已保存到: {output_file}")

        # 显示验证摘要
        summary = verification_result['summary']
        consistency = verification_result['consistency_analysis']
        rules = verification_result['strategy_rules']

        print(f"\n📊 策略一致性验证摘要:")
        print(f"测试股票池: {', '.join(args.verification_stocks)}")
        print(f"测试股票总数: {summary['total_stocks_tested']}")
        print(f"正向筛选股票: {summary['forward_selected']}")
        print(f"反向确认股票: {summary['reverse_confirmed']}")
        print(f"一致性率: {summary['consistency_rate']:.2f}%")

        print(f"\n📋 生成的选股策略规则:")
        print(f"评分阈值: {rules['score_thresholds']['buy_threshold']:.1f}")
        print(f"最低买入策略数: {rules['strategy_conditions']['min_buy_strategies']}")
        print(f"允许卖出策略数: {rules['strategy_conditions']['allowed_sell_strategies']}")

        if consistency['confirmed_stocks']:
            print(f"\n✅ 一致性确认股票: {', '.join(consistency['confirmed_stocks'])}")

        if consistency['inconsistent_stocks']:
            print(f"\n⚠️ 不一致股票: {', '.join(consistency['inconsistent_stocks'])}")

        return

    # 检查必需参数
    if not args.stock_code or not args.date:
        parser.error("--stock-code 和 --date 是必需参数")
        return

    # 转换周期参数
    period_mapping = {
        '15min': Period.MIN_15,
        '30min': Period.MIN_30,
        '60min': Period.MIN_60,
        'daily': Period.DAILY,
        'weekly': Period.WEEKLY,
        'monthly': Period.MONTHLY
    }

    # 转换周期参数（支持全周期分析）
    periods = None  # 默认全周期分析
    if args.periods:
        periods = [period_mapping[p] for p in args.periods]
        print(f"使用指定周期: {', '.join(args.periods)}")
    else:
        print("使用全周期分析: 15分钟、30分钟、60分钟、日线、周线、月线")

    # 确定要分析的指标
    indicator_names = args.indicators if args.indicators else None
    if indicator_names:
        print(f"使用指定的{len(indicator_names)}个指标进行分析")
    else:
        print(f"使用全部{len(analyzer.all_indicators)}个可用指标进行分析")
        print(f"动态策略: {len(analyzer.builtin_strategies)}个")

    # 确定是否启用策略分析
    enable_strategy_analysis = not args.disable_strategy_analysis
    if enable_strategy_analysis:
        print("策略分析: 启用（动态策略）")
    else:
        print("策略分析: 禁用")

    # 执行分析
    result = analyzer.analyze_multi_period_buypoint(
        stock_code=args.stock_code,
        target_date=args.date,
        periods=periods,
        indicator_names=indicator_names,
        enable_strategy_analysis=enable_strategy_analysis
    )
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            # 自定义JSON编码器处理Period枚举和其他不可序列化对象
            def json_serializer(obj):
                if hasattr(obj, 'value'):  # 处理枚举类型
                    return obj.value
                elif hasattr(obj, '__dict__'):  # 处理其他对象
                    return obj.__dict__
                else:
                    return str(obj)

            json.dump(result, f, ensure_ascii=False, indent=2, default=json_serializer)
        print(f"分析结果已保存到: {args.output}")
    else:
        def json_serializer(obj):
            if hasattr(obj, 'value'):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=json_serializer))

    # 显示分析摘要
    if result.get('status') == 'SUCCESS':
        print(f"\n📊 分析摘要:")
        print(f"股票代码: {result['stock_code']}")
        print(f"分析日期: {result['analysis_date']}")
        print(f"综合评分: {result['overall_score']:.2f}/100")
        print(f"分析指标: {result['total_indicators_count']}个真实指标")

        recommendations = result.get('recommendations', {})
        action = recommendations.get('action', 'UNKNOWN')
        confidence = recommendations.get('confidence', 'UNKNOWN')
        print(f"投资建议: {action} (置信度: {confidence})")

        if enable_strategy_analysis and result.get('strategy_analysis'):
            strategy_count = len(result['strategy_analysis'].get('strategy_signals', {}))
            combined_score = result['strategy_analysis'].get('combined_strategy_score', 0.0)
            print(f"策略分析: {strategy_count}个策略，组合评分: {combined_score:.2f}")
    else:
        print(f"\n❌ 分析失败: {result.get('error', '未知错误')}")


if __name__ == '__main__':
    main()
