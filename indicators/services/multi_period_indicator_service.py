#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多周期指标计算服务
支持在不同周期上计算技术指标，如15分钟KDJ金叉和日线KDJ金叉
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

from utils.container import container
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from indicators.base_indicator import BaseIndicator
from db.services.multi_period_data_service import MultiPeriodDataService, Period
from indicators.complete_indicator_registry import CompleteIndicatorRegistry, get_indicator
from indicators.signal_method_adapter import get_unified_indicator_signal
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class MultiPeriodIndicatorService(BaseIndicator):
    """
    多周期指标计算服务
    
    功能：
    - 在不同周期上计算相同指标
    - 比较不同周期的指标信号
    - 提供多周期信号聚合分析
    - 支持周期间的信号验证
    """
    
    def __init__(self, **kwargs):
        """初始化多周期指标服务"""
        super().__init__(name="MultiPeriodIndicatorService", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        self.data_service = MultiPeriodDataService()
        self.indicator_registry = CompleteIndicatorRegistry()
        self.indicator_registry.register_all_indicators()
        
        # 周期权重配置（用于信号聚合）
        self.period_weights = {
            Period.MIN_15: 0.1,   # 15分钟线权重较低
            Period.MIN_30: 0.15,  # 30分钟线  # TODO: 将魔法数字提取到配置中
            Period.MIN_60: 0.2,   # 60分钟线
            Period.DAILY: 0.35,   # 日线权重最高  # TODO: 将魔法数字提取到配置中
            Period.WEEKLY: 0.15,  # 周线  # TODO: 将魔法数字提取到配置中
            Period.MONTHLY: 0.05  # 月线权重较低  # TODO: 将魔法数字提取到配置中
        }
        
        logger.info("多周期指标计算服务初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def calculate_multi_period_indicators(self,
                                        stock_code: str,
                                        target_date: str,
                                        indicator_names: Optional[List[str]] = None,
                                        periods: Optional[List[Period]] = None) -> Dict[str, Any]:
        """
        计算多周期指标
        
        Args:
            stock_code: 股票代码
            target_date: 目标分析日期
            indicator_names: 指标名称列表，None表示计算所有指标
            periods: 周期列表，None表示计算所有周期
            
        Returns:
            Dict[str, Any]: 多周期指标结果
        """
        if periods is None:
            periods = [Period.MIN_15, Period.MIN_30, Period.MIN_60, Period.DAILY, Period.WEEKLY]
        
        if indicator_names is None:
            # 使用全部128个指标进行多周期分析
            all_indicators = self.indicator_registry.get_all_indicators()
            indicator_names = list(all_indicators.keys())
            logger.info(f"使用全部{len(indicator_names)}个指标进行多周期分析")
        
        # 获取多周期数据
        logger.info(f"开始获取{stock_code}的多周期数据")
        multi_period_data = self.data_service.get_stock_multi_period_data(
            stock_code=stock_code,
            target_date=target_date,
            periods=periods
        )
        
        # 计算各周期的指标
        results = {
            'stock_code': stock_code,
            'target_date': target_date,
            'periods': {},
            'cross_period_analysis': {},
            'aggregated_signals': {}
        }
        
        for period in periods:
            period_data = multi_period_data.get(period, pd.DataFrame())
            if not period_data.empty:
                period_results = self._calculate_period_indicators(
                    period_data, indicator_names, period
                )
                results['periods'][period.value] = period_results
                logger.info(f"完成{period.value}周期指标计算，{len(period_results)}个指标")
            else:
                logger.warning(f"{period.value}周期数据为空，跳过计算")
                results['periods'][period.value] = {}
        
        # 进行跨周期分析
        results['cross_period_analysis'] = self._analyze_cross_period_signals(
            results['periods'], indicator_names
        )
        
        # 生成聚合信号
        results['aggregated_signals'] = self._generate_aggregated_signals(
            results['periods'], indicator_names
        )
        
        return results
    
    def _calculate_period_indicators(self,
                                   data: pd.DataFrame,
                                   indicator_names: List[str],
                                   period: Period) -> Dict[str, Any]:
        """
        计算单个周期的指标
        
        Args:
            data: 股票数据
            indicator_names: 指标名称列表
            period: 周期
            
        Returns:
            Dict[str, Any]: 周期指标结果
        """
        period_results = {
            'period': period.value,
            'data_points': len(data),
            'indicators': {}
        }
        
        for indicator_name in indicator_names:
            try:
                indicator = get_indicator(indicator_name)
                if indicator:
                    # 计算指标
                    calc_result = indicator.calculate(data)
                    
                    # 获取统一信号
                    signal_result = get_unified_indicator_signal(indicator, data, indicator_name)
                    
                    period_results['indicators'][indicator_name] = {
                        'signal': signal_result.get('signal', 'UNKNOWN'),
                        'strength': signal_result.get('strength', 0.0),
                        'value': signal_result.get('value', None),
                        'method_used': signal_result.get('method_used', 'unknown'),
                        'period': period.value,
                        'calculation_success': True
                    }
                    
                    # 特殊处理：检测金叉死叉等形态
                    if indicator_name in ['KDJ', 'MACD', 'STOCH']:
                        cross_analysis = self._detect_cross_patterns(calc_result, indicator_name)
                        period_results['indicators'][indicator_name].update(cross_analysis)
                
                else:
                    period_results['indicators'][indicator_name] = {
                        'signal': 'NOT_AVAILABLE',
                        'error': f'指标{indicator_name}未找到',
                        'period': period.value,
                        'calculation_success': False
                    }
                    
            except Exception as e:
                logger.warning(f"计算{period.value}周期{indicator_name}指标失败: {e}")
                period_results['indicators'][indicator_name] = {
                    'signal': 'ERROR',
                    'error': str(e),
                    'period': period.value,
                    'calculation_success': False
                }
        
        return period_results
    
    def _detect_cross_patterns(self, calc_result: Any, indicator_name: str) -> Dict[str, Any]:
        """
        检测金叉死叉等交叉形态
        
        Args:
            calc_result: 指标计算结果
            indicator_name: 指标名称
            
        Returns:
            Dict[str, Any]: 交叉形态分析
        """
        cross_info = {
            'has_golden_cross': False,
            'has_death_cross': False,
            'cross_type': 'NONE',
            'cross_strength': 0.0
        }
        
        try:
            if not hasattr(calc_result, 'columns') or len(calc_result) < 2:
                return cross_info
            
            latest_row = calc_result.iloc[-1]
            prev_row = calc_result.iloc[-2]
            
            if indicator_name == 'KDJ':
                # KDJ金叉死叉检测
                if 'K' in calc_result.columns and 'D' in calc_result.columns:
                    k_curr, d_curr = latest_row['K'], latest_row['D']
                    k_prev, d_prev = prev_row['K'], prev_row['D']
                    
                    if k_prev <= d_prev and k_curr > d_curr:
                        cross_info.update({
                            'has_golden_cross': True,
                            'cross_type': 'GOLDEN_CROSS',
                            'cross_strength': min(abs(k_curr - d_curr) / 10, 1.0)
                        })
                    elif k_prev >= d_prev and k_curr < d_curr:
                        cross_info.update({
                            'has_death_cross': True,
                            'cross_type': 'DEATH_CROSS',
                            'cross_strength': min(abs(k_curr - d_curr) / 10, 1.0)
                        })
            
            elif indicator_name == 'MACD':
                # MACD金叉死叉检测
                if 'DIF' in calc_result.columns and 'DEA' in calc_result.columns:
                    dif_curr, dea_curr = latest_row['DIF'], latest_row['DEA']
                    dif_prev, dea_prev = prev_row['DIF'], prev_row['DEA']
                    
                    if dif_prev <= dea_prev and dif_curr > dea_curr:
                        cross_info.update({
                            'has_golden_cross': True,
                            'cross_type': 'GOLDEN_CROSS',
                            'cross_strength': min(abs(dif_curr - dea_curr) * 100, 1.0)
                        })
                    elif dif_prev >= dea_prev and dif_curr < dea_curr:
                        cross_info.update({
                            'has_death_cross': True,
                            'cross_type': 'DEATH_CROSS',
                            'cross_strength': min(abs(dif_curr - dea_curr) * 100, 1.0)
                        })
            
        except Exception as e:
            logger.debug(f"检测{indicator_name}交叉形态失败: {e}")
        
        return cross_info
    
    def _analyze_cross_period_signals(self,
                                    periods_data: Dict[str, Dict],
                                    indicator_names: List[str]) -> Dict[str, Any]:
        """
        分析跨周期信号一致性
        
        Args:
            periods_data: 各周期数据
            indicator_names: 指标名称列表
            
        Returns:
            Dict[str, Any]: 跨周期分析结果
        """
        cross_analysis = {
            'signal_consistency': {},
            'period_divergence': {},
            'trend_confirmation': {}
        }
        
        for indicator_name in indicator_names:
            # 收集各周期的信号
            period_signals = {}
            for period_name, period_data in periods_data.items():
                if indicator_name in period_data.get('indicators', {}):
                    signal_info = period_data['indicators'][indicator_name]
                    period_signals[period_name] = signal_info.get('signal', 'UNKNOWN')
            
            if len(period_signals) >= 2:
                # 分析信号一致性
                unique_signals = set(period_signals.values())
                consistency_score = 1.0 - (len(unique_signals) - 1) / len(period_signals)
                
                cross_analysis['signal_consistency'][indicator_name] = {
                    'score': consistency_score,
                    'signals': period_signals,
                    'is_consistent': len(unique_signals) <= 2
                }
                
                # 检测周期背离
                short_term_signals = [period_signals.get(p, 'UNKNOWN') 
                                    for p in ['15分钟', '30分钟', '60分钟']]
                long_term_signals = [period_signals.get(p, 'UNKNOWN') 
                                   for p in ['日线', '周线']]
                
                short_bullish = sum(1 for s in short_term_signals if s == 'BUY')
                long_bullish = sum(1 for s in long_term_signals if s == 'BUY')
                
                has_divergence = (short_bullish > 0 and long_bullish == 0) or \
                               (short_bullish == 0 and long_bullish > 0)
                
                cross_analysis['period_divergence'][indicator_name] = {
                    'has_divergence': has_divergence,
                    'short_term_bias': 'BULLISH' if short_bullish > 0 else 'BEARISH',
                    'long_term_bias': 'BULLISH' if long_bullish > 0 else 'BEARISH'
                }
        
        return cross_analysis
    
    def _generate_aggregated_signals(self,
                                   periods_data: Dict[str, Dict],
                                   indicator_names: List[str]) -> Dict[str, Any]:
        """
        生成聚合信号
        
        Args:
            periods_data: 各周期数据
            indicator_names: 指标名称列表
            
        Returns:
            Dict[str, Any]: 聚合信号结果
        """
        aggregated = {}
        
        for indicator_name in indicator_names:
            weighted_score = 0.0
            total_weight = 0.0
            signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for period_name, period_data in periods_data.items():
                if indicator_name in period_data.get('indicators', {}):
                    signal_info = period_data['indicators'][indicator_name]
                    signal = signal_info.get('signal', 'UNKNOWN')
                    strength = signal_info.get('strength', 0.0)
                    
                    # 获取周期权重
                    period_enum = None
                    for p in Period:
                        if p.value == period_name:
                            period_enum = p
                            break
                    
                    if period_enum and period_enum in self.period_weights:
                        weight = self.period_weights[period_enum]
                        
                        # 计算加权分数
                        if signal == 'BUY':
                            weighted_score += weight * strength
                            signal_count['BUY'] += 1
                        elif signal == 'SELL':
                            weighted_score -= weight * strength
                            signal_count['SELL'] += 1
                        else:
                            signal_count['HOLD'] += 1
                        
                        total_weight += weight
            
            # 生成最终聚合信号
            if total_weight > 0:
                final_score = weighted_score / total_weight
                
                if final_score > 0.3:  # TODO: 将魔法数字提取到配置中
                    final_signal = 'BUY'
                elif final_score < -0.3:  # TODO: 将魔法数字提取到配置中
                    final_signal = 'SELL'
                else:
                    final_signal = 'HOLD'
                
                aggregated[indicator_name] = {
                    'signal': final_signal,
                    'strength': abs(final_score),
                    'weighted_score': final_score,
                    'signal_distribution': signal_count,
                    'confidence': min(total_weight, 1.0)
                }
        
        return aggregated

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据，包含OHLCV等字段
            
        Returns:
            pd.DataFrame: 包含指标计算结果的数据框
        """
        if not self.validate_data(data):
            raise ValueError("输入数据不符合要求")
        
        # 预处理数据
        processed_data = self.preprocess_data(data)
        
        # TODO: 实现具体的指标计算逻辑
        result = processed_data.copy()
        result[f'{self.name}_value'] = processed_data['close'].rolling(window=self.period).mean()
        
        # 后处理结果
        result = self.postprocess_result(result)
        
        # 保存结果
        self._result = result
        
        return result

    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号
        
        Args:
            data: 包含指标计算结果的数据
            
        Returns:
            Dict[str, Any]: 交易信号信息
        """
        if data.empty:
            return {'signal': 'hold', 'strength': 0.0, 'timestamp': None}
        
        # TODO: 实现具体的信号生成逻辑
        latest_close = data['close'].iloc[-1] if 'close' in data.columns else 0
        
        return {
            'signal': 'hold',
            'strength': 0.0,
            'timestamp': data.index[-1] if not data.empty else None,
            'price': latest_close,
            'indicator': self.name
        }
