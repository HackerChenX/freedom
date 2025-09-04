#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略执行引擎
实现基于回测结果生成的策略进行实际选股
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from utils.logger import get_logger
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from indicators.complete_indicator_registry import get_indicator_registry
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler

logger = get_logger(__name__)


class StrategyExecutionEngine:
    """
    策略执行引擎
    
    根据买点回测生成的策略，从ClickHouse中选出符合条件的股票
    实现双向验证功能
    """
    
    def __init__(self):
        """初始化策略执行引擎"""
        self.data_access = get_service(DataAccessInterface)
        self.indicator_registry = get_indicator_registry()
        
        # 股票池配置
        self.stock_universe = self._get_stock_universe()
        logger.info(f"策略执行引擎初始化完成，股票池包含{len(self.stock_universe)}只股票")
    
    def _get_stock_universe(self) -> List[str]:
        """获取股票池"""
        try:
            # 这里应该从数据库获取所有可交易股票
            # 暂时使用模拟数据
            mock_universe = [
                '000001', '000002', '000858', '000876', '000977',
                '600000', '600036', '600519', '600887', '601318',
                '603359', '603501', '688001', '688036', '688111'
            ]
            return mock_universe
            
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return []
    
    @performance_monitor(threshold_seconds=60.0)
    @exception_handler(reraise=True)
    def execute_strategy(self, strategy: Dict[str, Any], 
                        target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        执行选股策略
        
        Args:
            strategy: 策略配置
            target_date: 目标日期，默认为最新交易日
            
        Returns:
            Dict[str, Any]: 选股结果
        """
        logger.info(f"🎯 执行策略: {strategy.get('name', 'Unknown')}")
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y%m%d')
        
        result = {
            'strategy_id': strategy.get('id', 'unknown'),
            'strategy_name': strategy.get('name', 'Unknown'),
            'execution_date': target_date,
            'selected_stocks': [],
            'analysis_details': {},
            'execution_summary': {}
        }
        
        try:
            # 解析策略条件
            conditions = strategy.get('conditions', {})
            required_patterns = conditions.get('required_patterns', [])
            
            if not required_patterns:
                logger.warning("策略没有定义必要的形态条件")
                return result
            
            # 逐个分析股票池中的股票
            analyzed_stocks = 0
            matched_stocks = []
            
            for stock_code in self.stock_universe:
                try:
                    analyzed_stocks += 1
                    logger.debug(f"分析股票 {stock_code} ({analyzed_stocks}/{len(self.stock_universe)})")
                    
                    # 分析单只股票
                    stock_analysis = self._analyze_single_stock(
                        stock_code, target_date, required_patterns
                    )
                    
                    if stock_analysis['matched']:
                        matched_stocks.append({
                            'stock_code': stock_code,
                            'match_score': stock_analysis['score'],
                            'matched_patterns': stock_analysis['matched_patterns'],
                            'analysis_details': stock_analysis['details']
                        })
                        
                        logger.info(f"✅ {stock_code} 符合策略条件，评分: {stock_analysis['score']:.1f}")
                    
                    # 保存分析详情
                    result['analysis_details'][stock_code] = stock_analysis
                    
                except Exception as e:
                    logger.debug(f"分析股票{stock_code}失败: {e}")
                    continue
            
            # 按评分排序
            matched_stocks.sort(key=lambda x: x['match_score'], reverse=True)
            
            # 生成最终结果
            result['selected_stocks'] = [stock['stock_code'] for stock in matched_stocks]
            result['execution_summary'] = {
                'total_analyzed': analyzed_stocks,
                'total_matched': len(matched_stocks),
                'match_rate': len(matched_stocks) / analyzed_stocks if analyzed_stocks > 0 else 0,
                'top_stocks': matched_stocks[:10],  # 前10只股票
                'avg_score': np.mean([s['match_score'] for s in matched_stocks]) if matched_stocks else 0
            }
            
            logger.info(f"🎉 策略执行完成: {len(matched_stocks)}/{analyzed_stocks} 股票符合条件")
            return result
            
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            result['error'] = str(e)
            return result
    
    def _analyze_single_stock(self, stock_code: str, target_date: str, 
                            required_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析单只股票是否符合策略条件
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            required_patterns: 必要的形态列表
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        analysis = {
            'matched': False,
            'score': 0.0,
            'matched_patterns': [],
            'details': {}
        }
        
        try:
            # 按周期分组所需形态
            patterns_by_period = {}
            for pattern in required_patterns:
                period = pattern.get('period', 'daily')
                if period not in patterns_by_period:
                    patterns_by_period[period] = []
                patterns_by_period[period].append(pattern)
            
            total_score = 0.0
            total_weight = 0.0
            matched_count = 0
            
            # 逐周期检查
            for period, period_patterns in patterns_by_period.items():
                period_result = self._check_period_patterns(
                    stock_code, target_date, period, period_patterns
                )
                
                analysis['details'][period] = period_result
                
                if period_result['matched']:
                    matched_count += 1
                    weight = self._get_period_weight(period)
                    total_score += period_result['score'] * weight
                    total_weight += weight
                    
                    analysis['matched_patterns'].extend(period_result['matched_patterns'])
            
            # 计算最终匹配结果
            required_period_count = len(patterns_by_period)
            match_rate = matched_count / required_period_count if required_period_count > 0 else 0
            
            # 至少要有70%的周期匹配才算成功
            analysis['matched'] = match_rate >= 0.7
            analysis['score'] = total_score / total_weight if total_weight > 0 else 0
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析股票{stock_code}失败: {e}")
            analysis['error'] = str(e)
            return analysis
    
    def _check_period_patterns(self, stock_code: str, target_date: str, 
                             period: str, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查特定周期的形态匹配"""
        result = {
            'matched': False,
            'score': 0.0,
            'matched_patterns': [],
            'pattern_details': {}
        }
        
        try:
            # 获取周期数据
            period_data = self._get_period_data(stock_code, period, target_date)
            if period_data is None or period_data.empty:
                return result
            
            # 按指标分组形态
            patterns_by_indicator = {}
            for pattern in patterns:
                indicator = pattern.get('indicator')
                if indicator not in patterns_by_indicator:
                    patterns_by_indicator[indicator] = []
                patterns_by_indicator[indicator].append(pattern)
            
            matched_indicators = 0
            total_score = 0.0
            
            # 逐指标检查
            for indicator_name, indicator_patterns in patterns_by_indicator.items():
                indicator_result = self._check_indicator_patterns(
                    period_data, indicator_name, indicator_patterns
                )
                
                result['pattern_details'][indicator_name] = indicator_result
                
                if indicator_result['matched']:
                    matched_indicators += 1
                    total_score += indicator_result['score']
                    result['matched_patterns'].extend(indicator_result['matched_patterns'])
            
            # 计算周期匹配结果
            required_indicator_count = len(patterns_by_indicator)
            indicator_match_rate = matched_indicators / required_indicator_count if required_indicator_count > 0 else 0
            
            # 至少要有50%的指标匹配
            result['matched'] = indicator_match_rate >= 0.5
            result['score'] = total_score / matched_indicators if matched_indicators > 0 else 0
            
            return result
            
        except Exception as e:
            logger.debug(f"检查周期{period}形态失败: {e}")
            result['error'] = str(e)
            return result
    
    def _check_indicator_patterns(self, data: pd.DataFrame, indicator_name: str, 
                                patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查指标的形态匹配"""
        result = {
            'matched': False,
            'score': 0.0,
            'matched_patterns': [],
            'calculation_result': None
        }
        
        try:
            # 创建指标实例
            indicator = self.indicator_registry.create_indicator(indicator_name)
            if indicator is None:
                return result
            
            # 计算指标
            indicator_result = indicator.calculate(data)
            if indicator_result is None or indicator_result.empty:
                return result
            
            result['calculation_result'] = indicator_result
            
            # 获取形态
            indicator_patterns = indicator.get_patterns(indicator_result)
            if not indicator_patterns:
                return result
            
            # 检查最新数据点的形态
            latest_index = len(indicator_result) - 1
            matched_pattern_count = 0
            
            for pattern_config in patterns:
                pattern_name = pattern_config.get('pattern')
                if pattern_name in indicator_patterns:
                    pattern_data = indicator_patterns[pattern_name]
                    
                    # 检查最新点是否命中形态
                    if self._is_pattern_matched(pattern_data, latest_index):
                        matched_pattern_count += 1
                        result['matched_patterns'].append({
                            'indicator': indicator_name,
                            'pattern': pattern_name,
                            'value': self._get_pattern_value(pattern_data, latest_index)
                        })
            
            # 计算匹配结果
            required_pattern_count = len(patterns)
            pattern_match_rate = matched_pattern_count / required_pattern_count if required_pattern_count > 0 else 0
            
            result['matched'] = pattern_match_rate >= 0.8  # 80%的形态要匹配
            
            # 计算评分
            if result['matched']:
                score_result = indicator.calculate_score(indicator_result)
                result['score'] = score_result.get('score', 50.0)
            
            return result
            
        except Exception as e:
            logger.debug(f"检查指标{indicator_name}形态失败: {e}")
            result['error'] = str(e)
            return result
    
    def _get_period_data(self, stock_code: str, period: str, target_date: str) -> Optional[pd.DataFrame]:
        """获取指定周期的股票数据"""
        try:
            # 这里应该调用数据访问层获取真实数据
            # 暂时返回模拟数据
            dates = pd.date_range(end=target_date, periods=100, freq='D')
            data = pd.DataFrame({
                'date': dates,
                'open': np.random.uniform(10, 20, 100),
                'high': np.random.uniform(15, 25, 100),
                'low': np.random.uniform(8, 15, 100),
                'close': np.random.uniform(10, 20, 100),
                'volume': np.random.uniform(1000000, 10000000, 100)
            })
            return data
            
        except Exception as e:
            logger.debug(f"获取{stock_code}的{period}数据失败: {e}")
            return None
    
    def _is_pattern_matched(self, pattern_data: Any, index: int) -> bool:
        """检查指定索引位置是否匹配形态"""
        try:
            if isinstance(pattern_data, pd.Series) and index < len(pattern_data):
                value = pattern_data.iloc[index]
                return bool(value) and value != 0
            elif isinstance(pattern_data, dict):
                return True  # 字典格式的形态认为匹配
            return False
        except:
            return False
    
    def _get_pattern_value(self, pattern_data: Any, index: int) -> Any:
        """获取指定索引位置的形态值"""
        try:
            if isinstance(pattern_data, pd.Series) and index < len(pattern_data):
                return pattern_data.iloc[index]
            elif isinstance(pattern_data, dict):
                return pattern_data
            return None
        except:
            return None
    
    def _get_period_weight(self, period: str) -> float:
        """获取周期权重"""
        weights = {
            '15min': 0.1,
            '30min': 0.15,
            '60min': 0.2,
            'daily': 0.3,
            'weekly': 0.15,
            'monthly': 0.1
        }
        return weights.get(period, 0.1)
