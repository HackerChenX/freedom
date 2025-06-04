#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动指标分析器

自动分析多个技术指标并识别形态
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from indicators.factory import IndicatorFactory
from indicators.scoring_framework import IndicatorScoreManager
from indicators.base_indicator import BaseIndicator

# 尝试导入pattern_recognition_analyzer，如果talib不可用则跳过
try:
    from analysis.pattern_recognition_analyzer import PatternRecognitionAnalyzer
    PATTERN_ANALYZER_AVAILABLE = True
except ImportError as e:
    PATTERN_ANALYZER_AVAILABLE = False
    print(f"警告: 形态识别分析器不可用，跳过K线形态分析: {e}")

logger = get_logger(__name__)

class AutoIndicatorAnalyzer:
    """自动指标分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.indicator_factory = IndicatorFactory()
        self.score_manager = IndicatorScoreManager()
        
        # 获取所有可用指标
        self.all_indicators = self._get_all_indicators()
        
        # 仅在可用时初始化形态分析器
        self.pattern_analyzer = None
        if PATTERN_ANALYZER_AVAILABLE:
            try:
                self.pattern_analyzer = PatternRecognitionAnalyzer()
            except Exception as e:
                logger.error(f"初始化形态分析器时出错: {e}")
        
    def _get_all_indicators(self) -> List[str]:
        """
        获取所有可用指标
        
        Returns:
            List[str]: 所有可用指标列表
        """
        # 获取工厂中所有注册的指标
        indicators = self.indicator_factory.get_all_registered_indicators()
        logger.info(f"已加载 {len(indicators)} 个技术指标")
        return indicators
    
    def analyze_all_indicators(self, 
                      stock_data: Dict[str, pd.DataFrame],
                      target_rows: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
        """
        分析所有指标
        
        Args:
            stock_data: 多周期股票数据
            target_rows: 每个周期的目标行索引
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 按周期组织的指标分析结果
        """
        try:
            results = {}
            
            # 对每个周期的数据分别进行分析
            for period, df in stock_data.items():
                if df is None or df.empty:
                    logger.warning(f"周期 {period} 的数据为空，跳过分析")
                    continue
                    
                # 获取目标行索引
                target_row = target_rows.get(period)
                if target_row is None:
                    logger.warning(f"周期 {period} 没有提供目标行索引，使用最后一行")
                    target_row = len(df) - 1
                    
                if target_row < 0 or target_row >= len(df):
                    logger.warning(f"周期 {period} 的目标行索引 {target_row} 超出范围，跳过分析")
                    continue
                    
                # 分析该周期的所有指标
                period_results = self._analyze_period_indicators(df, target_row)
                
                if period_results:
                    results[period] = period_results
                    
            return results
        except Exception as e:
            logger.error(f"分析所有指标时出错: {e}")
            return {}
    
    def _find_closest_date_row(self, df: pd.DataFrame, target_date: str) -> Optional[int]:
        """
        找到最接近目标日期的数据行索引
        
        Args:
            df: 股票数据
            target_date: 目标日期
            
        Returns:
            Optional[int]: 最接近的行索引，如果找不到则返回None
        """
        try:
            # 确保目标日期格式正确
            if isinstance(target_date, str):
                if len(target_date) == 8:  # YYYYMMDD格式
                    target_date = datetime.strptime(target_date, '%Y%m%d')
                else:
                    target_date = datetime.strptime(target_date, '%Y-%m-%d')
                    
            # 如果DataFrame中的日期是字符串，转换为datetime
            date_col = pd.to_datetime(df['date'])
                
            # 找到不超过目标日期的最大日期
            valid_dates = date_col[date_col <= target_date]
            
            if valid_dates.empty:
                return None
                
            closest_idx = valid_dates.index[-1]  # 取最后一个有效日期的索引
            return closest_idx
            
        except Exception as e:
            logger.error(f"查找最接近日期时出错: {e}")
            return None
    
    def _analyze_period_indicators(self, 
                               df: pd.DataFrame, 
                               target_idx: int) -> List[Dict[str, Any]]:
        """
        分析特定周期下所有指标
        
        Args:
            df: 股票数据
            target_idx: 目标行索引
            
        Returns:
            List[Dict[str, Any]]: 命中的指标列表
        """
        hit_indicators = []
        
        # 分析K线形态，仅在形态分析器可用时执行
        if self.pattern_analyzer is not None:
            pattern_results = self._analyze_patterns(df, target_idx)
            if pattern_results:
                hit_indicators.extend(pattern_results)
        
        # 分析所有技术指标
        for indicator_name in self.all_indicators:
            try:
                # 创建指标实例
                indicator = self.indicator_factory.create_indicator(indicator_name)
                
                # 检查指标实例是否有效
                if indicator is None:
                    logger.warning(f"无法创建指标 {indicator_name} 的实例，跳过分析")
                    continue
                    
                # 检查是否有calculate方法
                if not hasattr(indicator, 'calculate') or not callable(getattr(indicator, 'calculate')):
                    logger.warning(f"指标 {indicator_name} 没有calculate方法，跳过分析")
                    continue
                
                # 计算指标值
                indicator_df = indicator.calculate(df)
                
                # 识别形态
                indicator_hits = self._analyze_indicator_patterns(
                    indicator_name, 
                    indicator_df, 
                    target_idx
                )
                
                # 如果有命中形态，添加到结果中
                if indicator_hits:
                    hit_indicators.extend(indicator_hits)
                    
            except Exception as e:
                logger.error(f"分析指标 {indicator_name} 时出错: {e}")
                continue
                
        return hit_indicators
    
    def _analyze_patterns(self, 
                       df: pd.DataFrame, 
                       target_idx: int) -> List[Dict[str, Any]]:
        """
        分析K线形态
        
        Args:
            df: 股票数据
            target_idx: 目标行索引
            
        Returns:
            List[Dict[str, Any]]: 命中的K线形态列表
        """
        try:
            # 使用形态识别分析器
            patterns = self.pattern_analyzer.identify_patterns(df)
            
            # 过滤出目标日期的形态
            target_date = df.iloc[target_idx]['date']
            
            # 找出最接近但不超过目标日期的形态
            hit_patterns = []
            
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.empty:
                    continue
                    
                # 找到目标日期附近的形态
                date_col = pd.to_datetime(pattern_data['date'])
                target_date_dt = pd.to_datetime(target_date)
                
                # 找出目标日期或之前最近的形态
                valid_patterns = pattern_data[date_col <= target_date_dt]
                
                if valid_patterns.empty:
                    continue
                    
                # 获取最接近目标日期的形态
                closest_pattern = valid_patterns.iloc[-1]
                pattern_date = closest_pattern['date']
                
                # 计算日期差距（天数）
                if isinstance(pattern_date, str):
                    pattern_date = pd.to_datetime(pattern_date)
                date_diff = (target_date_dt - pattern_date).days
                
                # 只考虑最近5个交易日内的形态
                if date_diff <= 5:
                    # 计算形态得分
                    score = self.score_manager.score_pattern(
                        pattern_name, 
                        closest_pattern.to_dict()
                    )
                    
                    hit_patterns.append({
                        'type': 'pattern',
                        'name': pattern_name,
                        'score': score,
                        'date_diff': date_diff,
                        'pattern_date': pattern_date.strftime('%Y-%m-%d') if hasattr(pattern_date, 'strftime') else pattern_date,
                        'details': closest_pattern.to_dict()
                    })
            
            return hit_patterns
            
        except Exception as e:
            logger.error(f"分析K线形态时出错: {e}")
            return []
    
    def _analyze_indicator_patterns(self, 
                                indicator_name: str, 
                                indicator_df: pd.DataFrame, 
                                target_idx: int) -> List[Dict[str, Any]]:
        """
        分析指标形态
        
        Args:
            indicator_name: 指标名称
            indicator_df: 指标计算结果
            target_idx: 目标行索引
            
        Returns:
            List[Dict[str, Any]]: 命中的形态列表
        """
        hit_patterns = []
        
        # 获取指标实例
        indicator = self._get_indicator_instance(indicator_name)
        if indicator is None:
            return []
        
        # 尝试使用identify_patterns方法识别形态
        try:
            # 尝试调用identify_patterns方法
            patterns = []
            try:
                # 尝试直接传递数据
                patterns = indicator.identify_patterns(indicator_df)
            except TypeError as e:
                # 处理参数不匹配的情况
                if "takes 1 positional argument but 2 were given" in str(e):
                    # 如果方法不接受数据参数，直接调用无参方法
                    try:
                        patterns = indicator.identify_patterns()
                    except Exception as inner_e:
                        logger.warning(f"使用 identify_patterns 方法分析指标 {indicator_name} 形态时出错: {inner_e}")
                else:
                    logger.warning(f"使用 identify_patterns 方法分析指标 {indicator_name} 形态时出错: {e}")
            
            # 处理结果
            if patterns:
                for pattern in patterns:
                    # 计算形态得分
                    score = self.score_manager.score_pattern(pattern, {})
                    
                    hit_patterns.append({
                        'indicator': indicator_name,
                        'pattern_id': pattern,
                        'display_name': self._get_pattern_display_name(pattern),
                        'score_impact': score,
                        'date_diff': 0,  # 当前日期的形态
                        'type': 'indicator'
                    })
        except Exception as e:
            logger.error(f"使用 identify_patterns 方法分析指标 {indicator_name} 形态时出错: {e}")
        
        # 尝试使用get_patterns方法获取更详细的形态信息
        try:
            # 检查数据是否足够
            if indicator_df.empty or target_idx >= len(indicator_df):
                return hit_patterns
            
            # 尝试调用get_patterns方法
            detailed_patterns = []
            try:
                # 尝试直接传递数据
                detailed_patterns = indicator.get_patterns(indicator_df)
            except TypeError as e:
                # 处理参数不匹配的情况
                if "takes 1 positional argument but 2 were given" in str(e):
                    # 如果方法不接受数据参数，直接调用无参方法
                    try:
                        detailed_patterns = indicator.get_patterns()
                    except Exception as inner_e:
                        logger.warning(f"使用 get_patterns 方法分析指标 {indicator_name} 形态时出错: {inner_e}")
                else:
                    logger.warning(f"使用 get_patterns 方法分析指标 {indicator_name} 形态时出错: {e}")
            except Exception as e:
                logger.warning(f"使用 get_patterns 方法分析指标 {indicator_name} 形态时出错: {e}")
            
            # 处理详细形态
            if detailed_patterns:
                for pattern in detailed_patterns:
                    # 添加指标名称
                    pattern['indicator'] = indicator_name
                    # 添加到结果
                    hit_patterns.append(pattern)
        except Exception as e:
            logger.error(f"使用 get_patterns 方法分析指标 {indicator_name} 形态时出错: {e}")
        
        return hit_patterns
    
    def _get_indicator_instance(self, indicator_name: str) -> Optional[BaseIndicator]:
        """
        获取指标实例
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Optional[BaseIndicator]: 指标实例，获取失败则返回None
        """
        try:
            # 创建指标实例
            indicator = self.indicator_factory.create_indicator(indicator_name)
            return indicator
        except Exception as e:
            logger.error(f"创建指标 {indicator_name} 实例时出错: {e}")
            return None 